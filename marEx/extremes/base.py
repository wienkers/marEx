"""
Extreme-identification dispatcher.

Provides the public :func:`identify_extremes` entry point, which validates the
extreme-detection parameters and delegates to one of the concrete methods
(global constant-in-time threshold or day-of-year threshold) based on the
``method_extreme`` argument.
"""

import warnings
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Dict, Literal, Optional, Tuple

import dask
import numpy as np
import xarray as xr

from ..core.compute_mode import Materialiser
from ..core.dimensions import horizontal_dims
from ..core.time_axis import SeasonalCycle, resolve_cycle
from ..core.validation import _infer_dims_coords
from ..exceptions import ConfigurationError
from ..logging_config import configure_logging, get_logger
from .global_percentile import _identify_extremes_constant
from .histogram import _RangeSaturated
from .seasonal_percentile import _identify_extremes_seasonal

# Get module logger
logger = get_logger(__name__)


def supports_spatial_window(da: xr.DataArray, dimensions: Dict[str, str]) -> bool:
    """Whether a spatial rolling window is meaningful for this field.

    The window rolls over the HORIZONTAL dimensions only -- never over an extra
    dimension such as depth, which must not be smoothed across. So it needs two
    horizontal dims present, i.e. a structured grid.
    """
    return len([d for d in horizontal_dims(dimensions) if d in da.dims]) >= 2


def resolve_window_spatial(
    da: xr.DataArray,
    dimensions: Dict[str, str],
    method_extreme: str,
    method_percentile: str,
    window_spatial: Optional[int],
) -> Optional[int]:
    """Resolve the spatial window actually used, applying the gridded default.

    Sole definition of that default. :mod:`marEx.extremes.api` calls it to record
    the resolved value in the output attributes rather than restating the rule.

    The default applies only to the approximate seasonal path: the exact
    percentile path ignores ``window_spatial`` entirely, and validation rejects it
    when a caller supplies it there, so defaulting it would only inflate
    ``N_samples`` and hide the warning.
    """
    if (
        method_extreme == "seasonal_percentile"
        and method_percentile != "exact"
        and window_spatial is None
        and supports_spatial_window(da, dimensions)
    ):
        return 5  # Default to 5x5 spatial window for structured grids
    return window_spatial


# Bin geometry used when the data gives no usable scale (all-NaN, or constant). These are the
# historical SST-calibrated defaults, in kelvin.
_FALLBACK_PRECISION = 0.01
_FALLBACK_MAX_ANOMALY = 5.0
# A derived bin wider than this fraction of the anomaly std triggers the coarse-bin warning in
# `resolve_bin_spec` (bins of 0.0095 std over-flagged a seasonal p90 mask by 1.5 % against
# exact, bins of 0.088 std by 10 %).
_COARSE_BIN_FRACTION_OF_STD = 0.03
# Bin count over the derived range when `precision` is not given. The range is the
# tail's own estimate, so the bins land where thresholds can be: 0.25 deg OSTIA p95 gets
# +/-21.04 K, i.e. precision 0.014 K.
_TARGET_N_BINS = 3000
# Bin count used with the deprecated one-sided `max_anomaly` pass (the historical invariant).
_DEFAULT_N_BINS = 1000
# Above this many bins the histogram stage gets markedly slower (and, on the global path, its
# tiles smaller and more numerous), so say so.
_WARN_N_BINS = 10000
# Ceiling on any bin count. The bin index is uint16 and n_bins=65535 already realises 65536 bins
# (NEXT Discovered), so stay clear of the edge.
_MAX_DERIVED_N_BINS = 65000
# Safety factor on the per-cell normal estimate of the most extreme threshold,
# max over cells of (mean + z_p * std). Measured on 0.25 deg OSTIA p95 seasonal:
# the true largest threshold is 2.19x that estimate (variance concentrated in one season,
# e.g. at the ice edge). The estimate only ever LOWERS the range below the data's own
# extreme, and a range that turns out too narrow is regrown (see `_regrow_bin_spec`).
_RANGE_SAFETY = 3.0


@dataclass(frozen=True)
class BinSpec:
    """The approximate path's histogram geometry, and where it came from.

    ``cap`` is the data's own most extreme value on the requested tail's side
    (``max`` for ``tail='upper'``, ``-min`` for ``'lower'``). No threshold can lie
    beyond it, because a threshold is a quantile of those very samples.

    ``mode`` says what a threshold in the outermost bin means:

    * ``'pinned'``: the caller set the range (deprecated ``max_anomaly``); samples
      past it were clipped, so the threshold is meaningless and it raises.
    * ``'estimated'``: the range is the per-cell normal estimate, below ``cap``;
      samples past it were clipped, so the bins are regrown to ``cap`` and the
      threshold recomputed (once).
    * ``'data'``: the range is ``cap`` itself, so nothing on the tail's side was
      clipped; reaching the end bin only means too few samples, and it warns.
    """

    precision: float
    max_anomaly: float
    cap: float
    mode: Literal["pinned", "estimated", "data"]

    @property
    def n_bins(self) -> int:
        """Number of bins spanning ``[-max_anomaly, max_anomaly]`` at ``precision``."""
        return int(round(2.0 * self.max_anomaly / self.precision))


def _warn_deprecated_bin_args(max_anomaly: Optional[float], n_bins: Optional[int]) -> None:
    """``max_anomaly`` and ``n_bins`` are no longer part of the public interface."""
    for name, value in (("max_anomaly", max_anomaly), ("n_bins", n_bins)):
        if value is not None:
            warnings.warn(
                f"`{name}` is deprecated and will be removed: the histogram range is now derived from the "
                "data (the requested tail's side, regrown if a threshold reaches its edge), and `precision` "
                "alone sets the bin width.",
                FutureWarning,
                stacklevel=3,
            )


def reject_empty_series(da: xr.DataArray) -> None:
    """Reject a zero-length anomaly with a cause, not a symptom.

    An empty series reaches the histogram's range reduction as a zero-size array, where
    `nanmin` raises "zero-size array to reduction operation fmin which has no identity";
    on the explicit-bins path it survives to a bare `ZeroDivisionError` deeper in the
    histogram; and on the ``method_percentile="exact"`` path, which builds no histogram
    and so never reaches :func:`resolve_bin_spec`, it reached a bare `ZeroDivisionError`
    too. All three are unintelligible, and the cause is nearly always a baseline window
    longer than the series it is given (`shifting_baseline` trims the first
    `window_years` years, so `window_years=3` on 2.7 years of data leaves nothing).

    Lives outside :func:`resolve_bin_spec` so that the exact path -- which has no bin
    geometry to resolve -- still reports it identically.
    """
    if da.size != 0:
        return
    empty_dims = [str(d) for d, n in zip(da.dims, da.shape) if n == 0]
    raise ConfigurationError(
        f"Cannot identify extremes: the anomaly series is empty ({', '.join(f'{d}=0' for d in empty_dims)})",
        details=(
            "Identifying extremes needs at least one sample. "
            "An empty anomaly usually means the baseline window consumed the whole series: "
            "`shifting_baseline` removes the first `window_years` years before computing anomalies."
        ),
        suggestions=[
            "Reduce `window_years` so it is shorter than the input time series",
            "Lengthen the input time series",
            "Use `method_anomaly='detrend_harmonic'` or `'fixed_baseline'`, which do not trim the series",
        ],
        context={"shape": tuple(da.shape), "dims": tuple(str(d) for d in da.dims)},
    )


def _derive_bin_spec(
    da: xr.DataArray,
    precision: Optional[float],
    max_anomaly: Optional[float],
    n_bins: Optional[int] = None,
    threshold_percentile: Optional[float] = None,
    tail: Literal["upper", "lower"] = "upper",
    time_dim: Optional[str] = None,
) -> BinSpec:
    """Resolve the histogram geometry for the approximate path; see :func:`resolve_bin_spec`."""
    reject_empty_series(da)

    if n_bins is not None and n_bins < 2:
        raise ConfigurationError(
            f"n_bins must be at least 2, got {n_bins}",
            details="n_bins sets the number of histogram bins spanning [-max_anomaly, +max_anomaly]",
            suggestions=["Omit n_bins and set `precision` instead", "Increase n_bins for a finer threshold estimate"],
            context={"n_bins": n_bins},
        )
    # The bin index is stored as uint16 (`extremes/histogram.py`'s flox expected_groups),
    # so a count above 65535 would wrap silently rather than fail.
    if n_bins is not None and n_bins > 65535:
        raise ConfigurationError(
            f"n_bins must not exceed 65535, got {n_bins}",
            details=(
                "Histogram bin indices are stored as uint16; a larger count wraps silently " "and assigns samples to the wrong bins"
            ),
            suggestions=["Use a smaller n_bins", "Widen `precision` instead of adding bins"],
            context={"n_bins": n_bins, "max_supported": 65535},
        )

    # A caller-set range (deprecated): honoured exactly, as before, and a saturated threshold raises.
    if max_anomaly is not None:
        if precision is None:
            precision = 2.0 * max_anomaly / (n_bins or _DEFAULT_N_BINS)
        spec = BinSpec(float(precision), float(max_anomaly), float(max_anomaly), "pinned")
        _warn_on_bin_count(spec, precision_given=True)
        return spec

    # One fused pass: the global extremes (the tail's hard cap), the global std (coarse-bin
    # warning) and, when the percentile is known, the per-cell normal estimate of the most
    # extreme threshold, max over cells of (mean + z * std) along time. The per-cell maps are
    # reduced to a scalar inside the graph, so only four numbers come back.
    estimate_wanted = threshold_percentile is not None and time_dim is not None and time_dim in da.dims
    reductions = [da.min(), da.max(), da.std()]
    if estimate_wanted:
        z = NormalDist().inv_cdf(min(max(threshold_percentile / 100.0, 1e-9), 1 - 1e-9))
        cell_threshold = da.mean(time_dim) + z * da.std(time_dim)
        reductions.append(cell_threshold.max() if tail == "upper" else -cell_threshold.min())
    computed = dask.compute(*reductions)
    lo, hi, spread = (float(v) for v in computed[:3])
    cell_extreme = float(computed[3]) if estimate_wanted else np.nan

    # Every threshold is a quantile of samples on its own side, so the side's extreme bounds it.
    cap = hi if tail == "upper" else -lo
    if not np.isfinite(cap) or cap <= 0:
        # Nothing on the tail's side of zero: every threshold sits on the guard rail, so any
        # finite range will do. Use the other side's scale, then the historical defaults.
        cap = max(abs(lo), abs(hi))
    if not np.isfinite(cap) or cap <= 0:
        logger.warning(
            f"Could not derive a histogram range from the data (min={lo}, max={hi}); "
            f"falling back to precision={_FALLBACK_PRECISION}, max_anomaly={_FALLBACK_MAX_ANOMALY}."
        )
        return BinSpec(_FALLBACK_PRECISION, _FALLBACK_MAX_ANOMALY, _FALLBACK_MAX_ANOMALY, "data")

    estimate = _RANGE_SAFETY * cell_extreme
    if np.isfinite(estimate) and 0 < estimate < cap:
        max_anomaly, mode = estimate, "estimated"
    else:
        max_anomaly, mode = cap, "data"

    precision_given = precision is not None
    if precision is None:
        precision = 2.0 * max_anomaly / (n_bins or _TARGET_N_BINS)
    elif 2.0 * max_anomaly / precision > _MAX_DERIVED_N_BINS:
        raise ConfigurationError(
            f"precision={precision:.4g} needs {2.0 * max_anomaly / precision:.0f} histogram bins over the derived "
            f"range +/-{max_anomaly:.4g}, above the {_MAX_DERIVED_N_BINS} the uint16 bin index allows",
            details=(
                f"The range is derived from the data (the {tail} tail's extreme, or a per-cell estimate of the "
                f"largest threshold below it), and the bin count is 2 * range / precision."
            ),
            suggestions=[
                f"Use precision >= {2.0 * max_anomaly / _MAX_DERIVED_N_BINS:.3g}",
                f"Omit precision ({_TARGET_N_BINS} bins over the derived range)",
            ],
            context={"precision": precision, "max_anomaly": max_anomaly, "max_n_bins": _MAX_DERIVED_N_BINS},
        )
    spec = BinSpec(float(precision), float(max_anomaly), float(cap), mode)
    _warn_on_bin_count(spec, precision_given)

    # An explicit `precision` is the caller's choice and is never second-guessed.
    if not precision_given and np.isfinite(spread) and spread > 0 and spec.precision > _COARSE_BIN_FRACTION_OF_STD * spread:
        hint = (
            f" The {tail} tail's extreme ({cap:.4g}) is {cap / spread:.0f} std: check for unmasked fill values."
            if cap > 50 * spread
            else ""
        )
        logger.warning(
            f"Derived histogram bins are coarse: precision={spec.precision:.4g} is {spec.precision / spread:.3f} x the "
            f"anomaly std ({spread:.4g}) over the range +/-{spec.max_anomaly:.4g}. Approximate thresholds are then "
            f"accurate only to this bin width.{hint} Pass a smaller `precision` (e.g. ~0.01 x the std) if that "
            "matters for your variable."
        )
    logger.info(
        f"Histogram bins derived from the data: precision={spec.precision:.6g}, max_anomaly={spec.max_anomaly:.6g} "
        f"({spec.n_bins} bins; range {spec.mode}, {tail}-tail extreme {cap:.6g})"
    )
    return spec


def _warn_on_bin_count(spec: BinSpec, precision_given: bool) -> None:
    if spec.n_bins > _WARN_N_BINS:
        cause = "The requested precision" if precision_given else "This geometry"
        logger.warning(
            f"{cause} gives {spec.n_bins} histogram bins (precision={spec.precision:.4g} over "
            f"+/-{spec.max_anomaly:.4g}). Above {_WARN_N_BINS} bins the threshold stage is markedly slower and, "
            "for global_percentile, needs more memory per cell. A coarser `precision` reduces the bin count."
        )


def _regrow_bin_spec(spec: BinSpec) -> BinSpec:
    """Widen an ``'estimated'`` range to the data's own extreme after a threshold reached its edge.

    The bin width is kept, so every threshold that was inside the old range is reproduced
    bit-for-bit (the positive edges are ``arange(-p, max + p, p)``, the same floats at the same
    index whatever ``max``), and the saturated ones become exact quantiles. Past ``cap`` nothing
    is clipped, so one regrow is final. Only above the uint16 ceiling is the width widened.
    """
    regrown = replace(spec, max_anomaly=spec.cap, mode="data")
    if regrown.n_bins > _MAX_DERIVED_N_BINS:
        widened = 2.0 * spec.cap / _MAX_DERIVED_N_BINS
        logger.warning(
            f"Keeping precision={spec.precision:.4g} over +/-{spec.cap:.4g} would need {regrown.n_bins} bins; "
            f"widened to precision={widened:.4g} ({_MAX_DERIVED_N_BINS} bins)."
        )
        regrown = replace(regrown, precision=widened)
    else:
        _warn_on_bin_count(regrown, precision_given=False)
    return regrown


def resolve_bin_spec(
    da: xr.DataArray,
    precision: Optional[float],
    max_anomaly: Optional[float],
    n_bins: Optional[int] = None,
    *,
    threshold_percentile: Optional[float] = None,
    tail: Literal["upper", "lower"] = "upper",
    time_dim: Optional[str] = None,
) -> Tuple[float, float]:
    """Resolve the histogram bin width and range, deriving whatever was not supplied.

    Only ``precision`` is a public input. The range ``max_anomaly`` (the half-width of the
    symmetric bins) is derived from the data in whatever units it is in:

    1. **The tail's own extreme is a hard cap.** A threshold is a quantile of the samples,
       so it can never pass ``max(anomaly)`` (``tail='upper'``) or ``-min(anomaly)``
       (``'lower'``). The other side may be clipped freely: those samples still count in
       the tail's CDF.
    2. **A per-cell normal estimate lowers it** when that extreme is far out (heavy tails, a
       single storm): ``3 * max_cells(mean + z_p * std)`` along time, with ``z_p`` the normal
       quantile of ``threshold_percentile``. The factor 3 covers seasonal variance and heavy
       tails (on OSTIA p95 the largest threshold is 2.19x the bare estimate).
    3. **A threshold that still reaches the end bin regrows the range** to the cap at the same
       ``precision`` and recomputes it (once; :func:`_regrow_bin_spec`), so an estimate that
       is too low costs time, never a wrong threshold.

    ``precision`` defaults to ``2 * max_anomaly / 3000``. Passed explicitly, the bin count
    follows the derived range; above 10000 bins it warns, and above 65000 (the uint16 bin
    index) it raises. The deprecated ``max_anomaly`` pins the range as before, and the
    deprecated ``n_bins`` replaces the 3000-bin target.

    Deriving costs one fused ``dask.compute`` -- a full pass over the anomaly, cheap when it is
    already staged (``persist``), a walk of the anomaly graph otherwise. It is skipped for
    ``method_percentile='exact'``, which builds no histogram. ``threshold_percentile`` and
    ``time_dim`` enable step 2; without them the range is the cap.

    Returns ``(precision, max_anomaly)``.
    """
    spec = _derive_bin_spec(da, precision, max_anomaly, n_bins, threshold_percentile, tail, time_dim)
    return spec.precision, spec.max_anomaly


def identify_extremes(
    da: xr.DataArray,
    method_extreme: Literal["global_percentile", "seasonal_percentile"] = "seasonal_percentile",
    threshold_percentile: float = 95,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    window_days: int = 11,  # for seasonal_percentile
    window_spatial: Optional[int] = None,  # for seasonal_percentile
    method_percentile: Literal["exact", "approximate"] = "approximate",
    precision: Optional[float] = None,
    max_anomaly: Optional[float] = None,
    n_bins: Optional[int] = None,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
    materialiser: Optional[Materialiser] = None,
    threshold_label: str = "thresholds",
    cycle: Optional[SeasonalCycle] = None,
    tail: Literal["upper", "lower"] = "upper",
    bin_spec_out: Optional[list] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events exceeding a percentile threshold using specified method.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing anomalies
    method_extreme : str, default='seasonal_percentile'
        Method for threshold calculation ('global_percentile' or 'seasonal_percentile')
    threshold_percentile : float, default=95
        Percentile threshold (e.g., 95 for 95th percentile)
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    window_days : int, default=11
        Window for day-of-year threshold (seasonal_percentile only)
    window_spatial : int, default=None
        Width in cells (odd) of the square spatial pooling window (seasonal_percentile
        with method_percentile='approximate' on gridded data only). ``None`` resolves
        to 5 (a 5x5 window) on that path and to no pooling everywhere else.
    method_percentile : str, default='approximate'
        Method for percentile computation ('exact' or 'approximate')
    precision : float, optional
        Histogram bin width for the approximate method, in the variable's units. The
        binned range is derived from the data (the requested tail's extreme, lowered by a
        per-cell normal estimate of the largest threshold and regrown if a threshold
        reaches its edge; see ``resolve_bin_spec``). Omitted, it is ``range / 1500``
        (3000 bins over +/- range). Warns above 10000 bins; raises above 65000.
    max_anomaly : float, optional
        Deprecated. Pins the half-width of the binned range; a threshold reaching it raises.
    n_bins : int, optional
        Deprecated. Replaces the 3000-bin target when ``precision`` is omitted.
    tail : {'upper', 'lower'}, default='upper'
        Which side of the distribution counts as extreme. ``'upper'`` flags
        ``data >= threshold``, ``'lower'`` flags ``data <= threshold``. The
        threshold is the ``threshold_percentile``-th percentile in both cases, so
        the coldest 5 % is ``threshold_percentile=5, tail='lower'``.
    bin_spec_out : list, optional
        Internal. When given, the :class:`BinSpec` actually used (after any regrow; ``None``
        on the exact path) is appended, so a caller can record it in its output attributes.
    Returns
    -------
    tuple
        Tuple of (extremes, thresholds) where extremes is a boolean array
        identifying extreme events and thresholds contains the threshold values used

    Examples
    --------
    Basic extreme identification with global thresholds:

    >>> import xarray as xr
    >>> import marEx
    >>>
    >>> # Load anomaly data (from compute_normalised_anomaly)
    >>> anomalies = xr.open_dataset('anomalies.nc', chunks={}).dat_anomaly
    >>>
    >>> # Identify extreme events using global-in-time 95th percentile
    >>> extremes, thresholds = marEx.identify_extremes(
    ...     anomalies,
    ...     method_extreme="global_percentile",
    ...     threshold_percentile=95
    ... )
    >>> print(f"Extreme events shape: {extremes.shape}")
    Extreme events shape: (1461, 180, 360)
    >>> print(f"Thresholds shape: {thresholds.shape}")
    Thresholds shape: (180, 360)

    >>> # Count total extreme events
    >>> total_extremes = extremes.sum().compute()
    >>> print(f"Total extreme events: {total_extremes}")

    Using day-of-year specific thresholds (cf. Hobday et al. 2016 method):

    >>> # More sophisticated threshold calculation
    >>> extremes_seasonal, thresholds_seasonal = marEx.identify_extremes(
    ...     anomalies,
    ...     method_extreme="seasonal_percentile",
    ...     threshold_percentile=95,
    ...     window_days=11  # 11-day window around each day-of-year
    ...     window_spatial=3  # 3x3 spatial window for clustering percentile calcuation
    ... )
    >>> print(f"Seasonal thresholds shape: {thresholds_seasonal.shape}")
    Seasonal thresholds shape: (366, 180, 360)

    >>> # Compare seasonal variation in thresholds
    >>> summer_threshold = thresholds_seasonal.sel(dayofyear=200).mean()
    >>> winter_threshold = thresholds_seasonal.sel(dayofyear=50).mean()
    >>> print(f"Summer vs Winter thresholds: {summer_threshold:.3f} vs {winter_threshold:.3f}")

    Comparison of exact vs approximate percentile methods:

    >>> # Approximate method (faster, default)
    >>> extremes_approx, thresh_approx = marEx.identify_extremes(
    ...     anomalies, method_percentile="approximate"
    ... )
    >>>
    >>> # Exact method (slower & memory intensive)
    >>> extremes_exact, thresh_exact = marEx.identify_extremes(
    ...     anomalies, method_percentile="exact"
    ... )
    >>>
    >>> # Compare threshold precision — ~0.005C
    >>> threshold_diff = (thresh_exact - thresh_approx).std().compute()
    >>> print(f"Threshold difference (exact vs approx): {threshold_diff:.6f}")

    Different percentile thresholds for varying event rarity:

    >>> # Conservative threshold - very extreme events only
    >>> extremes_98, _ = marEx.identify_extremes(
    ...     anomalies, threshold_percentile=98
    ... )
    >>>
    >>> # Moderate threshold - more frequent events
    >>> extremes_90, _ = marEx.identify_extremes(
    ...     anomalies, threshold_percentile=90
    ... )
    >>>
    >>> # Compare event frequency
    >>> print(f"99th percentile events: {extremes_99.sum().compute()}")
    >>> print(f"90th percentile events: {extremes_90.sum().compute()}")

    Processing unstructured data:

    >>> # ICON ocean model data
    >>> icon_anomalies = xr.open_dataset('icon_anomalies.nc', chunks={}).dat_anomaly
    >>> extremes_unstructured, thresholds_unstructured = marEx.identify_extremes(
    ...     icon_anomalies,
    ...     dimensions={"time": "time", "x": "ncells"},
    ...     coordinates={"time": "time", "x": "lon", "y": "lat"},
    ...     threshold_percentile=95
    ... )
    >>> print(f"Unstructured extremes shape: {extremes_unstructured.shape}")

    Advanced seasonal method with custom temporal window:

    >>> # Longer temporal window for smoother thresholds
    >>> extremes_smooth, thresholds_smooth = marEx.identify_extremes(
    ...     anomalies,
    ...     method_extreme="seasonal_percentile",
    ...     window_days=31,  # Longer smoothing window
    ...     threshold_percentile=95
    ... )
    >>>
    >>> # Compare threshold smoothness
    >>> std_11day = thresholds_seasonal.std(dim='dayofyear').mean().compute()
    >>> std_31day = thresholds_smooth.std(dim='dayofyear').mean().compute()
    >>> print(f"Threshold variability: 11-day={std_11day:.3f}, 31-day={std_31day:.3f}")
    """
    # A None materialiser means "default to persist mode", which keeps every existing
    # caller, doctest and test working unchanged.
    if materialiser is None:
        materialiser = Materialiser("persist")

    # Configure logging if verbose/quiet parameters are provided
    if verbose is not None or quiet is not None:
        configure_logging(verbose=verbose, quiet=quiet)

    logger.debug(f"Identifying extremes using {method_extreme} method - {threshold_percentile}th percentile")
    _warn_deprecated_bin_args(max_anomaly, n_bins)

    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    # Validate method_percentile parameter
    valid_methods = ["exact", "approximate"]
    if method_percentile not in valid_methods:
        logger.error(f"Unknown method_percentile: {method_percentile}")
        raise ConfigurationError(
            f"Unknown method_percentile '{method_percentile}'",
            details="Invalid method_percentile parameter",
            suggestions=[
                "Use 'exact' for precise percentile computation (memory intensive)",
                "Use 'approximate' for efficient histogram-based computation (default)",
            ],
            context={
                "provided_method": method_percentile,
                "valid_methods": valid_methods,
            },
        )

    # Validate tail parameter
    valid_tails = ["upper", "lower"]
    if tail not in valid_tails:
        logger.error(f"Unknown tail: {tail}")
        raise ConfigurationError(
            f"Unknown tail '{tail}'",
            details="Invalid tail parameter",
            suggestions=[
                "Use 'upper' for extremes above the threshold (the default)",
                "Use 'lower' for extremes below the threshold, e.g. cold spells or drought",
            ],
            context={"provided_tail": tail, "valid_tails": valid_tails},
        )

    # Validate parameter compatibility for exact percentile method
    if method_percentile == "exact":
        # Detected by the SENTINEL, not by comparison against a literal default: the
        # defaults are now derived, so `precision != 0.01` would fire on every exact
        # run the moment auto-derivation set a value.
        if precision is not None:
            logger.error(f"Invalid parameter: precision={precision} with method_percentile='exact'")
            raise ConfigurationError(
                "Parameter 'precision' cannot be used with method_percentile='exact'",
                details=(
                    f"The precision parameter (precision={precision}) is only used by the approximate "
                    "histogram method and is ignored when using exact percentile computation"
                ),
                suggestions=[
                    "Remove the 'precision' parameter when using method_percentile='exact'",
                    "Use method_percentile='approximate' if you want to control histogram precision",
                ],
                context={
                    "method_percentile": method_percentile,
                    "provided_precision": precision,
                },
            )

        if max_anomaly is not None:
            logger.error(f"Invalid parameter: max_anomaly={max_anomaly} with method_percentile='exact'")
            raise ConfigurationError(
                "Parameter 'max_anomaly' cannot be used with method_percentile='exact'",
                details=(
                    f"The max_anomaly parameter (max_anomaly={max_anomaly}) is only used by the approximate "
                    "histogram method and is ignored when using exact percentile computation"
                ),
                suggestions=[
                    "Remove the 'max_anomaly' parameter when using method_percentile='exact'",
                    "Use method_percentile='approximate' if you want to control histogram binning range",
                ],
                context={
                    "method_percentile": method_percentile,
                    "provided_max_anomaly": max_anomaly,
                },
            )

    # NOTE: the approximate method used to reject `threshold_percentile < 60`. That was
    # correct under the old asymmetric bins, where every negative value shared a single
    # bin and any percentile falling into it was undefined by construction. The bins are
    # now symmetric about zero (`extremes/histogram.py::_symmetric_bin_edges`), so a low
    # percentile is resolved at exactly the same precision as a high one and the
    # rejection is obsolete. The genuine remaining failure mode -- a threshold landing in
    # a clipped end bin -- raises a ConfigurationError in
    # `extremes/histogram.py::_apply_threshold_bounds` when the caller pinned the range, and
    # warns when it was derived (nothing clipped), symmetrically at both ends.

    # Validate window_spatial parameter
    if window_spatial is not None:
        # A spatial window needs two horizontal dims. Extra dims (depth, level) do
        # not count: the window never rolls over them.
        if not supports_spatial_window(da, dimensions):
            logger.error(f"window_spatial={window_spatial} specified for unstructured grid")
            raise ConfigurationError(
                "window_spatial is not supported for unstructured grids",
                details=(
                    "Spatial smoothing with window_spatial requires structured grids with both x and y dimensions. "
                    "It applies to the horizontal dimensions only, never to extra dimensions such as depth. "
                    "Unstructured grids do not support spatial window operations due to computational and memory "
                    "limitations of the algorithms."
                ),
                suggestions=[
                    "Remove the window_spatial parameter for unstructured grids",
                    "Use structured grid data if spatial smoothing is required",
                    "Set window_spatial=None to use default behavior",
                ],
                context={
                    "grid_type": "unstructured",
                    "window_spatial": window_spatial,
                    "dimensions": dimensions,
                    "available_dims": list(da.dims),
                },
            )

        # Check if window_spatial is specified when seasonal_percentile is not used
        if method_extreme != "seasonal_percentile":
            logger.error(f"window_spatial={window_spatial} specified with method_extreme='{method_extreme}'")
            raise ConfigurationError(
                "window_spatial can only be used with method_extreme='seasonal_percentile'",
                details=(
                    "The window_spatial parameter is only implemented for the seasonal_percentile method. "
                    "Other extreme methods do not support spatial smoothing due to computational and memory "
                    "limitations of the algorithms."
                ),
                suggestions=[
                    "Remove the window_spatial parameter when using method_extreme='global_percentile'",
                    "Use method_extreme='seasonal_percentile' if spatial smoothing is required",
                    "Set window_spatial=None to use default behavior",
                ],
                context={
                    "method_extreme": method_extreme,
                    "window_spatial": window_spatial,
                    "compatible_methods": ["seasonal_percentile"],
                },
            )

        # Check if window_spatial is specified when method_percentile is "exact"
        if method_percentile == "exact":
            logger.error(f"window_spatial={window_spatial} specified with method_percentile='exact'")
            raise ConfigurationError(
                "window_spatial is not supported with method_percentile='exact'",
                details=(
                    "The window_spatial parameter is only implemented for the approximate percentile method. "
                    "Exact percentile computation does not support spatial smoothing due to computational and memory "
                    "limitations of the algorithms."
                ),
                suggestions=[
                    "Remove the window_spatial parameter when using method_percentile='exact'",
                    "Use method_percentile='approximate' if spatial smoothing is required",
                    "Set window_spatial=None to use default behavior",
                ],
                context={
                    "method_percentile": method_percentile,
                    "window_spatial": window_spatial,
                    "compatible_methods": ["approximate"],
                },
            )

    # Validate that window parameters are odd numbers (only for seasonal_percentile method).
    #
    # Oddness is a property of the window in TIMESTEPS, not in days -- the window must be
    # symmetric about a centre step. On a daily axis the two coincide, which is why this
    # has always been expressed in days. On any other cadence they do not: an 11-day
    # window on 6-hourly data is 44 steps, and demanding an odd number of *days* there
    # would reject a perfectly well-posed request. `SeasonalCycle.window_steps` forces
    # the step count odd on those axes, so the check is only needed, and only meaningful,
    # for daily data.
    #
    # Resolved ONLY for the seasonal method. `infer_cycle` raises on a mixed-cadence
    # axis, and `global_percentile` has no within-year cycle at all -- resolving
    # unconditionally would make it fail on axes where it has always worked, naming a
    # problem it does not have.
    resolved_cycle = resolve_cycle(da, coordinates["time"], cycle) if method_extreme == "seasonal_percentile" else cycle
    if method_extreme == "seasonal_percentile" and resolved_cycle.is_daily and window_days is not None and window_days % 2 == 0:
        logger.error(f"window_days={window_days} is not an odd number")
        raise ConfigurationError(
            "window_days must be an odd number",
            details=(
                f"Window parameters require odd numbers to ensure symmetric windows around a central point. "
                f"window_days={window_days} is even, which would create asymmetric temporal windows."
            ),
            suggestions=[
                f"Use window_days={window_days + 1} or {window_days - 1}",
                "Choose an odd number",
            ],
            context={
                "window_days": window_days,
                "is_odd": False,
            },
        )

    # Set default spatial window (only for the approximate seasonal_percentile method).
    window_spatial = resolve_window_spatial(da, dimensions, method_extreme, method_percentile, window_spatial)

    if method_extreme == "seasonal_percentile" and window_spatial is not None and window_spatial % 2 == 0:
        logger.error(f"window_spatial={window_spatial} is not an odd number")
        raise ConfigurationError(
            "window_spatial must be an odd number",
            details=(
                f"Window parameters require odd numbers to ensure symmetric windows around a central point. "
                f"window_spatial={window_spatial} is even, which would create asymmetric spatial windows."
            ),
            suggestions=[
                f"Use window_days={window_days + 1} or {window_days - 1}",
                "Choose an odd number.",
            ],
            context={
                "window_spatial": window_spatial,
                "is_odd": False,
            },
        )

    if method_extreme not in ("global_percentile", "seasonal_percentile"):
        logger.error(f"Unknown extreme method: {method_extreme}")
        raise ConfigurationError(
            f"Unknown extreme method '{method_extreme}'",
            details="Invalid method_extreme parameter",
            suggestions=[
                "Use 'global_percentile' for efficient constant percentile threshold",
                "Use 'seasonal_percentile' for day-of-year specific thresholds",
            ],
            context={
                "provided_method": method_extreme,
                "valid_methods": ["global_percentile", "seasonal_percentile"],
            },
        )

    # Resolve the bin geometry once, here, and hand concrete numbers down. Skipped for
    # the exact path, which builds no histogram -- deriving there would cost a full pass
    # over the anomaly for nothing, and would defeat the sentinel check above.
    spec: Optional[BinSpec] = None
    if method_percentile != "exact":
        spec = _derive_bin_spec(da, precision, max_anomaly, n_bins, threshold_percentile, tail, dimensions["time"])

    def _dispatch(spec: Optional[BinSpec]) -> Tuple[xr.DataArray, xr.DataArray]:
        bin_precision = None if spec is None else spec.precision
        bin_max_anomaly = None if spec is None else spec.max_anomaly
        range_mode = "pinned" if spec is None else spec.mode
        if method_extreme == "global_percentile":
            logger.debug(f"Global extreme method - method_percentile={method_percentile}")
            return _identify_extremes_constant(
                da,
                threshold_percentile,
                method_percentile,
                dimensions,
                bin_precision,
                bin_max_anomaly,
                materialiser,
                threshold_label,
                tail=tail,
                range_mode=range_mode,
            )
        logger.debug(f"Seasonal percentile method - window_days={window_days}, method_percentile={method_percentile}")
        return _identify_extremes_seasonal(
            da,
            threshold_percentile,
            window_days,
            window_spatial,
            method_percentile,
            dimensions,
            coordinates,
            bin_precision,
            bin_max_anomaly,
            materialiser,
            threshold_label,
            resolved_cycle,
            tail=tail,
            range_mode=range_mode,
        )

    try:
        result = _dispatch(spec)
    except _RangeSaturated as saturated:
        # The estimated range was too narrow somewhere: regrow it to the data's own extreme at
        # the same bin width and recompute. Nothing past that extreme exists, so the
        # second attempt cannot saturate this way again. The first attempt staged nothing under
        # `threshold_label` (thresholds are staged only after the bounds check), so the label
        # is free; its anonymous pin is released with it.
        regrown = _regrow_bin_spec(spec)
        logger.warning(
            f"A threshold reached the estimated histogram range +/-{spec.max_anomaly:.4g} "
            f"({saturated.reached:.4g}); recomputing the thresholds over the data's {tail}-tail extreme "
            f"+/-{regrown.max_anomaly:.4g} ({regrown.n_bins} bins, precision={regrown.precision:.4g})."
        )
        spec = regrown
        result = _dispatch(spec)

    if bin_spec_out is not None:
        bin_spec_out.append(spec)
    return result
