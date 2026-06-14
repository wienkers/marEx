"""
Public entry point for extreme-event identification.

:func:`identify` turns anomalies into a boolean extreme-event field plus the
percentile thresholds that defined it. It is a peer of
:func:`marEx.anomaly.compute`, not a stage of it: it accepts anomalies from
anywhere, including a dataset this package never produced.

Nothing here knows how the anomalies were made, and nothing here mentions
standardisation -- a caller wanting thresholds on a standardised series simply
calls this function again with that series.
"""

import logging
from typing import Dict, List, Literal, Optional, Union

import xarray as xr

from ..core.compute_mode import Materialiser, create_staging_dir
from ..core.dimensions import resolve_dims
from ..core.finalise import finalise_dataset, split_large_chunks
from ..core.time_axis import SeasonalCycle
from ..core.validation import _infer_dims_coords
from ..exceptions import ConfigurationError, create_data_validation_error
from ..logging_config import configure_logging, get_logger, log_memory_usage, log_timing
from .base import identify_extremes, resolve_bin_spec, resolve_window_spatial

# Get module logger
logger = get_logger(__name__)

METHODS = ("seasonal_percentile", "global_percentile")


def _extreme_steps(
    method: str,
    window_days: int,
    window_spatial: Optional[int],
) -> List[str]:
    """Describe the extremes stage for ``ds.attrs["preprocessing_steps"]``."""
    steps: List[str] = []

    if method == "global_percentile":
        steps.append("Global percentile threshold applied to all days")
    elif method == "seasonal_percentile":
        if window_spatial is not None:
            steps.append(f"Day-of-year thresholds with {window_days} day window & {window_spatial} spatial neighbours")
        else:
            steps.append(f"Day-of-year thresholds with {window_days} day window")

    return steps


def _effective_window_spatial(
    method: str,
    window_spatial: Optional[int],
    dimensions: Dict[str, str],
    ds_or_da,
    method_percentile: str = "approximate",
) -> Optional[int]:
    """
    Resolve the spatial window actually used.

    ``identify_extremes`` silently defaults a gridded seasonal run to a 5x5 window
    when the caller leaves it ``None``, so the recorded attributes and steps must
    reflect that rather than the raw ``None``. Delegates to
    :func:`marEx.extremes.base.resolve_window_spatial`, which owns that rule.
    """
    return resolve_window_spatial(ds_or_da, dimensions, method, method_percentile, window_spatial)


def _extremes_core(
    anomalies: xr.DataArray,
    method: str,
    threshold_percentile: float,
    window_days: int,
    window_spatial: Optional[int],
    method_percentile: str,
    precision: Optional[float],
    max_anomaly: Optional[float],
    n_bins: int,
    dimensions: Dict[str, str],
    coordinates: Dict[str, str],
    materialiser: Materialiser,
    threshold_label: str = "thresholds",
    cycle: Optional[SeasonalCycle] = None,
    tail: str = "upper",
):
    """
    Identify extremes, up to but not including output finalisation.

    Shared by :func:`identify` and by :func:`marEx.preprocess_data`.

    ``threshold_label`` names the materialiser staging slot. It is a parameter
    rather than a constant because a caller may run this stage twice on two
    different series (raw and standardised anomalies), and ``Materialiser``
    labels are single-owner: reusing one would raise.

    Returns ``(extremes, thresholds, bin_spec)``. The bin geometry comes back
    because it may have been DERIVED from the data, and the caller records what was
    actually used in the output attributes. It is resolved per series, not once per
    run: a standardised series is in units of sigma and has no reason to share a
    range with the raw anomaly.
    """
    if method not in METHODS:
        raise ConfigurationError(
            f"Unknown extreme method '{method}'",
            details=f"Supported methods are: {', '.join(METHODS)}",
            suggestions=[f"Use method='{METHODS[0]}' (the default)"],
        )

    # Resolve the bin geometry ONCE, here, before anything is built on it. With both
    # `precision` and `max_anomaly` unset this costs a fused min/max pass over the
    # anomaly, so it must not happen twice -- `identify_extremes` re-resolves, but by
    # then both are concrete and the call is a no-op. Skipped for the exact path, which
    # builds no histogram.
    bin_spec = (None, None) if method_percentile == "exact" else resolve_bin_spec(anomalies, precision, max_anomaly, n_bins)
    precision, max_anomaly = bin_spec

    with log_timing(
        logger,
        f"Extreme event identification using {method} method",
        log_memory=True,
        show_progress=True,
    ):
        logger.debug(
            f"Identifying extremes with parameters: method={method}, "
            f"percentile={threshold_percentile}%, method_percentile={method_percentile}"
        )
        extremes, thresholds = identify_extremes(
            anomalies,
            method,
            threshold_percentile,
            dimensions,
            coordinates,
            window_days,
            window_spatial,
            method_percentile,
            precision,
            max_anomaly,
            n_bins=n_bins,
            materialiser=materialiser,
            threshold_label=threshold_label,
            cycle=cycle,
            tail=tail,
        )
        log_memory_usage(logger, "After extreme identification", logging.DEBUG)

    # `thresholds` was already anchored inside the method module (before the comparison
    # that builds `extremes` was constructed on top of it), so this pin only covers
    # `extremes` itself in persist mode.
    extremes, thresholds = materialiser.pin(extremes, thresholds)
    return extremes, thresholds, bin_spec


def _log_extreme_summary(ds: xr.Dataset, materialiser: Materialiser) -> None:
    """
    Report the event count, but only when it is free to do so.

    This sum is a pass over the whole field. Only pay for it when the INFO line
    will actually be emitted AND the field is already materialised -- in
    lazy/streaming mode it would execute the entire graph, silently defeating the
    point of the mode for any INFO-level user.
    """
    if logger.isEnabledFor(logging.INFO) and materialiser.mode == "persist":
        extreme_count = ds.extreme_events.sum()
        if hasattr(extreme_count, "compute"):
            extreme_count = extreme_count.compute()
        logger.info(f"Extreme identification completed - {extreme_count} extreme events identified")
    else:
        logger.info(f"Extreme identification graph constructed (compute_mode='{materialiser.mode}')")


def identify(
    data: Union[xr.DataArray, xr.Dataset],
    method: Literal["seasonal_percentile", "global_percentile"] = "seasonal_percentile",
    *,
    threshold_percentile: float = 95,
    tail: Literal["upper", "lower"] = "upper",
    window_days: int = 11,
    window_spatial: Optional[int] = None,
    method_percentile: Literal["exact", "approximate"] = "approximate",
    precision: Optional[float] = None,
    max_anomaly: Optional[float] = None,
    n_bins: int = 1000,
    dask_chunks: Optional[Dict[str, int]] = None,
    compute_mode: Literal["persist", "lazy", "streaming"] = "persist",
    scratch_dir: Optional[str] = None,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    cycle: Optional[SeasonalCycle] = None,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> xr.Dataset:
    """
    Identify extreme events exceeding a percentile threshold.

    Parameters
    ----------
    data
        Anomalies to threshold. A ``DataArray``, or a ``Dataset`` carrying a
        ``dat_anomaly`` variable -- so that the output of
        :func:`marEx.anomaly.compute` composes directly.
    method
        Thresholding method.

        * ``'seasonal_percentile'`` (default) -- day-of-year resolved
          thresholds computed over a rolling day window, optionally pooled over
          spatial neighbours. Follows the Hobday et al. (2016) definition.
        * ``'global_percentile'`` -- one constant-in-time threshold per cell.
    threshold_percentile
        Percentile defining an extreme, e.g. ``95``.
    tail
        Which side of the distribution counts as extreme. ``'upper'`` (default)
        flags ``data >= threshold``; ``'lower'`` flags ``data <= threshold``, for
        cold spells, drought, or any low-side extreme. The threshold is the
        ``threshold_percentile``-th percentile in both cases, so the coldest 5 %
        is ``threshold_percentile=5, tail='lower'``.
    window_days
        Width of the rolling day-of-year window
        (``seasonal_percentile`` only).
    window_spatial
        Width of the spatial pooling window (``seasonal_percentile`` on gridded
        data only). Defaults to 5 on gridded input, and is unavailable on
        unstructured meshes.
    method_percentile
        ``'approximate'`` (default) uses a histogram-based quantile, which is
        what allows the reduction to stream. ``'exact'`` computes a true
        quantile and needs the full series resident per cell.
    precision, max_anomaly, n_bins
        Histogram bin geometry for ``method_percentile='approximate'``.
        ``max_anomaly`` is the half-width of the binned range and ``precision`` the
        bin width; ``n_bins`` (default 1000) derives whichever of the two is left
        unset. With both unset the range is taken from the data, which is what makes
        the defaults work on a variable that is not an SST anomaly in kelvin --
        precipitation in mm/day, or pressure in Pa. Supplying ``precision=0.01``
        alone reproduces the historical ``+/-5.0`` range exactly.
    dask_chunks
        Output chunking. Defaults to ``{"time": 25}``.
    compute_mode
        Materialisation policy: ``'persist'``, ``'lazy'`` or ``'streaming'``.
    scratch_dir
        Staging directory, required by ``compute_mode='streaming'``.
    dimensions, coordinates
        Name mappings. Inferred when omitted.
    cycle
        Within-year axis the thresholds are resolved on, as a
        :class:`~marEx.SeasonalCycle`. Inferred from the median spacing of the
        time coordinate when omitted: ``dayofyear`` for daily data, ``month``
        for monthly, ``hourofyear`` for sub-daily.
    verbose, quiet
        Logging verbosity overrides.

    Returns
    -------
    xr.Dataset
        ``extreme_events`` (boolean) and ``thresholds``. When given a Dataset,
        its existing variables are carried through.

    Examples
    --------
    >>> import marEx
    >>> anomalies = marEx.anomaly.compute(t2m)
    >>> events = marEx.extremes.identify(anomalies, threshold_percentile=90)
    """
    if dask_chunks is None:
        dask_chunks = {"time": 25}

    if verbose is not None or quiet is not None:
        configure_logging(verbose=verbose, quiet=quiet)

    if isinstance(data, xr.Dataset):
        if "dat_anomaly" not in data.data_vars:
            raise create_data_validation_error(
                "Dataset has no 'dat_anomaly' variable",
                details="identify() accepts a DataArray of anomalies, or a Dataset carrying 'dat_anomaly'",
                suggestions=[
                    "Pass the anomaly DataArray directly, or",
                    "Use the output of marEx.anomaly.compute()",
                ],
                data_info={"available_variables": list(data.data_vars)},
            )
        ds = data
        anomalies = data.dat_anomaly
    else:
        anomalies = data
        ds = xr.Dataset({"dat_anomaly": anomalies})

    logger.info(f"Identifying extremes - method: {method}, percentile: {threshold_percentile}%")

    dimensions, coordinates = _infer_dims_coords(anomalies, dimensions, coordinates)

    # Resolve the dimension contract (see marEx.anomaly.compute for the rationale).
    dims = resolve_dims(anomalies, dimensions, coordinates)
    if dims.extra:
        logger.info(f"Extra (non-horizontal) dimensions detected and carried through: {list(dims.extra)}")

    # NOT resolved here. `identify_extremes` resolves it inside the seasonal branch and
    # `_identify_extremes_seasonal` resolves it again for its own use; both are cheap and
    # deterministic from the same coordinate. Resolving eagerly would run `infer_cycle`
    # -- which raises on a mixed-cadence axis -- on the `global_percentile` path, which
    # needs no within-year cycle and must keep working there.

    staging_dir = create_staging_dir(scratch_dir) if compute_mode == "streaming" and scratch_dir else None
    materialiser = Materialiser(compute_mode, staging_dir)

    with split_large_chunks():
        extremes, thresholds, (used_precision, used_max_anomaly) = _extremes_core(
            anomalies,
            method,
            threshold_percentile,
            window_days,
            window_spatial,
            method_percentile,
            precision,
            max_anomaly,
            n_bins,
            dimensions,
            coordinates,
            materialiser,
            cycle=cycle,
            tail=tail,
        )
        ds = ds.copy()
        ds["extreme_events"] = extremes
        ds["thresholds"] = thresholds

        effective_window_spatial = _effective_window_spatial(method, window_spatial, dimensions, ds)
        ds.attrs.update({"method_extreme": method, "threshold_percentile": threshold_percentile, "tail": tail})
        ds.attrs["preprocessing_steps"] = list(ds.attrs.get("preprocessing_steps", [])) + _extreme_steps(
            method, window_days, effective_window_spatial
        )
        if method == "seasonal_percentile":
            ds.attrs.update({"window_days": window_days})
            if effective_window_spatial is not None:
                ds.attrs.update({"window_spatial": effective_window_spatial})
        # The RESOLVED geometry, not what the caller passed: with both left unset these
        # were derived from the data, and the output has to say which bins produced it.
        ds.attrs.update({"method_percentile": method_percentile, "precision": used_precision, "max_anomaly": used_max_anomaly})

        ds = finalise_dataset(ds, dimensions, coordinates, dask_chunks, materialiser, staging_dir, extra_dims=dims.extra)

    # After finalisation, so that in persist mode the count reads the materialised
    # field rather than re-walking the graph that built it.
    _log_extreme_summary(ds, materialiser)
    return ds
