"""
Extreme-identification dispatcher.

Provides the public :func:`identify_extremes` entry point, which validates the
extreme-detection parameters and delegates to one of the concrete methods
(global constant-in-time threshold or day-of-year threshold) based on the
``method_extreme`` argument.
"""

from typing import Dict, Literal, Optional, Tuple

import xarray as xr

from ..core.compute_mode import Materialiser
from ..core.dimensions import horizontal_dims
from ..core.time_axis import SeasonalCycle, resolve_cycle
from ..core.validation import _infer_dims_coords
from ..exceptions import ConfigurationError
from ..logging_config import configure_logging, get_logger
from .global_percentile import _identify_extremes_constant
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


def identify_extremes(
    da: xr.DataArray,
    method_extreme: Literal["global_percentile", "seasonal_percentile"] = "seasonal_percentile",
    threshold_percentile: float = 95,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    window_days: int = 11,  # for seasonal_percentile
    window_spatial: Optional[int] = None,  # for seasonal_percentile
    method_percentile: Literal["exact", "approximate"] = "approximate",
    precision: float = 0.01,
    max_anomaly: float = 5.0,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
    materialiser: Optional[Materialiser] = None,
    threshold_label: str = "thresholds",
    cycle: Optional[SeasonalCycle] = None,
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
        Window for day-of-year threshold spatial clustering (seasonal_percentile only)
    method_percentile : str, default='approximate'
        Method for percentile computation ('exact' or 'approximate')
    precision : float, default=0.01
        Precision for histogram bins in approximate method
    max_anomaly : float, default=5.0
        Maximum anomaly value for histogram binning

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

    # Validate parameter compatibility for exact percentile method
    if method_percentile == "exact":
        default_precision = 0.01
        default_max_anomaly = 5.0

        # Check if precision parameter was explicitly set to a non-default value
        if precision != default_precision:
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
                    "default_precision": default_precision,
                },
            )

        # Check if max_anomaly parameter was explicitly set to a non-default value
        if max_anomaly != default_max_anomaly:
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
                    "default_max_anomaly": default_max_anomaly,
                },
            )

    # Validate percentile parameter when using approximate method
    if threshold_percentile < 60 and method_percentile == "approximate":
        logger.error(f"Invalid percentile threshold: {threshold_percentile}% with method_percentile='approximate'")
        raise ConfigurationError(
            f"Percentile threshold {threshold_percentile}% is not supported with method_percentile='approximate'",
            details=(
                "Low percentile thresholds (<60%) produce undefined and unsupported behaviour "
                "when using approximate histogram methods"
            ),
            suggestions=[
                "Use method_percentile='exact' for percentiles below 60%",
                "Use a higher percentile threshold (≥60%) with method_percentile='approximate'",
                "Consider if such low percentiles are appropriate for extreme event identification",
            ],
            context={
                "threshold_percentile": threshold_percentile,
                "method_percentile": method_percentile,
                "min_supported_percentile": 60,
            },
        )

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
    # problem it does not have. Same shape as the Phase B `validate_rank` finding.
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

    if method_extreme == "global_percentile":
        logger.debug(f"Global extreme method - method_percentile={method_percentile}")
        return _identify_extremes_constant(
            da, threshold_percentile, method_percentile, dimensions, precision, max_anomaly, materialiser, threshold_label
        )
    elif method_extreme == "seasonal_percentile":
        logger.debug(f"Seasonal percentile method - window_days={window_days}, method_percentile={method_percentile}")

        return _identify_extremes_seasonal(
            da,
            threshold_percentile,
            window_days,
            window_spatial,
            method_percentile,
            dimensions,
            coordinates,
            precision,
            max_anomaly,
            materialiser,
            threshold_label,
            resolved_cycle,
        )
    else:
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
