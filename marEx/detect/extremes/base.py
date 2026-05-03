"""
Extreme-identification dispatcher.

Provides the public :func:`identify_extremes` entry point, which validates the
extreme-detection parameters and delegates to one of the concrete methods
(global constant-in-time threshold or Hobday day-of-year threshold) based on the
``method_extreme`` argument.
"""

from typing import Dict, Literal, Optional, Tuple

import xarray as xr

from ...exceptions import ConfigurationError
from ...logging_config import configure_logging, get_logger
from ..validation import _infer_dims_coords
from .global_extreme import _identify_extremes_constant
from .hobday import _identify_extremes_hobday

# Get module logger
logger = get_logger(__name__)


def identify_extremes(
    da: xr.DataArray,
    method_extreme: Literal["global_extreme", "hobday_extreme"] = "hobday_extreme",
    threshold_percentile: float = 95,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    window_days_hobday: int = 11,  # for hobday_extreme
    window_spatial_hobday: Optional[int] = None,  # for hobday_extreme
    method_percentile: Literal["exact", "approximate"] = "approximate",
    precision: float = 0.01,
    max_anomaly: float = 5.0,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events exceeding a percentile threshold using specified method.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing anomalies
    method_extreme : str, default='hobday_extreme'
        Method for threshold calculation ('global_extreme' or 'hobday_extreme')
    threshold_percentile : float, default=95
        Percentile threshold (e.g., 95 for 95th percentile)
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    window_days_hobday : int, default=11
        Window for day-of-year threshold (hobday_extreme only)
    window_spatial_hobday : int, default=None
        Window for day-of-year threshold spatial clustering (hobday_extreme only)
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
    ...     method_extreme="global_extreme",
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
    >>> extremes_hobday, thresholds_hobday = marEx.identify_extremes(
    ...     anomalies,
    ...     method_extreme="hobday_extreme",
    ...     threshold_percentile=95,
    ...     window_days_hobday=11  # 11-day window around each day-of-year
    ...     window_spatial_hobday=3  # 3x3 spatial window for clustering percentile calcuation
    ... )
    >>> print(f"Hobday thresholds shape: {thresholds_hobday.shape}")
    Hobday thresholds shape: (366, 180, 360)

    >>> # Compare seasonal variation in thresholds
    >>> summer_threshold = thresholds_hobday.sel(dayofyear=200).mean()
    >>> winter_threshold = thresholds_hobday.sel(dayofyear=50).mean()
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

    Advanced Hobday method with custom temporal window:

    >>> # Longer temporal window for smoother thresholds
    >>> extremes_smooth, thresholds_smooth = marEx.identify_extremes(
    ...     anomalies,
    ...     method_extreme="hobday_extreme",
    ...     window_days_hobday=31,  # Longer smoothing window
    ...     threshold_percentile=95
    ... )
    >>>
    >>> # Compare threshold smoothness
    >>> std_11day = thresholds_hobday.std(dim='dayofyear').mean().compute()
    >>> std_31day = thresholds_smooth.std(dim='dayofyear').mean().compute()
    >>> print(f"Threshold variability: 11-day={std_11day:.3f}, 31-day={std_31day:.3f}")
    """
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

    # Validate window_spatial_hobday parameter
    if window_spatial_hobday is not None:
        # Check if window_spatial_hobday is specified for unstructured grid
        has_y_dim = "y" in dimensions and dimensions["y"] in da.dims

        if not has_y_dim:
            logger.error(f"window_spatial_hobday={window_spatial_hobday} specified for unstructured grid")
            raise ConfigurationError(
                "window_spatial_hobday is not supported for unstructured grids",
                details=(
                    "Spatial smoothing with window_spatial_hobday requires structured grids with both x and y dimensions. "
                    "Unstructured grids do not support spatial window operations due to computational and memory "
                    "limitations of the algorithms."
                ),
                suggestions=[
                    "Remove the window_spatial_hobday parameter for unstructured grids",
                    "Use structured grid data if spatial smoothing is required",
                    "Set window_spatial_hobday=None to use default behavior",
                ],
                context={
                    "grid_type": "unstructured",
                    "window_spatial_hobday": window_spatial_hobday,
                    "dimensions": dimensions,
                    "available_dims": list(da.dims),
                },
            )

        # Check if window_spatial_hobday is specified when hobday_extreme is not used
        if method_extreme != "hobday_extreme":
            logger.error(f"window_spatial_hobday={window_spatial_hobday} specified with method_extreme='{method_extreme}'")
            raise ConfigurationError(
                "window_spatial_hobday can only be used with method_extreme='hobday_extreme'",
                details=(
                    "The window_spatial_hobday parameter is only implemented for the Hobday extreme method. "
                    "Other extreme methods do not support spatial smoothing due to computational and memory "
                    "limitations of the algorithms."
                ),
                suggestions=[
                    "Remove the window_spatial_hobday parameter when using method_extreme='global_extreme'",
                    "Use method_extreme='hobday_extreme' if spatial smoothing is required",
                    "Set window_spatial_hobday=None to use default behavior",
                ],
                context={
                    "method_extreme": method_extreme,
                    "window_spatial_hobday": window_spatial_hobday,
                    "compatible_methods": ["hobday_extreme"],
                },
            )

        # Check if window_spatial_hobday is specified when method_percentile is "exact"
        if method_percentile == "exact":
            logger.error(f"window_spatial_hobday={window_spatial_hobday} specified with method_percentile='exact'")
            raise ConfigurationError(
                "window_spatial_hobday is not supported with method_percentile='exact'",
                details=(
                    "The window_spatial_hobday parameter is only implemented for the approximate percentile method. "
                    "Exact percentile computation does not support spatial smoothing due to computational and memory "
                    "limitations of the algorithms."
                ),
                suggestions=[
                    "Remove the window_spatial_hobday parameter when using method_percentile='exact'",
                    "Use method_percentile='approximate' if spatial smoothing is required",
                    "Set window_spatial_hobday=None to use default behavior",
                ],
                context={
                    "method_percentile": method_percentile,
                    "window_spatial_hobday": window_spatial_hobday,
                    "compatible_methods": ["approximate"],
                },
            )

    # Validate that window parameters are odd numbers (only for hobday_extreme method)
    if method_extreme == "hobday_extreme" and window_days_hobday is not None and window_days_hobday % 2 == 0:
        logger.error(f"window_days_hobday={window_days_hobday} is not an odd number")
        raise ConfigurationError(
            "window_days_hobday must be an odd number",
            details=(
                f"Window parameters require odd numbers to ensure symmetric windows around a central point. "
                f"window_days_hobday={window_days_hobday} is even, which would create asymmetric temporal windows."
            ),
            suggestions=[
                f"Use window_days_hobday={window_days_hobday + 1} or {window_days_hobday - 1}",
                "Choose an odd number",
            ],
            context={
                "window_days_hobday": window_days_hobday,
                "is_odd": False,
            },
        )

    # Set default spatial window (only for the approximate hobday_extreme method). The
    # exact percentile path ignores window_spatial_hobday entirely, and the validation
    # above rejects it when user-supplied with method_percentile='exact', so it must not
    # be silently defaulted there (which only inflated N_samples and hid the warning).
    if (
        method_extreme == "hobday_extreme"
        and method_percentile != "exact"
        and window_spatial_hobday is None
        and "y" in dimensions
        and dimensions["y"] in da.dims
    ):
        window_spatial_hobday = 5  # Default to 5x5 spatial window for structured grids

    if method_extreme == "hobday_extreme" and window_spatial_hobday is not None and window_spatial_hobday % 2 == 0:
        logger.error(f"window_spatial_hobday={window_spatial_hobday} is not an odd number")
        raise ConfigurationError(
            "window_spatial_hobday must be an odd number",
            details=(
                f"Window parameters require odd numbers to ensure symmetric windows around a central point. "
                f"window_spatial_hobday={window_spatial_hobday} is even, which would create asymmetric spatial windows."
            ),
            suggestions=[
                f"Use window_days_hobday={window_days_hobday + 1} or {window_days_hobday - 1}",
                "Choose an odd number.",
            ],
            context={
                "window_spatial_hobday": window_spatial_hobday,
                "is_odd": False,
            },
        )

    if method_extreme == "global_extreme":
        logger.debug(f"Global extreme method - method_percentile={method_percentile}")
        return _identify_extremes_constant(da, threshold_percentile, method_percentile, dimensions, precision, max_anomaly)
    elif method_extreme == "hobday_extreme":
        logger.debug(f"Hobday extreme method - window_days={window_days_hobday}, method_percentile={method_percentile}")

        return _identify_extremes_hobday(
            da,
            threshold_percentile,
            window_days_hobday,
            window_spatial_hobday,
            method_percentile,
            dimensions,
            coordinates,
            precision,
            max_anomaly,
        )
    else:
        logger.error(f"Unknown extreme method: {method_extreme}")
        raise ConfigurationError(
            f"Unknown extreme method '{method_extreme}'",
            details="Invalid method_extreme parameter",
            suggestions=[
                "Use 'global_extreme' for efficient constant percentile threshold",
                "Use 'hobday_extreme' for day-of-year specific thresholds",
            ],
            context={
                "provided_method": method_extreme,
                "valid_methods": ["global_extreme", "hobday_extreme"],
            },
        )
