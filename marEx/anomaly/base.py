"""
Anomaly-computation dispatcher.

Provides the public :func:`compute_normalised_anomaly` entry point, which selects
and delegates to one of the concrete anomaly methods (harmonic detrending,
shifting baseline, fixed baseline, or detrended fixed baseline) based on the
``method_anomaly`` argument.
"""

from typing import Dict, List, Literal, Optional, Tuple

import xarray as xr
from dask.base import is_dask_collection

from ..core.compute_mode import Materialiser
from ..core.time_axis import SeasonalCycle
from ..core.validation import _infer_dims_coords
from ..exceptions import ConfigurationError, create_data_validation_error
from ..logging_config import configure_logging, get_logger
from .fixed_baseline import _compute_anomaly_detrend_fixed_baseline, _compute_anomaly_fixed_baseline
from .harmonic import _compute_anomaly_detrended
from .shifting_baseline import _compute_anomaly_shifting_baseline

# Get module logger
logger = get_logger(__name__)


def compute_normalised_anomaly(
    da: xr.DataArray,
    method_anomaly: Literal[
        "detrend_harmonic", "shifting_baseline", "fixed_baseline", "detrend_fixed_baseline"
    ] = "shifting_baseline",
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    window_years: int = 15,  # for shifting_baseline
    smooth_days: int = 21,  # "
    standardise: bool = False,  # for detrend_harmonic
    detrend_orders: Optional[List[int]] = None,  # "
    force_zero_mean: bool = True,  # "
    reference_period: Optional[Tuple[int, int]] = None,  # for fixed_baseline & detrend_fixed_baseline
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
    materialiser: Optional[Materialiser] = None,
    cycle: Optional[SeasonalCycle] = None,
) -> xr.Dataset:
    """
    Generate normalised anomalies using specified methodology.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with dimensions matching the 'dimensions' parameter
    method_anomaly : str, default='shifting_baseline'
        Anomaly computation method. Options:
        - 'detrend_harmonic': Detrending with harmonics and polynomials (efficient, biased)
        - 'shifting_baseline': Rolling climatology (accurate, shortens time series)
        - 'fixed_baseline': Daily climatology using full time series (keeps long-term trends in the anomaly)
        - 'detrend_fixed_baseline': Polynomial detrending + fixed climatology (does not shorten time series,
          keeps trends in seasonal timing in the anomaly)
    dimensions : dict, optional
        Mapping of conceptual dimensions to actual dimension names in the data
    coordinates : dict, optional
        Mapping of conceptual coordinates to actual coordinate names in the data
    window_years : int, default=15
        Number of years for rolling climatology (shifting_baseline only)
    smooth_days : int, default=21
        Days for smoothing rolling climatology (shifting_baseline only)
    standardise : bool, default=False
        Whether to normalise by 30-day rolling standard deviation (detrend_harmonic only)
    detrend_orders : list, default=[1]
        Polynomial orders for trend removal (detrend_harmonic and detrend_fixed_baseline only)
    force_zero_mean : bool, default=True
        Explicitly enforce zero mean in final anomalies (detrend_harmonic and detrend_fixed_baseline only)
    reference_period : tuple of (int, int), optional
        Year range (start_year, end_year) inclusive for computing the daily climatology
        (fixed_baseline and detrend_fixed_baseline only). If None (default), uses all
        available years. Anomalies are computed for the full time series regardless.

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies, mask, and metadata

    Examples
    --------
    Basic detrended baseline anomaly computation:

    >>> import xarray as xr
    >>> import marEx
    >>>
    >>> # Load chunked SST data
    >>> sst = xr.open_dataset('sst_data.nc', chunks={}).sst.chunk({'time': 30})
    >>>
    >>> # Compute anomalies using shifting baseline (default)
    >>> result = marEx.compute_normalised_anomaly(sst)
    >>> print(result.data_vars)
    Data variables:
        dat_anomaly  (time, lat, lon) float32 dask.array<chunksize=(30, 180, 360)>
        mask         (lat, lon) bool dask.array<chunksize=(180, 360)>

    >>> # Check that anomalies have approximately zero mean
    >>> print(f"Mean anomaly: {result.dat_anomaly.mean().compute():.6f}")
    Mean anomaly: 0.000023

    Previous configuration (marEx v2.0 default) of detrended baseline with higher-order polynomials and standardisation.
    Note: marEx v3.0+ uses shifting_baseline as the default method:

    >>> result_advanced = marEx.compute_normalised_anomaly(
    ...     sst,
    ...     method_anomaly="detrend_harmonic",
    ...     detrend_orders=[1, 2, 3],  # Linear, quadratic, cubic trends
    ...     standardise=True,        # Add standardised anomalies
    ...     force_zero_mean=True
    ... )
    >>> print(result_advanced.data_vars)
    Data variables:
        dat_anomaly  (time, lat, lon) float32 dask.array<chunksize=(30, 180, 360)>
        mask         (lat, lon) bool dask.array<chunksize=(180, 360)>
        dat_stn      (time, lat, lon) float32 dask.array<chunksize=(30, 180, 360)>
        STD          (dayofyear, lat, lon) float32 dask.array<chunksize=(366, 180, 360)>

    >>> # Standardised anomalies have unit variance
    >>> print(f"STD of standardised anomalies: {result_advanced.dat_stn.std().compute():.3f}")

    Accurate shifting baseline method for climate-aware anomalies:

    >>> result_shifting = marEx.compute_normalised_anomaly(
    ...     sst,
    ...     method_anomaly="shifting_baseline",
    ...     window_years=10,   # Use 10-year rolling climatology
    ...     smooth_days=31    # 31-day smoothing window
    ... )
    >>> # Anomalies computed relative to recent past climatology

    Processing unstructured data:

    >>> # ICON ocean model with ncells dimension
    >>> icon_data = xr.open_dataset('icon_sst.nc', chunks={}).to.chunk({'time': 25})
    >>> result_unstructured = marEx.compute_normalised_anomaly(
    ...     icon_data,
    ...     dimensions={"time": "time", "x": "ncells"}
    ...     coordinates={"time": "time", "x": "lon", "y": "lat"},
    ... )
    >>> print(result_unstructured.dims)
    Frozen({'time': 1461, 'ncells': 83886})

    Comparison of methods - detrended vs shifting baseline:

    >>> # Detrended baseline - faster, slight bias
    >>> detrended = marEx.compute_normalised_anomaly(
    ...     sst, method_anomaly="detrend_harmonic"
    ... )
    >>>
    >>> # Shifting baseline - slower, more accurate
    >>> shifting = marEx.compute_normalised_anomaly(
    ...     sst, method_anomaly="shifting_baseline",
    ...     window_years=15
    ... )
    >>>
    >>> # Compare anomaly magnitudes
    >>> print(f"Detrended RMS: {detrended.dat_anomaly.std().compute():.3f}")
    >>> print(f"Shifting RMS: {shifting.dat_anomaly.std().compute():.3f}")

    Fixed baseline climatology:

    >>> # Use full time series for daily climatology
    >>> result_fixed = marEx.compute_normalised_anomaly(
    ...     sst,
    ...     method_anomaly="fixed_baseline"
    ... )
    >>> # Climatology computed from all available years

    Fixed baseline with a restricted reference period:

    >>> # Compute climatology from 1990-2020 only, but output anomalies for all years
    >>> result_ref = marEx.compute_normalised_anomaly(
    ...     sst,
    ...     method_anomaly="fixed_baseline",
    ...     reference_period=(1990, 2020)
    ... )

    Fixed detrended baseline:

    >>> # Remove long-term trends then compute fixed climatology
    >>> result_fixed_detrended = marEx.compute_normalised_anomaly(
    ...     sst,
    ...     method_anomaly="detrend_fixed_baseline",
    ...     detrend_orders=[1],  # Remove linear trend
    ...     force_zero_mean=True
    ... )
    >>> # Combines trend removal with fixed climatology
    """
    # A None materialiser means "default to persist mode", which keeps every existing
    # caller, doctest and test working unchanged.
    if materialiser is None:
        materialiser = Materialiser("persist")

    # Set default values for mutable parameters
    if detrend_orders is None:
        detrend_orders = [1]

    # Configure logging if verbose/quiet parameters are provided
    if verbose is not None or quiet is not None:
        configure_logging(verbose=verbose, quiet=quiet)

    logger.debug(f"Computing normalised anomaly using {method_anomaly} method")

    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    # Same guard as the ``anomaly.compute`` stage entry point. This function is a public
    # entry point in its own right, and without the check a numpy-backed array fails deep
    # inside a method-specific chunking call with an incidental KeyError/TypeError that
    # names neither the input nor the fix.
    if not is_dask_collection(da.data):
        logger.error("Input DataArray is not Dask-backed - anomaly computation requires chunked data")
        raise create_data_validation_error(
            "Input DataArray must be Dask-backed",
            details="Anomaly computation requires chunked data for efficient computation",
            suggestions=[
                "Convert to Dask array: da = da.chunk({'time': 30})",
                "Load with chunking: xr.open_dataset('file.nc', chunks={'time': 30})",
            ],
            data_info={"data_type": type(da.data).__name__, "shape": da.shape},
        )

    # Validate reference_period is only used with compatible methods
    if reference_period is not None and method_anomaly not in ("fixed_baseline", "detrend_fixed_baseline"):
        raise ConfigurationError(
            f"reference_period is not supported for method_anomaly='{method_anomaly}'",
            details="reference_period is only applicable to 'fixed_baseline' and 'detrend_fixed_baseline' methods",
            suggestions=[
                "Remove the reference_period parameter, or",
                "Use method_anomaly='fixed_baseline' or 'detrend_fixed_baseline'",
            ],
        )

    if method_anomaly == "detrend_harmonic":
        logger.debug(
            f"Detrended baseline parameters: standardise={standardise}, orders={detrend_orders}, zero_mean={force_zero_mean}"
        )
        return _compute_anomaly_detrended(da, standardise, detrend_orders, dimensions, coordinates, force_zero_mean, cycle=cycle)
    elif method_anomaly == "shifting_baseline":
        logger.debug(f"Shifting baseline parameters: window_years={window_years}, smooth_days={smooth_days}")
        return _compute_anomaly_shifting_baseline(da, window_years, smooth_days, dimensions, coordinates, cycle)
    elif method_anomaly == "fixed_baseline":
        logger.debug(f"Fixed baseline parameters: reference_period={reference_period}")
        return _compute_anomaly_fixed_baseline(da, dimensions, coordinates, reference_period, materialiser, cycle)
    elif method_anomaly == "detrend_fixed_baseline":
        logger.debug(
            f"Fixed detrended baseline parameters: orders={detrend_orders}, "
            f"zero_mean={force_zero_mean}, reference_period={reference_period}"
        )
        return _compute_anomaly_detrend_fixed_baseline(
            da, detrend_orders, dimensions, coordinates, force_zero_mean, reference_period, materialiser, cycle
        )
    else:
        logger.error(f"Unknown anomaly method: {method_anomaly}")
        raise ConfigurationError(
            f"Unknown anomaly method '{method_anomaly}'",
            details="Invalid method_anomaly parameter",
            suggestions=[
                "Use 'detrend_harmonic' for efficient processing with trend and harmonic removal",
                "Use 'shifting_baseline' for accurate climatology (requires more data)",
                "Use 'fixed_baseline' to remove a single daily climatology across all years "
                "(keeps any long-term trend in the anomaly)",
                "Use 'detrend_fixed_baseline' for trend removal followed by fixed climatology",
            ],
            context={
                "provided_method": method_anomaly,
                "valid_methods": ["detrend_harmonic", "shifting_baseline", "fixed_baseline", "detrend_fixed_baseline"],
            },
        )
