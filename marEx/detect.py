"""
MarEx-Detect: Marine Extremes Detection Module

Preprocessing toolkit for marine extremes identification from scalar oceanographic data.
Converts raw time series into standardised anomalies and identifies extreme events
(e.g., Marine Heatwaves using Sea Surface Temperature).

Core capabilities:
- Two preprocessing methodologies: Detrended Baseline and Shifting Baseline
- Two definitions for extreme events: Global Extreme and Hobday Extreme
- Threshold-based extreme event identification
- Efficient processing of both structured (gridded) and unstructured data

Compatible data formats:
- Structured data:   3D arrays (time, lat, lon)
- Unstructured data: 2D arrays (time, cell)
"""

import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import dask
import dask.array as da
import flox.xarray
import numpy as np
import pandas as pd
import xarray as xr
from dask.base import is_dask_collection
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from xhistogram.xarray import histogram

# Coordinate validation imports removed
from .exceptions import (
    ConfigurationError,
    CoordinateError,
    DataValidationError,
    ProcessingError,
    create_coordinate_error,
    create_data_validation_error,
    create_processing_error,
)
from .helper import fix_dask_tuple_array
from .logging_config import configure_logging, get_logger, log_dask_info, log_memory_usage, log_progress, log_timing, progress_bar

# Get module logger
logger = get_logger(__name__)

# Suppress noisy distributed logging
logging.getLogger("distributed.shuffle._scheduler_plugin").setLevel(logging.ERROR)


# ============================
# Validation Functions
# ============================


def _validate_dimensions_exist(da: xr.DataArray, dimensions: Dict[str, str]) -> None:
    """
    Validate that all specified dimensions exist in the dataset.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array to validate
    dimensions : dict
        Mapping of conceptual dimensions to actual dimension names

    Raises
    ------
    DataValidationError
        If any specified dimension does not exist in the dataset
    """
    missing_dims = []
    for concept_dim, actual_dim in dimensions.items():
        if actual_dim not in da.dims:
            missing_dims.append(f"'{actual_dim}' (for {concept_dim})")

    if missing_dims:
        available_dims = list(da.dims)
        raise create_data_validation_error(
            f"Missing required dimensions: {', '.join(missing_dims)}",
            details=f"Dataset has dimensions: {available_dims}",
            suggestions=[
                "Check dimension names in your data",
                "Update the 'dimensions' parameter to match your data structure",
                f"Available dimensions: {available_dims}",
            ],
            data_info={
                "missing_dimensions": missing_dims,
                "available_dimensions": available_dims,
                "provided_dimensions": dimensions,
            },
        )


def _validate_coordinates_exist(da: xr.DataArray, coordinates: Dict[str, str]) -> None:
    """
    Validate that all specified coordinates exist in the dataset.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array to validate
    coordinates : dict
        Mapping of conceptual coordinates to actual coordinate names

    Raises
    ------
    DataValidationError
        If any specified coordinate does not exist in the dataset
    """
    missing_coords = []
    for concept_coord, actual_coord in coordinates.items():
        if actual_coord not in da.coords:
            missing_coords.append(f"'{actual_coord}' (for {concept_coord})")

    if missing_coords:
        available_coords = list(da.coords.keys())
        raise create_data_validation_error(
            f"Missing required coordinates: {', '.join(missing_coords)}",
            details=f"Dataset has coordinates: {available_coords}",
            suggestions=[
                "Check coordinate names in your data",
                "Update the 'coordinates' parameter to match your data structure",
                f"Available coordinates: {available_coords}",
            ],
            data_info={
                "missing_coordinates": missing_coords,
                "available_coordinates": available_coords,
                "provided_coordinates": coordinates,
            },
        )


# ============================
# Methodology Selection
# ============================


def preprocess_data(
    da: xr.DataArray,
    method_anomaly: Literal["detrended_baseline", "shifting_baseline"] = "detrended_baseline",
    method_extreme: Literal["global_extreme", "hobday_extreme"] = "global_extreme",
    threshold_percentile: float = 95,
    window_year_baseline: int = 15,  # for shifting_baseline
    smooth_days_baseline: int = 21,  # "
    window_days_hobday: int = 11,  # for hobday_extreme
    std_normalise: bool = False,  # for detrended_baseline
    detrend_orders: List[int] = [1],  # "
    force_zero_mean: bool = True,  # "
    exact_percentile: bool = False,
    dask_chunks: Dict[str, int] = {"time": 25},
    dimensions: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
    coordinates: Optional[Dict[str, str]] = None,
    neighbours: Optional[xr.DataArray] = None,
    cell_areas: Optional[xr.DataArray] = None,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> xr.Dataset:
    """
    Complete preprocessing pipeline for marine extreme event identification.

    Supports separate methods for anomaly computation and extreme identification:

    Anomaly Methods:
    - 'detrended_baseline': Detrending with harmonics and polynomials -- more efficient, but biases statistics
    - 'shifting_baseline': Rolling climatology using previous window_year_baseline years -- more "correct", but shortens time series by window_year_baseline years

    Extreme Methods:
    - 'global_extreme': Global-in-time threshold value
    - 'hobday_extreme': Local day-of-year specific thresholds with windowing

    Parameters
    ----------
    da : xarray.DataArray
        Raw input data

    Method Selection:
    method_anomaly : str, optional
        Anomaly computation method ('detrended_baseline' or 'shifting_baseline')
    method_extreme : str, optional
        Extreme identification method ('global_extreme' or 'hobday_extreme')

    General Parameters:
    threshold_percentile : float, optional
        Percentile threshold for extreme event identification
    dask_chunks : dict, optional
        Chunking specification for distributed computation
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    neighbours : xarray.DataArray, optional
        Neighbour connectivity for spatial clustering (optional)
    cell_areas : xarray.DataArray, optional
        Cell areas for weighted spatial statistics (optional)

    Shifting Baseline Method Parameters:
    window_year_baseline : int, optional
        Number of years for rolling climatology (shifting_baseline method only)
    smooth_days_baseline : int, optional
        Days for smoothing rolling climatology (shifting_baseline method only)

    Hobday Extreme Method Parameters:
    window_days_hobday : int, optional
        Window for day-of-year threshold calculation (hobday_extreme method only)

    Detrended Baseline Method Parameters:
    std_normalise : bool, optional
        Whether to standardise anomalies by rolling standard deviation (detrended_baseline only)
    detrend_orders : list, optional
        Polynomial orders for detrending (detrended_baseline method only)
    force_zero_mean : bool, optional
        Whether to enforce zero mean in detrended anomalies (detrended_baseline method only)

    Extreme Method Parameters:
    exact_percentile : bool, optional
        Whether to use exact or approximate percentile calculation (both global_extreme & hobday_extreme methods)
        N.B. Using exact percentile calculation requires both careful/thoughtful chunking & sufficient memory.


    Logging Parameters:
    verbose : bool, optional
        Enable verbose logging with detailed progress information.
        If None, uses current global logging configuration.
    quiet : bool, optional
        Enable quiet logging with minimal output (warnings and errors only).
        If None, uses current global logging configuration.
        Note: quiet takes precedence over verbose if both are True.

    Returns
    -------
    xarray.Dataset
        Processed dataset with anomalies and extreme event identification

    Examples
    --------
    Basic usage with gridded SST data for marine heatwave detection:

    >>> import xarray as xr
    >>> import marEx
    >>>
    >>> # Load and chunk SST data
    >>> sst = xr.open_dataset('sst_data.nc', chunks={}).sst.chunk({'time': 30})
    >>>
    >>> # Basic preprocessing with default detrended baseline method
    >>> result = marEx.preprocess_data(sst, threshold_percentile=90)
    >>> print(result)
    <xarray.Dataset>
    Dimensions:         (time: 1461, lat: 180, lon: 360)
    Data variables:
        dat_anomaly     (time, lat, lon) float32 dask.array<chunksize=(30, 180, 360)>
        mask            (lat, lon) bool dask.array<chunksize=(180, 360)>
        extreme_events  (time, lat, lon) bool dask.array<chunksize=(30, 180, 360)>
        thresholds      (lat, lon) float32 dask.array<chunksize=(180, 360)>

    >>> # Check which locations have extreme events
    >>> print(f"Total extreme events: {result.extreme_events.sum().compute()}")
    Total extreme events: 15847

    Using shifting baseline method for more accurate climatology:

    >>> # Requires at least 15 years of data by default
    >>> result_shifting = marEx.preprocess_data(
    ...     sst,
    ...     method_anomaly="shifting_baseline",
    ...     window_year_baseline=10,  # Use shorter window if needed
    ...     smooth_days_baseline=31   # Longer smoothing window
    ... )
    >>> # Note: First 10 years will be removed from output

    Using Hobday extreme method with day-of-year specific thresholds:

    >>> result_hobday = marEx.preprocess_data(
    ...     sst,
    ...     method_extreme="hobday_extreme",
    ...     window_days_hobday=11,  # 11-day window for each day-of-year
    ...     threshold_percentile=95
    ... )
    >>> print(result_hobday.thresholds.dims)
    ('dayofyear', 'lat', 'lon')

    Previous configuration (marEx v2.0 default) with polynomial detrending and standardisation:

    >>> result_advanced = marEx.preprocess_data(
    ...     sst,
    ...     method_anomaly="detrended_baseline",
    ...     detrend_orders=[1, 2],  # Linear and quadratic trends
    ...     std_normalise=True,     # Standardise by rolling std
    ...     force_zero_mean=True,
    ...     threshold_percentile=95
    ... )
    >>> # Result includes both raw and standardised anomalies
    >>> print('dat_stn' in result_advanced)
    True

    Processing unstructured data:

    >>> # For ICON ocean model data
    >>> icon_sst = xr.open_dataset('icon_sst.nc', chunks={}).to.chunk({'time': 50})
    >>> result_unstructured = marEx.preprocess_data(
    ...     icon_sst,
    ...     dimensions={"x": "ncells"},   # Must specify the name of the spatial dimension
    ...     dask_chunks={"time": 50}
    ... )

    Error handling - insufficient data for shifting baseline:

    >>> short_data = sst.isel(time=slice(0, 1000))  # Only ~3 years
    >>> try:
    ...     result = marEx.preprocess_data(
    ...         short_data,
    ...         method_anomaly="shifting_baseline",
    ...         window_year_baseline=15
    ...     )
    ... except ValueError as e:
    ...     print(f"Error: {e}")
    Error: Insufficient data for shifting_baseline method. Dataset spans 3 years but window_year_baseline requires at least 15 years.

    Performance considerations with chunking:

    >>> # For large datasets, adjust chunking for memory management
    >>> large_sst = sst.chunk({"time": 25, "lat": 90, "lon": 180})
    >>> result = marEx.preprocess_data(
    ...     large_sst,
    ...     dask_chunks={"time": 25},
    ...     exact_percentile=False  # Use approximate method for long time-series calculations
    ... )

    Integration with tracking workflow:

    >>> # Preprocess data then track events
    >>> processed = marEx.preprocess_data(sst, threshold_percentile=95)
    >>> tracker = marEx.tracker(
    ...     processed.extreme_events,
    ...     processed.mask,
    ...     R_fill=8,
    ...     area_filter_quartile=0.5
    ... )
    >>> events = tracker.run()
    >>> print(f"Identified {events.event.max().compute()} distinct events")
    """
    # Configure logging if verbose/quiet parameters are provided
    if verbose is not None or quiet is not None:
        configure_logging(verbose=verbose, quiet=quiet)

    # Log preprocessing start with parameters
    logger.info(f"Starting data preprocessing - Method: {method_anomaly} -> {method_extreme}")
    logger.info(f"Parameters: percentile={threshold_percentile}%, exact={exact_percentile}")
    logger.debug(
        f"Anomaly method parameters: window_year={window_year_baseline}, smooth_days={smooth_days_baseline}, "
        + f"std_normalise={std_normalise}, detrend_orders={detrend_orders}, force_zero_mean={force_zero_mean}"
    )
    logger.debug(f"Extreme method parameters: window_days_hobday={window_days_hobday}")

    # Log input data info
    log_dask_info(logger, da, "Input data")
    log_memory_usage(logger, "Initial memory state")

    # Handle coordinates parameter based on data structure
    if coordinates is None:
        if "y" not in dimensions:
            # Unstructured (2D) data - requires explicit coordinate specification
            logger.error("Coordinates parameter required for unstructured data")
            raise create_data_validation_error(
                "Coordinates parameter must be explicitly specified for unstructured data",
                details="Unstructured data requires coordinate names for x and y spatial coordinates",
                suggestions=[
                    "Specify coordinates parameter with spatial coordinate names",
                    "Example: coordinates={'time': 'time', 'x': 'lon', 'y': 'lat'}",
                    f"Your x dimension '{dimensions['x']}' needs associated coordinate names",
                ],
                data_info={
                    "data_structure": "unstructured (2D)",
                    "dimensions": dimensions,
                    "missing_coordinates": "x and y spatial coordinates",
                },
            )
        else:
            # Gridded (3D) data - copy dimensions to coordinates
            coordinates = dimensions.copy()
            logger.debug("Gridded data detected - copying dimensions to coordinates")

    # Validate dimensions and coordinates exist in dataset
    logger.debug("Validating dimensions and coordinates")
    _validate_dimensions_exist(da, dimensions)
    _validate_coordinates_exist(da, coordinates)

    # Check if input data is dask-backed
    if not is_dask_collection(da.data):
        logger.error("Input DataArray is not Dask-backed - preprocessing requires chunked data")
        raise create_data_validation_error(
            "Input DataArray must be Dask-backed",
            details="Preprocessing requires chunked data for efficient computation",
            suggestions=[
                "Convert to Dask array: da = da.chunk({'time': 30})",
                "Load with chunking: xr.open_dataset('file.nc', chunks={'time': 30})",
            ],
            data_info={"data_type": type(da.data).__name__, "shape": da.shape},
        )

    logger.debug("Enabling Dask large chunk splitting for preprocessing")
    dask.config.set({"array.slicing.split_large_chunks": True})

    # Step 1: Compute anomalies
    with log_timing(
        logger,
        f"Anomaly computation using {method_anomaly} method",
        log_memory=True,
        show_progress=True,
    ):
        logger.debug(
            f"Computing anomalies with parameters: method={method_anomaly}, "
            f"std_normalise={std_normalise}, force_zero_mean={force_zero_mean}"
        )
        ds = compute_normalised_anomaly(
            da.astype(np.float32),
            method_anomaly,
            dimensions,
            coordinates,
            window_year_baseline,
            smooth_days_baseline,
            std_normalise,
            detrend_orders,
            force_zero_mean,
        )
        log_memory_usage(logger, "After anomaly computation", logging.DEBUG)

    # For shifting baseline, remove first window_year_baseline years (insufficient climatology data)
    if method_anomaly == "shifting_baseline":
        min_year = int(ds[coordinates["time"]].dt.year.min().values.item())
        max_year = int(ds[coordinates["time"]].dt.year.max().values.item())
        total_years = max_year - min_year + 1

        logger.info(f"Shifting baseline data validation: {total_years} years available ({min_year}-{max_year})")

        if total_years < window_year_baseline:
            logger.error(f"Insufficient data: {total_years} years < {window_year_baseline} required")
            raise create_data_validation_error(
                "Insufficient data for shifting_baseline method",
                details=f"Dataset spans {total_years} years but requires at least {window_year_baseline} years",
                suggestions=[
                    "Use more years of data to meet minimum requirement",
                    f"Reduce window_year_baseline parameter (currently {window_year_baseline})",
                    "Consider using detrended_baseline method instead",
                ],
                data_info={
                    "available_years": int(total_years),
                    "required_years": int(window_year_baseline),
                },
            )

        start_year = int(min_year + window_year_baseline)
        logger.info(f"Trimming data to start from {start_year} (removing first {window_year_baseline} years)")
        time_sel = ds[coordinates["time"]].dt.year >= start_year
        ds = ds.isel({coordinates["time"]: time_sel})

    anomalies = ds.dat_anomaly

    # Step 2: Identify extreme events (both methods now return consistent tuple structures)
    with log_timing(
        logger,
        f"Extreme event identification using {method_extreme} method",
        log_memory=True,
        show_progress=True,
    ):
        logger.debug(
            f"Identifying extremes with parameters: method={method_extreme}, "
            f"percentile={threshold_percentile}%, exact={exact_percentile}"
        )
        extremes, thresholds = identify_extremes(
            anomalies,
            method_extreme,
            threshold_percentile,
            dimensions,
            coordinates,
            window_days_hobday,
            exact_percentile,
        )
        log_memory_usage(logger, "After extreme identification", logging.DEBUG)

    # Add extreme events and thresholds to dataset
    ds["extreme_events"] = extremes
    ds["thresholds"] = thresholds

    # Handle standardised anomalies if requested (only for detrended_baseline)
    if std_normalise and method_anomaly == "detrended_baseline":
        logger.info("Processing standardised anomalies for extreme identification")
        with log_timing(
            logger,
            "Standardised extreme identification",
            log_memory=True,
            show_progress=True,
        ):
            extremes_stn, thresholds_stn = identify_extremes(
                ds.dat_stn,
                method_extreme,
                threshold_percentile,
                dimensions,
                coordinates,
                window_days_hobday,
                exact_percentile,
            )
            ds["extreme_events_stn"] = extremes_stn
            ds["thresholds_stn"] = thresholds_stn

    # Add optional spatial metadata
    if neighbours is not None:
        logger.debug("Adding neighbour connectivity data")
        chunk_dict = {dim: -1 for dim in neighbours.dims}
        ds["neighbours"] = neighbours.astype(np.int32).chunk(chunk_dict)
        if "nv" in neighbours.dims:
            ds = ds.assign_coords(nv=neighbours.nv)

    if cell_areas is not None:
        logger.debug("Adding cell area data")
        chunk_dict = {dim: -1 for dim in cell_areas.dims}
        ds["cell_areas"] = cell_areas.astype(np.float32).chunk(chunk_dict)

    # Add processing parameters to metadata
    ds.attrs.update(
        {
            "method_anomaly": method_anomaly,
            "method_extreme": method_extreme,
            "threshold_percentile": threshold_percentile,
            "preprocessing_steps": _get_preprocessing_steps(
                method_anomaly,
                method_extreme,
                std_normalise,
                detrend_orders,
                window_year_baseline,
                smooth_days_baseline,
                window_days_hobday,
            ),
        }
    )

    # Add method-specific parameters
    if method_anomaly == "detrended_baseline":
        ds.attrs.update(
            {
                "detrend_orders": detrend_orders,
                "force_zero_mean": force_zero_mean,
                "std_normalise": std_normalise,
            }
        )
    elif method_anomaly == "shifting_baseline":
        ds.attrs.update(
            {
                "window_year_baseline": window_year_baseline,
                "smooth_days_baseline": smooth_days_baseline,
            }
        )

    if method_extreme == "hobday_extreme":
        ds.attrs.update({"window_days_hobday": window_days_hobday})

    ds.attrs.update({"exact_percentile": exact_percentile})

    # Final rechunking
    time_chunks = dask_chunks.get(dimensions["time"], dask_chunks.get("time", 10))
    logger.debug(f"Final rechunking with time chunks: {time_chunks}")
    chunk_dict = {dimensions[dim]: -1 for dim in ["x", "y"] if dim in dimensions}
    chunk_dict[dimensions["time"]] = time_chunks
    if method_extreme == "hobday_extreme":
        chunk_dict["dayofyear"] = time_chunks
    ds = ds.chunk(chunk_dict)

    # Fix encoding issue with saving when calendar & units attribute is present
    if "calendar" in ds[coordinates["time"]].attrs:
        logger.debug("Removing calendar attribute for Zarr compatibility")
        del ds[coordinates["time"]].attrs["calendar"]
    if "units" in ds[coordinates["time"]].attrs:
        logger.debug("Removing units attribute for Zarr compatibility")
        del ds[coordinates["time"]].attrs["units"]

    logger.info("Persisting final dataset and optimising task graph")
    with log_timing(
        logger,
        "Dataset persistence and optimisation",
        log_memory=True,
        show_progress=True,
    ):
        ds = ds.persist(optimize_graph=True)
        ds["thresholds"] = ds.thresholds.compute()  # Patch for a dask-Zarr bug that has problems saving this data array...
        ds["dat_anomaly"] = fix_dask_tuple_array(ds.dat_anomaly)

        # Patch for same dask-Zarr bug:
        ds[coordinates["x"]] = ds[coordinates["x"]].compute()
        ds[coordinates["y"]] = ds[coordinates["y"]].compute()
        if "neighbours" in ds.data_vars:
            ds["neighbours"] = ds.neighbours.compute()
        if "cell_areas" in ds.data_vars:
            ds["cell_areas"] = ds.cell_areas.compute()

        log_memory_usage(logger, "After dataset persistence", logging.DEBUG)

    # Final success reporting with summary
    extreme_count = ds.extreme_events.sum()
    if hasattr(extreme_count, "compute"):
        extreme_count = extreme_count.compute()

    logger.info(f"Preprocessing completed successfully - {extreme_count} extreme events identified")
    logger.debug(f"Final dataset shape: {ds.dims}")
    log_dask_info(logger, ds, "Final preprocessed dataset")

    return ds


def _get_preprocessing_steps(
    method_anomaly: str,
    method_extreme: str,
    std_normalise: bool,
    detrend_orders: List[int],
    window_year_baseline: int,
    smooth_days_baseline: int,
    window_days_hobday: int,
) -> List[str]:
    """
    Generate preprocessing steps description based on selected methods.
    """
    steps = []

    if method_anomaly == "detrended_baseline":
        steps.append(f"Removed polynomial trend orders={detrend_orders} & seasonal cycle")
        if std_normalise:
            steps.append("Normalised by 30-day rolling STD")
    elif method_anomaly == "shifting_baseline":
        steps.append(f"Rolling climatology using {window_year_baseline} years")
        steps.append(f"Smoothed with {smooth_days_baseline}-day window")

    # Extreme method steps
    if method_extreme == "global_extreme":
        steps.append("Global percentile threshold applied to all days")
    elif method_extreme == "hobday_extreme":
        steps.append(f"Day-of-year thresholds with {window_days_hobday} day window")

    return steps


def compute_normalised_anomaly(
    da: xr.DataArray,
    method_anomaly: Literal["detrended_baseline", "shifting_baseline"] = "detrended_baseline",
    dimensions: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
    coordinates: Optional[Dict[str, str]] = None,
    window_year_baseline: int = 15,  # for shifting_baseline
    smooth_days_baseline: int = 21,  # "
    std_normalise: bool = False,  # for detrended_baseline
    detrend_orders: List[int] = [1],  # "
    force_zero_mean: bool = True,  # "
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> xr.Dataset:
    """
    Generate normalised anomalies using specified methodology.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with dimensions matching the 'dimensions' parameter

    Method Selection:
    method_anomaly : str, optional
        Anomaly computation method ('detrended_baseline' or 'shifting_baseline')

    General Parameters:
    dimensions : dict, optional
        Mapping of conceptual dimensions to actual dimension names in the data
    coordinates : dict, optional
        Mapping of conceptual coordinates to actual coordinate names in the data

    Shifting Baseline Method Parameters:
    window_year_baseline : int, optional
        Number of years for rolling climatology (shifting_baseline only)
    smooth_days_baseline : int, optional
        Days for smoothing rolling climatology (shifting_baseline only)

    Detrended Baseline Method Parameters:
    std_normalise : bool, optional
        Whether to normalise by 30-day rolling standard deviation (detrended_baseline only)
    detrend_orders : list, optional
        Polynomial orders for trend removal (detrended_baseline only)
    force_zero_mean : bool, optional
        Explicitly enforce zero mean in final anomalies (detrended_baseline only)


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
    >>> # Compute anomalies using detrended baseline (default)
    >>> result = marEx.compute_normalised_anomaly(sst)
    >>> print(result.data_vars)
    Data variables:
        dat_anomaly  (time, lat, lon) float32 dask.array<chunksize=(30, 180, 360)>
        mask         (lat, lon) bool dask.array<chunksize=(180, 360)>

    >>> # Check that anomalies have approximately zero mean
    >>> print(f"Mean anomaly: {result.dat_anomaly.mean().compute():.6f}")
    Mean anomaly: 0.000023

    Previous configuration (marEx v2.0 default) of detrended baseline with higher-order polynomials and standardisation:

    >>> result_advanced = marEx.compute_normalised_anomaly(
    ...     sst,
    ...     method_anomaly="detrended_baseline",
    ...     detrend_orders=[1, 2, 3],  # Linear, quadratic, cubic trends
    ...     std_normalise=True,        # Add standardised anomalies
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
    ...     window_year_baseline=10,   # Use 10-year rolling climatology
    ...     smooth_days_baseline=31    # 31-day smoothing window
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
    ...     sst, method_anomaly="detrended_baseline"
    ... )
    >>>
    >>> # Shifting baseline - slower, more accurate
    >>> shifting = marEx.compute_normalised_anomaly(
    ...     sst, method_anomaly="shifting_baseline",
    ...     window_year_baseline=15
    ... )
    >>>
    >>> # Compare anomaly magnitudes
    >>> print(f"Detrended RMS: {detrended.dat_anomaly.std().compute():.3f}")
    >>> print(f"Shifting RMS: {shifting.dat_anomaly.std().compute():.3f}")
    """
    # Configure logging if verbose/quiet parameters are provided
    if verbose is not None or quiet is not None:
        configure_logging(verbose=verbose, quiet=quiet)

    logger.debug(f"Computing normalised anomaly using {method_anomaly} method")

    # Handle coordinates parameter based on data structure
    if coordinates is None:
        if "y" not in dimensions:
            # Unstructured (2D) data - requires explicit coordinate specification
            logger.error("Coordinates parameter required for unstructured data")
            raise create_data_validation_error(
                "Coordinates parameter must be explicitly specified for unstructured data",
                details="Unstructured data requires coordinate names for x and y spatial coordinates",
                suggestions=[
                    "Specify coordinates parameter with spatial coordinate names",
                    "Example: coordinates={'time': 'time', 'x': 'lon', 'y': 'lat'}",
                    f"Your x dimension '{dimensions['x']}' needs associated coordinate names",
                ],
                data_info={
                    "data_structure": "unstructured (2D)",
                    "dimensions": dimensions,
                    "missing_coordinates": "x and y spatial coordinates",
                },
            )
        else:
            # Gridded (3D) data - copy dimensions to coordinates
            coordinates = dimensions.copy()
            logger.debug("Gridded data detected - copying dimensions to coordinates")

    # Validate dimensions and coordinates exist in dataset
    logger.debug("Validating dimensions and coordinates")
    _validate_dimensions_exist(da, dimensions)
    _validate_coordinates_exist(da, coordinates)

    if method_anomaly == "detrended_baseline":
        logger.debug(
            f"Detrended baseline parameters: std_normalise={std_normalise}, orders={detrend_orders}, zero_mean={force_zero_mean}"
        )
        return _compute_anomaly_detrended(da, std_normalise, detrend_orders, dimensions, coordinates, force_zero_mean)
    elif method_anomaly == "shifting_baseline":
        logger.debug(f"Shifting baseline parameters: window_years={window_year_baseline}, smooth_days={smooth_days_baseline}")
        return _compute_anomaly_shifting_baseline(da, window_year_baseline, smooth_days_baseline, dimensions, coordinates)
    else:
        logger.error(f"Unknown anomaly method: {method_anomaly}")
        raise ConfigurationError(
            f"Unknown anomaly method '{method_anomaly}'",
            details="Invalid method_anomaly parameter",
            suggestions=[
                "Use 'detrended_baseline' for efficient processing",
                "Use 'shifting_baseline' for accurate climatology (requires more data)",
            ],
            context={
                "provided_method": method_anomaly,
                "valid_methods": ["detrended_baseline", "shifting_baseline"],
            },
        )


def identify_extremes(
    da: xr.DataArray,
    method_extreme: Literal["global_extreme", "hobday_extreme"] = "global_extreme",
    threshold_percentile: float = 95,
    dimensions: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
    coordinates: Optional[Dict[str, str]] = None,
    window_days_hobday: int = 11,  # for hobday_extreme
    exact_percentile: bool = False,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events exceeding a percentile threshold using specified method.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing anomalies

    Method Selection:
    method_extreme : str, optional
        Method for threshold calculation ('global_extreme' or 'hobday_extreme')

    General Parameters:
    threshold_percentile : float, optional
        Percentile threshold (e.g., 95 for 95th percentile)
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data

    Hobday Extreme Method Parameters:
    window_days_hobday : int, optional
        Window for day-of-year threshold (hobday_extreme only)

    Global Extreme Method Parameters:
    exact_percentile : bool, optional
        Whether to compute exact percentiles

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
    ...     anomalies, exact_percentile=False
    ... )
    >>>
    >>> # Exact method (slower & memory intensive)
    >>> extremes_exact, thresh_exact = marEx.identify_extremes(
    ...     anomalies, exact_percentile=True
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

    # Handle coordinates parameter based on data structure
    if coordinates is None:
        if "y" not in dimensions:
            # Unstructured (2D) data - requires explicit coordinate specification
            logger.error("Coordinates parameter required for unstructured data")
            raise create_data_validation_error(
                "Coordinates parameter must be explicitly specified for unstructured data",
                details="Unstructured data requires coordinate names for x and y spatial coordinates",
                suggestions=[
                    "Specify coordinates parameter with spatial coordinate names",
                    "Example: coordinates={'time': 'time', 'x': 'lon', 'y': 'lat'}",
                    f"Your x dimension '{dimensions['x']}' needs associated coordinate names",
                ],
                data_info={
                    "data_structure": "unstructured (2D)",
                    "dimensions": dimensions,
                    "missing_coordinates": "x and y spatial coordinates",
                },
            )
        else:
            # Gridded (3D) data - copy dimensions to coordinates
            coordinates = dimensions.copy()
            logger.debug("Gridded data detected - copying dimensions to coordinates")

    # Validate dimensions and coordinates exist in dataset
    logger.debug("Validating dimensions and coordinates")
    _validate_dimensions_exist(da, dimensions)
    _validate_coordinates_exist(da, coordinates)

    # Validate percentile parameter when using approximate method
    if threshold_percentile < 60 and not exact_percentile:
        logger.error(f"Invalid percentile threshold: {threshold_percentile}% with exact_percentile=False")
        raise ConfigurationError(
            f"Percentile threshold {threshold_percentile}% is not supported with exact_percentile=False",
            details="Low percentile thresholds (<60%) produce undefined and unsupported behaviour when using approximate histogram methods",
            suggestions=[
                "Use exact_percentile=True for percentiles below 60%",
                "Use a higher percentile threshold (≥60%) with exact_percentile=False",
                "Consider if such low percentiles are appropriate for extreme event identification",
            ],
            context={
                "threshold_percentile": threshold_percentile,
                "exact_percentile": exact_percentile,
                "min_supported_percentile": 60,
            },
        )

    if method_extreme == "global_extreme":
        logger.debug(f"Global extreme method - exact_percentile={exact_percentile}")
        return _identify_extremes_constant(da, threshold_percentile, exact_percentile, dimensions, coordinates)
    elif method_extreme == "hobday_extreme":
        logger.debug(f"Hobday extreme method - window_days={window_days_hobday}, exact_percentile={exact_percentile}")
        return _identify_extremes_hobday(
            da,
            threshold_percentile,
            window_days_hobday,
            exact_percentile,
            dimensions,
            coordinates,
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


# ===============================================
# Shifting Baseline Anomaly Method (New Method)
# ===============================================


def rolling_climatology(
    da: xr.DataArray,
    window_year_baseline: int = 15,
    dimensions: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
    coordinates: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
) -> xr.DataArray:
    """
    Compute rolling climatology efficiently using flox cohorts.
    Uses the previous `window_year_baseline` years of data and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    window_year_baseline : int, default=15
        Number of years to include in each climatology window
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data

    Returns
    -------
    xarray.DataArray
        Rolling climatology with same shape as input data

    Examples
    --------
    Basic rolling climatology computation:

    >>> import xarray as xr
    >>> import marEx
    >>>
    >>> # Load 20 years of SST data
    >>> sst = xr.open_dataset('sst_data.nc', chunks={}).sst.chunk({'time': 30})
    >>>
    >>> # Compute 15-year rolling climatology
    >>> climatology = marEx.rolling_climatology(sst, window_year_baseline=15)
    >>> print(climatology.shape)
    (7305, 180, 360)  # Same as input
    >>>
    >>> # First 15 years will be NaN (insufficient history)
    >>> print(f"NaN values in first year: {climatology.isel(time=slice(0, 365)).isnull().all().compute()}")
    True

    Shorter window for datasets with limited time span:

    >>> # For datasets with only 10 years, use shorter window
    >>> short_climatology = marEx.rolling_climatology(
    ...     sst, window_year_baseline=5
    ... )
    >>> # First 5 years will be NaN instead of 15

    Processing unstructured data:

    >>> # ICON ocean model data
    >>> icon_sst = xr.open_dataset('icon_sst.nc', chunks={}).to.chunk({'time': 25})
    >>> icon_climatology = marEx.rolling_climatology(
    ...     icon_sst,
    ...     dimensions={"time": "time", "x": "ncells"}
    ...     coordinates={"time": "time", "x": "lon", "y": "lat"}
    ... )
    >>> print(icon_climatology.dims)
    Frozen({'time': 7305, 'ncells': 83886})

    Comparing with fixed climatology:

    >>> # Fixed climatology (traditional approach)
    >>> fixed_clim = sst.groupby(sst.time.dt.dayofyear).mean()
    >>>
    >>> # Rolling climatology (adaptive approach)
    >>> rolling_clim = marEx.rolling_climatology(sst)
    >>>
    >>> # Rolling climatology adapts to climate change
    >>> clim_2000 = rolling_clim.sel(time='2000').mean()
    >>> clim_2020 = rolling_clim.sel(time='2020').mean()
    >>> print(f"Climate change signal: {(clim_2020 - clim_2000).compute():.3f} °C")

    Memory considerations for large datasets:

    >>> # Ensure appropriate chunking for memory efficiency
    >>> large_sst = sst.chunk({'time': 30, 'lat': 45, 'lon': 90})
    >>> large_climatology = marEx.rolling_climatology(large_sst)
    >>> # Output maintains input chunking structure
    """

    time_coord = coordinates["time"]
    time_dim = dimensions["time"]
    original_chunk_dict = {dim: chunks for dim, chunks in zip(da.dims, da.chunks)}

    # Add temporal coordinates
    years = da[time_coord].dt.year
    doys = da[time_coord].dt.dayofyear
    da = da.assign_coords({"year": years, "dayofyear": doys})

    # Get temporal bounds
    year_vals = years.compute().values
    doy_vals = doys.compute().values
    unique_years = np.unique(year_vals)
    min_year = int(unique_years.min().item())

    # Create long-form grouping variables
    # For each time point, determine which target years it contributes to
    contributing_time_indices = []
    contributing_target_years = []
    contributing_dayofyears = []

    for t_idx, (year_val, doy_val) in enumerate(zip(year_vals, doy_vals)):
        # Convert numpy scalars to Python ints to avoid dtype issues
        year_val = int(year_val)
        doy_val = int(doy_val)

        # Find target years this time point contributes to
        # A time point from year Y contributes to target years where:
        # target_year - window_year_baseline <= Y < target_year
        # Which means: Y < target_year <= Y + window_year_baseline
        candidate_targets = unique_years[(unique_years > year_val) & (unique_years <= year_val + window_year_baseline)]

        # Only include target years that have sufficient history
        valid_targets = candidate_targets[candidate_targets >= min_year + window_year_baseline]

        # Add entries for each valid target year
        n_targets = len(valid_targets)
        contributing_time_indices.extend([t_idx] * n_targets)
        contributing_target_years.extend(valid_targets.tolist())
        contributing_dayofyears.extend([doy_val] * n_targets)

    # Convert to numpy arrays with explicit dtypes
    time_indices = np.array(contributing_time_indices, dtype=np.int64)
    target_year_groups = np.array(contributing_target_years, dtype=np.int64)
    dayofyear_groups = np.array(contributing_dayofyears, dtype=np.int64)

    # Create long-form dataset by selecting the contributing time points
    long_form_data = da.isel({time_dim: time_indices})

    # Create a new time dimension for the long-form data
    long_time_dim = f"{time_dim}_contrib"
    long_form_data = long_form_data.rename({time_dim: long_time_dim})

    # Convert grouping arrays to DataArrays with the correct dimension
    target_year_da = xr.DataArray(target_year_groups, dims=[long_time_dim], name="target_year")
    dayofyear_da = xr.DataArray(dayofyear_groups, dims=[long_time_dim], name="dayofyear")

    # Use flox with both grouping variables to compute climatologies
    climatologies = flox.xarray.xarray_reduce(
        long_form_data,
        target_year_da,
        dayofyear_da,
        dim=long_time_dim,
        func="nanmean",
        expected_groups=(unique_years, np.arange(1, 367)),
        isbin=(False, False),
        dtype=np.float32,
        fill_value=np.nan,
    )

    # Create index arrays for final mapping
    year_to_idx = pd.Series(range(len(unique_years)), index=unique_years)
    year_indices = year_to_idx[year_vals].values

    # Select appropriate climatology for each time point
    result = climatologies.isel(
        target_year=xr.DataArray(year_indices, dims=[time_dim]),
        dayofyear=xr.DataArray(doy_vals - 1, dims=[time_dim]),
    )

    # Clean up dimensions and coordinates
    result = result.drop_vars(["target_year", "dayofyear"])

    return result.chunk(original_chunk_dict)


def smoothed_rolling_climatology(
    da: xr.DataArray,
    window_year_baseline: int = 15,
    smooth_days_baseline: int = 21,
    dimensions: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
    coordinates: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
) -> xr.DataArray:
    """
    Compute a smoothed rolling climatology using the previous `window_year_baseline` years of data and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    window_year_baseline : int, default=15
        Number of years to include in each climatology window
    smooth_days_baseline : int, default=21
        Number of days for temporal smoothing window
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data

    Returns
    -------
    xarray.DataArray
        Smoothed rolling climatology with same shape as input data

    Examples
    --------
    Basic smoothed rolling climatology:

    >>> import xarray as xr
    >>> import marEx
    >>>
    >>> # Load SST data
    >>> sst = xr.open_dataset('sst_data.nc', chunks={}).sst.chunk({'time': 30})
    >>>
    >>> # Compute smoothed rolling climatology
    >>> smooth_clim = marEx.smoothed_rolling_climatology(
    ...     sst,
    ...     window_year_baseline=15,
    ...     smooth_days_baseline=21
    ... )
    >>> print(smooth_clim.shape)
    (7305, 180, 360)

    Comparing different smoothing windows:

    >>> # Short smoothing - more day-to-day variability
    >>> clim_short = marEx.smoothed_rolling_climatology(
    ...     sst, smooth_days_baseline=7
    ... )
    >>>
    >>> # Long smoothing - smoother seasonal cycle
    >>> clim_long = marEx.smoothed_rolling_climatology(
    ...     sst, smooth_days_baseline=61
    ... )
    >>>
    >>> # Compare variability
    >>> var_short = clim_short.std(dim='time').mean().compute()
    >>> var_long = clim_long.std(dim='time').mean().compute()
    >>> print(f"Variability: short={var_short:.3f}, long={var_long:.3f}")

    Climatology for anomaly computation:

    >>> # Compute smoothed climatology then anomalies
    >>> climatology = marEx.smoothed_rolling_climatology(sst)
    >>> anomalies = sst - climatology
    >>>
    >>> # Check that anomalies have reasonable properties
    >>> print(f"Anomaly mean: {anomalies.mean().compute():.6f}")
    >>> print(f"Anomaly std: {anomalies.std().compute():.3f}")

    Unstructured data processing:

    >>> # ICON ocean data
    >>> icon_sst = xr.open_dataset('icon_sst.nc', chunks={}).to.chunk({'time': 25})
    >>> icon_smooth_clim = marEx.smoothed_rolling_climatology(
    ...     icon_sst,
    ...     dimensions={"time": "time", "x": "ncells"},
    ...     coordinates={"time": "time", "x": "lon", "y": "lat"},
    ...     window_year_baseline=10,
    ...     smooth_days_baseline=31
    ... )

    Effect of smoothing on seasonal cycle:

    >>> # Raw rolling climatology (no temporal smoothing)
    >>> raw_clim = marEx.rolling_climatology(sst, window_year_baseline=15)
    >>>
    >>> # Smoothed rolling climatology
    >>> smooth_clim = marEx.smoothed_rolling_climatology(
    ...     sst, window_year_baseline=15, smooth_days_baseline=21
    ... )
    >>>
    >>> # Compare seasonal cycle smoothness
    >>> # Extract annual cycle for a point
    >>> point_raw = raw_clim.isel(lat=90, lon=180).sel(time='2010')
    >>> point_smooth = smooth_clim.isel(lat=90, lon=180).sel(time='2010')
    >>>
    >>> print(f"Raw climatology range: {(point_raw.max() - point_raw.min()).compute():.3f}")
    >>> print(f"Smooth climatology range: {(point_smooth.max() - point_smooth.min()).compute():.3f}")

    Performance considerations:

    >>> # Efficient implementation smooths raw data first, then computes climatology
    >>> # This is more memory-efficient than smoothing the climatology
    >>> large_sst = sst.chunk({'time': 25, 'lat': 45, 'lon': 90})
    >>> efficient_clim = marEx.smoothed_rolling_climatology(large_sst)
    """

    # N.B.: It is more efficient (chunking-wise) to smooth the raw data rather than the climatology
    da_smoothed = (
        da.rolling({dimensions["time"]: smooth_days_baseline}, center=True)
        .mean()
        .chunk({dim: chunks for dim, chunks in zip(da.dims, da.chunks)})
    )

    clim = rolling_climatology(da_smoothed, window_year_baseline, dimensions, coordinates)

    return clim


def _compute_anomaly_shifting_baseline(
    da: xr.DataArray,
    window_year_baseline: int = 15,
    smooth_days_baseline: int = 21,
    dimensions: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
    coordinates: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
) -> xr.Dataset:
    """
    Compute anomalies using shifting baseline method with smoothed rolling climatology.

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # Compute smoothed rolling climatology
    climatology_smoothed = smoothed_rolling_climatology(da, window_year_baseline, smooth_days_baseline, dimensions, coordinates)

    # Compute anomaly as difference from climatology
    anomalies = da - climatology_smoothed

    # Create ocean/land mask from first time step
    chunk_dict_mask = {dimensions[dim]: -1 for dim in ["x", "y"] if dim in dimensions}
    mask = np.isfinite(da.isel({dimensions["time"]: 0})).drop_vars({coordinates["time"]}).chunk(chunk_dict_mask)

    # Build output dataset
    return xr.Dataset({"dat_anomaly": anomalies, "mask": mask})


# ==========================
# Hobday Extreme Definition
# ==========================


def _identify_extremes_hobday(
    da: xr.DataArray,
    threshold_percentile: float = 95,
    window_days_hobday: int = 11,
    exact_percentile: bool = False,
    dimensions: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
    coordinates: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events using day-of-year (i.e. climatological percentile threshold).

    For each spatial point and day-of-year, computes the p-th percentile of values within a window_days_hobday day window across all years.
    This implements the standard methodology for marine heatwave detection threshold calculation.

    Parameters:
    -----------
    da : xarray.DataArray
        Anomaly data with dimensions (time, lat, lon)
        Must be chunked with time dimension unbounded (time: -1)
    threshold_percentile : float, default 95
        Percentile to compute (0-100)
    window_days_hobday : int, default 11
        Window in days
    exact_percentile : bool, optional
        Whether to compute exact percentiles

    Returns:
    --------
    tuple
        (extreme_bool, thresholds)
        extreme_bool : xarray.DataArray
            Boolean mask indicating extreme events (True for extreme days)
        thresholds : xarray.DataArray
            Threshold values with dimensions (dayofyear, lat, lon)
    """
    # Add day-of-year coordinate
    da = da.assign_coords(dayofyear=da[dimensions["time"]].dt.dayofyear)

    # Group by day-of-year and compute percentile
    if exact_percentile:
        # Construct rolling window dimension
        da_windowed = da.rolling({dimensions["time"]: window_days_hobday}, center=True).construct("window")

        thresholds = da_windowed.groupby("dayofyear").reduce(
            np.nanpercentile, q=threshold_percentile, dim=("window", dimensions["time"])
        )
    else:  # Optimised histogram approximation method
        thresholds = compute_histogram_quantile_2d(
            da,
            threshold_percentile / 100.0,
            window_days_hobday=window_days_hobday,
            dimensions=dimensions,
            coordinates=coordinates,
        )

    # Ensure spatial dimensions are fully loaded for efficient comparison
    spatial_chunks = {dimensions[dim]: -1 for dim in ["x", "y"] if dim in dimensions}
    thresholds = thresholds.chunk(spatial_chunks)

    # Compare anomalies to day-of-year specific thresholds
    extremes = da.groupby(da[coordinates["time"]].dt.dayofyear) >= thresholds
    extremes = extremes.astype(bool).chunk(spatial_chunks)

    return extremes, thresholds


# ===============================================
# Detrended Baseline Anomaly Method (Old Method)
# ===============================================


def add_decimal_year(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """
    Add decimal year coordinate to DataArray for trend analysis.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with datetime coordinate
    dim : str, optional
        Name of the time dimension

    Returns
    -------
    xarray.DataArray
        Input data with added 'decimal_year' coordinate
    """
    time = pd.to_datetime(da[dim])
    start_of_year = pd.to_datetime(time.year.astype(str) + "-01-01")
    start_of_next_year = pd.to_datetime((time.year + 1).astype(str) + "-01-01")
    year_elapsed = (time - start_of_year).days
    year_duration = (start_of_next_year - start_of_year).days

    decimal_year = time.year + year_elapsed / year_duration
    return da.assign_coords(decimal_year=(dim, decimal_year))


def _compute_anomaly_detrended(
    da: xr.DataArray,
    std_normalise: bool = False,
    detrend_orders: List[int] = [1],
    dimensions: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
    coordinates: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
    force_zero_mean: bool = True,
) -> xr.Dataset:
    """
    Generate normalised anomalies by removing trends, seasonal cycles, and optionally
    standardising by local temporal variability using the detrended baseline method.
    """
    da = da.astype(np.float32)

    # Ensure time is the first dimension for efficient processing
    if da.dims[0] != dimensions["time"]:
        da = da.transpose(dimensions["time"], ...)

    # Warn if using higher-order detrending without linear component
    if 1 not in detrend_orders and len(detrend_orders) > 1:
        print("Warning: Higher-order detrending without linear term may be unstable")

    # Add decimal year for trend modelling
    da = add_decimal_year(da, dim=dimensions["time"])
    dy = da.decimal_year.compute()

    # Build model matrix with constant term, trends, and seasonal harmonics
    model_components = [np.ones(len(dy))]  # Constant term

    # Add polynomial trend terms
    centered_time = da.decimal_year - np.mean(dy)
    for order in detrend_orders:
        model_components.append(centered_time**order)

    # Add annual and semi-annual cycles (harmonics)
    model_components.extend(
        [
            np.sin(2 * np.pi * dy),  # Annual sine
            np.cos(2 * np.pi * dy),  # Annual cosine
            np.sin(4 * np.pi * dy),  # Semi-annual sine
            np.cos(4 * np.pi * dy),  # Semi-annual cosine
        ]
    )

    # Convert to numpy array for matrix operations
    model = np.array(model_components)

    # Orthogonalise model components for numerical stability
    for i in range(1, model.shape[0]):
        model[i] = model[i] - np.mean(model[i]) * model[0]

    # Compute pseudo-inverse for model fitting
    pmodel = np.linalg.pinv(model)
    n_coeffs = len(model_components)

    # Convert model matrices to xarray
    model_da = xr.DataArray(
        model.T,
        dims=[dimensions["time"], "coeff"],
        coords={
            dimensions["time"]: da[coordinates["time"]].values,
            "coeff": np.arange(1, n_coeffs + 1),
        },
    ).chunk({dimensions["time"]: da.chunks[0]})

    pmodel_da = xr.DataArray(
        pmodel.T,
        dims=["coeff", dimensions["time"]],
        coords={
            "coeff": np.arange(1, n_coeffs + 1),
            dimensions["time"]: da[coordinates["time"]].values,
        },
    ).chunk({dimensions["time"]: da.chunks[0]})

    # Prepare dimensions for model coefficients based on data structure
    dims = ["coeff"]
    coords = {"coeff": np.arange(1, n_coeffs + 1)}

    # Handle both 2D (unstructured) and 3D (gridded) data
    if "y" in dimensions:  # 3D gridded case
        dims.extend([dimensions["y"], dimensions["x"]])
        coords[dimensions["y"]] = da[coordinates["y"]].values
        coords[dimensions["x"]] = da[coordinates["x"]].values
    else:  # 2D unstructured case
        dims.append(dimensions["x"])
        coords.update(da[coordinates["x"]].coords)

    # Fit model to data
    model_fit_da = xr.DataArray(pmodel_da.dot(da), dims=dims, coords=coords)

    # Remove trend and seasonal cycle
    da_detrend = da.drop_vars({"decimal_year"}) - model_da.dot(model_fit_da).astype(np.float32)

    # Force zero mean if requested
    if force_zero_mean:
        da_detrend = da_detrend - da_detrend.mean(dim=dimensions["time"])

    # Create ocean/land mask from first time step
    chunk_dict_mask = {dimensions[dim]: -1 for dim in ["x", "y"] if dim in dimensions}
    mask_temp = np.isfinite(da.isel({dimensions["time"]: 0})).chunk(chunk_dict_mask)
    # Drop time-related coordinates to create spatial mask
    vars_to_drop = []
    if "decimal_year" in mask_temp.coords:
        vars_to_drop.append("decimal_year")
    if dimensions["time"] in mask_temp.coords:
        vars_to_drop.append(dimensions["time"])
    if coordinates["time"] in mask_temp.coords:
        vars_to_drop.append(coordinates["time"])
    mask = mask_temp.drop_vars(vars_to_drop) if vars_to_drop else mask_temp

    # Initialise output dataset
    data_vars = {"dat_anomaly": da_detrend, "mask": mask}
    
    # Ensure all original coordinates are preserved in the dataset
    coords_to_preserve = {}
    for coord_name in da.coords:
        if coord_name not in data_vars:  # Don't override data variables
            coords_to_preserve[coord_name] = da.coords[coord_name]

    # Standardise anomalies by temporal variability if requested
    if std_normalise:

        # Calculate day-of-year standard deviation using cohorts
        std_day = flox.xarray.xarray_reduce(
            da_detrend,
            da_detrend[coordinates["time"]].dt.dayofyear,
            dim=dimensions["time"],
            func="std",
            isbin=False,
            method="cohorts",
        )

        # Calculate 30-day rolling standard deviation with annual wrapped padding
        std_day_wrap = std_day.pad(dayofyear=16, mode="wrap")
        std_rolling = np.sqrt((std_day_wrap**2).rolling(dayofyear=30, center=True).mean()).isel(dayofyear=slice(16, 366 + 16))

        # Divide anomalies by rolling standard deviation
        # Replace any zeros or extremely small values with NaN to avoid division warnings
        std_rolling_safe = std_rolling.where(std_rolling > 1e-10, np.nan)
        da_stn = da_detrend.groupby(da_detrend[coordinates["time"]].dt.dayofyear) / std_rolling_safe

        # Rechunk data for efficient processing
        chunk_dict_std = chunk_dict_mask.copy()
        chunk_dict_std["dayofyear"] = -1

        da_stn = da_stn.chunk(chunk_dict_mask)
        std_rolling = std_rolling.chunk(chunk_dict_std)

        # Add standardised data to output
        data_vars["dat_stn"] = da_stn
        data_vars["STD"] = std_rolling

    # Build output dataset with metadata
    return xr.Dataset(data_vars=data_vars, coords=coords_to_preserve)


def _rolling_histogram_quantile(
    hist_chunk: NDArray[np.float64],
    window_days_hobday: int,
    q: float,
    bin_centers: NDArray[np.float64],
) -> NDArray[np.float32]:
    """
    Efficiently compute quantile thresholds from histogram data using vectorised numpy operations.

    Parameters
    ----------
    hist_chunk : numpy.ndarray
        Histogram data with shape (dayofyear, da_bin)
    window_days_hobday : int
        Rolling window size for day-of-year smoothing
    q : float
        Quantile to compute (0-1)
    bin_centers : numpy.ndarray
        Bin centre values for interpolation

    Returns
    -------
    numpy.ndarray
        Quantile thresholds with shape (dayofyear,)
    """
    n_doy, n_bins = hist_chunk.shape

    # Pad histogram with wrap mode for day-of-year cycling
    pad_size = window_days_hobday // 2
    hist_pad = np.concatenate([hist_chunk[-pad_size:], hist_chunk, hist_chunk[:pad_size]], axis=0)

    # Apply rolling sum using stride tricks (FTW)
    windowed_view = sliding_window_view(hist_pad, window_days_hobday, axis=0)
    hist_windowed = np.sum(windowed_view, axis=-1)

    # Compute PDF and CDF in single pass
    hist_sum = np.sum(hist_windowed, axis=1, keepdims=True) + 1e-10
    pdf = hist_windowed / hist_sum
    cdf = np.cumsum(pdf, axis=1)

    # Find first bin exceeding quantile threshold
    mask = cdf >= q
    first_true = np.argmax(mask, axis=1)
    idx_prev = np.clip(first_true - 1, 0, n_bins - 1)

    # Extract CDF values for linear interpolation
    doy_indices = np.arange(n_doy)
    cdf_prev = cdf[doy_indices, idx_prev]
    cdf_next = cdf[doy_indices, first_true]

    bin_prev = bin_centers[idx_prev]
    bin_next = bin_centers[first_true]

    denom = cdf_next - cdf_prev
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    frac = (q - cdf_prev) / denom
    threshold = bin_prev + frac * (bin_next - bin_prev)

    return threshold.astype(np.float32)


def compute_histogram_quantile_2d(
    da: xr.DataArray,
    q: float,
    window_days_hobday: int = 11,
    bin_edges: Optional[NDArray[np.float64]] = None,
    dimensions: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
    coordinates: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
) -> xr.DataArray:
    """
    Efficiently compute quantiles using binned histograms optimised for extreme values.
    Uses fine-grained bins for positive anomalies and a single bin for negative values.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    q : float
        Quantile to compute (0-1)

    Returns
    -------
    xarray.DataArray
        Computed quantile value for each spatial location
    """
    if bin_edges is None:
        # Default asymmetric bins
        precision = 0.01
        max_anomaly = 5.0
        bin_edges = np.concatenate([[-np.inf, 0.0], np.arange(precision, max_anomaly + precision, precision)])

    bin_centers_array = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_centers_array[0] = 0.0

    bin_centers = xr.DataArray(
        bin_centers_array,
        dims=["da_bin"],
        coords={"da_bin": np.arange(len(bin_centers_array))},
        name="bin_centers",
    )

    chunk_dict = {dimensions["time"]: -1}
    for d in ["x", "y"]:
        if d in dimensions:
            chunk_dict[dimensions[d]] = 10

    da_bin = xr.DataArray(
        np.digitize(da.data, bin_edges) - 1,  # -1 so first bin is 0
        dims=da.dims,
        coords=da.coords,
        name="da_bin",
    ).chunk(chunk_dict)

    # Construct 2D histogram using flox (in doy & anomaly)
    hist_raw = flox.xarray.xarray_reduce(
        da_bin,
        da_bin.dayofyear,
        da_bin,
        dim=[dimensions["time"]],
        func="count",
        expected_groups=(np.arange(1, 367), np.arange(len(bin_edges) - 1)),
        isbin=(False, False),
        dtype=np.int32,
        fill_value=0,
    )
    hist_raw.name = None

    def _compute_quantile_with_params(hist_chunk, bin_centers_chunk):
        return _rolling_histogram_quantile(hist_chunk, window_days_hobday, q, bin_centers_chunk)

    # Apply the optimised computation using apply_ufunc
    threshold = xr.apply_ufunc(
        _compute_quantile_with_params,
        hist_raw,
        bin_centers,
        input_core_dims=[["dayofyear", "da_bin"], ["da_bin"]],
        output_core_dims=[["dayofyear"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"dayofyear": 366}},
        keep_attrs=True,
    )

    return threshold


def compute_histogram_quantile_1d(
    da: xr.DataArray,
    q: float,
    dim: str = "time",
    bin_edges: Optional[NDArray[np.float64]] = None,
) -> xr.DataArray:
    """
    Efficiently compute quantiles using binned histograms optimised for extreme values.
    Uses fine-grained bins for positive anomalies and a single bin for negative values.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    q : float
        Quantile to compute (0-1)
    dim : str, optional
        Dimension along which to compute quantile

    Returns
    -------
    xarray.DataArray
        Computed quantile value for each spatial location
    """
    if bin_edges is None:
        # Default asymmetric bins
        precision = 0.01
        max_anomaly = 5.0
        bin_edges = np.concatenate([[-np.inf, 0.0], np.arange(precision, max_anomaly + precision, precision)])

    # Compute histogram
    hist = histogram(da, bins=[bin_edges], dim=[dim])

    # Convert to PDF and CDF
    hist_sum = hist.sum(dim=f"{da.name}_bin") + 1e-10
    pdf = hist / hist_sum
    cdf = pdf.cumsum(dim=f"{da.name}_bin")

    # Get bin centers
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_centers[0] = 0.0  # Set negative bin centre to 0

    # Find first bin exceeding quantile
    mask = cdf >= q
    first_true = mask.argmax(dim=f"{da.name}_bin")

    # Linearly interpolate between the two points around the 0 crossing
    idx = first_true.compute()
    idx_prev = np.clip(idx - 1, 0, len(bin_centers) - 1)

    cdf_prev = cdf.isel({f"{da.name}_bin": xr.DataArray(idx_prev, dims=first_true.dims)}).data
    cdf_next = cdf.isel({f"{da.name}_bin": xr.DataArray(idx, dims=first_true.dims)}).data
    bin_prev = bin_centers[idx_prev]
    bin_next = bin_centers[idx]

    denom = cdf_next - cdf_prev
    frac = (q - cdf_prev) / denom
    result_data = bin_prev + frac * (bin_next - bin_prev)

    result = first_true.copy(data=result_data)

    return result


# ======================================
# Constant (in time) Extreme Definition
# ======================================


def _identify_extremes_constant(
    da: xr.DataArray,
    threshold_percentile: float = 95,
    exact_percentile: bool = False,
    dimensions: Dict[str, str] = {"time": "time", "x": "lon"},
    coordinates: Dict[str, str] = {"time": "time", "x": "lon", "y": "lat"},
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events exceeding a constant (in time) percentile threshold.
    i.e. There is 1 threshold for each spatial point, computed across all time.

    Returns both the extreme events boolean mask and the thresholds used.
    """
    if exact_percentile:  # Compute exact percentile (memory-intensive)
        # Determine appropriate chunk size based on data dimensions
        if "y" in dimensions:
            rechunk_size = "auto"
        else:
            rechunk_size = 100 * int(np.sqrt(da[dimensions["x"]].size) * 1.5 / 100)
        # N.B.: If this rechunk_size is too small, then dask will be overwhelmed by the number of tasks
        chunk_dict = {dimensions[dim]: rechunk_size for dim in ["x", "y"] if dim in dimensions}
        chunk_dict[dimensions["time"]] = -1
        da_rechunk = da.chunk(chunk_dict)

        # Calculate threshold
        threshold = da_rechunk.quantile(threshold_percentile / 100.0, dim=dimensions["time"])

    else:  # Use an efficient histogram-based method with specified accuracy
        threshold = compute_histogram_quantile_1d(da, threshold_percentile / 100.0, dim=dimensions["time"])

    # Clean up coordinates if needed
    if "quantile" in threshold.coords:
        threshold = threshold.drop_vars("quantile")

    # Ensure spatial dimensions are fully loaded for efficient comparison
    spatial_chunks = {dimensions[dim]: -1 for dim in ["x", "y"] if dim in dimensions}
    threshold = threshold.chunk(spatial_chunks)

    # Create boolean mask for values exceeding threshold
    extremes = da >= threshold

    # Clean up coordinates if needed
    if "quantile" in extremes.coords:
        extremes = extremes.drop_vars("quantile")

    extremes = extremes.astype(bool).chunk({dim: chunks for dim, chunks in zip(da.dims, da.chunks)})

    return extremes, threshold
