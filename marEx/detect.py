"""
MarEx-Detect: Marine Extremes Detection Module

Preprocessing toolkit for marine extremes identification from scalar oceanographic data.
Converts raw time series into standardised anomalies and identifies extreme events
(e.g., Marine Heatwaves using Sea Surface Temperature).

Core capabilities:

* Two preprocessing methodologies: Detrended Baseline and Shifting Baseline
* Two definitions for extreme events: Global Extreme and Hobday Extreme
* Threshold-based extreme event identification
* Efficient processing of both structured (gridded) and unstructured data

Compatible data formats:

* Structured data: 3D arrays (time, lat, lon)
* Unstructured data: 2D arrays (time, cell)
"""

import logging
import warnings
from typing import Dict, List, Literal, Optional, Tuple

import dask
import flox.xarray
import numpy as np
import pandas as pd
import xarray as xr
from dask import persist
from dask.base import is_dask_collection
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from xhistogram.xarray import histogram

# Coordinate validation imports removed
from .exceptions import ConfigurationError, create_data_validation_error
from .helper import checkpoint_to_zarr, fix_dask_tuple_array
from .logging_config import configure_logging, get_logger, log_dask_info, log_memory_usage, log_timing

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


def _infer_dims_coords(
    da: xr.DataArray, dimensions: Optional[Dict[str, str]], coordinates: Optional[Dict[str, str]]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Determine full set of dimensions and coordinates for the DataArray.
    Sets default (standard) dimension and coordinate names if unspecified.

    This function ensures the dimensions dictionary includes required keys and coordinates
    are properly set based on data structure. It validates that all specified dimensions
    and coordinates exist in the dataset.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array to infer dimensions and coordinates for
    dimensions : dict
        Mapping of conceptual dimensions to actual dimension names
    coordinates : dict, optional
        Mapping of conceptual coordinates to actual coordinate names

    Returns
    -------
    tuple
        Tuple of (dimensions, coordinates) dictionaries with defaults applied

    Raises
    ------
    DataValidationError
        If any specified dimension or coordinate does not exist in the dataset
    """
    if dimensions is None:
        dimensions = {"time": "time", "x": "lon", "y": "lat"}

    if "time" not in dimensions:
        dimensions = {"time": "time", **dimensions}  # Permit partial default dimensions --> "time"

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
                    "If data is gridded, ensure 'y' dimension is also specified",
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
    else:
        # Coordinates provided but ensure time coordinate is included if missing
        if "time" not in coordinates:
            coordinates = {"time": dimensions.get("time", "time"), **coordinates}
            logger.debug("Added default time coordinate to provided coordinates")

    # Validate dimensions and coordinates exist in dataset
    logger.debug("Validating dimensions and coordinates")
    _validate_dimensions_exist(da, dimensions)
    _validate_coordinates_exist(da, coordinates)

    return dimensions, coordinates


def _validate_data_values(da: xr.DataArray, dimensions: Dict[str, str]) -> None:
    """
    Validate that all unmasked data contains only finite values (no NaN or inf).

    Parameters
    ----------
    da : xarray.DataArray
        Input data array to validate
    dimensions : dict
        Mapping of conceptual dimensions to actual dimension names

    Raises
    ------
    DataValidationError
        If any unmasked data contains NaN or infinite values
    """
    # Create spatial mask from first time step (2D array)
    spatial_mask = np.isfinite(da.isel({dimensions["time"]: 0}))

    # Check if there's any valid data at all
    if not spatial_mask.any().compute():
        raise create_data_validation_error(
            "Dataset contains no valid (finite) data",
            details="All values in the first time step are NaN or infinite",
            suggestions=[
                "Check your input data for data quality issues",
                "Verify the data was loaded correctly",
                "Check for issues in data preprocessing steps",
            ],
            data_info={
                "total_values": int(da.size),
                "total_spatial_locations": int(np.prod([da.sizes[d] for d in da.dims if d != dimensions["time"]])),
            },
        )

    # Reduce first, then mask (avoids broadcasting across time)
    # Count invalid values at each spatial location across time dimension
    # This produces a 2D spatial array instead of a 3D array
    finite_mask = np.isfinite(da)
    invalid_per_location = (~finite_mask).sum(dim=dimensions["time"])

    # Now apply spatial mask to this 2D result (no broadcasting across time!)
    invalid_in_valid_locations = invalid_per_location.where(spatial_mask, 0)

    # Check if any valid ocean location has invalid data
    max_invalid = invalid_in_valid_locations.max().compute()

    if max_invalid > 0:
        total_invalid_in_ocean = int(invalid_in_valid_locations.sum().compute())
        total_ocean_locations = int(spatial_mask.sum().compute())
        locations_affected = int((invalid_in_valid_locations > 0).sum().compute())
        total_time_steps = int(da.sizes[dimensions["time"]])

        raise create_data_validation_error(
            f"Dataset contains {total_invalid_in_ocean} invalid values in {locations_affected} ocean locations",
            details=(
                f"Found invalid data across time series. Worst location has {int(max_invalid)} "
                f"invalid time steps out of {total_time_steps}."
            ),
            suggestions=[
                "Remove or interpolate NaN/infinite values before preprocessing",
                "Check data quality and loading procedures",
                "Consider using data.fillna() or data.interpolate_na() methods",
                "Verify coordinate/dimension alignment in your dataset",
                "For ocean data, ensure land mask is properly applied before preprocessing",
            ],
            data_info={
                "total_invalid_values_in_ocean": total_invalid_in_ocean,
                "locations_affected": locations_affected,
                "total_ocean_locations": total_ocean_locations,
                "max_invalid_at_one_location": int(max_invalid),
                "total_time_steps": total_time_steps,
                "percentage_affected": f"{100.0 * locations_affected / total_ocean_locations:.2f}%",
            },
        )


# ============================
# Methodology Selection
# ============================


def preprocess_data(
    da: xr.DataArray,
    method_anomaly: Literal[
        "detrend_harmonic", "shifting_baseline", "fixed_baseline", "detrend_fixed_baseline"
    ] = "shifting_baseline",
    method_extreme: Literal["global_extreme", "hobday_extreme"] = "hobday_extreme",
    threshold_percentile: float = 95,
    window_year_baseline: int = 15,  # for shifting_baseline
    smooth_days_baseline: int = 21,  # "
    window_days_hobday: int = 11,  # for hobday_extreme
    window_spatial_hobday: Optional[int] = None,  # "
    std_normalise: bool = False,  # for detrend_harmonic
    detrend_orders: Optional[List[int]] = None,  # "
    force_zero_mean: bool = True,  # "
    method_percentile: Literal["exact", "approximate"] = "approximate",
    precision: float = 0.01,
    max_anomaly: float = 5.0,
    dask_chunks: Optional[Dict[str, int]] = None,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    neighbours: Optional[xr.DataArray] = None,
    cell_areas: Optional[xr.DataArray] = None,
    use_temp_checkpoints: bool = False,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> xr.Dataset:
    """
    Complete preprocessing pipeline for marine extreme event identification.

    Supports separate methods for anomaly computation and extreme identification:

    Anomaly Methods:

    * 'detrend_harmonic': Detrending with harmonics and polynomials -- more efficient, but biases statistics
    * 'shifting_baseline': Rolling climatology using previous window_year_baseline years -- more "correct",
      but shortens time series by window_year_baseline years
    * 'fixed_baseline': Daily climatology using full time series -- does _not_ remove climate trends !
    * 'detrend_fixed_baseline': Polynomial detrending followed by fixed daily climatology -- keeps full time-series
      of data, but does not account for trends in the timing of seasonal transitions (which may appear as extremes)

    Extreme Methods:

    * 'global_extreme': Global-in-time threshold value
    * 'hobday_extreme': Local day-of-year specific thresholds with windowing

    Parameters
    ----------
    da : xarray.DataArray
        Raw input data
    method_anomaly : str, default='shifting_baseline'
        Anomaly computation method ('detrend_harmonic', 'shifting_baseline', 'fixed_baseline', or 'detrend_fixed_baseline').
    method_extreme : str, default='hobday_extreme'
        Extreme identification method ('global_extreme' or 'hobday_extreme').
    threshold_percentile : float, default=95
        Percentile threshold for extreme event detection.
    window_year_baseline : int, default=15
        Number of previous years for rolling climatology (shifting_baseline method only).
    smooth_days_baseline : int, default=21
        Days for smoothing rolling climatology (shifting_baseline method only).
    window_days_hobday : int, default=11
        Window size for day-of-year threshold calculation (hobday_extreme method only).
    window_spatial_hobday : int, default=None
        Spatial window size (2D centred window) for the day-of-year threshold calculation (hobday_extreme method only).
    std_normalise : bool, default=False
        Whether to standardise anomalies by rolling standard deviation (detrend_harmonic only).
    detrend_orders : list, default=[1]
        Polynomial orders for detrending (detrend_harmonic method only).
        Default is 1st order (linear) detrend. `[1,2]` e.g. would use a linear+quadratic detrending.
    force_zero_mean : bool, default=True
        Whether to enforce zero mean in detrended anomalies (detrend_harmonic method only).
    method_percentile : str, default='approximate'
        Method for percentile calculation ('exact' or 'approximate') for both global_extreme & hobday_extreme methods.
        N.B.: Using the exact percentile calculation requires both careful/thoughtful chunking & sufficient memory,
        in consideration of the limitations inherent to distributed parallel I/O & processing.
    precision : float, default=0.01
        Precision for histogram bins in approximate percentile method.
    max_anomaly : float, default=5.0
        Maximum anomaly value for histogram binning in the approximate percentile method.
    dask_chunks : dict, optional
        Chunking specification for distributed computation.
    dimensions : dict, default={"time": "time", "x": "lon", "y": "lat"}
        Mapping of dimensions to names in the data.
    coordinates : dict, optional
        Mapping of coordinates to names in the data. Defaults to dimensions mapping.
    neighbours : xarray.DataArray, optional
        Neighbour connectivity for spatial clustering.
    cell_areas : xarray.DataArray, optional
        Cell areas for weighted spatial statistics.
    use_temp_checkpoints : bool, default=False
        Enable checkpointing to temporary zarr stores to break Dask graph dependencies.
        When True, intermediate results (anomalies, thresholds, extremes) are saved to
        temporary zarr files and immediately reloaded, preventing expensive recomputations.
        Recommended for large datasets on HPC systems where the 2D histogram computation
        is expensive. Temporary files are automatically cleaned up after reloading.
    verbose : bool, default=None
        Enable verbose logging with detailed progress information.
        If None, uses current global logging configuration.
    quiet : bool, default=None
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
    >>> # Basic preprocessing with default shifting baseline method
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
    ...     method_anomaly="detrend_harmonic",
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
    Error: Insufficient data for shifting_baseline method. Dataset spans 3 years but window_year_baseline
    requires at least 15 years.

    Performance considerations with chunking:

    >>> # For large datasets, adjust chunking for memory management
    >>> large_sst = sst.chunk({"time": 25, "lat": 90, "lon": 180})
    >>> result = marEx.preprocess_data(
    ...     large_sst,
    ...     dask_chunks={"time": 25},
    ...     method_percentile="approximate"  # Use approximate method (Default) for long time-series calculations
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

    Simple fixed baseline approach:

    >>> # Basic daily climatology across all years
    >>> result_fixed = marEx.preprocess_data(
    ...     sst,
    ...     method_anomaly="fixed_baseline",
    ...     threshold_percentile=95
    ... )
    >>> # Uses all available data for climatology computation

    Combined trend removal and fixed climatology:

    >>> # Remove long-term trends then compute daily climatology
    >>> result_combined = marEx.preprocess_data(
    ...     sst,
    ...     method_anomaly="detrend_fixed_baseline",
    ...     detrend_orders=[1],  # Linear trend
    ...     threshold_percentile=95,
    ...     force_zero_mean=True
    ... )
    >>> # Balances trend removal with simple climatology
    """
    # Set default values for mutable parameters
    if detrend_orders is None:
        detrend_orders = [1]
    if dask_chunks is None:
        dask_chunks = {"time": 25}

    # Configure logging if verbose/quiet parameters are provided
    if verbose is not None or quiet is not None:
        configure_logging(verbose=verbose, quiet=quiet)

    # Log preprocessing start with parameters
    logger.info(f"Starting data preprocessing - Method: {method_anomaly} -> {method_extreme}")
    logger.info(f"Parameters: percentile={threshold_percentile}%, method_percentile={method_percentile}")
    logger.debug(
        f"Anomaly method parameters: window_year={window_year_baseline}, smooth_days={smooth_days_baseline}, "
        + f"std_normalise={std_normalise}, detrend_orders={detrend_orders}, force_zero_mean={force_zero_mean}"
    )
    logger.debug(f"Extreme method parameters: window_days_hobday={window_days_hobday}")

    # Log input data info
    log_dask_info(logger, da, "Input data")
    log_memory_usage(logger, "Initial memory state")

    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

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

    # Validate that all unmasked data is valid (finite values only)
    logger.debug("Validating data values for NaN/infinite values")
    _validate_data_values(da, dimensions)

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
            use_temp_checkpoints,
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
                    "Consider using detrend_fixed_baseline or detrend_harmonic method instead",
                ],
                data_info={
                    "available_years": int(total_years),
                    "required_years": int(window_year_baseline),
                },
            )

        start_year = int(min_year + window_year_baseline)
        logger.info(f"Trimming data to start from {start_year} (removing first {window_year_baseline} years)")
        time_sel = (ds[coordinates["time"]].dt.year >= start_year).compute()
        ds = ds.isel({dimensions["time"]: time_sel})

    # Break graph after expensive anomaly computation
    if use_temp_checkpoints:
        logger.debug("Checkpointing anomaly dataset to break graph dependencies")
        ds = checkpoint_to_zarr(ds, name="anomalies", timedim=dimensions["time"])

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
            f"percentile={threshold_percentile}%, method_percentile={method_percentile}"
        )
        extremes, thresholds = identify_extremes(
            anomalies,
            method_extreme,
            threshold_percentile,
            dimensions,
            coordinates,
            window_days_hobday,
            window_spatial_hobday,
            method_percentile,
            precision,
            max_anomaly,
            use_temp_checkpoints,
        )
        log_memory_usage(logger, "After extreme identification", logging.DEBUG)

    # Add extreme events and thresholds to dataset
    ds_temp = persist(extremes, thresholds)
    extremes, thresholds = ds_temp

    ds["extreme_events"] = extremes
    ds["thresholds"] = thresholds

    # Handle standardised anomalies if requested (only for detrend_harmonic)
    if std_normalise and method_anomaly == "detrend_harmonic":
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
                window_spatial_hobday,
                method_percentile,
                precision,
                max_anomaly,
                use_temp_checkpoints,
            )

            # Break graph after standardised extremes computation
            if use_temp_checkpoints:
                logger.debug("Checkpointing standardised extremes and thresholds to break graph dependencies")
                extremes_stn = checkpoint_to_zarr(extremes_stn, name="extremes_stn", timedim=dimensions["time"])
                thresholds_stn = checkpoint_to_zarr(thresholds_stn, name="thresholds_stn", timedim="dayofyear")

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
                window_spatial_hobday,
            ),
        }
    )

    # Add method-specific parameters
    if method_anomaly == "detrend_harmonic":
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
    elif method_anomaly == "fixed_baseline":
        ds.attrs.update({})  # No method-specific parameters
    elif method_anomaly == "detrend_fixed_baseline":
        ds.attrs.update(
            {
                "detrend_orders": detrend_orders,
                "force_zero_mean": force_zero_mean,
            }
        )

    if method_extreme == "hobday_extreme":
        ds.attrs.update({"window_days_hobday": window_days_hobday})

    ds.attrs.update({"method_percentile": method_percentile, "precision": precision, "max_anomaly": max_anomaly})

    # Final rechunking
    time_chunks = dask_chunks.get(dimensions["time"], dask_chunks.get("time", 10))
    logger.debug(f"Final rechunking with time chunks: {time_chunks}")
    chunk_dict = {dimensions[dim]: -1 for dim in ["x", "y"] if dim in dimensions}
    chunk_dict[dimensions["time"]] = time_chunks
    if method_extreme == "hobday_extreme":
        chunk_dict["dayofyear"] = time_chunks
    ds = ds.chunk(chunk_dict)

    # Clear encoding metadata that may conflict with actual Dask chunks
    # This encoding carries over from checkpointing and can cause chunk misalignment errors
    logger.debug("Clearing encoding metadata for Dask-backed variables")
    for var in ds.data_vars:
        if hasattr(ds[var].data, "chunks"):  # Only for Dask-backed variables
            if hasattr(ds[var], "encoding") and "chunks" in ds[var].encoding:
                del ds[var].encoding["chunks"]

    # Fix encoding issue with saving when calendar & units attribute is present
    if "calendar" in ds[coordinates["time"]].attrs:  # pragma: no cover
        logger.debug("Removing calendar attribute for Zarr compatibility")
        del ds[coordinates["time"]].attrs["calendar"]
    if "units" in ds[coordinates["time"]].attrs:  # pragma: no cover
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
        ds["mask"] = ds.mask.compute()
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
    window_spatial_hobday: Optional[int],
) -> List[str]:
    """Generate preprocessing steps description based on selected methods."""
    steps = []

    if method_anomaly == "detrend_harmonic":
        steps.append(f"Removed polynomial trend orders={detrend_orders} & seasonal cycle")
        if std_normalise:
            steps.append("Normalised by 30-day rolling STD")
    elif method_anomaly == "shifting_baseline":
        steps.append(f"Rolling climatology using {window_year_baseline} years")
        steps.append(f"Smoothed with {smooth_days_baseline}-day window")
    elif method_anomaly == "fixed_baseline":
        steps.append("Daily climatology computed from full time series")
    elif method_anomaly == "detrend_fixed_baseline":
        steps.append(f"Removed polynomial trend orders={detrend_orders}")
        steps.append("Daily climatology computed from detrended data")

    # Extreme method steps
    if method_extreme == "global_extreme":
        steps.append("Global percentile threshold applied to all days")
    elif method_extreme == "hobday_extreme":
        if window_spatial_hobday is not None:
            steps.append(
                f"Day-of-year thresholds with {window_days_hobday} day window & {window_spatial_hobday} spatial neighbours"
            )
        else:
            steps.append(f"Day-of-year thresholds with {window_days_hobday} day window")

    return steps


def compute_normalised_anomaly(
    da: xr.DataArray,
    method_anomaly: Literal[
        "detrend_harmonic", "shifting_baseline", "fixed_baseline", "detrend_fixed_baseline"
    ] = "shifting_baseline",
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    window_year_baseline: int = 15,  # for shifting_baseline
    smooth_days_baseline: int = 21,  # "
    std_normalise: bool = False,  # for detrend_harmonic
    detrend_orders: Optional[List[int]] = None,  # "
    force_zero_mean: bool = True,  # "
    use_temp_checkpoints: bool = False,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
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
    window_year_baseline : int, default=15
        Number of years for rolling climatology (shifting_baseline only)
    smooth_days_baseline : int, default=21
        Days for smoothing rolling climatology (shifting_baseline only)
    std_normalise : bool, default=False
        Whether to normalise by 30-day rolling standard deviation (detrend_harmonic only)
    detrend_orders : list, default=[1]
        Polynomial orders for trend removal (detrend_harmonic and detrend_fixed_baseline only)
    force_zero_mean : bool, default=True
        Explicitly enforce zero mean in final anomalies (detrend_harmonic and detrend_fixed_baseline only)


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
    ...     sst, method_anomaly="detrend_harmonic"
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

    Fixed baseline climatology:

    >>> # Use full time series for daily climatology
    >>> result_fixed = marEx.compute_normalised_anomaly(
    ...     sst,
    ...     method_anomaly="fixed_baseline"
    ... )
    >>> # Climatology computed from all available years

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
    # Set default values for mutable parameters
    if detrend_orders is None:
        detrend_orders = [1]

    # Configure logging if verbose/quiet parameters are provided
    if verbose is not None or quiet is not None:
        configure_logging(verbose=verbose, quiet=quiet)

    logger.debug(f"Computing normalised anomaly using {method_anomaly} method")

    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    if method_anomaly == "detrend_harmonic":
        logger.debug(
            f"Detrended baseline parameters: std_normalise={std_normalise}, orders={detrend_orders}, zero_mean={force_zero_mean}"
        )
        return _compute_anomaly_detrended(da, std_normalise, detrend_orders, dimensions, coordinates, force_zero_mean)
    elif method_anomaly == "shifting_baseline":
        logger.debug(f"Shifting baseline parameters: window_years={window_year_baseline}, smooth_days={smooth_days_baseline}")
        return _compute_anomaly_shifting_baseline(
            da, window_year_baseline, smooth_days_baseline, dimensions, coordinates, use_temp_checkpoints
        )
    elif method_anomaly == "fixed_baseline":
        logger.debug("Fixed baseline parameters: using full time series for daily climatology")
        return _compute_anomaly_fixed_baseline(da, dimensions, coordinates)
    elif method_anomaly == "detrend_fixed_baseline":
        logger.debug(f"Fixed detrended baseline parameters: orders={detrend_orders}, zero_mean={force_zero_mean}")
        return _compute_anomaly_detrend_fixed_baseline(da, detrend_orders, dimensions, coordinates, force_zero_mean)
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
    use_temp_checkpoints: bool = False,
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

    # Set default spatial window (only for hobday_extreme method)
    if method_extreme == "hobday_extreme" and window_spatial_hobday is None and "y" in dimensions and dimensions["y"] in da.dims:
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
            use_temp_checkpoints,
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
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    use_temp_checkpoints: bool = False,
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
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)
    timedim = dimensions["time"]
    time_coord = coordinates["time"]
    original_chunk_dict = dict(zip(da.dims, da.chunks))

    # Add temporal coordinates
    years = da[time_coord].dt.year
    doys = da[time_coord].dt.dayofyear
    da = da.assign_coords({"year": years, "dayofyear": doys})

    # Get temporal bounds
    years, doys = persist(years, doys)
    year_vals = years.values
    doy_vals = doys.values
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
    time_indices = np.array(contributing_time_indices, dtype=np.int32)
    target_year_groups = np.array(contributing_target_years, dtype=np.int32)
    dayofyear_groups = np.array(contributing_dayofyears, dtype=np.int32)

    # Create long-form dataset by selecting the contributing time points
    long_form_data = da.isel({timedim: time_indices})

    # Create a new time dimension for the long-form data
    long_timedim = f"{timedim}_contrib"
    long_form_data = long_form_data.rename({timedim: long_timedim})

    # Convert grouping arrays to DataArrays with the correct dimension
    target_year_da = xr.DataArray(target_year_groups, dims=[long_timedim], name="target_year")
    dayofyear_da = xr.DataArray(dayofyear_groups, dims=[long_timedim], name="dayofyear")

    # Use flox with both grouping variables to compute climatologies
    climatologies = flox.xarray.xarray_reduce(
        long_form_data,
        target_year_da,
        dayofyear_da,
        dim=long_timedim,
        func="nanmean",
        expected_groups=(unique_years, np.arange(1, 367, dtype=np.int32)),
        isbin=(False, False),
        dtype=np.float32,
        fill_value=np.nan,
    ).chunk({"dayofyear": -1})

    if use_temp_checkpoints:
        logger.debug("Checkpointing climatologies to break graph dependencies")
        climatologies = checkpoint_to_zarr(climatologies, name="climatologies", timedim=timedim)

    # Create index arrays for final mapping
    year_to_idx = pd.Series(range(len(unique_years)), index=unique_years)
    year_indices = year_to_idx[year_vals].values

    # Select appropriate climatology for each time point
    result = climatologies.isel(
        target_year=xr.DataArray(year_indices, dims=[timedim]),
        dayofyear=xr.DataArray(doy_vals - 1, dims=[timedim]),
    )

    # Clean up dimensions and coordinates
    result = result.drop_vars(["target_year", "dayofyear"])

    return result.chunk(original_chunk_dict)


def smoothed_rolling_climatology(
    da: xr.DataArray,
    window_year_baseline: int = 15,
    smooth_days_baseline: int = 21,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    use_temp_checkpoints: bool = False,
) -> xr.DataArray:
    """
    Compute a smoothed rolling climatology using the previous `window_year_baseline` years of data
    and reassemble it to match the original data structure.
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
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)
    timedim = dimensions["time"]

    # N.B.: It is more efficient (chunking-wise) to smooth the raw data rather than the climatology
    da_smoothed = (
        da.rolling({timedim: smooth_days_baseline}, center=True).mean().chunk(dict(zip(da.dims, da.chunks))).astype(np.float32)
    )

    clim = rolling_climatology(da_smoothed, window_year_baseline, dimensions, coordinates, use_temp_checkpoints)

    return clim


def _compute_anomaly_shifting_baseline(
    da: xr.DataArray,
    window_year_baseline: int = 15,
    smooth_days_baseline: int = 21,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    use_temp_checkpoints: bool = False,
) -> xr.Dataset:
    """
    Compute anomalies using shifting baseline method with smoothed rolling climatology.

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    # Compute smoothed rolling climatology
    climatology_smoothed = smoothed_rolling_climatology(
        da, window_year_baseline, smooth_days_baseline, dimensions, coordinates, use_temp_checkpoints
    )

    # Compute anomaly as difference from climatology
    anomalies = da - climatology_smoothed

    # Create ocean/land mask from first time step
    mask = np.isfinite(da.isel({dimensions["time"]: 0})).drop_vars({coordinates["time"]})

    # Build output dataset
    return xr.Dataset({"dat_anomaly": anomalies, "mask": mask})


# ==========================
# Hobday Extreme Definition
# ==========================


def _identify_extremes_hobday(
    da: xr.DataArray,
    threshold_percentile: float = 95,
    window_days_hobday: int = 11,
    window_spatial_hobday: Optional[int] = None,
    method_percentile: Literal["exact", "approximate"] = "approximate",
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    precision: float = 0.01,
    max_anomaly: float = 5.0,
    use_temp_checkpoints: bool = False,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events using day-of-year (i.e. climatological percentile threshold).

    For each spatial point and day-of-year, computes the p-th percentile of values within a
    window_days_hobday day window across all years.
    This implements the standard methodology for marine heatwave detection threshold calculation.

    Parameters:
    -----------
    da : xarray.DataArray
        Anomaly data with dimensions (time, lat, lon)
        Must be chunked with time dimension unbounded (time: -1)
    threshold_percentile : float, default=95
        Percentile to compute (0-100)
    window_days_hobday : int, default=11
        Window in days
    window_spatial_hobday : int, default=None
        Window size in cells
    method_percentile : str, default='approximate'
        Method for percentile computation ('exact' or 'approximate')
    precision : float, default=0.01
        Precision for histogram bins in approximate method
    max_anomaly : float, default=5.0
        Maximum anomaly value for histogram binning

    Returns:
    --------
    tuple
        (extreme_bool, thresholds)
        extreme_bool : xarray.DataArray
            Boolean mask indicating extreme events (True for extreme days)
        thresholds : xarray.DataArray
            Threshold values with dimensions (dayofyear, lat, lon)
    """
    # Check if there is sufficient samples
    N_years = np.unique(da[coordinates["time"]].dt.year).size
    N_samples = N_years * window_days_hobday * (window_spatial_hobday if window_spatial_hobday is not None else 1) ** 2
    N_above_threshold = N_samples * (1.0 - threshold_percentile / 100.0)
    if N_above_threshold < 50:
        # Make warning
        logger.warning(
            f"Not enough samples for accurate extreme detection: {N_above_threshold} < 50. "
            "Consider using a lower threshold_percentile, increasing your time-series size, "
            "increasing the window_days_hobday, or using a larger window_spatial_hobday."
            "If your time-series is very short, consider using method_percentile='exact'."
        )

    # Add day-of-year coordinate (compute it to avoid chunked groupby issues)
    da = da.assign_coords(dayofyear=da[coordinates["time"]].dt.dayofyear.compute()).chunk(dict(zip(da.dims, da.chunks))).persist()

    # Group by day-of-year and compute percentile
    if method_percentile == "exact":
        # Construct rolling window dimension
        da_windowed = da.rolling({dimensions["time"]: window_days_hobday}, center=True).construct("window")

        # Ensure dayofyear coordinate is computed for groupby (required by newer xarray)
        if "dayofyear" in da_windowed.coords:
            da_windowed = da_windowed.assign_coords(dayofyear=da_windowed.dayofyear.compute())

        thresholds = da_windowed.groupby("dayofyear").reduce(
            np.nanpercentile, q=threshold_percentile, dim=("window", dimensions["time"])
        )
    else:  # Optimised histogram approximation method
        thresholds = _compute_histogram_quantile_2d(
            da,
            threshold_percentile / 100.0,
            window_days_hobday=window_days_hobday,
            window_spatial_hobday=window_spatial_hobday,
            dimensions=dimensions,
            precision=precision,
            max_anomaly=max_anomaly,
            use_temp_checkpoints=use_temp_checkpoints,
        )

    if use_temp_checkpoints:
        logger.debug("Checkpointing thresholds to break graph dependencies")
        thresholds = checkpoint_to_zarr(thresholds, name="thresholds", timedim="dayofyear")

    # Extract spatial chunk sizes from input data for alignment
    # Use most common chunk size to handle irregular chunks robustly
    spatial_chunks = {}
    for dim_key in ["x", "y"]:
        if dim_key in dimensions:
            dim_name = dimensions[dim_key]
            if dim_name in da.dims:
                dim_index = da.dims.index(dim_name)
                chunks_tuple = da.chunks[dim_index]
                # Get the most common chunk size (handles irregular chunks better)
                spatial_chunks[dim_name] = max(set(chunks_tuple), key=chunks_tuple.count)

    # Drop time coordinate/dimension to avoid conflicts when comparing with data grouped by dayofyear
    coords_to_drop = []
    if coordinates["time"] in thresholds.coords:
        coords_to_drop.append(coordinates["time"])
    if dimensions["time"] in thresholds.coords and dimensions["time"] not in thresholds.dims:
        coords_to_drop.append(dimensions["time"])
    if "time" in thresholds.coords and "time" not in thresholds.dims:
        coords_to_drop.append("time")
    if coords_to_drop:
        thresholds = thresholds.drop_vars(coords_to_drop)

    # Rechunk thresholds BEFORE comparison to align with input data
    # This eliminates expensive implicit rechunking during the groupby operation
    logger.debug(f"Aligning threshold chunks to match input data spatial chunks: {spatial_chunks}")
    thresholds = thresholds.chunk(spatial_chunks)

    # Compare anomalies to day-of-year specific thresholds
    # Assign dayofyear coordinate and use UniqueGrouper for chunked arrays
    da = da.assign_coords(dayofyear=da[coordinates["time"]].dt.dayofyear)
    extremes = da.groupby(dayofyear=xr.groupers.UniqueGrouper(labels=np.arange(1, 367))) >= thresholds

    # Drop unnecessary dayofyear coordinate
    if "dayofyear" in extremes.coords:
        extremes = extremes.drop_vars("dayofyear")

    # Rechunk to fix irregular time chunks created by groupby operation
    # Zarr requires uniform chunks, so we rechunk to match input data's time chunks
    time_dim_index = da.dims.index(dimensions["time"])
    time_chunk_size = max(set(da.chunks[time_dim_index]), key=da.chunks[time_dim_index].count)
    rechunk_dict = {dimensions["time"]: time_chunk_size}
    rechunk_dict.update(spatial_chunks)
    logger.debug(f"Rechunking extremes to fix irregular chunks from groupby: {rechunk_dict}")
    extremes = extremes.chunk(rechunk_dict)

    if use_temp_checkpoints:
        logger.debug("Checkpointing extremes to break graph dependencies")
        extremes = checkpoint_to_zarr(extremes, name="extremes", timedim=dimensions["time"])

    return extremes, thresholds


# ===============================================
# Detrended Baseline Anomaly Method (Old Method)
# ===============================================


def add_decimal_year(da: xr.DataArray, dim: str = "time", coord: Optional[str] = None) -> xr.DataArray:
    """
    Add decimal year coordinate to DataArray for trend analysis.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with datetime coordinate
    dim : str, optional
        Name of the time dimension
    coord : str, optional
        Name of the time coordinate (if different from dimension name)

    Returns
    -------
    xarray.DataArray
        Input data with added 'decimal_year' coordinate
    """
    # Use coordinate name if provided, otherwise use dimension name
    coord_name = coord if coord is not None else dim
    time = pd.to_datetime(da[coord_name])
    start_of_year = pd.to_datetime(time.year.astype(str) + "-01-01")
    start_of_next_year = pd.to_datetime((time.year + 1).astype(str) + "-01-01")
    year_elapsed = (time - start_of_year).days
    year_duration = (start_of_next_year - start_of_year).days

    decimal_year = time.year + year_elapsed / year_duration
    return da.assign_coords(decimal_year=(dim, decimal_year))


def _compute_anomaly_detrended(
    da: xr.DataArray,
    std_normalise: bool = False,
    detrend_orders: Optional[List[int]] = None,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    force_zero_mean: bool = True,
    remove_harmonics: bool = True,
) -> xr.Dataset:
    """
    Generate normalised anomalies by removing trends, seasonal cycles, and optionally
    standardising by local temporal variability using the detrended baseline method.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    std_normalise : bool, default=False
        Whether to standardise anomalies by temporal variability
    detrend_orders : list, optional
        Polynomial orders for trend removal (default: [1] for linear)
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    force_zero_mean : bool, default=True
        Whether to enforce zero mean in detrended data
    remove_harmonics : bool, default=True
        Whether to remove seasonal harmonics (annual and semi-annual cycles)

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies, mask, and optionally standardised data
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    # Default detrend_orders to linear if not specified
    if detrend_orders is None:
        detrend_orders = [1]

    # Validate detrend_orders is not empty and contains valid values
    if not detrend_orders:
        raise ConfigurationError(
            "detrend_orders cannot be empty",
            details="At least one polynomial order must be specified for detrending",
            suggestions=[
                "Use detrend_orders=[1] for linear detrending",
                "Use detrend_orders=[1, 2] for linear + quadratic detrending",
                "Remove detrend_orders optional parameter to use default [1]",
            ],
        )

    # Validate all orders are positive integers
    if any(order < 1 for order in detrend_orders):
        invalid_orders = [order for order in detrend_orders if order < 1]
        raise ConfigurationError(
            f"Invalid polynomial orders: {invalid_orders}",
            details="Polynomial orders must be positive integers (≥ 1)",
            suggestions=[
                "Use only positive integers for polynomial orders",
                "Common values: [1] for linear, [1,2] for linear+quadratic",
                f"Remove invalid orders: {invalid_orders}",
            ],
        )

    da = da.astype(np.float32)

    # Ensure time is the first dimension for efficient processing
    if da.dims[0] != dimensions["time"]:
        da = da.transpose(dimensions["time"], ...)

    # Warn if using higher-order detrending without linear component
    if 1 not in detrend_orders and len(detrend_orders) > 1:
        print("Warning: Higher-order detrending without linear term may be unstable")

    # Add decimal year for trend modelling
    da = add_decimal_year(da, dim=dimensions["time"], coord=coordinates["time"])
    dy = da.decimal_year.compute()

    # Build model matrix with constant term, trends, and seasonal harmonics
    model_components = [np.ones(len(dy))]  # Constant term

    # Add polynomial trend terms
    centered_time = da.decimal_year - np.mean(dy)
    for order in detrend_orders:
        model_components.append(centered_time**order)

    # Add annual and semi-annual cycles (harmonics) if requested
    if remove_harmonics:
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

    # Handle 1D (time series), 2D (unstructured) and 3D (gridded) data
    if "y" in dimensions:  # 3D gridded case
        dims.extend([dimensions["y"], dimensions["x"]])
        coords[dimensions["y"]] = da[coordinates["y"]].values
        coords[dimensions["x"]] = da[coordinates["x"]].values
    elif "x" in dimensions:  # 2D unstructured case
        dims.append(dimensions["x"])
        coords.update(da[coordinates["x"]].coords)
    # else: 1D time series case - no spatial dimensions to add

    # Fit model to data - use the actual dimensions of the result
    dot_result = pmodel_da.dot(da)
    # For dot product result, dimensions match input data's spatial dimensions
    spatial_dims = [dim for dim in da.dims if dim != dimensions["time"]]
    result_dims = ["coeff"] + spatial_dims

    # Build coordinates for the result
    result_coords = {"coeff": np.arange(1, n_coeffs + 1)}
    for dim in spatial_dims:
        if dim in da.coords:
            result_coords[dim] = da.coords[dim]

    model_fit_da = xr.DataArray(dot_result, dims=result_dims, coords=result_coords)

    # Remove trend and seasonal cycle
    da_detrend = (da.drop_vars({"decimal_year"}) - model_da.dot(model_fit_da).astype(np.float32)).persist()

    # Force zero mean if requested
    if force_zero_mean:
        da_detrend = da_detrend - da_detrend.mean(dim=dimensions["time"])

    # Create ocean/land mask from first time step
    # Handle both spatial (3D) and time-series (1D) data
    spatial_dims = [dim for dim in ["x", "y"] if dim in dimensions]
    if spatial_dims:
        # Spatial data - create 2D/3D mask
        chunk_dict_mask = {dimensions[dim]: -1 for dim in spatial_dims}
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
    else:
        # 1D time series - create scalar mask indicating if any finite values exist
        chunk_dict_mask = {}  # Empty for 1D case
        mask = xr.DataArray(np.any(np.isfinite(da.values)), dims=[], attrs={"description": "Time series validity mask"})

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
        da_detrend = da_detrend.assign_coords(dayofyear=da_detrend[coordinates["time"]].dt.dayofyear)
        da_stn = da_detrend.groupby(dayofyear=xr.groupers.UniqueGrouper(labels=np.arange(1, 367))) / std_rolling_safe

        # Drop dayofyear coordinate to avoid merge conflicts
        if "dayofyear" in da_stn.coords:
            da_stn = da_stn.drop_vars("dayofyear")

        # Rechunk data for efficient processing
        chunk_dict_std = chunk_dict_mask.copy()
        chunk_dict_std["dayofyear"] = -1

        da_stn = da_stn.chunk(chunk_dict_mask)
        std_rolling = std_rolling.chunk(chunk_dict_std)

        # Add standardised data to output
        data_vars["dat_stn"] = da_stn
        data_vars["STD"] = std_rolling

    # Build output dataset with metadata
    return xr.Dataset(data_vars=data_vars, coords=coords_to_preserve).drop_vars("decimal_year")


def _compute_anomaly_fixed_baseline(
    da: xr.DataArray,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
) -> xr.Dataset:
    """
    Compute anomalies using fixed baseline method with full time series climatology.

    This method computes a daily climatology using all available years in the dataset,
    then subtracts this climatology from the original data to obtain anomalies.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    # Compute daily climatology across all years using flox for efficiency
    logger.debug("Computing daily climatology across all years")
    daily_climatology = flox.xarray.xarray_reduce(
        da,
        da[coordinates["time"]].dt.dayofyear,
        dim=dimensions["time"],
        func="nanmean",
        isbin=False,
        method="cohorts",
        dtype=np.float32,
    ).persist()

    # Compute anomalies by subtracting daily climatology from original data
    logger.debug("Computing anomalies by subtracting daily climatology")
    da = da.assign_coords(dayofyear=da[coordinates["time"]].dt.dayofyear)
    anomalies = da.groupby(dayofyear=xr.groupers.UniqueGrouper(labels=np.arange(1, 367))) - daily_climatology
    anomalies = anomalies.astype(np.float32)

    # Drop dayofyear coordinate to avoid merge conflicts
    if "dayofyear" in anomalies.coords:
        anomalies = anomalies.drop_vars("dayofyear")

    # Create ocean/land mask from first time step
    # Handle both spatial (3D) and time-series (1D) data
    spatial_dims = [dim for dim in ["x", "y"] if dim in dimensions]
    if spatial_dims:
        # Spatial data - create 2D/3D mask
        chunk_dict_mask = {dimensions[dim]: -1 for dim in spatial_dims}
        mask = np.isfinite(da.isel({dimensions["time"]: 0})).drop_vars({coordinates["time"]}).chunk(chunk_dict_mask)
    else:
        # 1D time series - create scalar mask indicating if any finite values exist
        mask = xr.DataArray(np.any(np.isfinite(da.values)), dims=[], attrs={"description": "Time series validity mask"})

    # Build output dataset
    return xr.Dataset({"dat_anomaly": anomalies, "mask": mask})


def _compute_anomaly_detrend_fixed_baseline(
    da: xr.DataArray,
    detrend_orders: Optional[List[int]] = None,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    force_zero_mean: bool = True,
) -> xr.Dataset:
    """
    Compute anomalies using fixed detrended baseline method.

    This method first removes polynomial trends (without harmonics) from the data,
    then removes a full-time-series daily climatology from the detrended signal.
    Uses _compute_anomaly_detrended internally to perform the trend removal.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    detrend_orders : list, optional
        Polynomial orders for trend removal (default: [1] for linear)
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    force_zero_mean : bool, default=True
        Whether to enforce zero mean in detrended data

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    logger.debug(f"Removing polynomial trends of orders: {detrend_orders}")

    # Step 1: Remove polynomial trends (without harmonics) using _compute_anomaly_detrended
    detrended_result = _compute_anomaly_detrended(
        da=da,
        std_normalise=False,
        detrend_orders=detrend_orders,
        dimensions=dimensions,
        coordinates=coordinates,
        force_zero_mean=force_zero_mean,
        remove_harmonics=False,  # Only remove trends, not harmonics
    )["dat_anomaly"].persist()

    # Step 2: Compute daily climatology and anomalies using _compute_anomaly_fixed_baseline
    logger.debug("Computing daily climatology and anomalies from detrended data")
    final_result = _compute_anomaly_fixed_baseline(
        da=detrended_result,
        dimensions=dimensions,
        coordinates=coordinates,
    )

    return final_result


def _rolling_histogram_quantile(
    hist_chunk: NDArray[np.int32],
    window_days_hobday: int,
    q: float,
    bin_centers: NDArray[np.float64],
) -> NDArray[np.float32]:
    """
    Efficiently compute quantile thresholds from histogram data using vectorised numpy operations.
    Improved robust interpolation handles sparse histograms, especially in the tails.

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
    eps = 1e-10

    # Pad histogram with wrap mode for day-of-year cycling
    pad_size = window_days_hobday // 2
    hist_pad = np.concatenate([hist_chunk[-pad_size:], hist_chunk, hist_chunk[:pad_size]], axis=0)

    # Apply rolling sum using stride tricks FTW
    windowed_view = sliding_window_view(hist_pad, window_days_hobday, axis=0)
    hist_windowed = np.sum(windowed_view, axis=-1)

    # Apply gaussian smoothing along bin dimension
    # sigma = 2
    # hist_smoothed = gaussian_filter1d(
    #     hist_windowed.astype(np.float32), sigma=sigma, axis=1, mode="constant", cval=0.0  # Along bin dimension
    # ).astype(np.float32)

    # Count-based interpolation (rather than interpolating CDF in probability space)
    # Calculate cumulative counts (not normalized CDF)
    cumsum = np.cumsum(hist_windowed, axis=1, dtype=np.int32)
    total_counts = cumsum[:, -1]  # Total count for each day

    # Calculate the exact position where the quantile should be
    # For n samples, the q-th quantile is at position q*(n-1)
    # It is q*n here since we're working with cumulative counts
    quantile_position = q * total_counts

    # Vectorised search for the bins containing the quantile position
    # searchsorted with side='right' gives the first bin where cumsum > quantile_position
    idx_upper = np.zeros(n_doy, dtype=np.int32)

    for i in range(n_doy):
        if total_counts[i] <= 0:  # No data
            idx_upper[i] = 0
        else:
            # Find first bin where cumulative count exceeds target position
            idx_upper[i] = np.searchsorted(cumsum[i], quantile_position[i], side="right")

    # Clip to valid range
    idx_upper = np.clip(idx_upper, 0, n_bins - 1)
    idx_lower = np.maximum(0, idx_upper - 1)

    # Extract values for vectorised interpolation
    doy_indices = np.arange(n_doy, dtype=np.int32)

    # Get cumulative counts at the boundaries
    count_lower = np.where(idx_lower >= 0, cumsum[doy_indices, idx_lower], 0)
    count_upper = cumsum[doy_indices, idx_upper]

    # Bin centers for interpolation
    bin_lower = bin_centers[idx_lower]
    bin_upper = bin_centers[idx_upper]

    # Compute interpolation fraction based on counts
    count_diff = count_upper - count_lower
    safe_diff = np.where(count_diff > eps, count_diff, 1.0)
    frac = np.where(count_diff > eps, (quantile_position - count_lower) / safe_diff, 0.5)  # If no difference, use midpoint

    # Linear interpolation between bin centers
    threshold = bin_lower + frac * (bin_upper - bin_lower)

    # Handle edge cases
    # If total_counts is 0, return NaN
    threshold = np.where(total_counts > 0, threshold, np.nan)

    # If at the first bin (all data is negative), use the first bin center
    threshold = np.where((idx_upper == 0) & (total_counts > 0), bin_centers[0], threshold)

    return threshold.astype(np.float32)


def _compute_histogram_quantile_2d(
    da: xr.DataArray,
    q: float,
    window_days_hobday: int = 11,
    window_spatial_hobday: Optional[int] = None,
    bin_edges: Optional[NDArray[np.float64]] = None,
    dimensions: Optional[Dict[str, str]] = None,
    precision: float = 0.01,
    max_anomaly: float = 5.0,
    use_temp_checkpoints: bool = False,
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
    window_days_hobday : int, default=11
        Rolling window size for day-of-year quantiles
    window_spatial_hobday : int, default=None
        Spatial window size for day-of-year quantiles
    bin_edges : numpy.ndarray, optional
        Custom bin edges for histogram computation
    dimensions : dict, optional
        Dimension mapping dictionary
    precision : float, default=0.01
        Precision for positive anomaly bins
    max_anomaly : float, default=5.0
        Maximum anomaly value for binnin

    Returns
    -------
    xarray.DataArray
        Computed quantile value for each spatial location
    """
    if bin_edges is None:
        # Create optimised asymmetric bins
        bin_edges = np.concatenate(
            [[-np.inf], np.arange(-precision, max_anomaly + precision, precision, dtype=np.float32)], dtype=np.float32
        )

    bin_centers_array = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_centers_array[0] = 0.0

    bin_centers = xr.DataArray(
        bin_centers_array.astype(np.float32),
        dims=["da_bin"],
        coords={"da_bin": np.arange(len(bin_centers_array), dtype=np.uint16)},
        name="bin_centers",
    )

    chunk_dict = {dimensions["time"]: -1}
    chunk_dict[dimensions["x"]] = 16
    if "y" in dimensions:
        chunk_dict[dimensions["y"]] = 16

    da_bin = (
        xr.DataArray(
            np.digitize(da.data, bin_edges) - 1,  # -1 so first bin is 0
            dims=da.dims,
            coords=da.coords,
            name="da_bin",
        )
        .chunk(chunk_dict)
        .astype(np.uint16)
    )

    if use_temp_checkpoints:
        logger.debug("Checkpointing binned data to break graph dependencies")
        da_bin = checkpoint_to_zarr(da_bin, name="da_bin", timedim=dimensions["time"]).chunk(chunk_dict)

    # Construct 2D histogram using flox (in doy & anomaly)
    hist_raw = flox.xarray.xarray_reduce(
        da_bin,
        da_bin.dayofyear,
        da_bin,
        dim=[dimensions["time"]],
        func="count",
        expected_groups=(np.arange(1, 367, dtype=np.uint16), np.arange(len(bin_edges) - 1, dtype=np.uint16)),
        isbin=(False, False),
        dtype=np.uint16,
        fill_value=0,
    )
    hist_raw.name = None

    # Apply spatial-kernel smoothing to the histogram
    if window_spatial_hobday is not None and window_spatial_hobday > 1:
        pad_size = window_spatial_hobday // 2
        lon_dim, lat_dim = dimensions.get("x"), dimensions.get("y")

        hist_rolled = hist_raw

        # Periodic padding in longitude, rolling mean in both dimensions, then trim
        if lon_dim in hist_raw.dims:
            hist_rolled = hist_rolled.pad({lon_dim: pad_size}, mode="wrap")
            hist_rolled = hist_rolled.rolling({lon_dim: window_spatial_hobday}, center=True, min_periods=1).sum()
            hist_rolled = hist_rolled.isel({lon_dim: slice(pad_size, pad_size + hist_raw.sizes[lon_dim])})

        # Standard rolling in latitude
        if lat_dim in hist_raw.dims:
            hist_rolled = hist_rolled.rolling({lat_dim: window_spatial_hobday}, center=True, min_periods=1).sum()

        hist_raw = hist_rolled

    def _compute_quantile_with_params(hist_chunk, bin_centers_chunk):
        return _rolling_histogram_quantile(hist_chunk, window_days_hobday, q, bin_centers_chunk)

    # Rechunk histogram so core dimensions are unchunked for apply_ufunc
    # Create chunk dict for hist_raw that preserves spatial chunks but drops time
    hist_chunk_dict = {dimensions["x"]: chunk_dict.get(dimensions["x"], 16), "dayofyear": -1, "da_bin": -1}
    if "y" in dimensions:
        hist_chunk_dict[dimensions["y"]] = chunk_dict.get(dimensions["y"], 16)

    hist_raw = hist_raw.chunk(hist_chunk_dict)

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

    if use_temp_checkpoints:
        logger.debug("Checkpointing threshold to break graph dependencies")
        threshold = checkpoint_to_zarr(threshold, name="threshold")

    # Drop time coordinate to avoid conflicts when comparing with data grouped by dayofyear
    if dimensions["time"] in threshold.coords:
        threshold = threshold.drop_vars(dimensions["time"])

    # Set threshold to NaN for spatial points that contain NaN values
    nan_mask = da.isel({dimensions["time"]: 0}).isnull().compute()
    threshold = threshold.where(~nan_mask).persist()

    # Validate threshold values against bounds
    upper_bound = bin_edges[-2]
    lower_bound = bin_edges[3]  # We want this to be positive so that constant=0 anomalies will not be "extreme"

    # Check if any values are too high (ignore NaN values)
    too_high = threshold > upper_bound
    if too_high.any():
        warnings.warn(
            f"Quantile values exceed expected range: max={threshold.max().compute():.4f} > {upper_bound:.4f}. "
            f"Consider increasing max_anomaly parameter (currently {max_anomaly:.2f}) or using a lower percentile threshold.",
            UserWarning,
            stacklevel=2,
        )

    # Check if any values are too low (ignore NaN values)
    too_low = threshold < lower_bound
    if too_low.any():
        warnings.warn(
            f"Quantile values below expected range in some locations: min={threshold.min().compute():.4f} < {lower_bound:.4f}. "
            "This is likely due to a constant anomaly in certain (e.g. due to sea ice). "
            "Double check the computed threshold values are correct.",
            UserWarning,
            stacklevel=2,
        )
        # Set too low values to lower bound -- This is to ensure that constant=0 anomalies will not be "extreme"
        threshold = threshold.where(~too_low, lower_bound)

    return threshold


def _compute_histogram_quantile_1d(
    da: xr.DataArray,
    q: float,
    dim: str = "time",
    bin_edges: Optional[NDArray[np.float64]] = None,
    precision: float = 0.01,
    max_anomaly: float = 5.0,
) -> xr.DataArray:
    """
    Efficiently compute quantiles using binned histograms optimised for extreme values.
    Uses fine-grained bins for positive anomalies and a single bin for negative values.
    Improved robust interpolation handles empty bins in the tails.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    q : float
        Quantile to compute (0-1)
    dim : str, optional
        Dimension along which to compute quantile
    bin_edges : numpy.ndarray, optional
        Custom bin edges for histogram computation
    precision : float, default=0.01
        Precision for positive anomaly bins
    max_anomaly : float, default=5.0
        Maximum anomaly value for binning

    Returns
    -------
    xarray.DataArray
        Computed quantile value for each spatial location
    """
    if bin_edges is None:
        # Create optimised asymmetric bins
        bin_edges = np.concatenate([[-np.inf], np.arange(-precision, max_anomaly + precision, precision)])

    # Compute histogram
    hist = histogram(da, bins=[bin_edges], dim=[dim]).persist()

    # Convert to PDF and CDF
    hist_sum = hist.sum(dim=f"{da.name}_bin") + 1e-10
    pdf = hist / hist_sum
    cdf = pdf.cumsum(dim=f"{da.name}_bin").persist()

    # Get bin centers
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_centers[0] = 0.0  # Set negative bin centre to 0
    eps = 1e-10

    # Find bins for interpolation
    # Find first bin where CDF >= (q - eps) - this becomes upper bound
    cdf_above_q = cdf >= (q - eps)
    idx_upper = cdf_above_q.argmax(dim=f"{da.name}_bin")

    # Get CDF value one point to the left of idx_upper
    idx_before_upper = xr.where(idx_upper - 1 > 0, idx_upper - 1, 0)

    # Extract the target CDF value (avoiding negative indexing issues)
    idx_before_upper_computed = idx_before_upper.compute()
    cdf_target = cdf.isel({f"{da.name}_bin": idx_before_upper_computed})

    # Find idx_lower: first bin where CDF > cdf_target
    cdf_above_target = cdf > cdf_target
    idx_lower = cdf_above_target.argmax(dim=f"{da.name}_bin")

    # Ensure bounds are valid
    idx_lower = xr.where(idx_lower < 0, 0, xr.where(idx_lower > len(bin_centers) - 2, len(bin_centers) - 2, idx_lower))
    idx_upper = xr.where(idx_upper < 1, 1, xr.where(idx_upper > len(bin_centers) - 1, len(bin_centers) - 1, idx_upper))

    # Extract CDF and bin values for interpolation
    idx_lower_computed = idx_lower.compute()
    idx_upper_computed = idx_upper.compute()

    cdf_lower = cdf.isel({f"{da.name}_bin": idx_lower_computed})
    cdf_upper = cdf.isel({f"{da.name}_bin": idx_upper_computed})
    bin_lower = bin_centers[idx_lower_computed]
    bin_upper = bin_centers[idx_upper_computed]

    # Robust interpolation with proper handling of degenerate cases
    denom = cdf_upper - cdf_lower

    # Handle exact matches and zero denominators
    exact_match = (xr.ufuncs.fabs(cdf_lower - q) < eps).persist()
    zero_denom = (xr.ufuncs.fabs(denom) <= eps).persist()

    # Standard interpolation
    frac = (q - cdf_lower) / xr.where(xr.ufuncs.fabs(denom) > eps, denom, 1.0)
    threshold = bin_lower + frac * (bin_upper - bin_lower)

    # For exact matches, use the lower bin center
    threshold = xr.where(exact_match, bin_lower, threshold)

    # For zero denominator without exact match, use bin midpoint
    no_exact_match = zero_denom & ~exact_match
    threshold = xr.where(no_exact_match, (bin_lower + bin_upper) / 2, threshold)

    # Set threshold to NaN for spatial points that contain NaN values
    nan_mask = da.isnull().any(dim=dim)
    threshold = threshold.where(~nan_mask).drop_vars(f"{da.name}_bin").persist()

    # Validate threshold against bounds
    upper_bound = bin_edges[-2]
    lower_bound = bin_edges[3]  # We want this to be positive so that constant=0 anomalies will not be "extreme"

    # Check if any values are too high (ignore NaN values)
    too_high = (threshold > upper_bound) & threshold.notnull()
    if too_high.any():
        warnings.warn(
            f"Quantile values exceed expected range: max={threshold.max().compute():.4f} > {upper_bound:.4f}. "
            f"Consider increasing max_anomaly parameter (currently {max_anomaly:.2f}) or using a lower percentile threshold.",
            UserWarning,
            stacklevel=2,
        )

    # Check if any values are too low (ignore NaN values)
    too_low = (threshold < lower_bound) & threshold.notnull()
    if too_low.any():
        warnings.warn(
            f"Quantile values below expected range in some locations: min={threshold.min().compute():.4f} < {lower_bound:.4f}. "
            "This is likely due to a constant anomaly in certain (e.g. due to sea ice). "
            "Double check the computed threshold values are correct.",
            UserWarning,
            stacklevel=2,
        )
        # Set too low values to lower bound -- This is to ensure that constant=0 anomalies will not be "extreme"
        threshold = threshold.where(~too_low, lower_bound).persist()

    return threshold


# ======================================
# Constant (in time) Extreme Definition
# ======================================


def _identify_extremes_constant(
    da: xr.DataArray,
    threshold_percentile: float = 95,
    method_percentile: Literal["exact", "approximate"] = "approximate",
    dimensions: Optional[Dict[str, str]] = None,
    precision: float = 0.01,
    max_anomaly: float = 5.0,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events exceeding a constant (in time) percentile threshold.
    i.e. There is 1 threshold for each spatial point, computed across all time.

    Returns both the extreme events boolean mask and the thresholds used.
    """
    if method_percentile == "exact":  # Compute exact percentile (memory-intensive)
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
        threshold = _compute_histogram_quantile_1d(
            da, threshold_percentile / 100.0, dim=dimensions["time"], precision=precision, max_anomaly=max_anomaly
        )

    # Clean up coordinates if needed
    if "quantile" in threshold.coords:
        threshold = threshold.drop_vars("quantile")

    # Ensure spatial dimensions are fully loaded for efficient comparison
    spatial_chunks = {dimensions[dim]: -1 for dim in ["x", "y"] if dim in dimensions}
    threshold = threshold.chunk(spatial_chunks).persist()

    # Create boolean mask for values exceeding threshold
    extremes = da >= threshold

    # Clean up coordinates if needed
    if "quantile" in extremes.coords:
        extremes = extremes.drop_vars("quantile")

    extremes = extremes.astype(bool).chunk(dict(zip(da.dims, da.chunks))).persist()

    return extremes, threshold
