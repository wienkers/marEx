"""
Top-level preprocessing orchestrator for the marEx detection pipeline.

Provides the public :func:`preprocess_data` entry point, a thin orchestrator that
chains anomaly computation and extreme identification into a single dataset, plus
the :func:`_get_preprocessing_steps` metadata helper. This module sits at the top
of the detect package dependency layering and ties together the anomaly,
extremes, and validation subsystems.
"""

import logging
from typing import Dict, List, Literal, Optional, Tuple

import dask
import numpy as np
import xarray as xr
from dask import persist
from dask.base import is_dask_collection

from ..exceptions import ConfigurationError, create_data_validation_error
from ..logging_config import configure_logging, get_logger, log_dask_info, log_memory_usage, log_timing
from .anomaly.base import compute_normalised_anomaly
from .config import PreprocessConfig
from .extremes.base import identify_extremes
from .utils import make_netcdf_safe_attrs
from .validation import _infer_dims_coords, _validate_data_values

# Get module logger
logger = get_logger(__name__)


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
    reference_period: Optional[Tuple[int, int]] = None,  # for fixed_baseline & detrend_fixed_baseline
    method_percentile: Literal["exact", "approximate"] = "approximate",
    precision: float = 0.01,
    max_anomaly: float = 5.0,
    dask_chunks: Optional[Dict[str, int]] = None,
    dimensions: Optional[Dict[str, str]] = None,
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
    reference_period : tuple of (int, int), optional
        Year range (start_year, end_year) inclusive for computing the daily climatology
        (fixed_baseline and detrend_fixed_baseline only). If None (default), uses all
        available years. Anomalies are computed for the full time series regardless.
        Example: reference_period=(1990, 2020) computes the climatology from 1990-2020
        but outputs anomalies for the entire input time range.
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

    # Bundle the resolved tuning parameters into a single validated, immutable
    # configuration object that drives the rest of the pipeline. The public
    # keyword-argument signature above is preserved exactly; this config is an
    # internal container only (behaviour-preserving). Data arrays (``da``,
    # ``neighbours``, ``cell_areas``) and logging flags (``verbose``/``quiet``)
    # remain plain locals and are not part of the config.
    config = PreprocessConfig(
        method_anomaly=method_anomaly,
        method_extreme=method_extreme,
        threshold_percentile=threshold_percentile,
        window_year_baseline=window_year_baseline,
        smooth_days_baseline=smooth_days_baseline,
        window_days_hobday=window_days_hobday,
        window_spatial_hobday=window_spatial_hobday,
        std_normalise=std_normalise,
        detrend_orders=detrend_orders,
        force_zero_mean=force_zero_mean,
        reference_period=reference_period,
        method_percentile=method_percentile,
        precision=precision,
        max_anomaly=max_anomaly,
        dask_chunks=dask_chunks,
    )

    # Unpack the validated config back into local names so the orchestration body
    # below is driven by the config values while remaining verbatim.
    method_anomaly = config.method_anomaly
    method_extreme = config.method_extreme
    threshold_percentile = config.threshold_percentile
    window_year_baseline = config.window_year_baseline
    smooth_days_baseline = config.smooth_days_baseline
    window_days_hobday = config.window_days_hobday
    window_spatial_hobday = config.window_spatial_hobday
    std_normalise = config.std_normalise
    detrend_orders = config.detrend_orders
    force_zero_mean = config.force_zero_mean
    reference_period = config.reference_period
    method_percentile = config.method_percentile
    precision = config.precision
    max_anomaly = config.max_anomaly
    dask_chunks = config.dask_chunks

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

    # Validate reference_period before triggering any computation
    if reference_period is not None and method_anomaly not in ("fixed_baseline", "detrend_fixed_baseline"):
        raise ConfigurationError(
            f"reference_period is not supported for method_anomaly='{method_anomaly}'",
            details="reference_period is only applicable to 'fixed_baseline' and 'detrend_fixed_baseline' methods",
            suggestions=[
                "Remove the reference_period parameter, or",
                "Use method_anomaly='fixed_baseline' or 'detrend_fixed_baseline'",
            ],
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
            reference_period,
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
                window_spatial_hobday,
                reference_period,
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
        attrs = {}
        if reference_period is not None:
            attrs["reference_period"] = list(reference_period)
        ds.attrs.update(attrs)
    elif method_anomaly == "detrend_fixed_baseline":
        attrs = {
            "detrend_orders": detrend_orders,
            "force_zero_mean": force_zero_mean,
        }
        if reference_period is not None:
            attrs["reference_period"] = list(reference_period)
        ds.attrs.update(attrs)

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
    # (stale ``chunks`` encoding can otherwise trigger chunk-misalignment errors on save)
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

        log_memory_usage(logger, "After dataset persistence", logging.DEBUG)

    # Final success reporting with summary
    extreme_count = ds.extreme_events.sum()
    if hasattr(extreme_count, "compute"):
        extreme_count = extreme_count.compute()

    logger.info(f"Preprocessing completed successfully - {extreme_count} extreme events identified")
    logger.debug(f"Final dataset shape: {ds.dims}")
    log_dask_info(logger, ds, "Final preprocessed dataset")

    # Ensure the returned dataset is directly saveable to *both* Zarr and NetCDF.
    # Booleans/None in attrs round-trip through Zarr but break Dataset.to_netcdf.
    ds = make_netcdf_safe_attrs(ds)

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
    reference_period: Optional[Tuple[int, int]] = None,
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
        if reference_period is not None:
            steps.append(f"Daily climatology computed from {reference_period[0]}-{reference_period[1]}")
        else:
            steps.append("Daily climatology computed from full time series")
    elif method_anomaly == "detrend_fixed_baseline":
        steps.append(f"Removed polynomial trend orders={detrend_orders}")
        if reference_period is not None:
            steps.append(f"Daily climatology computed from detrended data ({reference_period[0]}-{reference_period[1]})")
        else:
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
