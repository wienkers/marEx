"""
Seasonal (day-of-year specific) extreme identification.

Provides :func:`_identify_extremes_seasonal`, which computes day-of-year resolved
percentile thresholds (with a rolling day window) to identify extreme events.
This backs the ``seasonal_percentile`` extreme-detection method. The exact-percentile
branch uses a nested ``_doy_percentiles`` closure that must remain inside the
parent function.

The day-of-year percentile definition implemented here follows Hobday, A. J.,
et al. (2016), "A hierarchical approach to defining marine heatwaves",
*Progress in Oceanography* 141, 227-238. The definition itself is not specific to
any variable or domain: it applies unchanged to air temperature, precipitation,
soil moisture, or a biogeochemical tracer.
"""

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import xarray as xr

from ..core.compute_mode import Materialiser
from ..logging_config import get_logger
from .histogram import _chunk_spatial_for_histogram, _compute_histogram_quantile_2d

# Get module logger
logger = get_logger(__name__)


def _identify_extremes_seasonal(
    da: xr.DataArray,
    threshold_percentile: float = 95,
    window_days: int = 11,
    window_spatial: Optional[int] = None,
    method_percentile: Literal["exact", "approximate"] = "approximate",
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    precision: float = 0.01,
    max_anomaly: float = 5.0,
    materialiser: Optional[Materialiser] = None,
    threshold_label: str = "thresholds",
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events using day-of-year (i.e. climatological percentile threshold).

    For each spatial point and day-of-year, computes the p-th percentile of values within a
    window_days day window across all years.
    This implements the standard methodology for marine heatwave detection threshold calculation.

    Parameters:
    -----------
    da : xarray.DataArray
        Anomaly data with dimensions (time, lat, lon)
        Must be chunked with time dimension unbounded (time: -1)
    threshold_percentile : float, default=95
        Percentile to compute (0-100)
    window_days : int, default=11
        Window in days
    window_spatial : int, default=None
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
    # A None materialiser means "default to persist mode", which keeps every existing
    # caller, doctest and test working unchanged.
    if materialiser is None:
        materialiser = Materialiser("persist")

    # Check if there is sufficient samples
    N_years = np.unique(da[coordinates["time"]].dt.year).size
    N_samples = N_years * window_days * (window_spatial if window_spatial is not None else 1) ** 2
    N_above_threshold = N_samples * (1.0 - threshold_percentile / 100.0)
    if N_above_threshold < 50:
        # Make warning
        logger.warning(
            f"Not enough samples for accurate extreme detection: {N_above_threshold} < 50. "
            "Consider using a lower threshold_percentile, increasing your time-series size, "
            "increasing the window_days, or using a larger window_spatial."
            "If your time-series is very short, consider using method_percentile='exact'."
        )

    # Add day-of-year coordinate (compute it to avoid chunked groupby issues).
    # No persist and no rechunk here: the rechunk restated the array's own chunks (a no-op)
    # and the persist duplicated the pipeline-level anomaly persist, pinning a second
    # full-size copy -- ~38 GB at 0.25 deg / 25 yr (review finding 3.14).
    da = da.assign_coords(dayofyear=da[coordinates["time"]].dt.dayofyear.compute())

    # Group by day-of-year and compute percentile
    if method_percentile == "exact":
        # Use apply_ufunc to compute DOY percentiles per spatial chunk in pure numpy.
        # Tile the spatial dims alongside time:-1. Rechunking only time leaves a
        # spatially-unchunked pipeline anomaly as a single (time, y, x) task, which is a
        # guaranteed worker OOM at scale; the constant-threshold exact path already tiles
        # for exactly this reason. Per-cell percentiles are independent of the tiling, so
        # this changes task granularity only (review finding 3.10).
        # Each cell yields 366 day-of-year percentiles, so budget the tile against that as
        # well as against the time slab: a series shorter than 366 days would otherwise get
        # a tile whose output exceeds the budget the tile was sized by.
        da_ufunc = _chunk_spatial_for_histogram(da, dimensions["time"], output_elements_per_cell=366)
        dayofyear_vals = da_ufunc[coordinates["time"]].dt.dayofyear.values
        half_w = window_days // 2

        # Pre-compute boolean masks (which time indices contribute to each DOY)
        doy_masks = []
        for doy in range(1, 367):
            mask = np.zeros(len(dayofyear_vals), dtype=bool)
            for offset in range(-half_w, half_w + 1):
                target = ((doy - 1 + offset) % 366) + 1
                mask |= dayofyear_vals == target
            doy_masks.append(mask)

        def _doy_percentiles(data, doy_masks, percentile):
            """Compute per-DOY percentiles. data: (*spatial, time) -> (*spatial, 366)."""
            result = np.full(data.shape[:-1] + (366,), np.nan, dtype=np.float32)
            for i, mask in enumerate(doy_masks):
                if mask.any():
                    result[..., i] = np.nanpercentile(data[..., mask], percentile, axis=-1)
            return result

        thresholds = xr.apply_ufunc(
            _doy_percentiles,
            da_ufunc,
            input_core_dims=[[dimensions["time"]]],
            output_core_dims=[["dayofyear"]],
            dask="parallelized",
            kwargs={"doy_masks": doy_masks, "percentile": threshold_percentile},
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"output_sizes": {"dayofyear": 366}},
        )

        # Assign dayofyear coordinate values and move dayofyear to first dimension
        thresholds = thresholds.assign_coords(dayofyear=np.arange(1, 367)).transpose("dayofyear", ...)
    else:  # Optimised histogram approximation method
        thresholds = _compute_histogram_quantile_2d(
            da,
            threshold_percentile / 100.0,
            window_days=window_days,
            window_spatial=window_spatial,
            dimensions=dimensions,
            precision=precision,
            max_anomaly=max_anomaly,
            materialiser=materialiser,
        )

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
    # Anchor the thresholds BEFORE the comparison is built on top of them. `extremes` is
    # a lazy expression over `thresholds`; anchoring after this line would leave it
    # pointing at the original graph and the whole threshold reduction would run a second
    # time. On an unstructured mesh `thresholds` is 366 x ncells x 4 B (21.8 GB at ICON
    # R02B09) -- space-scaled, so no amount of time-chunking shrinks it, which is why
    # streaming mode stages it to disk rather than pinning it in RAM.
    thresholds = materialiser.stage(thresholds, threshold_label)
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

    return extremes, thresholds
