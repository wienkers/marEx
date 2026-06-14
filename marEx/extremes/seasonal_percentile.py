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
from ..core.dimensions import spatial_dims
from ..core.time_axis import SeasonalCycle, resolve_cycle
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
    cycle: Optional[SeasonalCycle] = None,
    tail: Literal["upper", "lower"] = "upper",
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
    tail : {'upper', 'lower'}, default='upper'
        Which side of the distribution counts as extreme. The threshold is the
        ``threshold_percentile``-th percentile either way; only the comparison
        flips, so ``threshold_percentile=5, tail='lower'`` gives the coldest 5 %.

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
    # Samples landing beyond the threshold -- the count the estimate actually rests
    # on. In the lower tail that is the percentile itself, not its complement; using
    # the complement there would keep the warning silent in exactly the sparse case
    # it exists for (`threshold_percentile=5` has 5 % of samples below it, not 95 %).
    tail_fraction = (1.0 - threshold_percentile / 100.0) if tail == "upper" else (threshold_percentile / 100.0)
    N_above_threshold = N_samples * tail_fraction
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
    cycle = resolve_cycle(da, coordinates["time"], cycle)
    cycle_dim = cycle.index_name
    # `window_days` is a physical duration; convert it to whole timesteps. It must be
    # odd so the window is symmetric about its centre step -- both the exact path's
    # `half_w` offsets and the histogram path's wrap-padded sliding window depend on
    # that. On daily data the conversion is the identity (11 days -> 11 steps), which
    # is what keeps every existing seasonal threshold bit-identical.
    window_steps = cycle.window_steps(window_days)

    da = da.assign_coords({cycle_dim: cycle.index_of(da[coordinates["time"]]).compute()})

    # Group by cycle index and compute percentile
    if method_percentile == "exact":
        # Use apply_ufunc to compute DOY percentiles per spatial chunk in pure numpy.
        # Tile the spatial dims alongside time:-1. Rechunking only time leaves a
        # spatially-unchunked pipeline anomaly as a single (time, y, x) task, which is a
        # guaranteed worker OOM at scale; the constant-threshold exact path already tiles
        # for exactly this reason. Per-cell percentiles are independent of the tiling, so
        # this changes task granularity only (review finding 3.10).
        # Each cell yields one percentile per cycle slot (366 on daily data), so budget
        # the tile against that as well as against the time slab: a series shorter than
        # the cycle would otherwise get a tile whose output exceeds its own budget.
        da_ufunc = _chunk_spatial_for_histogram(da, dimensions["time"], output_elements_per_cell=cycle.length)
        cycle_vals = cycle.index_of(da_ufunc[coordinates["time"]]).values
        half_w = window_steps // 2

        # Pre-compute boolean masks (which time indices contribute to each cycle slot)
        doy_masks = []
        for slot in range(1, cycle.length + 1):
            mask = np.zeros(len(cycle_vals), dtype=bool)
            for offset in range(-half_w, half_w + 1):
                target = ((slot - 1 + offset) % cycle.length) + 1
                mask |= cycle_vals == target
            doy_masks.append(mask)

        n_slots = cycle.length

        def _doy_percentiles(data, doy_masks, percentile):
            """Per-slot percentiles. data: (*spatial, time) -> (*spatial, cycle.length)."""
            result = np.full(data.shape[:-1] + (n_slots,), np.nan, dtype=np.float32)
            for i, mask in enumerate(doy_masks):
                if mask.any():
                    result[..., i] = np.nanpercentile(data[..., mask], percentile, axis=-1)
            return result

        thresholds = xr.apply_ufunc(
            _doy_percentiles,
            da_ufunc,
            input_core_dims=[[dimensions["time"]]],
            output_core_dims=[[cycle_dim]],
            dask="parallelized",
            kwargs={"doy_masks": doy_masks, "percentile": threshold_percentile},
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"output_sizes": {cycle_dim: cycle.length}},
        )

        # Assign cycle coordinate values and move the cycle dim to first position
        thresholds = thresholds.assign_coords({cycle_dim: np.arange(1, cycle.length + 1)}).transpose(cycle_dim, ...)
    else:  # Optimised histogram approximation method
        thresholds = _compute_histogram_quantile_2d(
            da,
            threshold_percentile / 100.0,
            window_steps=window_steps,
            window_spatial=window_spatial,
            dimensions=dimensions,
            precision=precision,
            max_anomaly=max_anomaly,
            materialiser=materialiser,
            cycle=cycle,
            tail=tail,
        )

    # Extract spatial chunk sizes from input data for alignment
    # Use most common chunk size to handle irregular chunks robustly
    # Every spatial dim of the input, extra dims (depth, level) included -- the
    # thresholds carry them as broadcast axes and must be aligned on all of them.
    spatial_chunks = {}
    for dim_name in spatial_dims(da, dimensions):
        chunks_tuple = da.chunksizes[dim_name]
        # Get the most common chunk size (handles irregular chunks better)
        spatial_chunks[dim_name] = max(set(chunks_tuple), key=chunks_tuple.count)

    # Drop time coordinate/dimension to avoid conflicts when comparing with data grouped by cycle slot
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
    # Assign the cycle-index coordinate and use UniqueGrouper for chunked arrays
    da = da.assign_coords({cycle_dim: cycle.index_of(da[coordinates["time"]])})
    # Anchor the thresholds BEFORE the comparison is built on top of them. `extremes` is
    # a lazy expression over `thresholds`; anchoring after this line would leave it
    # pointing at the original graph and the whole threshold reduction would run a second
    # time. On an unstructured mesh `thresholds` is 366 x ncells x 4 B (21.8 GB at ICON
    # R02B09) -- space-scaled, so no amount of time-chunking shrinks it, which is why
    # streaming mode stages it to disk rather than pinning it in RAM.
    thresholds = materialiser.stage(thresholds, threshold_label)
    grouped = da.groupby({cycle_dim: xr.groupers.UniqueGrouper(labels=np.arange(1, cycle.length + 1))})
    extremes = (grouped >= thresholds) if tail == "upper" else (grouped <= thresholds)

    # Drop the now-unnecessary cycle-index coordinate
    if cycle_dim in extremes.coords:
        extremes = extremes.drop_vars(cycle_dim)

    # Rechunk to fix irregular time chunks created by groupby operation
    # Zarr requires uniform chunks, so we rechunk to match input data's time chunks
    time_chunks = da.chunksizes[dimensions["time"]]
    time_chunk_size = max(set(time_chunks), key=time_chunks.count)
    rechunk_dict = {dimensions["time"]: time_chunk_size}
    rechunk_dict.update(spatial_chunks)
    logger.debug(f"Rechunking extremes to fix irregular chunks from groupby: {rechunk_dict}")
    extremes = extremes.chunk(rechunk_dict)

    return extremes, thresholds
