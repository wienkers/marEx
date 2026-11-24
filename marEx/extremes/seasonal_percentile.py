"""
Seasonal (day-of-year specific) extreme identification.

Provides :func:`_identify_extremes_seasonal`, which computes day-of-year resolved
percentile thresholds (with a rolling day window) to identify extreme events.
This backs the ``seasonal_percentile`` extreme-detection method.

The day-of-year percentile definition implemented here follows Hobday, A. J.,
et al. (2016), "A hierarchical approach to defining marine heatwaves",
*Progress in Oceanography* 141, 227-238. The definition itself is not specific to
any variable or domain: it applies unchanged to air temperature, precipitation,
soil moisture, or a biogeochemical tracer.
"""

from typing import Dict, Literal, Optional, Tuple

import dask.array
import numpy as np
import xarray as xr

from ..core.compute_mode import Materialiser
from ..core.dimensions import TASK_ELEMENTS, horizontal_dims, spatial_dims
from ..core.seasonal_apply import apply_seasonal_field
from ..core.time_axis import SeasonalCycle, resolve_cycle
from ..logging_config import get_logger
from .histogram import _chunk_spatial_for_histogram, _clamp_exact_threshold, _compute_histogram_quantile_2d, _x_is_periodic

# Get module logger
logger = get_logger(__name__)

# Fewest expected threshold exceedances per cell and cycle slot below which the threshold is
# called unreliable.
_MIN_EXCEEDANCES = 10

# Allowance for one concurrent task, including its overlapped source and index.
_EXACT_POOL_TASK_BYTES = 128 * 2**20


def _exact_pool_tiles(n_time, n_slots, shape, halo, half, wrap, itemsize, max_active, full_cycle):
    """Size a single extra-dimension slice, including the halo and reduction workspace."""
    ny, nx = shape
    hy, hx = halo
    allowed = (max(1, ny // max(1, hy)), 1 if wrap and hx == 0 else max(1, nx // max(1, hx)))
    counts = [1, 1]
    width = 2 * half + 1
    rows = min(ny, width)
    samples = max_active * rows * (width if wrap else min(nx, width))
    # A replay holds a native sample crop plus NaN indices, masks and, for
    # wrapped windows, the ordered horizontal indices.
    oracle_bytes = (2 * itemsize + 16) * samples + (8 * (rows + width) if wrap and hx == 0 else 0)

    def working_set(indexed):
        cy, cx = ((size + count - 1) // count for size, count in zip(shape, counts))
        ay, ax = cy + 2 * hy, cx + 2 * hx
        cells, core = ay * ax, cy * cx
        # Source and overlap assembly, output thresholds and the shared slot index.
        shared = 3 * itemsize * n_time * cells + 4 * n_slots * core + 32 * (n_time + n_slots)
        if not indexed:
            return shared + oracle_bytes
        # Native-to-float64 copy, uint16 IDs, 8-byte postings, sort keys and
        # slot-update temporaries. The cell/bucket allowance includes extrema,
        # offsets, cursors, counts and both strip-prefix workspaces.
        return (
            shared
            + 48 * n_time * cells
            + 256 * (80 * cells + 32 * core + 8 * (ax + cy + 3))
            + (1 if full_cycle else n_slots) * core
            + max(16 * samples, oracle_bytes)
            + 80 * core
            + 512 * 1024
        )

    direct = False
    while working_set(not direct) > _EXACT_POOL_TASK_BYTES:
        candidates = [axis for axis in range(2) if counts[axis] < allowed[axis]]
        if not candidates:
            if direct:
                raise MemoryError(f"The exact {width}-cell spatial window exceeds the per-task memory budget")
            direct = True
            counts = [1, 1]
            continue
        axis = max(candidates, key=lambda axis: shape[axis] / counts[axis])
        counts[axis] = min(2 * counts[axis], allowed[axis])

    return tuple(
        (size // count + 1,) * (size % count) + (size // count,) * (count - size % count) for size, count in zip(shape, counts)
    ), direct


def _exact_pool_block(block, slots, n_slots, window_steps, half, halo, shape, percentile, x_periodic, direct, block_info=None):
    """Drop artificial edge padding before reducing the block's core centres."""
    from ._exact_spatial_pool import _pooled_percentiles

    # Extra axes have unit chunks. Time is whole for threshold calculation.
    values = block.reshape(block.shape[-3:])
    origin, core_shape, crop = [], [], []
    for axis, (depth, size) in enumerate(zip(halo, shape), start=-2):
        start, end = block_info[None]["array-location"][axis]
        left_pad = max(0, depth - start)
        right_pad = max(0, end + depth - size)
        if axis == -1 and x_periodic:
            left_pad = right_pad = 0
        crop.append(slice(left_pad, values.shape[axis] - right_pad))
        origin.append(depth - left_pad)
        core_shape.append(end - start)
    result = _pooled_percentiles(
        values[:, crop[0], crop[1]],
        slots,
        n_slots,
        window_steps,
        half,
        origin,
        core_shape,
        percentile,
        periodic=x_periodic and halo[1] == 0,
        direct=direct,
    )
    return result.reshape(block.shape[:-3] + result.shape)


def _exact_spatial_threshold(da, dimensions, cycle, window_steps, window_spatial, percentile):
    """Build haloed horizontal tasks and a lazy all-time dry-centre mask."""
    time_dim, y_dim, x_dim = (dimensions[key] for key in ("time", "y", "x"))
    spatial = tuple(dim for dim in da.dims if dim != time_dim)
    extra = tuple(dim for dim in spatial if dim not in (y_dim, x_dim))
    cycle_dim = cycle.index_name
    n_slots = cycle.length
    n_time, ny, nx = (da.sizes[dim] for dim in (time_dim, y_dim, x_dim))
    half = int(window_spatial) // 2
    x_periodic = _x_is_periodic(da, x_dim)
    # A wide global window can visit one longitude several times. Keep the
    # axis whole and let the kernel wrap the offsets in that case.
    wrap_in_kernel = x_periodic and half >= nx
    halo = (min(half, max(0, ny - 1)), 0 if wrap_in_kernel else half if x_periodic else min(half, max(0, nx - 1)))

    raw_slots = np.asarray(da[cycle_dim].values)
    valid = (raw_slots >= 1) & (raw_slots <= n_slots) & (raw_slots == np.floor(raw_slots))
    slots = np.where(valid, raw_slots, 0).astype(np.int64)
    # Number of timestamps in the largest temporal window, including repeats
    # and excluding invalid labels. Used to bound a single refinement buffer.
    width = min(window_steps, n_slots)
    counts = np.bincount(slots, minlength=n_slots + 1)[1:]
    counts = np.concatenate(([0], np.tile(counts, 2))).cumsum()
    max_active = int(np.max(counts[width : width + n_slots] - counts[:n_slots]))
    (y_chunks, x_chunks), direct = _exact_pool_tiles(
        n_time, n_slots, (ny, nx), halo, half, x_periodic, da.dtype.itemsize, max_active, width == n_slots
    )
    data = (
        da.transpose(*extra, time_dim, y_dim, x_dim)
        .chunk({**dict.fromkeys(extra, 1), time_dim: -1, y_dim: y_chunks, x_dim: x_chunks})
        .data
    )
    # Integer padding cannot be NaN. The block function crops the artificial
    # padding on both regional axes before either reducer sees it.
    pad = np.nan if da.dtype.kind in "fc" else 0
    axes = (data.ndim - 2, data.ndim - 1)
    data = dask.array.overlap.overlap(
        data,
        depth=dict(zip(axes, halo)),
        boundary={axes[0]: pad, axes[1]: "periodic" if x_periodic else pad},
        allow_rechunk=False,
    )

    results = dask.array.map_blocks(
        _exact_pool_block,
        data,
        slots=slots,
        n_slots=n_slots,
        window_steps=window_steps,
        half=half,
        halo=halo,
        shape=(ny, nx),
        percentile=percentile,
        x_periodic=x_periodic,
        direct=direct,
        chunks=data.chunks[:-3] + ((n_slots,), y_chunks, x_chunks),
        drop_axis=len(extra),
        new_axis=len(extra),
        dtype=np.float32,
        meta=np.array((), dtype=np.float32),
    )
    dims = (*extra, cycle_dim, y_dim, x_dim)
    coords = {name: coord for name, coord in da.coords.items() if time_dim not in coord.dims and cycle_dim not in coord.dims}
    coords[cycle_dim] = np.arange(1, n_slots + 1)
    thresholds = xr.DataArray(results, dims=dims, coords=coords).transpose(cycle_dim, *spatial)
    return thresholds.where(~da.isnull().all(dim=time_dim))


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
    range_mode: Literal["pinned", "estimated", "data"] = "pinned",
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
        Width in cells of the square spatial pooling window. Used as given here: the
        approximate default of 5 on gridded data is applied upstream by
        :func:`marEx.extremes.base.resolve_window_spatial`, so ``None`` means no pooling.
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
    # 10 is below a 30-year, 11-day climatology at the 95th percentile (16.5 on one cell), the
    # standard Hobday et al. (2016) set-up, so that set-up passes and only thinner windows warn.
    if N_above_threshold < _MIN_EXCEEDANCES:
        spatial_advice = " or using a larger window_spatial" if all(key in dimensions for key in ("x", "y")) else ""
        logger.warning(
            f"Not enough samples for accurate extreme detection: {N_above_threshold:g} < {_MIN_EXCEEDANCES}. "
            "Consider using a lower threshold_percentile, increasing your time-series size, "
            f"increasing the window_days{spatial_advice}. Below this count the threshold depends on the percentile convention "
            "as much as on the data: exact and approximate can differ by many histogram bins."
        )

    # Add day-of-year coordinate (compute it to avoid chunked groupby issues).
    # No persist and no rechunk here: the rechunk restated the array's own chunks (a no-op)
    # and the persist duplicated the pipeline-level anomaly persist, pinning a second
    # full-size copy -- ~38 GB at 0.25 deg / 25 yr.
    cycle = resolve_cycle(da, coordinates["time"], cycle)
    cycle_dim = cycle.index_name
    # `window_days` is a physical duration, and the window is applied along the cycle axis
    # (the exact path's `half_w` slot offsets, the histogram path's wrap-padded rows), so it is
    # converted into cycle SLOTS, forced odd to sit symmetric about its centre slot. On daily
    # and evenly sub-daily data a slot is one step (11 days -> 11 daily slots), which keeps
    # those thresholds bit-identical; 5-day data still has one slot per day. A window wider
    # than the cycle pools every slot once, on both paths.
    window_steps = cycle.window_slots(window_days, calendar=da[coordinates["time"]].dt.calendar)

    da = da.assign_coords({cycle_dim: cycle.index_of(da[coordinates["time"]]).compute()})

    # Group by cycle index and compute percentile
    if method_percentile == "exact" and window_spatial not in (None, 1):
        thresholds = _exact_spatial_threshold(da, dimensions, cycle, window_steps, window_spatial, threshold_percentile)
        thresholds = _clamp_exact_threshold(thresholds, tail)
    elif method_percentile == "exact":
        from ._exact_seasonal import _doy_percentiles as _exact_doy_percentiles

        # Use apply_ufunc to compute DOY percentiles per spatial chunk.
        # Tile the spatial dims alongside time:-1. Rechunking only time leaves a
        # spatially-unchunked pipeline anomaly as a single (time, y, x) task, which is a
        # guaranteed worker OOM at scale; the constant-threshold exact path already tiles
        # for exactly this reason. Per-cell percentiles are independent of the tiling, so
        # this changes task granularity only.
        # Each cell yields one percentile per cycle slot (366 on daily data), so budget
        # the tile against that as well as against the time slab: a series shorter than
        # the cycle would otherwise get a tile whose output exceeds its own budget.
        da_ufunc = _chunk_spatial_for_histogram(
            da, dimensions["time"], output_elements_per_cell=cycle.length, horizontal=horizontal_dims(dimensions)
        )
        cycle_vals = np.asarray(cycle.index_of(da_ufunc[coordinates["time"]]).values, dtype=np.int32)
        n_slots = cycle.length

        def _doy_percentiles(data, cycle_vals, percentile):
            """Per-slot percentiles. data: (*spatial, time) -> (*spatial, cycle.length)."""
            return _exact_doy_percentiles(data, cycle_vals, n_slots, window_steps, percentile)

        thresholds = xr.apply_ufunc(
            _doy_percentiles,
            da_ufunc,
            input_core_dims=[[dimensions["time"]]],
            output_core_dims=[[cycle_dim]],
            dask="parallelized",
            kwargs={"cycle_vals": cycle_vals, "percentile": threshold_percentile},
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"output_sizes": {cycle_dim: cycle.length}},
        )

        # Assign cycle coordinate values and move the cycle dim to first position
        thresholds = thresholds.assign_coords({cycle_dim: np.arange(1, cycle.length + 1)}).transpose(cycle_dim, ...)
        thresholds = _clamp_exact_threshold(thresholds, tail)
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
            range_mode=range_mode,
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

    # Keep the threshold storage chunks aligned with the input's modal spatial chunks.
    # Cap the stored cycle chunks as well. On a spatially whole grid, holding the cycle
    # whole here could make a single multi-gigabyte block before the comparison is tiled.
    spatial_block = int(np.prod(list(spatial_chunks.values()), dtype=np.int64))
    cycle_chunk = max(1, min(cycle.length, TASK_ELEMENTS // max(1, spatial_block)))
    align_chunks = {**spatial_chunks, cycle_dim: cycle_chunk}
    logger.debug(f"Aligning threshold chunks to match input data spatial chunks: {align_chunks}")
    thresholds = thresholds.chunk(align_chunks)

    # Anchor the thresholds BEFORE the comparison is built on top of them. `extremes` is
    # a lazy expression over `thresholds`; anchoring after this line would leave it
    # pointing at the original graph and the whole threshold reduction would run a second
    # time. On an unstructured mesh `thresholds` is 366 x ncells x 4 B (21.8 GB at ICON
    # R02B09) -- space-scaled, so no amount of time-chunking shrinks it. In streaming mode
    # the histogram driver has already pinned it in cluster RAM for the bounds check
    # (`pin_bounded`); staging it here writes it to disk so the pin is released and
    # every downstream consumer reads the store.
    thresholds = materialiser.stage(thresholds, threshold_label)
    extremes = apply_seasonal_field(
        da,
        thresholds,
        da[cycle_dim].values,
        timedim=dimensions["time"],
        cycledim=cycle_dim,
        operation=tail,
    )

    # Keep the established modal chunk layout for Zarr.
    time_chunks = da.chunksizes[dimensions["time"]]
    time_chunk_size = max(set(time_chunks), key=time_chunks.count)
    rechunk_dict = {dimensions["time"]: time_chunk_size}
    rechunk_dict.update(spatial_chunks)
    logger.debug(f"Rechunking extremes to the input's modal chunks: {rechunk_dict}")
    extremes = extremes.chunk(rechunk_dict)

    return extremes, thresholds
