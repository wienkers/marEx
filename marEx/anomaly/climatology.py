"""
Rolling-climatology computation for the shifting-baseline anomaly method.

Provides :func:`rolling_climatology` and :func:`smoothed_rolling_climatology`,
both exported from :mod:`marEx.anomaly`. They use flox to compute
multi-year rolling day-of-year climatologies.
"""

from dataclasses import replace
from typing import Dict, Optional

import dask.array
import flox.xarray
import numpy as np
import pandas as pd
import xarray as xr
from dask import persist

from ..core import dimensions as core_dimensions
from ..core.dimensions import spatial_dims, tile_spatial_chunks
from ..core.encoding import clear_store_encoding
from ..core.numerics import rolling_numerics
from ..core.time_axis import SeasonalCycle, resolve_cycle
from ..core.validation import _infer_dims_coords, unmask_dataarray
from ..exceptions import ConfigurationError
from ..logging_config import get_logger

# Get module logger
logger = get_logger(__name__)


def last_day_pool(time_coord: xr.DataArray, cycle: SeasonalCycle, read_coord: Optional[xr.DataArray] = None) -> Optional[tuple]:
    """Time positions and cycle slots that feed the climatology of a leap year's last day.

    Grouping by calendar day of year gives day 366 only the leap years' 31 December, a
    quarter of the samples every other slot gets, and none at all when no leap year is in
    the baseline. On a trending series that lone sample sits years from the target, so a leap
    year's 31 December inherits the trend as a whole-field anomaly. Day 366 instead takes
    every year's 31 December: a non-leap year's, which is day 365, is mapped onto the day-366
    slot at the same time of day.

    Returns ``(positions, slots, labels)``: positions into ``time_coord``, the day-366 slot
    each feeds, and the day-366 slot labels. ``None`` for a monthly cycle, which has no
    day-366 slot, and for a series with no day 366 at all (a no-leap calendar, or a span
    without a leap year), whose day-366 slot no timestep reads. ``read_coord`` is the series
    the climatology is subtracted from when that differs from ``time_coord`` (a reference
    period); it decides whether day 366 is read.
    """
    if not (cycle.is_daily or cycle.is_subdaily):
        return None
    steps_per_day = cycle.steps_per_day
    first_slot = cycle.length - steps_per_day + 1
    index = np.asarray(cycle.index_of(time_coord).values)
    own = index >= first_slot
    read = own if read_coord is None else np.asarray(cycle.index_of(read_coord).values) >= first_slot
    if not read.any():
        return None
    borrowed = (np.asarray(time_coord.dt.month.values) == 12) & (np.asarray(time_coord.dt.day.values) == 31) & ~own
    positions = np.flatnonzero(own | borrowed)
    slots = np.where(borrowed, index + steps_per_day, index)[positions].astype(np.int32)
    return positions, slots, np.arange(first_slot, cycle.length + 1, dtype=np.int32)


def _gather_slots(block: np.ndarray, flat_index: np.ndarray) -> np.ndarray:
    """Read one value per time step from the trailing group axis of a tile."""
    return block[..., flat_index]


def _group_contributions(
    positions: np.ndarray, slots: np.ndarray, years: np.ndarray, target_offsets: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Expand samples in input order, then in ascending target year within each sample."""
    time_indices, keys = [], []
    for position, slot in zip(positions.tolist(), slots.tolist()):
        if slot < 0:
            continue
        targets = target_offsets[years[position]]
        time_indices.extend([position] * targets.size)
        keys.extend((targets + slot).tolist())
    return np.asarray(time_indices, dtype=np.int32), np.asarray(keys, dtype=np.int64)


def _reduce_occupied(data: xr.DataArray, dim: str, keys: np.ndarray, occupied: np.ndarray, method: Optional[str]) -> xr.DataArray:
    """Reduce compact group codes without changing the order or chunks of the samples."""
    return flox.xarray.xarray_reduce(
        data,
        xr.DataArray(np.searchsorted(occupied, keys), dims=[dim], name="target_year"),
        dim=dim,
        func="nanmean",
        expected_groups=(pd.RangeIndex(occupied.size),),
        isbin=False,
        method=method,
        dtype=np.float32,
        fill_value=np.nan,
    )


def _slot_positions(keys: np.ndarray, targets: np.ndarray, offset: int, missing: int) -> np.ndarray:
    """Find target keys in a sorted table; absent keys read the shared NaN row."""
    if not keys.size:
        return np.full(targets.shape, missing, dtype=np.intp)
    positions = np.searchsorted(keys, targets)
    found = (positions < keys.size) & (keys[np.minimum(positions, keys.size - 1)] == targets)
    return np.where(found, positions + offset, missing)


def _restore_chunks(data: xr.DataArray, timedim: str, chunks: dict) -> xr.DataArray:
    """Split time into caller blocks before joining the spatial tiles."""
    return data.chunk({timedim: chunks[timedim]}).chunk({dim: size for dim, size in chunks.items() if dim != timedim})


def rolling_climatology(
    da: xr.DataArray,
    window_years: int = 15,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    cycle: Optional[SeasonalCycle] = None,
) -> xr.DataArray:
    """
    Compute a rolling climatology with flox.

    Each year's climatology uses the previous `window_years` years of data and is reassembled to
    match the input's structure. Years without enough previous data are NaN.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    window_years : int, default=15
        Number of years to include in each climatology window
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    cycle : SeasonalCycle, optional
        Within-year axis the climatology is resolved on. Inferred from the time
        coordinate's cadence when omitted (``dayofyear`` for daily data).

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
    >>> climatology = marEx.anomaly.rolling_climatology(sst, window_years=15)
    >>> print(climatology.shape)
    (7305, 180, 360)  # Same as input
    >>>
    >>> # First 15 years will be NaN (insufficient history)
    >>> print(f"NaN values in first year: {climatology.isel(time=slice(0, 365)).isnull().all().compute()}")
    True

    Shorter window for datasets with limited time span:

    >>> # For datasets with only 10 years, use shorter window
    >>> short_climatology = marEx.anomaly.rolling_climatology(
    ...     sst, window_years=5
    ... )
    >>> # First 5 years will be NaN instead of 15

    Processing unstructured data:

    >>> # ICON ocean model data
    >>> icon_sst = xr.open_dataset('icon_sst.nc', chunks={}).to.chunk({'time': 25})
    >>> icon_climatology = marEx.anomaly.rolling_climatology(
    ...     icon_sst,
    ...     dimensions={"time": "time", "x": "ncells"},
    ...     coordinates={"time": "time", "x": "lon", "y": "lat"}
    ... )
    >>> print(icon_climatology.dims)
    Frozen({'time': 7305, 'ncells': 83886})

    Comparing with fixed climatology:

    >>> # Fixed climatology (traditional approach)
    >>> fixed_clim = sst.groupby(sst.time.dt.dayofyear).mean()
    >>>
    >>> # Rolling climatology (adaptive approach)
    >>> rolling_clim = marEx.anomaly.rolling_climatology(sst).assign_coords(time=sst.time)
    >>>
    >>> # Rolling climatology adapts to climate change
    >>> clim_2000 = rolling_clim.sel(time='2000').mean()
    >>> clim_2020 = rolling_clim.sel(time='2020').mean()
    >>> print(f"Climate change signal: {(clim_2020 - clim_2000).compute():.3f} °C")

    Memory considerations for large datasets:

    >>> # Ensure appropriate chunking for memory efficiency
    >>> large_sst = sst.chunk({'time': 30, 'lat': 45, 'lon': 90})
    >>> large_climatology = marEx.anomaly.rolling_climatology(large_sst)
    >>> # Output maintains input chunking structure
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)
    da = unmask_dataarray(da)
    timedim = dimensions["time"]
    time_coord = coordinates["time"]
    cycle = resolve_cycle(da, time_coord, cycle)
    original_chunk_dict = dict(zip(da.dims, da.chunks))

    # Add temporal coordinates
    years = da[time_coord].dt.year
    doys = cycle.index_of(da[time_coord])
    da = da.assign_coords({"year": years, cycle.index_name: doys})

    # Get temporal bounds
    years, doys = persist(years, doys)
    year_vals = years.values
    doy_vals = doys.values
    unique_years = np.unique(year_vals)
    cycle_index = pd.Index(cycle.labels)
    n_cycle = len(cycle_index)
    doy_pos = cycle_index.get_indexer(doy_vals)

    # A sample from year Y contributes to observed targets in (Y, Y + window_years]
    # after the initial warm-up. Encode each target year as the start of its cycle.
    eligible_years = unique_years[unique_years >= int(unique_years.min()) + window_years]
    target_offsets = {
        year: eligible_years[(eligible_years > year) & (eligible_years <= int(year) + window_years)].astype(np.int64) * n_cycle
        for year in unique_years
    }
    time_indices, keys = _group_contributions(np.arange(year_vals.size), doy_pos, year_vals, target_offsets)

    pool_time = pool_keys = np.empty(0, dtype=np.int64)
    pool = last_day_pool(da[time_coord], cycle)
    if pool is not None:
        positions, slots, last_labels = pool
        pool_time, pool_keys = _group_contributions(positions, cycle_index.get_indexer(slots), year_vals, target_offsets)

    if not time_indices.size and not pool_time.size:
        # No eligible history. Keep the static coordinates and the usual output order and dtype.
        template = da.drop_vars([name for name, coord in da.coords.items() if timedim in coord.dims])
        result = xr.full_like(template.transpose(..., timedim), np.nan, dtype=np.float32).chunk(original_chunk_dict)
        clear_store_encoding(result)
        return result

    # Expected groups with no observations make flox build a Dask setitem graph whose
    # client cost is quadratic in the block count. Keep only occupied pairs and one
    # shared NaN row for all missing pairs.
    occupied = np.unique(keys)
    pool_occupied = np.unique(pool_keys)
    missing = occupied.size + pool_occupied.size

    # Split spatial chunks before combining the time axis so Dask never forms an untiled
    # whole-time block. Budget the input record, both expansions and the combined table.
    spatial_tile = tile_spatial_chunks(
        da,
        spatial_dims(da, dimensions),
        input_elements_per_cell=max(da.sizes[timedim], time_indices.size, pool_time.size),
        output_elements_per_cell=missing + 1,
    )
    da = da.chunk(spatial_tile).chunk({timedim: -1})

    long_timedim = f"{timedim}_contrib"
    targets = year_vals.astype(np.int64) * n_cycle + np.maximum(doy_pos, 0)
    indices = np.full(targets.shape, missing, dtype=np.intp)
    tables = []
    if time_indices.size:
        long_form_data = da.isel({timedim: time_indices}).rename({timedim: long_timedim})
        # Name the main reduction method:
        # flox's automatic choice searches for cohorts with a label-by-label containment matrix
        # that grows with the square of the (target year, cycle) group count, client-side before
        # any task runs (a 60-year daily record peaked near 9 GB against 2 GB with map-reduce).
        # Whether flox then picks cohorts depends on how many blocks the long-form axis has, not
        # on the record length, so naming the method also makes the result independent of that
        # heuristic.
        tables.append(_reduce_occupied(long_form_data, long_timedim, keys, occupied, "map-reduce"))
        indices = _slot_positions(occupied, targets, 0, missing)

    # Replace day-366 targets with the separate pool, including missing pool groups.
    if pool_time.size:
        pool_data = da.isel({timedim: pool_time}).rename({timedim: long_timedim})
        last_day = _reduce_occupied(pool_data, long_timedim, pool_keys, pool_occupied, None)
        tables.append(last_day.assign_coords(target_year=occupied.size + np.arange(pool_occupied.size)))
        use_pool = (doy_pos >= 0) & (doy_vals >= last_labels[0])
        indices[use_pool] = _slot_positions(pool_occupied, targets[use_pool], occupied.size, missing)
    indices[doy_pos < 0] = missing
    nan_row = xr.full_like(tables[0].isel(target_year=slice(0, 1)), np.nan).assign_coords(target_year=[missing])
    climatologies = xr.concat([*tables, nan_row], dim="target_year").chunk({"target_year": -1}).transpose(..., "target_year")

    # Share one Dask time index across all tiles. An embedded index would grow the graph
    # as tiles x time steps (about 1 GB on a century of daily data over 7 x 161 x 321 cells).
    tile_cells = int(np.prod([max(chunks) for chunks in climatologies.chunks[:-1]]))
    flat_index = dask.array.from_array(
        indices,
        chunks=max(1, core_dimensions.TASK_ELEMENTS // tile_cells),
    )
    spatial_index = tuple(range(climatologies.ndim - 1))
    gather_axis = (climatologies.ndim,)
    result = xr.DataArray(
        dask.array.blockwise(
            _gather_slots,
            spatial_index + gather_axis,
            climatologies.data,
            tuple(range(climatologies.ndim)),
            flat_index,
            gather_axis,
            dtype=climatologies.dtype,
            concatenate=True,
        ),
        dims=climatologies.dims[:-1] + (timedim,),
        coords={name: coord for name, coord in climatologies.coords.items() if "target_year" not in coord.dims},
        attrs=climatologies.attrs,
    )
    result.name = climatologies.name  # an unnamed input stays unnamed rather than taking the dask array's name

    result = _restore_chunks(result, timedim, original_chunk_dict)
    clear_store_encoding(result)  # the caller may write it; a format-2 input's codecs would not survive zarr 3
    return result


_FLOAT32_EXACT_COUNT = 2**24


def _min_count(steps: int, max_missing: float) -> int:
    """Fewest valid samples a window of ``steps`` needs when ``max_missing`` of it may be missing."""
    return max(1, steps - int(np.floor(max_missing * steps + 1e-9)))


def _check_smooth_max_missing(smooth_max_missing: float) -> float:
    """Validate ``smooth_max_missing``: a fraction in [0, 1)."""
    if (
        isinstance(smooth_max_missing, (bool, np.bool_))
        or not isinstance(smooth_max_missing, (int, float, np.integer, np.floating))
        or not 0 <= smooth_max_missing < 1
    ):
        raise ConfigurationError(
            f"smooth_max_missing must be a fraction in [0, 1), got {smooth_max_missing!r}",
            details="It is the fraction of each smoothing window that may be missing before the window is invalid.",
            suggestions=[
                "Use 0.1 (the default) to allow a tenth of each window to be missing",
                "Use 0 to require complete windows",
            ],
        )
    return float(smooth_max_missing)


def _subdaily_grid_positions(time_coord: xr.DataArray, stride: int) -> Optional[np.ndarray]:
    """Index of each timestep on the regular grid of ``stride`` steps a day that starts at the first one.

    ``None`` when the axis is that grid already. Raises when a timestep is off the grid, so that a
    window taking every ``stride``-th step really holds one time of day.
    """
    values = np.asarray(time_coord.values)
    if values.size < 2:
        return None
    step_seconds, off_grid = divmod(86400, stride)
    problem = None
    if off_grid:
        problem = f"{stride} steps a day is not a whole number of seconds apart"
    elif np.issubdtype(values.dtype, np.datetime64):
        if np.isnat(values).any():
            problem = "the time axis holds NaT"
        else:
            microseconds = values.astype("datetime64[us]")
            offsets = (microseconds - microseconds[0]).astype(np.int64)
    else:
        try:
            deltas = [value - values[0] for value in values]
            offsets = np.array([(d.days * 86400 + d.seconds) * 10**6 + d.microseconds for d in deltas], dtype=np.int64)
        except (AttributeError, TypeError):
            problem = "the time axis is not datetime-like"
    if problem is None:
        positions, remainder = np.divmod(offsets, step_seconds * 10**6)
        if remainder.any():
            problem = f"a timestep is not a whole number of {step_seconds} s steps after the first"
        elif (np.diff(positions) <= 0).any():
            problem = "the time axis is not strictly increasing"
    if problem is not None:
        # Labels decoded from float offsets can sit a few nanoseconds off the grid; snapping them
        # to the step is then the whole fix.
        snap = []
        if problem.startswith("a timestep is not a whole number"):
            rounding = f"da['time'] = da['time'].dt.round('{step_seconds}s')"
            snap = [f"If the labels are only slightly off the grid, snap them first: {rounding}"]
        raise ConfigurationError(
            "shifting_baseline cannot smooth this sub-daily time axis by time of day",
            details=f"Smoothing takes every {stride}th step as the same time of day, but {problem}.",
            suggestions=snap
            + [
                "Regrid the data onto a regular sub-daily axis (missing timesteps may be left out)",
                "Use method='fixed_baseline', which groups by time of day directly",
            ],
        )
    return None if positions[-1] == values.size - 1 else positions


def _blocked_window_sum(
    block: np.ndarray, time_axis: int, span: int, stride: int, n_out: int, block_elements: int, **kwargs
) -> np.ndarray:
    """Trailing window sums of a time-padded block, a few rows at a time.

    ``block`` holds ``span - 1`` rows of history before its ``n_out`` output rows (and possibly
    rows after them, which are not used). Output row ``i`` sums every ``stride``-th sample of
    block rows ``i .. i + span - 1``. Each row block's windows are copied to a C-ordered
    (rows, ..., span) array before the stepped sum, the layout xarray's ``construct`` gives the
    whole tile, so the sums are rounded exactly as they would be there; only one row block's
    copy exists at a time instead of ``span`` copies of the tile.
    """
    windows = np.lib.stride_tricks.sliding_window_view(
        np.take(block, np.arange(n_out + span - 1), axis=time_axis), span, axis=time_axis
    )
    per_row = max(1, windows.size // max(1, n_out))
    rows = max(1, block_elements // per_row)
    out_shape = windows.shape[:-1]
    out = None
    index = [slice(None)] * windows.ndim
    for start in range(0, n_out, rows):
        index[time_axis] = slice(start, start + rows)
        part = np.ascontiguousarray(windows[tuple(index)])[..., ::stride].sum(axis=-1, **kwargs)
        if out is None:
            out = np.empty(out_shape, dtype=part.dtype)
        out[tuple(index[:-1])] = part
    if out is None:  # no output rows
        out = np.ascontiguousarray(windows)[..., ::stride].sum(axis=-1, **kwargs)
    return out


def _strided_window_sum(
    obj: xr.DataArray, timedim: str, steps: int, stride: int, fill_value, block_elements: int = 2**23, **kwargs
) -> xr.DataArray:
    """Trailing window sum over ``steps`` samples ``stride`` apart, padded with ``fill_value``.

    The window is the trailing ``(steps - 1) * stride + 1`` steps with every ``stride``-th sample
    kept, the windows ``obj.rolling(...).construct(...)`` would build. Built through ``construct``
    on a dask array, each task holds its tile's windows densely, ``span`` copies of the tile. Here
    each time chunk gets ``span - 1`` steps of history from its neighbours (``fill_value`` before
    the record) and builds its windows a few rows at a time (about ``block_elements`` window
    samples per block).
    """
    span = (steps - 1) * stride + 1
    if obj.sizes[timedim] <= span - 1 or not isinstance(obj.data, dask.array.Array):
        # A record no longer than the history window cannot take its halo from neighbours, and the
        # dense windows of so short a series cost nothing; an in-memory array has no tiles to protect
        # and sums its strided view in a different order. Both build the windows as xarray does.
        window = f"_{timedim}_window"
        while window in obj.dims or window in obj.coords:
            window = f"_{window}"
        windows = obj.rolling({timedim: span}).construct(
            window, fill_value=fill_value, sliding_window_view_kwargs={"automatic_rechunk": False}
        )
        return windows.isel({window: slice(None, None, stride)}).sum(window, skipna=False, **kwargs)
    data = obj.data
    t = obj.get_axis_num(timedim)
    original_chunks = data.chunks
    # Every time chunk must hold a whole halo; merge neighbours locally, restore the layout below
    time_chunks = dask.array.overlap.ensure_minimum_chunksize(span - 1, data.chunks[t])
    if time_chunks != data.chunks[t]:
        data = data.rechunk({t: time_chunks})
    depth = {i: (span - 1 if i == t else 0) for i in range(obj.ndim)}
    boundary = {i: (fill_value if i == t else "none") for i in range(obj.ndim)}
    padded = dask.array.overlap.overlap(data, depth=depth, boundary=boundary)
    out_dtype = np.zeros(1, dtype=obj.dtype).sum(**kwargs).dtype
    out_chunks = data.chunks
    n_rows = out_chunks[t]

    def func(block, block_info=None):
        n_out = n_rows[block_info[0]["chunk-location"][t]]
        return _blocked_window_sum(block, t, span, stride, n_out, block_elements, **kwargs)

    summed = padded.map_blocks(func, chunks=out_chunks, dtype=out_dtype)
    if summed.chunks != original_chunks:
        summed = summed.rechunk(original_chunks)
    return xr.DataArray(summed, dims=obj.dims, coords=obj.coords, name=obj.name)


def _rolling_mean_keep_tiles(da: xr.DataArray, timedim: str, steps: int, stride: int = 1, max_missing: float = 0.0) -> xr.DataArray:
    """Centred rolling mean over ``steps`` samples ``stride`` apart that keeps ``da``'s spatial chunks.

    xarray's ``rolling().mean()`` counts the valid samples per window through a ``construct``
    that lets dask rechunk automatically, which splits every spatial tile about ``steps`` ways.
    The split carries into the climatology's flox reduction and multiplies its task count. This
    is the same arithmetic as xarray's numpy path (zero-filled window sum, valid count,
    ``min_periods`` equal to the window, masked sum divided by count in its dtype), with both
    windows built in place. xarray already builds the float sum's windows in place, so only the
    layout of the integer count changes, and an integer count is exact: the values do not move.

    With ``stride`` > 1 the window takes every ``stride``-th step, e.g. the same time of day on
    consecutive days, centred as xarray centres ``steps`` (an even window reaches one sample
    further back than forward): a trailing window of the spanned steps, every ``stride``-th
    sample kept, shifted back by its forward reach.

    ``max_missing`` is the fraction of a window that may be missing (NaN) before the window is
    invalid; the mean is then over the valid samples. It applies to gaps inside the record only:
    a window that reaches past either end of the series stays NaN, as it is at 0.
    """
    window = f"_{timedim}_window"
    while window in da.dims or window in da.coords:
        window = f"_{window}"
    in_place = {"automatic_rechunk": False}

    def window_sum(obj, fill_value, **kwargs):
        if stride == 1:
            windows = obj.rolling({timedim: steps}, center=True).construct(
                window, fill_value=fill_value, sliding_window_view_kwargs=in_place
            )
        else:
            return _strided_window_sum(obj, timedim, steps, stride, fill_value, **kwargs)
        return windows.sum(window, skipna=False, **kwargs)

    total = window_sum(da.fillna(0), 0)
    # Counted in float32 rather than int64 then float64: a quarter of the bytes on the tile, and
    # the same numbers once cast to the mean's dtype. float32 is exact only up to 2**24, so a
    # longer window counts in float64.
    counts = window_sum(da.notnull(), False, dtype=np.float32 if steps <= _FLOAT32_EXACT_COUNT else np.float64)
    min_count = _min_count(steps, max_missing)
    enough = counts >= min_count
    mean = total.where(enough)
    mean = mean / counts.where(enough).astype(mean.dtype)
    if stride > 1:
        reach = ((steps - 1) // 2) * stride
        # A window reaching further forward than the record is long fits nowhere.
        mean = mean.shift({timedim: -reach}) if reach < da.sizes[timedim] else mean.where(False)
    if min_count < steps:
        # The padding beyond the record counts as missing in `counts`; keep those windows NaN.
        nt = da.sizes[timedim]
        back, fwd = (steps // 2) * stride, ((steps - 1) // 2) * stride
        interior = np.zeros(nt, dtype=bool)
        interior[back : max(back, nt - fwd)] = True
        mean = mean.where(xr.DataArray(interior, dims=timedim))
    if xr.get_options()["keep_attrs"] is not False:
        mean.attrs = dict(da.attrs)
    return mean


def smoothed_rolling_climatology(
    da: xr.DataArray,
    window_years: int = 15,
    smooth_days: int = 21,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    cycle: Optional[SeasonalCycle] = None,
    smooth_max_missing: float = 0.1,
) -> xr.DataArray:
    """
    Compute a smoothed rolling climatology.

    Each year's climatology uses the previous `window_years` years of data and is reassembled to
    match the input's structure. Years without enough previous data are NaN.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    window_years : int, default=15
        Number of years to include in each climatology window
    smooth_days : int, default=21
        Number of days for temporal smoothing window
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    cycle : SeasonalCycle, optional
        Within-year axis the climatology is resolved on. Inferred from the time
        coordinate's cadence when omitted.
    smooth_max_missing : float, default=0.1
        Fraction of each smoothing window that may be missing (NaN, or a timestep absent from a
        sub-daily axis) before the smoothed value is NaN. Otherwise it is the mean of the valid
        samples. Windows that reach past either end of the record are NaN, and a missing step
        is excluded from the climatology itself. ``0`` requires complete windows.

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
    >>> smooth_clim = marEx.anomaly.smoothed_rolling_climatology(
    ...     sst,
    ...     window_years=15,
    ...     smooth_days=21
    ... )
    >>> print(smooth_clim.shape)
    (7305, 180, 360)

    Comparing different smoothing windows:

    >>> # Short smoothing - more day-to-day variability
    >>> clim_short = marEx.anomaly.smoothed_rolling_climatology(
    ...     sst, smooth_days=7
    ... )
    >>>
    >>> # Long smoothing - smoother seasonal cycle
    >>> clim_long = marEx.anomaly.smoothed_rolling_climatology(
    ...     sst, smooth_days=61
    ... )
    >>>
    >>> # Compare variability
    >>> var_short = clim_short.std(dim='time').mean().compute()
    >>> var_long = clim_long.std(dim='time').mean().compute()
    >>> print(f"Variability: short={var_short:.3f}, long={var_long:.3f}")

    Climatology for anomaly computation:

    >>> # Compute smoothed climatology then anomalies
    >>> climatology = marEx.anomaly.smoothed_rolling_climatology(sst)
    >>> anomalies = sst - climatology
    >>>
    >>> # Check that anomalies have reasonable properties
    >>> print(f"Anomaly mean: {anomalies.mean().compute():.6f}")
    >>> print(f"Anomaly std: {anomalies.std().compute():.3f}")

    Unstructured data processing:

    >>> # ICON ocean data
    >>> icon_sst = xr.open_dataset('icon_sst.nc', chunks={}).to.chunk({'time': 25})
    >>> icon_smooth_clim = marEx.anomaly.smoothed_rolling_climatology(
    ...     icon_sst,
    ...     dimensions={"time": "time", "x": "ncells"},
    ...     coordinates={"time": "time", "x": "lon", "y": "lat"},
    ...     window_years=10,
    ...     smooth_days=31
    ... )

    Effect of smoothing on seasonal cycle:

    >>> # Raw rolling climatology (no temporal smoothing)
    >>> raw_clim = marEx.anomaly.rolling_climatology(sst, window_years=15)
    >>>
    >>> # Smoothed rolling climatology
    >>> smooth_clim = marEx.anomaly.smoothed_rolling_climatology(
    ...     sst, window_years=15, smooth_days=21
    ... )
    >>>
    >>> # Compare seasonal cycle smoothness
    >>> # Extract annual cycle for a point
    >>> point_raw = raw_clim.assign_coords(time=sst.time).isel(lat=90, lon=180).sel(time='2010')
    >>> point_smooth = smooth_clim.assign_coords(time=sst.time).isel(lat=90, lon=180).sel(time='2010')
    >>>
    >>> print(f"Raw climatology range: {(point_raw.max() - point_raw.min()).compute():.3f}")
    >>> print(f"Smooth climatology range: {(point_smooth.max() - point_smooth.min()).compute():.3f}")

    Performance considerations:

    >>> # Efficient implementation smooths raw data first, then computes climatology
    >>> # This is more memory-efficient than smoothing the climatology
    >>> large_sst = sst.chunk({'time': 25, 'lat': 45, 'lon': 90})
    >>> efficient_clim = marEx.anomaly.smoothed_rolling_climatology(large_sst)
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)
    da = unmask_dataarray(da)
    timedim = dimensions["time"]
    cycle = resolve_cycle(da, coordinates["time"], cycle)
    smooth_max_missing = _check_smooth_max_missing(smooth_max_missing)

    # `smooth_days` is a physical duration, so convert it to whole timesteps. On daily
    # data this is the identity (21 days -> 21 steps) and every existing result is
    # unchanged. It is deliberately NOT forced odd: an even smoothing window is legal
    # here and forcing it would move existing daily output. On a monthly axis it
    # collapses to 1 step, which `steps_for_days` warns about -- the smoothing then
    # degenerates to `rolling_climatology`, which is correct but worth saying out loud.
    #
    # On a sub-daily axis the window runs across DAYS at a fixed time of day, as the fixed
    # baseline's smoothing does: a mean over consecutive steps would average the diurnal cycle
    # out of the climatology and leave it in the anomaly.
    stride = cycle.steps_per_day if cycle.is_subdaily else 1
    smooth_steps = replace(cycle, step_days=cycle.step_days * stride).steps_for_days(smooth_days, name="smooth_days")
    if smooth_steps == 1:
        # Warning, not info. `steps_for_days` only warns when the realised duration is
        # more than half a step from the request, and a 21-day ask on a 31-day axis is
        # inside that -- 31 days genuinely is the closest representable window. But the
        # user asked for smoothing and is getting none, which is a different method from
        # the one they requested, so it is said out loud regardless.
        logger.warning(
            "smooth_days=%s resolves to a single timestep on this %g-day axis: no smoothing "
            "is applied and smoothed_rolling_climatology reduces to rolling_climatology.",
            smooth_days,
            cycle.step_days,
        )

    # Smooth on a canonical layout: bottleneck's move_mean restarts its running sum at every dask
    # block boundary, so a chunked time axis shifts the smoothed field by a few float32 ULP, enough
    # to move a 0.01 threshold bin. With time whole the only boundary left is the one xarray's
    # centred padding adds near the end of the series, which depends on its length alone, so the result
    # does not depend on the caller's chunking, and a time chunk shorter than the window is no longer an error.
    caller_chunks = dict(zip(da.dims, da.chunks))
    # The time-of-day window is positional, so a sub-daily axis with missing timesteps is smoothed on
    # its regular grid, the missing steps as NaN, and read back at the original timesteps.
    positions = _subdaily_grid_positions(da[coordinates["time"]], stride) if stride > 1 else None
    grid_length = da.sizes[timedim] if positions is None else int(positions[-1]) + 1
    # The strided smoother pads both sides of each tile when it can use overlap.
    halo = (smooth_steps - 1) * stride
    padded_length = grid_length + 2 * halo if stride > 1 and grid_length > halo else grid_length
    spatial_tile = tile_spatial_chunks(da, spatial_dims(da, dimensions), input_elements_per_cell=padded_length)
    # Split space before either assembling the regular grid or merging time.
    grid = da.chunk(spatial_tile)
    if positions is not None:
        source = np.zeros(grid_length, dtype=np.int64)
        present = np.zeros(grid_length, dtype=bool)
        source[positions] = np.arange(positions.size)
        present[positions] = True
        grid = grid.isel({timedim: source}).where(xr.DataArray(present, dims=timedim))
    grid = grid.chunk({timedim: -1})

    # N.B.: It is more efficient (chunking-wise) to smooth the raw data rather than the climatology
    with rolling_numerics():
        da_smoothed = _rolling_mean_keep_tiles(grid, timedim, smooth_steps, stride, smooth_max_missing).astype(np.float32)
    if _min_count(smooth_steps, smooth_max_missing) < smooth_steps:
        # A window may now be valid around a missing sample. The climatology still takes only observed
        # steps, so a missing step adds no value of its own, whether it is NaN or absent from the axis.
        da_smoothed = da_smoothed.where(grid.notnull())
    if positions is not None:
        da_smoothed = da_smoothed.isel({timedim: positions})

    clim = rolling_climatology(da_smoothed, window_years, dimensions, coordinates, cycle)

    clim = _restore_chunks(clim, timedim, caller_chunks)
    clear_store_encoding(clim)
    return clim
