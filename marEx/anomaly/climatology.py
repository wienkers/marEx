"""
Rolling-climatology computation for the shifting-baseline anomaly method.

Provides :func:`rolling_climatology` and :func:`smoothed_rolling_climatology`,
both public entry points re-exported at the marEx top level. These use flox
cohorts to efficiently compute multi-year rolling day-of-year climatologies.
"""

from dataclasses import replace
from typing import Dict, Optional

import flox.xarray
import numpy as np
import pandas as pd
import xarray as xr
from dask import persist

from ..core.dimensions import canonical_time_chunks, spatial_dims, tile_spatial_chunks
from ..core.encoding import clear_store_encoding
from ..core.numerics import rolling_numerics
from ..core.time_axis import SeasonalCycle, resolve_cycle
from ..core.validation import _infer_dims_coords
from ..exceptions import ConfigurationError
from ..logging_config import get_logger

# Get module logger
logger = get_logger(__name__)


def rolling_climatology(
    da: xr.DataArray,
    window_years: int = 15,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    cycle: Optional[SeasonalCycle] = None,
) -> xr.DataArray:
    """
    Compute rolling climatology efficiently using flox cohorts.
    Uses the previous `window_years` years of data and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.

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
    >>> rolling_clim = marEx.anomaly.rolling_climatology(sst)
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
    timedim = dimensions["time"]
    time_coord = coordinates["time"]
    cycle = resolve_cycle(da, time_coord, cycle)
    cycle_dim = cycle.index_name
    original_chunk_dict = dict(zip(da.dims, da.chunks))

    # Add temporal coordinates
    years = da[time_coord].dt.year
    doys = cycle.index_of(da[time_coord])
    da = da.assign_coords({"year": years, cycle_dim: doys})

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
    contributing_cycle_indices = []

    for t_idx, (year_val, doy_val) in enumerate(zip(year_vals, doy_vals)):
        # Convert numpy scalars to Python ints to avoid dtype issues
        year_val = int(year_val)
        doy_val = int(doy_val)

        # Find target years this time point contributes to
        # A time point from year Y contributes to target years where:
        # target_year - window_years <= Y < target_year
        # Which means: Y < target_year <= Y + window_years
        candidate_targets = unique_years[(unique_years > year_val) & (unique_years <= year_val + window_years)]

        # Only include target years that have sufficient history
        valid_targets = candidate_targets[candidate_targets >= min_year + window_years]

        # Add entries for each valid target year
        n_targets = len(valid_targets)
        contributing_time_indices.extend([t_idx] * n_targets)
        contributing_target_years.extend(valid_targets.tolist())
        contributing_cycle_indices.extend([doy_val] * n_targets)

    # Convert to numpy arrays with explicit dtypes
    time_indices = np.array(contributing_time_indices, dtype=np.int32)
    target_year_groups = np.array(contributing_target_years, dtype=np.int32)
    cycle_groups = np.array(contributing_cycle_indices, dtype=np.int32)

    # Bound the long-form expansion before building it.
    #
    # The `isel` below materialises roughly `window_years` times the input along time,
    # and the flox reduction after it writes an `(n_years, cycle)` block per spatial
    # cell, which `.chunk({"dayofyear": -1})` then forces whole. Neither side had a
    # budget: on a field left spatially whole one task is the entire array (174 GB on
    # the ICON mesh, where it thrashed rather than failing), and an extra dimension
    # such as depth multiplies that directly.
    #
    # Tiling the spatial dims -- every one of them, so depth is tiled exactly like
    # latitude -- bounds both sides. It is a pure rechunk of SPATIAL dims only: the
    # reduction is independent per cell and each cell's reduced axis stays whole inside
    # its tile, so flox's ordering along that axis is unchanged and the result is
    # bit-identical (verified against both goldens and pinned by
    # tests/test_climatology_tiling.py).
    #
    # The time axis is held whole in the same rechunk. Flox's grouped mean and the rolling mean
    # before it accumulate block by block, so a time chunk boundary moves the result,
    # and flox's task count scales with tiles x time chunks. `original_chunk_dict` was
    # captured before this point, so the restore at the end returns the caller's own layout.
    spatial_tile = tile_spatial_chunks(
        da,
        spatial_dims(da, dimensions),
        input_elements_per_cell=len(time_indices),
        output_elements_per_cell=len(unique_years) * cycle.length,
    )
    da = da.chunk({timedim: -1, **spatial_tile})

    # Create long-form dataset by selecting the contributing time points
    long_form_data = da.isel({timedim: time_indices})

    # Create a new time dimension for the long-form data
    long_timedim = f"{timedim}_contrib"
    long_form_data = long_form_data.rename({timedim: long_timedim})

    # Convert grouping arrays to DataArrays with the correct dimension
    target_year_da = xr.DataArray(target_year_groups, dims=[long_timedim], name="target_year")
    cycle_da = xr.DataArray(cycle_groups, dims=[long_timedim], name=cycle_dim)

    # flox fills an expected group that received no data by assigning NaN through dask's
    # `__setitem__`, whose graph build is quadratic in the number of blocks: hours of client
    # time on the ICON mesh before any task runs. Here the only empty groups are normally
    # whole target years (the first `window_years` have no history), so reduce over the
    # years that do have data and pad the rest back with a reindex, which is one indexing
    # task per block. Any other gap falls back to letting flox fill it.
    data_years = np.unique(target_year_groups)
    n_pairs = len(set(zip(target_year_groups.tolist(), cycle_groups.tolist())))
    complete = len(data_years) > 0 and n_pairs == len(data_years) * len(cycle.labels)
    reduce_years = data_years if complete else unique_years

    # Use flox with both grouping variables to compute climatologies
    climatologies = flox.xarray.xarray_reduce(
        long_form_data,
        target_year_da,
        cycle_da,
        dim=long_timedim,
        func="nanmean",
        expected_groups=(reduce_years, cycle.labels),
        isbin=(False, False),
        dtype=np.float32,
        fill_value=np.nan,
    )
    if len(reduce_years) < len(unique_years):
        climatologies = climatologies.reindex(target_year=unique_years, fill_value=np.nan)
    climatologies = climatologies.chunk({cycle_dim: -1})

    # Create index arrays for final mapping
    year_to_idx = pd.Series(range(len(unique_years)), index=unique_years)
    year_indices = year_to_idx[year_vals].values

    # Select appropriate climatology for each time point
    result = climatologies.isel(
        {
            "target_year": xr.DataArray(year_indices, dims=[timedim]),
            cycle_dim: xr.DataArray(doy_vals - 1, dims=[timedim]),
        }
    )

    # Clean up dimensions and coordinates
    result = result.drop_vars(["target_year", cycle_dim])

    result = result.chunk(original_chunk_dict)
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
        raise ConfigurationError(
            "shifting_baseline cannot smooth this sub-daily time axis by time of day",
            details=f"Smoothing takes every {stride}th step as the same time of day, but {problem}.",
            suggestions=[
                "Regrid the data onto a regular sub-daily axis (missing timesteps may be left out)",
                "Use method='fixed_baseline', which groups by time of day directly",
            ],
        )
    return None if positions[-1] == values.size - 1 else positions


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
            span = (steps - 1) * stride + 1
            windows = obj.rolling({timedim: span}).construct(window, fill_value=fill_value, sliding_window_view_kwargs=in_place)
            windows = windows.isel({window: slice(None, None, stride)})
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
    Compute a smoothed rolling climatology using the previous `window_years` years of data
    and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.

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
        sub-daily axis) before the smoothed value is NaN; otherwise it is the mean of the valid
        samples. Windows that reach past either end of the record stay NaN, and a missing step
        adds nothing to the climatology itself. ``0`` requires complete windows.

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
    >>> point_raw = raw_clim.isel(lat=90, lon=180).sel(time='2010')
    >>> point_smooth = smooth_clim.isel(lat=90, lon=180).sel(time='2010')
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
    da = da.chunk(canonical_time_chunks(da, dimensions))

    # The time-of-day window is positional, so a sub-daily axis with missing timesteps is smoothed on
    # its regular grid, the missing steps as NaN, and read back at the original timesteps.
    positions = _subdaily_grid_positions(da[coordinates["time"]], stride) if stride > 1 else None
    if positions is not None:
        source = np.zeros(positions[-1] + 1, dtype=np.int64)
        present = np.zeros(positions[-1] + 1, dtype=bool)
        source[positions] = np.arange(positions.size)
        present[positions] = True
        grid = da.isel({timedim: source}).where(xr.DataArray(present, dims=timedim))
        grid = grid.chunk(canonical_time_chunks(grid, dimensions))
    else:
        grid = da

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

    clim = clim.chunk(caller_chunks)
    clear_store_encoding(clim)
    return clim
