"""Structure and values of the rolling and smoothed climatologies.

``rolling_climatology`` builds a long-form array about ``window_years`` times the input along
time and flox-reduces it to ``(target_year, cycle, *spatial)``; ``smoothed_rolling_climatology``
smooths the field on a canonical time-whole layout first. A value-only test passes while the
graph is broken (a full suite was green with an all-to-all rechunk live), so these tests pin
the structure directly: the spatial tile the smoothing keeps, the size of the graph, the
per-task working set and the time chunking the climatology receives, alongside bit-identity
across tilings and caller layouts and the day-366 pooling.
"""

import itertools
import pickle
import warnings

import dask
import dask.array as dsa
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.array.core import PerformanceWarning
from xarray.core.options import OPTIONS

import marEx
import marEx.core.dimensions as D
from marEx.anomaly import climatology
from marEx.anomaly.climatology import last_day_pool, rolling_climatology, smoothed_rolling_climatology
from marEx.core.dimensions import canonical_time_chunks, tile_spatial_chunks
from marEx.core.time_axis import SeasonalCycle, resolve_cycle

from ..conftest import assert_no_failures
from .conftest import GRID_DIMS, MESH_COORDS, MESH_DIMS, dec31_values, gridded_sst, trend

WINDOW_YEARS = 2


class _Stop(Exception):
    pass


def _cells(nt, nc, chunks):
    t = pd.date_range("2000-01-01", periods=nt, freq="D")
    return xr.DataArray(
        dsa.zeros((nt, nc), dtype=np.float32, chunks=chunks),
        dims=("time", "ncells"),
        coords={"time": t, "lat": ("ncells", np.zeros(nc, np.float32)), "lon": ("ncells", np.zeros(nc, np.float32))},
        name="sst",
    )


def _grid(nt, nlat, nlon, chunks):
    t = pd.date_range("2000-01-01", periods=nt, freq="D")
    return xr.DataArray(
        dsa.zeros((nt, nlat, nlon), dtype=np.float32, chunks=chunks),
        dims=("time", "lat", "lon"),
        coords={"time": t, "lat": np.linspace(-89.5, 89.5, nlat), "lon": np.linspace(0.5, 359.5, nlon)},
        name="sst",
    )


def _chunks_reaching_the_climatology(monkeypatch, da, dimensions, coordinates, smooth_days, **kwargs):
    """Chunks of the smoothed field as ``rolling_climatology`` receives it, before the caller's layout is restored."""
    seen = []

    def spy(da_smoothed, *args, **kwargs):
        seen.append(dict(zip(da_smoothed.dims, da_smoothed.chunks)))
        raise _Stop

    with monkeypatch.context() as m:
        m.setattr(climatology, "rolling_climatology", spy)
        with pytest.raises(_Stop):
            smoothed_rolling_climatology(da, smooth_days=smooth_days, dimensions=dimensions, coordinates=coordinates, **kwargs)
    assert len(seen) == 1
    return seen[0]


@pytest.mark.parametrize("grid", ["unstructured", "gridded"])
def test_smoothing_keeps_the_spatial_tile(monkeypatch, grid):
    """The smoothing hands the climatology the canonical spatial tile, unsplit, with or without a window.

    xarray's rolling ``mean`` counts valid samples through a ``construct`` that lets dask split
    every spatial tile about ``window`` ways; the split reaches the flox reduction and is undone
    by the restore to the caller's layout, so only the field as the climatology receives it shows it.
    """
    failures = []
    for window in (1, 21):
        if grid == "unstructured":
            da = _cells(1096, 50_000, (21, 25_000))
            tile = canonical_time_chunks(da, MESH_DIMS)
            if tile != {"time": -1, "ncells": 25_000}:
                failures.append(f"window {window}: canonical tile {tile}")
            chunks = _chunks_reaching_the_climatology(monkeypatch, da, MESH_DIMS, MESH_COORDS, window, window_years=2)
            if chunks["ncells"] != (25_000, 25_000):
                failures.append(f"window {window}: ncells chunks {chunks['ncells']}")
        else:
            da = _grid(1096, 60, 360, (30, 30, 360))
            expected = da.chunk(canonical_time_chunks(da, GRID_DIMS))
            chunks = _chunks_reaching_the_climatology(monkeypatch, da, GRID_DIMS, None, window, window_years=2)
            if not (chunks["lat"] == expected.chunks[1] == (30, 30)):
                failures.append(f"window {window}: lat chunks {chunks['lat']}, canonical {expected.chunks[1]}")
            if not (chunks["lon"] == expected.chunks[2] == (360,)):
                failures.append(f"window {window}: lon chunks {chunks['lon']}, canonical {expected.chunks[2]}")
    assert_no_failures(failures)


def _anomaly_tasks(nc, smooth_days):
    ds = marEx.anomaly.compute(
        _cells(5844, nc, (21, 25_000)),
        "shifting_baseline",
        window_years=5,
        smooth_days=smooth_days,
        compute_mode="lazy",
        validate=False,
        dimensions=MESH_DIMS,
        coordinates=MESH_COORDS,
        quiet=True,
    )
    return len(dict(ds["dat_anomaly"].data.__dask_graph__()))


def test_smoothing_task_overhead_does_not_scale_with_window():
    """The graph the smoothing adds is the same at every window and small next to the rest."""
    base = _anomaly_tasks(50_000, 1)
    smoothed = {window: _anomaly_tasks(50_000, window) for window in (11, 21, 61)}
    assert len(set(smoothed.values())) == 1, smoothed
    assert smoothed[21] - base < 0.15 * base, (base, smoothed)


def _working_set(tile, sizes, n_long, output_per_cell):
    """Elements one task touches: the larger of what it reads and what it writes."""
    cells = 1
    for dim, size in sizes.items():
        cells *= tile.get(dim, size)
    return cells * max(n_long, output_per_cell)


def test_working_set_invariant_in_depth_length():
    """The per-task bound does not grow with the extra dimension's length.

    ICON-scale arithmetic on lazy zeros: 8 years daily reduced over a 15-year window, on a
    14.9M-cell mesh. Invariance, not an absolute bound: a bound loose enough for depth=1 would
    pass while depth=50 allocated 50x.
    """
    n_long = 2922 * 15
    output_per_cell = 8 * 366
    budget = D.TASK_ELEMENTS
    working_sets = {}
    for ndepth in (1, 4, 25, 50):
        da = xr.DataArray(dsa.zeros((ndepth, 14_900_000)), dims=("depth", "ncells")).chunk({"depth": -1, "ncells": -1})
        tile = tile_spatial_chunks(
            da, ("depth", "ncells"), input_elements_per_cell=n_long, output_elements_per_cell=output_per_cell
        )
        working = _working_set(tile, dict(da.sizes), n_long, output_per_cell)
        assert working <= budget, f"depth={ndepth}: task touches {working} > budget {budget}"
        working_sets[ndepth] = working
    spread = max(working_sets.values()) / min(working_sets.values())
    assert spread < 2.0, f"per-task working set varies {spread:.1f}x across depth lengths: {working_sets}"
    assert min(working_sets.values()) > budget / 4, f"tiles far below budget: {working_sets}"


def _synthetic(nt=4 * 365, nlat=12, nlon=16, ndepth=None, seed=0):
    """Daily gridded series, optionally with a depth axis; unnamed."""
    rng = np.random.default_rng(seed)
    shape = (nt, nlat, nlon) if ndepth is None else (nt, ndepth, nlat, nlon)
    dims = ("time", "lat", "lon") if ndepth is None else ("time", "depth", "lat", "lon")
    values = rng.standard_normal(shape).astype(np.float32) * 2 + 285
    coords = {
        "time": xr.date_range("2001-01-01", periods=nt, freq="D"),
        "lat": np.linspace(-40, 40, nlat, dtype=np.float32),
        "lon": np.linspace(-60, 60, nlon, dtype=np.float32),
    }
    if ndepth is not None:
        coords["depth"] = np.arange(ndepth, dtype=np.float32)
    return xr.DataArray(values, dims=dims, coords=coords)


def test_tile_caps_but_never_raises_and_floors_only_window_dims():
    """The budget is a cap, never a target; the window floor widens horizontal dims, not depth."""
    fine = _synthetic(nt=10, nlat=64, nlon=64).chunk({"time": -1, "lat": 4, "lon": 4})
    tile = tile_spatial_chunks(fine, ("lat", "lon"), input_elements_per_cell=1, output_elements_per_cell=1)
    assert tile["lat"] == 4 and tile["lon"] == 4, f"cap raised the caller's chunks: {tile}"

    deep = _synthetic(nt=10, nlat=64, nlon=64, ndepth=8).chunk({"time": -1, "depth": -1, "lat": -1, "lon": -1})
    tile = tile_spatial_chunks(
        deep,
        ("lat", "lon", "depth"),
        input_elements_per_cell=10_000_000,
        output_elements_per_cell=1,
        floor_dims=("lat", "lon"),
        floor=5,
    )
    assert tile["lat"] >= 5 and tile["lon"] >= 5, tile
    assert tile["depth"] < 5, tile  # depth carries no rolling window, so it is not widened to the floor


def test_tiled_climatology_is_bit_identical_in_the_callers_layout(monkeypatch):
    """Tiling changes task granularity, never values, and the tile does not leak into the returned layout.

    A 3-D field (time, depth, lat, lon) under a budget small enough that the tiling bites; the
    unnamed input stays unnamed.
    """
    da = _synthetic(ndepth=3).chunk({"time": 30, "depth": -1, "lat": -1, "lon": -1})
    untiled = rolling_climatology(da, WINDOW_YEARS, GRID_DIMS, GRID_DIMS).compute()

    monkeypatch.setattr(D, "TASK_ELEMENTS", 200_000)
    tiled_chunks = tile_spatial_chunks(
        da,
        [d for d in da.dims if d != "time"],
        input_elements_per_cell=len(da.time) * WINDOW_YEARS,
        output_elements_per_cell=4 * 366,
    )
    assert any(tiled_chunks[d] < da.sizes[d] for d in tiled_chunks), (
        f"budget did not force a tiling; test would prove nothing: {tiled_chunks}"
    )

    tiled = rolling_climatology(da, WINDOW_YEARS, GRID_DIMS, GRID_DIMS)
    for dim in ("lat", "lon", "depth"):
        assert tiled.chunksizes[dim] == da.chunksizes[dim], f"{dim}: returned {tiled.chunksizes[dim]}, caller {da.chunksizes[dim]}"
    assert tiled.name is None, tiled.name
    np.testing.assert_array_equal(untiled.values, tiled.compute().values, err_msg="spatial tiling moved the climatology")


def test_padding_the_years_without_history_does_not_multiply_chunks(monkeypatch):
    """The first ``window_years`` have no climatology; padding them back stays one indexing task per block.

    A dask-backed ``reindex(fill_value=nan)`` OR-ed per-dimension masks of different chunking and
    raised dask's "Increasing number of chunks" PerformanceWarning once tiled (a 0.25 deg global
    grid is ~190 spatial tiles; this grid under this budget is a couple of cells per tile). The
    tiled smoothed climatology is bit-identical to the untiled one and an unnamed input stays unnamed.
    """
    # Whole calendar years (1982-1987), so every (year, day) pair the climatology expects is present
    # and the years without history are padded rather than filled by flox.
    da = _synthetic(nt=6 * 365 + 1, nlat=12, nlon=24)
    da = da.assign_coords(time=xr.date_range("1982-01-01", periods=da.sizes["time"], freq="D"))
    da = da.chunk({"time": 30, "lat": -1, "lon": -1})
    untiled = smoothed_rolling_climatology(da, 3, 21, GRID_DIMS, GRID_DIMS).compute()
    monkeypatch.setattr(D, "TASK_ELEMENTS", 5_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        result = smoothed_rolling_climatology(da, 3, 21, GRID_DIMS, GRID_DIMS)
    assert result.name is None, result.name
    values = result.values
    assert np.isnan(values).any() and not np.isnan(values).all(), "fixture has no padded years; test would prove nothing"
    np.testing.assert_array_equal(values, untiled.values)


def test_the_grouped_mean_skips_flox_cohort_search(monkeypatch):
    """The climatology reduction names flox's method.

    flox's automatic choice builds a label-by-label containment matrix that grows with the square
    of the number of (target year, day) groups; on a century of daily data that alone exhausts a
    16 GB client before any task runs. The day-366 pool is one block along the reduced axis and
    may still be resolved by flox.
    """
    import flox.xarray

    calls = []
    real_reduce = flox.xarray.xarray_reduce

    def spy(obj, *by, **kwargs):
        calls.append((len(obj.chunksizes[kwargs["dim"]]), kwargs.get("method")))
        return real_reduce(obj, *by, **kwargs)

    monkeypatch.setattr(flox.xarray, "xarray_reduce", spy)
    da = _synthetic(nt=6 * 365 + 1).chunk({"time": 30, "lat": -1, "lon": -1})
    with dask.config.set({"array.chunk-size": "256KiB"}):  # splits the long-form array along time
        rolling_climatology(da, WINDOW_YEARS, GRID_DIMS, GRID_DIMS)
    multi_block = [method for n_blocks, method in calls if n_blocks > 1]
    assert multi_block, f"no grouped reduction over more than one block along the reduced axis: {calls}"
    assert all(method == "map-reduce" for method in multi_block), f"(blocks, method) per flox call: {calls}"


def test_the_per_time_step_gather_does_not_repeat_the_time_index_per_tile(monkeypatch):
    """What each task added by a finer tiling weighs does not depend on the record length.

    When every tile carries its own copy of the time index, each task's payload grows with the
    record (about 1 GB of graph on a century of daily data over 7 x 161 x 321 cells).
    """

    def graph_size(nt, task_elements):
        monkeypatch.setattr(D, "TASK_ELEMENTS", task_elements)
        da = xr.DataArray(
            dsa.zeros((nt, 24, 32), chunks=(30, -1, -1), dtype=np.float32),
            dims=("time", "lat", "lon"),
            coords={
                "time": xr.date_range("2001-01-01", periods=nt, freq="D"),
                "lat": np.linspace(-40, 40, 24, dtype=np.float32),
                "lon": np.linspace(-60, 60, 32, dtype=np.float32),
            },
        )
        graph = dict(rolling_climatology(da, WINDOW_YEARS, GRID_DIMS, GRID_DIMS).data.__dask_graph__())
        return len(pickle.dumps(graph, protocol=5)), len(graph)

    per_task = {}
    for nt in (4 * 365 + 1, 16 * 365 + 4):
        coarse_bytes, coarse_tasks = graph_size(nt, 50_000_000)
        fine_bytes, fine_tasks = graph_size(nt, nt * WINDOW_YEARS * 48)  # ~16 tiles
        assert fine_tasks > coarse_tasks, "the finer budget did not tile the field; the test would prove nothing"
        per_task[nt] = (fine_bytes - coarse_bytes) / (fine_tasks - coarse_tasks)
    short, long = sorted(per_task)
    assert per_task[long] < 1.5 * per_task[short], f"bytes per added task grow with the record: {per_task}"


def test_the_per_time_step_gather_stays_within_the_task_budget(monkeypatch):
    """The gather back onto the record stays within the budget when the long-form input is shorter than the record.

    That happens with a one-year window, or a cycle coarser than the time step.
    """
    blocks = []
    real_blockwise = dask.array.blockwise

    def spy(func, *args, **kwargs):
        out = real_blockwise(func, *args, **kwargs)
        if getattr(func, "__name__", "") == "_gather_slots":
            blocks.append(max(int(np.prod(shape)) for shape in itertools.product(*out.chunks)))
        return out

    monkeypatch.setattr(dask.array, "blockwise", spy)
    monkeypatch.setattr(D, "TASK_ELEMENTS", 200_000)
    nt = 2 * 365
    da = xr.DataArray(
        dsa.zeros((nt, 60, 80), chunks=(30, -1, -1), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": xr.date_range("1990-01-01", periods=nt, freq="D"), "lat": np.arange(60.0), "lon": np.arange(80.0)},
    )
    rolling_climatology(da, 1, GRID_DIMS, GRID_DIMS, cycle=SeasonalCycle("month", 12, 30.4))
    assert blocks, "the spy never saw the gather; the test would prove nothing"
    assert max(blocks) <= D.TASK_ELEMENTS, f"gather block of {max(blocks)} elements exceeds the budget of {D.TASK_ELEMENTS}"


def _mesh_series(nt=1100, ncells=60, seed=0):
    """Daily series with a seasonal cycle and permanently-NaN land cells."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((nt, ncells)).astype(np.float32) * 5 + 280
    values += 3 * np.sin(2 * np.pi * np.arange(nt)[:, None] / 365.25).astype(np.float32)
    values[:, :4] = np.nan
    return xr.DataArray(
        values,
        dims=("time", "ncells"),
        coords={
            "time": xr.date_range("2007-01-01", periods=nt, freq="D"),
            "lon": ("ncells", np.linspace(-180, 180, ncells, dtype=np.float32)),
            "lat": ("ncells", np.linspace(-80, 80, ncells, dtype=np.float32)),
        },
    )


def test_smoothed_climatology_is_chunk_invariant(monkeypatch):
    """The smoothed climatology does not depend on how the input is chunked, and is returned in the caller's layout.

    Each layout is compared bit for bit against a time-whole input, including time chunks shorter
    than the window (2 against 21 days) and one that does not divide the window's padding
    (4 against 11). The smoothing runs on a canonical time-whole layout: values alone cannot show
    that (the numerics are chunk-independent), so each row also asserts the smoothed field reaches
    the climatology in exactly the time chunks a time-whole input gives it.
    """
    da = _mesh_series()
    failures = []
    for smooth_days in (21, 11):
        whole = da.chunk({"time": -1, "ncells": -1})
        ref = smoothed_rolling_climatology(whole, 2, smooth_days, MESH_DIMS, MESH_COORDS).compute()
        canonical = _chunks_reaching_the_climatology(monkeypatch, whole, MESH_DIMS, MESH_COORDS, smooth_days, window_years=2)
        layouts = (
            ({"time": 2, "ncells": -1}, {"time": 21, "ncells": -1}, {"time": 40, "ncells": 7})
            if smooth_days == 21
            else ({"time": 4, "ncells": -1},)
        )
        for layout in layouts:
            case = f"smooth_days={smooth_days} layout={layout}"
            chunked = da.chunk(layout)
            seen = _chunks_reaching_the_climatology(monkeypatch, chunked, MESH_DIMS, MESH_COORDS, smooth_days, window_years=2)
            if seen["time"] != canonical["time"]:
                failures.append(
                    f"{case}: the climatology received {len(seen['time'])} time chunks, "
                    f"not the canonical layout's {len(canonical['time'])}"
                )
            result = smoothed_rolling_climatology(chunked, 2, smooth_days, MESH_DIMS, MESH_COORDS)
            if dict(result.chunksizes) != dict(chunked.chunksizes):
                failures.append(f"{case}: returned chunks {dict(result.chunksizes)}, caller {dict(chunked.chunksizes)}")
            if not np.array_equal(result.values, ref.values, equal_nan=True):
                failures.append(f"{case}: climatology moved")
    assert_no_failures(failures)


def test_window_longer_than_series_gives_all_nan():
    """10 days under a 21-day window computes, and every value is NaN: the correct answer, not an error."""
    da = _mesh_series(nt=10, ncells=6).chunk({"time": 5, "ncells": -1})
    result = smoothed_rolling_climatology(da, 2, 21, MESH_DIMS, MESH_COORDS).compute()
    assert dict(result.sizes) == dict(da.sizes)
    assert bool(result.isnull().all()), f"{int(result.notnull().sum())} finite values"


def test_day_366_pools_every_dec31():
    """Slot 366 of the rolling climatology takes every 31 December in its window, not only the leap years'.

    Grouping by calendar day gives slot 366 one sample in five years; on a trending series that
    lone sample sits years from the target and the leap year's 31 December carries the trend.
    Day 365 keeps its own calendar grouping. A window with no leap year still has a day-366
    value, and a series with no day 366 (no-leap calendar, or no leap year) needs no pool.
    The input is a pure linear trend, so every expectation is an exact mean of known values.
    """
    failures = []
    da = trend()
    clim = rolling_climatology(da, window_years=5).transpose("time", ...).compute()
    t = da.time.to_index()
    for target in (2008, 2009):
        leap = pd.Timestamp(f"{target}-12-31").dayofyear == 366
        if leap:
            got = float(clim.values[t.get_loc(pd.Timestamp(f"{target}-12-31")), 0, 0])
            want = dec31_values(da, range(target - 5, target)).mean()
            if not np.isclose(got, want, rtol=1e-6, atol=0):
                failures.append(f"target {target}, slot 366: {got}, expected {want}")
        day365 = [f"{y}-12-30" if pd.Timestamp(f"{y}-12-31").dayofyear == 366 else f"{y}-12-31" for y in range(target - 5, target)]
        want365 = np.mean([float(da.values[t.get_loc(pd.Timestamp(d)), 0, 0]) for d in day365])
        on_365 = f"{target}-12-30" if leap else f"{target}-12-31"
        got365 = float(clim.values[t.get_loc(pd.Timestamp(on_365)), 0, 0])
        if not np.isclose(got365, want365, rtol=1e-6, atol=0):
            failures.append(f"target {target}, slot 365: {got365}, expected {want365}")

    # 2005-2007 holds no leap year; target 2008 used to get a NaN climatology on 31 December.
    short = trend("2005-01-01", "2008-12-31")
    clim = rolling_climatology(short, window_years=3).transpose("time", ...).compute()
    got = float(clim.values[short.time.to_index().get_loc(pd.Timestamp("2008-12-31")), 0, 0])
    want = dec31_values(short, range(2005, 2008)).mean()
    if not np.isclose(got, want, rtol=1e-6, atol=0):
        failures.append(f"window without a leap year, 2008-12-31: {got}, expected {want}")

    for calendar, (start, end) in (("noleap", ("2000-01-01", "2005-12-31")), ("standard", ("2001-01-01", "2003-12-31"))):
        time = xr.date_range(start, end, freq="D", calendar=calendar, use_cftime=calendar == "noleap")
        series = xr.DataArray(np.zeros(len(time)), coords={"time": time}, dims="time")
        if last_day_pool(series.time, resolve_cycle(series, "time", None)) is not None:
            failures.append(f"{calendar} {start}..{end}: a day-366 pool for a series without day 366")
    assert_no_failures(failures)


def test_every_rolling_mean_runs_pinned(client, monkeypatch):
    """Each float rolling mean in detect runs under ``rolling_numerics``, for every method that has one.

    A spy on xarray's rolling ``mean`` and ``construct`` (the climatology builds its windows
    directly) records the accelerator options in force when it is called, which is when xarray
    reads them. The run is under a global ``use_bottleneck=True``, so a site that loses its pin
    records True, and so does a site whose pin is flipped; comparing output values cannot see
    the second case.
    """
    seen = []
    classes = {type(xr.DataArray([0.0]).rolling(dim_0=1)), type(xr.Dataset({"v": ("t", [0.0])}).rolling(t=1))}
    for cls in classes:
        for name in ("mean", "construct"):
            original = getattr(cls, name)

            def spy(self, *args, _original=original, **kwargs):
                seen.append({key: OPTIONS.get(key) for key in ("use_bottleneck", "use_numbagg")})
                return _original(self, *args, **kwargs)

            monkeypatch.setattr(cls, name, spy)

    sst = gridded_sst().isel(time=slice(-6 * 365, None), lat=slice(0, 4), lon=slice(0, 4)).load().chunk({"time": 25})
    methods = {
        "shifting_baseline": {"method_anomaly": "shifting_baseline", "window_years": 5, "smooth_days": 11},
        "fixed_baseline": {"method_anomaly": "fixed_baseline"},
        "detrend_harmonic standardised": {"method_anomaly": "detrend_harmonic", "standardise": True},
    }
    failures = []
    for case, kwargs in methods.items():
        seen.clear()
        with xr.set_options(use_bottleneck=True):
            marEx.preprocess_data(
                sst, dimensions=GRID_DIMS, precision=0.01, max_anomaly=5.0, method_extreme="global_percentile", **kwargs
            )
        if not seen:
            failures.append(f"{case}: no rolling mean was reached; the guard is vacuous")
        unpinned = [options for options in seen if options["use_bottleneck"] or options["use_numbagg"]]
        if unpinned:
            failures.append(f"{case}: {len(unpinned)} of {len(seen)} rolling means ran unpinned: {unpinned}")
    assert_no_failures(failures)
