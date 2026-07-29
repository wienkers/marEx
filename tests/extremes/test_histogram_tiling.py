"""Tile budgets of the histogram and exact paths, and the invariances they rest on.

Each histogram tile holds every cell's full reduced (time) axis, so the computed quantile must be
bit for bit the same under any spatial tiling: that is the "memory-safe and bit-exact" guarantee.
What the tiling itself must bound is a task's working set: what it reads (``ntime`` per cell) and
what it writes (the per-cell output), constant in the series length, per level of an extra
dimension, and reported when the budget cannot be honoured (a ``window_spatial`` floor wider than
the budget's tile, or a single cell over the whole budget). Values cannot catch these: every
tiling is value-identical by construction.
"""

import dask.array as dsa
import numpy as np
import pandas as pd
import xarray as xr

import marEx
import marEx.extremes.global_percentile as global_percentile
import marEx.extremes.histogram as H
import marEx.extremes.seasonal_percentile as seasonal_percentile
from marEx.core import dimensions as dims_mod
from marEx.core.dimensions import FIT_WARNING_MARKER, canonical_time_chunks, extra_dim_chunks, horizontal_dims, tile_spatial_chunks

from ..conftest import assert_no_failures
from .conftest import DATA_DIR, GRID_DIMS, MESH_DIMS, check

PRECISION = 0.01
MAX_ANOMALY = 5.0
DEPTH_LENGTHS = (1, 4, 25, 50)


# ── the 1-D and dense 2-D tile budgets ────────────────────────────────


def test_1d_tile_budget():
    """The tile is budgeted against the array produced, not only the one consumed.

    A task reads ``ntime x tile_area`` and writes ``output_elements_per_cell x tile_area``. Sizing on
    ``ntime`` alone bounds only the read, and since ``tile_area = target // ntime`` the write is
    then ``target * output_per_cell / ntime``: over budget exactly when the series is shorter than
    the per-cell output. For a long series (the normal case) the tiling must not move.
    """
    target, ntime, output_per_cell = 1_000_000, 100, 500  # output deliberately > ntime
    da = xr.DataArray(np.zeros((ntime, 200, 200), dtype=np.float32), dims=("time", "lat", "lon")).chunk(
        {"time": -1, "lat": -1, "lon": -1}
    )
    tiled = H._chunk_spatial_for_histogram(da, "time", target_elements=target, output_elements_per_cell=output_per_cell)
    cells = tiled.chunks[1][0] * tiled.chunks[2][0]
    # The tolerance absorbs the integer rounding of the tile side (round(sqrt(area))**2 can exceed
    # area slightly); a few percent, not the 5x the input-only budget gives here.
    tol = 1.1
    assert ntime * cells <= target * tol, "input slab over budget"
    assert output_per_cell * cells <= target * tol, "produced array over budget"
    # And the old input-only budget genuinely violated it, so this test can fail.
    input_only = H._chunk_spatial_for_histogram(da, "time", target_elements=target)
    assert output_per_cell * input_only.chunks[1][0] * input_only.chunks[2][0] > target * tol

    long = xr.DataArray(np.zeros((5000, 180, 360), dtype=np.float32), dims=("time", "lat", "lon")).chunk(
        {"time": -1, "lat": -1, "lon": -1}
    )
    assert (
        H._chunk_spatial_for_histogram(long, "time").chunks
        == H._chunk_spatial_for_histogram(long, "time", output_elements_per_cell=502).chunks
    )


def test_dense_2d_tile_budget_is_constant_in_series_length():
    """The 2-D path holds time whole, so its tile is budgeted against ``max(ntime, 366 * n_bins)``.

    Sizing only on the ``366 x n_bins`` output leaves the slab each task READS growing linearly
    with the series. Asserted on the production dense tiler, not on a copy of its formula.
    """
    n_bins = 502
    out_per_cell = 366 * n_bins
    budget = H._HISTOGRAM_TASK_ELEMENTS

    def cells(ntime):
        da = xr.DataArray(dsa.zeros((ntime, 10_000), chunks=(ntime, 1_000), dtype=np.float32), dims=("time", "ncells"))
        chunks = H._histogram_tile_chunks(da, MESH_DIMS, n_bins, None, 366)
        assert chunks["time"] == -1
        return chunks["ncells"]

    # Realistic series: the output term dominates, so tiling is unchanged.
    assert cells(1_825) == cells(5_475) == cells(20_000)
    assert out_per_cell * cells(5_475) <= budget < out_per_cell * (cells(5_475) + 1)
    # Absurdly long series: the tile shrinks so the read slab stays inside the budget.
    huge = out_per_cell * 4
    assert cells(huge) < cells(5_475)
    assert huge * cells(huge) <= budget
    for ntime in (10, 366, 5_475, 100_000, out_per_cell, out_per_cell * 10):
        c = cells(ntime)
        assert ntime * c <= budget, f"read slab over budget at ntime={ntime}"
        assert out_per_cell * c <= budget, f"histogram over budget at ntime={ntime}"


# ── tiling invariance of the values ───────────────────────────────────


def _sst_anomaly(n_years, depth=False):
    """The shipped gridded fixture's first ``n_years``, harmonically detrended; values in memory, never persisted."""
    sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.isel(time=slice(0, n_years * 365))
    if depth:
        sst = xr.concat([sst, sst * 0.8 - 1.5, sst * 1.3 + 2.25], dim="depth")
        sst = sst.assign_coords(depth=np.arange(3, dtype=np.float32)).transpose("time", "depth", "lat", "lon")
    sst = sst.compute().chunk({d: (30 if d == "time" else -1) for d in sst.dims})
    anom = marEx.anomaly.compute_normalised_anomaly(sst, method_anomaly="detrend_harmonic", dimensions=GRID_DIMS).dat_anomaly
    anom = anom.compute().chunk({d: (30 if d == "time" else -1) for d in anom.dims})
    anom.name = "dat_anomaly"
    return anom


def test_1d_quantile_is_tiling_invariant(monkeypatch):
    """The 1-D quantile is identical under one tile and many; each row checks that it really tiled.

    Rows: the real ``_chunk_spatial_for_histogram`` at a small budget (not a stub); the output-bound
    budget on a short series; the harmonic anomaly of the gridded fixture under a one-tile and a
    7 x 7 stub; and the same with a depth axis under a 3 x 3 x 3 stub, where the default tiling
    must cover every spatial dim, depth included, with the reduced axis whole.
    """
    failures = []

    def real_chunker_multi_tile():
        da = xr.DataArray(np.random.default_rng(4).normal(0.0, 1.0, size=(2000, 60)), dims=("time", "x"), name="dat_anomaly").chunk(
            {"time": -1, "x": -1}
        )
        tiled = H._chunk_spatial_for_histogram(da, "time", target_elements=2000 * 8)
        assert len(tiled.chunks[tiled.dims.index("x")]) > 1, "real chunker did not multi-tile"
        with monkeypatch.context() as m:
            m.setattr(H, "_HISTOGRAM_TASK_ELEMENTS", 10**12)  # one tile
            big = H._compute_histogram_quantile_1d(da, 0.95, dim="time").compute()
            m.setattr(H, "_HISTOGRAM_TASK_ELEMENTS", 2000 * 8)  # ~8 cells per tile
            small = H._compute_histogram_quantile_1d(da, 0.95, dim="time").compute()
        np.testing.assert_array_equal(big.values, small.values)

    def output_bound_budget():
        ntime = 120  # short on purpose: this is where the input-only and output-bound budgets diverge
        da = xr.DataArray(
            np.random.default_rng(20260807).normal(size=(ntime, 240)).astype(np.float32),
            dims=("time", "x"),
            coords={"time": xr.date_range("2000-01-01", periods=ntime, freq="D")},
            name="dat",
        ).chunk({"time": -1, "x": 60})
        with monkeypatch.context() as m:
            m.setattr(H, "_HISTOGRAM_TASK_ELEMENTS", 10**12)
            reference = H._compute_histogram_quantile_1d(da, 0.95, dim="time").compute()
            m.setattr(H, "_HISTOGRAM_TASK_ELEMENTS", ntime * 40)  # the output bound binds
            tiled = H._compute_histogram_quantile_1d(da, 0.95, dim="time").compute()
        np.testing.assert_array_equal(tiled.values, reference.values)

    def stubbed(anom, side):
        def single_tile(da, dim, **kwargs):
            return da.chunk({d: -1 for d in da.dims})

        def many_tiles(da, dim, **kwargs):
            return da.chunk({**{d: side for d in da.dims if d != dim}, dim: -1})

        with monkeypatch.context() as m:
            m.setattr(H, "_chunk_spatial_for_histogram", single_tile)
            single = H._compute_histogram_quantile_1d(anom, 0.95, dim="time").compute()
            m.setattr(H, "_chunk_spatial_for_histogram", many_tiles)
            multi = H._compute_histogram_quantile_1d(anom, 0.95, dim="time").compute()
        assert many_tiles(anom, "time").data.npartitions > 1
        return single, multi

    def gridded_anomaly():
        single, multi = stubbed(_sst_anomaly(4), 7)  # 7 x 7 cells: several tiles over the 20 x 40 fixture
        np.testing.assert_array_equal(single.values, multi.values)

    def gridded_anomaly_with_depth():
        anom = _sst_anomaly(3, depth=True)
        tiled = H._chunk_spatial_for_histogram(anom, "time")
        assert set(tiled.chunksizes) == {"time", "depth", "lat", "lon"}
        assert tiled.chunksizes["time"] == (anom.sizes["time"],), "the reduced axis must stay whole"
        single, multi = stubbed(anom, 3)
        assert set(single.dims) == {"depth", "lat", "lon"}
        np.testing.assert_array_equal(single.values, multi.values)

    for row in (real_chunker_multi_tile, output_bound_budget, gridded_anomaly, gridded_anomaly_with_depth):
        check(failures, row.__name__, row)
    assert_no_failures(failures)


def test_2d_quantile_is_tiling_invariant(monkeypatch):
    """The 2-D per-day-of-year quantile, with spatial smoothing, does not depend on the tile chosen internally."""
    rng = np.random.default_rng(3)
    time = xr.date_range("2001-01-01", periods=365 * 3, freq="D", use_cftime=False)
    da = xr.DataArray(
        rng.normal(0.0, 1.0, size=(time.size, 6, 8)).astype(np.float32),
        dims=["time", "y", "x"],
        coords={"time": time},
        name="dat_anomaly",
    )
    da = da.assign_coords(dayofyear=da["time"].dt.dayofyear).chunk({"time": -1, "y": -1, "x": -1})
    dims = {"time": "time", "x": "x", "y": "y"}
    n_centres = len(np.arange(-PRECISION, MAX_ANOMALY + PRECISION, PRECISION)) + 1

    def run(target):
        with monkeypatch.context() as m:
            m.setattr(H, "_HISTOGRAM_TASK_ELEMENTS", target)
            return H._compute_histogram_quantile_2d(
                da, 0.9, window_steps=3, window_spatial=3, dimensions=dims, precision=PRECISION, max_anomaly=MAX_ANOMALY
            ).compute()

    big = run(50_000_000)  # one tile covering the whole field
    # A budget that tiles into several chunks while staying >= the smoothing window (~4x4 tiles over 6x8).
    small = run(366 * n_centres * 16)
    np.testing.assert_array_equal(big.values, small.values, err_msg="2D quantile changed with tiling")


# ── extra dimensions ──────────────────────────────────────────────────


def test_extra_dim_chunks_rule():
    """The one rule every site uses: one level per chunk while the slice is tiled, levels stacked once a whole slice fits."""
    rows = {
        "a 2-D field gets nothing": (({"lat": 10, "lon": 20}, [], 200, 200, 10_000), {}),
        "one level per chunk while the slice is tiled": (({"depth": 50}, ["depth"], 7225, 1_036_800, 7225), {"depth": 1}),
        "a whole slice stacks levels shortest first": (
            ({"member": 4, "depth": 100}, ["depth", "member"], 10, 10, 1000),
            {"member": 4, "depth": 25},
        ),
        "a floor over budget stacks nothing": (({"depth": 50}, ["depth"], 400, 400, 300), {"depth": 1}),
    }
    failures = []
    for case, (args, expected) in rows.items():

        def row(args=args, expected=expected):
            assert extra_dim_chunks(*args) == expected

        check(failures, case, row)
    assert_no_failures(failures)


def test_extra_dim_tile_is_per_level():
    """Each level of an extra dim gets the tile its horizontal slice would, at any depth length 1..50.

    Pinned as invariance in the extra dim's length, not an absolute bound: a bound loose enough to
    pass at depth 1 also passes while depth 50 allocates 50x. Asserted on the chunk dict the module
    chooses, not on values, which every tiling leaves identical. Lazy fields stand in for large
    grids (only sizes are read); each row first checks that its slice is actually tiled.

    Rows: ``_chunk_spatial_for_histogram`` (the 1-D path and the exact seasonal path) on a
    720 x 1440 grid and a 5 M cell mesh, at per-cell outputs 366 and 503; ``_histogram_tile_chunks``
    (the dense 2-D path) on a 2 M cell mesh, and on the gridded fixture's own 3-D shape with
    ``window_spatial=5``, whose horizontal floor applies to the horizontal dims only.
    """
    failures = []

    def chunk_spatial(grid, output_elements_per_cell):
        horizontal = {"gridded": {"lat": 720, "lon": 1440}, "mesh": {"ncells": 5_000_000}}[grid]
        dims = GRID_DIMS if grid == "gridded" else MESH_DIMS

        def field(*extra):
            names = ("time", *(("depth",) if extra else ()), *horizontal)
            shape = (7000, *extra, *horizontal.values())
            return xr.DataArray(dsa.zeros(shape, chunks=(100, *extra, *horizontal.values()), dtype=np.float32), dims=names)

        def tile(da):
            out = H._chunk_spatial_for_histogram(
                da, "time", output_elements_per_cell=output_elements_per_cell, horizontal=horizontal_dims(dims)
            )
            return {d: out.chunksizes[d][0] for d in out.dims}

        slice_tile = tile(field())
        assert all(slice_tile[d] < n for d, n in horizontal.items()), f"{slice_tile} is not tiled; vacuous"
        for ndepth in DEPTH_LENGTHS:
            got = tile(field(ndepth))
            assert got == {**slice_tile, "depth": 1}, f"depth={ndepth}: {got} != per-level {slice_tile}"

    def dense_mesh():
        ncells, ntime, n_bins = 2_000_000, 4 * 365, 503

        def mesh(*extra):
            dims = ("time", "depth", "ncells") if extra else ("time", "ncells")
            return xr.DataArray(dsa.zeros((ntime, *extra, ncells), chunks=(30, *extra, ncells), dtype=np.float32), dims=dims)

        slice_tile = H._histogram_tile_chunks(mesh(), MESH_DIMS, n_bins, window_spatial=None)
        assert slice_tile["ncells"] < ncells, f"slice tile {slice_tile} is not tiled; vacuous"
        for ndepth in DEPTH_LENGTHS:
            tile = H._histogram_tile_chunks(mesh(ndepth), MESH_DIMS, n_bins, window_spatial=None)
            assert tile == {**slice_tile, "depth": 1}, f"depth={ndepth}: {tile} != per-level {slice_tile}"

    def dense_gridded():
        # The old rank-th root put (3, 5, 5) tiles here, which peaked at 1.8x the RSS of the
        # per-level (1, 12, 12) layout on a crop of the slice oracle.
        n_bins = 503  # default precision=0.01, max_anomaly=5.0
        base = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.isel(time=slice(0, 4 * 365))
        level = base.expand_dims(depth=np.array([0.0], dtype=np.float32)).transpose("time", "depth", "lat", "lon")
        level = level.chunk({"time": 30, "depth": -1, "lat": -1, "lon": -1})
        budget_cells = H._HISTOGRAM_TASK_ELEMENTS // max(int(level.sizes["time"]), 366 * n_bins)
        slice_tile = H._histogram_tile_chunks(level.isel(depth=0, drop=True), GRID_DIMS, n_bins, window_spatial=5)
        assert slice_tile["lat"] * slice_tile["lon"] <= budget_cells, slice_tile
        assert slice_tile["lat"] < level.sizes["lat"], f"{slice_tile} is not tiled; vacuous"
        for ndepth in DEPTH_LENGTHS:
            da = level.reindex(depth=np.arange(ndepth, dtype=np.float32), method="nearest")
            tile = H._histogram_tile_chunks(da, GRID_DIMS, n_bins, window_spatial=5)
            assert tile == {**slice_tile, "depth": 1}, f"depth={ndepth}: {tile} != per-level {slice_tile}"
        assert slice_tile["lat"] >= 5 and slice_tile["lon"] >= 5, slice_tile

    for grid in ("gridded", "mesh"):
        for out in (366, 503):
            check(failures, f"chunk_spatial {grid} output={out}", lambda grid=grid, out=out: chunk_spatial(grid, out))
    check(failures, "dense mesh", dense_mesh)
    check(failures, "dense gridded ws=5", dense_gridded)
    assert_no_failures(failures)


def test_a_partial_horizontal_mapping_keeps_the_old_root():
    """A mapping naming only one of ``da``'s horizontal dims must not give it a 1-D side."""
    da = xr.DataArray(dsa.zeros((7000, 3, 720, 1440), chunks=(100, 3, 720, 1440)), dims=("time", "depth", "lat", "lon"))
    old = H._chunk_spatial_for_histogram(da, "time", output_elements_per_cell=366)
    partial = H._chunk_spatial_for_histogram(da, "time", output_elements_per_cell=366, horizontal=("lat", "x"))
    assert partial.chunksizes == old.chunksizes


def test_a_small_horizontal_extent_stacks_levels():
    """A mooring-like field: one whole slice fits, so levels share a tile instead of one task each."""
    da = xr.DataArray(dsa.zeros((1460, 4, 100, 10), chunks=(100, 4, 100, 10)), dims=("time", "member", "depth", "ncells"))
    out = H._chunk_spatial_for_histogram(da, "time", output_elements_per_cell=503, horizontal=("ncells",))
    tile_area = 50_000_000 // 1460
    chunks = {d: out.chunksizes[d] for d in ("member", "depth", "ncells")}
    assert chunks["ncells"] == (10,)
    assert max(chunks["member"]) * max(chunks["depth"]) * 10 <= tile_area, chunks
    assert int(np.prod([len(c) for c in chunks.values()])) <= 2, f"{chunks}: one task per level is back"


# ── the fit warning ───────────────────────────────────────────────────


def _field(n_time=200, n_lat=20, n_lon=30, lazy=False):
    """Small gridded field, chunked the way a caller usually hands one over."""
    shape = (n_time, n_lat, n_lon)
    data = dsa.zeros(shape, chunks=(25, n_lat, n_lon), dtype=np.float32) if lazy else np.zeros(shape, dtype=np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(n_time), "lat": np.arange(n_lat), "lon": np.arange(n_lon)},
        name="sst",
    )
    return da.chunk({"time": 25, "lat": -1, "lon": -1})


def _fit_warnings(capture):
    return [m for m in capture.messages if FIT_WARNING_MARKER in m]


def test_fit_warning_stays_silent_on_the_ordinary_path(marex_warnings, monkeypatch):
    """A warning that fires on the normal path is worse than no warning. Each row checks it really tiled, or really overshot."""
    failures = []

    def canonical_rechunk():
        with marex_warnings as cap:
            chunks = canonical_time_chunks(_field(), GRID_DIMS)
        assert _fit_warnings(cap) == []
        assert chunks["time"] == -1

    def forced_small_budget():
        # The budget is honoured by shrinking the tile: not a misfit. The kept-chunk path is the next row.
        with monkeypatch.context() as m:
            m.setattr(dims_mod, "TASK_ELEMENTS", 30_000)
            m.setattr(dims_mod, "KEEP_INPUT_CHUNK_FACTOR", 1)
            with marex_warnings as cap:
                chunks = canonical_time_chunks(_field(), GRID_DIMS)
        assert _fit_warnings(cap) == []
        assert chunks["lat"] < 20 and chunks["lon"] < 30  # it really did tile

    def kept_input_chunk():
        # A caller-sized spatial chunk within the keep band is the tile (10 x 30 x 200 = 2x the budget).
        with monkeypatch.context() as m:
            m.setattr(dims_mod, "TASK_ELEMENTS", 30_000)
            with marex_warnings as cap:
                chunks = canonical_time_chunks(_field().chunk({"lat": 10}), GRID_DIMS)
        assert _fit_warnings(cap) == []
        assert (chunks["lat"], chunks["lon"]) == (10, 30)

    def tile_side_rounding_alone():
        # The day-of-year path takes a flat round(cells ** (1/rank)) side, which can overshoot its own
        # cell budget; the greedy loop in tile_spatial_chunks cannot, so written against it this row
        # would be VACUOUS. Budget 3,000,000 over 1,000,000 per cell gives 3 cells; the flat side is
        # round(sqrt(3)) = 2, so 4 cells = 1.33x the budget with NO floor and NO single cell over.
        with monkeypatch.context() as m:
            m.setattr(H, "_HISTOGRAM_TASK_ELEMENTS", 3_000_000)
            with marex_warnings as cap:
                chunks = H._histogram_tile_chunks(_field(), GRID_DIMS, n_bins=1000, window_spatial=None, cycle_length=1000)
        estimate = chunks["lat"] * chunks["lon"] * 1_000_000
        assert estimate > 3_000_000, f"vacuous: {chunks} gives {estimate}, inside the budget"
        assert _fit_warnings(cap) == []

    def floor_inside_the_budget():
        with marex_warnings as cap:
            chunks = tile_spatial_chunks(
                _field(), ("lat", "lon"), input_elements_per_cell=1, target_elements=10_000, floor_dims=("lat", "lon"), floor=3
            )
        assert _fit_warnings(cap) == []
        assert min(chunks.values()) >= 3

    def same_window_on_a_daily_cycle():
        # The control for the hourly row of the warning test: only the cadence changes.
        with marex_warnings as cap:
            chunks = H._histogram_tile_chunks(
                _field(n_lat=60, n_lon=80), GRID_DIMS, n_bins=1000, window_spatial=5, cycle_length=366
            )
        assert _fit_warnings(cap) == []
        assert chunks == {"time": -1, "lat": 12, "lon": 12}

    def daily_default():
        with marex_warnings as cap:
            H._histogram_tile_chunks(_field(), GRID_DIMS, n_bins=200, window_spatial=None, cycle_length=366)
        assert _fit_warnings(cap) == []

    def window_inside_the_budget():
        with marex_warnings as cap:
            H._histogram_tile_chunks(_field(), GRID_DIMS, n_bins=200, window_spatial=5, cycle_length=366)
        assert _fit_warnings(cap) == []

    def coarse_bins_on_the_even_window_path():
        # The control for the 3000-bin row of the warning test: a coarser precision is a real lever.
        with marex_warnings as cap:
            H._histogram_tile_chunks(
                _field(n_time=87840, n_lat=40, n_lon=40, lazy=True), GRID_DIMS, n_bins=30, window_spatial=4, cycle_length=8784
            )
        assert _fit_warnings(cap) == []

    for row in (
        canonical_rechunk,
        forced_small_budget,
        kept_input_chunk,
        tile_side_rounding_alone,
        floor_inside_the_budget,
        same_window_on_a_daily_cycle,
        daily_default,
        window_inside_the_budget,
        coarse_bins_on_the_even_window_path,
    ):
        check(failures, row.__name__, row)
    assert_no_failures(failures)


def test_fit_warning_fires_where_the_budget_cannot_hold(marex_warnings, monkeypatch):
    """A window floor wider than the budget's tile, or a single cell over the whole budget: warn once, and never nudge the tile.

    The message names the window, the elements, the budget and the levers the user actually has
    (``window_spatial``, ``streaming``, ``precision``), and never suggests chunking the input finer:
    the floor is applied before the caller's chunks are taken into account, so that would narrow the
    tile below the rolling window.
    """
    failures = []

    def floor_names_the_window():
        with marex_warnings as cap:
            chunks = tile_spatial_chunks(
                _field(), ("lat", "lon"), input_elements_per_cell=1000, target_elements=10_000, floor_dims=("lat", "lon"), floor=15
            )
        messages = _fit_warnings(cap)
        assert len(messages) == 1, messages
        assert "15" in messages[0]  # the window that forced it
        assert "225,000" in messages[0]  # 15 x 15 cells x 1000 elements
        assert "10,000" in messages[0]  # the budget it broke
        assert chunks == {"lat": 15, "lon": 15}  # observation only
        for lever in ("window_spatial", "streaming", "precision"):
            assert lever in messages[0], lever
        assert "chunk your input" not in messages[0].lower()

    def single_cell_over_the_budget():
        with marex_warnings as cap:
            chunks = tile_spatial_chunks(_field(), ("lat", "lon"), input_elements_per_cell=5_000, target_elements=100)
        messages = _fit_warnings(cap)
        assert len(messages) == 1, messages
        assert "single" in messages[0].lower()
        assert chunks == {"lat": 1, "lon": 1}

    def output_side_counts_too():
        # The budget divisor is the larger of the two sides (dd43301).
        with marex_warnings as cap:
            tile_spatial_chunks(
                _field(), ("lat", "lon"), input_elements_per_cell=1, output_elements_per_cell=5_000, target_elements=100
            )
        assert len(_fit_warnings(cap)) == 1

    def sub_daily_cycle_with_a_wide_window():
        with monkeypatch.context() as m:
            m.setattr(H, "_HISTOGRAM_TASK_ELEMENTS", 1_000_000)
            with marex_warnings as cap:
                chunks = H._histogram_tile_chunks(_field(), GRID_DIMS, n_bins=1000, window_spatial=15, cycle_length=8784)
        messages = _fit_warnings(cap)
        assert len(messages) == 1, messages
        assert "15" in messages[0]
        assert chunks == {"time": -1, "lat": 15, "lon": 15}  # unchanged: time whole, both dims at the window

    def hourly_cadence_at_the_shipped_budget():
        # No monkeypatching: a sub-daily cycle multiplies the per-cell output by the steps per day,
        # so the cell budget collapses (50e6 / (8784 x 1000) = 5 cells) while window_spatial=5 does
        # not move: 5 x 5 cells at 8,784,000 elements each is 4.4x the budget.
        with marex_warnings as cap:
            chunks = H._histogram_tile_chunks(
                _field(n_lat=60, n_lon=80), GRID_DIMS, n_bins=1000, window_spatial=5, cycle_length=8784
            )
        messages = _fit_warnings(cap)
        assert len(messages) == 1, messages
        assert "219,600,000" in messages[0]
        assert "50,000,000" in messages[0]
        assert chunks == {"time": -1, "lat": 5, "lon": 5}

    def fine_bins_on_the_even_window_path():
        # The message offers a coarser precision for an even window_spatial; the silent test has the 30-bin control.
        with marex_warnings as cap:
            H._histogram_tile_chunks(
                _field(n_time=87840, n_lat=40, n_lon=40, lazy=True), GRID_DIMS, n_bins=3000, window_spatial=4, cycle_length=8784
            )
        assert _fit_warnings(cap), "3000 bins on the even-window path did not warn"

    for row in (
        floor_names_the_window,
        single_cell_over_the_budget,
        output_side_counts_too,
        sub_daily_cycle_with_a_wide_window,
        hourly_cadence_at_the_shipped_budget,
        fine_bins_on_the_even_window_path,
    ):
        check(failures, row.__name__, row)
    assert_no_failures(failures)


# ── the exact path's layout ───────────────────────────────────────────


def _mesh(n_time, n_cells=20000):
    t = pd.date_range("2000-01-01", periods=n_time, freq="D")
    return xr.DataArray(np.zeros((n_time, n_cells), dtype=np.float32), dims=("time", "ncells"), coords={"time": t}, name="a").chunk(
        {"time": 10}
    )


def test_mesh_exact_tile(monkeypatch):
    """Twice the series, half the cells: the per-task element count is invariant in n_time, above a 100-cell floor.

    50,000 cells give a 300-cell tile from n_cells alone, which both lengths got before. A budget
    smaller than one 100-cell tile cannot shred the mesh into single cells.
    """
    seen = {}
    real = xr.DataArray.quantile

    def spy(self, *args, **kwargs):
        seen["tile"] = max(self.chunks[self.dims.index("ncells")])
        return real(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "quantile", spy)

    def tile(da, budget):
        with monkeypatch.context() as m:
            m.setattr(global_percentile, "TASK_ELEMENTS", budget)
            global_percentile._identify_extremes_constant(da, 95, "exact", MESH_DIMS)
        return seen["tile"]

    assert (tile(_mesh(100, n_cells=50_000), 24_000), tile(_mesh(200, n_cells=50_000), 24_000)) == (240, 120)
    assert tile(_mesh(200), 1_000) == 100


def test_seasonal_exact_graph_carries_the_cycle_index(monkeypatch):
    """The ufunc kwargs scale with n_time, not with cycle.length x n_time (a boolean mask table)."""
    captured = {}
    real = seasonal_percentile.xr.apply_ufunc

    def spy(func, *args, **kwargs):
        captured.update(kwargs.get("kwargs", {}))
        return real(func, *args, **kwargs)

    monkeypatch.setattr(seasonal_percentile.xr, "apply_ufunc", spy)
    n_time = 3653
    t = pd.date_range("2000-01-01", periods=n_time, freq="D")
    da = xr.DataArray(
        np.zeros((n_time, 2, 3), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": t, "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
        name="a",
    ).chunk({"time": -1})
    seasonal_percentile._identify_extremes_seasonal(
        da, 95, method_percentile="exact", dimensions=GRID_DIMS, coordinates={"time": "time"}
    )
    payload = sum(v.nbytes for v in captured.values() if isinstance(v, np.ndarray))
    payload += sum(a.nbytes for v in captured.values() if isinstance(v, list) for a in v if isinstance(a, np.ndarray))
    assert captured, "the exact seasonal path never reached apply_ufunc"
    assert payload <= 4 * n_time, f"{payload} B of arrays ride in the graph (a 366 x n_time table is {366 * n_time} B)"
