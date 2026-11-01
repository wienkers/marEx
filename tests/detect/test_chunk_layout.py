"""Detect values do not depend on the input or internal layout, and the layout itself is bounded.

Every reduction along time -- the smoothing's running sum, flox's grouped means, the harmonic
fit -- accumulates block by block, so the floating-point result used to move with the input's
time chunk boundaries: ~1e-4 K, enough to move a threshold across a 0.01 bin and flip extreme
events. The anomaly stage now reduces with the time axis whole inside bounded spatial tiles
(``canonical_time_chunks``), so every layout must reproduce the time-whole reference bit for bit,
for every anomaly, threshold and percentile method (all 16 upper-tail triples, plus a designed
lower-tail subset with absolute anchors).

The canonical tile keeps an input spatial chunk the caller already sized: splitting a
25,000-cell chunk into ~6.8k-cell tiles and merging it back multiplies the task count without
lowering a memory figure the caller did not accept. The tile is the input chunk while one task on
it stays within ``KEEP_INPUT_CHUNK_FACTOR`` times ``TASK_ELEMENTS``; beyond that the budget tile
applies. Values never move with the spatial layout, and the graph is asserted by its size, not
only its values.

Tables (one behaviour, several inputs) run every row, collect failures with the row's case id,
and fail once at the end.
"""

from types import SimpleNamespace

import dask.array as dka
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
import marEx.core.dimensions as dims
from marEx.anomaly.base import compute_normalised_anomaly
from marEx.core.compute_mode import Materialiser
from marEx.core.finalise import finalise_dataset
from marEx.extremes import seasonal_percentile

from ..conftest import assert_no_failures
from .conftest import COMPARED_VARS, GRID_DIMS, MESH_COORDS, MESH_DIMS, check, open_sst_gridded

# ── every layout matches the time-whole reference ─────────────────────

ANOMALY_METHODS = ["shifting_baseline", "fixed_baseline", "detrend_fixed_baseline", "detrend_harmonic"]
EXTREME_METHODS = ["global_percentile", "seasonal_percentile"]
PERCENTILE_METHODS = ["exact", "approximate"]
# (input layout, per-task element budget). The fixture fits one spatial tile at the default budget, so
# the last case shrinks the budget until the canonical layout splits space into several tiles.
LAYOUTS = [({"time": 17}, None), ({"time": 60, "lat": 4, "lon": 7}, None), ({"lat": 3, "lon": 5}, None), ({"time": 30}, 30_000)]


@pytest.fixture(scope="module")
def sst():
    """6 years of an 8 x 14 corner of the gridded fixture, in memory (numpy)."""
    return open_sst_gridded().isel(time=slice(-6 * 365, None), lat=slice(0, 8), lon=slice(0, 14)).load()


def _detect(sst, layout, method_anomaly, method_extreme, method_percentile, tail="upper"):
    kw = {
        "method_anomaly": method_anomaly,
        "method_extreme": method_extreme,
        "method_percentile": method_percentile,
        "threshold_percentile": 95 if tail == "upper" else 5,
        "tail": tail,
        "dimensions": GRID_DIMS,
        "dask_chunks": {"time": 25},
    }
    if method_anomaly == "shifting_baseline":
        kw.update(window_years=3, smooth_days=11)
    if method_extreme == "seasonal_percentile":
        kw.update(window_days=3)
    if method_percentile == "approximate":
        kw.update(precision=0.01, max_anomaly=5.0)
    return marEx.preprocess_data(sst.chunk(layout), **kw).compute()


def _assert_layouts_match(sst, ref, monkeypatch, *args, tail="upper", check_tiled=False):
    for layout, budget in LAYOUTS:
        if budget is not None:
            monkeypatch.setattr(dims, "TASK_ELEMENTS", budget)
            if check_tiled:
                tile = dims.canonical_time_chunks(sst.chunk(layout), GRID_DIMS)
                assert tile["lat"] < sst.sizes["lat"] and tile["lon"] < sst.sizes["lon"], f"budget {budget} did not tile: {tile}"
        got = _detect(sst, layout, *args, tail=tail)
        monkeypatch.undo()
        for name in COMPARED_VARS:
            if name not in ref.data_vars:
                continue
            a, b = ref[name].values, got[name].values
            assert a.dtype == b.dtype, f"{layout}: {name} dtype moved"
            assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), f"{layout}: {name} is not bit-identical"


@pytest.mark.parametrize("method_percentile", PERCENTILE_METHODS)
@pytest.mark.parametrize("method_extreme", EXTREME_METHODS)
@pytest.mark.parametrize("method_anomaly", ANOMALY_METHODS)
def test_every_layout_matches_the_time_whole_reference(sst, method_anomaly, method_extreme, method_percentile, monkeypatch):
    ref = _detect(sst, {"time": -1}, method_anomaly, method_extreme, method_percentile)
    _assert_layouts_match(sst, ref, monkeypatch, method_anomaly, method_extreme, method_percentile, check_tiled=True)


# A designed subset, not the full grid: the histogram path (`_apply_threshold_bounds`) under both
# anomaly families and both threshold methods, and the exact path's lower-tail clamp under both as well.
LOWER_TAIL_CASES = [
    ("shifting_baseline", "global_percentile", "approximate"),
    ("detrend_harmonic", "seasonal_percentile", "approximate"),
    ("fixed_baseline", "global_percentile", "exact"),
    ("detrend_fixed_baseline", "seasonal_percentile", "exact"),
]
FLAT_CELL = {"lat": 2, "lon": 3}


@pytest.fixture(scope="module")
def sst_with_flat_cell(sst):
    """The same field with one cell held constant, so its anomaly is zero and the guard rail must bind."""
    flat = sst.copy(deep=True)
    flat[:, FLAT_CELL["lat"], FLAT_CELL["lon"]] = 15.0
    return flat


@pytest.mark.parametrize(
    ("method_anomaly", "method_extreme", "method_percentile"), LOWER_TAIL_CASES, ids=["-".join(c) for c in LOWER_TAIL_CASES]
)
def test_lower_tail_is_chunk_invariant_and_stays_below_zero(
    sst_with_flat_cell, method_anomaly, method_extreme, method_percentile, monkeypatch
):
    """tail='lower' has its own guard and range branches: bit-identical across layouts AND anchored in absolute terms.

    Invariance alone cannot see a lower-tail branch that is wrong the same way for every layout, so the reference
    is also held to what a lower-tail threshold must be: negative, nowhere near all clamped onto the guard rail,
    and on the guard rail (one bin from zero, or the smallest normal float) in the constant cell.
    """
    sst = sst_with_flat_cell
    ref = _detect(sst, {"time": -1}, method_anomaly, method_extreme, method_percentile, tail="lower")

    thr = ref["thresholds"]
    assert float(thr.max()) < 0, "a lower-tail threshold must lie below zero"
    assert float(thr.median()) < -0.1, "nearly every lower-tail threshold was clamped onto the guard rail"
    # The seasonal approximate path smooths thresholds across neighbouring cells, so the flat cell borrows theirs.
    if not (method_extreme == "seasonal_percentile" and method_percentile == "approximate"):
        flat_thr = thr.isel(**FLAT_CELL).values
        guard = -0.01 if method_percentile == "approximate" else -np.finfo(np.float32).tiny
        assert np.allclose(flat_thr, guard, rtol=1e-6, atol=0), (
            f"flat cell threshold {flat_thr.ravel()[:3]} is not on the guard rail {guard}"
        )
        assert int(ref["extreme_events"].isel(**FLAT_CELL).sum()) == 0, "a constant anomaly was flagged extreme"

    _assert_layouts_match(sst, ref, monkeypatch, method_anomaly, method_extreme, method_percentile, tail="lower")


# ── the canonical tile ────────────────────────────────────────────────


def _field(nt, ncells, chunks, seed=None):
    time = pd.date_range("2000-01-01", periods=nt, freq="D")
    if seed is None:
        data = dka.zeros((nt, ncells), chunks=chunks, dtype="float32")
    else:
        data = dka.from_array(np.random.default_rng(seed).normal(size=(nt, ncells)).astype("float32"), chunks=chunks)
    return xr.DataArray(
        data,
        dims=("time", "ncells"),
        coords={"time": time, "lat": ("ncells", np.zeros(ncells)), "lon": ("ncells", np.zeros(ncells))},
        name="sst",
    )


def _grid(nt, nlat, nlon, chunks):
    time = pd.date_range("2000-01-01", periods=nt, freq="D")
    return xr.DataArray(
        dka.zeros((nt, nlat, nlon), chunks=chunks, dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": np.linspace(-89.5, 89.5, nlat), "lon": np.linspace(0.5, 359.5, nlon)},
        name="sst",
    )


def test_canonical_tile_rules(monkeypatch):
    """Table of the keep rule, on lazy fields (only sizes are read); the factor-1 row is the negative control."""
    failures = []

    def tiles_by_factor(da, dimensions, budget, **kw):
        tiles = {}
        with monkeypatch.context() as m:
            m.setattr(dims, "TASK_ELEMENTS", budget)
            for factor in (1, 4):
                m.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
                tiles[factor] = dims.canonical_time_chunks(da, dimensions, **kw)
        return tiles

    def icon_shaped_chunk_is_kept():
        # 25,000 cells x 7305 steps = 183M elements: inside 4 x 50M, so the tile is the chunk.
        assert dims.canonical_time_chunks(_field(7305, 100_000, (21, 25_000)), MESH_DIMS) == {"time": -1, "ncells": 25_000}

    def oversized_chunk_falls_back():
        # One 100,000-cell chunk costs 730M elements, above the 200M ceiling: the budget tile.
        assert dims.canonical_time_chunks(_field(7305, 200_000, (21, 100_000)), MESH_DIMS)["ncells"] == dims.TASK_ELEMENTS // 7305

    def small_chunk_never_widened():
        assert dims.canonical_time_chunks(_field(7305, 100_000, (21, 2_000)), MESH_DIMS)["ncells"] == 2_000

    def keep_factor_one_restores_the_budget_tile():
        # Negative control: with the factor at 1 the ICON-shaped chunk is split again.
        with monkeypatch.context() as m:
            m.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", 1)
            tile = dims.canonical_time_chunks(_field(7305, 100_000, (21, 25_000)), MESH_DIMS)["ncells"]
        assert tile == dims.TASK_ELEMENTS // 7305

    def single_cell_over_budget_never_raised():
        # The budget is not raised, so the single-cell fit warning still fires.
        with monkeypatch.context() as m:
            m.setattr(dims, "TASK_ELEMENTS", 600)
            assert dims._kept_input_tile(_field(1000, 100, (21, 2)), ["ncells"], 1000) is None

    def spatially_whole_field_never_kept():
        # 21,600 cells: 2.2x the budget, inside the keep band; several tiles keep the time-whole stages parallel.
        tiles = tiles_by_factor(_grid(365, 180, 120, (30, -1, -1)), GRID_DIMS, 365 * 10_000)
        assert tiles[4] == tiles[1]
        assert tiles[4]["lat"] * tiles[4]["lon"] < 180 * 120

    def keep_band_inclusive_at_the_ceiling():
        # 20 cells x 1000 steps = exactly 4 x the budget: still kept.
        with monkeypatch.context() as m:
            m.setattr(dims, "TASK_ELEMENTS", 5_000)
            assert dims.canonical_time_chunks(_field(1000, 100, (21, 20)), MESH_DIMS)["ncells"] == 20

    def sliver(da, dimensions):
        def row():
            # One chunk holding all but a sliver of the field is spatially whole in effect: never kept.
            tiles = tiles_by_factor(da, dimensions, 365 * 10_000)
            assert tiles[4] == tiles[1]

        return row

    def output_elements_count_against_the_band():
        # 25,000 x 100 read = 2.5x, x 200 written = 5x the budget: costed on the larger, so not kept.
        tiles = tiles_by_factor(
            _field(100, 100_000, (21, 25_000)), MESH_DIMS, 1_000_000, input_elements_per_cell=100, output_elements_per_cell=200
        )
        assert tiles[4] == tiles[1]
        assert tiles[4]["ncells"] < 25_000

    def empty_spatial_dim_never_kept():
        # A zero-length tiled dim has no tile to keep: the rule declines instead of dividing by zero.
        assert dims._kept_input_tile(_field(730, 0, (30, -1)), ["ncells"], 730) is None

    for case, row in {
        "icon-shaped-chunk-is-kept": icon_shaped_chunk_is_kept,
        "oversized-chunk-falls-back-to-the-budget-tile": oversized_chunk_falls_back,
        "small-chunk-is-never-widened": small_chunk_never_widened,
        "keep-factor-one-restores-the-budget-tile": keep_factor_one_restores_the_budget_tile,
        "single-cell-over-budget-is-never-raised": single_cell_over_budget_never_raised,
        "spatially-whole-field-is-never-kept": spatially_whole_field_never_kept,
        "keep-band-is-inclusive-at-the-ceiling": keep_band_inclusive_at_the_ceiling,
        "mesh-one-cell-sliver": sliver(_field(365, 20_000, (30, (19_999, 1))), MESH_DIMS),
        "grid-one-row-sliver": sliver(_grid(365, 180, 120, (30, (179, 1), -1)), GRID_DIMS),
        "output-elements-count-against-the-keep-band": output_elements_count_against_the_band,
        "an-empty-spatial-dim-is-never-kept": empty_spatial_dim_never_kept,
    }.items():
        check(failures, case, row)
    assert_no_failures(failures)


def _graph_size(da):
    return len(dict(da.data.__dask_graph__()))


def test_kept_tile_shrinks_the_graph(monkeypatch):
    """Chunk STRUCTURE, not values: the whole anomaly graph is no larger (mesh: much smaller) when the tile is the input chunk."""
    failures = []

    def mesh_kept_tile():
        with monkeypatch.context() as m:
            m.setattr(dims, "TASK_ELEMENTS", 1460 * 6_800)
            da = _field(1460, 100_000, (21, 25_000))
            sizes = {}
            for factor in (1, 4):
                m.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
                sizes[factor] = _graph_size(
                    compute_normalised_anomaly(da, "detrend_fixed_baseline", MESH_DIMS, MESH_COORDS).dat_anomaly
                )
        assert sizes[4] < 0.6 * sizes[1], sizes

    def two_tiled_dims():
        # Lat and lon tiled together: the tile is the input chunk on BOTH axes. Raising only the budget
        # let the greedy tiler pick a side that straddles the input chunks.
        with monkeypatch.context() as m:
            m.setattr(dims, "TASK_ELEMENTS", 365 * 1_500)
            da = _grid(365, 180, 240, (30, 60, 40))
            sizes = {}
            for factor in (1, 4):
                m.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
                tile = dims.canonical_time_chunks(da, GRID_DIMS)
                if factor == 4:
                    assert (tile["lat"], tile["lon"]) == (60, 40), tile
                sizes[factor] = _graph_size(
                    compute_normalised_anomaly(da, "detrend_fixed_baseline", GRID_DIMS, GRID_DIMS).dat_anomaly
                )
        assert sizes[4] <= sizes[1], sizes

    def irregular_chunks():
        # An index-selected mesh: the tile is the largest chunk. The rechunk splits chunks that straddle
        # tile boundaries, so it is not a pure merge, but it still costs no more tasks than the budget tile.
        with monkeypatch.context() as m:
            m.setattr(dims, "TASK_ELEMENTS", 730 * 400)
            da = _field(730, 2_000, (30, (300, 500, 200, 500, 500)), seed=7)
            sizes = {}
            for factor in (1, 4):
                m.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
                assert dims.canonical_time_chunks(da, MESH_DIMS)["ncells"] == (400 if factor == 1 else 500)
                sizes[factor] = _graph_size(
                    compute_normalised_anomaly(da, "detrend_fixed_baseline", MESH_DIMS, MESH_COORDS).dat_anomaly
                )
        assert sizes[4] <= sizes[1], sizes

    for case, row in {
        "mesh-kept-tile": mesh_kept_tile,
        "two-tiled-dims": two_tiled_dims,
        "irregular-chunks": irregular_chunks,
    }.items():
        check(failures, case, row)
    assert_no_failures(failures)


def test_values_do_not_depend_on_the_kept_tile(monkeypatch):
    """Table: the kept tile (factor 4) against the budget tile (factor 1), bit for bit.

    Rows: detrend_harmonic and detrend_fixed_baseline (tile 500 vs 300), irregular chunks (500 vs
    400), and shifting_baseline, the ICON notebook's method, where a spy on the keep rule proves the
    rule fired inside the anomaly graph in one arm only.
    """
    failures = []

    def by_factor(da, budget, method, expected_tiles=None, **kw):
        outs, raised = {}, {}
        real_keep = dims._kept_input_tile
        with monkeypatch.context() as m:
            m.setattr(dims, "TASK_ELEMENTS", budget)
            for factor in (1, 4):
                m.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
                if expected_tiles is not None:
                    assert dims.canonical_time_chunks(da, MESH_DIMS)["ncells"] == expected_tiles[factor]
                seen = []
                m.setattr(dims, "_kept_input_tile", lambda *a, _s=seen: _s.append(real_keep(*a)) or _s[-1])
                outs[factor] = compute_normalised_anomaly(da, method, MESH_DIMS, MESH_COORDS, **kw).dat_anomaly.compute()
                raised[factor] = any(t is not None for t in seen)
        return outs, raised

    def tiled(method):
        def row():
            outs, _ = by_factor(_field(730, 2_000, (30, 500), seed=3), 730 * 300, method, {1: 300, 4: 500})
            assert np.array_equal(outs[1].values, outs[4].values), "values moved with the kept tile"

        return row

    def irregular():
        outs, _ = by_factor(
            _field(730, 2_000, (30, (300, 500, 200, 500, 500)), seed=7), 730 * 400, "detrend_fixed_baseline", {1: 400, 4: 500}
        )
        assert np.array_equal(outs[1].values, outs[4].values), "values moved with the kept tile"

    def shifting():
        nt = 4 * 365
        outs, raised = by_factor(
            _field(nt, 2_000, (30, 500), seed=5), nt * 300, "shifting_baseline", window_years=2, smooth_days=11
        )
        assert raised == {1: False, 4: True}, raised
        assert np.array_equal(outs[1].values, outs[4].values, equal_nan=True), "values moved with the kept tile"

    for case, row in {
        "detrend_harmonic": tiled("detrend_harmonic"),
        "detrend_fixed_baseline": tiled("detrend_fixed_baseline"),
        "irregular-chunks": irregular,
        "shifting_baseline-with-keep-spy": shifting,
    }.items():
        check(failures, case, row)
    assert_no_failures(failures)


# ── finalise: tuple time chunks and the cycle axis ────────────────────


def _seasonal_ds(n_time=10):
    time = xr.date_range("2001-01-01", periods=n_time, freq="D")
    return xr.Dataset(
        {
            "extreme_events": (("time", "lat", "lon"), np.zeros((n_time, 3, 4), dtype=bool)),
            "thresholds": (("dayofyear", "lat", "lon"), np.zeros((366, 3, 4), dtype=np.float32)),
        },
        coords={"time": time, "dayofyear": np.arange(1, 367), "lat": np.arange(3.0), "lon": np.arange(4.0)},
    ).chunk({"time": 1})


def test_finalise_time_chunks_on_the_cycle_axis():
    """A tuple time entry lists block sizes that sum to the time length, so it cannot be applied to the
    366-entry cycle axis: time takes the tuple as given, the cycle axis takes the largest block as its
    step. An integer still drives both."""
    failures = []
    rows = {
        "tuple-(4, 6)": ((4, 6), (4, 6), (6,) * 61),
        "tuple-(3, 3, 4)": ((3, 3, 4), (3, 3, 4), (4,) * 91 + (2,)),
        "list-[4, 6]": ([4, 6], (4, 6), (6,) * 61),
        "integer-5": (5, (5, 5), (5,) * 73 + (1,)),
    }
    for case, (time_chunks, want_time, want_cycle) in rows.items():

        def row(time_chunks=time_chunks, want_time=want_time, want_cycle=want_cycle):
            out = finalise_dataset(_seasonal_ds(), GRID_DIMS, GRID_DIMS, {"time": time_chunks}, SimpleNamespace(mode="lazy"))
            assert out.extreme_events.chunks == (want_time, (3,), (4,)), out.extreme_events.chunks
            assert out.thresholds.chunks == (want_cycle, (3,), (4,)), out.thresholds.chunks[0]

        check(failures, case, row)
    assert_no_failures(failures)


# ── the seasonal threshold block ──────────────────────────────────────
#
# On a lat/lon grid the input is spatially whole, so aligning the thresholds to the input's spatial
# chunks alone left the cycle axis whole too: one block of 366 x 720 x 1440 x 4 B = 1.5 GB at 0.25 deg.
# The groupby comparison indexes that block once per run of consecutive slots, every such task holding
# a copy, and a full-resolution global streaming run died on it. Fixture fields are far too small to
# reach the budget, so the budget is shrunk here instead; the block bound is pinned at two field sizes,
# because a bound that only holds at one size says nothing about how it scales.


def _block_field(n_y, n_x, n_years=4, seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2000-01-01", periods=365 * n_years, freq="D")
    doy = time.dayofyear.values[:, None, None]
    data = (np.sin(2 * np.pi * doy / 365.25) + rng.normal(0.0, 0.5, size=(time.size, n_y, n_x))).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": np.linspace(-10, 10, n_y), "lon": np.linspace(0, 20, n_x)},
        name="sst",
    )
    return da.chunk({"time": 30, "lat": -1, "lon": -1})


def _block_kwargs(method_percentile):
    return {
        "method_anomaly": "shifting_baseline",
        "method_extreme": "seasonal_percentile",
        "method_percentile": method_percentile,
        "window_years": 2,
        "smooth_days": 21,
        "threshold_percentile": 95,
        "dimensions": GRID_DIMS,
        "dask_chunks": {"time": 30},
    }


@pytest.mark.parametrize("method_percentile", ["approximate", "exact"])
def test_seasonal_threshold_block_is_bounded(monkeypatch, method_percentile):
    """The threshold anchored before the comparison stays inside the budget at two field sizes (6 x 8 holds
    20 slots, 12 x 16 holds 5), and bounding the cycle axis leaves every output alone (the bounded 6 x 8
    runs, at 20 and at 7 slots per block, against an unbounded reference)."""
    da = _block_field(6, 8)
    reference = marEx.preprocess_data(da, **_block_kwargs(method_percentile)).compute()

    budget = 20 * 6 * 8
    monkeypatch.setattr(seasonal_percentile, "TASK_ELEMENTS", budget, raising=False)
    seen = []
    original = Materialiser.stage

    def spy(self, obj, label, *args, **kwargs):
        if label == "thresholds":
            seen.append(dict(zip(obj.dims, obj.chunks)))
        return original(self, obj, label, *args, **kwargs)

    monkeypatch.setattr(Materialiser, "stage", spy)

    for n_y, n_x in [(6, 8), (12, 16)]:
        seen.clear()
        field = da if (n_y, n_x) == (6, 8) else _block_field(n_y, n_x)
        result = marEx.preprocess_data(field, **_block_kwargs(method_percentile))
        assert seen, f"{n_y}x{n_x}: the comparison's threshold was never staged"
        for chunks in seen:
            assert chunks["lat"] == (n_y,) and chunks["lon"] == (n_x,), chunks
            block = max(chunks["dayofyear"]) * n_y * n_x
            assert block <= budget, f"{n_y}x{n_x}: threshold block {block} elements > budget {budget}: {chunks}"
        if (n_y, n_x) == (6, 8):
            bounded = result.compute()
            for var in ("extreme_events", "thresholds", "dat_anomaly", "mask"):
                xr.testing.assert_identical(bounded[var], reference[var])

    # A tighter block, 7 slots, splits the cycle axis into many more pieces; the values still may not move.
    budget = 7 * 6 * 8
    monkeypatch.setattr(seasonal_percentile, "TASK_ELEMENTS", budget, raising=False)
    seen.clear()
    bounded = marEx.preprocess_data(da, **_block_kwargs(method_percentile)).compute()
    assert seen, "6x8 at 7 slots: the comparison's threshold was never staged"
    for chunks in seen:
        assert max(chunks["dayofyear"]) <= 7 and len(chunks["dayofyear"]) > 1, f"6x8 at 7 slots: {chunks}"
    for var in ("extreme_events", "thresholds", "dat_anomaly", "mask"):
        xr.testing.assert_identical(bounded[var], reference[var])
