"""The canonical tile keeps an input spatial chunk the caller already sized.

Splitting a 25,000-cell chunk into ~6.8k-cell tiles and merging it back multiplies the task count
without lowering a memory figure the caller did not accept. The tile is the input chunk while one
task on it stays within ``KEEP_INPUT_CHUNK_FACTOR`` times ``TASK_ELEMENTS``; beyond that the budget
tile applies as before. Values never move with the spatial layout.
"""

import dask.array as dka
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from marEx.anomaly.base import compute_normalised_anomaly
from marEx.core import dimensions as dims

DIMENSIONS = {"time": "time", "x": "ncells"}
COORDINATES = {"time": "time", "x": "lon", "y": "lat"}


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


def test_icon_shaped_chunk_is_kept():
    """25,000 cells x 7305 steps = 183M elements: inside 4 x 50M, so the tile is the chunk."""
    da = _field(7305, 100_000, (21, 25_000))
    assert dims.canonical_time_chunks(da, DIMENSIONS) == {"time": -1, "ncells": 25_000}


def test_oversized_chunk_falls_back_to_the_budget_tile():
    """One 100,000-cell chunk costs 730M elements, above the 200M ceiling: the old budget tile."""
    da = _field(7305, 200_000, (21, 100_000))
    tile = dims.canonical_time_chunks(da, DIMENSIONS)["ncells"]
    assert tile == dims.TASK_ELEMENTS // 7305


def test_small_chunk_is_never_widened():
    da = _field(7305, 100_000, (21, 2_000))
    assert dims.canonical_time_chunks(da, DIMENSIONS)["ncells"] == 2_000


def test_keep_factor_one_restores_the_budget_tile(monkeypatch):
    """Negative control: with the factor at 1 the ICON-shaped chunk is split again."""
    monkeypatch.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", 1)
    da = _field(7305, 100_000, (21, 25_000))
    assert dims.canonical_time_chunks(da, DIMENSIONS)["ncells"] == dims.TASK_ELEMENTS // 7305


def test_kept_tile_cuts_the_anomaly_graph(monkeypatch):
    """Chunk STRUCTURE, not values: the whole anomaly graph is smaller when the tile is the chunk."""
    monkeypatch.setattr(dims, "TASK_ELEMENTS", 1460 * 6_800)
    da = _field(1460, 100_000, (21, 25_000))
    sizes = {}
    for factor in (1, 4):
        monkeypatch.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
        out = compute_normalised_anomaly(da, "detrend_fixed_baseline", DIMENSIONS, COORDINATES).dat_anomaly
        sizes[factor] = len(dict(out.data.__dask_graph__()))
    assert sizes[4] < 0.6 * sizes[1], sizes


@pytest.mark.parametrize("method", ["detrend_harmonic", "detrend_fixed_baseline"])
def test_values_do_not_depend_on_the_kept_tile(monkeypatch, method):
    """Tile 500 (kept) against tile 300 (budget) must agree bit for bit."""
    monkeypatch.setattr(dims, "TASK_ELEMENTS", 730 * 300)
    da = _field(730, 2_000, (30, 500), seed=3)
    outs = {}
    for factor in (1, 4):
        monkeypatch.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
        tile = dims.canonical_time_chunks(da, DIMENSIONS)["ncells"]
        assert tile == (300 if factor == 1 else 500)
        outs[factor] = compute_normalised_anomaly(da, method, DIMENSIONS, COORDINATES).dat_anomaly.compute()
    assert np.array_equal(outs[1].values, outs[4].values)


def test_shifting_baseline_values_do_not_depend_on_the_kept_tile(monkeypatch):
    """The ICON notebook's method: tile 500 (kept) against tile 300 (budget), bit for bit.

    A spy on the keep rule proves the keep rule fired inside the anomaly graph in one arm only.
    """
    nt = 4 * 365
    monkeypatch.setattr(dims, "TASK_ELEMENTS", nt * 300)
    da = _field(nt, 2_000, (30, 500), seed=5)
    real_keep = dims._kept_input_tile
    outs, raised = {}, {}
    for factor in (1, 4):
        monkeypatch.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
        seen = []
        monkeypatch.setattr(dims, "_kept_input_tile", lambda *a, _s=seen: _s.append(real_keep(*a)) or _s[-1])
        outs[factor] = compute_normalised_anomaly(
            da, "shifting_baseline", DIMENSIONS, COORDINATES, window_years=2, smooth_days=11
        ).dat_anomaly.compute()
        raised[factor] = any(t is not None for t in seen)
    assert raised == {1: False, 4: True}, raised
    assert np.array_equal(outs[1].values, outs[4].values, equal_nan=True)


def test_single_cell_over_budget_is_never_raised(monkeypatch):
    """One cell above the budget: the budget is not raised, so the single-cell fit warning still fires."""
    monkeypatch.setattr(dims, "TASK_ELEMENTS", 600)
    da = _field(1000, 100, (21, 2))
    assert dims._kept_input_tile(da, ["ncells"], 1000) is None


def _grid(nt, nlat, nlon, chunks):
    time = pd.date_range("2000-01-01", periods=nt, freq="D")
    return xr.DataArray(
        dka.zeros((nt, nlat, nlon), chunks=chunks, dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": np.linspace(-89.5, 89.5, nlat), "lon": np.linspace(0.5, 359.5, nlon)},
        name="sst",
    )


GRID_DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}


def test_two_tiled_dims_keep_the_exact_input_chunk(monkeypatch):
    """Lat and lon tiled together: the tile is the input chunk on BOTH axes, and the graph does not grow.

    Raising only the budget let the greedy tiler pick a side that straddles the input chunks.
    """
    monkeypatch.setattr(dims, "TASK_ELEMENTS", 365 * 1_500)
    da = _grid(365, 180, 240, (30, 60, 40))
    sizes = {}
    for factor in (1, 4):
        monkeypatch.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
        tile = dims.canonical_time_chunks(da, GRID_DIMENSIONS)
        if factor == 4:
            assert (tile["lat"], tile["lon"]) == (60, 40), tile
        out = compute_normalised_anomaly(da, "detrend_fixed_baseline", GRID_DIMENSIONS, GRID_DIMENSIONS).dat_anomaly
        sizes[factor] = len(dict(out.data.__dask_graph__()))
    assert sizes[4] <= sizes[1], sizes


def test_spatially_whole_field_is_never_kept(monkeypatch):
    """A lat/lon field left whole in space keeps several tiles, so the time-whole stages stay parallel."""
    monkeypatch.setattr(dims, "TASK_ELEMENTS", 365 * 10_000)
    da = _grid(365, 180, 120, (30, -1, -1))  # 21,600 cells: 2.2x the budget, inside the keep band
    tiles = {}
    for factor in (1, 4):
        monkeypatch.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
        tiles[factor] = dims.canonical_time_chunks(da, GRID_DIMENSIONS)
    assert tiles[4] == tiles[1]
    assert tiles[4]["lat"] * tiles[4]["lon"] < 180 * 120


def test_keep_band_is_inclusive_at_the_ceiling(monkeypatch):
    """A chunk costing exactly KEEP_INPUT_CHUNK_FACTOR x TASK_ELEMENTS is still kept."""
    monkeypatch.setattr(dims, "TASK_ELEMENTS", 5_000)
    da = _field(1000, 100, (21, 20))  # 20 cells x 1000 steps = 4 x the budget
    assert dims.canonical_time_chunks(da, DIMENSIONS)["ncells"] == 20


def test_irregular_chunks_keep_the_largest_chunk(monkeypatch):
    """Irregular input (an index-selected mesh): the tile is the largest chunk, the graph does not grow and values agree.

    The rechunk then splits the chunks that straddle tile boundaries, so it is not a pure merge, but it
    still costs fewer tasks than the budget tile.
    """
    monkeypatch.setattr(dims, "TASK_ELEMENTS", 730 * 400)
    da = _field(730, 2_000, (30, (300, 500, 200, 500, 500)), seed=7)
    outs, sizes = {}, {}
    for factor in (1, 4):
        monkeypatch.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
        assert dims.canonical_time_chunks(da, DIMENSIONS)["ncells"] == (400 if factor == 1 else 500)
        out = compute_normalised_anomaly(da, "detrend_fixed_baseline", DIMENSIONS, COORDINATES).dat_anomaly
        sizes[factor] = len(dict(out.data.__dask_graph__()))
        outs[factor] = out.compute()
    assert sizes[4] <= sizes[1], sizes
    assert np.array_equal(outs[1].values, outs[4].values)


@pytest.mark.parametrize(
    "da, dimensions",
    [
        (_field(365, 20_000, (30, (19_999, 1))), DIMENSIONS),
        (_grid(365, 180, 120, (30, (179, 1), -1)), GRID_DIMENSIONS),
    ],
    ids=["mesh-one-cell-sliver", "grid-one-row-sliver"],
)
def test_a_sliver_does_not_pass_as_a_split_field(monkeypatch, da, dimensions):
    """One chunk holding all but a sliver of the field is spatially whole in effect: never kept."""
    monkeypatch.setattr(dims, "TASK_ELEMENTS", 365 * 10_000)
    tiles = {}
    for factor in (1, 4):
        monkeypatch.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
        tiles[factor] = dims.canonical_time_chunks(da, dimensions)
    assert tiles[4] == tiles[1]


def test_output_elements_count_against_the_keep_band(monkeypatch):
    """A caller writing more per cell than it reads is costed on the larger of the two."""
    monkeypatch.setattr(dims, "TASK_ELEMENTS", 1_000_000)
    da = _field(100, 100_000, (21, 25_000))  # 25,000 x 100 read = 2.5x, x 200 written = 5x the budget
    tiles = {}
    for factor in (1, 4):
        monkeypatch.setattr(dims, "KEEP_INPUT_CHUNK_FACTOR", factor)
        tiles[factor] = dims.canonical_time_chunks(da, DIMENSIONS, input_elements_per_cell=100, output_elements_per_cell=200)
    assert tiles[4] == tiles[1]
    assert tiles[4]["ncells"] < 25_000


def test_an_empty_spatial_dim_is_never_kept():
    """A zero-length tiled dim has no tile to keep: the rule declines instead of dividing by zero."""
    assert dims._kept_input_tile(_field(730, 0, (30, -1)), ["ncells"], 730) is None
