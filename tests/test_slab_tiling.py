"""The slab engine's tile is capped in cells, so its task count grows with the mesh.

A slab task is cheap per cell but holds a cell's whole record, so the element budget alone gave a
0.5 M-cell mesh 4 tasks and a 14.9 M-cell one 109. The tile is now also capped at
``_SLAB_TILE_CELLS`` cells, which bounds a task's wall instead of leaving whole nodes idle, and it
is cut from the input's own spatial chunks (a divisor or a multiple of them) so the transpose
never makes a tile straddle two input chunks. Structure is asserted here; values (the slab result
under any tiling equals the dense engine's) are asserted with a cap small enough to cut the
fixtures into many tiles.
"""

import dask.array
import numpy as np
import pytest
import xarray as xr

from marEx.core.time_axis import DAILY_CYCLE
from marEx.extremes import histogram as H

from .test_slab_engine import GRID_DIMS, MESH_DIMS, _both, _edges_and_centres, _field

DEPTH_DIMS = {"time": "time", "x": "ncells"}


def _lazy(shape, chunks, dims):
    data = dask.array.zeros(shape, chunks=chunks, dtype=np.float32)
    slots = (np.arange(shape[0]) % 365 + 1).astype(np.int64)
    return xr.DataArray(data, dims=dims, coords={"dayofyear": ("time", slots)})


def _threshold_blocks(da, dims, window_spatial=None):
    raw = H._slab_seasonal_threshold(
        da,
        *_edges_and_centres(),
        dimensions=dims,
        cycle=DAILY_CYCLE,
        window_steps=11,
        window_spatial=window_spatial,
        q=0.95,
        q_mirror=0.05,
        tail="upper",
    ).data
    return raw.chunks[:-1]


def _boundaries(chunks):
    return set(np.cumsum(chunks).tolist())


@pytest.mark.parametrize("ncells", [1_000_000, 4_000_000])
def test_mesh_tile_count_scales_with_the_mesh(ncells):
    # 0.5 M cells used to be 4 tasks; the count must now be linear in the mesh, not fixed by the budget.
    da = _lazy((365, ncells), (21, 25_000), ("time", "ncells"))
    (tiles,) = _threshold_blocks(da, MESH_DIMS)
    assert max(tiles) <= H._SLAB_TILE_CELLS
    assert len(tiles) == ncells // max(tiles)


def test_mesh_tile_count_ratio_is_the_mesh_ratio():
    small = _threshold_blocks(_lazy((365, 1_000_000), (21, 25_000), ("time", "ncells")), MESH_DIMS)[0]
    large = _threshold_blocks(_lazy((365, 4_000_000), (21, 25_000), ("time", "ncells")), MESH_DIMS)[0]
    assert len(large) == 4 * len(small)


@pytest.mark.parametrize("in_chunk", [25_000, 1_000, 1_536, 600])
def test_mesh_tiles_never_straddle_an_input_chunk(in_chunk):
    da = _lazy((365, 300_000), (21, in_chunk), ("time", "ncells"))
    (tiles,) = _threshold_blocks(da, MESH_DIMS)
    inputs = _boundaries(da.chunks[1])
    outputs = _boundaries(tiles)
    # Either every input boundary is a tile boundary (tiles split chunks) or the reverse (tiles merge them).
    assert inputs <= outputs or outputs <= inputs
    assert max(tiles) <= H._SLAB_TILE_CELLS


def test_gridded_tile_is_capped_and_aligned():
    # A spatially whole 0.25 deg globe: 720 x 1440 in one chunk is split into tiles that divide it.
    da = _lazy((365, 720, 1440), (21, 720, 1440), ("time", "lat", "lon"))
    lat, lon = _threshold_blocks(da, GRID_DIMS)
    assert max(lat) * max(lon) <= H._SLAB_TILE_CELLS
    assert 720 % max(lat) == 0 and 1440 % max(lon) == 0
    assert len(lat) * len(lon) >= 720 * 1440 // H._SLAB_TILE_CELLS


def test_extra_dim_does_not_restack_into_a_starved_tile():
    # A small mesh with many levels: the leftover budget must stack levels only up to the cell cap.
    da = _lazy((365, 50, 1_000), (21, 1, 1_000), ("time", "depth", "ncells"))
    chunks = H._slab_tile_chunks(da, DEPTH_DIMS, None, 366)
    assert chunks["ncells"] * chunks["depth"] <= H._SLAB_TILE_CELLS
    assert chunks["ncells"] == 1_000


@pytest.mark.parametrize("window_spatial", [None, 3, 5])
@pytest.mark.parametrize("tail, q", [("upper", 0.95), ("lower", 0.05)])
def test_gridded_values_unchanged_under_a_small_cap(monkeypatch, window_spatial, tail, q):
    # A cap of 4 cuts the 6 x 7 fixture into tiles narrower than the halo and across the x wrap.
    monkeypatch.setattr(H, "_SLAB_TILE_CELLS", 4)
    da, cycle = _field(shape=(6, 7))
    tiles = H._slab_tile_chunks(da, GRID_DIMS, window_spatial, cycle.length)
    assert tiles["lat"] < 6 and tiles["lon"] < 7
    dense, slab = _both(da, cycle, GRID_DIMS, 30, q=q, tail=tail, window_spatial=window_spatial)
    xr.testing.assert_identical(dense, slab)


@pytest.mark.parametrize("spatial_chunk", [-1, 21, 6])
def test_mesh_values_unchanged_under_a_small_cap(monkeypatch, spatial_chunk):
    monkeypatch.setattr(H, "_SLAB_TILE_CELLS", 8)
    da, cycle = _field(mesh=True)
    da = da.chunk({"ncells": spatial_chunk})
    dense, slab = _both(da, cycle, MESH_DIMS, 30, q=0.95, tail="upper")
    xr.testing.assert_identical(dense, slab)
    assert H._slab_tile_chunks(da, MESH_DIMS, None, cycle.length)["ncells"] <= 8
