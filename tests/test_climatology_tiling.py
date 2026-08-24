"""Spatial tiling of the rolling climatology: bounded tasks, unchanged values.

``rolling_climatology`` builds a long-form array roughly ``window_years`` times
the input along time and flox-reduces it to ``(target_year, dayofyear, *spatial)``
with the cycle axis forced whole. Neither side had an element budget, so a field
left spatially whole made one task the entire array -- and an extra dimension such
as depth multiplied that directly.

Two properties are pinned here, because a value-only test would have passed while
the tiling was broken (CLAUDE.md records exactly that: 440 tests, the window
harness and the coverage tripwires were all green while an all-to-all rechunk was
live):

1. **structure** -- the tile bounds the per-task working set, and the bound is
   invariant in the length of the extra dimension rather than a fixed byte count;
2. **values** -- tiling is a pure rechunk of spatial dims, so the climatology is
   bit-identical to the untiled one.
"""

import numpy as np
import pytest
import xarray as xr
from dask.array import zeros as da_zeros

import marEx.core.dimensions as D
from marEx.anomaly.climatology import rolling_climatology
from marEx.core.dimensions import tile_spatial_chunks

WINDOW_YEARS = 2


def _synthetic(nt=4 * 365, nlat=12, nlon=16, ndepth=None, seed=0):
    """Daily gridded series, optionally with a depth axis."""
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


def _working_set(tile, sizes, n_long, output_per_cell):
    """Elements one task touches: the larger of what it reads and what it writes."""
    cells = 1
    for dim, size in sizes.items():
        cells *= tile.get(dim, size)
    return cells * max(n_long, output_per_cell)


class TestTileIsRankAgnostic:
    """The per-task bound must not grow with the extra dimension's length."""

    def test_working_set_invariant_in_depth_length(self):
        # ICON-scale arithmetic on lazy zeros: 8 years daily reduced over a 15-year
        # window, on a 14.9M-cell mesh. Nothing is computed; only the tile is.
        n_long = 2922 * 15
        output_per_cell = 8 * 366
        budget = D.TASK_ELEMENTS
        ncells = 14_900_000

        working_sets = {}
        for ndepth in (1, 4, 25, 50):
            da = xr.DataArray(
                da_zeros((ndepth, ncells)),
                dims=("depth", "ncells"),
            ).chunk({"depth": -1, "ncells": -1})

            tile = tile_spatial_chunks(
                da,
                ("depth", "ncells"),
                input_elements_per_cell=n_long,
                output_elements_per_cell=output_per_cell,
            )
            working = _working_set(tile, dict(da.sizes), n_long, output_per_cell)
            assert working <= budget, f"depth={ndepth}: task touches {working} > budget {budget}"
            working_sets[ndepth] = working

        # The whole point: lengthening the extra dimension 50x must not grow the
        # per-task working set. Assert invariance, not an absolute byte bound -- a
        # bound loose enough for depth=1 would pass while depth=50 allocated 50x.
        spread = max(working_sets.values()) / min(working_sets.values())
        assert spread < 2.0, f"per-task working set varies {spread:.1f}x across depth lengths: {working_sets}"

        # And the budget is actually being spent, not silently under-used.
        assert min(working_sets.values()) > budget / 4, f"tiles far below budget: {working_sets}"

    def test_tile_only_ever_shrinks_existing_chunks(self):
        """The budget is a cap, never a target: a finer chunking is left alone."""
        da = _synthetic(nt=10, nlat=64, nlon=64).chunk({"time": -1, "lat": 4, "lon": 4})
        tile = tile_spatial_chunks(da, ("lat", "lon"), input_elements_per_cell=1, output_elements_per_cell=1)
        assert tile["lat"] == 4 and tile["lon"] == 4, f"cap raised the caller's chunks: {tile}"

    def test_tile_respects_the_horizontal_window_floor_but_not_extra_dims(self):
        da = _synthetic(nt=10, nlat=64, nlon=64, ndepth=8).chunk({"time": -1, "depth": -1, "lat": -1, "lon": -1})
        tile = tile_spatial_chunks(
            da,
            ("lat", "lon", "depth"),
            input_elements_per_cell=10_000_000,
            output_elements_per_cell=1,
            floor_dims=("lat", "lon"),
            floor=5,
        )
        assert tile["lat"] >= 5 and tile["lon"] >= 5, tile
        # depth carries no rolling window, so it is not widened to the floor
        assert tile["depth"] < 5, tile


class TestClimatologyTilingEquivalence:
    """Tiling changes task granularity, never values."""

    @pytest.mark.parametrize("ndepth", [None, 3])
    def test_tiled_climatology_is_bit_identical(self, ndepth, monkeypatch):
        da = _synthetic(ndepth=ndepth)
        chunks = {"time": 30, "lat": -1, "lon": -1}
        if ndepth is not None:
            chunks["depth"] = -1
        da = da.chunk(chunks)
        dims = {"time": "time", "x": "lon", "y": "lat"}
        coords = {"time": "time", "x": "lon", "y": "lat"}

        untiled = rolling_climatology(da, WINDOW_YEARS, dims, coords).compute()

        # Force a budget small enough that the tiling actually bites on this fixture.
        monkeypatch.setattr(D, "TASK_ELEMENTS", 200_000)
        tiled_chunks = tile_spatial_chunks(
            da,
            [d for d in da.dims if d != "time"],
            input_elements_per_cell=len(da.time) * WINDOW_YEARS,
            output_elements_per_cell=4 * 366,
        )
        assert any(
            tiled_chunks[d] < da.sizes[d] for d in tiled_chunks
        ), f"budget did not force a tiling; test would prove nothing: {tiled_chunks}"

        tiled = rolling_climatology(da, WINDOW_YEARS, dims, coords).compute()

        np.testing.assert_array_equal(
            untiled.values,
            tiled.values,
            err_msg="spatial tiling moved the climatology",
        )

    def test_returned_chunking_is_the_callers_own(self, monkeypatch):
        """The tile must not leak into the returned layout."""
        da = _synthetic(ndepth=3).chunk({"time": 30, "depth": -1, "lat": -1, "lon": -1})
        dims = {"time": "time", "x": "lon", "y": "lat"}
        coords = {"time": "time", "x": "lon", "y": "lat"}

        monkeypatch.setattr(D, "TASK_ELEMENTS", 200_000)
        result = rolling_climatology(da, WINDOW_YEARS, dims, coords)

        assert result.chunksizes["lat"] == da.chunksizes["lat"]
        assert result.chunksizes["lon"] == da.chunksizes["lon"]
        assert result.chunksizes["depth"] == da.chunksizes["depth"]
