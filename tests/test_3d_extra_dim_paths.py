"""Extra-dimension paths that ``test_3d_preprocessing.py`` does not reach.

That file pins the gridded, persist, approximate path: slice equivalence by value, and
the day-of-year histogram tile as invariant in the depth length (D-127). Three paths
were left open, and each is pinned here the same two ways -- by VALUE against the
per-level run, and by CHUNK STRUCTURE, because every tiling is value-identical and a
value check alone passes while a tile quietly grows or shatters with depth:

- ``compute_mode="streaming"`` on a 3-D field (staging has never carried an extra dim);
- an unstructured mesh with depth, ``(time, depth, ncells)``;
- ``_chunk_spatial_for_histogram``, the tiler behind the 1-D histogram (the DEFAULT
  ``global_percentile`` path) and the exact seasonal path. It used to take the budget's
  root over every non-time dim, so a depth-1 field on a 720x1440 grid got 19x19 tiles
  (2888 tasks) where its 2-D slice got 85x85 (153), and a depth-3 field got the
  multi-level (3, 19, 19) tile that D-127 found costly on the approximate path. It now
  tiles per level, like ``_histogram_tile_chunks``;
- the other sites that sized an extra dim like a horizontal one: ``global_percentile``'s
  exact rechunk on a mesh, and the output layout (``finalise``, harmonic ``dat_stn`` /
  ``STD``), which held depth whole.

Every site follows one rule, :func:`marEx.core.dimensions.extra_dim_chunks` (D-128): one
level per chunk while the horizontal tile is smaller than the whole slice, and levels
stacked with the leftover budget once a whole slice fits. The crops below fit in one
tile at the shipped budgets, which is the STACKING regime; the per-level tests shrink the
budget so the crop is tiled, which is what a real field looks like.

The fixtures are crops of the shipped zarrs (10x20 cells gridded, 405 cells mesh,
4 years) so each test runs serially on a 16 GB node. The per-level legs are handed the
3-D run's own bin geometry, for the reason ``test_3d_preprocessing.py`` gives.
"""

from pathlib import Path

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.core.compute_mode import clear_staging
from marEx.core.dimensions import horizontal_dims

from .test_compute_mode_equivalence import _assert_identical

DATA_DIR = Path(__file__).parent / "data"

GRIDDED_DIMS = {"time": "time", "x": "lon", "y": "lat"}
MESH_DIMS = {"time": "time", "x": "ncells"}
MESH_COORDS = {"time": "time", "x": "lon", "y": "lat"}
TIME_CHUNK = 30
NYEARS = 4

# Distinct offset and scale per level, so a reduction over depth (or a level borrowed
# from another) cannot pass unnoticed.
LEVEL_TRANSFORMS = [(0.0, 1.0), (-1.5, 0.8), (2.25, 1.3)]
REAL_LEVELS = tuple(range(len(LEVEL_TRANSFORMS)))
DEPTH_LENGTHS = (1, 4, 25, 50)
# A budget small enough that the 10x20 crop is tiled (~5x5 per tile at ~730 anomaly steps).
SMALL_BUDGET = 20_000


def _stack_levels(base, spatial):
    levels = [base * scale + offset for offset, scale in LEVEL_TRANSFORMS]
    da = xr.concat(levels, dim="depth").assign_coords(depth=np.arange(len(levels), dtype=np.float32))
    da = da.transpose("time", "depth", *spatial)
    da.name = "to"
    return da.chunk({"time": TIME_CHUNK, "depth": -1, **{d: -1 for d in spatial}})


@pytest.fixture(scope="module")
def gridded_3d():
    base = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to
    base = base.isel(time=slice(0, NYEARS * 365), lat=slice(0, 10), lon=slice(0, 20))
    return _stack_levels(base, ("lat", "lon"))


@pytest.fixture(scope="module")
def mesh_3d():
    base = xr.open_zarr(str(DATA_DIR / "sst_unstructured.zarr"), chunks={}).to.isel(time=slice(0, NYEARS * 365))
    n = base.sizes["ncells"]
    base = base.assign_coords(lat=("ncells", np.linspace(-60, 60, n)), lon=("ncells", np.linspace(-180, 180, n)))
    return _stack_levels(base, ("ncells",))


def _kwargs(method_extreme, dimensions, method_percentile="approximate", **extra):
    kw = {
        "method_anomaly": "shifting_baseline",
        "method_extreme": method_extreme,
        "method_percentile": method_percentile,
        "window_years": 2,
        "smooth_days": 21,
        "threshold_percentile": 95,
        "dimensions": dimensions,
        "dask_chunks": {"time": TIME_CHUNK},
    }
    if dimensions is MESH_DIMS:
        kw["coordinates"] = MESH_COORDS
    kw.update(extra)
    return kw


def _assert_levels_match_own_runs(da, kw, spatial):
    """Each real level of the 3-D result equals that level run on its own."""
    result_3d = marEx.preprocess_data(da, **kw).compute()
    # The exact path has no bins: its attrs hold the string "None", which the exact guard rejects.
    bin_spec = (
        {}
        if kw["method_percentile"] == "exact"
        else {"precision": result_3d.attrs["precision"], "max_anomaly": result_3d.attrs["max_anomaly"]}
    )
    for level in REAL_LEVELS:
        slice_nd = da.isel(depth=level, drop=True).chunk({"time": TIME_CHUNK, **{d: -1 for d in spatial}})
        result = marEx.preprocess_data(slice_nd, **{**kw, **bin_spec}).compute()
        got = result_3d.isel(depth=level, drop=True)
        for var in ("dat_anomaly", "mask", "extreme_events", "thresholds"):
            np.testing.assert_array_equal(
                got[var].transpose(*result[var].dims).values,
                result[var].values,
                err_msg=f"{kw['method_extreme']}/{kw['method_percentile']}: '{var}' at depth={level} differs from its own run",
            )


class TestStreamingOnExtraDim:
    """``streaming`` must reproduce ``persist`` exactly when the field carries depth."""

    @pytest.mark.parametrize(
        ("grid", "method_extreme"),
        [("gridded", "seasonal_percentile"), ("gridded", "global_percentile"), ("mesh", "seasonal_percentile")],
    )
    def test_streaming_matches_persist(self, gridded_3d, mesh_3d, grid, method_extreme, tmp_path):
        da, dims = (gridded_3d, GRIDDED_DIMS) if grid == "gridded" else (mesh_3d, MESH_DIMS)
        kw = _kwargs(method_extreme, dims)
        ref = marEx.preprocess_data(da, **kw)
        streamed = marEx.preprocess_data(da, compute_mode="streaming", scratch_dir=str(tmp_path), **kw)
        try:
            assert streamed.encoding.get("marex_staging_dir"), "streaming fell back: no staging dir recorded"
            assert Path(streamed.encoding["marex_staging_dir"]).is_dir()
            assert streamed.sizes["depth"] == da.sizes["depth"]
            for var in ("dat_anomaly", "extreme_events"):
                assert streamed[var].chunksizes["depth"] == ref[var].chunksizes["depth"], var
            _assert_identical(ref, streamed, f"streaming/{grid}/{method_extreme}")
        finally:
            clear_staging(streamed)


class TestUnstructuredExtraDim:
    """``(time, depth, ncells)``: a clean broadcast, tiled per level."""

    @pytest.mark.parametrize("method_extreme", ["global_percentile", "seasonal_percentile"])
    def test_each_level_matches_its_own_run(self, mesh_3d, method_extreme):
        _assert_levels_match_own_runs(mesh_3d, _kwargs(method_extreme, MESH_DIMS), ("ncells",))

    def test_seasonal_tiling_is_the_per_level_mesh_tiling(self):
        """The day-of-year tile on a mesh is its 1-D slice's tile at any depth length.

        Lazy zeros stand in for a large mesh: only sizes are read, and the fixture's
        405 cells fit in one tile, which would make the assertion vacuous.
        """
        from marEx.extremes.histogram import _histogram_tile_chunks

        ncells, ntime, n_bins = 2_000_000, NYEARS * 365, 503

        def mesh(*extra):
            shape = (ntime, *extra, ncells)
            dims = ("time", "depth", "ncells") if extra else ("time", "ncells")
            return xr.DataArray(dsa.zeros(shape, chunks=(TIME_CHUNK, *extra, ncells), dtype=np.float32), dims=dims)

        slice_tile = _histogram_tile_chunks(mesh(), MESH_DIMS, n_bins, window_spatial=None)
        assert slice_tile["ncells"] < ncells, f"slice tile {slice_tile} is not tiled; the test would be vacuous"
        for ndepth in DEPTH_LENGTHS:
            tile = _histogram_tile_chunks(mesh(ndepth), MESH_DIMS, n_bins, window_spatial=None)
            assert tile == {**slice_tile, "depth": 1}, f"depth={ndepth}: {tile} != per-level {slice_tile}"


class TestHistogramTilerPerLevel:
    """``_chunk_spatial_for_histogram`` tiles each level like its horizontal slice."""

    @pytest.mark.parametrize("grid", ["gridded", "mesh"])
    @pytest.mark.parametrize("output_elements_per_cell", [366, 503])
    def test_tile_is_invariant_in_the_depth_length(self, grid, output_elements_per_cell):
        from marEx.extremes.histogram import _chunk_spatial_for_histogram

        ntime = 7000
        horizontal = {"gridded": {"lat": 720, "lon": 1440}, "mesh": {"ncells": 5_000_000}}[grid]
        dims = GRIDDED_DIMS if grid == "gridded" else MESH_DIMS

        def field(*extra):
            names = ("time", *(("depth",) if extra else ()), *horizontal)
            shape = (ntime, *extra, *horizontal.values())
            chunks = (100, *extra, *horizontal.values())
            return xr.DataArray(dsa.zeros(shape, chunks=chunks, dtype=np.float32), dims=names)

        def tile(da):
            out = _chunk_spatial_for_histogram(
                da, "time", output_elements_per_cell=output_elements_per_cell, horizontal=horizontal_dims(dims)
            )
            return {d: out.chunksizes[d][0] for d in out.dims}

        slice_tile = tile(field())
        assert all(slice_tile[d] < n for d, n in horizontal.items()), f"{slice_tile} is not tiled; vacuous"
        for ndepth in DEPTH_LENGTHS:
            got = tile(field(ndepth))
            assert got == {**slice_tile, "depth": 1}, f"{grid} depth={ndepth}: {got} != per-level {slice_tile}"

    def test_a_partial_horizontal_mapping_keeps_the_old_root(self):
        """A mapping naming only one of ``da``'s horizontal dims must not give it a 1-D side."""
        from marEx.extremes.histogram import _chunk_spatial_for_histogram

        da = xr.DataArray(dsa.zeros((7000, 3, 720, 1440), chunks=(100, 3, 720, 1440)), dims=("time", "depth", "lat", "lon"))
        old = _chunk_spatial_for_histogram(da, "time", output_elements_per_cell=366)
        partial = _chunk_spatial_for_histogram(da, "time", output_elements_per_cell=366, horizontal=("lat", "x"))
        assert partial.chunksizes == old.chunksizes

    def test_a_small_horizontal_extent_stacks_levels(self):
        """A mooring-like field: one whole slice fits, so levels share a tile instead of one task each."""
        from marEx.extremes.histogram import _chunk_spatial_for_histogram

        da = xr.DataArray(dsa.zeros((1460, 4, 100, 10), chunks=(100, 4, 100, 10)), dims=("time", "member", "depth", "ncells"))
        out = _chunk_spatial_for_histogram(da, "time", output_elements_per_cell=503, horizontal=("ncells",))
        tile_area = 50_000_000 // 1460
        chunks = {d: out.chunksizes[d] for d in ("member", "depth", "ncells")}
        assert chunks["ncells"] == (10,)
        cells = max(chunks["member"]) * max(chunks["depth"]) * 10
        assert cells <= tile_area, chunks
        assert int(np.prod([len(c) for c in chunks.values()])) <= 2, f"{chunks}: one task per level is back"

    @pytest.mark.parametrize(
        ("method_extreme", "method_percentile", "module"),
        [
            ("global_percentile", "approximate", "marEx.extremes.histogram"),
            ("seasonal_percentile", "exact", "marEx.extremes.seasonal_percentile"),
        ],
    )
    def test_both_callers_tile_per_level(self, gridded_3d, monkeypatch, method_extreme, method_percentile, module):
        """Through the public API: each caller hands the tiler the horizontal dims.

        Spied at the call site, so a caller that stops passing them fails here even
        though the tiler's own test still passes. The spy shrinks the budget so the
        10x20 crop is actually tiled (at the shipped budget it fits in one tile, the
        stacking regime); the old every-dim root then gave depth chunks (3,).
        """
        import importlib

        from marEx.extremes import histogram

        target = importlib.import_module(module)
        original = histogram._chunk_spatial_for_histogram
        seen = []

        def spy(*args, **kwargs):
            kwargs.setdefault("target_elements", SMALL_BUDGET)
            out = original(*args, **kwargs)
            seen.append({d: out.chunksizes[d] for d in out.dims})
            return out

        monkeypatch.setattr(target, "_chunk_spatial_for_histogram", spy)
        kw = _kwargs(method_extreme, GRIDDED_DIMS, method_percentile)

        marEx.preprocess_data(gridded_3d, **kw).compute()
        assert seen, f"{module} never called the tiler for {method_extreme}/{method_percentile}"
        for chunks in seen:
            assert set(chunks["depth"]) == {1}, f"depth chunks {chunks['depth']}: levels share a tile"
            assert max(chunks["lat"]) < gridded_3d.sizes["lat"], f"{chunks}: crop not tiled, the check is vacuous"

        seen_3d, seen[:] = list(seen), []
        slice_2d = gridded_3d.isel(depth=0, drop=True).chunk({"time": TIME_CHUNK, "lat": -1, "lon": -1})
        marEx.preprocess_data(slice_2d, **kw).compute()
        assert len(seen) == len(seen_3d), (len(seen), len(seen_3d))
        for got, want in zip(seen_3d, seen):
            assert {d: got[d] for d in ("lat", "lon")} == {d: want[d] for d in ("lat", "lon")}, (got, want)


def extra_dim_chunks(*args):
    from marEx.core.dimensions import extra_dim_chunks as rule

    return rule(*args)


class TestExtraDimChunks:
    """The one rule every site uses (D-128)."""

    def test_a_2d_field_gets_nothing(self):
        assert extra_dim_chunks({"lat": 10, "lon": 20}, [], 200, 200, 10_000) == {}

    def test_one_level_per_chunk_while_the_slice_is_tiled(self):
        assert extra_dim_chunks({"depth": 50}, ["depth"], 7225, 1_036_800, 7225) == {"depth": 1}

    def test_a_whole_slice_stacks_levels_shortest_first(self):
        sizes = {"member": 4, "depth": 100}
        assert extra_dim_chunks(sizes, ["depth", "member"], 10, 10, 1000) == {"member": 4, "depth": 25}

    def test_a_floor_over_budget_stacks_nothing(self):
        assert extra_dim_chunks({"depth": 50}, ["depth"], 400, 400, 300) == {"depth": 1}


class TestGlobalExactOnExtraDim:
    """``global_percentile`` exact on a mesh: depth is not sized like the cell axis."""

    def test_mesh_depth_is_per_level(self, mesh_3d, monkeypatch):
        seen = []
        original = xr.DataArray.quantile

        def spy(self, *args, **kwargs):
            seen.append(dict(self.chunksizes))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(xr.DataArray, "quantile", spy)
        kw = _kwargs("global_percentile", MESH_DIMS, "exact")
        marEx.preprocess_data(mesh_3d, **kw).compute()
        assert seen, "the exact path never called quantile"
        for chunks in seen:
            assert max(chunks["ncells"]) < mesh_3d.sizes["ncells"], chunks
            assert set(chunks["depth"]) == {1}, f"depth chunks {chunks['depth']}: sized like the cell axis"

    @pytest.mark.parametrize(("grid", "method_percentile"), [("gridded", "exact"), ("mesh", "exact")])
    def test_each_level_matches_its_own_run(self, gridded_3d, mesh_3d, grid, method_percentile):
        da, dims, spatial = (gridded_3d, GRIDDED_DIMS, ("lat", "lon")) if grid == "gridded" else (mesh_3d, MESH_DIMS, ("ncells",))
        _assert_levels_match_own_runs(da, _kwargs("global_percentile", dims, method_percentile), spatial)


class TestOutputLayoutOnExtraDim:
    """``finalise`` and harmonic ``dat_stn`` / ``STD``: depth follows the D-128 rule, not held whole."""

    @pytest.mark.parametrize("module", ["marEx.core.finalise"])
    def test_output_is_per_level_on_a_large_slice(self, gridded_3d, monkeypatch, module):
        """Shrinking the budget below one whole-horizontal time block is what a real grid looks like."""
        import importlib

        monkeypatch.setattr(importlib.import_module(module), "TASK_ELEMENTS", 100 * TIME_CHUNK, raising=False)
        out = marEx.preprocess_data(gridded_3d, **_kwargs("global_percentile", GRIDDED_DIMS))
        for var in ("dat_anomaly", "extreme_events"):
            assert set(out[var].chunksizes["depth"]) == {1}, (var, out[var].chunksizes["depth"])
            assert out[var].chunksizes["lat"] == (gridded_3d.sizes["lat"],)

    @pytest.mark.parametrize("time_chunk", ["auto", None, -1, (400, 330)])
    @pytest.mark.parametrize("with_depth", [True, False])
    def test_any_time_chunk_spelling_is_accepted(self, gridded_3d, monkeypatch, time_chunk, with_depth):
        """``dask_chunks`` time may be "auto", None, -1 or a tuple: none may crash, and depth is
        sized from the time chunk dask actually chose (-1 is the whole series, not one step)."""
        from marEx.core import finalise

        monkeypatch.setattr(finalise, "TASK_ELEMENTS", 200 * 400, raising=False)
        da = gridded_3d if with_depth else gridded_3d.isel(depth=0, drop=True)
        kw = _kwargs("global_percentile", GRIDDED_DIMS, dask_chunks={"time": time_chunk})
        out = marEx.preprocess_data(da, **kw)
        if with_depth:
            steps = max(out.dat_anomaly.chunksizes["time"])
            per_chunk = steps * 200 * max(out.dat_anomaly.chunksizes["depth"])
            assert per_chunk <= max(200 * 400, steps * 200), f"time {steps} x depth {out.dat_anomaly.chunksizes['depth']}"

    def test_output_stacks_levels_on_a_small_slice(self, gridded_3d):
        out = marEx.preprocess_data(gridded_3d, **_kwargs("global_percentile", GRIDDED_DIMS))
        assert out.dat_anomaly.chunksizes["depth"] == (gridded_3d.sizes["depth"],)

    @pytest.mark.parametrize("budget", ["small", "shipped"])
    def test_standardised_fields_follow_the_rule(self, gridded_3d, monkeypatch, budget):
        from marEx.anomaly import harmonic

        if budget == "small":
            monkeypatch.setattr(harmonic, "TASK_ELEMENTS", 100 * TIME_CHUNK, raising=False)
        ds = harmonic._compute_anomaly_detrended(gridded_3d, standardise=True, dimensions=GRIDDED_DIMS, coordinates=GRIDDED_DIMS)
        want = {1} if budget == "small" else {gridded_3d.sizes["depth"]}
        for var in ("dat_stn", "STD"):
            assert set(ds[var].chunksizes["depth"]) == want, (var, ds[var].chunksizes["depth"])
            assert ds[var].chunksizes["lat"] == (gridded_3d.sizes["lat"],)


class TestExactSeasonalOnExtraDim:
    """The exact day-of-year path had no 3-D value coverage: pin it per level."""

    def test_each_level_matches_its_own_2d_run(self, gridded_3d, monkeypatch):
        """The 3-D leg runs tiled (forced budget), the 2-D legs in one tile: values must not care."""
        from marEx.extremes import histogram, seasonal_percentile

        original = histogram._chunk_spatial_for_histogram
        calls = []

        def forced(*args, **kwargs):
            kwargs.setdefault("target_elements", SMALL_BUDGET)
            out = original(*args, **kwargs)
            calls.append(max(out.chunksizes["lat"]))
            return out

        kw = _kwargs("seasonal_percentile", GRIDDED_DIMS, "exact")
        with monkeypatch.context() as m:
            m.setattr(seasonal_percentile, "_chunk_spatial_for_histogram", forced)
            result_3d = marEx.preprocess_data(gridded_3d, **kw).compute()
        assert calls and max(calls) < gridded_3d.sizes["lat"], f"3-D leg not tiled: {calls}"

        for level in REAL_LEVELS:
            slice_2d = gridded_3d.isel(depth=level, drop=True).chunk({"time": TIME_CHUNK, "lat": -1, "lon": -1})
            result = marEx.preprocess_data(slice_2d, **kw).compute()
            got = result_3d.isel(depth=level, drop=True)
            for var in ("dat_anomaly", "mask", "extreme_events", "thresholds"):
                np.testing.assert_array_equal(
                    got[var].transpose(*result[var].dims).values,
                    result[var].values,
                    err_msg=f"exact seasonal: '{var}' at depth={level} differs from its own 2-D run",
                )
