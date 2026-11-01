"""3D+time support: an extra dimension (depth) must be a broadcast, by value and by chunk structure.

marEx fields are ``(time, *extra, *horizontal)``. The maths was always dimension-agnostic --
every ``apply_ufunc`` declares its core dims over time/dayofyear/da_bin and broadcasts the rest
-- so what a depth axis actually tests is whether the *memory management* carries it: chunk
dicts built from a hardcoded ``["x", "y"]`` silently left it unchunked and unmasked.

The core test is **slice equivalence** (``TestSliceEquivalence``): run the pipeline on the 3-D field,
run it separately on each ``isel(depth=k)`` slice, and require the two to agree with zero
tolerance. It is the only test that can distinguish a genuine broadcast from a silent reduction
over the extra axis, and it is why it compares values rather than shapes.

There is exactly one input that is legitimately not slice-invariant, and the test concedes it
explicitly rather than loosening its tolerance: the **histogram bin geometry**.
``resolve_bin_spec`` derives ``max_anomaly`` as ``max(|min|, |max|)`` over the whole series, and
a 3-D field is one series, so the range is pooled over depth -- the same way a 2-D run already
pools over lat/lon. The per-level leg is therefore given the 3-D run's own resolved geometry, and
everything else stays exact. Do not "fix" this by pinning constants: that would stop the 3-D path
exercising ``resolve_bin_spec`` at all.

The ``sst_3d`` fixture carries one entirely-NaN level, standing in for a bathymetry mask: it must
yield an all-False mask and NaN thresholds rather than an exception. That level is excluded from
the per-level leg, because a 2-D run on an all-NaN field correctly raises "no valid data".

The other paths (streaming, the unstructured mesh, the exact paths, the 1-D histogram tiler and
the output layout) are pinned the same two ways -- by VALUE against the per-level run, and by
CHUNK STRUCTURE, because every tiling is value-identical and a value check alone passes while a
tile quietly grows or shatters with depth. Every site follows one rule,
:func:`marEx.core.dimensions.extra_dim_chunks`: one level per chunk while the horizontal tile is
smaller than the whole slice, and levels stacked with the leftover budget once a whole slice
fits. Their fixtures (``gridded_3d``, ``mesh_3d``) are crops (10 x 20 cells gridded, 405 cells
mesh, 4 years) that fit in one tile at the shipped budgets, the STACKING regime; the per-level
tests shrink the budget so the crop is tiled, which is what a real field looks like.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.core.compute_mode import clear_staging

from ..conftest import assert_no_failures
from .conftest import GRID_DIMS, MESH_COORDS, MESH_DIMS, assert_identical, check, open_sst_gridded, open_sst_mesh

TIME_CHUNK = 30
NYEARS = 4

# Distinct offset and scale per level, so a reduction over depth (or a level borrowed from
# another) cannot pass unnoticed.
LEVEL_TRANSFORMS = [(0.0, 1.0), (-1.5, 0.8), (2.25, 1.3)]
REAL_LEVELS = tuple(range(len(LEVEL_TRANSFORMS)))
NAN_LEVEL = len(LEVEL_TRANSFORMS)
# A budget small enough that the 10x20 crop is tiled (~5x5 per tile at ~730 anomaly steps).
SMALL_BUDGET = 20_000

ANOMALY_METHODS = ["shifting_baseline", "detrend_harmonic", "fixed_baseline", "detrend_fixed_baseline"]
EXTREME_METHODS = ["global_percentile", "seasonal_percentile"]
VARS = ("dat_anomaly", "mask", "extreme_events", "thresholds")


def _stack_levels(levels, spatial):
    da = xr.concat(levels, dim="depth").assign_coords(depth=np.arange(len(levels), dtype=np.float32))
    da = da.transpose("time", "depth", *spatial)
    da.name = "to"
    return da.chunk({"time": TIME_CHUNK, "depth": -1, **{d: -1 for d in spatial}})


@pytest.fixture(scope="module")
def sst_3d():
    """``(time, depth, lat, lon)``, the whole 20 x 40 grid: three distinct levels plus an all-NaN one.

    Built here rather than shipped as a zarr: deriving it costs milliseconds.
    """
    base = open_sst_gridded().isel(time=slice(0, NYEARS * 365))
    levels = [base * scale + offset for offset, scale in LEVEL_TRANSFORMS] + [xr.full_like(base, np.nan)]
    return _stack_levels(levels, ("lat", "lon"))


@pytest.fixture(scope="module")
def gridded_3d():
    base = open_sst_gridded().isel(time=slice(0, NYEARS * 365), lat=slice(0, 10), lon=slice(0, 20))
    return _stack_levels([base * scale + offset for offset, scale in LEVEL_TRANSFORMS], ("lat", "lon"))


@pytest.fixture(scope="module")
def mesh_3d():
    base = open_sst_mesh(60.0, 180.0).isel(time=slice(0, NYEARS * 365))
    return _stack_levels([base * scale + offset for offset, scale in LEVEL_TRANSFORMS], ("ncells",))


def _slice_run(da, method_anomaly, method_extreme, **extra):
    """The slice-equivalence test's keyword set (``sst_3d``)."""
    return marEx.preprocess_data(
        da,
        method_anomaly=method_anomaly,
        method_extreme=method_extreme,
        window_years=2,
        smooth_days=21,
        threshold_percentile=95,
        dimensions=GRID_DIMS,
        dask_chunks={"time": TIME_CHUNK},
        **extra,
    )


def _kwargs(method_extreme, dimensions, method_percentile="approximate", **extra):
    """The crops' keyword set (``gridded_3d``, ``mesh_3d``)."""
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


class TestSliceEquivalence:
    """The 3-D result must equal the stack of independent per-level 2-D results."""

    @pytest.mark.slow
    @pytest.mark.parametrize("method_anomaly", ANOMALY_METHODS)
    @pytest.mark.parametrize("method_extreme", EXTREME_METHODS)
    def test_each_level_matches_its_own_2d_run(self, sst_3d, method_anomaly, method_extreme, monkeypatch):
        # The seasonal path's histogram is (cycle x n_bins) per cell: at the default 1000
        # bins this fixture's is 3200 x 366 x 1000 counts, and the spatial window keeps
        # neighbouring tiles live, which on a 2x4-tile level is all of them -- 15.2 GB peak
        # for ONE param on the synchronous scheduler, so it could only run in a 96 GB batch
        # job. 100 bins keeps the data-derived range (neither precision nor max_anomaly is
        # given, so `resolve_bin_spec` still runs), and a 10x smaller task budget keeps the
        # 12x12 tiling the default gives -- lat AND lon tile boundaries, under the window.
        extra = {}
        if method_extreme == "seasonal_percentile":
            from marEx.extremes import histogram

            extra = {"n_bins": 100}
            monkeypatch.setattr(histogram, "_HISTOGRAM_TASK_ELEMENTS", histogram._HISTOGRAM_TASK_ELEMENTS // 10)
            tile = histogram._histogram_tile_chunks(sst_3d, GRID_DIMS, 100, window_spatial=None)
            assert tile["lat"] < sst_3d.sizes["lat"] and tile["lon"] < sst_3d.sizes["lon"], f"{tile}: fixture not tiled"
        result_3d = _slice_run(sst_3d, method_anomaly, method_extreme, **extra).compute()

        # The 3-D leg runs with the production default: bin geometry derived from the
        # data. That geometry is the one input which is legitimately NOT slice-invariant
        # -- `resolve_bin_spec` pools `max(|min|, |max|)` over the whole series, and a
        # 3-D field is one series, so it pools over depth. The fixture's levels scale the
        # anomaly by 1.0 / 0.8 / 1.3, so the pooled range is level 2's; measured, levels 0
        # and 1 then disagree with their own 2-D runs on ~0.03 % of `extreme_events` while
        # level 2 stays bit-identical. That is the bin spec differing, not the depth axis
        # being reduced over: pinning the geometry makes all three exact.
        #
        # So hand the 3-D run's OWN resolved geometry to each 2-D leg. The test keeps
        # asking its real question -- is the extra dimension a clean broadcast? -- and
        # still exercises `resolve_bin_spec` on the 3-D path, which pinning constants
        # here would not.
        bin_spec = {
            "precision": result_3d.attrs["precision"],
            "max_anomaly": result_3d.attrs["max_anomaly"],
        }

        for level in REAL_LEVELS:
            slice_2d = sst_3d.isel(depth=level, drop=True).chunk({"time": TIME_CHUNK, "lat": -1, "lon": -1})
            result_2d = _slice_run(slice_2d, method_anomaly, method_extreme, **bin_spec, **extra).compute()
            got = result_3d.isel(depth=level, drop=True)

            assert list(result_2d.time.values) == list(got.time.values), f"level {level}: time axis differs"

            for var in VARS:
                np.testing.assert_array_equal(
                    got[var].transpose(*result_2d[var].dims).values,
                    result_2d[var].values,
                    err_msg=(
                        f"{method_anomaly}/{method_extreme}: '{var}' at depth={level} differs from "
                        f"its own 2-D run -- the extra dimension is not a clean broadcast"
                    ),
                )


@pytest.mark.parametrize("method_extreme", EXTREME_METHODS)
def test_extra_dim_shape_and_nan_level(sst_3d, method_extreme):
    """One fixed_baseline 3-D run per extreme method: the output carries depth, the all-NaN level
    masks out rather than raising, and (lazily, before compute) finalise makes the horizontal dims
    whole and chunks time as asked.

    Depth comes out whole here only because a 20x40 slice is far below the task budget, so
    ``extra_dim_chunks`` stacks every level; on a large grid it is one level per chunk
    (``test_output_depth_chunking``).
    """
    lazy = _slice_run(sst_3d, "fixed_baseline", method_extreme)
    if method_extreme == "global_percentile":
        assert lazy.dat_anomaly.chunksizes["depth"] == (sst_3d.sizes["depth"],)
        assert lazy.dat_anomaly.chunksizes["lat"] == (sst_3d.sizes["lat"],)
        assert max(lazy.dat_anomaly.chunksizes["time"]) == TIME_CHUNK
    result = lazy.compute()

    assert set(result.mask.dims) == {"depth", "lat", "lon"}, result.mask.dims
    assert set(result.dat_anomaly.dims) == {"time", "depth", "lat", "lon"}, result.dat_anomaly.dims
    assert "depth" in result.thresholds.dims, result.thresholds.dims
    assert result.sizes["depth"] == sst_3d.sizes["depth"]

    assert not bool(result.mask.isel(depth=NAN_LEVEL).values.any()), "all-NaN level should mask to all-False"
    assert bool(np.isnan(result.thresholds.isel(depth=NAN_LEVEL).values).all()), "all-NaN level should give NaN thresholds"
    assert int(result.extreme_events.isel(depth=NAN_LEVEL).sum()) == 0
    # ... and the real levels are untouched by its presence.
    assert bool(result.mask.isel(depth=0).values.all())


def test_streaming_matches_persist_on_extra_dim(gridded_3d, mesh_3d, tmp_path):
    """Table over (grid, method): ``streaming`` reproduces ``persist`` exactly when the field carries depth
    (staging had never carried an extra dim), records a live staging dir, and keeps depth's chunking."""
    failures = []
    for grid, method_extreme in [
        ("gridded", "seasonal_percentile"),
        ("gridded", "global_percentile"),
        ("mesh", "seasonal_percentile"),
    ]:

        def row(grid=grid, method_extreme=method_extreme):
            da, dims = (gridded_3d, GRID_DIMS) if grid == "gridded" else (mesh_3d, MESH_DIMS)
            kw = _kwargs(method_extreme, dims)
            ref = marEx.preprocess_data(da, **kw)
            streamed = marEx.preprocess_data(
                da, compute_mode="streaming", scratch_dir=str(tmp_path / f"{grid}-{method_extreme}"), **kw
            )
            try:
                assert streamed.encoding.get("marex_staging_dir"), "streaming fell back: no staging dir recorded"
                assert Path(streamed.encoding["marex_staging_dir"]).is_dir()
                assert streamed.sizes["depth"] == da.sizes["depth"]
                for var in ("dat_anomaly", "extreme_events"):
                    assert streamed[var].chunksizes["depth"] == ref[var].chunksizes["depth"], var
                assert_identical(ref, streamed, f"streaming/{grid}/{method_extreme}")
            finally:
                clear_staging(streamed)

        check(failures, f"{grid}-{method_extreme}", row)
    assert_no_failures(failures)


def _assert_levels_match(result_3d, da, kw, spatial, label, runner=None):
    """Each real level of ``result_3d`` equals that level run on its own, handed the 3-D run's bin geometry."""
    # The exact path has no bins: its attrs hold the string "None", which the exact guard rejects.
    bin_spec = (
        {}
        if kw["method_percentile"] == "exact"
        else {"precision": result_3d.attrs["precision"], "max_anomaly": result_3d.attrs["max_anomaly"]}
    )
    for level in REAL_LEVELS:
        slice_nd = da.isel(depth=level, drop=True).chunk({"time": TIME_CHUNK, **{d: -1 for d in spatial}})
        run = runner(level) if runner is not None else marEx.preprocess_data
        result = run(slice_nd, **{**kw, **bin_spec}).compute()
        got = result_3d.isel(depth=level, drop=True)
        for var in VARS:
            np.testing.assert_array_equal(
                got[var].transpose(*result[var].dims).values,
                result[var].values,
                err_msg=f"{label}: '{var}' at depth={level} differs from its own run",
            )


def _tiler_spy(monkeypatch, module, forced=True):
    """Spy on ``module``'s ``_chunk_spatial_for_histogram``; with ``forced`` the budget shrinks so the crop is tiled."""
    from marEx.extremes import histogram

    original = histogram._chunk_spatial_for_histogram
    seen = []

    def spy(*args, **kwargs):
        if forced:
            kwargs.setdefault("target_elements", SMALL_BUDGET)
        out = original(*args, **kwargs)
        seen.append({d: out.chunksizes[d] for d in out.dims})
        return out

    monkeypatch.setattr(importlib.import_module(module), "_chunk_spatial_for_histogram", spy)
    return seen


@pytest.mark.parametrize("case", ["mesh-approximate", "exact-global", "exact-seasonal"])
def test_other_paths_match_their_per_level_runs(gridded_3d, mesh_3d, monkeypatch, case):
    """Each real level of a 3-D run equals its own run, on the paths the slice test does not reach.

    - ``mesh-approximate``: ``(time, depth, ncells)``, global and seasonal, a clean broadcast.
    - ``exact-global``: ``global_percentile`` exact on the gridded crop and the mesh; on the mesh
      the quantile spy shows depth is not sized like the cell axis (one level per chunk, cells tiled).
    - ``exact-seasonal``: the exact day-of-year path; its 3-D leg runs tiled (forced budget) and
      the caller hands the tiler the horizontal dims (depth chunks 1, crop actually tiled). Level 0's
      2-D leg runs under the same forced budget and must get the same lat/lon tiles; levels 1 and 2
      run in one tile, so the values must not care about the tiling.
    """
    if case == "mesh-approximate":
        for method_extreme in EXTREME_METHODS:
            kw = _kwargs(method_extreme, MESH_DIMS)
            _assert_levels_match(marEx.preprocess_data(mesh_3d, **kw).compute(), mesh_3d, kw, ("ncells",), f"mesh/{method_extreme}")

    elif case == "exact-global":
        kw = _kwargs("global_percentile", GRID_DIMS, "exact")
        _assert_levels_match(
            marEx.preprocess_data(gridded_3d, **kw).compute(), gridded_3d, kw, ("lat", "lon"), "exact-global/gridded"
        )

        seen = []
        original = xr.DataArray.quantile
        kw = _kwargs("global_percentile", MESH_DIMS, "exact")
        with monkeypatch.context() as m:

            def spy(self, *args, **kwargs):
                seen.append(dict(self.chunksizes))
                return original(self, *args, **kwargs)

            m.setattr(xr.DataArray, "quantile", spy)
            result_3d = marEx.preprocess_data(mesh_3d, **kw).compute()
        assert seen, "the exact path never called quantile"
        for chunks in seen:
            assert max(chunks["ncells"]) < mesh_3d.sizes["ncells"], chunks
            assert set(chunks["depth"]) == {1}, f"depth chunks {chunks['depth']}: sized like the cell axis"
        _assert_levels_match(result_3d, mesh_3d, kw, ("ncells",), "exact-global/mesh")

    else:
        module = "marEx.extremes.seasonal_percentile"
        kw = _kwargs("seasonal_percentile", GRID_DIMS, "exact")
        with monkeypatch.context() as m:
            seen_3d = _tiler_spy(m, module)
            result_3d = marEx.preprocess_data(gridded_3d, **kw).compute()
        assert seen_3d, f"{module} never called the tiler"
        for chunks in seen_3d:
            assert set(chunks["depth"]) == {1}, f"depth chunks {chunks['depth']}: levels share a tile"
            assert max(chunks["lat"]) < gridded_3d.sizes["lat"], f"{chunks}: crop not tiled, the check is vacuous"

        seen_2d = []

        def runner(level):
            if level != 0:
                return marEx.preprocess_data

            def forced_run(da, **kwargs):
                with monkeypatch.context() as m:
                    spied = _tiler_spy(m, module)
                    result = marEx.preprocess_data(da, **kwargs).compute()
                seen_2d.extend(spied)
                return result

            return forced_run

        _assert_levels_match(result_3d, gridded_3d, kw, ("lat", "lon"), "exact seasonal", runner=runner)
        assert len(seen_2d) == len(seen_3d), (len(seen_2d), len(seen_3d))
        for got, want in zip(seen_3d, seen_2d):
            assert {d: got[d] for d in ("lat", "lon")} == {d: want[d] for d in ("lat", "lon")}, (got, want)


def test_global_histogram_caller_tiles_per_level(gridded_3d, monkeypatch):
    """Through the public API, the 1-D histogram caller hands ``_chunk_spatial_for_histogram`` the horizontal dims.

    Spied at the call site, so a caller that stops passing them fails here even though the
    tiler's own test still passes. The spy shrinks the budget so the 10x20 crop is actually tiled
    (at the shipped budget it fits in one tile, the stacking regime); the old every-dim root then
    gave depth chunks (3,). The 2-D slice under the same budget must get the same lat/lon tiles.
    """
    module = "marEx.extremes.histogram"
    seen = _tiler_spy(monkeypatch, module)
    kw = _kwargs("global_percentile", GRID_DIMS, "approximate")

    marEx.preprocess_data(gridded_3d, **kw).compute()
    assert seen, f"{module} never called the tiler for global_percentile/approximate"
    for chunks in seen:
        assert set(chunks["depth"]) == {1}, f"depth chunks {chunks['depth']}: levels share a tile"
        assert max(chunks["lat"]) < gridded_3d.sizes["lat"], f"{chunks}: crop not tiled, the check is vacuous"

    seen_3d, seen[:] = list(seen), []
    slice_2d = gridded_3d.isel(depth=0, drop=True).chunk({"time": TIME_CHUNK, "lat": -1, "lon": -1})
    marEx.preprocess_data(slice_2d, **kw).compute()
    assert len(seen) == len(seen_3d), (len(seen), len(seen_3d))
    for got, want in zip(seen_3d, seen):
        assert {d: got[d] for d in ("lat", "lon")} == {d: want[d] for d in ("lat", "lon")}, (got, want)


# The 2-D output time chunks each spelling yields on the crop (730 output steps after the 2-year
# shifting window), MEASURED on 7b9b4e3 (pins/detect/pins_steps_chunks.out): "auto" and -1 are the
# whole series, a tuple is taken as given, and None leaves the pipeline's own time chunks alone.
SPELLING_TIME_CHUNKS = {"auto": (730,), None: (29,) * 24 + (34,), -1: (730,), (400, 330): (400, 330)}


def test_any_time_chunk_spelling_is_accepted(gridded_3d, monkeypatch):
    """Table over ``dask_chunks`` time = "auto", None, -1 and a tuple, with and without depth: none may crash.

    With depth, depth is sized from the time chunk dask actually chose (-1 is the whole series,
    not one step). Without depth, the output time chunks are the ones the spelling asks for.
    """
    from marEx.core import finalise

    monkeypatch.setattr(finalise, "TASK_ELEMENTS", 200 * 400, raising=False)
    failures = []
    for with_depth in (True, False):
        for time_chunk, want in SPELLING_TIME_CHUNKS.items():

            def row(with_depth=with_depth, time_chunk=time_chunk, want=want):
                da = gridded_3d if with_depth else gridded_3d.isel(depth=0, drop=True)
                out = marEx.preprocess_data(da, **_kwargs("global_percentile", GRID_DIMS, dask_chunks={"time": time_chunk}))
                if with_depth:
                    steps = max(out.dat_anomaly.chunksizes["time"])
                    per_chunk = steps * 200 * max(out.dat_anomaly.chunksizes["depth"])
                    assert per_chunk <= max(200 * 400, steps * 200), f"time {steps} x depth {out.dat_anomaly.chunksizes['depth']}"
                else:
                    assert out.dat_anomaly.chunksizes["time"] == want, out.dat_anomaly.chunksizes["time"]
                    assert out.extreme_events.chunksizes["time"] == want, out.extreme_events.chunksizes["time"]

            check(failures, f"depth={with_depth}-time={time_chunk!r}", row)
    assert_no_failures(failures)


def test_output_depth_chunking(gridded_3d, monkeypatch):
    """finalise sizes depth by ``extra_dim_chunks``: one level per chunk once the budget is below one
    whole-horizontal time block (what a real grid looks like), every level stacked on the small crop."""
    from marEx.core import finalise

    kw = _kwargs("global_percentile", GRID_DIMS)
    small = marEx.preprocess_data(gridded_3d, **kw)
    assert small.dat_anomaly.chunksizes["depth"] == (gridded_3d.sizes["depth"],)

    monkeypatch.setattr(finalise, "TASK_ELEMENTS", 100 * TIME_CHUNK, raising=False)
    large = marEx.preprocess_data(gridded_3d, **kw)
    for var in ("dat_anomaly", "extreme_events"):
        assert set(large[var].chunksizes["depth"]) == {1}, (var, large[var].chunksizes["depth"])
        assert large[var].chunksizes["lat"] == (gridded_3d.sizes["lat"],)


def test_standardised_fields_follow_the_rule(gridded_3d, monkeypatch):
    """Harmonic ``dat_stn`` / ``STD``: depth one level per chunk under a small budget, stacked at the shipped one."""
    from marEx.anomaly import harmonic

    for budget in ("shipped", "small"):
        if budget == "small":
            monkeypatch.setattr(harmonic, "TASK_ELEMENTS", 100 * TIME_CHUNK, raising=False)
        ds = harmonic._compute_anomaly_detrended(gridded_3d, standardise=True, dimensions=GRID_DIMS, coordinates=GRID_DIMS)
        want = {1} if budget == "small" else {gridded_3d.sizes["depth"]}
        for var in ("dat_stn", "STD"):
            assert set(ds[var].chunksizes["depth"]) == want, (budget, var, ds[var].chunksizes["depth"])
            assert ds[var].chunksizes["lat"] == (gridded_3d.sizes["lat"],)
