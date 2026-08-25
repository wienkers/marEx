"""3D+time support: the extra dimension must be a broadcast, not a reduction.

marEx fields are ``(time, *extra, *horizontal)``. The maths was always
dimension-agnostic -- every ``apply_ufunc`` declares its core dims over
time/dayofyear/da_bin and broadcasts the rest -- so what a depth axis actually
tests is whether the *memory management* carries it: chunk dicts built from a
hardcoded ``["x", "y"]`` silently left it unchunked and unmasked.

The gate here is **slice equivalence**: run the pipeline on the 3-D field, run it
separately on each ``isel(depth=k)`` slice, and require the two to agree with zero
tolerance. It is the only test that can distinguish a genuine broadcast from a
silent reduction over the extra axis, and it is why it compares values rather than
shapes.

The fixture carries one entirely-NaN level, standing in for a bathymetry mask: it
must yield an all-False mask and NaN thresholds rather than an exception. That
level is excluded from the per-level leg, because a 2-D run on an all-NaN field
correctly raises "no valid data" -- it is only valid *within* a field that has
other levels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import TrackingError, VisualisationError

DATA_DIR = Path(__file__).parent / "data"

DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}
WINDOW_YEARS = 2
SMOOTH_DAYS = 21
TIME_CHUNK = 30

# Levels 0-2 carry a distinct offset and scaling so a reduction over depth (or a
# level accidentally broadcast from another) cannot pass unnoticed. Level 3 is
# entirely NaN.
LEVEL_TRANSFORMS = [(0.0, 1.0), (-1.5, 0.8), (2.25, 1.3)]
REAL_LEVELS = tuple(range(len(LEVEL_TRANSFORMS)))
NAN_LEVEL = len(LEVEL_TRANSFORMS)

ANOMALY_METHODS = ["shifting_baseline", "detrend_harmonic", "fixed_baseline", "detrend_fixed_baseline"]
EXTREME_METHODS = ["global_percentile", "seasonal_percentile"]


def _base(nyears=4):
    """Gridded 2-D source field, a subset of the shipped fixture."""
    return xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.isel(time=slice(0, nyears * 365))


@pytest.fixture(scope="module")
def sst_3d():
    """``(time, depth, lat, lon)`` field: three distinct levels plus an all-NaN one.

    Built here rather than shipped as a zarr: broadcasting the 14611-step fixture
    across depth is ~140 MB of derived data, and deriving it costs milliseconds.
    """
    base = _base()
    levels = [base * scale + offset for offset, scale in LEVEL_TRANSFORMS]
    levels.append(xr.full_like(base, np.nan))
    da = xr.concat(levels, dim="depth").assign_coords(depth=np.arange(len(levels), dtype=np.float32))
    da = da.transpose("time", "depth", "lat", "lon")
    da.name = "to"
    return da.chunk({"time": TIME_CHUNK, "depth": -1, "lat": -1, "lon": -1})


def _run(da, method_anomaly, method_extreme):
    return marEx.preprocess_data(
        da,
        method_anomaly=method_anomaly,
        method_extreme=method_extreme,
        window_years=WINDOW_YEARS,
        smooth_days=SMOOTH_DAYS,
        threshold_percentile=95,
        dimensions=DIMENSIONS,
        dask_chunks={"time": TIME_CHUNK},
    )


class TestSliceEquivalence:
    """The 3-D result must equal the stack of independent per-level 2-D results."""

    @pytest.mark.slow
    @pytest.mark.parametrize("method_anomaly", ANOMALY_METHODS)
    @pytest.mark.parametrize("method_extreme", EXTREME_METHODS)
    def test_each_level_matches_its_own_2d_run(self, sst_3d, method_anomaly, method_extreme):
        result_3d = _run(sst_3d, method_anomaly, method_extreme).compute()

        for level in REAL_LEVELS:
            slice_2d = sst_3d.isel(depth=level, drop=True).chunk({"time": TIME_CHUNK, "lat": -1, "lon": -1})
            result_2d = _run(slice_2d, method_anomaly, method_extreme).compute()
            got = result_3d.isel(depth=level, drop=True)

            assert list(result_2d.time.values) == list(got.time.values), f"level {level}: time axis differs"

            for var in ("dat_anomaly", "mask", "extreme_events", "thresholds"):
                np.testing.assert_array_equal(
                    got[var].transpose(*result_2d[var].dims).values,
                    result_2d[var].values,
                    err_msg=(
                        f"{method_anomaly}/{method_extreme}: '{var}' at depth={level} differs from "
                        f"its own 2-D run -- the extra dimension is not a clean broadcast"
                    ),
                )


class TestExtraDimensionShape:
    """Structural guarantees that hold for every method."""

    @pytest.mark.parametrize("method_extreme", EXTREME_METHODS)
    def test_mask_and_threshold_carry_the_extra_dim(self, sst_3d, method_extreme):
        result = _run(sst_3d, "fixed_baseline", method_extreme).compute()

        assert set(result.mask.dims) == {"depth", "lat", "lon"}, result.mask.dims
        assert set(result.dat_anomaly.dims) == {"time", "depth", "lat", "lon"}, result.dat_anomaly.dims
        assert "depth" in result.thresholds.dims, result.thresholds.dims
        assert result.sizes["depth"] == sst_3d.sizes["depth"]

    @pytest.mark.parametrize("method_extreme", EXTREME_METHODS)
    def test_all_nan_level_masks_out_rather_than_raising(self, sst_3d, method_extreme):
        result = _run(sst_3d, "fixed_baseline", method_extreme).compute()

        assert not bool(result.mask.isel(depth=NAN_LEVEL).values.any()), "all-NaN level should mask to all-False"
        assert bool(np.isnan(result.thresholds.isel(depth=NAN_LEVEL).values).all()), "all-NaN level should give NaN thresholds"
        assert int(result.extreme_events.isel(depth=NAN_LEVEL).sum()) == 0
        # ... and the real levels are untouched by its presence.
        assert bool(result.mask.isel(depth=0).values.all())

    def test_spatial_dims_are_whole_and_time_is_chunked(self, sst_3d):
        """finalise makes every spatial dim whole -- the extra one included."""
        result = _run(sst_3d, "fixed_baseline", "global_percentile")

        assert result.dat_anomaly.chunksizes["depth"] == (sst_3d.sizes["depth"],)
        assert result.dat_anomaly.chunksizes["lat"] == (sst_3d.sizes["lat"],)
        assert max(result.dat_anomaly.chunksizes["time"]) == TIME_CHUNK

    def test_seasonal_tiling_bounds_the_task_over_all_spatial_dims(self, sst_3d):
        """The histogram tile shrinks with rank instead of multiplying by depth.

        Asserted on the chunk dict the module actually chooses, not on values: a
        rank-blind tiling leaves the extra dimension whole and the horizontal tile
        unchanged, so the per-task cell count is multiplied by the depth length --
        and every value-based check still passes while it does. That is precisely
        the failure mode CLAUDE.md records for the shifted-window rechunk.
        """
        from marEx.extremes.histogram import _HISTOGRAM_TASK_ELEMENTS, _histogram_tile_chunks

        n_bins = 503  # default precision=0.01, max_anomaly=5.0
        ntime = int(sst_3d.sizes["time"])
        budget_cells = _HISTOGRAM_TASK_ELEMENTS // max(ntime, 366 * n_bins)

        cells = {}
        for ndepth in (1, 4, 25, 50):
            da = sst_3d.isel(depth=slice(0, 1)).reindex(depth=np.arange(ndepth, dtype=np.float32), method="nearest")
            tile = _histogram_tile_chunks(da, DIMENSIONS, n_bins, window_spatial=5)
            assert set(tile) == {"time", "lat", "lon", "depth"}, tile
            assert tile["time"] == -1, "the reduced axis must stay whole"
            cells[ndepth] = tile["lat"] * tile["lon"] * tile["depth"]
            assert cells[ndepth] <= budget_cells, f"depth={ndepth}: {cells[ndepth]} cells > budget {budget_cells}"

        # Invariance in the extra dim's length, not an absolute bound: a bound loose
        # enough to pass at depth=1 would also pass while depth=50 allocated 50x.
        assert max(cells.values()) / min(cells.values()) < 8, f"tile grows with the depth axis: {cells}"

        # The horizontal window floor applies, and only to the horizontal dims.
        tile = _histogram_tile_chunks(sst_3d, DIMENSIONS, n_bins, window_spatial=5)
        assert tile["lat"] >= 5 and tile["lon"] >= 5, tile
        assert tile["depth"] <= sst_3d.sizes["depth"], tile


class TestRankGuards:
    """3-D tracking and 3-D rendering are refused, with the fix in the message."""

    def test_tracker_rejects_an_extra_dimension(self, sst_3d):
        binary = (sst_3d > 0).chunk({"time": TIME_CHUNK, "depth": -1, "lat": -1, "lon": -1})
        mask = xr.ones_like(sst_3d.isel(time=0, depth=0, drop=True), dtype=bool)

        with pytest.raises(TrackingError, match="depth"):
            marEx.tracker(binary, mask, R_fill=1, area_filter_quartile=0.5, dimensions=DIMENSIONS)

    def test_tracker_error_names_the_fix(self, sst_3d):
        binary = (sst_3d > 0).chunk({"time": TIME_CHUNK, "depth": -1, "lat": -1, "lon": -1})
        mask = xr.ones_like(sst_3d.isel(time=0, depth=0, drop=True), dtype=bool)

        with pytest.raises(TrackingError) as excinfo:
            marEx.tracker(binary, mask, R_fill=1, area_filter_quartile=0.5, dimensions=DIMENSIONS)
        assert "isel(depth=0)" in str(excinfo.value)

    def test_plotx_rejects_an_extra_dimension(self, sst_3d):
        plotter = sst_3d.isel(time=0).plotX(dimensions=DIMENSIONS, coordinates=DIMENSIONS)
        with pytest.raises(VisualisationError, match="depth"):
            plotter.single_plot(marEx.plotX.PlotConfig(title="3-D"))

    def test_plotx_still_plots_a_selected_level(self, sst_3d):
        """The guard must not block the 2-D field the user is steered towards."""
        level = sst_3d.isel(time=0, depth=0, drop=True)
        plotter = level.plotX(dimensions=DIMENSIONS, coordinates=DIMENSIONS)
        fig, ax, im = plotter.single_plot(marEx.plotX.PlotConfig(title="level 0"))
        assert im is not None
        # Close it. This file sorts first, so a figure left open here stays the
        # current figure for the whole session, and the mocked-pyplot tests in
        # test_plotx.py then get a real Figure back from plt.colorbar's gcf()
        # instead of their MagicMock.
        plt.close(fig)


class TestFinaliseChunksTheFieldNotTheAttachments:
    """``finalise`` makes the FIELD's spatial dims whole -- and nothing else's.

    ``ds.dims`` on the output is the union over every variable, so deriving the
    spatial dims from it would also pick up ``neighbours``' own ``nv`` axis on the
    unstructured path. That axis is not a spatial dimension of the field and must
    not be rechunked here. Neither golden covers it (both are gridded), so it is
    pinned directly.
    """

    def test_neighbour_axis_is_not_treated_as_a_spatial_dim(self):
        from marEx.core.dimensions import extra_dims, resolve_dims

        sst = xr.open_zarr(str(DATA_DIR / "sst_unstructured.zarr"), chunks={}).to.isel(time=slice(0, 3 * 365))
        ncells = sst.sizes["ncells"]
        sst = sst.assign_coords(
            lat=xr.DataArray(np.linspace(-90, 90, ncells), dims=["ncells"]),
            lon=xr.DataArray(np.linspace(-180, 180, ncells), dims=["ncells"]),
        ).chunk({"time": 30, "ncells": -1})
        dims = {"time": "time", "x": "ncells"}
        coords = {"time": "time", "x": "lon", "y": "lat"}
        neighbours = xr.DataArray(
            np.zeros((3, ncells), dtype=np.int32),
            dims=("nv", "ncells"),
            coords={"nv": np.arange(3)},
        )
        cell_areas = xr.DataArray(np.ones(ncells, dtype=np.float32), dims=("ncells",))

        result = marEx.preprocess_data(
            sst,
            method_anomaly="fixed_baseline",
            method_extreme="global_percentile",
            threshold_percentile=95,
            dimensions=dims,
            coordinates=coords,
            neighbours=neighbours,
            cell_areas=cell_areas,
            dask_chunks={"time": TIME_CHUNK},
        )

        # The field has no extra dims, so finalise must chunk exactly ncells and time.
        assert resolve_dims(sst, dims, coords).extra == ()
        # `nv` is visible on the output Dataset but is not the field's spatial dim.
        assert "nv" in result.dims
        assert "nv" in extra_dims(result, dims, exclude=("dayofyear",))
        assert result.neighbours.chunksizes["nv"] == (3,)
        assert result.dat_anomaly.chunksizes["ncells"] == (ncells,)
