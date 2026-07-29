"""Animation: frame geometry, centroid markers, and the persisted-coordinate fix that keeps frames computable.

Frame geometry has two properties that pull against each other: every frame in one movie has identical
pixel dimensions (h264 rejects a stream whose size changes), and the domain should fill the frame.
``_domain_figsize`` resolves both by deriving one canvas aspect from the projected extent of the
domain, up front.

The persisted-coordinate fix guards against a graph-construction bug in the xarray/dask/distributed
stack: a dask-backed coordinate that does not vary with time, shared by every per-frame
``dask.delayed`` call, is submitted with an unresolvable dependency and cancels every frame with
``FutureCancelledError ... lost dependencies``. ``_animate`` persists such coordinates once, before the
frame loop. ``persist`` is used rather than ``compute`` because a computed coordinate would be embedded
as a literal in every per-frame task.

Tests that need a distributed scheduler request ``client``; the rest run threaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cartopy.crs as ccrs
import dask
import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

from marEx.plotX import animation as animation_mod
from marEx.plotX.animation import (
    _DEFAULT_FIG_ASPECT,
    _FIG_WIDTH_INCHES,
    _FRAME_DPI,
    _MAX_FRAME_PIXELS,
    _MIN_FIG_ASPECT,
    _domain_figsize,
    _even_height,
    _frame_centroids,
    _materialise_dask_coords,
    make_frame,
)
from marEx.plotX.config import PlotConfig

from ..conftest import assert_no_failures, check, on_client

GLOBAL_LON = np.arange(-179.5, 180.0, 1.0)
GLOBAL_LAT = np.arange(-89.5, 90.0, 1.0)
REGIONAL_LON = np.arange(-80.0, 20.0, 0.25)
REGIONAL_LAT = np.arange(-10.0, 60.0, 0.25)
# 10 deg lon by 80 deg lat: an ordinary regional strip that projects to an aspect near 8, past the pixel cap.
TALL_NARROW = (np.arange(-5.0, 5.0, 0.1), np.arange(-40.0, 40.0, 0.1))

DEFAULT_SIZE = (_FIG_WIDTH_INCHES, _even_height(_FIG_WIDTH_INCHES * _DEFAULT_FIG_ASPECT))


def _pixels(figsize):
    return round(figsize[0] * _FRAME_DPI), round(figsize[1] * _FRAME_DPI)


def _zarr_values(seed, n_time, n_cells):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 5, size=(n_time, n_cells)).astype("int32")
    lat = rng.uniform(-90, 90, size=n_cells)
    return ids, lat


def _build_zarr_da(tmp_path, name, seed=0, n_time=4, n_cells=100):
    """A real zarr-backed DataArray with one dask coordinate that varies with nothing but the cells.

    It must come from a real zarr store: xarray's zarr backend is what constructs the vulnerable
    coordinate variables. ``name`` and ``seed`` pick a distinct store path and distinct values, so two
    calls give independently tokenised dask coordinates (persisting one must not persist the other).
    """
    ids, lat = _zarr_values(seed, n_time, n_cells)
    ds = xr.Dataset(
        {"ID": (("time", "ncells"), ids)},
        coords={"lat": ("ncells", lat), "time": np.datetime64("2020-01-01") + np.arange(n_time)},
    )
    path = Path(tmp_path) / f"{name}.zarr"
    ds.chunk({"time": 1, "ncells": -1}).to_zarr(path, mode="w")
    return xr.open_zarr(path)["ID"]


def _is_persisted(dask_array):
    """A persisted dask array's graph holds exactly one task per chunk and nothing else."""
    return len(dict(dask_array.__dask_graph__())) == dask_array.npartitions


def _touch(slice_):
    return slice_.values.sum()


class _StubPlotter:
    """What ``_animate`` reads from a plotter, without any plotting."""

    def __init__(self, da):
        self.da = da
        self.dimensions = {"time": "time"}
        self.coordinates = {"time": "time", "y": "lat", "x": "lon"}

    def _setup_common_params(self, config):
        return "viridis", None, None, "", "both"


def _stub_animate_deps(monkeypatch, make_frame_task):
    """Stub ffmpeg and frame drawing; return the list that records each frame set ffmpeg was handed."""
    monkeypatch.setattr(animation_mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    encoded = []

    def fake_run(cmd, *args, **kwargs):
        frames = Path(cmd[cmd.index("-i") + 1]).parent
        encoded.append(sorted(p.name for p in frames.glob("time_*.jpg")))

    monkeypatch.setattr(animation_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(animation_mod, "make_frame", make_frame_task)
    return encoded


@dask.delayed
def _frame_that_touches_everything(data_slice, time_ind, temp_dir, plot_params, grid_info):
    # The same touch the real frame makes; it is what trips the upstream bug. Tasks may run in
    # worker processes, so nothing is recorded in the test process: a frame that fails raises
    # out of ``_animate``'s ``dask.compute``, and the files ffmpeg is handed prove the others ran.
    data_slice.values
    if plot_params["object_ids"] is not None:
        plot_params["object_ids"].values
    filename = f"time_{time_ind:04d}.jpg"
    (Path(temp_dir) / filename).write_bytes(b"\x00")
    return filename


@pytest.mark.slow
def test_animate_with_where_cut_coords_over_several_batches(client, monkeypatch, tmp_path):
    """``_animate`` on where-cut fields: every frame computes across several batches and sees its own IDs.

    Without an ID field every frame is handed ``object_ids=None`` explicitly, not a missing key.

    The ID field comes from its own store: a persisted coordinate is registered under its content-
    addressed token, so an ID field built from the main field's store would ride on the main field's
    persist and the test would stop discriminating. ``plot_IDs`` is checked inside the task, because
    the frame function only knows the field holds IDs if ``_animate`` says so.
    """
    assert on_client(client)
    n_time, n_cells = 6, 100
    monkeypatch.setattr(animation_mod, "_FRAME_BATCH_SIZE", 2)
    batches = []
    real_compute = dask.compute

    def spy_compute(*args, **kwargs):
        batches.append(len(args))
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(animation_mod.dask, "compute", spy_compute)
    failures = []

    def run(plot_IDs, with_ids=True):
        ids_values, lat = _zarr_values(4, n_time, n_cells)
        expected_ids = np.where(lat < 45.0, ids_values, 0)

        @dask.delayed
        def checking_frame(data_slice, time_ind, temp_dir, plot_params, grid_info):
            data_slice.values
            if plot_params["plot_IDs"] is not plot_IDs:
                raise AssertionError(f"plot_IDs={plot_params['plot_IDs']!r} reached frame {time_ind}, expected {plot_IDs!r}")
            if not with_ids:
                if "object_ids" not in plot_params or plot_params["object_ids"] is not None:
                    raise AssertionError(f"frame {time_ind} without an ID field was not handed object_ids=None")
            elif not np.array_equal(plot_params["object_ids"].values, expected_ids[time_ind]):
                raise AssertionError(f"frame {time_ind} was handed the wrong ID slice")
            filename = f"time_{time_ind:04d}.jpg"
            (Path(temp_dir) / filename).write_bytes(b"\x00")
            return filename

        encoded = _stub_animate_deps(monkeypatch, checking_frame)
        main = _build_zarr_da(tmp_path, f"main_{plot_IDs}_{with_ids}", seed=3, n_time=n_time, n_cells=n_cells)
        object_ids = None
        if with_ids:
            ids_da = _build_zarr_da(tmp_path, f"ids_{plot_IDs}", seed=4, n_time=n_time, n_cells=n_cells)
            object_ids = ids_da.where(ids_da.lat < 45.0, 0)
        plotter = _StubPlotter(main.where(main.lat < 45.0, 0))
        batches.clear()
        result = animation_mod._animate(plotter, PlotConfig(plot_IDs=plot_IDs), plot_dir=str(tmp_path), object_ids=object_ids)
        assert result is not None and result.endswith(".mp4"), f"result {result!r}"
        assert batches == [2, 2, 2], f"frames were computed in batches of {batches}, expected [2, 2, 2]"
        want = [f"time_{i:04d}.jpg" for i in range(n_time)]
        assert encoded == [want], f"ffmpeg was handed {encoded}, expected one set {want}"

    for plot_IDs in (True, False):
        check(failures, f"plot_IDs={plot_IDs}", lambda plot_IDs=plot_IDs: run(plot_IDs))
    check(failures, "no ID field", lambda: run(False, with_ids=False))
    assert_no_failures(failures)


@pytest.mark.slow
def test_animate_without_materialising_loses_dependencies(client, monkeypatch, tmp_path):
    """Without the persisted coordinates ``_animate`` fails with the upstream error: for either array.

    The raw probe checks the upstream bug still exists; when it does not, the fix may be removable and
    the test says so instead of failing. The two rows then stub the fix out for every array and for the
    ID field only, each on its own store, and both must fail the same way.
    """
    from distributed.client import FutureCancelledError

    assert on_client(client)
    probe_da = _build_zarr_da(tmp_path, "probe", seed=0)
    cut = probe_da.where(probe_da.lat < 45.0, 0)
    tasks = [dask.delayed(_touch)(cut.isel(time=i)) for i in range(cut.sizes["time"])]
    try:
        dask.compute(*tasks)
        reproduced = False
    except FutureCancelledError:
        reproduced = True
    del tasks, cut, probe_da
    if not reproduced:
        pytest.xfail("upstream fixed; _materialise_dask_coords may be removable")

    real = animation_mod._materialise_dask_coords
    failures = []

    def run(case, noop_for):
        _stub_animate_deps(monkeypatch, _frame_that_touches_everything)
        main = _build_zarr_da(tmp_path, f"main_{case}", seed=1)
        ids_da = _build_zarr_da(tmp_path, f"ids_{case}", seed=2)
        object_ids = ids_da.where(ids_da.lat < 45.0, 0)

        def stubbed(da, time_dim):
            return da if noop_for(da, object_ids) else real(da, time_dim)

        monkeypatch.setattr(animation_mod, "_materialise_dask_coords", stubbed)
        plotter = _StubPlotter(main.where(main.lat < 45.0, 0))
        with pytest.raises(FutureCancelledError):
            animation_mod._animate(plotter, PlotConfig(), plot_dir=str(tmp_path), object_ids=object_ids)

    check(failures, "no coordinate persisted", lambda: run("all", lambda da, ids: True))
    check(failures, "only the ID field left unpersisted", lambda: run("ids", lambda da, ids: da is ids))
    assert_no_failures(failures)


def test_canvas_aspect_follows_projected_domain():
    """The canvas aspect tracks the projection and the domain; a global map is no longer drawn on 7x5."""
    failures = []

    def global_robinson():
        figsize = _domain_figsize(ccrs.Robinson(), GLOBAL_LON, GLOBAL_LAT, show_colorbar=False)
        aspect = figsize[1] / figsize[0]
        assert aspect < _DEFAULT_FIG_ASPECT, f"aspect {aspect} is not wider than the old fixed canvas"
        assert 0.40 < aspect < 0.62, f"Robinson draws the globe near 2:1, got canvas aspect {aspect}"

    def platecarree_tracks_the_box():
        lon = np.arange(-100.0, 100.0, 1.0)  # 199 deg wide
        lat = np.arange(-25.0, 25.0, 1.0)  # 49 deg tall
        figsize = _domain_figsize(ccrs.PlateCarree(), lon, lat, show_colorbar=False)
        axes_ratio = (figsize[1] * 0.77) / (figsize[0] * 0.775)
        assert axes_ratio == pytest.approx(np.ptp(lat) / np.ptp(lon), rel=0.02)

    def polar_stereographic_follows_the_projected_extent():
        """The raw lon/lat box of this strip is 0.24; a polar stereographic map of it is about twice as tall."""
        lon, lat = np.arange(-60.0, 60.0, 1.0), np.arange(50.0, 80.0, 1.0)
        figsize = _domain_figsize(ccrs.NorthPolarStereo(), lon, lat, show_colorbar=False)
        aspect = figsize[1] / figsize[0]
        assert aspect > 0.4, f"canvas aspect {aspect} follows the lon/lat box ({np.ptp(lat) / np.ptp(lon):.2f}), not the projection"

    def taller_domain_is_taller():
        wide = _domain_figsize(ccrs.PlateCarree(), np.arange(-180.0, 180.0, 1.0), np.arange(-10.0, 10.0, 1.0), False)
        tall = _domain_figsize(ccrs.PlateCarree(), np.arange(-20.0, 20.0, 1.0), np.arange(-80.0, 80.0, 1.0), False)
        assert tall[1] > wide[1], f"tall domain canvas {tall} is not taller than wide {wide}"

    def colorbar_shortens_the_canvas():
        with_cb = _domain_figsize(ccrs.Robinson(), GLOBAL_LON, GLOBAL_LAT, show_colorbar=True)
        without_cb = _domain_figsize(ccrs.Robinson(), GLOBAL_LON, GLOBAL_LAT, show_colorbar=False)
        assert with_cb[1] < without_cb[1], f"with colorbar {with_cb}, without {without_cb}"

    check(failures, "global Robinson is wide", global_robinson)
    check(failures, "PlateCarree tracks the lon/lat box", platecarree_tracks_the_box)
    check(failures, "polar stereographic strip", polar_stereographic_follows_the_projected_extent)
    check(failures, "taller domain, taller canvas", taller_domain_is_taller)
    check(failures, "colorbar narrows the map", colorbar_shortens_the_canvas)
    assert_no_failures(failures)


@pytest.mark.filterwarnings("ignore:All-NaN slice")
def test_canvas_falls_back_to_default(marex_records):
    """Anything the sizing cannot measure degrades to the previous fixed canvas instead of raising, and says why."""

    class ExplodingProjection:
        def transform_points(self, *args, **kwargs):
            raise RuntimeError("projection blew up")

    class NonFiniteProjection:
        def transform_points(self, src, x, y):
            return np.full((x.size, 3), np.inf)

    rows = {
        "no coordinates": (ccrs.Robinson(), None, None),
        "no latitude": (ccrs.Robinson(), GLOBAL_LON, None),
        "no longitude": (ccrs.Robinson(), None, GLOBAL_LAT),
        "empty": (ccrs.Robinson(), np.array([]), np.array([])),
        "single point": (ccrs.Robinson(), np.array([10.0]), np.array([10.0])),
        "all NaN": (ccrs.Robinson(), np.array([np.nan, np.nan]), np.array([np.nan, np.nan])),
        "projection raises": (ExplodingProjection(), GLOBAL_LON, GLOBAL_LAT),
        "projection returns non-finite points": (NonFiniteProjection(), GLOBAL_LON, GLOBAL_LAT),
    }
    # The logged reason for the fallback, where the row pins one.
    reasons = {
        "projection raises": "RuntimeError while projecting the domain",
        "projection returns non-finite points": "projection returned no finite points for this domain",
    }
    records = marex_records(logging.DEBUG)
    failures = []
    for case, (proj, lon, lat) in rows.items():

        def row(case=case, proj=proj, lon=lon, lat=lat):
            start = len(records)
            got = _domain_figsize(proj, lon, lat, show_colorbar=True)
            assert got == DEFAULT_SIZE, f"got {got}, expected {DEFAULT_SIZE}"
            if case in reasons:
                logged = [r.getMessage() for r in records[start:] if "fell back to the default canvas" in r.getMessage()]
                assert len(logged) == 1 and f"({reasons[case]})" in logged[0], f"fallback reason logged: {logged}"

        check(failures, case, row)
    assert_no_failures(failures)


def test_canvas_invariants_hold_for_every_domain():
    """Across projections, domains and colorbar settings: constant width, aspect floor, even pixels, pixel cap."""
    domains = {"global": (GLOBAL_LON, GLOBAL_LAT), "regional": (REGIONAL_LON, REGIONAL_LAT), "tall narrow": TALL_NARROW}
    projections = {
        "Robinson": ccrs.Robinson(),
        "PlateCarree": ccrs.PlateCarree(),
        "Mollweide": ccrs.Mollweide(),
        "EqualEarth": ccrs.EqualEarth(),
    }
    failures = []
    for pname, proj in projections.items():
        for dname, (lon, lat) in domains.items():
            for colorbar in (True, False):

                def row(proj=proj, lon=lon, lat=lat, colorbar=colorbar):
                    figsize = _domain_figsize(proj, lon, lat, colorbar)
                    assert figsize[0] == _FIG_WIDTH_INCHES, f"width {figsize[0]} is not held at {_FIG_WIDTH_INCHES}"
                    assert figsize[1] / figsize[0] >= _MIN_FIG_ASPECT, f"aspect {figsize[1] / figsize[0]} is under the floor"
                    width_px, height_px = _pixels(figsize)
                    assert height_px % 2 == 0, f"odd pixel height {height_px}"
                    assert width_px <= _MAX_FRAME_PIXELS and height_px <= _MAX_FRAME_PIXELS, f"{width_px}x{height_px} px"
                    assert _domain_figsize(proj, lon, lat, colorbar) == figsize, "same domain gave a different canvas"

                check(failures, f"{pname}/{dname}/colorbar={colorbar}", row)

    def cap_binds():
        """If this stops clamping, the cap rows above prove nothing."""
        _, height_px = _pixels(_domain_figsize(ccrs.PlateCarree(), *TALL_NARROW, show_colorbar=False))
        assert height_px == _MAX_FRAME_PIXELS, f"tall narrow PlateCarree is {height_px} px tall, expected the cap"

    def floor_binds():
        """A zonal band far flatter than the floor lands on it; without this the floor rows prove nothing."""
        figsize = _domain_figsize(ccrs.PlateCarree(), np.arange(-179.5, 180.0, 1.0), np.arange(-2.5, 2.5, 1.0), False)
        assert abs(figsize[1] / figsize[0] - _MIN_FIG_ASPECT) <= 2.0 / (_FRAME_DPI * _FIG_WIDTH_INCHES), f"canvas {figsize}"

    check(failures, "the cap binds for the tall narrow domain", cap_binds)
    check(failures, "the floor binds for a zonal band", floor_binds)
    assert_no_failures(failures)


def test_even_height():
    """h264 needs even dimensions: the pixel height is always even, within one pixel pair of the request, and positive."""
    failures = []
    for inches in (0.001, 1.0, 2.8655, 3.57, 5.0, 7.13, 12.0):

        def lands_even(inches=inches):
            assert round(_even_height(inches) * _FRAME_DPI) % 2 == 0, f"{_even_height(inches) * _FRAME_DPI} px is odd"

        check(failures, f"{inches} in is even", lands_even)
    for inches in (1.0, 2.8655, 3.57, 7.13):

        def stays_close(inches=inches):
            moved = abs(_even_height(inches) - inches)
            assert moved <= 2.0 / _FRAME_DPI, f"moved by {moved} in"

        check(failures, f"{inches} in stays close", stays_close)
    for inches in (0.0, -5.0):

        def clamped(inches=inches):
            assert _even_height(inches) > 0, f"got {_even_height(inches)}"

        check(failures, f"{inches} in is clamped positive", clamped)
    assert_no_failures(failures)


def test_frame_centroids_follow_the_id_field():
    """Centroid markers belong to the objects present in the frame, taken from the ID field and not the values."""
    # (component, ID): component 0 = lat, 1 = lon; ID 2 is absent at this time step (NaN).
    values = np.array([[10.0, np.nan, 30.0], [100.0, np.nan, 300.0]])
    centroids = xr.DataArray(values, dims=("component", "ID"), coords={"ID": [1, 2, 3]})
    rows = {
        "only ID 3 present": (np.array([[0, 3], [3, 0]]), [(300.0, 30.0)]),
        "NaN background of a masked ID field": (np.array([[np.nan, 3.0], [np.nan, np.nan]]), [(300.0, 30.0)]),
        "no ID field": (None, [(100.0, 10.0), (300.0, 30.0)]),
    }
    failures = []
    for case, (object_ids, expected) in rows.items():

        def row(object_ids=object_ids, expected=expected):
            got = _frame_centroids(centroids, object_ids)
            assert got == expected, f"got {got}, expected {expected}"

        check(failures, case, row)
    assert_no_failures(failures)


def test_make_frame_picks_the_centroid_id_source(monkeypatch, tmp_path):
    """``make_frame`` takes the marker IDs from ``object_ids``, never from the animated values of an anomaly movie."""
    lat = np.linspace(-10, 10, 4)
    lon = np.linspace(0, 30, 4)
    animated = np.array([[1.0, 3.0, 0.5, -1.0]] * 4)
    anomaly = xr.DataArray(animated, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    ids = np.zeros((4, 4), dtype=int)
    ids[0, 0] = 2
    object_ids = xr.DataArray(ids, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    centroids = xr.DataArray(np.array([[0.0, 5.0, 8.0], [10.0, 15.0, 20.0]]), dims=("component", "ID"), coords={"ID": [1, 2, 3]})
    seen = []

    def spy(centroids_, id_field):
        seen.append(None if id_field is None else np.array(id_field))
        return _frame_centroids(centroids_, id_field)

    monkeypatch.setattr(animation_mod, "_frame_centroids", spy)

    def frame_id_field(plot_IDs, with_object_ids):
        seen.clear()
        plot_params = {
            "cmap": "viridis",
            "norm": None,
            "clim": (-1.0, 3.0),
            "var_units": "",
            "extend": "both",
            "show_colorbar": False,
            "grid_labels": False,
            "projection": None,
            "x_coord": "lon",
            "y_coord": "lat",
            "plot_IDs": plot_IDs,
            "centroids": centroids,
            "object_ids": object_ids if with_object_ids else None,
        }
        make_frame(anomaly, 0, tmp_path, plot_params).compute(scheduler="synchronous")
        assert len(seen) == 1, f"the ID source was read {len(seen)} times"
        return seen[0]

    failures = []

    def anomaly_with_ids():
        got = frame_id_field(plot_IDs=False, with_object_ids=True)
        np.testing.assert_array_equal(got, ids)

    def anomaly_without_ids():
        got = frame_id_field(plot_IDs=False, with_object_ids=False)
        assert got is None, f"got {got}, expected no ID field"

    def id_movie_without_ids():
        got = frame_id_field(plot_IDs=True, with_object_ids=False)
        np.testing.assert_array_equal(got, animated)

    check(failures, "anomaly movie with object_ids", anomaly_with_ids)
    check(failures, "anomaly movie without object_ids", anomaly_without_ids)
    check(failures, "ID movie without object_ids", id_movie_without_ids)
    assert_no_failures(failures)


def test_materialise_persists_only_shared_dask_coords(tmp_path):
    """Shared dask coordinates are persisted (still dask, not computed); everything else is left as it was."""
    da = _build_zarr_da(tmp_path, "store", seed=0)
    before = da.lat.values
    time_varying = da.assign_coords(frame_label=("time", dsa.arange(da.sizes["time"], chunks=1) * 2))
    failures = []

    def shared_coordinate_is_persisted():
        assert hasattr(da.lat.data, "dask") and not _is_persisted(da.lat.data), "premise: lat starts as a lazy graph"
        out = _materialise_dask_coords(da, time_dim="time")
        assert hasattr(out.lat.data, "dask"), "the persisted coordinate became numpy; it would be embedded in every frame"
        assert _is_persisted(out.lat.data), "the coordinate still carries its lazy lineage"
        np.testing.assert_array_equal(out.lat.values, before)
        assert hasattr(out.data, "dask"), "the animated field itself must stay lazy"

    def time_varying_coordinate_is_untouched():
        assert not _is_persisted(time_varying.frame_label.data), "premise: frame_label starts as a lazy graph"
        out = _materialise_dask_coords(time_varying, time_dim="time")
        assert hasattr(out.frame_label.data, "dask"), "a time-varying coordinate was computed"
        assert not _is_persisted(out.frame_label.data), "a time-varying coordinate was persisted"

    def no_dask_coordinates_is_a_noop():
        plain = xr.DataArray(np.zeros((2, 3)), dims=("time", "x"), coords={"x": [1, 2, 3]})
        assert _materialise_dask_coords(plain, time_dim="time") is plain

    check(failures, "shared coordinate", shared_coordinate_is_persisted)
    check(failures, "time-varying coordinate", time_varying_coordinate_is_untouched)
    check(failures, "no dask coordinates", no_dask_coordinates_is_a_noop)
    assert_no_failures(failures)


def test_per_frame_slice_does_not_grow_with_coord_size(client, tmp_path):
    """A per-frame slice's pickled size stays near the unfixed baseline instead of scaling with the coordinate.

    ``compute`` would embed the coordinate in every frame's task (0.01 MB to 4.80 MB at 300k cells), which
    crashed on graph size at ICON scale. The scenario needs an active distributed client: with only the
    local scheduler ``persist`` has nowhere else to keep the data and embeds it exactly like ``compute``.
    """
    import cloudpickle

    assert on_client(client)
    da = _build_zarr_da(tmp_path, "big", seed=2, n_time=2, n_cells=300_000)
    baseline = len(cloudpickle.dumps(da.isel(time=0)))
    fixed = _materialise_dask_coords(da, time_dim="time")
    fixed_size = len(cloudpickle.dumps(fixed.isel(time=0)))
    computed_size = len(cloudpickle.dumps(da.assign_coords(lat=da.lat.compute()).isel(time=0)))
    del fixed

    assert fixed_size < 2 * baseline, f"persisted slice grew to {fixed_size} bytes vs baseline {baseline}"
    assert computed_size > 10 * fixed_size, "the compute-based regression this guards against did not reproduce"
