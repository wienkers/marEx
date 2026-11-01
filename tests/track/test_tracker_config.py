"""Tracker construction: validation and normalisation (including the rank guard), user-facing errors,
chunking warnings, the checkpoint round trip and the single-use guard, the coordinate state, single-step
records, and the shared futures tripwire.

Each row runs on the scheduler its pre-consolidation test ran on: a test that requests ``client``
wraps its threaded-origin rows in the conftest pin (``THREADED``).
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import time
import warnings
from pathlib import Path

import dask
import numpy as np
import pytest
import xarray as xr
from dask import is_dask_collection

import marEx
from marEx.exceptions import ConfigurationError, CoordinateError, DataValidationError, TrackingError
from marEx.track.grid import unify_coordinates
from marEx.track.validation import validate_unstructured_chunking

from ..conftest import released
from .conftest import (
    CHUNK_SIZE,
    MODE_KWARGS,
    OPENERS,
    THREADED,
    assert_no_failures,
    check,
    on_client,
    open_gridded,
    open_unstructured,
    polar_mask,
)

MESH = {
    "unstructured_grid": True,
    "dimensions": {"x": "ncells"},
    "coordinates": {"x": "lon", "y": "lat"},
    "coordinate_units": "degrees",
}


def test_shared_fixtures_hold_no_futures(client):
    """Re-adding ``.persist()`` to an opener pins it to whichever client was current when it ran, and the
    tests fail with ``lost dependencies`` once that client closes."""
    from distributed.client import futures_of

    assert on_client(client)
    for name, opener in OPENERS.items():
        held = futures_of(opener())
        assert not held, f"{name}() holds {len(held)} future(s); keep it rooted in the zarr store"


def _small_grid(lat=None, lon=None, attrs=None):
    data = xr.DataArray(
        np.random.default_rng(0).random((10, 5, 4)) > 0.7,
        dims=["time", "lat", "lon"],
        coords={
            "time": range(10),
            "lat": np.linspace(-20, 20, 5) if lat is None else lat,
            "lon": np.linspace(-10, 10, 4) if lon is None else lon,
        },
        attrs={} if attrs is None else attrs,
    )
    return data


def _mesh_tracker(ds, temp_dir, **kwargs):
    params = dict(
        MESH, area_filter_quartile=0.5, R_fill=0, T_fill=0, neighbours=ds.neighbours, cell_areas=ds.cell_areas, quiet=True
    )
    params.update(kwargs)
    if temp_dir is not None:
        params["temp_dir"] = str(temp_dir)
    return marEx.tracker(ds.extreme_events, ds.mask, **params)


def test_checkpoint_save_and_load(client, tmp_path):
    """checkpoint='save' writes the store and stats and returns the computed stats; 'load' returns the saved values."""
    assert on_client(client)
    ds = open_unstructured()
    tracker = _mesh_tracker(ds, tmp_path)
    saved, saved_stats = tracker.run_preprocess(checkpoint="save")
    checkpoint = Path(tracker.scratch_dir) / "marEx_checkpoint_proc_bin.zarr"
    assert checkpoint.exists(), "Checkpoint zarr file not created"
    assert (Path(tracker.scratch_dir) / "marEx_checkpoint_stats.npz").exists(), "Checkpoint stats file not created"
    assert isinstance(saved_stats, tuple) and len(saved_stats) == 6
    assert "data_bin_preproc" in xr.open_zarr(str(checkpoint))
    saved_values = np.asarray(saved.compute().values)
    # Drop the trackers and lazy results before asserting: a failure's traceback keeps them, and
    # their futures, alive past the module's leak check.
    del tracker, saved

    # run_preprocess is single-use per tracker, so build one instance per run.
    plain, plain_stats = _mesh_tracker(ds, tmp_path).run_preprocess()
    del plain
    for i, (s, p) in enumerate(zip(saved_stats, plain_stats)):
        assert not isinstance(s, np.ndarray), f"stats[{i}] is a numpy array from the npz round-trip: {s!r}"
        assert type(s) is type(p), f"stats[{i}] type {type(s)} differs from uncheckpointed {type(p)}"
        assert s == pytest.approx(p), f"stats[{i}] value {s} differs from uncheckpointed {p}"

    loaded, loaded_stats = _mesh_tracker(ds, tmp_path).run_preprocess(checkpoint="load")
    loaded_values = np.asarray(loaded.compute().values)
    del loaded
    assert np.array_equal(loaded_values, saved_values), "loaded field values differ from the saved ones"
    assert len(loaded_stats) == len(saved_stats)
    for i, (s, v) in enumerate(zip(saved_stats, loaded_stats)):
        if isinstance(s, (int, np.integer)):
            assert s == v, f"Stats element {i} doesn't match"
        else:
            np.testing.assert_allclose(s, v, rtol=1e-10)


GEOMETRY_ROWS = [
    ("neighbours_ncells100", {"neighbours": {"ncells": 100}}, "Neighbours spatial dimension 'ncells' has multiple chunks"),
    ("neighbours_nv1", {"neighbours": {"nv": 1}}, "Neighbours dimension 'nv' has multiple chunks"),
    ("neighbours_nv2", {"neighbours": {"ncells": -1, "nv": 2}}, "Neighbours dimension 'nv' has multiple chunks"),
    ("cell_areas_ncells100", {"cell_areas": {"ncells": 100}}, "Cell areas spatial dimension 'ncells' has multiple chunks"),
    ("single_chunk", {}, None),
]


def test_unstructured_geometry_chunking_warns_and_is_rechunked(client, tmp_path):
    """Multi-chunk neighbours or cell_areas warn by name; the tracker's cell_area ends up contiguous with equal values."""
    assert on_client(client)
    ds = open_unstructured()
    failures = []
    for name, chunking, match in GEOMETRY_ROWS:

        def row(name=name, chunking=chunking, match=match):
            geometry = {key: getattr(ds, key).chunk(spec) for key, spec in chunking.items()}
            for key, arr in geometry.items():
                assert any(len(c) > 1 for c in arr.chunks), f"precondition: {key} must be multi-chunk"
            if match is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    tracker = _mesh_tracker(ds, tmp_path / name, **geometry)
            else:
                with pytest.warns(UserWarning, match=match):
                    tracker = _mesh_tracker(ds, tmp_path / name, **geometry)
            if "cell_areas" in geometry:
                chunksizes = tracker.cell_area.chunksizes
                if "ncells" in chunksizes:
                    assert len(chunksizes["ncells"]) == 1, f"tracker.cell_area still has {len(chunksizes['ncells'])} chunks"
                else:
                    assert not is_dask_collection(getattr(tracker.cell_area, "data", None))
                np.testing.assert_allclose(np.asarray(tracker.cell_area.values), ds.cell_areas.values)

        check(failures, name, row)
    assert_no_failures(failures)


def test_valid_inputs_are_normalised(client):
    """Empty attrs stay empty; radian coordinates are detected and converted exactly; regional_tracker is regional."""
    assert on_client(client)
    failures = []

    def store_without_attrs():
        ds = open_gridded()
        data = ds.extreme_events.copy(deep=True)
        data.attrs = {}
        assert marEx.tracker(data, ds.mask, area_filter_quartile=0.5, R_fill=4, quiet=True).data_attrs == {}

    def synthetic_without_attrs():
        for case, data in (("empty", _small_grid(attrs={})), ("cleared", _small_grid(attrs={"a": 1}))):
            data.attrs.clear()
            data = data.chunk({"time": 5})
            mask = xr.ones_like(data.isel(time=0), dtype=bool)
            tracker = marEx.tracker(data, mask, R_fill=2, area_filter_quartile=0.5, regional_mode=True, coordinate_units="degrees")
            assert tracker.data_attrs == {}, f"{case}: attrs {tracker.data_attrs}"

    def radians():
        lat_rad = np.linspace(-np.pi / 2, np.pi / 2, 5)
        for lon_rad in (np.linspace(0, 2 * np.pi, 4), np.linspace(-np.pi, np.pi, 4)):
            data = _small_grid(lat=lat_rad, lon=lon_rad).chunk({"time": 5})
            tracker = marEx.tracker(data, xr.ones_like(data.isel(time=0), dtype=bool), R_fill=2, area_filter_quartile=0.5)
            assert tracker.coordinate_units == "radians"
            np.testing.assert_allclose(tracker.data_bin["lat"].values, np.degrees(lat_rad), err_msg="lat not converted to degrees")
            np.testing.assert_allclose(tracker.data_bin["lon"].values, np.degrees(lon_rad), err_msg="lon not converted to degrees")

    def regional():
        data = _small_grid().chunk({"time": 5})
        tracker = marEx.regional_tracker(
            data, xr.ones_like(data.isel(time=0), dtype=bool), R_fill=2, area_filter_absolute=10.0, coordinate_units="degrees"
        )
        assert hasattr(tracker, "run")
        assert tracker.regional_mode is True
        assert tracker.area_filter_absolute == 10.0

    check(failures, "store_without_attrs", store_without_attrs)
    with dask.config.set(THREADED):
        for name, row in (("synthetic_without_attrs", synthetic_without_attrs), ("radians", radians), ("regional", regional)):
            check(failures, name, row)
    assert_no_failures(failures)


# -- the rejection table: (exception, match, builder, pre-consolidation scheduler) ------------


def _missing_mesh_coords(tmp_path):
    data = xr.DataArray(
        np.random.default_rng(0).random((10, 100)) > 0.5,
        dims=["time", "ncells"],
        coords={"time": np.arange(10), "ncells": np.arange(100)},
    ).chunk({"time": 5})
    mask = xr.DataArray(np.ones(100, dtype=bool), dims=["ncells"], coords={"ncells": np.arange(100)})
    marEx.tracker(
        data, mask, R_fill=0, unstructured_grid=True, dimensions={"x": "ncells"}, coordinates={"x": "lon", "y": "lat"}, quiet=True
    )


def _grid(**kwargs):
    data = _small_grid().chunk({"time": 5})
    marEx.tracker(
        data, xr.ones_like(data.isel(time=0), dtype=bool), R_fill=2, regional_mode=True, coordinate_units="degrees", **kwargs
    )


def _not_dask(tmp_path):
    data = _small_grid()
    marEx.tracker(
        data,
        xr.ones_like(data.isel(time=0), dtype=bool),
        R_fill=2,
        area_filter_quartile=0.5,
        regional_mode=True,
        coordinate_units="degrees",
    )


def _mode(**kwargs):
    ds = open_gridded()
    marEx.tracker(ds.extreme_events.chunk(CHUNK_SIZE), ds.mask, **kwargs, **MODE_KWARGS)


def _prefilter(bad):
    ds = open_gridded()
    marEx.tracker(
        ds.extreme_events.chunk({"time": 4}),
        ds.mask,
        prefilter_min_cells=bad,
        R_fill=4,
        area_filter_absolute=30,
        T_fill=2,
        allow_merging=True,
        nn_partitioning=True,
        quiet=True,
    )


def _bad_cell_areas(extra_dim):
    ds = open_gridded()
    if extra_dim:
        areas = xr.DataArray(
            np.ones((ds.sizes["lat"], ds.sizes["lon"], 2)),
            coords={"lat": ds.lat, "lon": ds.lon, "extra": [0, 1]},
            dims=["lat", "lon", "extra"],
        )
    else:
        areas = xr.DataArray(np.ones(ds.sizes["lon"]), coords={"lon": ds.lon}, dims=["lon"])
    marEx.tracker(ds.extreme_events, ds.mask, R_fill=4, cell_areas=areas, quiet=True)


def _grid_resolution(value):
    ds = open_gridded()
    marEx.tracker(ds.extreme_events, ds.mask, R_fill=4, grid_resolution=value, quiet=True)


def _with_depth():
    """A gridded field with an extra (depth) dimension: tracking runs on one horizontal level."""
    data = xr.DataArray(
        np.zeros((4, 2, 5, 6), dtype=bool),
        dims=("time", "depth", "lat", "lon"),
        coords={"time": np.arange(4), "depth": [0.0, 1.0], "lat": np.linspace(-10, 10, 5), "lon": np.linspace(0, 50, 6)},
    ).chunk({"time": 2})
    mask = xr.ones_like(data.isel(time=0, depth=0, drop=True), dtype=bool)
    marEx.tracker(data, mask, R_fill=1, area_filter_quartile=0.5, dimensions={"time": "time", "x": "lon", "y": "lat"})


REJECTIONS = [
    pytest.param(
        DataValidationError,
        "Missing required coordinates in unstructured data",
        _missing_mesh_coords,
        "client",
        id="mesh_missing_coords",
    ),
    pytest.param(
        DataValidationError,
        "grid_resolution parameter is not supported for unstructured grids",
        lambda tmp: _mesh_tracker(open_unstructured(), tmp, grid_resolution=0.1),
        "client",
        id="mesh_grid_resolution",
    ),
    pytest.param(
        ConfigurationError,
        "area_filter_absolute.*must be positive",
        lambda tmp: _grid(area_filter_absolute=-100.0),
        "threads",
        id="negative_area_filter",
    ),
    pytest.param(
        ConfigurationError,
        "Cannot specify both area filtering parameters",
        lambda tmp: _grid(area_filter_quartile=0.5, area_filter_absolute=1000.0),
        "threads",
        id="both_area_filters",
    ),
    pytest.param(DataValidationError, "Data must be chunked", _not_dask, "threads", id="not_dask"),
    pytest.param(
        DataValidationError,
        "temp_dir is required for unstructured grids",
        lambda tmp: _mesh_tracker(open_unstructured(), None),
        "client",
        id="mesh_no_temp_dir",
    ),
    pytest.param(ConfigurationError, "lazy", lambda tmp: _mode(compute_mode="lazy"), "threads", id="mode_lazy"),
    pytest.param(ConfigurationError, "compute_mode", lambda tmp: _mode(compute_mode="nonsense"), "threads", id="mode_unknown"),
    pytest.param(
        ConfigurationError, "temp_dir", lambda tmp: _mode(compute_mode="streaming"), "threads", id="streaming_no_temp_dir"
    ),
    *[
        pytest.param(
            ConfigurationError, "prefilter_min_cells", lambda tmp, bad=bad: _prefilter(bad), "threads", id=f"prefilter_{bad!r}"
        )
        for bad in (0, -3, 2.5, True, "23")
    ],
    pytest.param(
        DataValidationError,
        "Invalid cell_areas dimensions",
        lambda tmp: _bad_cell_areas(False),
        "client",
        id="cell_areas_missing_lat",
    ),
    pytest.param(
        DataValidationError, "Invalid cell_areas dimensions", lambda tmp: _bad_cell_areas(True), "client", id="cell_areas_extra_dim"
    ),
    pytest.param(
        DataValidationError,
        "grid_resolution must be a positive number",
        lambda tmp: _grid_resolution(-0.5),
        "client",
        id="grid_resolution_negative",
    ),
    pytest.param(
        DataValidationError,
        "grid_resolution must be a positive number",
        lambda tmp: _grid_resolution(0.0),
        "client",
        id="grid_resolution_zero",
    ),
    pytest.param(
        ConfigurationError,
        "mask must be an xarray.DataArray",
        lambda tmp: marEx.tracker(open_gridded().extreme_events, 8),
        "client",
        id="r_fill_positional",
    ),
    pytest.param(
        ConfigurationError,
        "R_fill is required",
        lambda tmp: marEx.tracker(open_gridded().extreme_events),
        "client",
        id="r_fill_missing",
    ),
    pytest.param(
        ConfigurationError,
        "coordinate_units is required",
        lambda tmp: marEx.regional_tracker(open_gridded().extreme_events, R_fill=4),
        "client",
        id="regional_no_units",
    ),
    pytest.param(
        TrackingError, re.escape("extra dimension(s) ['depth']"), lambda tmp: _with_depth(), "threads", id="rank_extra_dimension"
    ),
    pytest.param(TrackingError, re.escape("ds.isel(depth=0)"), lambda tmp: _with_depth(), "threads", id="rank_names_the_fix"),
    pytest.param(
        DataValidationError,
        "neighbours array is required",
        lambda tmp: _mesh_tracker(open_unstructured(), tmp, neighbours=None, cell_areas=None),
        "client",
        id="mesh_no_neighbours",
    ),
]


@pytest.mark.parametrize("error, match, build, scheduler", REJECTIONS)
def test_invalid_configuration_is_rejected(error, match, build, scheduler, client, tmp_path):
    assert on_client(client)
    with dask.config.set(THREADED if scheduler == "threads" else {}), pytest.raises(error) as excinfo:
        build(tmp_path)
    message = str(excinfo.value)
    # The raised error's traceback holds the half-built tracker and its persisted coordinates:
    # drop it before asserting, or a mismatch would also leak futures past the module.
    del excinfo
    assert re.search(match, message), f"{error.__name__} message does not match {match!r}: {message}"


@pytest.mark.parametrize("area_filter", [{"area_filter_absolute": 1}, {"area_filter_quartile": 0.5}])
def test_single_timestep_record_tracks(client, area_filter):
    """A single time step tracks to one event labelled 1 on the blob's cells."""
    blob = np.zeros((1, 10, 10), dtype=bool)
    blob[0, 2:6, 2:6] = True
    data = xr.DataArray(
        blob,
        dims=["time", "lat", "lon"],
        coords={"time": [0], "lat": np.linspace(-20, 20, 10), "lon": np.linspace(-10, 10, 10)},
    ).chunk({"time": 1})
    mask = xr.ones_like(data.isel(time=0), dtype=bool)

    tracked = marEx.tracker(
        data, mask, R_fill=0, T_fill=0, regional_mode=True, coordinate_units="degrees", quiet=True, **area_filter
    ).run()

    assert tracked.attrs["N_events_final"] == 1
    ids = tracked.ID_field.values
    assert ids.shape == (1, 10, 10)
    assert (ids[blob] == 1).all()
    assert (ids[~blob] == 0).all()


def test_unstructured_single_timestep(client, tmp_path):
    """A single time step of the shipped store tracks to contiguous IDs 1..N."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ds = open_unstructured()
    # Independent count: connected components of the event cells on the neighbour graph
    # (an absolute filter of one cell removes nothing). Track the step with the most objects.
    nb = ds.neighbours.values - 1  # stored 1-based, 0 = no neighbour
    n_cells = nb.shape[1]
    cells = np.broadcast_to(np.arange(n_cells), nb.shape)
    valid = nb >= 0
    adjacency = coo_matrix((np.ones(valid.sum()), (cells[valid], nb[valid])), shape=(n_cells, n_cells)).tocsr()
    flags = ds.extreme_events.values.astype(bool)
    counts = [connected_components(adjacency[m][:, m], directed=False)[0] if m.any() else 0 for m in flags]
    t = int(np.argmax(counts))
    expected = counts[t]
    assert expected > 1, "fixture step must hold more than one object"
    single = ds.extreme_events.isel(time=slice(t, t + 1)).chunk({"time": 1})

    tracked = marEx.tracker(
        single,
        ds.mask,
        R_fill=0,
        T_fill=0,
        area_filter_absolute=1,
        temp_dir=str(tmp_path),
        quiet=True,
        neighbours=ds.neighbours,
        cell_areas=ds.cell_areas,
        **MESH,
    ).run()

    ids = tracked.ID_field.values
    present = np.unique(ids[ids > 0])
    assert tracked.attrs["N_events_final"] == expected
    assert present.tolist() == list(range(1, expected + 1))


def test_prefilter_min_cells_is_keyword_only():
    p = inspect.signature(marEx.tracker.__init__).parameters["prefilter_min_cells"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None


def _gridded_no_merge(**kwargs):
    ds = open_gridded()
    return marEx.tracker(
        ds.extreme_events, ds.mask, area_filter_quartile=0.5, R_fill=4, T_fill=0, allow_merging=False, quiet=True, **kwargs
    )


def test_checkpoint_load_bypasses_the_single_use_guard(client, tmp_path):
    """``run()`` refuses a tracker whose ``data_bin`` is gone ("already been run"), but not a checkpoint load,
    which reads the zarr store and never ``data_bin``: the documented two-cluster pattern is
    ``run_preprocess(checkpoint="save")`` -> close the cluster -> ``run(checkpoint="load")``. The guard once
    fired unconditionally. With no checkpoint store present a load must still fail, LATER, on the missing
    store and not on the guard.

    The ``round_trip`` row is the end-to-end pattern on one client (no restart): tracker A saves the
    checkpoint, tracker B has no ``data_bin`` and is asked for the load only as the ``run()`` argument, so
    ``run()`` must hand that argument on to ``run_preprocess`` (B's own ``checkpoint`` is ``None``). Its ID
    field equals a plain run's. The guard rows keep the threads pin they ran under before.
    """
    assert on_client(client)
    failures = []

    def guard_fires_without_checkpoint():
        tracker = _gridded_no_merge()
        tracker.data_bin = None  # what run_preprocess() does after hole filling
        with pytest.raises(TrackingError, match="already been run"):
            tracker.run()

    def guard_passes_on_load(where):
        if where == "argument":
            tracker = _gridded_no_merge(temp_dir=str(tmp_path / where), checkpoint="save")
        else:
            tracker = _gridded_no_merge(temp_dir=str(tmp_path / where), checkpoint="load")
        tracker.data_bin = None
        with pytest.raises(Exception) as excinfo:  # noqa: PT011 - any error past the guard will do; the store does not exist
            tracker.run(checkpoint="load") if where == "argument" else tracker.run()
        message = f"{excinfo.type.__name__}: {excinfo.value}"
        del excinfo, tracker
        assert "already been run" not in message, f"a checkpoint load ({where}) was rejected by the single-use guard: {message}"

    def round_trip():
        store = str(tmp_path / "round_trip")
        saver = _gridded_no_merge(temp_dir=store)
        saver.run_preprocess(checkpoint="save")
        del saver
        loader = _gridded_no_merge(temp_dir=store)
        loader.data_bin = None
        try:
            loaded = loader.run(checkpoint="load").ID_field.values
        except Exception as e:  # noqa: BLE001 - reported below without the traceback's persisted intermediates
            failure = f"{type(e).__name__}: {e}"
        else:
            failure = None
        del loader
        assert failure is None, f"run(checkpoint='load') did not load the checkpoint: {failure}"
        plain = _gridded_no_merge(temp_dir=str(tmp_path / "plain")).run().ID_field.values
        assert (plain > 0).any(), "the plain run found no objects, so the comparison is vacuous"
        assert np.array_equal(loaded, plain), (
            "run(checkpoint='load') after run_preprocess(checkpoint='save') differs from a plain run"
        )

    with dask.config.set(THREADED):  # the guard rows ran under the threads pin before they gained the round trip
        check(failures, "no_checkpoint", guard_fires_without_checkpoint)
        check(failures, "load_as_run_argument", lambda: guard_passes_on_load("argument"))
        check(failures, "load_from_constructor", lambda: guard_passes_on_load("constructor"))
    check(failures, "round_trip", round_trip)
    assert_no_failures(failures)


def test_tracker_holds_no_dask_backed_coordinates(tmp_path):
    """The tracker's coordinate state is materialised at construction, never left dask-backed or persisted.

    ``tracker.__init__`` once ``persist()``-ed ``lat`` / ``lon`` / ``lat_init`` / ``lon_init`` / ``cell_area``,
    binding them to whichever client was active and replacing their graphs with futures; closing that client
    (the two-cluster pattern) orphaned them and tracking died with ``lost dependencies``. Unstructured only:
    on a gridded store these are small numpy coords and the persist was a silent no-op, so only a mesh whose
    geometry is really dask-backed (405 cells here, 14.9 M on ICON R02B09) discriminates.
    """
    data = xr.open_zarr(str(Path(__file__).parent.parent / "data" / "extremes_unstructured.zarr"), chunks={"time": 2, "ncells": -1})
    assert data.cell_areas.chunks is not None and data.lat.chunks is not None, "the mesh geometry must be dask-backed"
    tracker = marEx.tracker(
        data.extreme_events,
        data.mask,
        R_fill=2,
        area_filter_quartile=0.1,
        T_fill=0,
        allow_merging=False,
        regional_mode=False,
        quiet=True,
        neighbours=data.neighbours,
        cell_areas=data.cell_areas,
        temp_dir=str(tmp_path),
        **MESH,
    )
    checked = []
    for attr in ("lat", "lon", "lat_init", "lon_init", "cell_area"):
        value = getattr(tracker, attr, None)
        if value is None:
            continue
        checked.append(attr)
        assert not is_dask_collection(getattr(value, "data", None)), f"{attr} is still dask-backed"
    assert {"lat", "lon", "cell_area"} <= set(checked), checked


def test_grid_resolution_builds_cell_areas(client, marex_records):
    """grid_resolution gives the spherical cell areas elementwise, and overrides a supplied cell_areas with a warning."""
    assert on_client(client)
    ds = open_gridded()
    resolution = abs(float(ds.lat[1] - ds.lat[0]))
    lat_r = np.radians(np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")[0])
    d = np.radians(resolution)
    manual = (6378.0**2 * np.abs(np.sin(lat_r + d / 2) - np.sin(lat_r - d / 2)) * d).astype(np.float32)

    tracker = marEx.tracker(
        ds.extreme_events,
        polar_mask(ds),
        area_filter_quartile=0.5,
        R_fill=4,
        T_fill=2,
        allow_merging=True,
        grid_resolution=resolution,
        quiet=True,
    )
    # The areas depend on latitude only, and the tracker may keep them one-dimensional.
    template = xr.DataArray(manual, coords={"lat": ds.lat, "lon": ds.lon}, dims=["lat", "lon"])
    got = np.asarray(tracker.cell_area.broadcast_like(template).transpose("lat", "lon").values)
    np.testing.assert_allclose(got, manual, rtol=1e-6, err_msg="grid_resolution areas differ from the spherical formula")
    assert tracker.mean_cell_area == pytest.approx(float(manual.mean()), rel=1e-6)

    ones = xr.DataArray(
        np.ones((ds.sizes["lat"], ds.sizes["lon"]), np.float32), coords={"lat": ds.lat, "lon": ds.lon}, dims=["lat", "lon"]
    )
    records = marex_records(logging.WARNING)
    tracker = marEx.tracker(ds.extreme_events, polar_mask(ds), R_fill=4, cell_areas=ones, grid_resolution=1.0, quiet=True)
    messages = [r.getMessage() for r in records]
    assert any("grid_resolution parameter overrides provided cell_areas" in m for m in messages), messages
    assert tracker.mean_cell_area > 1000, "the supplied unit areas were used instead of the grid_resolution areas"


def test_grid_resolution_areas_reach_the_event_properties(client):
    """A merging run on grid_resolution areas reports each object's area as the sum of its cells' spherical areas.

    grid_resolution keeps the areas one-dimensional (latitude only), so the property kernel must broadcast them
    across longitude; a run on unit or two-dimensional areas never takes that path.
    """
    assert on_client(client)
    resolution = 4.0
    lat = np.arange(-18.0, 22.0, resolution)
    lon = np.arange(-18.0, 22.0, resolution)
    field = np.zeros((4, lat.size, lon.size), dtype=bool)
    for t in range(4):
        field[t, 1:4, 1 + t : 4 + t] = True  # moves east, south of the equator
        field[t, 6:9, 6:8] = True  # stays put, north of the equator
    data = xr.DataArray(field, dims=["time", "lat", "lon"], coords={"time": np.arange(4), "lat": lat, "lon": lon})
    data = data.chunk({"time": 1})
    lat_r = np.radians(np.meshgrid(lat, lon, indexing="ij")[0])
    d = np.radians(resolution)
    manual = (6378.0**2 * np.abs(np.sin(lat_r + d / 2) - np.sin(lat_r - d / 2)) * d).astype(np.float32)

    tracker = marEx.tracker(
        data,
        xr.ones_like(data.isel(time=0), dtype=bool),
        R_fill=0,
        T_fill=0,
        area_filter_absolute=1,
        allow_merging=True,
        regional_mode=True,
        coordinate_units="degrees",
        grid_resolution=resolution,
        quiet=True,
    )
    assert tracker.cell_area.ndim == 1, f"grid_resolution areas are {tracker.cell_area.dims}, expected latitude only"
    tracked = released(lambda: tracker.run().compute())

    ids = tracked.ID_field.values
    checked = 0
    for t in range(ids.shape[0]):
        for object_id in np.unique(ids[t][ids[t] > 0]):
            got = float(tracked.area.isel(time=t).sel(ID=object_id))
            want = float(manual[ids[t] == object_id].sum())
            assert got == pytest.approx(want, rel=1e-5), f"time {t} ID {object_id}: area {got}, expected {want}"
            checked += 1
    assert checked == 8, f"expected 2 objects at each of 4 steps, checked {checked}"


def test_prune_removes_only_stale_temp_stores(tmp_path):
    """A temp store older than the age limit goes; a fresh one, a checkpoint and an unrelated directory stay."""
    from marEx.track.tracker import _TEMP_STORE_MAX_AGE_S, _prune_stale_temp_stores

    old = time.time() - _TEMP_STORE_MAX_AGE_S - 3600
    names = {
        "marEx_temp_field_1234.zarr": old,
        "marEx_temp_field.zarr": old,  # the bare legacy name
        "marEx_temp_refresh_99.zarr": old,
        "marEx_temp_field_5678.zarr": None,
        "marEx_checkpoint_run.zarr": old,
        "other_output.zarr": old,
    }
    for name, mtime in names.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "0.0").write_bytes(b"x")
        if mtime is not None:
            os.utime(tmp_path / name, (mtime, mtime))

    _prune_stale_temp_stores(str(tmp_path))

    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["marEx_checkpoint_run.zarr", "marEx_temp_field_5678.zarr", "other_output.zarr"], left
    _prune_stale_temp_stores(str(tmp_path / "missing"))  # a temp_dir that does not exist is not an error


def _regional_field(ncells=50):
    return xr.DataArray(
        np.zeros((2, ncells), dtype=bool),
        dims=("time", "ncells"),
        coords={"time": np.arange(2), "lon": ("ncells", np.linspace(10.0, 60.0, ncells)), "lat": ("ncells", np.zeros(ncells))},
    )


def test_undetectable_coordinate_units_suggest_a_working_route():
    """A longitude range that is neither a full turn in degrees nor in radians.

    regional_mode raises NotImplementedError on an unstructured grid, so it is suggested only on a
    structured one; coordinate_units alone works on both.
    """
    failures = []

    def unstructured():
        with pytest.raises(CoordinateError) as excinfo:
            unify_coordinates(_regional_field(), False, None, "lon", "lat", unstructured_grid=True)
        suggestions = " ".join(excinfo.value.suggestions)
        assert "regional_mode" not in suggestions
        assert "coordinate_units='degrees'" in suggestions

    def structured():
        with pytest.raises(CoordinateError) as excinfo:
            unify_coordinates(_regional_field(), False, None, "lon", "lat")
        assert any("regional_mode=True" in s for s in excinfo.value.suggestions)

    check(failures, "unstructured", unstructured)
    check(failures, "structured", structured)
    assert_no_failures(failures)


def test_runs_that_leave_nothing_raise_clear_errors(client, tmp_path):
    """An area filter or pre-filter that removes every object, or zero cell areas, names the problem."""
    assert on_client(client)
    failures = []
    gridded = open_gridded().isel(time=slice(0, 10))
    mesh = open_unstructured()

    def mesh_tracker(ds, cell_areas, **kwargs):
        return marEx.tracker(
            ds.extreme_events,
            ds.mask,
            cell_areas=cell_areas,
            neighbours=ds.neighbours,
            R_fill=1,
            T_fill=0,
            allow_merging=False,
            quiet=True,
            **MESH,
            **kwargs,
        )

    def gridded_filter():
        tracker = marEx.tracker(gridded.extreme_events, gridded.mask, R_fill=0, T_fill=0, area_filter_absolute=10**9, quiet=True)
        with pytest.raises(TrackingError, match="area_filter_absolute"):
            tracker.run()

    def mesh_prefilter():
        short = mesh.isel(time=slice(0, 10))
        tracker = mesh_tracker(short, short.cell_areas, area_filter_absolute=10**9, temp_dir=str(tmp_path / "p"))
        with pytest.raises(TrackingError, match="area_filter_absolute=1000000000"):
            tracker.run()

    def zero_areas():
        tracker = mesh_tracker(mesh, mesh.cell_areas * 0, area_filter_absolute=6, temp_dir=str(tmp_path / "z"))
        with pytest.raises(TrackingError, match="zero total area"):
            tracker.run()

    for name, row in (("gridded_filter", gridded_filter), ("mesh_prefilter", mesh_prefilter), ("zero_areas", zero_areas)):
        check(failures, name, row)
    assert_no_failures(failures)


def test_validate_unstructured_chunking():
    """Multi-chunk geometry comes back single-chunked along nv and space with equal values; single-chunk passes silently."""
    n_cells, n_nv = 40, 3
    neighbours = xr.DataArray(np.arange(n_nv * n_cells, dtype=np.int32).reshape(n_nv, n_cells), dims=("nv", "ncells")).chunk(
        {"nv": 1, "ncells": 10}
    )
    cell_areas = xr.DataArray(np.linspace(1, 2, n_cells, dtype=np.float32), dims=("ncells",)).chunk({"ncells": 10})
    assert len(neighbours.chunksizes["ncells"]) > 1 and len(cell_areas.chunksizes["ncells"]) > 1  # precondition

    with pytest.warns(UserWarning, match="Rechunking to single chunk"):
        neighbours_out, cell_areas_out = validate_unstructured_chunking(neighbours, cell_areas, "ncells")
    assert len(neighbours_out.chunksizes["ncells"]) == 1
    assert len(neighbours_out.chunksizes["nv"]) == 1
    assert len(cell_areas_out.chunksizes["ncells"]) == 1
    np.testing.assert_array_equal(neighbours_out.values, neighbours.values)
    np.testing.assert_array_equal(cell_areas_out.values, cell_areas.values)

    single_nb = neighbours.chunk({"nv": -1, "ncells": -1})
    single_ca = cell_areas.chunk({"ncells": -1})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        neighbours_out, cell_areas_out = validate_unstructured_chunking(single_nb, single_ca, "ncells")
    assert len(neighbours_out.chunksizes["ncells"]) == 1
    assert len(cell_areas_out.chunksizes["ncells"]) == 1
