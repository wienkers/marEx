"""Event clustering of a volumetric ID field and the final per-event diagnostics."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.core.compute_mode import Materialiser, create_staging_dir
from marEx.track.merge_split import cluster_rename_objects_and_props
from marEx.track.volumetric.config import VolumetricDims
from marEx.track.volumetric.diagnostics import event_diagnostics
from marEx.track.volumetric.events import FLAT_DIM, cluster_events
from marEx.track.volumetric.geometry import build_vertical_geometry
from marEx.track.volumetric.ids import IDAllocator
from marEx.track.volumetric.merge_loop import MergeLedger, final_links, split_and_merge_volumetric
from marEx.track.volumetric.partition import ScaledCoords
from marEx.track.volumetric.tracker4d import identify_volumes

from ..conftest import on_client as _on_client
from .conftest import VDIMS, build_case, store_from_field
from .synthetic import Ball, paint, scenarios

PROBE_DIM = "__probe_voxel__"


# ---------------------------------------------------------------------------
# Step 1: the unchanged 2-D clusterer on a flattened field
# ---------------------------------------------------------------------------


def _probe_field():
    """(time=3, voxel=4): 1 at t0 links to 2 at t1; 3 at t2 is a separate event."""
    values = np.array([[1, 1, 0, 0], [0, 2, 2, 0], [0, 0, 0, 3]], dtype=np.int32)
    times = pd.date_range("2000-01-01", periods=3, freq="D")
    return xr.DataArray(values, dims=("time", PROBE_DIM), coords={"time": times}).chunk({"time": 2, PROBE_DIM: -1})


def test_cluster_rename_accepts_empty_props():
    """The unchanged 2-D clusterer runs on a flattened field with an ID-only property set and an empty ledger."""
    field = _probe_field()
    ledger = MergeLedger(time_dtype=field["time"].dtype).to_dataset()
    ids = np.array([1, 2, 3], dtype=np.int32)
    out = cluster_rename_objects_and_props(
        field,
        xr.Dataset(coords={"ID": ids}),
        np.array([[1, 2]], dtype=np.int64),
        ledger,
        True,
        "time",
        "time",
        2,
        None,
        PROBE_DIM,
        None,
        None,
        None,
        False,
        materialiser=None,
    )
    times = field["time"].values
    # Two events: {1, 2} and {3}.
    np.testing.assert_array_equal(out["ID"].values, [1, 2])
    np.testing.assert_array_equal(out["ID_field"].values, [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 2]])
    np.testing.assert_array_equal(out["global_ID"].transpose("time", "ID").values, [[1, 0], [2, 0], [0, 3]])
    np.testing.assert_array_equal(out["presence"].transpose("time", "ID").values, [[1, 0], [1, 0], [0, 1]])
    np.testing.assert_array_equal(out["time_start"].values, times[[0, 2]])
    np.testing.assert_array_equal(out["time_end"].values, times[[1, 2]])
    np.testing.assert_array_equal(out["merge_ledger"].values, np.full((3, 2, 1), -1))
    assert set(out.data_vars) >= {"global_ID", "presence", "time_start", "time_end", "merge_ledger"}


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _track_4d(id_field, geom, coords, threshold=0.5, materialiser=None):
    """Merge loop, final links and clustering of an identified ``(time, depth, lat, lon)`` field."""
    store = store_from_field(id_field.values, geom.cell_volume)
    allocator = IDAllocator(int(id_field.values.max()) + 1)
    merged, store, ledger = split_and_merge_volumetric(
        id_field, store, allocator, geom, coords, threshold, VDIMS, materialiser=None, id_field_path=None
    )
    links = final_links(merged, store, geom.cell_volume, threshold, VDIMS.time)
    events = cluster_events(merged, links, ledger.to_dataset(), VDIMS, VDIMS.time, id_field.chunks[0][0], materialiser=materialiser)
    return merged, ledger.to_dataset(), events


def _unit_geometry(ny, nx, lat, lon):
    """One 1 m level over 1 m^2 cells: every voxel has volume 1."""
    wet = xr.DataArray(np.ones((1, ny, nx), bool), dims=("depth", "lat", "lon"))
    bounds = xr.DataArray([[0.0, 1.0]], dims=("depth", "bnds"))
    areas = xr.DataArray(np.ones((ny, nx)), dims=("lat", "lon"), attrs={"units": "m2"})
    return build_vertical_geometry(
        np.array([0.5]),
        {"units": "m", "positive": "down"},
        wet,
        zdim="depth",
        ydim="lat",
        xdim="lon",
        lat_deg=lat,
        lon_deg=lon,
        layer_bounds=bounds,
        layer_thickness=None,
        cell_volumes=None,
        cell_areas=areas,
        grid_resolution=None,
    )


def test_nz1_events_match_2d_no_merge_input(client):
    """With one unit level the 4-D event outputs equal the 2-D tracker's on an input with births, deaths and a split."""
    assert _on_client(client)
    nt, ny, nx = 7, 12, 16
    balls = [
        Ball(0, 6, z=0.0, y=2.5, x=3.0, rz=0.4, ry=1.5, rx=2.0, vx=0.5),  # persists and drifts
        Ball(2, 4, z=0.0, y=8.5, x=12.5, rz=0.4, ry=1.5, rx=1.5),  # born at 2, dies after 4
        Ball(0, 2, z=0.0, y=8.5, x=4.0, rz=0.4, ry=1.5, rx=3.0),  # splits at 3 ...
        Ball(3, 5, z=0.0, y=8.5, x=2.0, rz=0.4, ry=1.5, rx=1.0),  # ... into this
        Ball(3, 5, z=0.0, y=8.5, x=6.0, rz=0.4, ry=1.5, rx=1.0),  # ... and this
    ]
    binary = paint(nt, 1, ny, nx, balls)[:, 0]
    times = pd.date_range("2000-01-01", periods=nt, freq="D")
    lat = np.linspace(-10.0, 10.0, ny)
    lon = np.linspace(0.0, 360.0, nx, endpoint=False)  # global, so the 2-D tracker infers degrees
    data2 = xr.DataArray(binary, dims=("time", "lat", "lon"), coords={"time": times, "lat": lat, "lon": lon}).chunk({"time": 3})
    mask2 = xr.DataArray(np.ones((ny, nx), bool), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    t2 = marEx.tracker(
        data2,
        mask2,
        R_fill=0,
        T_fill=0,
        area_filter_absolute=1,
        allow_merging=True,
        overlap_threshold=0.5,
        nn_partitioning=True,
        quiet=True,
    )
    preprocessed, _ = t2.run_preprocess()
    events2, merges2, _ = t2.track_objects(preprocessed)
    assert merges2.sizes.get("merge_ID", 0) == 0  # the input never merges

    data4 = preprocessed.compute().transpose("time", "lat", "lon").expand_dims(depth=[0.5], axis=1)
    data4 = data4.assign_coords(depth=("depth", [0.5], {"units": "m", "positive": "down"})).chunk({"time": 3})
    geom = _unit_geometry(ny, nx, lat, lon)
    ids4 = identify_volumes(data4, geom, 0.0, True, VDIMS, IDAllocator(), materialiser=None)
    coords = ScaledCoords(lat, lon, geom.midpoint, *geom.default_scales)
    _, ledger4, events4 = _track_4d(ids4, geom, coords)
    assert ledger4.sizes["merge_ID"] == 0

    id2 = events2["ID_field"].transpose("time", "lat", "lon")
    id4 = events4["ID_field"].isel(depth=0).transpose("time", "lat", "lon")
    assert id4.dtype == id2.dtype == np.int32
    assert int(id2.max()) >= 3
    np.testing.assert_array_equal(id4.values, id2.values)
    for name in ("global_ID", "presence", "time_start", "time_end", "merge_ledger"):
        a, b = events4[name], events2[name]
        assert a.dims == b.dims, name
        assert a.dtype == b.dtype, name
        np.testing.assert_array_equal(a.values, b.values, err_msg=name)
    np.testing.assert_array_equal(events4["ID"].values, events2["ID"].values)


def test_merge_ledger_records_parent_events():
    """At every merge time the merged event's ledger row holds the parents' event IDs; elsewhere it is -1."""
    binary, meta = scenarios()["one_merge"]
    case = build_case(binary, meta)
    merged, ledger, events = _track_4d(case.id_field, case.geom, case.coords)
    assert ledger.sizes["merge_ID"] >= 1

    # Old object ID -> event ID, read voxel by voxel from the two fields.
    old = merged.values.ravel()
    new = events["ID_field"].transpose(*VDIMS.internal_order()).values.ravel()
    occupied = old > 0
    old_to_new = dict(zip(old[occupied].tolist(), new[occupied].tolist()))
    assert len(old_to_new) == len(set(old[occupied].tolist()))

    event_ids = events["ID"].values
    times = events[VDIMS.time].values
    expected = np.full((times.size, event_ids.size, ledger.sizes["parent_idx"]), -1, dtype=np.int32)
    for m in range(ledger.sizes["merge_ID"]):
        t = int(np.flatnonzero(times == ledger["merge_time"].values[m])[0])
        for p in ledger["parent_IDs"].values[m]:
            if p > 0:
                e = old_to_new[int(p)]
                expected[t, int(np.flatnonzero(event_ids == e)[0]), :] = e
    assert (expected > 0).any()
    got = events["merge_ledger"].transpose(VDIMS.time, "ID", "sibling_ID").values
    np.testing.assert_array_equal(got, expected)


def _small_field(dims, nt=4, shape=(2, 3, 4), time_chunk=2):
    """A short (time, z, y, x) ID field with a linked pair and an unlinked object."""
    values = np.zeros((nt,) + shape, dtype=np.int32)
    values[0, 0, 0, 0:2] = 1
    values[1, 0:2, 0, 1:3] = 2
    values[2, 1, 2, 3] = 3
    values[3, 1, 2, 2:4] = 4
    coords = {
        dims.time: pd.date_range("2000-01-01", periods=nt, freq="D"),
        dims.z: [5.0, 15.0],
        dims.y: [0.0, 1.0, 2.0],
        dims.x: [10.0, 11.0, 12.0, 13.0],
    }
    da = xr.DataArray(values, dims=dims.internal_order(), coords=coords)
    return da.chunk({dims.time: time_chunk, dims.z: -1, dims.y: -1, dims.x: -1})


_LINKS = np.array([[1, 2], [3, 4]], dtype=np.int64)


def test_flat_dim_name_cannot_collide():
    """A user depth dimension literally named ``z`` (the name the 2-D clusterer stacks into) clusters the same."""
    ref_dims = VolumetricDims(time="time", z="depth", y="lat", x="lon")
    z_dims = VolumetricDims(time="time", z="z", y="y", x="x")
    empty = MergeLedger().to_dataset()
    ref = cluster_events(_small_field(ref_dims), _LINKS, empty, ref_dims, "time", 2, materialiser=None)
    out = cluster_events(_small_field(z_dims), _LINKS, empty, z_dims, "time", 2, materialiser=None)
    assert out["ID_field"].dims == ("time", "z", "y", "x")
    assert FLAT_DIM not in out.dims
    np.testing.assert_array_equal(out["ID_field"].values, ref["ID_field"].values)
    np.testing.assert_array_equal(out["ID_field"].values[:, 0, 0, :2], [[1, 1], [0, 1], [0, 0], [0, 0]])
    np.testing.assert_array_equal(out["ID_field"].values[3, 1, 2, 2:4], [2, 2])
    for name in ("global_ID", "presence", "time_start", "time_end", "merge_ledger"):
        np.testing.assert_array_equal(out[name].values, ref[name].values, err_msg=name)
    # Spatial coordinates survive the round trip.
    np.testing.assert_array_equal(out["z"].values, [5.0, 15.0])
    np.testing.assert_array_equal(out["x"].values, [10.0, 11.0, 12.0, 13.0])


@pytest.mark.parametrize("mode", ["persist", "streaming"])
def test_reshape_back_keeps_chunk_structure(tmp_path, mode):
    """The ID field comes back (time, z, y, x) with the input's time chunks and whole z, y, x."""
    dims = VolumetricDims(time="time", z="depth", y="lat", x="lon")
    field = _small_field(dims, nt=5, time_chunk=2)
    materialiser = Materialiser(mode, create_staging_dir(tmp_path) if mode == "streaming" else None)
    out = cluster_events(field, _LINKS, MergeLedger().to_dataset(), dims, "time", 2, materialiser=materialiser)
    assert out["ID_field"].dims == dims.internal_order()
    assert out["ID_field"].dtype == np.int32
    assert out["ID_field"].chunks == ((2, 2, 1), (2,), (3,), (4,))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _geometry(thickness, areas, lat, lon):
    nz = len(thickness)
    ny, nx = areas.shape
    edges = np.concatenate([[0.0], np.cumsum(np.asarray(thickness, dtype=np.float64))])
    centres = 0.5 * (edges[:-1] + edges[1:])
    wet = xr.DataArray(np.ones((nz, ny, nx), bool), dims=("depth", "lat", "lon"))
    bounds = xr.DataArray(np.stack([edges[:-1], edges[1:]], axis=1), dims=("depth", "bnds"))
    cell_areas = xr.DataArray(areas, dims=("lat", "lon"), attrs={"units": "m2"})
    return build_vertical_geometry(
        centres,
        {"units": "m", "positive": "down"},
        wet,
        zdim="depth",
        ydim="lat",
        xdim="lon",
        lat_deg=lat,
        lon_deg=lon,
        layer_bounds=bounds,
        layer_thickness=None,
        cell_volumes=None,
        cell_areas=cell_areas,
        grid_resolution=None,
    )


def _events(values, time_chunk=1):
    """Wrap a (time, z, y, x) event field and its (time, ID) presence."""
    nt = values.shape[0]
    times = pd.date_range("2000-01-01", periods=nt, freq="D")
    ids = np.arange(1, int(values.max()) + 1, dtype=np.int32)
    field = xr.DataArray(values.astype(np.int32), dims=VDIMS.internal_order(), coords={"time": times})
    field = field.chunk({"time": time_chunk, "depth": -1, "lat": -1, "lon": -1})
    present = np.stack([np.isin(ids, values[t]) for t in range(nt)])
    presence = xr.DataArray(present, dims=("time", "ID"), coords={"time": times, "ID": ids}).chunk({"time": time_chunk})
    return field, presence


def test_diagnostics_analytic():
    """A 2x2x2 box on known layers and areas: exact volume, footprint, depths and thickness; absent IDs are NaN."""
    thickness = [2.0, 4.0, 8.0]
    area = 1024.0
    lat = np.array([10.0, 10.5, 11.0])
    lon = np.array([20.0, 20.5, 21.0])
    geom = _geometry(thickness, np.full((3, 3), area), lat, lon)
    values = np.zeros((2, 3, 3, 3), dtype=np.int32)
    values[0, 0:2, 0:2, 0:2] = 1  # levels 0-1 (0 to 6 m) under four columns
    values[1, 2, 1, 1] = 2  # one voxel on the deepest level
    field, presence = _events(values)
    out = event_diagnostics(field, presence, geom, lat, lon, VDIMS, materialiser=None).compute()

    for name in ("volume", "footprint_area", "depth_centroid", "depth_top", "depth_bottom", "mean_thickness"):
        assert out[name].dims == ("time", "ID") and out[name].dtype == np.float64, name
    assert out["centroid"].dims == ("component", "time", "ID")
    np.testing.assert_array_equal(out["component"].values, [0, 1])

    nan = np.nan
    np.testing.assert_array_equal(out["volume"].values, [[4 * 6 * area, nan], [nan, 8 * area]])
    np.testing.assert_array_equal(out["footprint_area"].values, [[4 * area, nan], [nan, area]])
    np.testing.assert_array_equal(out["depth_top"].values, [[0.0, nan], [nan, 6.0]])
    np.testing.assert_array_equal(out["depth_bottom"].values, [[6.0, nan], [nan, 14.0]])
    # Volume-weighted midpoint: (2 * 1 + 4 * 4) / 6 = 3 m.
    np.testing.assert_array_equal(out["depth_centroid"].values, [[3.0, nan], [nan, 10.0]])
    np.testing.assert_array_equal(out["mean_thickness"].values, [[6.0, nan], [nan, 8.0]])
    # A single voxel's centroid is its cell's unit vector converted back to degrees.
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    a = _oracle(values[1], geom, lat2, lon2, presence["ID"].values)[2]
    np.testing.assert_array_equal(out["centroid"].values[:, 1, 1], _centroid_degrees(a, lon.min()))
    assert np.isnan(out["centroid"].values[:, 0, 1]).all()


def _oracle(values_t, geom, lat2, lon2, ids):
    """Python-loop sums over the voxels of one timestep in C (z, y, x) order."""
    nz, ny, nx = values_t.shape
    ex = np.cos(np.radians(lat2)) * np.cos(np.radians(lon2))
    ey = np.cos(np.radians(lat2)) * np.sin(np.radians(lon2))
    ez = np.sin(np.radians(lat2))
    acc = {int(i): dict(v=0.0, vm=0.0, sx=0.0, sy=0.0, sz=0.0, fa=0.0, top=np.inf, bot=-np.inf) for i in ids}
    columns = {int(i): set() for i in ids}
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                e = int(values_t[k, j, i])
                if e == 0:
                    continue
                v = float(geom.cell_volume[k, j, i])
                a = acc[e]
                a["v"] += v
                a["vm"] += v * float(geom.midpoint[k, j, i])
                a["sx"] += v * float(ex[j, i])
                a["sy"] += v * float(ey[j, i])
                a["sz"] += v * float(ez[j, i])
                a["top"] = min(a["top"], float(geom.top[k, j, i]))
                a["bot"] = max(a["bot"], float(geom.bottom[k, j, i]))
                columns[e].add(j * nx + i)
    for e, cols in columns.items():
        for c in sorted(cols):  # columns in C (y, x) order
            acc[e]["fa"] += float(geom.cell_area_m2.ravel()[c])
    return acc


def _centroid_degrees(a, lon_origin):
    """(lat, lon) in degrees of an oracle's summed unit vector, longitude into [lon_origin, lon_origin + 360)."""
    sx, sy, sz = (np.array([a[c]]) for c in ("sx", "sy", "sz"))
    lat_c = np.degrees(np.arctan2(sz, np.hypot(sx, sy)))[0]
    lon_c = np.degrees(np.arctan2(sy, sx))[0]
    offset = np.mod(lon_c - lon_origin, 360.0)
    return lat_c, lon_origin + (offset - 360.0 if offset > 360.0 - 1e-9 else offset)


def test_diagnostics_summation_order_pinned():
    """Non-dyadic layers, areas and shapes: every float diagnostic equals a C-order Python-loop sum, exactly."""
    rng = np.random.default_rng(20261101)
    nz, ny, nx = 3, 5, 6
    thickness = [0.1, 0.7, 13.0]
    lat = np.linspace(-33.3, -29.1, ny)
    lon = np.linspace(170.7, 183.1, nx)  # crosses 180 in a 0..360 convention
    areas = 1.0e6 / 3.0 * (1.0 + 0.137 * np.arange(ny)[:, None] + 0.0113 * np.arange(nx)[None, :])
    geom = _geometry(thickness, areas, lat, lon)
    values = np.zeros((2, nz, ny, nx), dtype=np.int32)
    values[0] = rng.integers(0, 3, size=(nz, ny, nx))  # two interleaved events
    values[1] = np.where(rng.random((nz, ny, nx)) < 0.6, 2, 0)
    field, presence = _events(values)
    out = event_diagnostics(field, presence, geom, lat, lon, VDIMS, materialiser=None).compute()

    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    ids = presence["ID"].values
    discriminating = False
    for t in range(values.shape[0]):
        acc = _oracle(values[t], geom, lat2, lon2, ids)
        for col, e in enumerate(ids):
            a = acc[int(e)]
            if a["v"] == 0.0:
                assert np.isnan(out["volume"].values[t, col]) and np.isnan(out["centroid"].values[:, t, col]).all()
                continue
            assert out["volume"].values[t, col] == a["v"]
            assert out["footprint_area"].values[t, col] == a["fa"]
            assert out["depth_centroid"].values[t, col] == a["vm"] / a["v"]
            assert out["depth_top"].values[t, col] == a["top"]
            assert out["depth_bottom"].values[t, col] == a["bot"]
            assert out["mean_thickness"].values[t, col] == a["v"] / a["fa"]
            lat_c, lon_c = _centroid_degrees(a, lon.min())
            assert out["centroid"].values[0, t, col] == lat_c
            assert out["centroid"].values[1, t, col] == lon_c
            # The inputs are chosen so that the order of the volume sum matters.
            occupied = values[t] == e
            backwards = 0.0
            for v in geom.cell_volume[occupied][::-1]:
                backwards += float(v)
            discriminating |= backwards != a["v"]
    assert discriminating, "inputs do not distinguish forward from reversed summation"
    # Longitudes stay in the coordinate's 0..360 convention (no wrap to negative values).
    lon_out = out["centroid"].values[1]
    assert np.nanmin(lon_out) >= lon.min() and np.nanmax(lon_out) <= lon.max()


def test_centroid_seam_safe():
    """An object straddling lon +-180 has its centroid on the seam, not on the far side of the globe."""
    lat = np.array([-0.5, 0.5])
    lon = np.arange(-180.0, 180.0, 30.0)  # 12 cells: -180, -150, ..., 150
    geom = _geometry([10.0], np.ones((2, 12)), lat, lon)
    values = np.zeros((1, 1, 2, 12), dtype=np.int32)
    values[0, 0, :, [11, 0, 1]] = 1  # 150, -180 and -150
    field, presence = _events(values)
    out = event_diagnostics(field, presence, geom, lat, lon, VDIMS, materialiser=None).compute()
    lon_c = float(out["centroid"].values[1, 0, 0])
    assert abs((lon_c % 360.0) - 180.0) < 0.01, lon_c


@pytest.mark.parametrize("mode", ["persist", "streaming"])
def test_diagnostics_keep_time_chunks_and_static_geometry(tmp_path, mode):
    """Outputs keep the field's time chunks and a whole ID axis."""
    lat = np.array([10.0, 10.5, 11.0])
    lon = np.array([20.0, 20.5, 21.0])
    geom = _geometry([2.0, 4.0, 8.0], np.full((3, 3), 5.0), lat, lon)
    values = np.zeros((5, 3, 3, 3), dtype=np.int32)
    values[:, 0, 0, 0] = 1
    values[2:, 1, 2, 2] = 2
    field, presence = _events(values, time_chunk=2)
    materialiser = Materialiser(mode, create_staging_dir(tmp_path) if mode == "streaming" else None)
    out = event_diagnostics(field, presence, geom, lat, lon, VDIMS, materialiser=materialiser)
    for name in ("volume", "footprint_area", "depth_centroid", "depth_top", "depth_bottom", "mean_thickness"):
        assert out[name].chunks == ((2, 2, 1), (2,)), name
    assert out["centroid"].chunks == ((2,), (2, 2, 1), (2,))
    np.testing.assert_array_equal(out["volume"].values[:, 0], np.full(5, 2.0 * 5.0))


def test_centroid_lon_stays_on_the_edge_columns():
    """A one-column event in the westmost or eastmost column keeps that column's longitude, never a 360-degree jump."""
    ny, nx = 20, 4
    lat = np.linspace(-80.0, 80.0, ny)
    for lon0 in np.linspace(-180.0, 179.0, 60):
        for dlon in (0.083, 0.25, 1.7):
            lon = lon0 + dlon * np.arange(nx)
            geom = _geometry([10.0], np.ones((ny, nx)), lat, lon)
            values = np.zeros((1, 1, ny, nx), dtype=np.int32)
            values[0, 0, :, 0] = np.arange(1, ny + 1)  # one event per row, in the westmost column
            values[0, 0, :, nx - 1] = np.arange(ny + 1, 2 * ny + 1)  # and in the eastmost column
            field, presence = _events(values)
            out = event_diagnostics(field, presence, geom, lat, lon, VDIMS, materialiser=None)
            lon_c = out["centroid"].values[1, 0]
            west, east = lon_c[:ny], lon_c[ny:]
            assert np.abs(west - lon[0]).max() < 1e-9, (lon0, dlon, west.max())
            assert np.abs(east - lon[-1]).max() < 1e-9, (lon0, dlon, east.max())


def _cluster_probe(ledger):
    field = _probe_field()
    out_merges = ledger.to_dataset()
    cluster_rename_objects_and_props(
        field,
        xr.Dataset(coords={"ID": np.array([1, 2, 3], dtype=np.int32)}),
        np.array([[1, 2]], dtype=np.int64),
        out_merges,
        True,
        "time",
        "time",
        2,
        None,
        PROBE_DIM,
        None,
        None,
        None,
        False,
    )
    return out_merges


def test_merge_event_ids_follow_renames_and_leave_dropped_ids_unresolved():
    """Objects 1 and 2 are one event, 3 another. 7 -> 8 -> 2 is a chain of consolidation renames; 9 never reached the field."""
    ledger = MergeLedger()
    ledger.times.append(np.datetime64("2000-01-02", "ns"))
    ledger.parent_ids.append(np.array([1, 3], dtype=np.int32))
    ledger.child_ids.append(np.array([7, 3, 9], dtype=np.int32))
    ledger.overlap_volumes.append(np.array([1.0, 1.0]))
    ledger.renamed.extend([(7, 8), (8, 2)])
    merges = _cluster_probe(ledger)
    first, second = merges["parent_event_IDs"].values[0]
    assert first > 0 and second > 0 and first != second
    np.testing.assert_array_equal(merges["child_event_IDs"].values, [[first, second, -1]])
    np.testing.assert_array_equal(merges["n_parent_events"].values, [2])
    assert "_renamed_from" not in merges.variables and "_renamed_to" not in merges.variables and "rename_ID" not in merges.sizes


def test_merge_event_ids_keep_padding_and_count_events_not_objects():
    ledger = MergeLedger()
    day = np.datetime64("2000-01-02", "ns")
    for parents in ([1, 2], [1]):  # two objects of one event; a shorter row is padded with -1
        ledger.times.append(day)
        ledger.parent_ids.append(np.array(parents, dtype=np.int32))
        ledger.child_ids.append(np.array([2], dtype=np.int32))
        ledger.overlap_volumes.append(np.ones(len(parents)))
    merges = _cluster_probe(ledger)
    event = merges["parent_event_IDs"].values[0, 0]
    np.testing.assert_array_equal(merges["parent_event_IDs"].values, [[event, event], [event, -1]])
    np.testing.assert_array_equal(merges["n_parent_events"].values, [1, 1])


def test_split_records_get_their_event():
    ledger = MergeLedger()
    ledger.split_times.append(np.datetime64("2000-01-02", "ns"))
    ledger.split_parents.append(1)
    ledger.split_children.append(np.array([2, 7], dtype=np.int32))
    ledger.renamed.append((7, 2))
    merges = _cluster_probe(ledger)
    event = merges["split_event_IDs"].values[0]
    assert event > 0 and merges["split_event_IDs"].dtype == np.int32
    np.testing.assert_array_equal(merges["split_child_IDs"].values, [[2, 7]])
