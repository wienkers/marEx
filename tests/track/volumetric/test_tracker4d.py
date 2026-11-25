"""The volumetric tracker end to end: preprocess, identify, merge loop, events and diagnostics."""

import os
import re
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.exceptions import DataValidationError
from marEx.track.volumetric import merge_loop as _merge_loop
from marEx.track.volumetric import tracker4d as _tracker4d
from marEx.track.volumetric.tracker4d import VolumetricTracker

from .synthetic import Ball, make_inputs, paint, scenarios

DIMS = {"time": "time", "z": "depth", "y": "lat", "x": "lon"}
CANONICAL = ("time", "depth", "lat", "lon")
EXPECTED_ATTRS = (
    "spatial_connectivity",
    "max_vertical_gap_m",
    "vertical_geometry",
    "volume_units",
    "partition_metric",
    "partition_scales_m",
    "overlap_threshold",
    "volume_threshold (m^3)",
    "R_fill",
    "T_fill",
    "N_objects_prefiltered",
    "N_objects_filtered",
    "N_events_final",
    "total_merges",
    "multi_parent_merges",
    "horizontal_y_dim",
    "horizontal_x_dim",
    "horizontal_y_coord",
    "horizontal_x_coord",
    "horizontal_coordinate_units",
)
DIAGNOSTICS = ("volume", "footprint_area", "centroid", "depth_centroid", "depth_top", "depth_bottom", "mean_thickness")


def _kwargs(**overrides):
    """Tracker keyword arguments that leave the synthetic events untouched by preprocessing."""
    kwargs = dict(
        R_fill=0,
        T_fill=0,
        dimensions=DIMS,
        nn_partitioning=True,
        regional_mode=True,
        coordinate_units="degrees",
        volume_filter_absolute=1.0,
        quiet=True,
    )
    kwargs.update(overrides)
    return kwargs


def _scenario_inputs(name, **make_kwargs):
    binary, meta = scenarios()[name]
    kwargs = {**meta["make_inputs"], **make_kwargs}
    return binary, meta, make_inputs(binary, **kwargs)


def _track(extremes, wet, bounds, areas, **overrides):
    """Construct a tracker and run it, returning ``(events, merges, tracker)``."""
    t = marEx.tracker(extremes, wet, layer_bounds=bounds, cell_areas=areas, **_kwargs(**overrides))
    assert type(t) is VolumetricTracker
    events, merges = t.run(return_merges=True)
    return events, merges, t


def _levels(id_field, event_id, t):
    """Depth indices occupied by ``event_id`` at timestep ``t`` (canonical order)."""
    slab = id_field.transpose(*CANONICAL).isel(time=t).values
    return sorted(set(np.nonzero(slab == event_id)[0].tolist()))


def _assert_datasets_equal(a, b, label=""):
    """Every data variable of ``a`` and ``b`` equal exactly, after putting ``b`` in ``a``'s dim order."""
    assert set(a.data_vars) == set(b.data_vars), label
    for name in a.data_vars:
        x, y = a[name], b[name].transpose(*a[name].dims)
        assert x.dtype == y.dtype, f"{label}{name}"
        np.testing.assert_array_equal(x.values, y.values, err_msg=f"{label}{name}")


# --------------------------------------------------------------------------------------
# Lineage end to end
# --------------------------------------------------------------------------------------


def test_planted_lineage_end_to_end():
    """Two events merge, are partitioned, and separate again: two events, three ledger rows."""
    binary, meta, (extremes, wet, bounds, areas) = _scenario_inputs("merge_then_split")
    events, merges, _ = _track(extremes, wet, bounds, areas)

    ids = events["ID_field"]
    assert ids.dtype == np.int32 and ids.dims == CANONICAL
    assert int(ids.max()) == 2 == events.attrs["N_events_final"]
    np.testing.assert_array_equal(events["ID"].values, [1, 2])
    # Both events live through all nine steps (a merge partitions; it does not end an event).
    assert events["presence"].values.all()
    # Occupied voxels are exactly the planted extremes, every one labelled.
    np.testing.assert_array_equal(ids.values > 0, binary)
    # One event spans several depth levels at once.
    assert len(_levels(ids, 1, 0)) >= 2

    # The ledger in object IDs, as the merge loop allocated them (identified IDs 1..15, merge IDs 16..18).
    np.testing.assert_array_equal(merges["parent_IDs"].values, [[5, 6], [7, 16], [8, 17]])
    np.testing.assert_array_equal(merges["child_IDs"].values, [[7, 16], [8, 17], [9, 18]])
    np.testing.assert_array_equal(merges["merge_time"].values, extremes["time"].values[[3, 4, 5]])
    np.testing.assert_array_equal(merges["n_parents"].values, [2, 2, 2])
    assert events.attrs["total_merges"] == 3
    assert events.attrs["multi_parent_merges"] == 0


@pytest.mark.parametrize("gap, n_events", [(0.0, 2), (10.0, 1)])
def test_vertical_gap_join_end_to_end(gap, n_events):
    """The two stacked blobs are separated by one 10 m level: joined only when 10 m is bridged."""
    binary, _, (extremes, wet, bounds, areas) = _scenario_inputs("vertical_gap_join")
    events, _, _ = _track(extremes, wet, bounds, areas, max_vertical_gap_m=gap)
    ids = events["ID_field"].transpose(*CANONICAL).values
    assert int(ids.max()) == n_events == events.attrs["N_events_final"]
    # The bridged level stays background and adds no volume.
    assert not (ids[:, 1] > 0).any()
    np.testing.assert_array_equal(ids > 0, binary)
    if n_events == 1:
        assert set(np.unique(ids[:, 0][ids[:, 0] > 0])) == set(np.unique(ids[:, 2][ids[:, 2] > 0])) == {1}
        np.testing.assert_array_equal(events["depth_top"].values, 0.0)
        np.testing.assert_array_equal(events["depth_bottom"].values, 40.0)


def test_true_on_dry_cells_end_to_end():
    """True on dry cells (a boolean that was never masked) is never labelled, counted or bridged."""
    binary, _, (extremes, wet, bounds, areas) = _scenario_inputs("vertical_gap_join")
    nt, nz, ny, nx = binary.shape
    dry = np.zeros((nz, ny, nx), bool)
    dry[1] = True  # the 10 m level between the blobs is land in every column
    dry[0, :, 9:] = True  # and a dry block beside the upper blob
    dirty = binary | dry[None]  # the user's boolean is True on every dry cell
    clean = binary & ~dry[None]
    _, _, (ext_dirty, wet_dry, bounds, areas) = _scenario_inputs("vertical_gap_join", dry=dry)
    ext_dirty = ext_dirty.copy(data=dirty).chunk({"time": 4})
    ext_clean = ext_dirty.copy(data=clean).chunk({"time": 4})

    # A 10 m gap would bridge the blobs if level 1 were wet; it is dry, so it is a barrier.
    ev_dirty, mg_dirty, t_dirty = _track(ext_dirty, wet_dry, bounds, areas, max_vertical_gap_m=10.0, R_fill=1)
    ev_clean, mg_clean, t_clean = _track(ext_clean, wet_dry, bounds, areas, max_vertical_gap_m=10.0, R_fill=1)

    ids = ev_dirty["ID_field"].transpose(*CANONICAL).values
    assert not (ids[:, dry] > 0).any()
    assert int(ids.max()) == 2
    _assert_datasets_equal(ev_clean, ev_dirty, "events ")
    _assert_datasets_equal(mg_clean, mg_dirty, "merges ")
    for key in ("N_objects_prefiltered", "N_objects_filtered", "volume_threshold (m^3)", "preprocessed_volume_fraction"):
        assert ev_dirty.attrs[key] == ev_clean.attrs[key], key


# --------------------------------------------------------------------------------------
# Layout invariance: dimension order, depth dtype, chunking, depth order and sign
# --------------------------------------------------------------------------------------

# Non-dyadic layer thicknesses (GLORYS-like), so the level centres are inexact in float32.
_IRREGULAR = [1.0216, 2.1123, 3.4987, 5.2207]


def _with_depth(extremes, wet, bounds, depth_values, attrs):
    """Give the three inputs one new depth coordinate (same order as the old one)."""
    depth = xr.DataArray(depth_values, dims="depth", attrs=attrs)
    return (
        extremes.assign_coords(depth=depth),
        wet.assign_coords(depth=depth),
        bounds.assign_coords(depth=depth),
    )


@pytest.mark.parametrize("mask_depth", ["same", "rounded", "float64"])
def test_transposed_dims_and_float32_depth_match(mask_depth):
    """
    A float32 depth coordinate and scrambled dimension orders. The mask carries the data's depth
    labels ("same"), labels that differ only by rounding ("rounded", accepted as the same grid),
    or the float64 originals ("float64", which differ by float32 rounding and are accepted as the same grid).
    """
    binary, _, (extremes, wet, bounds, areas) = _scenario_inputs("merge_then_split", thickness=_IRREGULAR)
    reference, ref_merges, _ = _track(extremes, wet, bounds, areas)

    depth32 = extremes["depth"].values.astype(np.float32)
    assert not np.array_equal(depth32.astype(np.float64), extremes["depth"].values)  # really inexact
    ext, msk, bnd = _with_depth(extremes, wet, bounds, depth32, {"units": "meters", "positive": "down"})
    if mask_depth == "rounded":
        msk = msk.assign_coords(depth=depth32.astype(np.float64) * (1.0 + 1.0e-12))
        assert not np.array_equal(msk["depth"].values, ext["depth"].values)
    elif mask_depth == "float64":
        msk = msk.assign_coords(depth=extremes["depth"].values)  # float64 originals
        assert msk["depth"].dtype == np.float64 and ext["depth"].dtype == np.float32
    ext = ext.transpose("time", "lat", "lon", "depth")
    msk = msk.transpose("lat", "lon", "depth")
    events, merges, _ = _track(ext, msk, bnd, areas)

    assert events["ID_field"].dims == ("time", "lat", "lon", "depth")
    assert events["depth"].dtype == np.float32
    np.testing.assert_array_equal(events["depth"].values, depth32)
    assert events["depth"].attrs == {"units": "meters", "positive": "down"}
    _assert_datasets_equal(reference, events)
    _assert_datasets_equal(ref_merges, merges)


def test_depth_chunked_input_warns_and_matches():
    binary, _, (extremes, wet, bounds, areas) = _scenario_inputs("merge_then_split")
    reference, ref_merges, _ = _track(extremes, wet, bounds, areas)

    chunked = extremes.chunk({"time": 4, "depth": 1, "lat": 3})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = marEx.tracker(chunked, wet, layer_bounds=bounds, cell_areas=areas, **_kwargs())
    rechunk = [w for w in caught if issubclass(w.category, UserWarning) and "Rechunking" in str(w.message)]
    assert len(rechunk) == 1
    assert all(len(t.data_bin.chunksizes[d]) == 1 for d in ("depth", "lat", "lon"))
    events, merges = t.run(return_merges=True)
    _assert_datasets_equal(reference, events)
    _assert_datasets_equal(ref_merges, merges)


@pytest.mark.parametrize("reverse, up", [(True, False), (False, True), (True, True)])
def test_depth_reversal_and_positive_up_invariance(reverse, up):
    binary, _, _ = _scenario_inputs("merge_then_split")
    # The planted balls are symmetric in depth; empty the deepest level so a field that is not
    # flipped back cannot pass by symmetry.
    binary = binary.copy()
    binary[:, -1] = False
    extremes, wet, bounds, areas = make_inputs(binary, thickness=_IRREGULAR)
    assert not np.array_equal(binary, binary[:, ::-1])
    reference, ref_merges, _ = _track(extremes, wet, bounds, areas)

    ext, msk, bnd = extremes, wet, bounds
    if up:
        ext, msk, bnd = _with_depth(ext, msk, -bnd, -extremes["depth"].values, {"units": "m", "positive": "up"})
    if reverse:
        ext, msk, bnd = (a.isel(depth=slice(None, None, -1)) for a in (ext, msk, bnd))
    events, merges, _ = _track(ext, msk, bnd, areas)

    # The output depth coordinate is the input's, in the input's order, with its attributes.
    np.testing.assert_array_equal(events["depth"].values, ext["depth"].values)
    assert events["depth"].attrs == ext["depth"].attrs
    assert events["ID_field"].dims == ext.dims
    back = events.isel(depth=slice(None, None, -1)) if reverse else events
    np.testing.assert_array_equal(back["ID_field"].values, reference["ID_field"].values)
    _assert_datasets_equal(reference.drop_vars("ID_field"), back.drop_vars("ID_field"))
    _assert_datasets_equal(ref_merges, merges)
    # Depth diagnostics stay positive-down metres whatever the input convention.
    assert float(events["depth_top"].min()) >= 0.0


# --------------------------------------------------------------------------------------
# Outputs, attributes and checkpointing
# --------------------------------------------------------------------------------------


def test_attrs_recorded():
    _, _, (extremes, wet, bounds, areas) = _scenario_inputs("merge_then_split")
    extremes = extremes.assign_attrs(
        N_events_final=987,
        volume_units="km3",
        overlap_threshold=0.9,
        horizontal_coordinate_units="radians",
        volume_filter_quartile=0.9,
        source="synthetic",
    )
    events, _, t = _track(extremes, wet, bounds, areas, max_vertical_gap_m=5.0, overlap_threshold=0.4)
    attrs = events.attrs
    missing = [k for k in EXPECTED_ATTRS if k not in attrs]
    assert not missing, missing
    assert attrs["N_events_final"] == events.sizes["ID"] == 2
    assert attrs["horizontal_coordinate_units"] == "degrees"
    assert attrs["source"] == "synthetic"
    assert attrs["volume_units"] == "m3"
    assert attrs["max_vertical_gap_m"] == 5.0
    assert attrs["vertical_geometry"] == "bounds"
    assert attrs["overlap_threshold"] == 0.4
    np.testing.assert_array_equal(attrs["partition_scales_m"], t.partition_scales)
    assert "5.0" in attrs["spatial_connectivity"] and "8-neighbour" in attrs["spatial_connectivity"]
    assert "L_H" in attrs["partition_metric"] and "L_Z" in attrs["partition_metric"]
    # Only the active volume filter is recorded (a None attribute cannot be written to netCDF).
    assert attrs["volume_filter_absolute"] == 1.0 and "volume_filter_quartile" not in attrs
    assert attrs["volume_threshold (m^3)"] == 1.0
    assert (attrs["R_fill"], attrs["T_fill"]) == (0, 0)
    assert attrs["N_objects_prefiltered"] == attrs["N_objects_filtered"] == 15
    assert not any(v is None for v in attrs.values())

    quantile_events, _, _ = _track(extremes, wet, bounds, areas, volume_filter_absolute=None, volume_filter_quartile=0.0)
    assert quantile_events.attrs["volume_filter_quartile"] == 0.0
    assert "volume_filter_absolute" not in quantile_events.attrs


def test_no_area_variable():
    _, _, (extremes, wet, bounds, areas) = _scenario_inputs("merge_then_split")
    events, _, _ = _track(extremes, wet, bounds, areas)
    assert "area" not in events.variables
    for name in DIAGNOSTICS:
        assert name in events.data_vars, name
    assert events["centroid"].sizes["component"] == 2
    assert events["centroid"].attrs["units"] == "degrees"


def _append_cyclic_endpoint(array):
    longitude = float(array.lon[0]) + 360.0
    return xr.concat([array, array.isel(lon=slice(0, 1)).assign_coords(lon=[longitude])], dim="lon")


@pytest.mark.parametrize("units, start", [("degrees", 0.0), ("radians", -180.0)])
def test_duplicate_endpoint_preserves_event_lineage_and_diagnostics(units, start):
    binary = np.zeros((2, 1, 1, 4), dtype=bool)
    binary[0, 0, 0, [0, 1]] = True
    binary[1, 0, 0, [3, 0]] = True
    extremes, wet, bounds, areas = make_inputs(binary, thickness=[1.0], lon0=start, dlon=90.0)
    inputs = (extremes, wet, areas)
    with_endpoint = tuple(_append_cyclic_endpoint(da) for da in inputs)
    if units == "radians":
        inputs = tuple(da.assign_coords(lat=np.deg2rad(da.lat), lon=np.deg2rad(da.lon)) for da in inputs)
        with_endpoint = tuple(da.assign_coords(lat=np.deg2rad(da.lat), lon=np.deg2rad(da.lon)) for da in with_endpoint)

    options = dict(regional_mode=False, coordinate_units=None, overlap_threshold=0.6)
    reference, ref_merges, _ = _track(inputs[0], inputs[1], bounds, inputs[2], **options)
    events, merges, _ = _track(with_endpoint[0], with_endpoint[1], bounds, with_endpoint[2], **options)
    assert reference.attrs["N_events_final"] == 2  # One of each object's two physical cells overlaps.
    assert events.ID_field.chunksizes["lon"] == (5,)
    xr.testing.assert_identical(events.isel(lon=slice(None, -1)), reference)
    xr.testing.assert_identical(merges, ref_merges)
    np.testing.assert_array_equal(events.ID_field.isel(lon=-1), reference.ID_field.isel(lon=0))
    xr.testing.assert_identical(events.lon, with_endpoint[0].lon)


@pytest.mark.parametrize("which", ["extremes", "mask"])
def test_inconsistent_duplicate_seam_raises(which):
    binary = np.ones((2, 1, 1, 5), dtype=bool)
    dry = np.zeros((1, 1, 5), dtype=bool)
    if which == "extremes":
        binary[1, 0, 0, -1] = False
    else:
        dry[0, 0, -1] = True
    extremes, wet, bounds, areas = make_inputs(binary, thickness=[1.0], dlon=90.0, dry=dry)
    with pytest.raises(DataValidationError, match=rf"{which}.*duplicate longitude endpoint"):
        _track(extremes, wet, bounds, areas, regional_mode=False)


@pytest.mark.parametrize("keep_attrs", [False, True])
def test_radian_centroid_declares_units_after_conversion(keep_attrs):
    inputs = make_inputs(np.ones((2, 1, 1, 1), dtype=bool), thickness=[1.0], lat0=30.5, lon0=-169.5)
    inputs = tuple(da.assign_coords(lat=np.deg2rad(da.lat), lon=np.deg2rad(da.lon)) if "lon" in da.dims else da for da in inputs)
    with xr.set_options(keep_attrs=keep_attrs):
        events, _, _ = _track(*inputs, coordinate_units="radians")
    assert events.centroid.attrs["units"] == events.attrs["horizontal_coordinate_units"] == "radians"
    np.testing.assert_allclose(events.centroid.isel(time=0, ID=0), np.deg2rad([30.5, -169.5]), rtol=0.0, atol=1e-14)


def test_centroid_uses_finite_wet_longitudes_for_its_origin():
    binary = np.array([[[[False, False, False, True]]]] * 2, dtype=bool)
    dry = np.array([[[False, True, True, False]]])
    inputs = make_inputs(binary, thickness=[1.0], lat0=25.0, dry=dry)
    inputs = tuple(da.assign_coords(lon=[10.0, np.nan, -20.0, 350.0]) if "lon" in da.dims else da for da in inputs)
    events, _, _ = _track(*inputs)
    np.testing.assert_allclose(events.centroid.isel(time=0, ID=0), [25.0, 350.0], rtol=0.0, atol=1e-12)


def test_checkpoint_save_then_load_matches(client, tmp_path):
    # Preprocessing that does something: a hole closed in space and a gap closed in time.
    balls = [Ball(0, 7, z=1.5, y=5.0, x=5.5, rz=2.0, ry=3.5, rx=3.5, vx=0.2)]
    binary = paint(8, 4, 10, 12, balls)
    binary[3] = False  # a one-step gap, closed by T_fill=2
    binary[5, :, 5, 5] = False  # a one-cell hole, closed by R_fill=1
    extremes, wet, bounds, areas = make_inputs(binary, thickness=[10.0, 10.0, 20.0, 20.0])
    kw = dict(R_fill=1, T_fill=2, temp_dir=str(tmp_path))

    surface = marEx.tracker(
        extremes.isel(depth=0, drop=True),
        wet.isel(depth=0, drop=True),
        R_fill=0,
        T_fill=0,
        area_filter_absolute=1,
        regional_mode=True,
        coordinate_units="degrees",
        temp_dir=str(tmp_path),
        quiet=True,
    )
    surface_field, surface_stats = surface.run_preprocess(checkpoint="save")
    surface_field = surface_field.compute()

    saved = marEx.tracker(extremes, wet, layer_bounds=bounds, cell_areas=areas, **_kwargs(checkpoint="save", **kw))
    events_saved, merges_saved = saved.run(return_merges=True)
    assert os.path.isdir(tmp_path / "marEx_tracker4d_checkpoint_proc_bin.zarr")
    stats = np.load(tmp_path / "marEx_tracker4d_checkpoint_stats.npz")
    assert sorted(stats.files) == sorted(
        [
            "total_volume_IDed",
            "N_objects_prefiltered",
            "N_objects_filtered",
            "volume_threshold",
            "accepted_volume_fraction",
            "preprocessed_volume_fraction",
        ]
    )
    # Preprocessing changed the field, so the checkpoint is not the raw input.
    ids = events_saved["ID_field"].transpose(*CANONICAL).values
    assert (ids[3] > 0).any() and ids[5, :, 5, 5].any()

    loaded = marEx.tracker(extremes, wet, layer_bounds=bounds, cell_areas=areas, **_kwargs(**kw))
    events_loaded, merges_loaded = loaded.run(return_merges=True, checkpoint="load")
    _assert_datasets_equal(events_saved, events_loaded)
    _assert_datasets_equal(merges_saved, merges_loaded)
    assert events_saved.attrs.keys() == events_loaded.attrs.keys()
    for key, value in events_saved.attrs.items():
        np.testing.assert_array_equal(events_loaded.attrs[key], value, err_msg=key)
    surface_reloaded, reloaded_stats = surface.run_preprocess(checkpoint="load")
    xr.testing.assert_identical(surface_reloaded, surface_field)
    assert reloaded_stats == surface_stats


def test_return_merges_has_overlap_volumes_int32_counts():
    _, _, (extremes, wet, bounds, areas) = _scenario_inputs("merge_then_split")
    events, merges, _ = _track(extremes, wet, bounds, areas)
    assert "overlap_areas" not in merges.variables
    assert merges["overlap_volumes"].dtype == np.float64
    assert merges["overlap_volumes"].dims == ("merge_ID", "parent_idx")
    assert (merges["overlap_volumes"].values > 0).all()
    for name in ("n_parents", "n_children", "parent_IDs", "child_IDs"):
        assert merges[name].dtype == np.int32, name
    # Object IDs (the merge loop's), not event IDs: the parents are numbered past the event count.
    assert int(merges["parent_IDs"].max()) > int(events["ID_field"].max())


# --------------------------------------------------------------------------------------
# Client memory model
# --------------------------------------------------------------------------------------


def test_client_memory_warning(monkeypatch):
    _, _, (extremes, wet, bounds, areas) = _scenario_inputs("merge_then_split")
    nt, nz, ny, nx = extremes.shape
    n_voxels = nz * ny * nx
    time_chunk = extremes.chunks[0][0]
    # The static term is one full volume array and four depth-only arrays.
    held = marEx.tracker(extremes, wet, layer_bounds=bounds, cell_areas=areas, **_kwargs())
    static = sum(getattr(held.geom, name).nbytes for name in ("cell_volume", "top", "bottom", "midpoint", "thickness"))
    assert static == (n_voxels + 4 * nz) * 8
    pristine = 2 * n_voxels * 4
    expected = 11 * time_chunk * n_voxels * 4 + static + pristine
    sizes = {"layer_elements": nz, "pristine_slices": 2}
    assert _merge_loop.merge_loop_client_bytes(n_voxels, time_chunk, "persist", **sizes) == expected
    assert _merge_loop.merge_loop_client_bytes(n_voxels, time_chunk, "streaming", **sizes) == (
        2 * time_chunk * n_voxels * 4 + static + pristine
    )
    assert _tracker4d.CLIENT_WARN_BYTES == 8 * 2**30

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        marEx.tracker(extremes, wet, layer_bounds=bounds, cell_areas=areas, **_kwargs())
    assert not [w for w in caught if "client" in str(w.message).lower()]

    monkeypatch.setattr(_tracker4d, "CLIENT_WARN_BYTES", expected - 1)
    with pytest.warns(UserWarning, match="merge loop") as record:
        marEx.tracker(extremes, wet, layer_bounds=bounds, cell_areas=areas, **_kwargs())
    message = " ".join(str(w.message) for w in record if "merge loop" in str(w.message))
    assert f"{expected:,} bytes" in message
    assert re.search(r"11 retained chunks x 4 time steps x 480 voxels x 4 B", message)
    assert f"(480 voxels + 4 x {nz} layer elements) x 8 B" in message
    assert f"{expected - 1:,}" in message

    monkeypatch.setattr(_tracker4d, "CLIENT_WARN_BYTES", expected)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        marEx.tracker(extremes, wet, layer_bounds=bounds, cell_areas=areas, **_kwargs())
    assert not [w for w in caught if "merge loop" in str(w.message)]


# --------------------------------------------------------------------------------------
# Silent-wrong sites
# --------------------------------------------------------------------------------------


def test_silent_wrong_sites_pinned():
    """
    One object spanning two levels, on a non-square grid with unusual dimension names given in
    a scrambled order. Per-level ID offsets would give it two IDs; inferring the latitude
    dimension from position would swap lat and lon; a 3-component centroid would mix depth into
    the horizontal centroid.
    """
    nt, nz, ny, nx = 2, 3, 5, 7
    lat = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    lon = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0])
    values = np.zeros((nt, nz, ny, nx), bool)
    values[:, 0, 1, 5] = True  # one column at lat -10, lon 150 ...
    values[:, 1, 1, 5] = True  # ... through the top two levels
    edges = np.array([0.0, 10.0, 40.0, 100.0])
    centres = 0.5 * (edges[:-1] + edges[1:])
    coords = {
        "t": pd.date_range("2000-01-01", periods=nt),
        "lev": xr.DataArray(centres, dims="lev", attrs={"units": "m"}),
        "latitude": lat,
        "longitude": lon,
    }
    data = xr.DataArray(values, dims=("t", "lev", "latitude", "longitude"), coords=coords)
    data = data.transpose("t", "longitude", "lev", "latitude").chunk({"t": 1})
    wet = xr.ones_like(data.isel(t=0, drop=True), dtype=bool).compute().transpose("latitude", "lev", "longitude")
    bounds = xr.DataArray(np.stack([edges[:-1], edges[1:]], axis=1), dims=("lev", "nv"))
    areas = xr.DataArray(np.full((ny, nx), 1.0e6), dims=("latitude", "longitude"), attrs={"units": "m2"})

    t = marEx.tracker(
        data,
        wet,
        layer_bounds=bounds,
        cell_areas=areas,
        **_kwargs(dimensions={"time": "t", "z": "lev", "y": "latitude", "x": "longitude"}),
    )
    events = t.run()

    ids = events["ID_field"].transpose("t", "lev", "latitude", "longitude").values
    assert ids.dtype == np.int32
    np.testing.assert_array_equal(np.unique(ids), [0, 1])  # one ID across both levels
    np.testing.assert_array_equal(ids > 0, values)
    np.testing.assert_array_equal(events["latitude"].values, lat)
    np.testing.assert_array_equal(events["longitude"].values, lon)

    centroid = events["centroid"]
    assert centroid.sizes["component"] == 2
    np.testing.assert_allclose(centroid.isel(component=0).values.ravel(), -10.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(centroid.isel(component=1).values.ravel(), 150.0, rtol=0, atol=1e-9)
    # Volumes 1e7 and 3e7 m^3 at midpoints 5 and 25 m: (5 * 1 + 25 * 3) / 4 = 20 m.
    np.testing.assert_allclose(events["depth_centroid"].values.ravel(), 20.0, rtol=1e-15)
    np.testing.assert_array_equal(events["volume"].values.ravel(), 4.0e7)
    np.testing.assert_array_equal(events["footprint_area"].values.ravel(), 1.0e6)
    np.testing.assert_array_equal(events["depth_top"].values.ravel(), 0.0)
    np.testing.assert_array_equal(events["depth_bottom"].values.ravel(), 40.0)


# --------------------------------------------------------------------------------------
# The merge dataset in event IDs, and split records
# --------------------------------------------------------------------------------------


def _event_at(events, t, x):
    """Event occupying the planted ball centred at column ``x`` of timestep ``t``."""
    return int(events["ID_field"].transpose(*CANONICAL).isel(time=t, depth=1, lat=5, lon=x).compute().item())


def test_merges_name_the_events_of_their_objects():
    _, _, (extremes, wet, bounds, areas) = _scenario_inputs("merge_then_split")
    events, merges, _ = _track(extremes, wet, bounds, areas)
    west, east = _event_at(events, 2, 3), _event_at(events, 2, 8)
    assert {west, east} == {1, 2}
    # Object 5 is the western ball at t=2 and 6 the eastern; the merge loop keeps that order all three days.
    np.testing.assert_array_equal(merges["parent_event_IDs"].values, [[west, east]] * 3)
    np.testing.assert_array_equal(merges["child_event_IDs"].values, [[west, east]] * 3)
    np.testing.assert_array_equal(merges["n_parent_events"].values, [2, 2, 2])
    for name in ("parent_event_IDs", "child_event_IDs", "n_parent_events"):
        assert merges[name].dtype == np.int32, name
    assert "_renamed_from" not in merges.variables and "rename_ID" not in merges.sizes


def test_one_split_is_recorded_once():
    """The kept object has two pieces on every later day, and those days are not new splits."""
    _, _, (extremes, wet, bounds, areas) = _scenario_inputs("one_split")
    events, merges, _ = _track(extremes, wet, bounds, areas)
    assert merges.sizes["split_ID"] == 1
    np.testing.assert_array_equal(merges["split_time"].values, extremes["time"].values[[3]])
    np.testing.assert_array_equal(merges["split_parent_IDs"].values, [3])
    np.testing.assert_array_equal(merges["split_child_IDs"].values, [[4, 5]])
    np.testing.assert_array_equal(merges["n_split_children"].values, [2])
    np.testing.assert_array_equal(merges["split_event_IDs"].values, [1])
    assert merges["split_event_IDs"].dtype == np.int32
    assert events.attrs["total_splits"] == 1


@pytest.mark.parametrize("name", ["merge_then_split", "one_merge", "persist_grow_decay"])
def test_no_split_is_recorded_without_one(name):
    """Two events that merge and separate keep their identities: that is a partition, not a split."""
    _, _, (extremes, wet, bounds, areas) = _scenario_inputs(name)
    events, merges, _ = _track(extremes, wet, bounds, areas)
    assert merges.sizes["split_ID"] == 0
    assert merges["split_time"].dtype == extremes["time"].dtype
    assert events.attrs["total_splits"] == 0


def test_merges_with_no_split_write_to_zarr(tmp_path):
    _, _, (extremes, wet, bounds, areas) = _scenario_inputs("merge_then_split")
    _, merges, _ = _track(extremes, wet, bounds, areas)
    merges.to_zarr(tmp_path / "merges.zarr", mode="w")
    back = xr.open_zarr(tmp_path / "merges.zarr").load()
    assert back.sizes["split_ID"] == 0
    np.testing.assert_array_equal(back["child_event_IDs"].values, merges["child_event_IDs"].values)


def test_checkpoint_reload_preserves_millisecond_datetimes(tmp_path):
    _, _, (extremes, wet, bounds, areas) = _scenario_inputs("one_merge")
    ms_time = extremes["time"].values.astype("datetime64[ms]")
    extremes_ms = extremes.assign_coords(time=ms_time, obs_ms=("time", ms_time + np.timedelta64(500, "ms"))).chunk({"time": 3})
    trk = marEx.regional_tracker(
        extremes_ms,
        wet,
        coordinate_units="degrees",
        R_fill=0,
        T_fill=0,
        volume_filter_absolute=1.0,
        layer_bounds=bounds,
        cell_areas=areas,
        dimensions={"time": "time", "z": "depth", "y": "lat", "x": "lon"},
        temp_dir=str(tmp_path),
        quiet=True,
    )
    trk.run_preprocess(checkpoint="save")
    events_same, merges_same = trk.run(checkpoint="load", return_merges=True)
    assert events_same["time"].dtype == np.dtype("datetime64[ms]")
    assert events_same["time_start"].dtype == np.dtype("datetime64[ms]")
    assert events_same["ID_field"].coords["obs_ms"].dtype == np.dtype("datetime64[ms]")
    assert merges_same["merge_time"].dtype == np.dtype("datetime64[ms]")


def test_regional_mode_keeps_360_span_boundary_adjacent(tmp_path):
    time = pd.date_range("2001-01-01", periods=2, freq="D")
    lon = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    lat = np.array([-10.0, 0.0, 10.0])
    depth = np.array([5.0])
    data = np.zeros((2, 1, 3, 5), dtype=bool)
    data[:, 0, 1, [0, 3, 4]] = True
    extremes = xr.DataArray(
        data,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": time, "depth": ("depth", depth, {"units": "m"}), "lat": lat, "lon": lon},
    ).chunk({"time": 1})
    wet = xr.DataArray(
        np.ones((1, 3, 5), dtype=bool), dims=("depth", "lat", "lon"), coords={"depth": depth, "lat": lat, "lon": lon}
    )
    bounds = xr.DataArray(np.array([[0.0, 10.0]]), dims=("depth", "bounds"), coords={"depth": depth})
    events = (
        marEx.regional_tracker(
            extremes,
            wet,
            coordinate_units="degrees",
            R_fill=0,
            T_fill=0,
            grid_resolution=90.0,
            volume_filter_absolute=1.0,
            layer_bounds=bounds,
            dimensions={"time": "time", "z": "depth", "y": "lat", "x": "lon"},
            temp_dir=str(tmp_path),
            quiet=True,
        )
        .run()
        .compute()
    )
    row = events["ID_field"].values[0, 0, 1, :]
    assert row[0] > 0 and row[3] > 0
    assert row[3] == row[4]
    assert row[0] != row[4]


def test_cf_encoded_cftime_zero_split_ledger_matches_across_compute_modes(tmp_path):
    _, _, (extremes, wet, bounds, areas) = _scenario_inputs("one_merge")
    cftime_axis = xr.date_range("2001-01-01", periods=extremes.sizes["time"], calendar="360_day", use_cftime=True)
    extremes = extremes.assign_coords(time=cftime_axis).chunk({"time": 2})
    extremes["time"].encoding = {"units": "hours since 1990-01-01", "calendar": "360_day"}
    common = dict(
        coordinate_units="degrees",
        R_fill=0,
        T_fill=0,
        volume_filter_absolute=1.0,
        layer_bounds=bounds,
        cell_areas=areas,
        dimensions={"time": "time", "z": "depth", "y": "lat", "x": "lon"},
        quiet=True,
    )
    ev_p, mg_p = marEx.regional_tracker(extremes, wet, compute_mode="persist", temp_dir=str(tmp_path / "p"), **common).run(
        return_merges=True
    )
    ev_s, mg_s = marEx.regional_tracker(extremes, wet, compute_mode="streaming", temp_dir=str(tmp_path / "s"), **common).run(
        return_merges=True
    )
    try:
        assert mg_p.sizes["split_ID"] == 0
        xr.testing.assert_identical(mg_s.compute(), mg_p.compute())
    finally:
        marEx.clear_staging(ev_s)


def test_duplicate_seam_validation_respects_flipped_depth_and_transposed_dims(tmp_path):
    import dask.array as dsa

    from marEx.exceptions import DataValidationError

    time = pd.date_range("2001-01-01", periods=4, freq="D")
    lon = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    lat = np.array([-30.0, 30.0])
    depth_desc = np.array([15.0, 5.0])  # descending -> flip_z=True; index 0 is deep dry, index 1 is shallow wet
    wet = xr.DataArray(
        np.array([np.zeros((2, 5), dtype=bool), np.ones((2, 5), dtype=bool)]),
        dims=("depth", "lat", "lon"),
        coords={"depth": depth_desc, "lat": lat, "lon": lon},
    )
    data = np.zeros((4, 2, 2, 5), dtype=bool)
    data[:, 1, 0, [0, 4]] = True
    months = dsa.from_array(np.datetime64("2020-01", "M") + np.arange(4).astype("timedelta64[M]"), chunks=2)
    extremes = xr.DataArray(
        dsa.from_array(data, chunks=(2, 2, 2, 5)),
        dims=("time", "depth", "lat", "lon"),
        coords={
            "time": time,
            "depth": ("depth", depth_desc, {"units": "m"}),
            "lat": lat,
            "lon": lon,
            "obs_month": ("time", months),
        },
    )
    bounds = xr.DataArray(np.array([[10.0, 20.0], [0.0, 10.0]]), dims=("depth", "bnds"), coords={"depth": depth_desc})

    # Dry-only mismatch on deep level (index 0) must be accepted; Dask datetime64[M] must survive checkpoint & streaming.
    vols_ok = np.full((2, 2, 5), 1e10, dtype=np.float64)
    vols_ok[0, :, 0] = np.nan
    vols_ok[0, :, 4] = 999.0
    cv_ok = xr.DataArray(vols_ok, dims=("depth", "lat", "lon")).transpose("lat", "depth", "lon")
    common = dict(
        R_fill=0,
        T_fill=0,
        dimensions={"time": "time", "z": "depth", "y": "lat", "x": "lon"},
        layer_bounds=bounds,
        cell_volumes=cv_ok,
        grid_resolution=90.0,
        volume_filter_absolute=1.0,
        quiet=True,
    )
    trk_ck = marEx.tracker(extremes, wet, temp_dir=str(tmp_path / "ck"), compute_mode="persist", **common)
    trk_ck.run_preprocess(checkpoint="save")
    ev_ck_lazy = trk_ck.run(checkpoint="load")
    ev_st = marEx.tracker(extremes, wet, temp_dir=str(tmp_path / "st"), compute_mode="streaming", **common).run()
    try:
        assert ev_ck_lazy["obs_month"].dtype == np.dtype("datetime64[M]")
        assert ev_st["obs_month"].dtype == np.dtype("datetime64[M]")
        ev_ck, ev_st_c = ev_ck_lazy.compute(), ev_st.compute()
        xr.testing.assert_identical(ev_st_c, ev_ck)
    finally:
        marEx.clear_staging(ev_st)

    # Wet mismatch on shallow level (index 1) under flip_z=True and transposed (lat, depth, lon) with nz == ny must fail.
    vols_bad = np.full((2, 2, 5), 1e10, dtype=np.float64)
    vols_bad[1, 0, 4] = 9e10
    cv_bad = xr.DataArray(vols_bad, dims=("depth", "lat", "lon")).transpose("lat", "depth", "lon")
    with pytest.raises(DataValidationError, match="cell_volumes differs on wet cells"):
        marEx.tracker(extremes, wet, temp_dir=str(tmp_path / "bad"), **{**common, "cell_volumes": cv_bad})
