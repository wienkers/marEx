"""Labelled geometry inputs are matched to the data by their coordinates, never by array position."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError

DIMS = {"time": "time", "z": "depth", "y": "lat", "x": "lon"}
LAT = np.array([60.0, 0.0])
LON = np.array([0.0, 1.0, 2.0])
DEPTH = np.array([5.0, 60.0])
TOP, BOTTOM = np.array([0.0, 10.0]), np.array([10.0, 110.0])


def _binary():
    """Left column at t0, top row at t1, both on level 0."""
    values = np.zeros((2, 2, 2, 3), dtype=bool)
    values[0, 0, :, 0] = True
    values[1, 0, 0, :] = True
    return values


def _data(values=None):
    depth = xr.DataArray(DEPTH, dims="depth", attrs={"units": "m", "positive": "down"})
    coords = {"time": pd.date_range("2000-01-01", periods=2), "depth": depth, "lat": LAT, "lon": LON}
    data = xr.DataArray(_binary() if values is None else values, dims=("time", "depth", "lat", "lon"), coords=coords)
    return data.chunk({"time": 1, "depth": -1, "lat": -1, "lon": -1})


def _wet():
    return xr.DataArray(
        np.ones((2, 2, 3), dtype=bool), dims=("depth", "lat", "lon"), coords={"depth": DEPTH, "lat": LAT, "lon": LON}
    )


def _bounds():
    return xr.DataArray(np.stack([TOP, BOTTOM], axis=1), dims=("depth", "nv"), coords={"depth": DEPTH}, attrs={"units": "m"})


def _areas():
    """Areas that differ along both axes, so a reversal of either moves the overlap fraction across 0.5."""
    return xr.DataArray(
        np.array([[1.0, 1.0, 2.0], [2.0, 2.0, 4.0]]),
        dims=("lat", "lon"),
        coords={"lat": LAT, "lon": LON},
        attrs={"units": "m2"},
    )


def _thickness():
    values = np.broadcast_to((BOTTOM - TOP)[:, None, None], (2, 2, 3)).copy()
    values[0, 1, 2] = 5.0  # a partial cell, at one labelled place
    return xr.DataArray(values, dims=("depth", "lat", "lon"), coords={"depth": DEPTH, "lat": LAT, "lon": LON})


def _volumes():
    return _thickness() * _areas().transpose("lat", "lon")


def _track(data=None, mask=None, dimensions=None, coordinates=None, **geometry):
    geometry.setdefault("layer_bounds", _bounds())
    geometry.setdefault("cell_areas", _areas())
    tracker = marEx.tracker(
        _data() if data is None else data,
        _wet() if mask is None else mask,
        R_fill=0,
        T_fill=0,
        dimensions=dimensions or DIMS,
        coordinates=coordinates,
        regional_mode=True,
        coordinate_units="degrees",
        volume_filter_absolute=0.5,
        quiet=True,
        **geometry,
    )
    return tracker.run().compute()


def _assert_same(a, b):
    np.testing.assert_array_equal(a["ID_field"].values, b["ID_field"].values)
    np.testing.assert_array_equal(a["volume"].values, b["volume"].values)


@pytest.mark.parametrize(
    "reorder",
    [lambda d: d.sortby("lat"), lambda d: d.sortby("lon", ascending=False)],
    ids=["lat", "lon"],
)
def test_reordered_cell_areas_give_the_same_events(reorder):
    # Sorting latitude alone took this field from two events to one when the areas were read by position.
    reference = _track()
    assert reference.attrs["N_events_final"] == 2
    _assert_same(_track(cell_areas=reorder(_areas())), reference)


def test_reordered_layer_bounds_keep_each_level_thickness():
    reference = _track()  # level 0 only, so the wrong thickness would change every volume
    _assert_same(_track(layer_bounds=_bounds().sortby("depth", ascending=False)), reference)


@pytest.mark.parametrize(
    "name, make", [("layer_thickness", _thickness), ("cell_volumes", _volumes)], ids=["layer_thickness", "cell_volumes"]
)
def test_reordered_3d_geometry_gives_the_same_events(name, make):
    values = _binary()
    values[1, 0, 1, 2] = True  # reach the partial cell
    reference = _track(data=_data(values), **{name: make()})
    shuffled = make().sortby("depth", ascending=False).sortby("lat").sortby("lon", ascending=False)
    _assert_same(_track(data=_data(values), **{name: shuffled}), reference)


def test_bounds_follow_the_labels_when_the_data_is_stored_deepest_first():
    # The data is flipped internally, so the bounds must be matched to its labels before that flip.
    reference = _track()
    flipped = _data().isel(depth=slice(None, None, -1))
    wet = _wet().isel(depth=slice(None, None, -1))
    result = _track(data=flipped, mask=wet)  # bounds still listed shallowest first
    np.testing.assert_array_equal(result["volume"].values, reference["volume"].values)


def test_float32_labels_reordered_are_still_matched():
    lat32 = xr.DataArray(np.array([60.1, 0.1], dtype=np.float32), dims="lat")
    data = _data().assign_coords(lat=lat32)
    mask = _wet().assign_coords(lat=lat32)
    areas = _areas().assign_coords(lat=np.array([60.1, 0.1], dtype=np.float64))
    reference = _track(data=data, mask=mask, cell_areas=areas)
    _assert_same(_track(data=data, mask=mask, cell_areas=areas.sortby("lat")), reference)


@pytest.mark.parametrize(
    "name, make, dim",
    [
        ("cell_areas", _areas, "lat"),
        ("layer_bounds", _bounds, "depth"),
        ("layer_thickness", _thickness, "lon"),
        ("cell_volumes", _volumes, "lat"),
    ],
    ids=["cell_areas", "layer_bounds", "layer_thickness", "cell_volumes"],
)
def test_labels_from_another_grid_raise(name, make, dim):
    wrong = make()
    wrong = wrong.assign_coords({dim: wrong[dim].values + 0.5})
    with pytest.raises(ConfigurationError, match=rf"{name}.*'{dim}'"):
        _track(**{name: wrong})


def test_unlabelled_geometry_stays_positional():
    _assert_same(_track(cell_areas=_areas().drop_vars(["lat", "lon"])), _track())


def test_grid_resolution_ignores_the_labels_of_the_areas_it_overrides():
    wrong = _areas().assign_coords(lat=LAT + 0.5)
    _assert_same(_track(cell_areas=wrong, grid_resolution=1.0), _track(cell_areas=None, grid_resolution=1.0))


def _on_ji(obj):
    return obj.rename(lat="j", lon="i").drop_vars(["j", "i"]).assign_coords(latitude=("j", LAT), longitude=("i", LON))


def test_labels_on_separate_dimensions_are_matched_through_the_coordinate_names():
    dims = {"time": "time", "z": "depth", "y": "j", "x": "i"}
    coords = {"y": "latitude", "x": "longitude"}
    kwargs = dict(data=_on_ji(_data()), mask=_on_ji(_wet()), dimensions=dims, coordinates=coords)
    reference = _track(cell_areas=_on_ji(_areas()), **kwargs)
    assert reference.attrs["N_events_final"] == 2
    areas = _on_ji(_areas()).isel(j=slice(None, None, -1))  # same labelled values, reversed along j
    _assert_same(_track(cell_areas=areas, **kwargs), reference)


def test_a_mask_on_separate_dimensions_is_matched_through_the_coordinate_names():
    dims = {"time": "time", "z": "depth", "y": "j", "x": "i"}
    coords = {"y": "latitude", "x": "longitude"}
    wet = _wet()
    wet[:, 0, 0] = False  # dry at latitude 60, longitude 0
    kwargs = dict(data=_on_ji(_data()), cell_areas=_on_ji(_areas()), dimensions=dims, coordinates=coords)
    reference = _track(mask=_on_ji(wet), **kwargs)
    _assert_same(_track(mask=_on_ji(wet).isel(j=slice(None, None, -1)), **kwargs), reference)


def _on_indexed_ji(obj, *, geographic=True):
    """(j, i) index labels, with or without the latitude(j), longitude(i) coordinates of the data."""
    out = obj.rename(lat="j", lon="i").assign_coords(j=np.arange(obj.sizes["lat"]), i=np.arange(obj.sizes["lon"]))
    return (
        out.assign_coords(latitude=("j", LAT), longitude=("i", LON))
        if geographic
        else out.drop_vars(["lat", "lon"], errors="ignore")
    )


@pytest.mark.parametrize("name", ["mask", "cell_areas"])
def test_an_input_labelled_only_by_the_dimension_index_is_matched_by_it(name):
    dims = {"time": "time", "z": "depth", "y": "j", "x": "i"}
    coords = {"y": "latitude", "x": "longitude"}
    wet = _wet()
    wet[:, 0, 0] = False
    inputs = {"mask": _on_indexed_ji(wet, geographic=False), "cell_areas": _on_indexed_ji(_areas(), geographic=False)}
    reference = _track(data=_on_indexed_ji(_data()), dimensions=dims, coordinates=coords, **inputs)
    inputs[name] = inputs[name].isel(j=slice(None, None, -1))
    _assert_same(_track(data=_on_indexed_ji(_data()), dimensions=dims, coordinates=coords, **inputs), reference)


def test_distinct_labels_closer_than_the_rounding_bar_are_still_put_in_order():
    # Neighbouring float32 values: within the rounding bar of each other, yet different cells.
    lon32 = np.nextafter(np.nextafter(np.float32(150.0), np.float32(np.inf)), np.float32(np.inf))
    labels = np.array([150.0, np.nextafter(np.float32(150.0), np.float32(np.inf)), lon32], dtype=np.float32)
    data = _data().assign_coords(lon=labels)
    mask = _wet().assign_coords(lon=labels)
    areas = _areas().assign_coords(lon=labels)
    reference = _track(data=data, mask=mask, cell_areas=areas)
    _assert_same(_track(data=data, mask=mask, cell_areas=areas.isel(lon=slice(None, None, -1))), reference)


def test_many_distinct_labels_cannot_pair_with_one_repeated_label():
    step = np.nextafter(np.float32(150.0), np.float32(np.inf))
    labels = np.array([150.0, step, np.nextafter(step, np.float32(np.inf))], dtype=np.float32)
    areas = _areas().assign_coords(lon=np.full(3, 150.0, dtype=np.float32))
    with pytest.raises(ConfigurationError, match="cell_areas.*'lon'"):
        _track(data=_data().assign_coords(lon=labels), mask=_wet().assign_coords(lon=labels), cell_areas=areas)


def test_integer_labels_are_compared_exactly():
    base = 2147483646
    indexed = dict(j=np.arange(base, base + 2, dtype=np.int64))
    dims = {"time": "time", "z": "depth", "y": "j", "x": "i"}
    data = _on_indexed_ji(_data()).assign_coords(indexed)
    wet = _on_indexed_ji(_wet(), geographic=False).assign_coords(indexed)
    areas = _on_indexed_ji(_areas(), geographic=False).assign_coords(j=np.arange(base + 1, base + 3, dtype=np.int64))
    with pytest.raises(ConfigurationError, match="cell_areas.*'j'"):
        _track(data=data, mask=wet, cell_areas=areas, dimensions=dims, coordinates={"y": "latitude", "x": "longitude"})
