"""Volumetric tracker configuration: dispatch, rejections and data preparation."""

import re
import warnings

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError, TrackingError
from marEx.track.volumetric.config import (
    FOUR_D_ONLY_KWARGS,
    VolumetricDims,
    check_volumetric_config,
    prepare_volumetric_data,
)
from marEx.track.volumetric.tracker4d import VolumetricTracker

NT, NZ, NY, NX = 4, 3, 5, 6
DIMS = {"time": "time", "z": "depth", "y": "lat", "x": "lon"}
VDIMS = VolumetricDims(time="time", z="depth", y="lat", x="lon")
DEPTHS = np.array([5.0, 15.0, 30.0])
BOUNDS = np.array([[0.0, 10.0], [10.0, 20.0], [20.0, 40.0]])


def field(depths=DEPTHS, units="m", positive=None):
    """A 4-D boolean extremes field, one chunk per spatial dim, time chunk 2."""
    rng = np.random.default_rng(0)
    attrs = {"units": units}
    if positive is not None:
        attrs["positive"] = positive
    return xr.DataArray(
        rng.random((NT, NZ, NY, NX)) < 0.3,
        dims=("time", "depth", "lat", "lon"),
        coords={
            "time": np.arange(NT),
            "depth": xr.DataArray(depths, dims="depth", attrs=attrs),
            "lat": np.linspace(-10, 10, NY),
            "lon": np.linspace(0, 50, NX),
        },
    ).chunk({"time": 2})


def wet_mask(data):
    return xr.ones_like(data.isel(time=0, drop=True), dtype=bool)


def layer_bounds():
    return xr.DataArray(BOUNDS, dims=("depth", "bnds"), coords={"depth": DEPTHS})


def cell_areas():
    return xr.DataArray(np.full((NY, NX), 1.0e8), dims=("lat", "lon"), attrs={"units": "m2"})


def valid_kwargs():
    """Keyword arguments of a configuration that builds."""
    return dict(
        R_fill=1,
        dimensions=DIMS,
        layer_bounds=layer_bounds(),
        cell_areas=cell_areas(),
        nn_partitioning=True,
        regional_mode=True,
        coordinate_units="degrees",
        quiet=True,
    )


def build(data=None, mask="default", **overrides):
    data = field() if data is None else data
    mask = wet_mask(data) if isinstance(mask, str) else mask
    kwargs = {**valid_kwargs(), **overrides}
    return marEx.tracker(data, mask, **kwargs)


def test_valid_baseline_constructs():
    """Negative control for the rejection table: the baseline every row perturbs does build."""
    t = build()
    assert type(t) is VolumetricTracker
    assert t.data_bin.dims == ("time", "depth", "lat", "lon")
    assert t.geom.wet.shape == (NZ, NY, NX)
    assert t.partition_scales == t.geom.default_scales


REJECTIONS = [
    pytest.param({"area_filter_quartile": 0.5}, "volume_filter_quartile", id="area_filter_quartile"),
    pytest.param({"area_filter_absolute": 10}, "volume_filter_absolute", id="area_filter_absolute"),
    pytest.param({"allow_merging": False}, "allow_merging=True", id="allow_merging_false"),
    pytest.param({"nn_partitioning": False}, "nn_partitioning=True", id="nn_partitioning_false"),
    pytest.param({"unstructured_grid": True}, "structured", id="unstructured_grid"),
    pytest.param({"prefilter_min_cells": 5}, "prefilter_min_cells", id="prefilter_min_cells"),
    pytest.param({"max_vertical_gap_m": -1}, "max_vertical_gap_m", id="negative_gap"),
    pytest.param({"max_vertical_gap_m": float("nan")}, "max_vertical_gap_m", id="nan_gap"),
    pytest.param({"partition_scales": (0, 10)}, "partition_scales", id="scale_zero"),
    pytest.param({"partition_scales": (10, -1)}, "partition_scales", id="scale_negative"),
    pytest.param({"partition_scales": (10,)}, "partition_scales", id="scale_wrong_length"),
    pytest.param({"mask": None}, "wet mask", id="missing_mask"),
    pytest.param({"mask": "2d"}, r"\(depth, lat, lon\) wet mask", id="mask_2d"),
    pytest.param({"volume_filter_quartile": 0.5, "volume_filter_absolute": 1e9}, "both volume filtering", id="both_volume_filters"),
    pytest.param({"volume_filter_quartile": 1.5}, "in \\[0, 1\\]", id="quartile_out_of_range"),
    pytest.param({"volume_filter_absolute": -1.0}, "positive volume", id="absolute_not_positive"),
    pytest.param({"checkpoint": "save"}, "Checkpointing requires a temporary directory", id="checkpoint_without_temp_dir"),
]


@pytest.mark.parametrize("overrides, match", REJECTIONS)
def test_rejections(overrides, match):
    data = field()
    overrides = dict(overrides)
    if "mask" in overrides:
        mask = None if overrides.pop("mask") is None else wet_mask(data).isel(depth=0, drop=True)
        call = {"mask": mask}
    else:
        call = {}
    with pytest.raises(ConfigurationError) as excinfo:
        build(data, **call, **overrides)
    message = str(excinfo.value)
    del excinfo
    assert re.search(match, message), f"message does not match {match!r}: {message}"


def test_nn_partitioning_defaults_to_true_for_volumetric():
    kwargs = valid_kwargs()
    del kwargs["nn_partitioning"]
    t = marEx.tracker(field(), wet_mask(field()), **kwargs)
    assert isinstance(t, VolumetricTracker)
    assert t.nn_partitioning is True


def test_dispatch_routes_and_is_a_tracker():
    data = field()
    t = marEx.tracker(data, wet_mask(data), **valid_kwargs())
    assert isinstance(t, VolumetricTracker)
    assert isinstance(t, marEx.tracker)

    # `dimensions` as a positional argument routes the same way (data_bin, mask, R_fill, area_filter_quartile,
    # area_filter_absolute, temp_dir, T_fill, allow_merging, nn_partitioning, overlap_threshold,
    # unstructured_grid, dimensions).
    rest = {k: v for k, v in valid_kwargs().items() if k not in ("R_fill", "nn_partitioning", "dimensions")}
    positional = marEx.tracker(data, wet_mask(data), 1, None, None, None, 2, True, True, 0.5, False, DIMS, **rest)
    assert isinstance(positional, VolumetricTracker)


def test_regional_tracker_routes_4d():
    data = field()
    kwargs = valid_kwargs()
    for key in ("R_fill", "regional_mode", "coordinate_units", "quiet"):
        kwargs.pop(key)
    t = marEx.regional_tracker(data, wet_mask(data), coordinate_units="degrees", R_fill=1, quiet=True, **kwargs)
    assert isinstance(t, VolumetricTracker)
    assert t.regional_mode is True


@pytest.mark.parametrize("name", FOUR_D_ONLY_KWARGS)
def test_4d_kwarg_without_z_raises(name):
    data = field().isel(depth=0, drop=True)
    value = {"layer_bounds": layer_bounds(), "layer_thickness": data, "cell_volumes": data}.get(name, 1.0)
    with pytest.raises(ConfigurationError) as excinfo:
        marEx.tracker(data, R_fill=1, **{name: value})
    message = str(excinfo.value)
    del excinfo
    assert 'dimensions["z"]' in message and name in message


def test_undeclared_extra_dim_message_names_both_fixes():
    data = field()
    with pytest.raises(TrackingError) as excinfo:
        marEx.tracker(data, wet_mask(data).isel(depth=0, drop=True), R_fill=1, dimensions={"time": "time", "y": "lat", "x": "lon"})
    message = str(excinfo.value)
    del excinfo
    assert 'dimensions["z"]' in message and "isel(depth=0)" in message


def test_extra_dim_beyond_z_raises():
    data = field().expand_dims({"member": 2}, axis=1).chunk({"member": -1})
    mask = wet_mask(field())
    with pytest.raises(TrackingError, match=re.escape("extra dimension(s) ['member']")):
        build(data, mask)


def _data_with_depth_coordinate(depths, **attrs):
    data = field()
    z = xr.DataArray(depths, dims="depth", attrs=attrs)
    return data.assign_coords(depth=z)


Z_CHECKS = [
    pytest.param(lambda: field().drop_vars("depth"), "has no coordinate", id="missing_coordinate"),
    pytest.param(lambda: _data_with_depth_coordinate([5.0, 30.0, 15.0], units="m"), "strictly monotonic", id="non_monotonic"),
    pytest.param(lambda: _data_with_depth_coordinate(["a", "b", "c"], units="m"), "must be numeric", id="non_numeric"),
    pytest.param(lambda: field(units="km"), "units in metres", id="non_metre_units"),
    pytest.param(lambda: _data_with_depth_coordinate(DEPTHS), "units in metres", id="no_units"),
]


@pytest.mark.parametrize("make, match", Z_CHECKS)
def test_z_coordinate_checks(make, match):
    data = make()
    with pytest.raises(ConfigurationError, match=match):
        build(data, wet_mask(field()))


def test_default_volume_filter_is_median():
    base = dict(
        area_filter_quartile=None,
        area_filter_absolute=None,
        volume_filter_quartile=None,
        volume_filter_absolute=None,
        allow_merging=True,
        nn_partitioning=True,
        unstructured_grid=False,
        prefilter_min_cells=None,
        max_vertical_gap_m=0.0,
        partition_scales=None,
    )
    assert check_volumetric_config(**base) == (0.5, None)
    assert check_volumetric_config(**{**base, "volume_filter_quartile": 0.2}) == (0.2, None)
    assert check_volumetric_config(**{**base, "volume_filter_absolute": 5.0e9}) == (None, 5.0e9)
    t = build()
    assert (t.volume_filter_quartile, t.volume_filter_absolute) == (0.5, None)


@pytest.mark.parametrize(
    "depths, positive, expect_flip",
    [
        (DEPTHS, None, False),
        (DEPTHS[::-1], None, True),
        (-DEPTHS, "up", False),
        (-DEPTHS[::-1], "up", True),
    ],
)
def test_prepare_reorders_to_ascending_depth(depths, positive, expect_flip):
    data = field(depths=depths, positive=positive).transpose("lon", "time", "lat", "depth")
    mask = wet_mask(field(depths=depths, positive=positive))
    out, wet, z_values, z_attrs, original_dims = prepare_volumetric_data(data, mask, VDIMS)
    assert out.dims == ("time", "depth", "lat", "lon") and wet.dims == ("depth", "lat", "lon")
    np.testing.assert_array_equal(z_values, depths)
    assert z_attrs["units"] == "m"
    assert original_dims == ("lon", "time", "lat", "depth")
    internal = out.depth.values * (-1.0 if positive == "up" else 1.0)
    assert np.all(np.diff(internal) > 0)
    # The data really moved with its coordinate.
    for k in range(NZ):
        np.testing.assert_array_equal(
            out.isel(depth=k).transpose("time", "lat", "lon").values,
            data.sel(depth=out.depth.values[k]).transpose("time", "lat", "lon").values,
        )
    assert (list(out.depth.values) != list(depths)) == expect_flip


def test_prepare_rechunks_with_one_warning():
    data = field().chunk({"time": 2, "depth": 1, "lat": 2, "lon": 3})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, *_ = prepare_volumetric_data(data, wet_mask(data), VDIMS)
    rechunk = [w for w in caught if issubclass(w.category, UserWarning) and "Rechunking" in str(w.message)]
    assert len(rechunk) == 1
    assert all(len(out.chunksizes[d]) == 1 for d in ("depth", "lat", "lon"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepare_volumetric_data(field(), wet_mask(field()), VDIMS)
    assert not [w for w in caught if "Rechunking" in str(w.message)]


def _mask_with_dry_deepest_level(depths, transpose=False):
    """A wet mask, listed on the given depth labels, whose 30 m level is dry."""
    wet = np.ones((len(depths), NY, NX), dtype=bool)
    wet[list(depths).index(30.0)] = False
    mask = xr.DataArray(
        wet,
        dims=("depth", "lat", "lon"),
        coords={"depth": depths, "lat": np.linspace(-10, 10, NY), "lon": np.linspace(0, 50, NX)},
    )
    return mask.transpose("lat", "depth", "lon") if transpose else mask


MASK_ORDERS = [
    pytest.param(DEPTHS[::-1], DEPTHS, False, id="descending_data_ascending_mask"),
    pytest.param(DEPTHS[::-1], DEPTHS[::-1], False, id="both_descending"),
    pytest.param(DEPTHS, DEPTHS[::-1], False, id="ascending_data_descending_mask"),
    pytest.param(DEPTHS, DEPTHS, True, id="transposed_mask"),
    pytest.param(DEPTHS[::-1], DEPTHS, True, id="transposed_mask_descending_data"),
]


@pytest.mark.parametrize("data_depths, mask_depths, transpose", MASK_ORDERS)
def test_prepare_mask_follows_its_coordinate(data_depths, mask_depths, transpose):
    data = field(depths=data_depths)
    mask = _mask_with_dry_deepest_level(mask_depths, transpose)
    out, wet, *_ = prepare_volumetric_data(data, mask, VDIMS)
    assert wet.dims == ("depth", "lat", "lon")
    np.testing.assert_array_equal(wet.depth.values, out.depth.values)
    np.testing.assert_array_equal(out.depth.values, DEPTHS)
    # The dry level is the deepest one and stays attached to its 30 m label.
    np.testing.assert_array_equal(wet.values.any(axis=(1, 2)), [True, True, False])


def test_prepare_mask_coordinate_mismatch_raises():
    mask = _mask_with_dry_deepest_level(DEPTHS)
    mask = mask.assign_coords(depth=[5.0, 15.0, 31.0])
    with pytest.raises(ConfigurationError, match="'depth' coordinate does not match"):
        prepare_volumetric_data(field(), mask, VDIMS)


def test_tracker_geometry_wet_follows_mask_label():
    """End to end: a dry level on descending data lands on the right internal level in the geometry."""
    data = field(depths=DEPTHS[::-1])
    mask = _mask_with_dry_deepest_level(DEPTHS)
    bounds = layer_bounds().isel(depth=slice(None, None, -1))
    t = build(data, mask, layer_bounds=bounds)
    np.testing.assert_array_equal(t.geom.wet.any(axis=(1, 2)), [True, True, False])


def test_checkpoint_is_stored_with_temp_dir(tmp_path):
    t = build(checkpoint="save", temp_dir=str(tmp_path))
    assert t.checkpoint == "save"


GLORYS_DEPTHS = np.array([0.494025, 1.541375, 2.645669, 3.819495, 5.078224, 6.440614, 7.92956, 9.572997])


def _glorys_pair(data_dtype, mask_dtype, mask_scale=1.0):
    """Extremes field on `data_dtype` depths and a wet mask on `mask_dtype` depths (optionally scaled)."""
    data = field(depths=GLORYS_DEPTHS[:NZ].astype(data_dtype))
    mask = wet_mask(field(depths=GLORYS_DEPTHS[:NZ].astype(np.float64)))
    mask = mask.assign_coords(depth=(GLORYS_DEPTHS[:NZ] * mask_scale).astype(mask_dtype))
    return data, mask


def test_float32_data_depth_matches_float64_mask_depth():
    data, mask = _glorys_pair(np.float32, np.float64)
    assert not np.array_equal(data["depth"].values.astype(np.float64), mask["depth"].values)
    t = build(data, mask, layer_bounds=layer_bounds())
    assert t.mask["depth"].dtype == np.float32
    np.testing.assert_array_equal(t.mask["depth"].values, data["depth"].values)


@pytest.mark.parametrize("coord", ["depth", "lat", "lon"])
def test_float32_coordinate_rounding_is_accepted_on_every_axis(coord):
    # Coordinates that are not exactly representable in float32 (and avoid zero, where a relative bar is undefined).
    data = field(depths=GLORYS_DEPTHS[:NZ]).assign_coords(
        lat=np.linspace(-9.87654321, 10.123456789, NY), lon=np.linspace(0.1234567, 50.7654321, NX)
    )
    mask = wet_mask(data)
    data = data.assign_coords({coord: data[coord].values.astype(np.float32)})
    out = marEx.track.volumetric.config._align_mask(mask, data, VDIMS)
    assert out[coord].dtype == np.float64
    np.testing.assert_array_equal(out[coord].values, mask[coord].values)


def test_float32_data_depth_offset_beyond_rounding_raises():
    data, mask = _glorys_pair(np.float32, np.float64, mask_scale=1.0 + 1.0e-3)
    with pytest.raises(ConfigurationError, match="'depth' coordinate does not match"):
        build(data, mask, layer_bounds=layer_bounds())


def test_float64_pair_keeps_the_tight_tolerance():
    data, mask = _glorys_pair(np.float64, np.float64, mask_scale=1.0 + 1.0e-7)
    with pytest.raises(ConfigurationError, match="'depth' coordinate does not match"):
        build(data, mask, layer_bounds=layer_bounds())
