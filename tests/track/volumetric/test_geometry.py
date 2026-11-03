"""Tests for the vertical geometry of volumetric tracking."""

import logging

import numpy as np
import pytest
import xarray as xr

from marEx.exceptions import ConfigurationError
from marEx.track.volumetric.geometry import EARTH_RADIUS_M, VerticalGeometry, build_vertical_geometry

Z, Y, X = "depth", "lat", "lon"
CENTRES = np.array([5.0, 15.0, 30.0])
BOUNDS = np.array([[0.0, 10.0], [10.0, 20.0], [20.0, 40.0]])


def _wet(nz=3, ny=1, nx=1, value=True):
    return xr.DataArray(np.full((nz, ny, nx), value, dtype=bool), dims=(Z, Y, X))


def _bounds(arr=BOUNDS, bdim="nv", dtype=np.float64):
    return xr.DataArray(np.asarray(arr, dtype=dtype), dims=(Z, bdim))


def _build(z=CENTRES, attrs=None, wet=None, **kw):
    ny = kw.pop("ny", 1)
    nx = kw.pop("nx", 1)
    if wet is None:
        wet = _wet(len(z), ny, nx)
    args = dict(
        z_values=np.asarray(z, dtype=float),
        z_attrs={"units": "m"} if attrs is None else attrs,
        wet_mask=wet,
        zdim=Z,
        ydim=Y,
        xdim=X,
        lat_deg=np.linspace(0.0, 1.0, ny) if ny > 1 else np.array([10.0]),
        lon_deg=np.arange(nx, dtype=float),
        layer_bounds=_bounds(),
        layer_thickness=None,
        cell_volumes=None,
        cell_areas=xr.DataArray(np.full((ny, nx), 4.0), dims=(Y, X), attrs={"units": "m2"}),
        grid_resolution=None,
    )
    args.update(kw)
    return build_vertical_geometry(**args)


def test_bounds_1d():
    g = _build()
    assert isinstance(g, VerticalGeometry)
    np.testing.assert_array_equal(g.thickness[:, 0, 0], [10, 10, 20])
    np.testing.assert_array_equal(g.midpoint[:, 0, 0], [5, 15, 30])
    np.testing.assert_array_equal(g.top[:, 0, 0], [0, 10, 20])
    np.testing.assert_array_equal(g.bottom[:, 0, 0], [10, 20, 40])
    np.testing.assert_array_equal(g.cell_volume[:, 0, 0], [40, 40, 80])
    assert g.source == "bounds"
    assert g.flip_z is False
    assert g.thickness.dtype == np.float64 and g.cell_volume.dtype == np.float64


def test_bounds_3d_per_column():
    b = np.zeros((3, 1, 2, 2))
    b[:, 0, 0, :] = BOUNDS
    b[:, 0, 1, :] = [[0, 5], [5, 12], [12, 30]]
    lb = xr.DataArray(b, dims=(Z, Y, X, "nv"))
    g = _build(layer_bounds=lb, nx=2)
    np.testing.assert_array_equal(g.thickness[:, 0, 0], [10, 10, 20])
    np.testing.assert_array_equal(g.thickness[:, 0, 1], [5, 7, 18])
    np.testing.assert_array_equal(g.midpoint[:, 0, 1], [2.5, 8.5, 21])


def test_partial_cell_thickness_shortens_from_below():
    t = np.array([10.0, 10.0, 12.0]).reshape(3, 1, 1)
    g = _build(layer_thickness=xr.DataArray(t, dims=(Z, Y, X)))
    assert g.top[2, 0, 0] == 20
    assert g.bottom[2, 0, 0] == 32
    assert g.midpoint[2, 0, 0] == 26
    assert g.thickness[2, 0, 0] == 12
    assert g.cell_volume[2, 0, 0] == 48
    assert g.source == "bounds+thickness"


@pytest.mark.parametrize("bad", [0.0, 21.0, np.nan])
def test_thickness_outside_bounds_raises(bad):
    t = np.array([10.0, 10.0, bad]).reshape(3, 1, 1)
    with pytest.raises(ConfigurationError, match="1 wet cell"):
        _build(layer_thickness=xr.DataArray(t, dims=(Z, Y, X)))


def test_thickness_on_dry_cell_is_ignored():
    t = np.array([10.0, 10.0, 0.0]).reshape(3, 1, 1)
    wet = _wet()
    wet[2] = False
    g = _build(wet=wet, layer_thickness=xr.DataArray(t, dims=(Z, Y, X)))
    assert g.thickness[2, 0, 0] == 0.0


def test_thickness_without_bounds_raises():
    t = xr.DataArray(np.ones((3, 1, 1)), dims=(Z, Y, X))
    with pytest.raises(ConfigurationError, match="layer_bounds"):
        _build(layer_bounds=None, layer_thickness=t)


def test_from_centres_reconstruction_warns_and_records(caplog):
    lg = logging.getLogger("marEx.track.volumetric.geometry")
    lg.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="marEx.track.volumetric.geometry"):
            g = _build(layer_bounds="from_centres")
    finally:
        lg.removeHandler(caplog.handler)
    np.testing.assert_array_equal(g.top[:, 0, 0], [0, 10, 22.5])
    np.testing.assert_array_equal(g.bottom[:, 0, 0], [10, 22.5, 37.5])
    assert g.source == "reconstructed_from_centres"
    assert any(r.levelname == "WARNING" and "from_centres" in r.getMessage() for r in caplog.records)


def test_from_centres_cropped_column_uses_midpoint_rule():
    wet = _wet(ny=1, nx=2)
    wet[0, 0, 1] = False
    g = _build(layer_bounds="from_centres", wet=wet, nx=2)
    assert g.top[1, 0, 1] == 10.0
    assert g.top[0, 0, 1] == 0.0  # dry cell
    assert g.top[1, 0, 0] == 10.0


@pytest.mark.parametrize(
    "centres,expect_top,expect_bottom",
    [
        # mirrored first half-thickness: 15 - (22.5 - 15) = 7.5
        ([15.0, 30.0, 50.0], [7.5, 22.5, 40.0], [22.5, 40.0, 60.0]),
        # mirrored value would be -3.5 (above the surface): floored at 0
        ([1.0, 10.0, 30.0], [0.0, 5.5, 20.0], [5.5, 20.0, 40.0]),
    ],
)
def test_from_centres_first_top_is_mirrored_then_floored_at_zero(centres, expect_top, expect_bottom):
    g = _build(z=centres, layer_bounds="from_centres")
    np.testing.assert_array_equal(g.top[:, 0, 0], expect_top)
    np.testing.assert_array_equal(g.bottom[:, 0, 0], expect_bottom)


@pytest.mark.parametrize(
    "z,bounds",
    [
        # positive-up values with the 'positive' attribute missing
        ([-5.0, -15.0, -30.0], [[-10.0, 0.0], [-20.0, -10.0], [-40.0, -20.0]]),
        ([-30.0, -15.0, -5.0], [[-40.0, -20.0], [-20.0, -10.0], [-10.0, 0.0]]),
        # shallowest interface above the surface
        ([5.0, 15.0, 30.0], [[-1.0, 10.0], [10.0, 20.0], [20.0, 40.0]]),
    ],
)
def test_negative_depth_without_positive_attr_raises(z, bounds):
    with pytest.raises(ConfigurationError, match="positive"):
        _build(z=z, layer_bounds=_bounds(bounds))


def test_negative_depth_on_dry_cell_is_ignored():
    b = BOUNDS.copy()
    b[0] = [-5.0, 10.0]
    wet = _wet()
    wet[0] = False
    g = _build(layer_bounds=_bounds(b), wet=wet)
    assert g.thickness[0, 0, 0] == 0.0


def test_dangling_bounds_attr_raises():
    with pytest.raises(ConfigurationError, match="layer_bounds"):
        _build(attrs={"units": "m", "bounds": "depth_bnds"}, layer_bounds=None)


def test_no_geometry_raises_naming_three_options():
    with pytest.raises(ConfigurationError) as ei:
        _build(layer_bounds=None)
    msg = str(ei.value)
    assert "layer_bounds" in msg and "layer_thickness" in msg and '"from_centres"' in msg


def test_positive_up_negated_and_descending_flipped():
    ref = _build()
    up = _build(
        z=-CENTRES,
        attrs={"units": "m", "positive": "up"},
        layer_bounds=_bounds(-BOUNDS),
    )
    assert up.flip_z is False
    for name in ("top", "bottom", "thickness", "midpoint", "cell_volume"):
        np.testing.assert_array_equal(getattr(up, name), getattr(ref, name))
    # descending positive-down input: bounds and z are in the original (deep to shallow) order
    desc = _build(z=CENTRES[::-1], layer_bounds=_bounds(BOUNDS[::-1]))
    assert desc.flip_z is True
    for name in ("top", "bottom", "thickness", "midpoint", "cell_volume"):
        np.testing.assert_array_equal(getattr(desc, name), getattr(ref, name))
    # positive-up and descending in the stored order (0 at the surface rising to -30 is ascending depth)
    up_desc = _build(z=-CENTRES[::-1], attrs={"units": "m", "positive": "up"}, layer_bounds=_bounds(-BOUNDS[::-1]))
    assert up_desc.flip_z is True
    np.testing.assert_array_equal(up_desc.midpoint, ref.midpoint)


def test_descending_thickness_is_flipped_with_bounds():
    t = np.array([12.0, 10.0, 10.0]).reshape(3, 1, 1)
    g = _build(
        z=CENTRES[::-1],
        layer_bounds=_bounds(BOUNDS[::-1]),
        layer_thickness=xr.DataArray(t, dims=(Z, Y, X)),
    )
    np.testing.assert_array_equal(g.thickness[:, 0, 0], [10, 10, 12])
    assert g.bottom[2, 0, 0] == 32


def test_non_monotonic_z_raises():
    with pytest.raises(ConfigurationError, match="monotonic"):
        _build(z=[5.0, 30.0, 15.0])


@pytest.mark.parametrize("units", ["m", "metre", "metres", "meter", "meters"])
def test_metre_units_accepted(units):
    _build(attrs={"units": units})


@pytest.mark.parametrize("attrs", [{"units": "km"}, {}])
def test_non_metre_units_raise(attrs):
    with pytest.raises(ConfigurationError, match="units"):
        _build(attrs=attrs)


@pytest.mark.parametrize(
    "units,factor",
    [("m2", 1.0), ("m^2", 1.0), ("km2", 1e6)],
)
def test_cell_areas_units(units, factor):
    a = xr.DataArray(np.full((1, 1), 3.0), dims=(Y, X), attrs={"units": units})
    g = _build(cell_areas=a)
    assert g.cell_area_m2[0, 0] == 3.0 * factor
    assert g.cell_area_m2.dtype == np.float64


@pytest.mark.parametrize("attrs", [{"units": "ha"}, {}])
def test_cell_areas_bad_units_raise(attrs):
    a = xr.DataArray(np.full((1, 1), 3.0), dims=(Y, X), attrs=attrs)
    with pytest.raises(ConfigurationError, match="units"):
        _build(cell_areas=a)


def test_grid_resolution_area_float64_m2():
    lat = np.array([-30.0, 0.0, 45.0, 80.0])
    res = 0.25
    g = _build(
        wet=_wet(ny=4, nx=3),
        ny=4,
        nx=3,
        lat_deg=lat,
        cell_areas=None,
        grid_resolution=res,
    )
    lat_r = np.radians(lat)
    d = np.radians(res)
    expect = EARTH_RADIUS_M**2 * np.abs(np.sin(lat_r + d / 2) - np.sin(lat_r - d / 2)) * d
    assert g.cell_area_m2.shape == (4, 3)
    assert g.cell_area_m2.dtype == np.float64
    np.testing.assert_array_equal(g.cell_area_m2, np.broadcast_to(expect[:, None], (4, 3)))
    assert EARTH_RADIUS_M == 6_378_000.0


def test_no_horizontal_area_raises():
    with pytest.raises(ConfigurationError, match="cell_areas"):
        _build(cell_areas=None, grid_resolution=None)


def test_cell_volumes_override():
    v = xr.DataArray(np.array([1.0, 2.0, 3.0]).reshape(3, 1, 1), dims=(Z, Y, X))
    g = _build(cell_volumes=v)
    np.testing.assert_array_equal(g.cell_volume[:, 0, 0], [1, 2, 3])
    np.testing.assert_array_equal(g.thickness[:, 0, 0], [10, 10, 20])
    with pytest.raises(ConfigurationError, match="layer_bounds"):
        _build(cell_volumes=v, layer_bounds=None)
    bad = xr.DataArray(np.array([1.0, -2.0, 3.0]).reshape(3, 1, 1), dims=(Z, Y, X))
    with pytest.raises(ConfigurationError, match="cell_volumes"):
        _build(cell_volumes=bad)


def test_dry_cells_zero_volume_and_ignored():
    wet = _wet(ny=1, nx=2)
    wet[1:, 0, 1] = False
    b = np.zeros((3, 1, 2, 2))
    b[:, 0, 0, :] = BOUNDS
    b[0, 0, 1, :] = [0, 10]
    b[1:, 0, 1, :] = np.nan
    g = _build(layer_bounds=xr.DataArray(b, dims=(Z, Y, X, "nv")), wet=wet, nx=2)
    assert np.all(g.cell_volume[1:, 0, 1] == 0.0)
    assert np.all(g.thickness[1:, 0, 1] == 0.0)
    assert np.all(g.midpoint[1:, 0, 1] == 0.0)
    assert np.isfinite(g.top).all() and np.isfinite(g.bottom).all()
    assert g.cell_volume[0, 0, 1] == 40.0
    np.testing.assert_array_equal(g.wet, wet.values)


def test_wet_cell_with_nan_bounds_raises():
    b = BOUNDS.copy()
    b[1] = np.nan
    with pytest.raises(ConfigurationError, match="1 wet cell"):
        _build(layer_bounds=_bounds(b))


def test_no_wet_cells_raises():
    with pytest.raises(ConfigurationError, match="wet"):
        _build(wet=_wet(value=False))


def test_default_scales():
    # column A: sqrt(area) = 1, one wet level; column B: sqrt(area) = 3, three wet levels.
    # Per-wet-cell values are [1, 3, 3, 3] -> median 3 (a per-column median would give 2).
    wet = _wet(ny=1, nx=2)
    wet[1:, 0, 0] = False
    area = xr.DataArray(np.array([[1.0, 9.0]]), dims=(Y, X), attrs={"units": "m2"})
    g = _build(wet=wet, nx=2, cell_areas=area)
    assert g.default_scales[0] == 3.0
    # thickness over wet cells: A -> [10]; B -> [10, 10, 20]; median of [10, 10, 10, 20] = 10
    assert g.default_scales[1] == 10.0
    # column A keeps one wet level, column B two: per-wet-cell values [1, 3, 3]
    wet2 = _wet(ny=1, nx=2)
    wet2[2, 0, 1] = False
    wet2[1:, 0, 0] = False
    g2 = _build(wet=wet2, nx=2, cell_areas=area)
    assert g2.default_scales[0] == np.median([1.0, 3.0, 3.0])
    assert all(isinstance(s, float) for s in g.default_scales)


def test_default_scales_even_count_matches_numpy_median():
    rng = np.random.default_rng(3)
    ny, nx = 4, 5
    wet = _wet(ny=ny, nx=nx, value=False)
    wet.values[:] = rng.random((3, ny, nx)) < 0.6
    wet.values[0, 0, 0] = True
    area_np = rng.uniform(1.0, 100.0, (ny, nx))
    area = xr.DataArray(area_np, dims=(Y, X), attrs={"units": "m2"})
    g = _build(wet=wet, ny=ny, nx=nx, cell_areas=area)
    w = wet.values
    expect = np.median(np.broadcast_to(np.sqrt(area_np), w.shape)[w])
    assert g.default_scales[0] == expect


@pytest.mark.parametrize("bdim", ["nv", "bnds", "d2"])
def test_bounds_dim_name_and_dtype_agnostic(bdim):
    ref = _build()
    g = _build(
        z=-CENTRES,
        attrs={"units": "m", "positive": "up"},
        layer_bounds=_bounds(-BOUNDS, bdim=bdim, dtype=np.float32),
    )
    for name in ("top", "bottom", "thickness", "midpoint", "cell_volume"):
        assert getattr(g, name).dtype == np.float64
        np.testing.assert_array_equal(getattr(g, name), getattr(ref, name))


def test_bounds_per_column_with_size_two_spatial_dims():
    b = np.broadcast_to(BOUNDS[:, None, None, :], (3, 2, 2, 2)).copy()
    lb = xr.DataArray(b, dims=(Z, Y, X, "bnds"))
    g = _build(layer_bounds=lb, ny=2, nx=2)
    np.testing.assert_array_equal(g.thickness[:, 1, 1], [10, 10, 20])


def test_bounds_extra_dim_raises():
    lb = xr.DataArray(np.zeros((3, 1, 2, 2)), dims=(Z, "ens", "nv", "nb"))
    with pytest.raises(ConfigurationError, match="layer_bounds"):
        _build(layer_bounds=lb)


def test_bounds_wrong_size_raises():
    lb = xr.DataArray(np.zeros((3, 3)), dims=(Z, "nv"))
    with pytest.raises(ConfigurationError, match="layer_bounds"):
        _build(layer_bounds=lb)
