"""Checked ID allocation and per-timestep volumetric identification."""

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import TrackingError
from marEx.track.volumetric.config import VolumetricDims
from marEx.track.volumetric.geometry import build_vertical_geometry
from marEx.track.volumetric.ids import INT32_MAX, IDAllocator
from marEx.track.volumetric.tracker4d import identify_volumes

from ..conftest import CHUNK_SIZE, GOLDEN_KWARGS, open_gridded, polar_mask
from ..conftest import on_client as _on_client

DIMS = VolumetricDims(time="time", z="depth", y="lat", x="lon")


def test_allocator_offsets_and_overflow_before_mutation():
    allocator = IDAllocator()
    offsets = allocator.reserve_offsets([3, 0, 2])
    np.testing.assert_array_equal(offsets, [0, 3, 3])
    assert offsets.dtype == np.int32 and allocator.next_id == 6

    near_full = IDAllocator(INT32_MAX - 1)
    with pytest.raises(TrackingError, match="exhausted"):
        near_full.allocate(3)
    assert near_full.next_id == INT32_MAX - 1
    with pytest.raises(TrackingError, match="exhausted"):
        near_full.reserve_offsets([1, 2])
    assert near_full.next_id == INT32_MAX - 1
    # The last valid ID is INT32_MAX itself.
    np.testing.assert_array_equal(near_full.allocate(2), [INT32_MAX - 1, INT32_MAX])
    assert near_full.next_id == INT32_MAX + 1 and near_full.next_id > INT32_MAX


def test_allocator_offsets_start_at_next_id():
    """The block base is next_id - 1, so a non-default start shifts every offset."""
    allocator = IDAllocator(10)
    np.testing.assert_array_equal(allocator.reserve_offsets([2, 3]), [9, 11])
    assert allocator.next_id == 15
    np.testing.assert_array_equal(allocator.allocate(2), [15, 16])


def _geometry(shape, mask=None):
    nz, ny, nx = shape
    wet = xr.DataArray(np.ones(shape, bool) if mask is None else mask, dims=("depth", "lat", "lon"))
    edges = np.arange(nz + 1, dtype=float) * 10.0
    bounds = xr.DataArray(np.stack([edges[:-1], edges[1:]], axis=1), dims=("depth", "bnds"))
    areas = xr.DataArray(np.full((ny, nx), 1.0e6), dims=("lat", "lon"), attrs={"units": "m2"})
    return build_vertical_geometry(
        edges[:-1] + 5.0,
        {"units": "m"},
        wet,
        zdim="depth",
        ydim="lat",
        xdim="lon",
        lat_deg=np.linspace(-5, 5, ny),
        lon_deg=np.linspace(0, 20, nx),
        layer_bounds=bounds,
        layer_thickness=None,
        cell_volumes=None,
        cell_areas=areas,
        grid_resolution=None,
    )


def _field(volume):
    return xr.DataArray(volume, dims=("time", "depth", "lat", "lon")).chunk({"time": 1})


def test_identify_ids_unique_across_time_and_levels():
    nt, nz, ny, nx = 2, 3, 4, 5
    vol = np.zeros((nt, nz, ny, nx), bool)
    vol[0, 0:2, 1, 1] = True  # object A spans levels 0 and 1
    vol[0, 2, 3, 3] = True  # object B, one level
    vol[1, 1:3, 2, 2] = True  # object C spans levels 1 and 2
    vol[1, 0, 0, 4] = True  # object D at level 0
    allocator = IDAllocator()
    ids = identify_volumes(_field(vol), _geometry((nz, ny, nx)), 0.0, False, DIMS, allocator, materialiser=None)
    out = ids.values
    assert ids.dtype == np.int32 and ids.dims == ("time", "depth", "lat", "lon")
    # One ID per object, however many levels it spans.
    assert set(np.unique(out[0, 0:2, 1, 1])) == {1}
    assert out[0, 2, 3, 3] == 2
    # Time 1 starts after time 0's largest ID (a per-(t, z) offset would reuse or skip IDs).
    assert set(np.unique(out[1, 1:3, 2, 2])) == {4}  # D is first in C order over (z, y, x)
    assert out[1, 0, 0, 4] == 3
    assert allocator.next_id == 5
    np.testing.assert_array_equal(out > 0, vol)


def test_identify_offsets_follow_allocator_state():
    vol = np.zeros((1, 2, 3, 3), bool)
    vol[0, 0, 0, 0] = True
    allocator = IDAllocator(100)
    ids = identify_volumes(_field(vol), _geometry((2, 3, 3)), 0.0, False, DIMS, allocator, materialiser=None)
    assert ids.values.max() == 100 and allocator.next_id == 101


def test_identify_overflow_leaves_allocator_unchanged():
    vol = np.zeros((1, 1, 3, 3), bool)
    vol[0, 0, 0, 0] = vol[0, 0, 2, 2] = True
    allocator = IDAllocator(INT32_MAX)
    with pytest.raises(TrackingError, match="exhausted"):
        identify_volumes(_field(vol), _geometry((1, 3, 3)), 0.0, False, DIMS, allocator, materialiser=None)
    assert allocator.next_id == INT32_MAX


def test_identify_nz1_matches_2d_on_golden_input(client):
    """With one unit-thickness level and unit areas the 4-D IDs equal the 2-D per-timestep IDs, bit for bit."""
    assert _on_client(client)
    ds = open_gridded()
    mask2d = polar_mask(ds)
    t2 = marEx.tracker(ds.extreme_events.chunk(CHUNK_SIZE), mask2d, **GOLDEN_KWARGS)
    preprocessed, _ = t2.run_preprocess()
    ids_2d, _, _ = t2.identify_objects(preprocessed, time_connectivity=False)
    ids_2d = ids_2d.compute().transpose("time", "lat", "lon").values

    data4 = preprocessed.compute().transpose("time", "lat", "lon").expand_dims(depth=[0.5], axis=1)
    data4 = data4.assign_coords(depth=("depth", [0.5], {"units": "m", "positive": "down"})).chunk({"time": 2})
    mask3 = mask2d.expand_dims(depth=[0.5], axis=0)
    bounds = xr.DataArray([[0.0, 1.0]], dims=("depth", "bnds"), coords={"depth": [0.5]})
    areas = xr.ones_like(mask2d, dtype=float)
    areas.attrs["units"] = "m2"
    t4 = marEx.tracker(
        data4,
        mask3,
        dimensions={"time": "time", "z": "depth", "y": "lat", "x": "lon"},
        layer_bounds=bounds,
        cell_areas=areas,
        **{k: v for k, v in GOLDEN_KWARGS.items() if not k.startswith("area_filter")},
    )
    ids_4d = t4.identify_volumes(t4.data_bin).compute().isel(depth=0).transpose("time", "lat", "lon").values
    assert ids_4d.dtype == ids_2d.dtype == np.int32
    assert ids_4d.max() > 0
    np.testing.assert_array_equal(ids_4d, ids_2d)
