"""
Structured object properties must not cost more for a slice whose labels are large.

The tracker's structured labels are offset by the running object count, so a late slice holds
IDs in the millions while containing only a few objects. The property step must cost what the
objects present cost, and return the same values whatever the labels are.
"""

import tracemalloc

import numpy as np
import xarray as xr

from marEx.track.objects import calculate_object_properties


def _slice(offset):
    """Three objects on a periodic 20 x 40 grid; one wraps across the x edge."""
    ids = np.zeros((20, 40), dtype=np.int32)
    ids[2:6, 3:9] = 1 + offset
    ids[10:14, 0:3] = 2 + offset  # wraps: also on the right edge
    ids[10:14, 37:40] = 2 + offset
    ids[15:18, 20:30] = 3 + offset
    return ids


def _props(field):
    return calculate_object_properties(
        field,
        unstructured_grid=False,
        lat=None,
        lon=None,
        cell_area=None,
        timedim="time",
        regional_mode=False,
        ydim="lat",
        xdim="lon",
        properties=["area", "centroid"],
    )


def test_properties_do_not_depend_on_label_values():
    small = _props(xr.DataArray(_slice(0), dims=["lat", "lon"]))
    large = _props(xr.DataArray(_slice(1_000_000), dims=["lat", "lon"]))
    assert list(large.ID.values) == [1_000_001, 1_000_002, 1_000_003]
    assert large.ID.dtype == small.ID.dtype
    xr.testing.assert_identical(small.drop_vars("ID"), large.drop_vars("ID"))


def test_properties_over_time_keep_the_original_labels():
    # Two time steps through the parallel path: the second slice carries the offset labels.
    field = np.stack([_slice(0), _slice(3)])
    props = _props(xr.DataArray(field, dims=["time", "lat", "lon"]).chunk({"time": 1}))
    assert list(props.ID.values) == [1, 2, 3, 4, 5, 6]
    np.testing.assert_array_equal(props.area.values[:3], props.area.values[3:])
    np.testing.assert_array_equal(props.centroid.values[:, :3], props.centroid.values[:, 3:])


def test_allocation_does_not_follow_label_magnitude():
    def peak(offset):
        field = xr.DataArray(_slice(offset), dims=["lat", "lon"])
        _props(field)  # first-call imports and caches are not the measurement
        tracemalloc.start()
        try:
            _props(field)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    small, large = peak(0), peak(2_000_000)
    assert large <= 2 * small + 256 * 1024, f"peak {small} B with labels near 1, {large} B with labels near 2e6"


def test_small_labels_cost_no_more_than_regionprops():
    """Compacting is not free: with labels already small it must not run at all."""
    from skimage.measure import regionprops_table

    ids = np.zeros((1200, 2400), dtype=np.int32)
    rng = np.random.default_rng(0)
    for k in range(200):
        y, x = rng.integers(0, 1180), rng.integers(0, 2380)
        ids[y : y + 15, x : x + 15] = k + 1

    def peak(fn):
        fn()
        tracemalloc.start()
        try:
            fn()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    ours = peak(lambda: _props(xr.DataArray(ids, dims=["lat", "lon"])))
    reference = peak(lambda: regionprops_table(ids, properties=["label", "area", "centroid", "bbox"]))
    # One compaction pass over this 2.9M-cell slice peaks ~17 MB above regionprops alone.
    assert ours <= reference + 4 * 1024 * 1024, f"peak {ours} B, regionprops alone {reference} B"
