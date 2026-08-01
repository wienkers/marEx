"""
Tests for the absolute-threshold branch of ``filter_small_objects``.

Coverage tripwires over the whole suite showed this branch was never executed: the
existing tests construct a tracker with ``area_filter_absolute`` and then only assert the
attribute round-trips, so the quartile branch was the only one any test ran. That matters
because it is the mode the example notebooks and the full-scale runs actually use
(``area_filter_absolute=600``), and because it is the branch that fuses the area census and
the keep-mask into a single labelling pass.

The reference here re-derives the expected outputs from ``scipy.ndimage.label`` +
``np.bincount`` per slice rather than calling the function a second way, so it pins what the
filter is supposed to produce, not what it currently produces.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import label as scipy_label

from marEx.track.morphology import _EIGHT_CONNECTIVITY, _merge_lon_seam, filter_small_objects


def _make_field(n_time=3, ny=24, nx=40, seed=0):
    """Binary field with a spread of object sizes, including some crossing the seam."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n_time, ny, nx), dtype=bool)
    for t in range(n_time):
        # a handful of solid blocks of varying size
        for size in (2, 3, 5, 7):
            y = int(rng.integers(0, ny - size))
            x = int(rng.integers(0, nx - size))
            data[t, y : y + size, x : x + size] = True
        # one object straddling the antimeridian seam, so the seam merge is exercised
        data[t, 4:8, :3] = True
        data[t, 4:8, -3:] = True
    return data


def _reference(data, threshold, regional_mode):
    """Independent expectation: per-slice 8-connected labelling, pixel counts, keep >= threshold."""
    keep_field = np.zeros_like(data)
    all_areas = []
    for t in range(data.shape[0]):
        labels, n_labels = scipy_label(data[t], structure=_EIGHT_CONNECTIVITY)
        if not regional_mode:
            labels = _merge_lon_seam(labels, n_labels)
        counts = np.bincount(labels.ravel())
        areas = counts[1:][counts[1:] > 0]
        all_areas.append(areas)
        keep = counts >= threshold
        keep[0] = False
        keep_field[t] = keep[labels]
    areas_np = np.concatenate(all_areas)
    return keep_field, areas_np, int(areas_np.size), int(np.sum(areas_np >= threshold))


def _call(data, threshold, regional_mode, chunks=1):
    n_time, ny, nx = data.shape
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(n_time),
            "lat": np.linspace(-10, 10, ny),
            "lon": np.linspace(0, 359, nx),
        },
    ).chunk({"time": chunks})
    mask = xr.DataArray(np.ones((ny, nx), dtype=bool), dims=("lat", "lon"), coords={"lat": da.lat, "lon": da.lon})
    cell_area = xr.DataArray(np.ones((ny, nx), dtype=np.float32), dims=("lat", "lon"), coords={"lat": da.lat, "lon": da.lon})

    return filter_small_objects(
        data_bin=da,
        unstructured_grid=False,
        xdim="lon",
        use_absolute_filtering=True,
        area_filter_absolute=threshold,
        area_filter_quartile=0.0,
        mask=mask,
        neighbours_int=None,
        regional_mode=regional_mode,
        lat=da.lat,
        lon=da.lon,
        cell_area=cell_area,
        timedim="time",
        ydim="lat",
    )


@pytest.mark.parametrize("regional_mode", [False, True])
@pytest.mark.parametrize("threshold", [4, 9, 25])
def test_absolute_filter_matches_reference(regional_mode, threshold):
    """The single-pass absolute branch must reproduce a plain label + bincount + threshold."""
    data = _make_field()
    filtered, area_threshold, object_areas, n_unfiltered, n_filtered = _call(data, threshold, regional_mode)

    exp_field, exp_areas, exp_unfiltered, exp_filtered = _reference(data, threshold, regional_mode)

    np.testing.assert_array_equal(filtered.values, exp_field, err_msg="filtered field differs from the reference")
    np.testing.assert_array_equal(np.sort(object_areas.values), np.sort(exp_areas), err_msg="object areas differ")
    assert area_threshold == threshold
    assert n_unfiltered == exp_unfiltered
    assert n_filtered == exp_filtered


def test_absolute_filter_keeps_ties():
    """An object of exactly the threshold size is kept (>=, matching the stats convention)."""
    data = np.zeros((1, 12, 12), dtype=bool)
    data[0, 2:4, 2:4] = True  # exactly 4 cells
    data[0, 7:8, 7:8] = True  # 1 cell, must be dropped

    filtered, _, object_areas, n_unfiltered, n_filtered = _call(data, 4, regional_mode=True)

    assert filtered.values.sum() == 4, "the 4-cell object must survive an area_filter_absolute of 4"
    assert n_unfiltered == 2
    assert n_filtered == 1
    assert sorted(object_areas.values.tolist()) == [1, 4]


def test_absolute_filter_is_chunk_invariant():
    """Chunking the time axis must not change what the filter keeps."""
    data = _make_field(n_time=4, seed=3)
    one_chunk = _call(data, 9, regional_mode=False, chunks=4)
    many_chunks = _call(data, 9, regional_mode=False, chunks=1)

    np.testing.assert_array_equal(one_chunk[0].values, many_chunks[0].values)
    np.testing.assert_array_equal(np.sort(one_chunk[2].values), np.sort(many_chunks[2].values))
    assert one_chunk[3] == many_chunks[3]
    assert one_chunk[4] == many_chunks[4]
