"""The unstructured merge loop must hand the partitioners only its real parents.

``split_and_merge_objects_parallel`` collects parents into fixed-width ``MAX_PARENTS`` arrays,
with unused centroid rows filled by a far-off sentinel. Both unstructured partitioners treat
every centroid row as a parent: the BFS kernel's great-circle fallback and the centroid-only
kernel take an ``argmin`` over all of them. On a sphere a sentinel row is just another point,
so it can win, and the result then indexes past ``child_ids``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dask.distributed import Client

import marEx
from marEx.track import merge_split as _merge_split
from marEx.track.partitioning import partition_centroid_unstructured, partition_nn_unstructured_optimised

TEST_DATA_DIR = Path(__file__).parent / "data"
MAX_PARENTS = 64
SENTINEL = -1.0e10


def _chain(n):
    """A path graph of ``n`` cells along the equator, 0.5 degrees apart."""
    neighbours = np.full((3, n), -1, dtype=np.int32)
    neighbours[0, 1:] = np.arange(n - 1, dtype=np.int32)
    neighbours[1, :-1] = np.arange(1, n, dtype=np.int32)
    lat = np.zeros(n, dtype=np.float32)
    lon = (np.arange(n) * 0.5).astype(np.float32)
    return neighbours, lat, lon


def _two_parent_chain(n=101):
    """A connected child spanning the chain, with one parent seed at each end."""
    neighbours, lat, lon = _chain(n)
    child_mask = np.ones(n, dtype=bool)
    parent_masks = np.full(n, 255, dtype=np.uint8)
    parent_masks[0] = 0
    parent_masks[n - 1] = 1
    centroids = np.full((MAX_PARENTS, 2), SENTINEL, dtype=np.float32)
    centroids[0] = (lat[0], lon[0])
    centroids[1] = (lat[n - 1], lon[n - 1])
    return child_mask, parent_masks, centroids, neighbours, lat, lon


def test_nn_fallback_with_real_parents_stays_in_range():
    """Cells beyond the hop cutoff fall back to the nearer of the two real centroids."""
    child_mask, parent_masks, centroids, neighbours, lat, lon = _two_parent_chain()
    got = partition_nn_unstructured_optimised(child_mask, parent_masks, centroids[:2], neighbours, lat, lon, max_distance=40)
    assert got.max() < 2
    # The midpoint splits the chain: the first half goes to parent 0, the second to parent 1.
    assert (got[:50] == 0).all() and (got[51:] == 1).all()


def test_centroid_partition_with_real_parents_returns_child_ids():
    child_mask, _, centroids, _, lat, lon = _two_parent_chain()
    child_ids = np.array([7, 9], dtype=np.int32)
    got = partition_centroid_unstructured(child_mask, centroids[:2], child_ids, lat, lon)
    assert set(np.unique(got)) <= {7, 9}


def _global_fallback_case():
    """Every cell of a 1-degree global grid is a child cell; two real parents, one seed cell each.

    The cells are threaded on a path graph and BFS is cut off after one hop, so nearly every
    cell falls back to the nearest centroid. With the padding rows included, 24153 of the
    64800 cells pick a padding row when all 64 rows are passed.
    """
    la, lo = np.meshgrid(np.arange(-89.5, 90.0, 1.0), np.arange(-179.5, 180.0, 1.0), indexing="ij")
    lat, lon = la.ravel().astype(np.float32), lo.ravel().astype(np.float32)
    n = lat.size
    neighbours, _, _ = _chain(n)
    seeds = [int(np.argmin(np.hypot(lat - 0.5, lon - 0.5))), int(np.argmin(np.hypot(lat - 0.5, lon - 90.5)))]
    parent_masks = np.full(n, 255, dtype=np.uint8)
    parent_masks[seeds] = (0, 1)
    centroids = np.full((MAX_PARENTS, 2), SENTINEL, dtype=np.float32)
    centroids[:2] = np.column_stack((lat[seeds], lon[seeds]))
    return np.ones(n, dtype=bool), parent_masks, centroids, neighbours, lat, lon


def test_nn_fallback_far_from_parents_stays_in_range():
    child_mask, parent_masks, centroids, neighbours, lat, lon = _global_fallback_case()
    got = partition_nn_unstructured_optimised(child_mask, parent_masks, centroids[:2], neighbours, lat, lon, max_distance=1)
    assert got.max() < 2


def test_centroid_partition_far_from_parents_returns_child_ids():
    child_mask, _, centroids, _, lat, lon = _global_fallback_case()
    child_ids = np.array([7, 9], dtype=np.int32)
    got = partition_centroid_unstructured(child_mask, centroids[:2], child_ids, lat, lon)
    assert set(np.unique(got)) == {7, 9}


UNSTRUCTURED_KWARGS = {
    "R_fill": 1,
    "area_filter_absolute": 5,
    "T_fill": 2,
    "allow_merging": True,
    "overlap_threshold": 0.8,
    "unstructured_grid": True,
    "dimensions": {"x": "ncells"},
    "coordinates": {"x": "lon", "y": "lat"},
    "regional_mode": False,
    "coordinate_units": "degrees",
    "quiet": True,
}


# Module-level spies: dask pickles a test-local closure by value, so its appends would land in
# a copy. Functions and lists at module level are pickled by reference and stay shared.
_CALLS: list = []


def _spy_nn(child_mask, parent_masks, parent_centroids, *args, **kwargs):
    n_parents = int(parent_masks[parent_masks < 255].max()) + 1
    _CALLS.append((len(parent_centroids), n_parents))
    return partition_nn_unstructured_optimised(child_mask, parent_masks, parent_centroids, *args, **kwargs)


def _spy_centroid(child_mask, parent_centroids, child_ids, lat, lon):
    _CALLS.append((len(parent_centroids), len(child_ids)))
    return partition_centroid_unstructured(child_mask, parent_centroids, child_ids, lat, lon)


@pytest.fixture(scope="module")
def inprocess_client():
    """Threads in this process, so the patched partitioners are the ones the merge tasks call."""
    client = Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None)
    yield client
    client.close()


@pytest.mark.parametrize("nn_partitioning", [True, False])
def test_merge_loop_passes_only_real_parent_centroids(inprocess_client, monkeypatch, tmp_path, nn_partitioning):
    _CALLS.clear()
    monkeypatch.setattr(_merge_split, "partition_nn_unstructured_optimised", _spy_nn)
    monkeypatch.setattr(_merge_split, "partition_centroid_unstructured", _spy_centroid)

    data = xr.open_zarr(str(TEST_DATA_DIR / "extremes_unstructured_merging.zarr"), chunks={})
    tr = marEx.tracker(
        data.extreme_events.chunk({"time": 4, "ncells": -1}),
        data.mask,
        nn_partitioning=nn_partitioning,
        neighbours=data.neighbours,
        cell_areas=data.cell_areas,
        temp_dir=str(tmp_path),
        **UNSTRUCTURED_KWARGS,
    )
    tr.run()

    calls = list(_CALLS)
    assert calls, "the merge loop never partitioned a child: the check is vacuous"
    wrong = [c for c in calls if c[0] != c[1]]
    assert not wrong, f"{len(wrong)} of {len(calls)} partitions got padded centroids, e.g. (rows, parents) = {wrong[0]}"
