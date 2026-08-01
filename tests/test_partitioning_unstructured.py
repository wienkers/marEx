"""
Tests for the unstructured nearest-parent partitioners.

``partition_nn_unstructured`` is reached only through the *serial* merge path with
``unstructured_grid=True``. The tracker routes unstructured runs to the parallel path
(``tracker.py``: ``split_and_merge_objects_parallel if self.unstructured_grid``), so no
end-to-end test exercises it -- measured with coverage tripwires over the whole tracking
suite, which hit ``partition_nn_unstructured_optimised`` but never this one.

These tests pin its behaviour against an *independent* implementation of the documented
rule, not against a transcription of the code:

    Each child cell is assigned to the parent whose (parent AND child) seed cells are
    fewest graph hops away. Ties go to the lowest parent index. Cells that no seed set
    reaches within ``max_distance`` hops fall back to the nearest parent centroid by
    great-circle distance.

That gives the frontier-queue rewrite a real oracle rather than a snapshot.
"""

from collections import deque

import numpy as np
import pytest

from marEx.track.partitioning import partition_nn_unstructured


def _chain_neighbours(n):
    """(3, n) neighbour table for a simple path graph, -1 padded."""
    neighbours = np.full((3, n), -1, dtype=np.int32)
    neighbours[0, 1:] = np.arange(n - 1, dtype=np.int32)  # left
    neighbours[1, :-1] = np.arange(1, n, dtype=np.int32)  # right
    return neighbours


def _random_triangular_neighbours(n, rng):
    """(3, n) symmetric neighbour table for a random connected graph of degree <= 3."""
    adjacency = [set() for _ in range(n)]
    order = rng.permutation(n)
    for i in range(1, n):  # random spanning tree keeps it connected
        a, b = int(order[i]), int(order[rng.integers(0, i)])
        if len(adjacency[a]) < 3 and len(adjacency[b]) < 3:
            adjacency[a].add(b)
            adjacency[b].add(a)
    for _ in range(n // 2):  # a few extra edges, still respecting degree 3
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b and len(adjacency[a]) < 3 and len(adjacency[b]) < 3:
            adjacency[a].add(b)
            adjacency[b].add(a)

    neighbours = np.full((3, n), -1, dtype=np.int32)
    for cell, nbrs in enumerate(adjacency):
        for slot, nbr in enumerate(sorted(nbrs)):
            neighbours[slot, cell] = nbr
    return neighbours


def _hop_distances(sources, neighbours, n_points, max_distance):
    """Plain BFS hop distance from a set of source cells (independent of the code under test)."""
    dist = np.full(n_points, np.inf)
    queue = deque()
    for s in sources:
        dist[s] = 0
        queue.append(s)
    while queue:
        cell = queue.popleft()
        if dist[cell] >= max_distance:
            continue
        for slot in range(3):
            nbr = int(neighbours[slot, cell])
            if nbr >= 0 and dist[nbr] == np.inf:
                dist[nbr] = dist[cell] + 1
                queue.append(nbr)
    return dist


def _reference_partition(child_mask, parent_masks, child_ids, parent_centroids, neighbours, lat, lon, max_distance):
    """The documented rule, written from the specification rather than from the implementation."""
    n_points = len(child_mask)
    n_parents = len(parent_masks)

    hop = np.full((n_parents, n_points), np.inf)
    for p in range(n_parents):
        seeds = np.where(parent_masks[p] & child_mask)[0]
        if seeds.size:
            hop[p] = _hop_distances(seeds, neighbours, n_points, max_distance)

    child_points = np.where(child_mask)[0]
    labels = np.empty(child_points.size, dtype=np.int32)
    for k, cell in enumerate(child_points):
        column = hop[:, cell]
        if np.isfinite(column).any():
            labels[k] = child_ids[int(np.argmin(column))]  # argmin takes the lowest index on ties
        else:
            lat_r, lon_r = np.deg2rad(lat[cell]), np.deg2rad(lon[cell])
            plat, plon = np.deg2rad(parent_centroids[:, 0]), np.deg2rad(parent_centroids[:, 1])
            a = np.sin((plat - lat_r) / 2) ** 2 + np.cos(lat_r) * np.cos(plat) * np.sin((plon - lon_r) / 2) ** 2
            labels[k] = child_ids[int(np.argmin(2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))))]
    return labels


@pytest.mark.parametrize("seed", range(12))
def test_matches_specification_on_random_graphs(seed):
    """Random meshes, parents and children: the partitioner must follow the documented rule."""
    rng = np.random.default_rng(seed)
    n = 60
    neighbours = _random_triangular_neighbours(n, rng)
    lat = np.linspace(-40.0, 40.0, n).astype(np.float32)
    lon = np.linspace(-80.0, 80.0, n).astype(np.float32)

    child_mask = rng.random(n) < 0.45
    n_parents = int(rng.integers(2, 4))
    parent_masks = np.zeros((n_parents, n), dtype=bool)
    for p in range(n_parents):
        parent_masks[p] = rng.random(n) < 0.2
        # guarantee at least one seed cell so the graph phase is exercised
        seed_cell = int(rng.choice(np.where(child_mask)[0]))
        parent_masks[p, seed_cell] = True

    child_ids = np.arange(10, 10 + n_parents, dtype=np.int32)
    parent_centroids = np.stack([lat[:n_parents].astype(np.float64), lon[:n_parents].astype(np.float64)], axis=1)

    got = partition_nn_unstructured(child_mask, parent_masks, child_ids, parent_centroids, neighbours, lat, lon, 20)
    expected = _reference_partition(child_mask, parent_masks, child_ids, parent_centroids, neighbours, lat, lon, 20)
    np.testing.assert_array_equal(got, expected)


def test_hop_distance_decides_between_two_parents():
    """On a chain seeded at both ends, each cell goes to the nearer end."""
    n = 21
    neighbours = _chain_neighbours(n)
    lat = np.zeros(n, dtype=np.float32)
    lon = np.linspace(0.0, 20.0, n).astype(np.float32)

    child_mask = np.ones(n, dtype=bool)
    parent_masks = np.zeros((2, n), dtype=bool)
    parent_masks[0, 0] = True
    parent_masks[1, n - 1] = True
    child_ids = np.array([7, 9], dtype=np.int32)
    parent_centroids = np.array([[0.0, 0.0], [0.0, 20.0]])

    got = partition_nn_unstructured(child_mask, parent_masks, child_ids, parent_centroids, neighbours, lat, lon, 30)

    assert np.all(got[:10] == 7), "cells nearer the left seed must take the left parent"
    assert np.all(got[11:] == 9), "cells nearer the right seed must take the right parent"
    assert got[10] == 7, "an exact hop tie must go to the lower parent index"


def test_direct_overlap_ties_go_to_lowest_parent_index():
    n = 9
    neighbours = _chain_neighbours(n)
    lat = np.zeros(n, dtype=np.float32)
    lon = np.arange(n, dtype=np.float32)

    child_mask = np.ones(n, dtype=bool)
    parent_masks = np.ones((2, n), dtype=bool)  # both parents cover every cell
    child_ids = np.array([3, 4], dtype=np.int32)
    parent_centroids = np.array([[0.0, 0.0], [0.0, 8.0]])

    got = partition_nn_unstructured(child_mask, parent_masks, child_ids, parent_centroids, neighbours, lat, lon, 20)
    np.testing.assert_array_equal(got, np.full(n, 3, dtype=np.int32))


def test_unreachable_cells_fall_back_to_nearest_centroid():
    """A child component with no seed and no path must use the great-circle fallback."""
    n = 12
    neighbours = _chain_neighbours(n)
    # sever the chain between cells 5 and 6, in both directions
    neighbours[1, 5] = -1
    neighbours[0, 6] = -1

    lat = np.zeros(n, dtype=np.float32)
    lon = np.concatenate([np.linspace(0.0, 5.0, 6), np.linspace(60.0, 65.0, 6)]).astype(np.float32)

    child_mask = np.ones(n, dtype=bool)
    parent_masks = np.zeros((2, n), dtype=bool)
    parent_masks[0, 0] = True  # only parent 0 has a seed inside the child
    child_ids = np.array([2, 5], dtype=np.int32)
    # parent 1's centroid sits on top of the severed component
    parent_centroids = np.array([[0.0, 0.0], [0.0, 62.0]])

    got = partition_nn_unstructured(child_mask, parent_masks, child_ids, parent_centroids, neighbours, lat, lon, 20)
    expected = _reference_partition(child_mask, parent_masks, child_ids, parent_centroids, neighbours, lat, lon, 20)

    np.testing.assert_array_equal(got, expected)
    assert np.all(got[:6] == 2), "the connected half is claimed by the seeded parent"
    assert np.all(got[6:] == 5), "the severed half falls back to the nearer centroid"
