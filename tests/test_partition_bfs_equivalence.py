"""
Equivalence gate for the ``partition_nn_unstructured_optimised`` frontier rewrite.

Two hot-path costs were removed from that kernel, both claimed to be behaviour-neutral:

1. the BFS seed queue was built by a counting sort (two linear passes) instead of one
   full ``n_points`` scan **per parent** -- O(n_parents x n_points), up to ~149 M
   iterations per merge event on the 14.9 M-cell ICON mesh;
2. the ``np.any(child_mask & (frontiers == 255))`` recomputed at every BFS level was
   replaced by an incrementally maintained counter, removing two whole-field temporaries
   per level.

Neither may move a single assignment. This module holds a reference implementation of the
**previous** semantics -- a per-parent seeding scan and a recomputed ``np.any`` -- and
asserts the kernel agrees with it exactly, including the parent-major seeding order that
decides ties between parents reaching a cell on the same level.

A value-only comparison of the final labels would be too weak: the seeding order is
observable only through contested cells, so the meshes below are built to have them.
"""

import numpy as np
import pytest

from marEx.track.partitioning import partition_nn_unstructured_optimised


def _random_triangular_neighbours(n, rng):
    """(3, n) symmetric neighbour table for a random connected graph of degree <= 3."""
    adjacency = [set() for _ in range(n)]
    order = rng.permutation(n)
    for i in range(1, n):  # random spanning tree keeps it connected
        a, b = int(order[i]), int(order[rng.integers(0, i)])
        if len(adjacency[a]) < 3 and len(adjacency[b]) < 3:
            adjacency[a].add(b)
            adjacency[b].add(a)
    for _ in range(n):  # extra edges to create contested cells, still degree <= 3
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b and b not in adjacency[a] and len(adjacency[a]) < 3 and len(adjacency[b]) < 3:
            adjacency[a].add(b)
            adjacency[b].add(a)

    neighbours = np.full((3, n), -1, dtype=np.int32)
    for cell, nbrs in enumerate(adjacency):
        for slot, nbr in enumerate(sorted(nbrs)):
            neighbours[slot, cell] = nbr
    return neighbours


def _reference_previous_semantics(child_mask, parent_frontiers, parent_centroids, neighbours, lat, lon, max_distance):
    """The kernel as it stood BEFORE the rewrite, transcribed in plain NumPy/Python.

    Deliberately a transcription, not an independent oracle: the point here is to detect
    any drift introduced by the two optimisations, so it must reproduce the old code's
    tie-breaks exactly -- including the per-parent seeding scan.
    """
    frontiers = parent_frontiers.copy()
    child = child_mask.copy()
    n_points = len(child)
    n_parents = int(np.max(frontiers[frontiers < 255])) + 1

    # Previous seeding: a full scan of every point, once per parent.
    queue = np.empty(n_points, dtype=np.int32)
    n_queued = 0
    for parent_idx in range(n_parents):
        for point in range(n_points):
            if frontiers[point] == parent_idx:
                queue[n_queued] = point
                n_queued += 1

    current_distance = 0
    any_unassigned = bool(np.any(child & (frontiers == 255)))
    level_start, level_end = 0, n_queued

    while current_distance < max_distance and any_unassigned:
        current_distance += 1
        updates_made = False
        new_end = level_end
        for k in range(level_start, level_end):
            cell = int(queue[k])
            parent_idx = frontiers[cell]
            for i in range(3):
                neighbour = int(neighbours[i, cell])
                if neighbour < 0 or frontiers[neighbour] != 255:
                    continue
                frontiers[neighbour] = parent_idx
                queue[new_end] = neighbour
                new_end += 1
                if child[neighbour]:
                    updates_made = True
        level_start, level_end = level_end, new_end
        if not updates_made:
            break
        any_unassigned = bool(np.any(child & (frontiers == 255)))

    unassigned_mask = child & (frontiers == 255)
    if np.any(unassigned_mask):
        parent_lat_rad = np.deg2rad(parent_centroids[:, 0])
        parent_lon_rad = np.deg2rad(parent_centroids[:, 1])
        cos_parent_lat = np.cos(parent_lat_rad)
        for point in np.where(unassigned_mask)[0]:
            dlat = parent_lat_rad - np.deg2rad(lat[point])
            dlon = parent_lon_rad - np.deg2rad(lon[point])
            a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat[point])) * cos_parent_lat * np.sin(dlon / 2) ** 2
            dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            frontiers[point] = np.int32(np.argmin(dist))

    return frontiers[child].copy()


def _make_case(seed, n_points=400, n_parents=4, max_distance=30):
    rng = np.random.default_rng(seed)
    neighbours = _random_triangular_neighbours(n_points, rng)

    lat = (rng.random(n_points).astype(np.float32) * 160.0 - 80.0).astype(np.float32)
    lon = (rng.random(n_points).astype(np.float32) * 360.0 - 180.0).astype(np.float32)

    # Parent seeds: small disjoint sets; the rest of the field is unassigned (255).
    frontiers = np.full(n_points, 255, dtype=np.uint8)
    seeds = rng.choice(n_points, size=n_parents * 5, replace=False)
    for slot, cell in enumerate(seeds):
        frontiers[cell] = slot % n_parents

    # Child covers a large, contiguous-ish share so parents genuinely contest cells.
    child = np.zeros(n_points, dtype=bool)
    child[rng.choice(n_points, size=n_points // 2, replace=False)] = True
    child[frontiers < 255] = False  # children are the unassigned region being partitioned

    centroids = np.full((10, 2), -1.0e10, dtype=np.float32)
    for parent_idx in range(n_parents):
        members = np.where(frontiers == parent_idx)[0]
        centroids[parent_idx, 0] = float(lat[members].mean())
        centroids[parent_idx, 1] = float(lon[members].mean())

    return child, frontiers, centroids, neighbours, lat, lon, max_distance


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_matches_previous_semantics(seed):
    """The rewritten kernel assigns every child cell exactly as the previous one did."""
    child, frontiers, centroids, neighbours, lat, lon, max_distance = _make_case(seed)

    expected = _reference_previous_semantics(child, frontiers, centroids, neighbours, lat, lon, max_distance)
    actual = partition_nn_unstructured_optimised(child, frontiers, centroids, neighbours, lat, lon, max_distance=max_distance)

    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("max_distance", [1, 2, 3, 100])
def test_matches_previous_semantics_across_max_distance(max_distance):
    """Short walls exercise the centroid fallback; long ones exercise pure traversal."""
    child, frontiers, centroids, neighbours, lat, lon, _ = _make_case(11, max_distance=max_distance)

    expected = _reference_previous_semantics(child, frontiers, centroids, neighbours, lat, lon, max_distance)
    actual = partition_nn_unstructured_optimised(child, frontiers, centroids, neighbours, lat, lon, max_distance=max_distance)

    np.testing.assert_array_equal(actual, expected)


def test_does_not_mutate_its_inputs():
    """The call sites no longer pass defensive copies, so the kernel must not write them.

    ``neighbours_int`` is the one that mattered: copying it per merge event allocated
    178 MB on the ICON mesh. ``child_mask`` is read after the call to place the new
    labels, so a mutation there would corrupt the relabelling directly.
    """
    child, frontiers, centroids, neighbours, lat, lon, max_distance = _make_case(3)
    before = (child.copy(), frontiers.copy(), neighbours.copy(), lat.copy(), lon.copy())

    partition_nn_unstructured_optimised(child, frontiers, centroids, neighbours, lat, lon, max_distance=max_distance)

    np.testing.assert_array_equal(child, before[0])
    np.testing.assert_array_equal(frontiers, before[1])
    np.testing.assert_array_equal(neighbours, before[2])
    np.testing.assert_array_equal(lat, before[3])
    np.testing.assert_array_equal(lon, before[4])


def test_seed_order_tiebreak_favours_lower_parent_index():
    """A cell equidistant from two parents goes to the lower index, as before.

    This is what the counting-sort seeding has to preserve: parent-major queue order, and
    ascending point order within a parent.
    """
    n_points = 7
    neighbours = np.full((3, n_points), -1, dtype=np.int32)
    # Path graph 0-1-2-3-4-5-6; parents seeded at both ends, child in the middle.
    neighbours[0, 1:] = np.arange(n_points - 1, dtype=np.int32)
    neighbours[1, :-1] = np.arange(1, n_points, dtype=np.int32)

    frontiers = np.full(n_points, 255, dtype=np.uint8)
    frontiers[0] = 1  # higher index on the left
    frontiers[6] = 0  # lower index on the right
    child = np.zeros(n_points, dtype=bool)
    child[1:6] = True

    lat = np.zeros(n_points, dtype=np.float32)
    lon = np.linspace(-30, 30, n_points).astype(np.float32)
    centroids = np.full((10, 2), -1.0e10, dtype=np.float32)
    centroids[0] = (0.0, 30.0)
    centroids[1] = (0.0, -30.0)

    expected = _reference_previous_semantics(child, frontiers, centroids, neighbours, lat, lon, 10)
    actual = partition_nn_unstructured_optimised(child, frontiers, centroids, neighbours, lat, lon, max_distance=10)

    np.testing.assert_array_equal(actual, expected)
    # Cell 3 is 3 hops from each seed: the previous code's parent-major seeding let the
    # lower index (0, seeded at cell 6) claim it first, and that must not have changed.
    assert actual[2] == 0
