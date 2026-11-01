"""Partition kernels against independent oracles: wrapped distances, the legacy grid search, the EDT
partition, the unstructured specification, the BFS rewrite, and the centroid kernels' fallbacks.

No test here needs a cluster: every kernel is called directly on numpy inputs.
"""

from __future__ import annotations

from collections import deque

import numpy as np

import marEx.track as track
from marEx.track.partitioning import (
    _partition_nn_grid_gridsearch,
    partition_centroid_unstructured,
    partition_nn_unstructured,
    partition_nn_unstructured_optimised,
)

from .conftest import assert_no_failures, check

# -- wrapped distances and the legacy grid index ----------------------------------------------------


def _mask(shape, *points):
    out = np.zeros(shape, dtype=bool)
    for y, x in points:
        out[y, x] = True
    return out


def test_wrapped_distance_mask():
    """``wrapped_euclidian_distance_mask_parallel``: one row per mask pixel, one column per centroid, with
    the x axis periodic. It serves the centroid fallback of the gridded partition."""
    failures = []

    def basic():
        result = track.wrapped_euclidian_distance_mask_parallel(_mask((10, 10), (5, 5)), np.array([[3.0, 3.0]]), Nx=10, wrap=True)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], np.sqrt(8), atol=1e-6), result

    def wrap():
        # Unwrapped the distance would be 8; across the seam it is 2.
        result = track.wrapped_euclidian_distance_mask_parallel(_mask((10, 10), (5, 9)), np.array([[5.0, 1.0]]), Nx=10, wrap=True)
        assert np.isclose(result[0, 0], 2.0, atol=1e-6), result

    def two_centroids():
        result = track.wrapped_euclidian_distance_mask_parallel(
            _mask((10, 10), (2, 2), (8, 8)), np.array([[1.0, 1.0], [7.0, 7.0]]), Nx=10, wrap=True
        )
        assert result.shape == (2, 2)
        assert result[0, 0] < result[0, 1] and result[1, 1] < result[1, 0], result

    def empty():
        result = track.wrapped_euclidian_distance_mask_parallel(np.zeros((5, 5), bool), np.array([[2.0, 2.0]]), Nx=5, wrap=True)
        assert result.shape == (0, 1)

    def on_centroid():
        result = track.wrapped_euclidian_distance_mask_parallel(_mask((5, 5), (2, 2)), np.array([[2.0, 2.0]]), Nx=5, wrap=True)
        assert result.shape == (1, 1) and np.isclose(result[0, 0], 0.0, atol=1e-6), result

    for case, func in (
        ("basic", basic),
        ("wrap", wrap),
        ("two_centroids", two_centroids),
        ("empty", empty),
        ("on_centroid", on_centroid),
    ):
        check(failures, case, func)
    assert_no_failures(failures)


# (y1, x1, y2, x2, wrap, expected) on a grid 10 wide: only x wraps, and exactly half the width does not.
DISTANCE_ROWS = [
    ("basic", 3.0, 4.0, 0.0, 0.0, False, 5.0),
    ("x_wrap", 0.0, 9.0, 0.0, 1.0, True, 2.0),
    ("x_wrap_negative", 0.0, 1.0, 0.0, 9.0, True, 2.0),
    ("no_y_wrap", 0.0, 0.0, 9.0, 0.0, True, 9.0),
    ("exact_half", 0.0, 0.0, 0.0, 5.0, True, 5.0),
    ("wrap_to_one", 0, 9, 0, 0, True, 1.0),
    ("six_does_not_wrap_past_half", 0, 6, 0, 0, True, 4.0),
    ("four", 0, 4, 0, 0, True, 4.0),
    ("diagonal", 0, 0, 5, 5, True, np.sqrt(50.0)),
    ("diagonal_wrap", 3, 2, 7, 8, True, np.sqrt(32.0)),
]


def test_wrapped_distance_points():
    """``wrapped_euclidian_distance_points`` gives the exact distance for every row, both ways round."""
    failures = []
    for case, y1, x1, y2, x2, wrap, expected in DISTANCE_ROWS:

        def row(y1=y1, x1=x1, y2=y2, x2=x2, wrap=wrap, expected=expected, case=case):
            there = track.wrapped_euclidian_distance_points(y1, x1, y2, x2, nx=10, half_nx=5.0, wrap=wrap)
            back = track.wrapped_euclidian_distance_points(y2, x2, y1, x1, nx=10, half_nx=5.0, wrap=wrap)
            assert np.isclose(there, expected, atol=1e-10), f"{case}: {there} != {expected}"
            assert np.isclose(back, there, atol=1e-10), f"{case}: not symmetric, {there} one way and {back} the other"

        check(failures, case, row)
    assert_no_failures(failures)


def test_grid_index_arrays():
    """``create_grid_index_arrays`` bins points into ``grid_size`` cells, the last cell included; it serves the
    legacy grid-search oracle."""
    failures = []

    def basic():
        points, counts = track.create_grid_index_arrays(
            np.array([1, 6, 8], np.int32), np.array([2, 3, 7], np.int32), grid_size=5, ny=10, nx=10
        )
        assert points.shape == (2, 2, 3) and counts.shape == (2, 2)
        assert counts.tolist() == [[1, 0], [1, 1]], counts.tolist()

    def boundaries():
        _, counts = track.create_grid_index_arrays(
            np.array([0, 4, 5, 9], np.int32), np.array([0, 4, 5, 9], np.int32), grid_size=5, ny=10, nx=10
        )
        assert counts.tolist() == [[2, 0], [0, 2]], counts.tolist()

    def empty():
        points, counts = track.create_grid_index_arrays(np.array([], np.int32), np.array([], np.int32), grid_size=5, ny=10, nx=10)
        assert points.shape == (2, 2, 0) and counts.shape == (2, 2) and np.all(counts == 0)

    def last_cell():
        points, counts = track.create_grid_index_arrays(np.array([9], np.int32), np.array([9], np.int32), grid_size=5, ny=10, nx=10)
        assert counts[1, 1] == 1 and points[1, 1, 0] == 0, (counts.tolist(), points[1, 1].tolist())

    for case, func in (("basic", basic), ("boundaries", boundaries), ("empty", empty), ("last_cell", last_cell)):
        check(failures, case, func)
    assert_no_failures(failures)


# -- the gridded nearest-parent partition ----------------------------------------------------------


def test_partition_nn_grid_hand_cases():
    """``partition_nn_grid`` gives each child pixel the parent owning the nearest pixel, across the seam, and
    falls back to the nearest centroid beyond ``max_distance``. An empty parent mask owns no pixel."""
    failures = []

    def nearer_parent():
        result = track.partition_nn_grid(
            _mask((10, 10), (5, 5)),
            np.array([_mask((10, 10), (3, 3)), _mask((10, 10), (8, 8))]),
            np.array([100, 200], np.int32),
            np.array([[3.0, 3.0], [8.0, 8.0]]),
            Nx=10,
        )
        assert result.tolist() == [100], result.tolist()

    def seam_wrap():
        # (5, 1) is 2 away across the seam; (5, 5) is 4 away.
        result = track.partition_nn_grid(
            _mask((10, 10), (5, 9)),
            np.array([_mask((10, 10), (5, 1)), _mask((10, 10), (5, 5))]),
            np.array([100, 200], np.int32),
            np.array([[5.0, 1.0], [5.0, 5.0]]),
            Nx=10,
        )
        assert result.tolist() == [100], result.tolist()

    def empty_parent():
        # The first parent has no pixel, so its centroid at (0, 0) is never considered; the second owns (1, 1).
        result = track.partition_nn_grid(
            _mask((5, 5), (2, 2)),
            np.array([np.zeros((5, 5), bool), _mask((5, 5), (1, 1))]),
            np.array([100, 200], np.int32),
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            Nx=5,
        )
        assert result.tolist() == [200], result.tolist()

    def beyond_max_distance():
        result = track.partition_nn_grid(
            _mask((20, 20), (10, 10)),
            np.array([_mask((20, 20), (0, 0))]),
            np.array([100], np.int32),
            np.array([[0.0, 0.0]]),
            Nx=20,
            max_distance=5,
        )
        assert result.tolist() == [100], result.tolist()

    for case, func in (
        ("nearer_parent", nearer_parent),
        ("seam_wrap", seam_wrap),
        ("empty_parent", empty_parent),
        ("beyond_max_distance", beyond_max_distance),
    ):
        check(failures, case, func)
    assert_no_failures(failures)


def test_partition_nn_grid_matches_legacy_gridsearch():
    """The EDT ``partition_nn_grid`` is physically equivalent to the legacy grid search.

    Both assign each child pixel to the parent owning the nearest pixel (within max_distance, else nearest
    centroid), so over a randomized battery -- including seam-straddling cases -- the assignments must agree
    everywhere except a small set of (near-)equidistant boundary pixels, where scipy's feature-transform
    tie-break differs from the grid search's first-parent-in-index-order. A large disagreement would
    indicate a real bug, not a tie.
    """
    rng = np.random.default_rng(2024)
    ny, nx = 50, 100

    def make_blob(cy, cx, r):
        yy, xx = np.ogrid[:ny, :nx]
        dx = np.abs(xx - cx)
        dx = np.minimum(dx, nx - dx)  # wrapped longitude distance
        return ((yy - cy) ** 2 + dx**2) <= r * r

    total = 0
    diff = 0
    trials = 0
    for trial in range(60):
        n_parents = int(rng.integers(2, 6))
        # Half the trials place the child near the antimeridian seam.
        seam = trial % 2 == 0
        base_x = 0 if seam else int(rng.integers(20, 80))
        centres, parent_masks = [], []
        for _ in range(n_parents):
            cy = int(rng.integers(8, ny - 8))
            cx = int((base_x + rng.integers(-12, 13)) % nx)
            centres.append((cy, cx))
            parent_masks.append(make_blob(cy, cx, int(rng.integers(2, 5))))
        parent_masks = np.array(parent_masks)
        if not parent_masks.any(axis=(1, 2)).all():
            continue  # skip trials with an empty parent (covered by the hand cases)
        parent_centroids = np.array([[float(c[0]), float(c[1])] for c in centres])
        ccy = int(np.mean([c[0] for c in centres]))
        child_mask = make_blob(ccy, base_x, int(rng.integers(8, 14)))
        if not child_mask.any():
            continue
        child_ids = np.arange(100, 100 + n_parents, dtype=np.int32)
        md = 60
        new_edt = track.partition_nn_grid(child_mask, parent_masks, child_ids, parent_centroids, Nx=nx, max_distance=md, wrap=True)
        new_leg = _partition_nn_grid_gridsearch(child_mask, parent_masks, child_ids, parent_centroids, nx, md, True)
        assert new_edt.shape == new_leg.shape
        total += new_edt.size
        diff += int(np.sum(new_edt != new_leg))
        trials += 1

    assert trials >= 20, "Too few valid trials generated"
    frac = diff / max(total, 1)
    assert frac < 0.02, f"EDT vs grid-search disagreement {frac:.3%} exceeds 2% (only equidistant ties expected)"


# -- the unstructured nearest-parent partition against its specification ------------------------------
#
# ``partition_nn_unstructured`` is reached only through the serial merge path with unstructured_grid=True,
# which the tracker never takes, so no end-to-end test exercises it. Its rule, written here independently
# of the code: each child cell goes to the parent whose (parent AND child) seed cells are fewest graph hops
# away; ties go to the lowest parent index; cells no seed set reaches within ``max_distance`` hops fall back
# to the nearest parent centroid by great-circle distance.


def _chain_neighbours(n):
    """(3, n) neighbour table for a simple path graph, -1 padded."""
    neighbours = np.full((3, n), -1, dtype=np.int32)
    neighbours[0, 1:] = np.arange(n - 1, dtype=np.int32)  # left
    neighbours[1, :-1] = np.arange(1, n, dtype=np.int32)  # right
    return neighbours


def _random_triangular_neighbours(n, rng, extra_edges, distinct=False):
    """(3, n) symmetric neighbour table for a random connected graph of degree <= 3."""
    adjacency = [set() for _ in range(n)]
    order = rng.permutation(n)
    for i in range(1, n):  # random spanning tree keeps it connected
        a, b = int(order[i]), int(order[rng.integers(0, i)])
        if len(adjacency[a]) < 3 and len(adjacency[b]) < 3:
            adjacency[a].add(b)
            adjacency[b].add(a)
    for _ in range(extra_edges):  # extra edges, still respecting degree 3
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b and (not distinct or b not in adjacency[a]) and len(adjacency[a]) < 3 and len(adjacency[b]) < 3:
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


def test_nn_unstructured_matches_specification():
    """Random meshes, parents and children (seeds 0-11): the partitioner follows the documented rule."""
    failures = []
    for seed in range(12):

        def row(seed=seed):
            rng = np.random.default_rng(seed)
            n = 60
            neighbours = _random_triangular_neighbours(n, rng, extra_edges=n // 2)
            lat = np.linspace(-40.0, 40.0, n).astype(np.float32)
            lon = np.linspace(-80.0, 80.0, n).astype(np.float32)

            child_mask = rng.random(n) < 0.45
            n_parents = int(rng.integers(2, 4))
            parent_masks = np.zeros((n_parents, n), dtype=bool)
            for p in range(n_parents):
                parent_masks[p] = rng.random(n) < 0.2
                # guarantee at least one seed cell so the graph phase is exercised
                parent_masks[p, int(rng.choice(np.where(child_mask)[0]))] = True

            child_ids = np.arange(10, 10 + n_parents, dtype=np.int32)
            centroids = np.stack([lat[:n_parents].astype(np.float64), lon[:n_parents].astype(np.float64)], axis=1)

            got = partition_nn_unstructured(child_mask, parent_masks, child_ids, centroids, neighbours, lat, lon, 20)
            expected = _reference_partition(child_mask, parent_masks, child_ids, centroids, neighbours, lat, lon, 20)
            np.testing.assert_array_equal(got, expected, err_msg=f"seed {seed}")

        check(failures, f"seed{seed}", row)
    assert_no_failures(failures)


def test_nn_unstructured_hand_cases():
    """Hop distance decides between two parents (an exact tie to the lower index), a direct overlap tie goes
    to the lowest parent index, and a component no seed reaches falls back to the nearer centroid."""
    failures = []

    def hop_distance():
        n = 21
        parent_masks = np.zeros((2, n), dtype=bool)
        parent_masks[0, 0] = parent_masks[1, n - 1] = True
        got = partition_nn_unstructured(
            np.ones(n, bool),
            parent_masks,
            np.array([7, 9], np.int32),
            np.array([[0.0, 0.0], [0.0, 20.0]]),
            _chain_neighbours(n),
            np.zeros(n, np.float32),
            np.linspace(0.0, 20.0, n).astype(np.float32),
            30,
        )
        assert np.all(got[:10] == 7), "cells nearer the left seed must take the left parent"
        assert np.all(got[11:] == 9), "cells nearer the right seed must take the right parent"
        assert got[10] == 7, "an exact hop tie must go to the lower parent index"

    def direct_overlap_tie():
        n = 9
        got = partition_nn_unstructured(
            np.ones(n, bool),
            np.ones((2, n), bool),  # both parents cover every cell
            np.array([3, 4], np.int32),
            np.array([[0.0, 0.0], [0.0, 8.0]]),
            _chain_neighbours(n),
            np.zeros(n, np.float32),
            np.arange(n, dtype=np.float32),
            20,
        )
        np.testing.assert_array_equal(got, np.full(n, 3, dtype=np.int32))

    def unreachable_fallback():
        n = 12
        neighbours = _chain_neighbours(n)
        neighbours[1, 5] = neighbours[0, 6] = -1  # sever the chain between cells 5 and 6
        lat = np.zeros(n, np.float32)
        lon = np.concatenate([np.linspace(0.0, 5.0, 6), np.linspace(60.0, 65.0, 6)]).astype(np.float32)
        child_mask = np.ones(n, bool)
        parent_masks = np.zeros((2, n), bool)
        parent_masks[0, 0] = True  # only parent 0 has a seed inside the child
        child_ids = np.array([2, 5], np.int32)
        centroids = np.array([[0.0, 0.0], [0.0, 62.0]])  # parent 1's centroid sits on the severed component
        got = partition_nn_unstructured(child_mask, parent_masks, child_ids, centroids, neighbours, lat, lon, 20)
        expected = _reference_partition(child_mask, parent_masks, child_ids, centroids, neighbours, lat, lon, 20)
        np.testing.assert_array_equal(got, expected)
        assert np.all(got[:6] == 2), "the connected half is claimed by the seeded parent"
        assert np.all(got[6:] == 5), "the severed half falls back to the nearer centroid"

    for case, func in (
        ("hop_distance", hop_distance),
        ("direct_overlap_tie", direct_overlap_tie),
        ("unreachable", unreachable_fallback),
    ):
        check(failures, case, func)
    assert_no_failures(failures)


# -- the optimised BFS kernel against its previous semantics -----------------------------------------
#
# Two hot-path costs were removed from ``partition_nn_unstructured_optimised``, both behaviour-neutral: the
# BFS seed queue is built by a counting sort instead of one full n_points scan per parent (up to ~149 M
# iterations per merge event on the 14.9 M-cell ICON mesh), and the per-level ``np.any(child & unassigned)``
# became an incrementally maintained counter. The reference below is a TRANSCRIPTION of the previous kernel,
# not an independent oracle: it must reproduce the old tie-breaks exactly, including the parent-major seeding
# order that decides contested cells. The meshes are built to have contested cells.


def _reference_previous_semantics(child_mask, parent_frontiers, parent_centroids, neighbours, lat, lon, max_distance):
    """The kernel as it stood BEFORE the rewrite, in plain NumPy/Python."""
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
    neighbours = _random_triangular_neighbours(n_points, rng, extra_edges=n_points, distinct=True)

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


def test_optimised_matches_previous_semantics():
    """The rewritten kernel assigns every child cell exactly as the previous one did: eight random meshes at
    max_distance 30, and one mesh at max_distance 1, 2, 3 (the centroid fallback) and 100 (pure traversal)."""
    failures = []
    rows = [(f"seed{seed}", seed, 30) for seed in range(8)] + [(f"max_distance{md}", 11, md) for md in (1, 2, 3, 100)]
    for case, seed, max_distance in rows:

        def row(seed=seed, max_distance=max_distance, case=case):
            child, frontiers, centroids, neighbours, lat, lon, md = _make_case(seed, max_distance=max_distance)
            expected = _reference_previous_semantics(child, frontiers, centroids, neighbours, lat, lon, md)
            actual = partition_nn_unstructured_optimised(child, frontiers, centroids, neighbours, lat, lon, max_distance=md)
            assert actual.shape == expected.shape, f"{case}: {actual.shape} != {expected.shape}"
            np.testing.assert_array_equal(actual, expected, err_msg=case)

        check(failures, case, row)
    assert_no_failures(failures)


def test_optimised_does_not_mutate_its_inputs():
    """The call sites no longer pass defensive copies, so the kernel must not write them.

    ``neighbours_int`` is the one that mattered: copying it per merge event allocated 178 MB on the ICON
    mesh. ``child_mask`` is read after the call to place the new labels, so a mutation there would corrupt
    the relabelling directly.
    """
    child, frontiers, centroids, neighbours, lat, lon, max_distance = _make_case(3)
    before = (child.copy(), frontiers.copy(), neighbours.copy(), lat.copy(), lon.copy())

    partition_nn_unstructured_optimised(child, frontiers, centroids, neighbours, lat, lon, max_distance=max_distance)

    np.testing.assert_array_equal(child, before[0])
    np.testing.assert_array_equal(frontiers, before[1])
    np.testing.assert_array_equal(neighbours, before[2])
    np.testing.assert_array_equal(lat, before[3])
    np.testing.assert_array_equal(lon, before[4])


def test_optimised_seed_order_tiebreak():
    """A cell equidistant from two parents goes to the lower index, as before: the counting-sort seeding has
    to preserve parent-major queue order, and ascending point order within a parent."""
    n_points = 7
    neighbours = _chain_neighbours(n_points)  # path graph 0-1-2-3-4-5-6; parents at both ends, child between
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
    # Cell 3 is 3 hops from each seed: the previous code's parent-major seeding let the lower index
    # (0, seeded at cell 6) claim it first, and that must not have changed.
    assert actual[2] == 0


# -- fallbacks and the centroid kernel ---------------------------------------------------------------

MAX_PARENTS = 64
SENTINEL = -1.0e10


def _two_parent_chain(n=101):
    """A connected child spanning an equatorial chain 0.5 degrees per cell, one parent seed at each end."""
    neighbours = _chain_neighbours(n)
    lat = np.zeros(n, dtype=np.float32)
    lon = (np.arange(n) * 0.5).astype(np.float32)
    parent_masks = np.full(n, 255, dtype=np.uint8)
    parent_masks[0] = 0
    parent_masks[n - 1] = 1
    centroids = np.full((MAX_PARENTS, 2), SENTINEL, dtype=np.float32)
    centroids[0] = (lat[0], lon[0])
    centroids[1] = (lat[n - 1], lon[n - 1])
    return np.ones(n, dtype=bool), parent_masks, centroids, neighbours, lat, lon


def _global_fallback_case():
    """Every cell of a 1-degree global grid is a child cell; two real parents, one seed cell each.

    The cells are threaded on a path graph and BFS is cut off after one hop, so nearly every cell falls back
    to the nearest centroid. With the padding rows included, 24153 of the 64800 cells pick a padding row
    when all 64 rows are passed.
    """
    la, lo = np.meshgrid(np.arange(-89.5, 90.0, 1.0), np.arange(-179.5, 180.0, 1.0), indexing="ij")
    lat, lon = la.ravel().astype(np.float32), lo.ravel().astype(np.float32)
    n = lat.size
    seeds = [int(np.argmin(np.hypot(lat - 0.5, lon - 0.5))), int(np.argmin(np.hypot(lat - 0.5, lon - 90.5)))]
    parent_masks = np.full(n, 255, dtype=np.uint8)
    parent_masks[seeds] = (0, 1)
    centroids = np.full((MAX_PARENTS, 2), SENTINEL, dtype=np.float32)
    centroids[:2] = np.column_stack((lat[seeds], lon[seeds]))
    return np.ones(n, dtype=bool), parent_masks, centroids, _chain_neighbours(n), lat, lon


def test_fallback_assigns_only_real_parents():
    """Given only the real parents, both unstructured kernels return their indices or IDs and nothing else.

    On the 101-cell chain the split is exact: cells 0-50 go to the first parent (cell 50 is equidistant and
    goes to the lower index), cells 51-100 to the second. On the global fallback case the result stays in range.
    """
    child_ids = np.array([7, 9], dtype=np.int32)
    expected_split = np.r_[np.zeros(51, np.int32), np.ones(50, np.int32)]
    failures = []

    def nn_chain():
        child_mask, parent_masks, centroids, neighbours, lat, lon = _two_parent_chain()
        got = partition_nn_unstructured_optimised(child_mask, parent_masks, centroids[:2], neighbours, lat, lon, max_distance=40)
        assert got.max() < 2, got.max()
        # The midpoint splits the chain: the first half goes to parent 0, the second to parent 1.
        assert (got[:50] == 0).all() and (got[51:] == 1).all(), got.tolist()

    def centroid_chain():
        child_mask, _, centroids, _, lat, lon = _two_parent_chain()
        got = partition_centroid_unstructured(child_mask, centroids[:2], child_ids, lat, lon)
        np.testing.assert_array_equal(got, child_ids[expected_split])

    def nn_global():
        child_mask, parent_masks, centroids, neighbours, lat, lon = _global_fallback_case()
        got = partition_nn_unstructured_optimised(child_mask, parent_masks, centroids[:2], neighbours, lat, lon, max_distance=1)
        assert got.max() < 2, got.max()

    def centroid_global():
        child_mask, _, centroids, _, lat, lon = _global_fallback_case()
        got = partition_centroid_unstructured(child_mask, centroids[:2], child_ids, lat, lon)
        assert set(np.unique(got)) == {7, 9}, set(np.unique(got))

    for case, func in (
        ("nn_chain", nn_chain),
        ("centroid_chain", centroid_chain),
        ("nn_global", nn_global),
        ("centroid_global", centroid_global),
    ):
        check(failures, case, func)
    assert_no_failures(failures)


def _centroid_mesh():
    """A 10-cell mesh; the child is 4 cells, split north/south around two parent centroids."""
    n = 10
    child_mask = np.zeros(n, dtype=bool)
    child_mask[[2, 3, 7, 8]] = True
    lat = np.array([0, 0, 10, 11, 0, 0, 0, -10, -11, 0], dtype=np.float32)
    lon = np.zeros(n, dtype=np.float32)
    return child_mask, np.array([[10.5, 0.0], [-10.5, 0.0]], dtype=np.float64), np.array([101, 202], np.int32), lat, lon


def test_partition_centroid_unstructured():
    """``partition_centroid_unstructured`` returns one label per CHILD cell (it once returned one per mesh
    cell, so the caller's ``data_t[child_mask] = new_labels`` raised whenever the child was smaller than the
    mesh), each the nearest parent's ID, in the child IDs' dtype."""
    failures = []

    def one_label_per_child_cell():
        child_mask, cent, ids, lat, lon = _centroid_mesh()
        out = partition_centroid_unstructured(child_mask, cent, ids, lat, lon)
        assert len(out) == int(child_mask.sum()) == 4, len(out)
        np.testing.assert_array_equal(out, np.array([101, 101, 202, 202], dtype=np.int32))
        assert out.dtype == ids.dtype, out.dtype
        data_t = np.zeros(len(child_mask), dtype=np.int32)
        data_t[child_mask] = out  # the exact assignment in merge_split
        np.testing.assert_array_equal(data_t, [0, 0, 101, 101, 0, 0, 0, 202, 202, 0])

    def whole_mesh_is_the_child():
        n = 6
        lat = np.linspace(-20, 20, n).astype(np.float32)
        out = partition_centroid_unstructured(
            np.ones(n, bool), np.array([[20.0, 0.0], [-20.0, 0.0]]), np.array([7, 9], np.int32), lat, np.zeros(n, np.float32)
        )
        assert len(out) == n
        assert out[0] == 9 and out[-1] == 7, out.tolist()  # southmost -> south parent, northmost -> north

    def single_parent_takes_everything():
        child_mask, _, _, lat, lon = _centroid_mesh()
        out = partition_centroid_unstructured(child_mask, np.array([[0.0, 0.0]]), np.array([42], np.int32), lat, lon)
        assert set(np.unique(out)) == {42} and len(out) == int(child_mask.sum()), out.tolist()

    for case, func in (
        ("one_label_per_child_cell", one_label_per_child_cell),
        ("whole_mesh", whole_mesh_is_the_child),
        ("single_parent", single_parent_takes_everything),
    ):
        check(failures, case, func)
    assert_no_failures(failures)
