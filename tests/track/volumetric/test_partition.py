"""Tests for the nearest-parent-voxel partition of a merged child."""

import numpy as np
import pytest
import xarray as xr

from marEx.exceptions import ConfigurationError, TrackingError
from marEx.track.volumetric.geometry import EARTH_RADIUS_M, build_vertical_geometry
from marEx.track.volumetric.partition import ScaledCoords, partition_child

from .oracles import _squared_distance


class TablePoints:
    """Stand-in for ScaledCoords that serves explicit 4-D points by flat index."""

    def __init__(self, table):
        self.table = np.asarray(table, dtype=np.float64)

    def points(self, flat_idx):
        return self.table[np.asarray(flat_idx, dtype=np.int64)]


def brute_nearest(child_pts, parent_pts, parent_of_pt, parent_ids):
    """
    Nearest parent per child point from the full distance matrix.

    ``parent_of_pt`` indexes ``parent_ids`` for every parent point. Exact float64
    distances; on an exact tie the smallest parent ID wins. Returns an index into
    ``parent_ids`` per child point. Does not use a spatial tree.
    """
    child_pts = np.asarray(child_pts, dtype=np.float64)
    parent_pts = np.asarray(parent_pts, dtype=np.float64)
    parent_of_pt = np.asarray(parent_of_pt)
    parent_ids = np.asarray(parent_ids)
    out = np.empty(len(child_pts), dtype=np.intp)
    for i, c in enumerate(child_pts):
        d = _squared_distance(c[None, :], parent_pts)
        tied = np.flatnonzero(d == d.min())
        ids = parent_ids[parent_of_pt[tied]]
        winner_id = ids.min()
        out[i] = int(np.flatnonzero(parent_ids == winner_id)[0])
    return out


def _ids(n, seed):
    """n distinct positive int32 parent IDs in a seeded random order."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(1, 4 * n + 1))[:n].astype(np.int32)


def _random_case(seed, nz=3, ny=12, nx=14):
    """A random grid with 2-4 parents (several voxels each) and a disjoint child."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-20.0, 20.0, ny)
    lon = np.linspace(170.0, 190.0, nx)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    midpoint = np.broadcast_to(np.array([5.0, 20.0, 60.0])[:nz, None, None], (nz, ny, nx)).copy()
    coords = ScaledCoords(lat, lon, midpoint, L_H=1.0e5, L_Z=25.0)
    n = nz * ny * nx
    perm = rng.permutation(n)
    n_parents = int(rng.integers(2, 5))
    sizes = rng.integers(1, 12, size=n_parents)
    parents, start = [], 0
    for s in sizes:
        parents.append(np.sort(perm[start : start + s]).astype(np.int64))
        start += int(s)
    child = np.sort(perm[start : start + int(rng.integers(5, 60))]).astype(np.int64)
    return coords, child, parents, _ids(n_parents, seed)


def _oracle(coords, child, parents, parent_ids):
    parent_pts = coords.points(np.concatenate(parents))
    owner = np.repeat(np.arange(len(parents)), [len(p) for p in parents])
    return brute_nearest(coords.points(child), parent_pts, owner, parent_ids)


# ---------------------------------------------------------------- ScaledCoords


def test_scaled_coords_formula_1d_axes():
    lat = np.array([-30.0, 10.0])
    lon = np.array([0.0, 90.0, 179.0])
    mid = np.arange(2 * 2 * 3, dtype=np.float64).reshape(2, 2, 3) * 7.0
    L_H, L_Z = 5.0e4, 13.0
    sc = ScaledCoords(lat, lon, mid, L_H, L_Z)
    flat = np.array([0, 4, 7, 11], dtype=np.int64)
    pts = sc.points(flat)
    assert pts.shape == (4, 4) and pts.dtype == np.float64
    z, y, x = np.unravel_index(flat, mid.shape)
    phi, lam = np.deg2rad(lat[y]), np.deg2rad(lon[x])
    np.testing.assert_allclose(pts[:, 0], EARTH_RADIUS_M * np.cos(phi) * np.cos(lam) / L_H, rtol=1e-14)
    np.testing.assert_allclose(pts[:, 1], EARTH_RADIUS_M * np.cos(phi) * np.sin(lam) / L_H, rtol=1e-14)
    np.testing.assert_allclose(pts[:, 2], EARTH_RADIUS_M * np.sin(phi) / L_H, rtol=1e-14)
    np.testing.assert_array_equal(pts[:, 3], mid[z, y, x] / L_Z)


def test_scaled_coords_accepts_2d_lat_lon():
    lat1, lon1 = np.array([0.0, 20.0]), np.array([10.0, 30.0, 50.0])
    lat2, lon2 = np.meshgrid(lat1, lon1, indexing="ij")
    mid = np.full((2, 2, 3), 40.0)
    flat = np.arange(12, dtype=np.int64)
    a = ScaledCoords(lat1, lon1, mid, 1.0e5, 10.0).points(flat)
    b = ScaledCoords(lat2, lon2, mid, 1.0e5, 10.0).points(flat)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("L_H, L_Z", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, np.nan), (np.inf, 1.0)])
def test_scaled_coords_rejects_bad_scales(L_H, L_Z):
    with pytest.raises(ConfigurationError):
        ScaledCoords(np.array([0.0]), np.array([0.0]), np.zeros((1, 1, 1)), L_H, L_Z)


def test_scaled_coords_rejects_bad_shapes():
    with pytest.raises(ConfigurationError):
        ScaledCoords(np.array([0.0, 1.0, 2.0]), np.array([0.0]), np.zeros((1, 1, 1)), 1.0, 1.0)
    with pytest.raises(ConfigurationError):
        ScaledCoords(np.array([0.0]), np.array([0.0, 1.0]), np.zeros((1, 1, 1)), 1.0, 1.0)


# ------------------------------------------------------------ nearest parent


@pytest.mark.parametrize("seed", range(50))
def test_matches_brute_force(seed):
    coords, child, parents, ids = _random_case(seed)
    got = partition_child(child, parents, ids, coords)
    assert got.dtype == np.intp and got.shape == child.shape
    np.testing.assert_array_equal(got, _oracle(coords, child, parents, ids))


def test_three_way_equidistant_tie_smallest_id():
    # Child at the origin; three parents at unit distance along different axes.
    table = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # 0: child
            [1.0, 0.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0, 0.0],  # 2
            [0.0, 0.0, -1.0, 0.0],  # 3
            [3.0, 3.0, 3.0, 3.0],  # 4: far extra voxel of the parent with the smallest ID
        ]
    )
    coords = TablePoints(table)
    child = np.array([0], dtype=np.int64)
    base = [np.array([1]), np.array([2]), np.array([3, 4])]
    ids = np.array([40, 30, 20], dtype=np.int32)
    for perm in [(0, 1, 2), (2, 1, 0), (1, 2, 0), (0, 2, 1)]:
        parents = [base[i] for i in perm]
        pid = ids[list(perm)]
        got = partition_child(child, parents, pid, coords)
        assert pid[got[0]] == 20


def test_several_seeds_one_parent():
    # A parent with many voxels, one tied with a lone rival: ID decides, not seed count.
    table = np.zeros((9, 4))
    table[1] = [1, 0, 0, 0]
    table[2] = [-1, 0, 0, 0]
    table[3] = [0, 1, 0, 0]
    table[4] = [0, -1, 0, 0]
    table[5] = [0, 0, 1, 0]
    table[6] = [0, 0, 0, 5]
    table[7] = [0, 0, 0, -5]
    table[8] = [9, 9, 9, 9]
    coords = TablePoints(table)
    child = np.array([0], dtype=np.int64)
    many = np.array([1, 2, 3, 4], dtype=np.int64)  # four tied voxels, one parent
    lone = np.array([5], dtype=np.int64)  # a fifth tied voxel, another parent
    for id_many, id_lone, expect in [(7, 3, 3), (3, 7, 3)]:
        got = partition_child(child, [many, lone], np.array([id_many, id_lone], dtype=np.int32), coords)
        assert np.array([id_many, id_lone])[got[0]] == expect
    # And several voxels of the winner at varied distances: the nearest still decides.
    far = np.array([6, 7, 8], dtype=np.int64)
    got = partition_child(child, [far, many], np.array([1, 2], dtype=np.int32), coords)
    assert got[0] == 1


@pytest.mark.parametrize("leafsize", [1, 4, 16, 64])
@pytest.mark.parametrize("batch_size", [1, 7, 65536])
def test_invariant_to_seed_order_leafsize_batch(leafsize, batch_size):
    # A coarse integer-valued lattice makes many exact ties.
    rng = np.random.default_rng(3)
    table = rng.integers(-3, 4, size=(300, 4)).astype(np.float64)
    coords = TablePoints(table)
    perm = rng.permutation(300)
    child = np.sort(perm[:80]).astype(np.int64)
    parents = [np.sort(perm[80 + 40 * i : 80 + 40 * (i + 1)]).astype(np.int64) for i in range(4)]
    ids = np.array([17, 5, 11, 2], dtype=np.int32)

    expect = ids[_oracle(coords, child, parents, ids)]
    got = ids[partition_child(child, parents, ids, coords, batch_size=batch_size, leafsize=leafsize)]
    np.testing.assert_array_equal(got, expect)

    order = rng.permutation(4)
    shuffled = [rng.permutation(parents[i]) for i in order]
    got2 = ids[order][partition_child(child, shuffled, ids[order], coords, batch_size=batch_size, leafsize=leafsize)]
    np.testing.assert_array_equal(got2, expect)


def test_ties_actually_occur_in_lattice_case():
    # Guard the tie tests: the lattice used above must contain exact ties between parents.
    rng = np.random.default_rng(3)
    table = rng.integers(-3, 4, size=(300, 4)).astype(np.float64)
    perm = rng.permutation(300)
    child = table[perm[:80]]
    parent_pts = table[perm[80:240]]
    owner = np.repeat(np.arange(4), 40)
    n_tied_parents = 0
    for c in child:
        d = _squared_distance(c[None, :], parent_pts)
        n_tied_parents += len(set(owner[d == d.min()])) > 1
    assert n_tied_parents > 0


def test_near_tie_stays_distinct():
    one = 1.0
    up = np.nextafter(one, 2.0)  # one ULP above 1
    assert _squared_distance(np.array([up, 0, 0, 0.0]), np.zeros(4)) != _squared_distance(np.array([one, 0, 0, 0.0]), np.zeros(4))
    table = np.array([[0, 0, 0, 0], [one, 0, 0, 0], [up, 0, 0, 0], [0, -one, 0, 0], [0, -up, 0, 0.0]])
    coords = TablePoints(table)
    child = np.array([0], dtype=np.int64)
    # The strictly nearer parent wins even though it has the larger ID, in both orders.
    for near_id, far_id in [(9, 2), (2, 9)]:
        for parents, ids in [
            ([np.array([1]), np.array([2])], np.array([near_id, far_id], dtype=np.int32)),
            ([np.array([2]), np.array([1])], np.array([far_id, near_id], dtype=np.int32)),
        ]:
            got = partition_child(child, parents, ids, coords)
            assert ids[got[0]] == near_id
    # Same with the two near-tied voxels on different axes (the winner is the one at distance 1).
    for parents, ids in [
        ([np.array([4]), np.array([3])], np.array([1, 8], dtype=np.int32)),
        ([np.array([3]), np.array([4])], np.array([8, 1], dtype=np.int32)),
    ]:
        got = partition_child(child, parents, ids, coords)
        assert ids[got[0]] == 8


def _adversarial_case(rng, i):
    horizontal = rng.normal(size=(8, 3))
    query = np.r_[rng.normal(size=3), rng.uniform(-2, 2)]
    depths = rng.uniform(-3, 3, size=(8, 3))
    case = i % 10
    if case in (0, 1, 2):
        lat = rng.choice([90.0, -90.0, np.nextafter(90.0, 0.0)]) if case < 2 else rng.uniform(-70, 70)
        lon = np.array([180.0, -180.0, np.nextafter(180.0, 0.0), 0.0, -0.0, -90.0, 90.0, 179.0])
        coords = ScaledCoords(np.array([lat]), lon, np.arange(1.0, 4.0), EARTH_RADIUS_M, 10.0)
        grid = coords.points(np.arange(24)).reshape(3, 8, 4)
        horizontal, depths = grid[0, :, :3].copy(), grid[:, :, 3].T.copy()
        query = grid[1, i % 8].copy()
        if case == 1:
            horizontal[:, :2] = query[:2] = 0.0  # exactly coincident polar columns
    elif case in (3, 4, 5):
        query[:] = 0.0
        horizontal[:3] = np.eye(3)
        depths[:3] = 0.0
        if case == 4:
            horizontal[0, 0] = np.nextafter(1.0, 2.0)
        if case == 5:
            depths[:3] = [0.0, 2**-30, -(2**-30)]  # rounded full distances tie
        horizontal[3:] *= 10.0
        horizontal[3:, 0] += 25.0
    elif case == 6:
        query[:] = 0.0
        horizontal *= 1e-170
        depths *= 1e-170
    elif case == 7:
        query[:3] = horizontal[0]
        horizontal[1] = horizontal[0]
        depths = np.round(depths)
        depths[0, 0] = depths[1, 1] = query[3]
    elif case == 8:
        horizontal *= 1e100
        depths *= 1e99
        query *= 1e98
    else:
        horizontal[:4] = horizontal[0]  # non-uniform partial-cell midpoints
    points = np.c_[np.repeat(horizontal, 3, axis=0), depths.ravel()]
    ids = rng.permutation(np.arange(1, 25, dtype=np.int32))
    return points, ids, query


@pytest.mark.parametrize("leafsize", [1, 2, 16])
@pytest.mark.parametrize("batch_size", [1, 7, 65536])
def test_adversarial_column_tree_matches_exhaustive(leafsize, batch_size):
    rng = np.random.default_rng(241124)
    parents = [np.array([i]) for i in range(24)]
    for case in range(500):
        seeds, ids, query = _adversarial_case(rng, case)
        queries = np.vstack([query, seeds[::3]])  # also score overlapping voxels across batch boundaries
        coords = TablePoints(np.vstack([seeds, queries]))
        child = np.arange(len(seeds), len(seeds) + len(queries))
        got = partition_child(child, parents, ids, coords, leafsize=leafsize, batch_size=batch_size)
        np.testing.assert_array_equal(got, _oracle(coords, child, parents, ids), err_msg=f"case {case}")


def test_distinct_grid_columns_with_coincident_horizontal_coordinates():
    lat = np.array([[90.0, 90.0, 90.0]])
    lon = np.array([[180.0, 180.0, -180.0]])
    midpoint = np.array([[[1.0, 1.0, 1.1]], [[2.0, 2.1, 2.2]], [[3.0, 3.1, 3.2]]])
    coords = ScaledCoords(lat, lon, midpoint, L_H=1e170, L_Z=10.0)
    child = np.array([0, 3, 4, 6, 7])
    parents = [np.array([0, 8]), np.array([1, 4]), np.array([2, 6])]
    ids = np.array([9, 3, 2], dtype=np.int32)
    got = partition_child(child, parents, ids, coords, leafsize=1)
    np.testing.assert_array_equal(got, _oracle(coords, child, parents, ids))
    assert ids[got[0]] == 3  # the overlapping seed owns a larger ID


def test_scaled_depth_rounding_changes_the_nearest_parent():
    coords = ScaledCoords(np.array([0.0]), np.array([0.0]), np.arange(1.0, 4.0), L_H=1.0, L_Z=10.0)
    parents = [np.array([0]), np.array([2])]
    ids = np.array([1, 2], dtype=np.int32)
    np.testing.assert_array_equal(partition_child([1], parents, ids, coords), [1])


def test_scaled_coords_subclass_can_change_horizontal_position_with_depth():
    class SlopingColumn(ScaledCoords):
        def points(self, flat_idx):
            points = super().points(flat_idx)
            points[flat_idx > 0, :2] = [0.0, 1.0]  # deeper voxels lie at 90° E on the equator
            return points

    coords = SlopingColumn(np.array([0.0]), np.array([0.0]), np.arange(1.0, 4.0), L_H=EARTH_RADIUS_M, L_Z=1.0)
    parents = [np.array([0]), np.array([2])]
    ids = np.array([3, 9], dtype=np.int32)
    np.testing.assert_array_equal(ids[partition_child([1], parents, ids, coords)], [9])


def test_finite_coordinates_with_squared_distance_overflow():
    max_float = np.finfo(np.float64).max
    coords = TablePoints([[max_float, 0, 0, max_float], [-max_float, 0, 0, -max_float], [0, 0, 0, 0]])
    parents = [np.array([0]), np.array([1])]
    ids = np.array([9, 3], dtype=np.int32)
    child = np.array([0, 2])
    got = partition_child(child, parents, ids, coords, leafsize=1)
    with np.errstate(over="ignore"):
        np.testing.assert_array_equal(got, _oracle(coords, child, parents, ids))
    np.testing.assert_array_equal(got, [0, 1])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("role", ["parent", "child"])
def test_non_finite_scaled_coordinates_raise(bad, role):
    table = np.zeros((3, 4))
    table[0 if role == "parent" else 2, 3] = bad
    with pytest.raises(TrackingError, match=f"Non-finite scaled coordinates among {role}"):
        partition_child([2], [[0], [1]], np.array([1, 2]), TablePoints(table))


@pytest.mark.parametrize("role", ["parent", "child"])
def test_scaled_coordinate_overflow_raises_even_with_one_parent(role):
    coords = ScaledCoords(np.array([0.0]), np.array([0.0]), np.array([0.0, 10.0]), L_H=1.0, L_Z=1e-308)
    parent, child = (1, 0) if role == "parent" else (0, 1)
    with pytest.raises(TrackingError, match=f"Non-finite scaled coordinates among {role}"):
        partition_child([child], [[parent]], np.array([1]), coords)


def test_horizontal_scale_overflow_raises():
    coords = ScaledCoords(np.array([0.0]), np.array([0.0, 1.0]), np.array([0.0, 10.0]), L_H=1e-308, L_Z=1.0)
    with pytest.raises(TrackingError, match="Non-finite scaled coordinates among parent"):
        partition_child([2], [[0], [1]], np.array([1, 2]), coords)


def test_highly_unequal_parents():
    rng = np.random.default_rng(11)
    ny, nx, nz = 20, 30, 3
    lat = np.linspace(-10.0, 10.0, ny)
    lon = np.linspace(-15.0, 15.0, nx)
    mid = np.broadcast_to(np.array([10.0, 40.0, 90.0])[:, None, None], (nz, ny, nx)).copy()
    coords = ScaledCoords(lat, lon, mid, 1.0e5, 30.0)
    perm = rng.permutation(nz * ny * nx)
    big = np.sort(perm[:400]).astype(np.int64)
    small_a = perm[400:402].astype(np.int64)
    small_b = perm[402:403].astype(np.int64)
    child = np.sort(perm[403:703]).astype(np.int64)
    ids = np.array([9, 4, 6], dtype=np.int32)
    got = partition_child(child, [big, small_a, small_b], ids, coords, batch_size=50)
    np.testing.assert_array_equal(got, _oracle(coords, child, [big, small_a, small_b], ids))


def test_giant_child_two_tiny_parents():
    nz, ny, nx = 4, 60, 80
    lat = np.linspace(-30.0, 30.0, ny)
    lon = np.linspace(-40.0, 40.0, nx)
    mid = np.broadcast_to(np.array([5.0, 25.0, 75.0, 200.0])[:, None, None], (nz, ny, nx)).copy()
    coords = ScaledCoords(lat, lon, mid, 8.0e4, 20.0)
    n = nz * ny * nx
    p1 = np.array([0], dtype=np.int64)
    p2 = np.array([n - 1], dtype=np.int64)
    child = np.arange(1, n - 1, dtype=np.int64)
    ids = np.array([5, 3], dtype=np.int32)
    got = partition_child(child, [p1, p2], ids, coords, batch_size=4096)
    np.testing.assert_array_equal(got, _oracle(coords, child, [p1, p2], ids))
    assert set(np.unique(got)) == {0, 1}


def test_single_parent_and_single_voxel():
    coords, child, parents, _ = _random_case(1)
    got = partition_child(child, [parents[0]], np.array([7], dtype=np.int32), coords)
    np.testing.assert_array_equal(got, np.zeros(len(child), dtype=np.intp))
    one = partition_child(child, [parents[0][:1]], np.array([7], dtype=np.int32), coords)
    np.testing.assert_array_equal(one, np.zeros(len(child), dtype=np.intp))


def test_empty_child_and_invalid_inputs(monkeypatch):
    coords, child, parents, ids = _random_case(2)

    def unexpected_points(flat_idx):
        pytest.fail(f"Coordinate gathering is unnecessary for an empty or invalid query: {flat_idx}")

    monkeypatch.setattr(coords, "points", unexpected_points)
    out = partition_child(np.empty(0, dtype=np.int64), parents, ids, coords)
    assert out.shape == (0,) and out.dtype == np.intp
    with pytest.raises(TrackingError):
        partition_child(child, [np.empty(0, dtype=np.int64)] * len(parents), ids, coords)
    with pytest.raises(TrackingError):
        partition_child(child, parents, ids[:-1], coords)
    with pytest.raises(ConfigurationError):
        partition_child(child, parents, ids, coords, batch_size=0)
    with pytest.raises(ConfigurationError):
        partition_child(child, parents, ids, coords, leafsize=0)


@pytest.mark.parametrize("volumes", [(1.0, 2.0, 4.0, 8.0, 16.0, 32.0), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)])
def test_partitions_disjoint_and_conserve_volume(volumes):
    coords, child, parents, ids = _random_case(5)
    child = child[:6]
    vol = dict(zip(child.tolist(), volumes))
    got = partition_child(child, parents, ids, coords)
    assert got.shape == child.shape
    assert got.min() >= 0 and got.max() < len(ids)
    pieces = [child[got == j] for j in range(len(ids))]
    # Each child voxel lands in exactly one piece, and the union is the child voxel set.
    assert sum(len(p) for p in pieces) == len(child)
    union = np.concatenate(pieces)
    assert len(np.unique(union)) == len(union)
    np.testing.assert_array_equal(np.sort(union), np.sort(child))
    if volumes[0] == 1.0:
        # Powers of two add exactly in any order, so the subtotals sum to the whole.
        total = sum(sum(vol[int(v)] for v in p) for p in pieces)
        assert total == sum(vol[int(v)] for v in child)
    # Non-binary-exact volumes: separately rounded subtotals need not equal the whole-child
    # sum (0.1+0.2+0.3 != 0.1+(0.2+0.3)), so only the voxel-set identity above is asserted.


def test_seam_chord_metric():
    lat = np.array([0.0])
    lon = np.array([179.9, -179.9, 170.0])
    mid = np.zeros((1, 1, 3))
    coords = ScaledCoords(lat, lon, mid, L_H=1.0e5, L_Z=10.0)
    child = np.array([0], dtype=np.int64)
    # Nearest by longitude difference would be the voxel at 170 (9.9 deg); the chord says -179.9 (0.2 deg).
    got = partition_child(child, [np.array([2]), np.array([1])], np.array([1, 2], dtype=np.int32), coords)
    assert got[0] == 1
    np.testing.assert_array_equal(got, _oracle(coords, child, [np.array([2]), np.array([1])], np.array([1, 2])))


def _small_geometry():
    """A 3-level, 1x3 grid at 10N with 1-degree cells, built through the real geometry builder."""
    z = np.array([50.0, 150.0, 250.0])
    bounds = xr.DataArray(np.stack([z - 50.0, z + 50.0], axis=1), dims=("depth", "nv"))
    wet = xr.DataArray(np.ones((3, 1, 3), dtype=bool), dims=("depth", "lat", "lon"))
    lat, lon = np.array([10.0]), np.array([0.0, 1.0, 2.0])
    geom = build_vertical_geometry(
        z_values=z,
        z_attrs={"units": "m"},
        wet_mask=wet,
        zdim="depth",
        ydim="lat",
        xdim="lon",
        lat_deg=lat,
        lon_deg=lon,
        layer_bounds=bounds,
        layer_thickness=None,
        cell_volumes=None,
        cell_areas=None,
        grid_resolution=1.0,
    )
    return geom, lat, lon


def test_default_scales_change_winning_parent():
    geom, lat, lon = _small_geometry()
    L_H, L_Z = geom.default_scales
    flat = lambda z, y, x: int(np.ravel_multi_index((z, y, x), (3, 1, 3)))  # noqa: E731
    child = np.array([flat(0, 0, 2)], dtype=np.int64)
    below = np.array([flat(2, 0, 2)], dtype=np.int64)  # same column, two layers deeper
    aside = np.array([flat(0, 0, 1)], dtype=np.int64)  # same level, one cell over
    parents = [below, aside]
    ids = np.array([1, 2], dtype=np.int32)

    results = []
    for l_z in (L_Z, L_Z * 100.0):
        coords = ScaledCoords(lat, lon, geom.midpoint, L_H, l_z)
        got = partition_child(child, parents, ids, coords)
        np.testing.assert_array_equal(got, _oracle(coords, child, parents, ids))
        results.append(int(ids[got[0]]))
    # With the grid-derived scales the neighbouring cell is nearer; stretching the
    # vertical scale makes the column below nearer.
    assert results == [2, 1]
