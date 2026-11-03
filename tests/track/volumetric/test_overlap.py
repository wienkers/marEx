"""Tests for volume-weighted overlap, VolumeStore and ObjectIndex."""

import numpy as np
import pytest

from marEx.exceptions import TrackingError
from marEx.track.volumetric.overlap import link_mask, volume_overlaps
from marEx.track.volumetric.store import ObjectIndex, VolumeStore


def _store(volumes):
    s = VolumeStore()
    for k, v in volumes.items():
        s.set(k, v)
    return s


def test_analytic_volumes():
    a = np.array([1, 1], dtype=np.int32)
    b = np.array([2, 2], dtype=np.int32)
    pairs, ov = volume_overlaps(a, b, np.array([2.5, 4.0]))
    assert pairs.tolist() == [[1, 2]]
    assert ov.dtype == np.float64
    assert ov[0] == 6.5


def test_link_at_threshold_equality():
    pairs = np.array([[1, 2]], dtype=np.int64)
    store = _store({1: 2.0, 2: 4.0})
    assert link_mask(pairs, np.array([1.0]), store, 0.5).tolist() == [True]
    assert link_mask(pairs, np.array([1.0 - 2.0**-40]), store, 0.5).tolist() == [False]


def test_zero_overlap_empty():
    a = np.array([1, 0], dtype=np.int32)
    b = np.array([0, 2], dtype=np.int32)
    pairs, ov = volume_overlaps(a, b, np.ones(2))
    assert pairs.shape == (0, 2)
    assert ov.shape == (0,)
    assert pairs.dtype == np.int64 and ov.dtype == np.float64


def test_sparse_high_ids_near_int32_max():
    hi, lo = 2**31 - 2, 2**31 - 3
    a = np.array([hi, lo, hi], dtype=np.int32)
    b = np.array([lo, hi, lo], dtype=np.int32)
    pairs, ov = volume_overlaps(a, b, np.array([1.0, 2.0, 4.0]))
    assert pairs.tolist() == [[lo, hi], [hi, lo]]
    assert ov.tolist() == [2.0, 5.0]


def test_key_overflow_raises():
    big = np.array([2**31 - 2], dtype=np.int64)
    # radix 2**31 - 1 squared is below 2**62, so fabricate a wider dtype to exceed it
    huge = np.array([2**32], dtype=np.int64)
    volume_overlaps(big, big, np.ones(1))
    with pytest.raises(TrackingError):
        volume_overlaps(huge, huge, np.ones(1))


def test_pairs_sorted_lexicographically():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 6, 200).astype(np.int32)
    b = rng.integers(0, 6, 200).astype(np.int32)
    pairs, _ = volume_overlaps(a, b, np.ones(200))
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    assert (order == np.arange(len(pairs))).all()


def test_stacked_levels_equal_scaled_nz1():
    n = 4
    rng = np.random.default_rng(1)
    ncol = 30
    prev = rng.integers(0, 4, ncol).astype(np.int32)
    nxt = rng.integers(0, 4, ncol).astype(np.int32)
    area = rng.integers(1, 9, ncol).astype(np.float64)  # exact in binary
    # n identical levels, each of thickness 1
    p3, o3 = volume_overlaps(np.tile(prev, n), np.tile(nxt, n), np.tile(area, n))
    # nz=1 path with volumes scaled by n
    p1, o1 = volume_overlaps(prev, nxt, area * n)
    assert (p3 == p1).all()
    assert (o3 == o1).all()
    s3 = VolumeStore.from_slice(np.tile(prev, n), np.tile(area, n))
    s1 = VolumeStore.from_slice(prev, area * n)
    m3 = link_mask(p3, o3, s3, 0.5)
    m1 = link_mask(p1, o1, s1, 0.5)
    assert (m3 == m1).all()
    f3 = o3 / np.minimum(s3.volumes(p3[:, 0]), s3.volumes(p3[:, 1]))
    f1 = o1 / np.minimum(s1.volumes(p1[:, 0]), s1.volumes(p1[:, 1]))
    assert (f3 == f1).all()


def test_negative_control_unequal_areas():
    # object 1 and 2 each two cells; they share cell 0 (small) only
    prev = np.array([1, 1, 0], dtype=np.int32)
    nxt = np.array([2, 0, 2], dtype=np.int32)
    vol = np.array([1.0, 1.0, 2.0])
    # pixel fraction: 1 shared / min(2, 2) = 1/2 -> links at 0.5
    px_pairs, px_ov = volume_overlaps(prev, nxt, np.ones(3))
    px_store = VolumeStore.from_slice(prev, np.ones(3))
    px_store.update_from_slice(nxt, np.ones(3))
    assert link_mask(px_pairs, px_ov, px_store, 0.5).tolist() == [True]
    # volume: V1 = 2, V2 = 3, overlap 1 -> 1/2 as well; make cell 0 smaller instead
    vol = np.array([0.5, 1.5, 2.0])
    pairs, ov = volume_overlaps(prev, nxt, vol)
    store = VolumeStore.from_slice(prev, vol)
    store.update_from_slice(nxt, vol)
    assert ov.tolist() == [0.5]
    assert store.volumes(np.array([1, 2])).tolist() == [2.0, 2.5]
    assert link_mask(pairs, ov, store, 0.5).tolist() == [False]  # 0.25


def test_link_mask_missing_ids_false():
    pairs = np.array([[1, 2], [1, 9]], dtype=np.int64)
    store = _store({1: 1.0, 2: 1.0})
    assert link_mask(pairs, np.array([1.0, 1.0]), store, 0.5).tolist() == [True, False]
    assert link_mask(pairs[:0], np.zeros(0), store, 0.5).shape == (0,)


def test_store_from_slice_and_update():
    ids = np.array([0, 3, 3, 5, 0, 5, 5], dtype=np.int32)
    vol = np.array([9.0, 1.0, 2.0, 0.5, 9.0, 0.25, 0.125])
    s = VolumeStore.from_slice(ids, vol)
    assert 3 in s and 5 in s and 0 not in s and 4 not in s
    assert s.volumes(np.array([3, 5])).tolist() == [3.0, 0.875]
    assert s.contains_many(np.array([3, 4, 5])).tolist() == [True, False, True]
    s.update_from_slice(np.array([3, 7], dtype=np.int32), np.array([10.0, 2.0]))
    assert s.volumes(np.array([3, 5, 7])).tolist() == [10.0, 0.875, 2.0]
    s.drop(3)
    s.drop(3)
    assert 3 not in s


def _check(idx, flat):
    for i in range(1, 12):
        assert idx.voxels(i).tolist() == np.flatnonzero(flat == i).tolist()


def test_object_index_roundtrip():
    for seed in range(50):
        rng = np.random.default_rng(seed)
        ref = rng.integers(0, 6, (4, 5, 6)).astype(np.int32)
        flat = ref.ravel().copy()
        idx = ObjectIndex.from_slice(ref)

        _check(idx, flat)
        owned = np.flatnonzero(flat == 2)
        if owned.size:
            labels = rng.choice([0, 2, 7, 8], owned.size).astype(np.int32)
            idx.reassign(2, owned, labels)
            flat[owned] = labels
            _check(idx, flat)
        idx.rename([1, 3, 7], 4)
        flat[np.isin(flat, [1, 3, 7])] = 4
        _check(idx, flat)
        idx.rename([4, 5], 5)  # target among the sources
        flat[flat == 4] = 5
        _check(idx, flat)
        idx.rename([99], 6)  # absent source: no-op
        _check(idx, flat)
