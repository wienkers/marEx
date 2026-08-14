"""
Bit-identity gate for the merge loop's mask-intersection rewrite.

The unstructured merge/split loop used to intersect two whole-mesh boolean masks to find
where a candidate parent overlaps the child being processed::

    if np.any(parent_mask & child_mask):
        overlap_area = area[parent_mask & child_mask].sum()

Objects are tiny next to the mesh (14.9 M cells on ICON R02B09), so it now evaluates the
parent mask *at the child's own cell indices* instead::

    overlap_cells = child_cells[parent_mask[child_cells]]
    overlap_area  = area[overlap_cells].sum()

That is O(child) rather than O(ncells), and the same substitution is used forward in time
against ``potential_child_mask``.

These tests exist because the substitution has to be **byte**-exact, not merely close.
Every one of these sums feeds a comparison against ``overlap_threshold``; a single ULP
flips a comparison, which changes which objects get queued for merging, which changes
labels, which fails the tracker's bit-identical output gate. So they assert byte equality
of float32 reductions, never ``allclose``.

The argument they encode: ``np.where(mask)[0]`` is ascending, boolean-indexing an
ascending index array preserves that order, and gathering through it yields a contiguous
array with the same length and the same elements in the same order as the boolean form --
so NumPy's pairwise summation walks an identical tree.
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def field_and_weights():
    rng = np.random.default_rng(20260814)
    n_cells = 200_000
    field = np.zeros(n_cells, dtype=np.int32)
    occupied = rng.random(n_cells) < 0.35
    field[occupied] = rng.integers(1, 400, size=int(occupied.sum()), dtype=np.int32)
    area = (rng.random(n_cells).astype(np.float32) * 50.0 + 1.0).astype(np.float32)
    return field, area


def test_overlap_cells_match_boolean_and(field_and_weights):
    """``child_cells[parent_mask[child_cells]]`` is ``np.where(parent & child)[0]``."""
    field, _ = field_and_weights
    other = np.roll(field, 31)
    checked = 0
    for child_id in np.unique(field)[1:40]:
        child_mask = field == child_id
        child_cells = np.where(child_mask)[0].astype(np.int32)
        for parent_id in np.unique(other)[1:8]:
            parent_mask = other == parent_id
            expected = np.where(parent_mask & child_mask)[0]
            actual = child_cells[parent_mask[child_cells]]
            np.testing.assert_array_equal(actual, expected)
            checked += 1
    assert checked > 0


def test_overlap_area_is_bitwise_identical(field_and_weights):
    """``area[overlap_cells].sum()`` matches ``area[parent & child].sum()`` to the byte."""
    field, area = field_and_weights
    other = np.roll(field, 31)
    differing = []
    non_empty = 0
    for child_id in np.unique(field)[1:60]:
        child_mask = field == child_id
        child_cells = np.where(child_mask)[0].astype(np.int32)
        for parent_id in np.unique(other)[1:10]:
            parent_mask = other == parent_id
            via_mask = area[parent_mask & child_mask].sum()
            via_cells = area[child_cells[parent_mask[child_cells]]].sum()
            if via_mask.tobytes() != via_cells.tobytes():
                differing.append((int(child_id), int(parent_id)))
            if (parent_mask & child_mask).any():
                non_empty += 1
    assert differing == []
    assert non_empty > 0, "fixture produced no overlaps, so the test proved nothing"


def test_any_test_agrees(field_and_weights):
    """``overlap_cells.size`` gates identically to ``np.any(parent & child)``."""
    field, _ = field_and_weights
    other = np.roll(field, 31)
    for child_id in np.unique(field)[1:40]:
        child_mask = field == child_id
        child_cells = np.where(child_mask)[0].astype(np.int32)
        for parent_id in np.unique(other)[1:8]:
            parent_mask = other == parent_id
            assert bool(child_cells[parent_mask[child_cells]].size) == bool(np.any(parent_mask & child_mask))


def test_child_area_hoist_is_bitwise_identical(field_and_weights):
    """Hoisting ``area[child_mask].sum()`` out of the parent loop cannot move it.

    It was recomputed once per candidate parent; the value is loop-invariant, but this
    pins that the hoisted single evaluation is byte-for-byte what each repeat produced.
    """
    field, area = field_and_weights
    for child_id in np.unique(field)[1:40]:
        child_mask = field == child_id
        hoisted = area[child_mask].sum()
        for _ in range(3):
            assert area[child_mask].sum().tobytes() == hoisted.tobytes()


def test_empty_overlap_sums_to_the_same_zero(field_and_weights):
    """A disjoint parent must still produce the same 0.0 the boolean form produced."""
    field, area = field_and_weights
    child_cells = np.where(field == np.unique(field)[1])[0].astype(np.int32)
    child_mask = field == np.unique(field)[1]
    parent_mask = np.zeros_like(child_mask)
    via_mask = area[parent_mask & child_mask].sum()
    via_cells = area[child_cells[parent_mask[child_cells]]].sum()
    assert via_mask.tobytes() == via_cells.tobytes()
    assert via_mask.dtype == via_cells.dtype


def test_forward_new_id_lookup_matches_field_compare():
    """``spatial_indices_all[new_labels == new_id]`` equals ``data_t == new_id``.

    The forward re-scan no longer compares the whole field to find the cells of a
    freshly-partitioned child: those cells are exactly the child's own, labelled in place.
    """
    rng = np.random.default_rng(7)
    n_cells = 50_000
    data_t = np.zeros(n_cells, dtype=np.int32)
    child_id = 42
    child_cells = np.sort(rng.choice(n_cells, size=900, replace=False)).astype(np.int32)
    data_t[child_cells] = child_id

    child_ids = np.array([child_id, 1001, 1002], dtype=np.int32)
    new_labels = child_ids[rng.integers(0, 3, size=child_cells.size)]
    data_t[child_cells] = new_labels

    for new_id in child_ids:
        expected = np.where(data_t == new_id)[0]
        actual = child_cells[new_labels == new_id]
        np.testing.assert_array_equal(actual, expected)
