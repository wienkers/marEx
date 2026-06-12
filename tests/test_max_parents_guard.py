"""The MAX_PARENTS guard fired on merges the fixed-width arrays could actually hold.

Job 27098021 (full ICON R02B09, nt=1096) died at global timestep ~767 with
``Child 23060 at timestep 3 has 10 parents (limit: 10)``. The guard sat at the *top* of the
candidate loop in ``process_chunk``::

    for parent_id in potential_parents[potential_parents > 0]:
        if n_parents >= MAX_PARENTS:
            raise TrackingError(...)
        ...
        if overlap_area / min_area < overlap_threshold:
            continue                       # <- candidate REJECTED, n_parents unchanged
        parent_ids[n_parents] = parent_id  # <- only here does it consume a slot

``parent_ids`` is ``np.full(MAX_PARENTS, -1)``, so indices ``0..MAX_PARENTS-1`` -- ten accepted
parents fit exactly. The top-of-loop placement therefore raised whenever MAX_PARENTS were
accepted and *any* further candidate id remained in ``potential_parents``, even when every one
of those was about to fail the overlap threshold. ``potential_parents`` is
``np.unique(data_m1[child_mask])``, which on a basin-scale child is mostly such rejects, so the
failure was also order-dependent: the identical set of parents passed if the rejects happened
to sort first.

See docs/superpowers/reports/REPORT_max_parents_diagnosis.md.
"""

import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "marEx" / "track" / "merge_split.py"


class Overflow(Exception):
    """Stands in for the TrackingError the real loop raises."""


def _scan(candidates, accepts, max_parents, *, guard_at_top):
    """Transcription of the candidate loop, with the guard at either placement.

    ``candidates`` is the ascending id list; ``accepts`` maps an id to whether it clears the
    overlap threshold. Returns the accepted-parent count, or raises Overflow.
    """
    n_parents = 0
    for parent_id in candidates:
        if guard_at_top and n_parents >= max_parents:
            raise Overflow(n_parents)
        if not accepts[parent_id]:
            continue  # below overlap_threshold: consumes no slot
        if not guard_at_top and n_parents >= max_parents:
            raise Overflow(n_parents)
        n_parents += 1
    return n_parents


class TestGuardPlacement:
    """The behavioural difference the fix is for."""

    MAX = 10

    def test_full_arrays_plus_a_reject_no_longer_raises(self):
        # 10 accepted parents, then one candidate that fails the threshold. The observed
        # full-scale failure shape.
        candidates = list(range(1, 12))
        accepts = dict.fromkeys(range(1, 11), True)
        accepts[11] = False

        with pytest.raises(Overflow):
            _scan(candidates, accepts, self.MAX, guard_at_top=True)

        assert _scan(candidates, accepts, self.MAX, guard_at_top=False) == self.MAX

    def test_a_genuine_overflow_still_raises(self):
        # 11 parents that all clear the threshold genuinely cannot be stored.
        candidates = list(range(1, 12))
        accepts = dict.fromkeys(candidates, True)

        for guard_at_top in (True, False):
            with pytest.raises(Overflow):
                _scan(candidates, accepts, self.MAX, guard_at_top=guard_at_top)

    def test_exactly_max_parents_and_nothing_else_never_raised(self):
        # The one case the old placement got right, kept so the fix is not over-claimed.
        candidates = list(range(1, 11))
        accepts = dict.fromkeys(candidates, True)

        for guard_at_top in (True, False):
            assert _scan(candidates, accepts, self.MAX, guard_at_top=guard_at_top) == self.MAX

    def test_old_placement_was_order_dependent(self):
        # Same parents, same accept/reject verdicts -- only the id order differs. np.unique
        # sorts, so which of these you got was decided by the id numbering alone.
        accepts = {1: False, **dict.fromkeys(range(2, 12), True)}
        reject_first = list(range(1, 12))
        assert _scan(reject_first, accepts, self.MAX, guard_at_top=True) == self.MAX

        accepts_reject_last = {**dict.fromkeys(range(1, 11), True), 11: False}
        with pytest.raises(Overflow):
            _scan(list(range(1, 12)), accepts_reject_last, self.MAX, guard_at_top=True)

        # The fixed placement is order-independent: both orders give the same answer.
        assert _scan(reject_first, accepts, self.MAX, guard_at_top=False) == self.MAX
        assert _scan(list(range(1, 12)), accepts_reject_last, self.MAX, guard_at_top=False) == self.MAX


class TestSourceStructure:
    """Pin the placement in the shipped source, so the transcription above stays honest."""

    @staticmethod
    def _source():
        return SOURCE.read_text()

    def test_guard_follows_the_overlap_threshold_reject(self):
        src = self._source()
        reject = src.index("if overlap_area / min_area < overlap_threshold:")
        guard = src.index("if n_parents >= MAX_PARENTS:")
        assert guard > reject, (
            "The MAX_PARENTS guard must sit after the overlap-threshold `continue`, so it only "
            "fires on parents that actually consume an array slot."
        )

    def test_guard_precedes_the_first_array_write(self):
        src = self._source()
        guard = src.index("if n_parents >= MAX_PARENTS:")
        write = src.index("parent_ids[n_parents] = parent_id")
        assert guard < write, "The guard must still run before any write at index n_parents."


class TestUpdateIdSpaceInvariant:
    """`updates_array` is uint8 with 255 as its sentinel, which bounds BOTH constants.

    Every new id minted at a timestep consumes one `updates_ids` slot, and each merge mints
    `n_parents - 1` of them. So MAX_MERGES and MAX_PARENTS are not independently tunable:
    raising either without checking this product overflows the slot table. Before the
    accompanying guard, that overflow died on a bare IndexError from an empty
    `np.where(updates_ids[t] == -1)[0][0]`, hours into a run.
    """

    @staticmethod
    def _constants():
        src = SOURCE.read_text()
        max_merges = int(re.search(r"^\s*MAX_MERGES = (\d+)", src, re.M).group(1))
        max_parents = int(re.search(r"^\s*MAX_PARENTS = (\d+)", src, re.M).group(1))
        slots = int(re.search(r"updates_ids = np\.full\(\(n_time, (\d+)\)", src).group(1))
        return max_merges, max_parents, slots

    def test_worst_case_new_ids_fit_the_slot_table(self):
        max_merges, max_parents, slots = self._constants()
        worst_case = max_merges * (max_parents - 1)
        assert worst_case <= slots, (
            f"MAX_MERGES({max_merges}) * (MAX_PARENTS-1)({max_parents - 1}) = {worst_case} "
            f"exceeds the {slots} per-timestep update slots. Raise these two together, or "
            f"widen updates_array to uint16 and updates_ids to match."
        )

    def test_max_parents_fits_the_uint8_parent_mask(self):
        _, max_parents, _ = self._constants()
        # parent_masks_uint is uint8 holding the parent index, with 255 as "unvisited".
        assert max_parents <= 255

    def test_the_slot_exhaustion_guard_exists(self):
        assert "Exhausted the per-timestep update-id space" in SOURCE.read_text()
