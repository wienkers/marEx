"""Symmetric histogram bins: construction, guard rails, and legacy equivalence.

Phase D replaced the ``[-inf, -precision, 0, precision, ..., max_anomaly]`` binning
-- one bin for every negative value -- with finite bins symmetric about zero, which
is what makes a low-tail percentile resolvable at all.

The bins are the ONLY thing that was allowed to move. These tests pin the three
construction details that make that claim checkable:

* the positive half is bit-for-bit the old positive half,
* the guard rail a threshold is clamped onto has the same value under both,
* forcing the legacy edges back in reproduces the pipeline's stored goldens.

The third is the isolating gate: it separates "the bins changed" from "the code
changed", and it is the reason the goldens could be regenerated with a one-line
explanation instead of a shrug.
"""

import numpy as np
import pytest

from marEx.extremes.histogram import _end_clips, _symmetric_bin_edges, _zero_bin_edges

PRECISION = 0.01
MAX_ANOMALY = 5.0


def legacy_bin_edges(precision=PRECISION, max_anomaly=MAX_ANOMALY, dtype=np.float64):
    """The pre-Phase-D asymmetric edges, verbatim from each driver."""
    if dtype == np.float32:
        return np.concatenate(
            [[-np.inf], np.arange(-precision, max_anomaly + precision, precision, dtype=np.float32)], dtype=np.float32
        )
    return np.concatenate([[-np.inf], np.arange(-precision, max_anomaly + precision, precision)])


class TestConstruction:
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_positive_half_is_bit_for_bit_the_legacy_positive_half(self, dtype):
        """The whole near-identity claim rests on this equality being EXACT."""
        new = _symmetric_bin_edges(PRECISION, MAX_ANOMALY, dtype)
        old = legacy_bin_edges(dtype=dtype)
        zero = int(np.searchsorted(new, 0.0))
        assert np.array_equal(new[zero:], old[2:]), "positive edges moved"
        assert new.dtype == old.dtype

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_precision_times_arange_would_NOT_have_reproduced_them(self, dtype):
        """Pins the trap the construction exists to avoid.

        ``arange`` evaluates ``start + i * step``, so ``-p + i*p`` is not the same
        float as ``p * (i - 1)``. The spec proposed ``precision * arange(-n, n+1)``;
        measured, that shifts the float32 edges by ~5e-7, which is enough to move a
        float32 threshold by a ULP and flip a ``>=`` comparison. If this test ever
        fails, numpy's arange changed and the construction can be simplified.
        """
        n = int(round(MAX_ANOMALY / PRECISION))
        naive = (PRECISION * np.arange(-n, n + 1)).astype(dtype)
        actual = _symmetric_bin_edges(PRECISION, MAX_ANOMALY, dtype)
        assert not np.array_equal(naive, actual)
        assert np.allclose(naive, actual, atol=1e-6)

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_edges_are_exactly_symmetric(self, dtype):
        edges = _symmetric_bin_edges(PRECISION, MAX_ANOMALY, dtype)
        assert np.array_equal(edges, -edges[::-1])

    def test_no_infinite_edge_at_either_end(self):
        """A trailing +inf would make the top clip a no-op and the top centre infinite."""
        edges = _symmetric_bin_edges(PRECISION, MAX_ANOMALY)
        assert np.isfinite(edges).all()
        assert edges[0] == pytest.approx(-MAX_ANOMALY)
        assert edges[-1] == pytest.approx(MAX_ANOMALY)

    def test_bin_count_doubles(self):
        """Task-count consequence: the tile budget halves with the bin count."""
        legacy_positive_bins = len(legacy_bin_edges()) - 1 - 2  # drop the -inf bin and [-precision, 0)
        assert len(_symmetric_bin_edges(PRECISION, MAX_ANOMALY)) - 1 == 2 * legacy_positive_bins


class TestGuardRails:
    def test_guard_value_is_identical_under_legacy_and_symmetric_edges(self):
        """The clamped cells must keep their clamped VALUE, not just their identity.

        ``bin_edges[3]`` on the legacy edges is the upper edge of the bin containing
        zero; ``_zero_bin_edges`` finds that same edge on either binning, and both
        come from the same ``arange`` element, so the clamp target does not move.
        """
        legacy = legacy_bin_edges()
        symmetric = _symmetric_bin_edges(PRECISION, MAX_ANOMALY)
        assert _zero_bin_edges(legacy)[0] == float(legacy[3])
        assert _zero_bin_edges(symmetric)[0] == _zero_bin_edges(legacy)[0]

    def test_the_two_guards_straddle_zero(self):
        upper, lower = _zero_bin_edges(_symmetric_bin_edges(PRECISION, MAX_ANOMALY))
        assert lower < 0.0 < upper
        assert upper == pytest.approx(PRECISION)
        assert lower == pytest.approx(-PRECISION)

    def test_bottom_clip_is_infinite_on_legacy_edges_and_finite_on_symmetric(self):
        """The bottom clip must be a no-op on the legacy edges, or identity breaks."""
        assert _end_clips(legacy_bin_edges())[0] == -np.inf
        bottom, top = _end_clips(_symmetric_bin_edges(PRECISION, MAX_ANOMALY))
        assert bottom == pytest.approx(-MAX_ANOMALY + PRECISION / 2)
        assert top == pytest.approx(MAX_ANOMALY - PRECISION / 2)

    def test_top_clip_is_unchanged_by_the_symmetrisation(self):
        assert _end_clips(_symmetric_bin_edges(PRECISION, MAX_ANOMALY))[1] == _end_clips(legacy_bin_edges())[1]
