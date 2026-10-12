"""The slab kernel builds each cell's counts over its occupied bin span only, bit for bit.

``_slab_quantile_block`` hands ``_rolling_histogram_quantile`` one cell's (cycle x bins) counts.
Bins below the cell's lowest sample and above its highest carry no counts, so the kernel slices
them off (keeping one empty bin below, which the interpolation reads) and slices the bin centres
to match. These tests call the kernel directly on numpy blocks and compare it with the full-width
kernel it replaced, on the raw float32 bits (NaN sign included), over cells built to sit on each
edge of the trim: a span touching bin 0 or the top bin, a single occupied bin, no samples at all,
sparse slots, both tails, the ulp-sensitive percentiles, q at 0 and 1, and a spatial halo.
"""

import numpy as np
import pytest

from marEx.extremes import histogram as H


def _full_width_block(block, slots, n_slots, n_bins, halo, window_steps, q, q_mirror, mirror, bin_centers):
    """The kernel before trimming: every cell's counts span all ``n_bins``."""
    core_shape = tuple(block.shape[i + 1] - 2 * h for i, h in enumerate(halo))
    out = np.full(core_shape + (n_slots,), np.nan, dtype=np.float32)
    ntime = block.shape[0]
    in_cycle = slots >= 1
    slot_offset = (slots.astype(np.int64) - 1) * n_bins
    for idx in np.ndindex(*core_shape):
        window = block[(slice(None),) + tuple(slice(i, i + 2 * h + 1) for i, h in zip(idx, halo))].reshape(ntime, -1)
        valid = (window < n_bins) & in_cycle[:, None]
        keys = (slot_offset[:, None] + window)[valid]
        counts = np.bincount(keys, minlength=n_slots * n_bins).reshape(n_slots, n_bins)
        if mirror:
            out[idx] = -H._rolling_histogram_quantile(counts[:, ::-1], window_steps, q_mirror, bin_centers)
        else:
            out[idx] = H._rolling_histogram_quantile(counts, window_steps, q, bin_centers)
    return out


def _centres(precision=0.05, max_anomaly=5.0):
    edges = H._symmetric_bin_edges(precision, max_anomaly, np.float32)
    return ((edges[1:] + edges[:-1]) / 2).astype(np.float32)


def _q_mirror(q):
    return (100.0 - round(q * 100.0, 10)) / 100.0


def _cells(n_bins, ntime, rng):
    """Columns of bin indices, one per cell, covering every edge of the trim; ``n_bins`` is a NaN sample."""
    mid = n_bins // 2
    cols = []
    for _ in range(40):  # narrow spans away from both ends: the ordinary case
        centre, width = rng.integers(10, n_bins - 10), rng.integers(1, 8)
        cols.append(np.clip(np.round(rng.normal(centre, width, ntime)), 1, n_bins - 2))
    cols.append(np.clip(np.round(rng.normal(3, 4, ntime)), 0, n_bins - 1))  # touches bin 0
    cols.append(np.clip(np.round(rng.normal(n_bins - 4, 4, ntime)), 0, n_bins - 1))  # touches the top bin
    cols.append(np.where(rng.random(ntime) < 0.5, 0, n_bins - 1))  # clipped into both end bins
    cols.append(np.full(ntime, mid))  # a single occupied bin
    cols.append(np.full(ntime, 1))  # a single bin next to bin 0
    cols.append(np.full(ntime, n_bins - 1))  # a single bin at the top
    cols.append(np.full(ntime, n_bins))  # no samples: all NaN
    sparse = np.full(ntime, n_bins)
    sparse[rng.choice(ntime, 5, replace=False)] = rng.integers(20, n_bins - 20, 5)
    cols.append(sparse)  # a few samples, most slots empty
    gappy = np.clip(np.round(rng.normal(mid, 3, ntime)), 0, n_bins - 1)
    gappy[: ntime // 2] = n_bins
    cols.append(gappy)  # the first half of the record is NaN
    return np.stack(cols, axis=1).astype(np.uint16)


def _slots(ntime, n_slots, rng, with_dropped=True):
    slots = (np.arange(ntime) % n_slots + 1).astype(np.int64)
    if with_dropped:
        slots[rng.choice(ntime, 7, replace=False)] = 0  # out-of-cycle steps the kernel drops
    return slots


QS = [0.95, 0.9, 0.5, 0.07, 0.32, 0.34, 0.05, 0.0, 1.0, 1.0 - 1e-12]


def _assert_bitwise(new, ref):
    assert new.shape == ref.shape and new.dtype == ref.dtype == np.float32
    np.testing.assert_array_equal(new.view(np.uint32), ref.view(np.uint32))


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("n_slots, window_steps", [(366, 11), (12, 1), (12, 3)])
def test_trimmed_kernel_is_bitwise_the_full_width_kernel(q, mirror, n_slots, window_steps):
    rng = np.random.default_rng(7)
    centres = _centres()
    n_bins = centres.size
    ntime = 3 * n_slots + 5
    block = _cells(n_bins, ntime, rng)
    args = (_slots(ntime, n_slots, rng), n_slots, n_bins, [0], window_steps, q, _q_mirror(q), mirror, centres)
    _assert_bitwise(H._slab_quantile_block(block, *args), _full_width_block(block, *args))


@pytest.mark.parametrize("q", [0.95, 0.07, 1.0])
@pytest.mark.parametrize("mirror", [False, True])
def test_trimmed_kernel_is_bitwise_with_a_halo(q, mirror):
    # A 2-D block with a (1, 2) halo: each core cell pools a 3 x 5 window, whose union sets the span.
    # The pad index n_bins (y rows beyond the edge) is a dropped sample, as in the driver.
    rng = np.random.default_rng(11)
    centres = _centres()
    n_bins = centres.size
    n_slots, ntime = 366, 2 * 366 + 3
    cols = _cells(n_bins, ntime, rng)  # 50 cells
    block = cols[:, : 7 * 7].reshape(ntime, 7, 7)
    block[:, 0, :] = n_bins  # a padded row
    args = (_slots(ntime, n_slots, rng), n_slots, n_bins, [1, 2], 11, q, _q_mirror(q), mirror, centres)
    _assert_bitwise(H._slab_quantile_block(block, *args), _full_width_block(block, *args))


def test_trimmed_kernel_is_bitwise_on_many_random_cells():
    # Broad random spans and offsets, upper and lower, daily cycle.
    rng = np.random.default_rng(3)
    centres = _centres(0.0131818, 19.7727)  # a 3000-bin geometry derived from real ICON data
    n_bins = centres.size
    n_slots, ntime = 366, 2 * 365 + 1
    centre = rng.integers(0, n_bins, 300)
    width = rng.choice([0.5, 5, 50, 400], 300)
    block = np.clip(np.round(rng.normal(centre, width, (ntime, 300))), 0, n_bins - 1).astype(np.uint16)
    block[rng.random((ntime, 300)) < 0.1] = n_bins
    slots = _slots(ntime, n_slots, rng)
    for q, mirror in [(0.95, False), (0.9, False), (0.05, True), (0.1, True)]:
        args = (slots, n_slots, n_bins, [0], 11, q, _q_mirror(q), mirror, centres)
        _assert_bitwise(H._slab_quantile_block(block, *args), _full_width_block(block, *args))


@pytest.mark.parametrize("mirror", [False, True])
def test_kernel_hands_the_estimator_only_the_occupied_span(monkeypatch, mirror):
    # Structure, not values: a cell occupying bins 100..104 of 3000 is counted over 6 bins (one empty
    # bin below), not 3000. The bitwise tests above cannot tell a trimmed kernel from an untrimmed one.
    centres = _centres(0.0131818, 19.7727)
    n_bins = centres.size
    widths = []
    original = H._rolling_histogram_quantile

    def record(hist, *a, **k):
        widths.append(hist.shape[1])
        return original(hist, *a, **k)

    monkeypatch.setattr(H, "_rolling_histogram_quantile", record)
    ntime = 400
    block = (100 + np.arange(ntime) % 5).astype(np.uint16)[:, None]
    H._slab_quantile_block(
        block, _slots(ntime, 366, np.random.default_rng(0), False), 366, n_bins, [0], 11, 0.95, 0.05, mirror, centres
    )
    assert widths == [6]
