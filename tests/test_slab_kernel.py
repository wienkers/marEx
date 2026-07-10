"""The slab kernel builds each cell's counts over its occupied bin span only, bit for bit.

``_slab_quantile_block`` hands ``_rolling_histogram_quantile`` one cell's (cycle x bins) counts.
Bins below the cell's lowest sample and above its highest carry no counts, so the kernel slices
them off (keeping one empty bin below, which the interpolation reads) and slices the bin centres
to match. These tests call the kernel directly on numpy blocks and compare it with the full-width
kernel it replaced, on the raw float32 bits (NaN sign included), over cells built to sit on each
edge of the trim: a span touching bin 0 or the top bin, a single occupied bin, no samples at all,
sparse slots, both tails, the ulp-sensitive percentiles, q at 0 and 1, and a spatial halo.
"""

import time
import tracemalloc

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
def test_kernel_counts_only_the_occupied_span(mirror):
    # Structure, not values: a cell occupying bins 100..104 of 3000 is counted over 6 bins (one empty
    # bin below), not 3000. The bitwise tests above cannot tell a trimmed kernel from an untrimmed one.
    centres = _centres(0.0131818, 19.7727)
    n_bins = centres.size
    ntime = 400
    block = (100 + np.arange(ntime) % 5).astype(np.uint16)[:, None]
    _, widths = H._slab_quantile_cells(
        block, _slots(ntime, 366, np.random.default_rng(0), False), 366, n_bins, [0], 11, 0.95, 0.05, mirror, centres
    )
    assert widths.tolist() == [6]


def _peak_bytes(core, halo, n_bins=20, ntime=2):
    block = np.zeros((ntime, core + 2 * halo, core + 2 * halo), dtype=np.uint16)
    centres = np.arange(n_bins, dtype=np.float32)
    slots = np.arange(1, ntime + 1, dtype=np.int64)
    tracemalloc.start()
    H._slab_quantile_cells(block, slots, ntime, n_bins, [halo, halo], 1, 0.95, 0.05, False, centres)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def test_pooling_metadata_does_not_scale_with_cells_times_window():
    # Structure: the pooled positions are cells + window integers, not a cells x window table. At a
    # 51-wide window, quadrupling the core cells quadrupled a per-cell neighbourhood table (~21 MB
    # at 32 x 32); here the peak grows only with the block itself.
    _peak_bytes(4, 25)  # compile outside the measurement
    small, large = _peak_bytes(16, 25), _peak_bytes(32, 25)
    assert large < 2 * small, f"{small} -> {large} bytes"


@pytest.mark.parametrize(
    "case",
    ["slot beyond the cycle", "too few bin centres", "no core cells"],
)
def test_kernel_refuses_what_it_would_index_out_of_bounds(case):
    # The compiled kernel does not bounds-check, so the wrapper refuses these instead of reading or
    # writing past its arrays; the NumPy kernel raised on all three.
    block = np.random.default_rng(0).integers(0, 20, (40, 3)).astype(np.uint16)
    slots = (np.arange(40) % 12 + 1).astype(np.int64)
    centres = np.linspace(-1, 1, 20, dtype=np.float32)
    halo = [0]
    if case == "slot beyond the cycle":
        slots[5] = 13
    elif case == "too few bin centres":
        centres = centres[:10]
    else:
        halo = [2]
    with pytest.raises(ValueError):
        H._slab_quantile_block(block, slots, 12, 20, halo, 3, 0.9, 0.1, False, centres)


def _numpy_kernel(block, slots, n_slots, n_bins, halo, window_steps, q, q_mirror, mirror, bin_centers):
    """The per-cell NumPy kernel the compiled one replaced, kept so its speed-up stays measurable.

    It built each cell's trimmed counts with ``np.bincount`` and ran ``_rolling_histogram_quantile``
    on them, about 1.6 ms per cell on a real record, which left most threads of a node idle
    behind the GIL.
    """
    core_shape = tuple(block.shape[i + 1] - 2 * h for i, h in enumerate(halo))
    out = np.full(core_shape + (n_slots,), np.nan, dtype=np.float32)
    ntime = block.shape[0]
    in_cycle = slots >= 1
    slot_index = slots.astype(np.int64) - 1
    q_used = q_mirror if mirror else q
    trim_top = q_used < 1.0 - 1e-6
    for idx in np.ndindex(*core_shape):
        window = block[(slice(None),) + tuple(slice(i, i + 2 * h + 1) for i, h in zip(idx, halo))].reshape(ntime, -1)
        valid = (window < n_bins) & in_cycle[:, None]
        samples = window[valid].astype(np.int64)
        if mirror:
            samples = (n_bins - 1) - samples
        sample_slots = np.broadcast_to(slot_index[:, None], window.shape)[valid]
        lo, hi = (int(samples.min()), int(samples.max())) if samples.size else (0, 0)
        start = lo - 1 if lo > 0 else 0
        stop = hi + 1 if trim_top else n_bins
        width = stop - start
        counts = np.bincount(sample_slots * width + (samples - start), minlength=n_slots * width).reshape(n_slots, width)
        threshold = H._rolling_histogram_quantile(counts, window_steps, q_used, bin_centers[start:stop])
        out[idx] = -threshold if mirror else threshold
    return out


def _icon_like_block(n_cells=400, ntime=1096, seed=0):
    """400 cells x 1096 daily steps over the 3000-bin ICON geometry, ~400 bins occupied per cell."""
    centres = _centres(0.0131818, 19.7727)
    n_bins = centres.size
    rng = np.random.default_rng(seed)
    centre = rng.integers(400, n_bins - 400, n_cells)
    block = np.clip(np.round(rng.normal(centre, 60.0, (ntime, n_cells))), 0, n_bins - 1).astype(np.uint16)
    block[rng.random((ntime, n_cells)) < 0.02] = n_bins  # NaN samples
    slots = (np.arange(ntime) % 365 + 1).astype(np.int64)
    return block, slots, centres


def _best_of(fn, repeats=3):
    best, result = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


@pytest.mark.slow
@pytest.mark.parametrize("q, mirror", [(0.95, False), (0.05, True)])
def test_kernel_is_three_times_the_numpy_kernel_and_bitwise(q, mirror):
    numba = pytest.importorskip("numba")
    block, slots, centres = _icon_like_block()
    args = (slots, 366, centres.size, [0], 11, q, _q_mirror(q), mirror, centres)
    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        H._slab_quantile_block(block[:, :2], *args)  # compile outside the timing
        new_time, new = _best_of(lambda: H._slab_quantile_block(block, *args))
        ref_time, ref = _best_of(lambda: _numpy_kernel(block, *args))
    finally:
        numba.set_num_threads(previous)
    _assert_bitwise(new, ref)
    assert ref_time / new_time >= 3.0, f"kernel {new_time:.3f} s vs NumPy {ref_time:.3f} s ({ref_time / new_time:.2f}x)"
