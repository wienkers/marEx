"""Exact seasonal quantiles over a horizontal neighbourhood."""

import numpy as np
from numba import config, njit
from numba.extending import register_jitable

from ._exact_seasonal import _slot_index

_BUCKETS = 256
_SAMPLE_SIZE = 16_384
_MIN_COMPILED_WORK = 1_000


@register_jitable
def _select(values, weights, size, rank):
    """Partition original values in place, carrying cell-constant counts as weights."""
    start, stop = 0, size
    while stop - start > 1:
        pivot = values[(start + stop) // 2]
        lower, upper, i = start, stop, start
        below, equal = 0, 0
        while i < upper:
            value = values[i]
            weight = weights[i]
            if value < pivot:
                values[i], values[lower] = values[lower], value
                weights[i], weights[lower] = weights[lower], weight
                lower += 1
                below += weight
            elif value > pivot:
                upper -= 1
                values[i], values[upper] = values[upper], value
                weights[i], weights[upper] = weights[upper], weight
                continue
            else:
                equal += weight
            i += 1
        if rank < below:
            stop = lower
        elif rank < below + equal:
            return pivot
        else:
            start = upper
            rank -= below + equal
    return values[start]


def _posting_index(source, edges):
    """Group source positions by cell and bucket, retaining seasonal slot order."""
    n_cells = source.shape[1]
    ids = np.searchsorted(edges, source, side="right").astype(np.uint16)
    ids[np.isnan(source)] = _BUCKETS
    cell_buckets = _BUCKETS * np.arange(n_cells, dtype=np.int64)
    keys = cell_buckets + ids
    n_keys = n_cells * _BUCKETS
    keys[ids == _BUCKETS] = n_keys
    keys = keys.ravel()
    offsets = np.empty(n_keys + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(np.bincount(keys, minlength=n_keys + 1)[:-1], out=offsets[1:])
    posting = np.argsort(keys, kind="stable")

    # The extra entry receives NaNs, then is reused for tile-wide extrema.
    extrema = np.empty((2, n_keys + _BUCKETS), dtype=np.float64)
    extrema[0] = np.inf
    extrema[1] = -np.inf
    np.fmin.at(extrema[0], keys, source.ravel())
    np.fmax.at(extrema[1], keys, source.ravel())
    extrema[0, n_keys:] = extrema[0, :n_keys].reshape(n_cells, _BUCKETS).min(axis=0)
    extrema[1, n_keys:] = extrema[1, :n_keys].reshape(n_cells, _BUCKETS).max(axis=0)
    return ids, offsets, extrema, posting, cell_buckets


def _slot_buckets(ids, cell_buckets, first, last):
    departing_or_entering = ids[first:last]
    return (cell_buckets + departing_or_entering)[departing_or_entering < _BUCKETS]


def _strip_sums(active, y_edges, x_edges, periodic, vertical, horizontal):
    """Sum over y first, then over x, using exact differences of prefix sums."""
    np.cumsum(active, axis=0, out=vertical[1:])
    y0, y1 = y_edges
    np.cumsum(vertical[y1] - vertical[y0], axis=1, out=horizontal[:, 1:])
    x0, x1 = x_edges
    if periodic:
        # Each complete traversal adds a whole strip. Remainders retain the
        # position of the seam, including windows wider than the entire grid.
        nx = active.shape[1]
        n0, x0 = np.divmod(x0, nx)
        n1, x1 = np.divmod(x1, nx)
        return np.take(horizontal, x1, axis=1) - np.take(horizontal, x0, axis=1) + (n1 - n0)[None, :, None] * horizontal[:, -1:]
    return np.take(horizontal, x1, axis=1) - np.take(horizontal, x0, axis=1)


@njit(nogil=True, no_rewrites=True, no_cfunc_wrapper=True)
def _crossing_values(source, offsets, posting, extrema, active, heads, queries, y_edges, x_edges, nx, values, weights, selected):
    """Recover original endpoints for the selected bucket and rank in each query."""
    core_x = x_edges.shape[1]
    n_core = selected.size // 2
    for centre in range(n_core):
        cy, cx = centre // core_x, centre % core_x
        gathered, size = -1, 0
        for query in range(centre, centre + 2 * n_core, n_core):
            bucket = queries[0, query]
            if bucket < 0:
                continue
            tile_bucket = extrema.shape[1] - _BUCKETS + bucket
            if extrema[0, tile_bucket] == extrema[1, tile_bucket]:
                selected[query] = extrema[0, tile_bucket]
                continue
            if bucket != gathered:
                gathered, size = bucket, 0
                for y in range(y_edges[0, cy], y_edges[1, cy]):
                    for x in range(x_edges[0, cx], x_edges[1, cx]):
                        index = (y * nx + x % nx) * _BUCKETS + bucket
                        count = active[index]
                        if not count:
                            continue
                        low = extrema[0, index]
                        if low == extrema[1, index]:
                            # One weighted entry replaces all its active samples.
                            values[size] = low
                            weights[size] = count
                            size += 1
                            continue
                        head = heads[index]
                        for _ in range(count):
                            values[size] = source[posting[head]]
                            weights[size] = 1
                            size += 1
                            head += 1
                            if head == offsets[index + 1]:
                                head = offsets[index]
            selected[query] = _select(values, weights, size, queries[1, query])


def _rolling_pool(source, edges, bounds, width, half, core_origin, periodic, fraction, max_active, result, replay, shape):
    """Advance the slot cursors and build the two spatial strip sums per slot."""
    ny, nx = shape
    ids, offsets, extrema, posting, cell_buckets = _posting_index(source, edges)
    n_slots = bounds.size - 1
    active = np.zeros(offsets.size - 1, dtype=np.int64)
    heads = offsets[:-1].copy()
    values = np.empty(max_active, dtype=np.float64)
    weights = np.empty(max_active, dtype=np.int64)
    y = core_origin[0] + np.arange(result.shape[1], dtype=np.int64)
    x = core_origin[1] + np.arange(result.shape[2], dtype=np.int64)
    y_edges = np.maximum(0, y - half), np.minimum(ny, y + half + 1)
    x_edges = x - half, x + half + 1
    if not periodic:
        x_edges = np.maximum(0, x_edges[0]), np.minimum(nx, x_edges[1])
    vertical = np.zeros((ny + 1, nx, _BUCKETS), dtype=np.int64)
    horizontal = np.zeros((y.size, nx + 1, _BUCKETS), dtype=np.int64)
    y_bounds, x_bounds = np.stack(y_edges), np.stack(x_edges)
    selected = np.empty((2, y.size, x.size), dtype=np.float64)
    ranks = np.empty(selected.shape, dtype=np.int64)
    queries = np.empty((2, *ranks.shape), dtype=np.int64)
    np.add.at(active, _slot_buckets(ids, cell_buckets, 0, bounds[width]), 1)

    bits = source.view(np.uint64)
    mixed_zeros = np.any(bits == 1 << 63) and np.any(bits == 0)
    dtype = fraction.dtype
    for slot in range(result.shape[0]):
        if slot:
            depart = _slot_buckets(ids, cell_buckets, bounds[slot - 1], bounds[slot])
            np.subtract.at(active, depart, 1)
            # Duplicate postings in this slot each advance the circular head once.
            np.add.at(heads, depart, 1)
            starts = offsets[depart]
            heads[depart] = starts + (heads[depart] - starts) % (offsets[depart + 1] - starts)
            entering = (slot + width - 1) % n_slots
            arrive = _slot_buckets(ids, cell_buckets, bounds[entering], bounds[entering + 1])
            np.add.at(active, arrive, 1)

        pool = _strip_sums(active.reshape(ny, nx, _BUCKETS), y_edges, x_edges, periodic, vertical, horizontal)
        np.cumsum(pool, axis=-1, out=pool)
        totals = pool[..., -1]
        if not np.any(totals):
            continue
        virtual = np.multiply(totals - 1, fraction, dtype=dtype)
        ranks[0] = np.where(totals, virtual, -1)
        ranks[1] = np.minimum(ranks[0] + 1, totals - 1)
        # At the maximum NumPy indexes both endpoints by -1. It uses that
        # index to calculate the weight, even though the endpoints are equal.
        gamma = np.subtract(virtual, np.where(ranks[0] < totals - 1, ranks[0], -1), dtype=dtype)
        # The number of prefixes at or below a rank is its crossing bucket.
        buckets = np.count_nonzero(pool <= ranks[..., None], axis=-1)
        before = np.take_along_axis(pool, np.maximum(0, buckets - 1).transpose(1, 2, 0), axis=-1).transpose(2, 0, 1)
        queries[0] = np.where(totals, buckets, -1)
        queries[1] = ranks - np.where(buckets, before, 0)
        selected.fill(np.nan)
        _crossing_values(
            source.ravel(),
            offsets,
            posting,
            extrema,
            active,
            heads,
            queries.reshape(2, -1),
            y_bounds,
            x_bounds,
            nx,
            values,
            weights,
            selected.ravel(),
        )
        lo, hi = selected.astype(dtype, copy=False)
        with np.errstate(invalid="ignore", over="ignore"):
            difference = hi - lo
            result[slot] = np.where(gamma >= 0.5, hi - difference * (1 - gamma), lo + difference * gamma)
        if mixed_zeros:
            replay[slot] = (lo == 0) | (hi == 0)


def _sample_edges(values):
    # Flat striding samples the whole tile without copying it when non-contiguous.
    stride = max(1, (values.size + _SAMPLE_SIZE - 1) // _SAMPLE_SIZE)
    sample = values.flat[::stride]
    sample = np.sort(sample[np.isfinite(sample)])
    z = np.arange(1, _BUCKETS) / _BUCKETS
    cuts = np.where(z < 0.5, 2 * z**2, 1 - 2 * (1 - z) ** 2)
    return sample[(cuts * (sample.size - 1)).astype(np.intp)].astype(np.float64) if sample.size else np.zeros(_BUCKETS - 1)


def _numpy_pool(values, slots, n_slots, width, half, core_origin, percentile, periodic, result, replay=None):
    """Replay selected reductions in the original time, y and x-offset order."""
    ny, nx = values.shape[-2:]
    y_core, x_core = core_origin
    for slot in range(result.shape[0]):
        if replay is not None and not replay[slot].any():
            continue
        valid = (slots >= 1) & (slots <= n_slots)
        if width != n_slots:
            valid &= (slots - 1 - slot + width // 2) % n_slots < width
        times = np.flatnonzero(valid)
        if not times.size:
            continue
        for cy in range(result.shape[1]):
            y = y_core + cy
            ys = slice(max(0, y - half), min(ny, y + half + 1))
            for cx in range(result.shape[2]):
                if replay is not None and not replay[slot, cy, cx]:
                    continue
                x = x_core + cx
                if periodic:
                    xs = np.arange(x - half, x + half + 1)
                    xs %= nx
                    pool = values[np.ix_(times, np.arange(ys.start, ys.stop), xs)]
                else:
                    xs = slice(max(0, x - half), min(nx, x + half + 1))
                    pool = values[times, ys, xs]
                if not np.isnan(pool).all():
                    with np.errstate(invalid="ignore", over="ignore"):
                        result[slot, cy, cx] = np.nanpercentile(pool, percentile, method="linear", overwrite_input=True)


def _pooled_percentiles(
    values, slots, n_slots, window_steps, half, core_origin, core_shape, percentile, *, periodic=False, direct=False
):
    """Reduce a haloed time/y/x tile using 1-based integer slot labels."""
    width = min(n_slots, 2 * (window_steps // 2) + 1)
    n_reductions = 1 if width == n_slots else n_slots
    result = np.full((n_reductions, *core_shape), np.nan, dtype=np.float32)
    if not values.size:
        return np.broadcast_to(result, (n_slots, *core_shape))

    order, bounds = _slot_index(slots, n_slots, width)
    ny, nx = values.shape[-2:]
    counts = np.concatenate((bounds, bounds[1:] + order.size))
    populations = counts[width : width + n_reductions] - bounds[:n_reductions]
    max_active = int(populations.max()) * min(ny, 2 * half + 1) * (2 * half + 1 if periodic else min(nx, 2 * half + 1))
    if not max_active:
        return np.broadcast_to(result, (n_slots, *core_shape))

    dtype = values.dtype
    fraction = np.true_divide(percentile, dtype.type(100)) if dtype in (np.float32, np.float64) else None
    # With NumPy 1, float32 quantiles promote to float64 under division by Python 100.0.
    native_rank = (
        fraction is not None
        and fraction.ndim == 0
        and fraction.dtype == dtype
        and np.true_divide(fraction, 1.0).dtype == dtype
        and (1 * fraction).dtype == dtype
    )
    if (
        direct
        or config.DISABLE_JIT
        or not native_rank
        or fraction.dtype != dtype
        or not 0 <= fraction <= 1
        or (dtype == np.float32 and max_active > 2**24)
        or np.count_nonzero(populations) * core_shape[0] * core_shape[1] < _MIN_COMPILED_WORK
    ):
        _numpy_pool(values, slots, n_slots, width, half, core_origin, percentile, periodic, result)
    else:
        # The only compiled signature takes a contiguous float64 source.
        # This conversion is exact for float32, including zeros and subnormals.
        edges = _sample_edges(values)
        source = np.ascontiguousarray(values[order], dtype=np.float64).reshape(order.size, ny * nx)
        replay = np.zeros(result.shape, dtype=np.bool_)
        _rolling_pool(source, edges, bounds, width, half, core_origin, periodic, fraction, max_active, result, replay, (ny, nx))
        if replay.any():
            _numpy_pool(values, slots, n_slots, width, half, core_origin, percentile, periodic, result, replay)

    return np.broadcast_to(result, (n_slots, *core_shape))
