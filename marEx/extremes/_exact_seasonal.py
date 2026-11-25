"""Exact per-cell seasonal percentiles from radix-sorted rolling ranks."""

from math import prod
from sys import byteorder
from typing import Literal

import numpy as np
from llvmlite import ir
from numba import config, njit, types
from numba.extending import intrinsic

_STRIP_BYTES = 1 << 20
_STRIP_ROWS = 64
# A cold compile costs about a second. NumPy takes about 60 microseconds per
# cell-slot on a daily tile, so allow headroom for smaller windows and gappy tiles.
_MIN_COMPILED_WORK = 50_000


@intrinsic
def _popcount(typingctx, word):
    def codegen(context, builder, signature, args):
        return builder.ctpop(args[0])

    return types.int64(word), codegen


@intrinsic
def _trailing_zeros(typingctx, word):
    def codegen(context, builder, signature, args):
        return builder.cttz(args[0], ir.Constant(ir.IntType(1), 1))

    return types.int64(word), codegen


# All dtype-specific work happens in NumPy. The dispatcher sees one writable,
# contiguous integer layout and can compile in a read-only installation.
@njit(nogil=True, no_rewrites=True, no_cfunc_wrapper=True)
def _rolling_rank_percentiles(order, sizes, time_order, bounds, width, lower_ranks, ranks, active, selected):
    """Write each window's lower sample index, upper sample index and valid count."""
    n_slots = bounds.size - 1
    n_reductions = 1 if width == n_slots else n_slots
    for row in range(sizes.size):
        count = sizes[row]
        if count == 0:
            continue
        row_start = row * time_order.size
        for j in range(count):
            ranks[order[row_start + j]] = j
        for j in range(count, time_order.size):
            ranks[order[row_start + j]] = -1
        for word_index in range(active.size):
            active[word_index] = 0
        total = 0
        leaving_start = 0
        leaving_end = 0
        entering_start = 0
        entering_end = bounds[width]
        for slot in range(n_reductions):
            for change in range(2):
                start = leaving_start if change == 0 else entering_start
                stop = leaving_end if change == 0 else entering_end
                delta = 2 * change - 1
                for i in range(start, stop):
                    rank = ranks[i]
                    if rank >= 0:
                        # Each rank is unique, so entering and leaving can both toggle it.
                        active[rank >> 6] ^= np.uint64(1) << np.uint64(rank & 63)
                        total += delta

            if total:
                lo = lower_ranks[total]
                remaining = lo
                word_index = 0
                word = active[word_index]
                while remaining >= _popcount(word):
                    remaining -= _popcount(word)
                    word_index += 1
                    word = active[word_index]
                for _ in range(remaining):
                    word &= word - np.uint64(1)
                lower = time_order[order[row_start + (word_index << 6) + _trailing_zeros(word)]]
                upper = lower
                if lo + 1 < total:
                    word &= word - np.uint64(1)
                    while word == 0:
                        word_index += 1
                        word = active[word_index]
                    upper = time_order[order[row_start + (word_index << 6) + _trailing_zeros(word)]]
                index = 3 * (row * n_reductions + slot)
                selected[index] = lower
                selected[index + 1] = upper
                selected[index + 2] = total

            entering = slot + width
            if entering >= n_slots:
                entering -= n_slots
            leaving_start = bounds[slot]
            leaving_end = bounds[slot + 1]
            entering_start = bounds[entering]
            entering_end = bounds[entering + 1]


def _radix_order(bits, time_order, sign, magnitude, infinity, exceptional):
    """Use stable byte-key passes to sort the IEEE representations, with NaNs last."""
    raw = bits[:, time_order]
    missing = (raw & magnitude) > infinity
    # Complement negatives and flip the sign bit of the non-negative values.
    keys = np.ascontiguousarray(np.where(raw & sign, ~raw, raw ^ sign))
    keys[missing] = sign | magnitude
    sizes = np.int32(time_order.size) - np.count_nonzero(missing, axis=1).astype(np.int32)
    sizes[exceptional] = 0
    digits = keys.view(np.uint8).reshape(keys.shape + (keys.itemsize,)).transpose(2, 0, 1)
    # lexsort applies the least significant byte first.
    if byteorder == "big":
        digits = digits[::-1]
    return np.lexsort(digits, axis=-1).astype(np.int32), sizes


def _slot_index(slots, n_slots, width):
    """Order valid samples by slot, starting at the left edge of the first window."""
    valid = np.flatnonzero((slots >= 1) & (slots <= n_slots))
    rotated = (slots[valid].astype(np.intp) - 1 + width // 2) % n_slots
    order = np.argsort(rotated, kind="stable")
    bounds = np.searchsorted(rotated[order], np.arange(n_slots + 1))
    return valid[order], bounds


def _native_fraction(percentile, dtype):
    """Return a scalar fraction when native rank arithmetic matches NumPy."""
    if dtype not in (np.float32, np.float64):
        return None
    fraction = np.true_divide(percentile, dtype.type(100))
    # NumPy 1.x divides by Python 100.0, which promotes float32 quantiles.
    if (
        fraction.ndim == 0
        and fraction.dtype == dtype
        and np.true_divide(fraction, 1.0).dtype == dtype
        and (1 * fraction).dtype == dtype
        and 0 <= fraction <= 1
    ):
        return fraction
    return None


def _numpy_doy_percentiles(values, time_order, bounds, width, percentile, result):
    """Reduce one slot at a time, preserving the original sample order within it."""
    n_slots = bounds.size - 1
    for slot in range(1 if width == n_slots else n_slots):
        end = slot + width
        if end <= n_slots:
            members = time_order[bounds[slot] : bounds[end]]
        else:
            members = np.concatenate((time_order[bounds[slot] :], time_order[: bounds[end - n_slots]]))
        if not members.size:
            continue
        samples = values[:, np.sort(members)]
        start = 0
        if values.dtype.type is np.float16 and np.isnan(samples[0]).all():
            # NumPy sizes the output from the first row. A dry float16 row
            # would round valid quantiles computed with wider percentile scalars.
            wet = (~np.isnan(samples)).any(axis=1)
            start = wet.argmax()
            if not wet[start]:
                continue
        result[start:, slot] = np.nanpercentile(samples[start:], percentile, axis=-1)


def _doy_percentiles(values, slots, n_slots, window_steps, percentile, *, _backend: Literal["auto", "compiled", "numpy"] = "auto"):
    """Compute exact thresholds for a time-last block, with float32 output."""
    if isinstance(values, np.ma.MaskedArray):
        out_dt = np.result_type(values.dtype, float) if values.dtype.kind in "bui" else values.dtype
        values = np.asarray(
            np.ma.filled(values.astype(out_dt, copy=False), np.nan) if np.ma.is_masked(values) else np.ma.getdata(values),
            dtype=out_dt,
        )
    spatial_shape = values.shape[:-1]
    n_time = values.shape[-1]
    n_cells = prod(spatial_shape)
    result = np.full((n_cells, n_slots), np.nan, dtype=np.float32)
    if not n_cells or not n_time:
        return result.reshape(spatial_shape + (n_slots,))

    # The symmetric window contains 2 * half + 1 slots. A full-cycle window
    # counts each slot once and needs only one reduction per cell.
    width = min(n_slots, 2 * (window_steps // 2) + 1)
    time_order, bounds = _slot_index(slots, n_slots, width)
    if not time_order.size:
        return result.reshape(spatial_shape + (n_slots,))
    dtype = values.dtype
    n_reductions = 1 if width == n_slots else n_slots
    populated = 1
    if _backend == "auto" and width < n_slots:
        # Only windows with samples call nanpercentile. Count their contiguous
        # and wrapped spans separately, including when the cycle exceeds the record.
        populated = np.count_nonzero(bounds[width:] != bounds[:-width])
        populated += np.count_nonzero((bounds[-width:-1] < bounds[-1]) | (bounds[1:width] > 0))
    work = n_cells * populated
    fraction = None
    if (
        not config.DISABLE_JIT
        and n_time <= 2**24
        and (_backend == "compiled" or (_backend == "auto" and work >= _MIN_COMPILED_WORK))
    ):
        fraction = _native_fraction(percentile, dtype)

    rows_per_strip = max(1, min(_STRIP_ROWS, _STRIP_BYTES // (max(n_time, n_reductions) * dtype.itemsize)))
    # Adjacent spatial strides must agree before reshaping. NumPy can allocate a
    # temporary copy even when an in-place shape assignment eventually fails.
    axes = [(size, stride) for size, stride in zip(spatial_shape, values.strides) if size > 1]
    collapsible = all(left == size * right for (_, left), (size, right) in zip(axes, axes[1:]))
    rows = values.reshape(n_cells, n_time) if collapsible else None

    if fraction is not None:
        strip_buffer = np.empty((min(n_cells, rows_per_strip), n_time), dtype=dtype)
        unsigned = np.uint32 if dtype == np.float32 else np.uint64
        sign = unsigned(1 << (8 * dtype.itemsize - 1))
        magnitude = sign - unsigned(1)
        inf = np.array(np.inf, dtype=dtype).view(unsigned)[()]
        time_order = time_order.astype(np.int32)
        bounds = bounds.astype(np.int32)
        size = time_order.size
        # Entry n uses the virtual rank (n - 1) * fraction in the input dtype.
        # Entry zero carries NaN for windows without any valid samples.
        virtual_ranks = np.arange(-1, size, dtype=dtype) * fraction
        lower_ranks = virtual_ranks.astype(np.int32)
        weights = np.subtract(virtual_ranks, lower_ranks, dtype=dtype)
        weights[0] = np.nan
        selected = np.empty((strip_buffer.shape[0], n_reductions, 3), dtype=np.int32)
        ranks = np.empty(size, dtype=np.int32)
        active = np.empty((size + 63) // 64, dtype=np.uint64)

    for start in range(0, n_cells, rows_per_strip):
        stop = min(start + rows_per_strip, n_cells)
        if rows is None:
            strip = values[np.unravel_index(np.arange(start, stop), spatial_shape)]
        else:
            strip = rows[start:stop]
        destination = result[start:stop]
        if fraction is not None:
            buffer = strip_buffer[: stop - start]
            buffer[:] = strip
            bits = buffer.view(unsigned)
            exceptional = np.any((bits == sign) | ((bits & magnitude) == inf), axis=1)
            if not exceptional.all():
                order, sizes = _radix_order(bits, time_order, sign, magnitude, inf, exceptional)
                neighbours = selected[: stop - start]
                neighbours.fill(0)
                _rolling_rank_percentiles(
                    order.ravel(),
                    sizes,
                    time_order,
                    bounds,
                    width,
                    lower_ranks,
                    ranks,
                    active,
                    neighbours.ravel(),
                )
                lower = np.take_along_axis(buffer, neighbours[..., 0], axis=1)
                upper = np.take_along_axis(buffer, neighbours[..., 1], axis=1)
                weight = weights[neighbours[..., 2]]
                # NumPy evaluates the adjacent difference even at integer ranks.
                # With finite endpoints it can overflow, giving inf * 0 and NaN.
                with np.errstate(invalid="ignore", over="ignore"):
                    difference = upper - lower
                    interpolated = np.where(
                        weight >= 0.5, upper - difference * (dtype.type(1) - weight), lower + difference * weight
                    )
                    destination[:, :n_reductions] = np.where(neighbours[..., 2] == 0, np.float32(np.nan), interpolated)
            if exceptional.any():
                fallback = np.full((np.count_nonzero(exceptional), n_reductions), np.nan, dtype=np.float32)
                _numpy_doy_percentiles(buffer[exceptional], time_order, bounds, width, percentile, fallback)
                destination[exceptional, :n_reductions] = fallback
        else:
            _numpy_doy_percentiles(strip, time_order, bounds, width, percentile, destination)

    if width == n_slots:
        result[:, 1:] = result[:, :1]
    return result.reshape(spatial_shape + (n_slots,))
