"""
MarEx Track: checked int32 event-ID allocator for 4-D tracking.

Every new event ID, whether it comes from per-timestep identification or from a
merge, is drawn from one allocator. All arithmetic is in Python integers and each
request is validated against the int32 range before any state changes, so an
exhausted ID space raises instead of wrapping silently inside an int32 field.
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ...exceptions import TrackingError

INT32_MAX = 2**31 - 1


class IDAllocator:
    """
    Hand out consecutive positive int32 IDs, never exceeding ``INT32_MAX``.

    Parameters
    ----------
    next_id : int, default=1
        First ID that will be issued. 0 is the background label and is never issued.
    """

    def __init__(self, next_id: int = 1) -> None:
        """Start issuing at ``next_id``."""
        next_id = int(next_id)
        if next_id < 1:
            raise TrackingError(
                f"The first event ID must be at least 1, got {next_id}",
                details="0 is the background label",
                context={"next_id": next_id},
            )
        self._next_id = next_id

    @property
    def next_id(self) -> int:
        """The next ID to be issued (a Python int)."""
        return self._next_id

    def _check(self, n: int) -> None:
        last = self._next_id + n - 1
        if last > INT32_MAX:
            raise TrackingError(
                "The event ID space is exhausted: IDs are int32",
                details=f"Allocating {n} ID(s) from {self._next_id} would reach {last}, above the int32 maximum {INT32_MAX}",
                suggestions=[
                    "Raise volume_filter_quartile or volume_filter_absolute to keep fewer, larger events",
                    "Track a shorter time range",
                ],
                context={"next_id": self._next_id, "requested": n, "int32_max": INT32_MAX},
            )

    def allocate(self, n: int) -> NDArray[np.int32]:
        """
        Issue ``n`` consecutive IDs.

        Raises
        ------
        TrackingError
            If the last ID would exceed ``INT32_MAX``. The allocator is unchanged.
        """
        n = int(n)
        if n < 0:
            raise TrackingError(f"Cannot allocate a negative number of IDs ({n})")
        self._check(n)
        first = self._next_id
        # np.arange with explicit int64 bounds: the range was checked above, so the cast is exact.
        ids = np.arange(first, first + n, dtype=np.int64).astype(np.int32)
        self._next_id = first + n
        return ids

    def reserve_offsets(self, per_slice_counts: Sequence[int]) -> NDArray[np.int32]:
        """
        Reserve a block of IDs per slice and return each block's offset.

        Slice ``i`` owns the IDs ``next_id + offset_i + 1 .. next_id + offset_i + counts[i]``
        counted from the allocator's state at the call; the returned offset is
        ``(next_id - 1) + exclusive_cumsum(counts)[i]``, so adding it to a slice's local
        labels ``1..counts[i]`` gives globally unique IDs. The sum is formed in Python
        integers and checked before the allocator advances.

        Raises
        ------
        TrackingError
            If the last ID would exceed ``INT32_MAX``. The allocator is unchanged.
        """
        counts = [int(c) for c in per_slice_counts]
        if any(c < 0 for c in counts):
            raise TrackingError("Per-slice object counts must be non-negative", context={"counts": counts[:10]})
        total = sum(counts)
        self._check(total)
        offsets = []
        running = self._next_id - 1
        for c in counts:
            offsets.append(running)
            running += c
        self._next_id += total
        return np.asarray(offsets, dtype=np.int64).astype(np.int32)
