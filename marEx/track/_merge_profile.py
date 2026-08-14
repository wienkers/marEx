"""Env-var-gated instrumentation for the unstructured merge/split inner loop.

Set ``MAREX_MERGE_PROFILE=<dir>`` to have every ``process_chunk`` invocation append
one JSON line to ``<dir>/mergeprof_<pid>.jsonl``.  Unset (the default), the whole
thing collapses to a singleton no-op object whose methods return immediately.

Why a file and not a counter in the enclosing scope: ``process_chunk`` is a closure
that cloudpickle serialises **by value** to the workers, so a counter in
``split_and_merge_objects_parallel`` is incremented on a copy and reads back zero.
The same trap is recorded in ``NEXT.md``.

The instrumentation records, per merging child object:

* how many candidate parents the backward scan considered, and how many it accepted
* how many ``(new_id, potential_child)`` pairs the forward re-scan considered --
  the loop with no ``MAX_PARENTS``-style bound, and the prime suspect for the
  observed super-linear wall-clock growth
* wall time in each block: child setup, backward scan, partitioning, relabel,
  ``gc.collect()``, update bookkeeping, forward re-scan

Nothing here changes any value the tracker computes.
"""

import json
import os
import time

# Resolved at import time on each worker process (the module is re-imported there,
# so the environment the workers inherit is what counts).
_DIR = os.environ.get("MAREX_MERGE_PROFILE") or ""
ENABLED = bool(_DIR)

# Cap the per-child detail so a pathological run cannot fill the filesystem; the
# aggregate totals below are always complete regardless of this cap.
MAX_ROWS = int(os.environ.get("MAREX_MERGE_PROFILE_MAXROWS", 200000))


class _NullProfile:
    """No-op stand-in used when profiling is disabled."""

    __slots__ = ()
    enabled = False

    def add(self, key, dt):
        pass

    def count(self, key, n=1):
        pass

    def row(self, **kwargs):
        pass

    def dump(self, **meta):
        pass


class ChunkProfile:
    """Accumulator for one ``process_chunk`` call, dumped as a single JSON line."""

    __slots__ = ("t", "n", "rows", "_t0")
    enabled = True

    def __init__(self):
        """Start a fresh accumulator timed from construction."""
        self.t = {}
        self.n = {}
        self.rows = []
        self._t0 = time.perf_counter()

    def add(self, key, dt):
        """Accumulate dt seconds into the named timing bucket."""
        self.t[key] = self.t.get(key, 0.0) + dt

    def count(self, key, n=1):
        """Accumulate n into the named counter."""
        self.n[key] = self.n.get(key, 0) + n

    def row(self, **kwargs):
        """Append one per-child detail row, capped at MAX_ROWS."""
        if len(self.rows) < MAX_ROWS:
            self.rows.append(kwargs)

    def dump(self, **meta):
        """Write the accumulated record as one JSON line under _DIR."""
        record = {
            "pid": os.getpid(),
            "wall_s": time.perf_counter() - self._t0,
            "totals_s": self.t,
            "counts": self.n,
            "rows": self.rows,
        }
        record.update(meta)
        path = os.path.join(_DIR, f"mergeprof_{os.getpid()}.jsonl")
        try:
            os.makedirs(_DIR, exist_ok=True)
            with open(path, "a") as handle:
                handle.write(json.dumps(record, default=float) + "\n")
        except OSError:  # instrumentation must never break the run
            pass


_NULL = _NullProfile()


def make():
    """Return a live profile when enabled, else the shared no-op singleton."""
    return ChunkProfile() if ENABLED else _NULL
