"""Shared machinery for the larger-than-memory squeeze demonstrations.

These scripts are deliberately NOT part of the pytest suite: each one wants a whole
SLURM allocation and tens of minutes to hours of wall clock.  What they establish is
the one claim the unit tests structurally cannot -- that a workload which does *not*
complete under ``compute_mode="persist"`` *does* complete under ``"streaming"`` at the
same memory budget.

Every guard rail in here exists because its absence has already produced a wrong result
somewhere in this project's history:

* **The effective per-worker memory limit is asserted, not assumed.**  Dask reads the
  host's total RAM, not the cgroup the batch system put the job in.  Without an explicit
  ``memory_limit`` the workers believe they own the whole node, the squeeze never binds,
  ``persist`` completes, and the test "passes" for entirely the wrong reason.
* **A failure is classified from evidence, never inferred.**  A wall-clock kill is not a
  proof of OOM, and an OOM is not proven by slowness.  We watch for ``KilledWorker``,
  ``MemoryError``, and the nanny's own "exceeded 95%" warnings, and report
  ``inconclusive`` when none of them fired.
* **Bytes pinned are counted, not guessed.**  An array is *still* a dask collection after
  ``.persist()``, so ``is_dask_collection`` proves nothing about laziness.  The accountant
  patches the three persist entry points and attributes every byte to the marEx source
  line that asked for it -- including the ``from dask import persist`` sites, which a
  naive patch of ``dask.persist`` silently misses.
* **A breadcrumb summary is written before the work starts**, so a run killed by the wall
  clock still leaves a record saying what it was attempting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict

GB = 1e9


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the CLI flags shared by every squeeze leg (cluster shape, mode, scratch, deadline)."""
    parser.add_argument("--label", required=True, help="Identifier for this leg; names the summary file.")
    parser.add_argument("--outdir", required=True, help="Directory for the JSON summary and memory series.")
    parser.add_argument(
        "--mode",
        default="persist",
        choices=("persist", "lazy", "streaming"),
        help="compute_mode under test. 'lazy' is detect-only.",
    )
    parser.add_argument("--nt", type=int, default=0, help="Truncate the input to this many timesteps (0 = all).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mem-per-worker", default="6GB", help="Per-worker memory_limit, e.g. '6GB'.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--scratch", required=True, help="Scratch root for dask spill dirs and staging.")
    parser.add_argument(
        "--no-spill",
        dest="spill",
        action="store_false",
        default=True,
        help="Disable dask's spill-to-disk. MEASURED 2026-08-26 to be a trap, kept only as a "
        "labelled control: with spilling off, `pause: 0.90` makes a worker that crosses the "
        "threshold pause with no way back down, so the cluster deadlocks instead of finishing, "
        "and peak memory RISES because what would have spilled stays resident. It changed the "
        "outcome on both sides of two separate legs, which destroys the comparison rather than "
        "sharpening it. Default is dask's real behaviour.",
    )
    parser.add_argument(
        "--deadline",
        type=int,
        default=0,
        help="Seconds after which the leg aborts itself (0 = rely on the SLURM wall). A self-imposed "
        "deadline produces a summary file; a wall kill does not.",
    )
    parser.add_argument("--validate", action="store_true", help="Pass validate=True to preprocess_data.")
    return parser


def parse_memory(text: str) -> int:
    """'6GB' -> 6_000_000_000. Accepts GB/GiB/MB/MiB or a bare byte count."""
    text = str(text).strip()
    units = {"KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}
    for suffix, scale in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if text.upper().endswith(suffix):
            return int(float(text[: -len(suffix)]) * scale)
    return int(float(text))


# --------------------------------------------------------------------------------------
# Cluster construction, with the limit assertion that makes the squeeze real
# --------------------------------------------------------------------------------------
def build_cluster(args) -> tuple:
    """Start a LocalCluster whose *effective* memory limit is verified against the job's.

    Returns ``(client, metadata)``. Raises SystemExit if the workers ended up with more
    memory than the batch allocation grants, because in that case nothing is being tested.
    """
    import dask
    from dask.distributed import Client, LocalCluster

    # Dask's defaults, unless a control leg explicitly asks for spilling off.
    #
    # Disabling spill looks like it should sharpen the result -- "fits in memory" would then
    # mean RAM, and the outcome would be binary.  Measured, it does the opposite.  A worker
    # that crosses `pause` can no longer spill its way back down, so it pauses permanently and
    # the cluster deadlocks; and peak memory RISES, because everything that would have spilled
    # stays resident.  On the unstructured tracker at 192 GB, the same configuration that
    # completes in 70 min at 118.7 GB peak with spilling on (f4_stream 27255851) was killed at
    # 169.5 GB peak with it off.  Both modes then fail, and a leg where neither mode completes
    # proves nothing about either.
    spill_config = (
        {}
        if args.spill
        else {
            "distributed.worker.memory.target": False,
            "distributed.worker.memory.spill": False,
            "distributed.worker.memory.pause": 0.90,
            "distributed.worker.memory.terminate": 0.95,
        }
    )
    dask.config.set(spill_config)

    scratch = Path(args.scratch) / "dask" / args.label
    scratch.mkdir(parents=True, exist_ok=True)

    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads,
        memory_limit=args.mem_per_worker,
        processes=True,
        local_directory=str(scratch),
    )
    client = Client(cluster)
    client.wait_for_workers(args.workers)

    # Ask the workers themselves. Client.scheduler_info() can serve a cached identity,
    # and the whole point here is to learn what actually bound.
    limits = client.run(lambda dask_worker: dask_worker.memory_manager.memory_limit)
    distinct = sorted({int(v) for v in limits.values()})
    total = sum(int(v) for v in limits.values())

    slurm_mem_mb = os.environ.get("SLURM_MEM_PER_NODE")
    slurm_mem = int(slurm_mem_mb) * 2**20 if slurm_mem_mb else None

    meta = {
        # `n_workers_requested` is what was asked for and what `wait_for_workers` gated on;
        # `n_workers` is what `client.run` answered from. The ABORT above makes them equal on
        # any run that gets this far, so a consumer comparing the two is NOT cross-checking a
        # live cluster -- that check happens here, once, and again on every sample in
        # `SpillSampler.run`. Recording both is for the RETROSPECTIVE case: summaries written
        # before 2026-09-08 carry neither key, and must read UNMEASURED rather than inherit a
        # coverage guarantee that did not exist when they were produced.
        "n_workers_requested": args.workers,
        "n_workers": len(limits),
        "threads_per_worker": args.threads,
        "worker_memory_limits_bytes": distinct,
        "cluster_memory_limit_bytes": total,
        "slurm_mem_per_node_bytes": slurm_mem,
        "spill_enabled": bool(args.spill),
    }
    print(
        f"[{args.label}] cluster: {len(limits)} x {args.threads}t, effective per-worker limit "
        f"{[f'{v / GB:.2f} GB' for v in distinct]}, aggregate {total / GB:.1f} GB"
        + (f" (SLURM grants {slurm_mem / GB:.1f} GB)" if slurm_mem else " (SLURM --mem not visible)"),
        flush=True,
    )

    if len(limits) != args.workers:
        client.close()
        cluster.close()
        sys.exit(
            f"ABORT: requested {args.workers} workers but client.run() answered from {len(limits)}. "
            "Every width derived downstream -- `n_workers`, and with it the spill sampler's "
            "coverage check -- would inherit the undercount instead of catching it."
        )
    if len(distinct) != 1 or distinct[0] != parse_memory(args.mem_per_worker):
        client.close()
        cluster.close()
        sys.exit(
            f"ABORT: requested memory_limit={args.mem_per_worker} but workers report {distinct}. "
            "The squeeze would not bind; refusing to produce a meaningless pass."
        )
    if slurm_mem is not None and total > slurm_mem:
        client.close()
        cluster.close()
        sys.exit(
            f"ABORT: aggregate worker limit {total / GB:.1f} GB exceeds the SLURM allocation "
            f"{slurm_mem / GB:.1f} GB. The cgroup, not compute_mode, would decide the outcome."
        )
    return client, meta


# --------------------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------------------
class NannyWatcher(logging.Handler):
    """Count the nanny warnings that distinguish a memory cascade from mere slowness."""

    PHRASES = ("exceeded 95", "exceeded 95%", "memory budget", "worker exceeded", "restarting worker")

    def __init__(self) -> None:
        """Build an unattached handler; call install() to start watching the logger tree."""
        super().__init__(level=logging.WARNING)
        self.events: list = []
        self._seen: set = set()
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            message = record.getMessage().lower()
        except Exception:
            return
        if any(p in message for p in self.PHRASES) or "killed by signal" in message:
            with self._lock:
                # The handler is attached at several points in the logger tree, so the same
                # record can arrive twice; count each distinct message once.
                key = (record.name, message)
                if key in self._seen:
                    return
                self._seen.add(key)
                if len(self.events) < 200:
                    self.events.append({"logger": record.name, "message": record.getMessage()[:300]})

    # `distributed` installs its own handlers and does not always propagate to root, so
    # attach to its loggers directly as well -- a watcher that sees nothing would turn
    # every failure into "inconclusive" and quietly destroy the classification.
    LOGGERS = ("", "distributed", "distributed.nanny", "distributed.worker", "distributed.worker.memory")

    def install(self) -> "NannyWatcher":
        """Attach to every logger in LOGGERS and return self, for chaining at construction."""
        for name in self.LOGGERS:
            logging.getLogger(name).addHandler(self)
        return self

    def uninstall(self) -> None:
        """Detach from every logger in LOGGERS."""
        for name in self.LOGGERS:
            logging.getLogger(name).removeHandler(self)


def _is_count(value) -> bool:
    """Report whether `value` is a non-negative integer byte count, excluding bool.

    `isinstance(True, int)` is True, so a bool would otherwise pass as a byte count and
    `True` would sum as 1. The negative test rejects `SpillSampler.UNREADABLE`.
    """
    return type(value) is int and value >= 0


class SpillSampler(threading.Thread):
    """Sample bytes resident in the workers' spill directories.

    `max_disk` is the largest CONCURRENT total seen across a 5 s sampling grid, not the
    cumulative bytes ever written: `SpillBuffer.spilled_total` falls again when a key is
    dropped (measured -- deleting one of four spilled 1 MB keys took the total from
    4000912 to 3000684 B).  A short spike between two samples is missed, so the figure is a
    LOWER BOUND on the true peak.

    `unmeasured` and `samples_ok` are the load-bearing fields.  A probe that cannot read the
    number must never report 0, because 0 is also a legitimate answer; see D-038 and D-041.
    `samples_ok` is what distinguishes "sampled every 5 s and never saw a byte on disk" from
    "never managed to sample at all" -- without it those two are the same `max_disk = 0`.

    A sample that reached only part of the cluster under-counts just as badly, so the expected
    width is taken from `n_workers` as REQUESTED at the command line and any sample of a
    different width latches `unmeasured`.  Note what `workers_expected` therefore is: the width
    every counted sample was REQUIRED to have, not a width observed and recorded.  It is
    reported as `spill_workers_sampled` because the latch makes the two equivalent for the
    samples that count -- a sample of any other width is not counted at all -- but no artefact
    holds an independently observed per-sample width.  It must not be learned from the cluster: a client
    that persistently answers from 1 of 4 workers would teach the sampler to expect 1 and then
    satisfy it on every sample, which is the defect this paragraph replaces.  For the same
    reason it must not be taken from `build_cluster`'s `n_workers`, which is itself
    `len(client.run(...))` and would inherit the same undercount one call earlier.

    `max_managed_worker` is the per-worker series `memory.target` actually thresholds on.
    `peak_cluster_bytes` is MemorySampler's cluster-SUMMED PROCESS series, so it cannot be
    compared against the per-worker managed fraction; recording both here makes that comparison
    possible without a second round trip.  Its `unmeasured` flag is independent: managed bytes
    failing to read never suppresses a disk measurement, and vice versa.

    `max_process_worker` is the THIRD quantity, and it is a different threshold, not a finer
    reading of the second.  `WorkerMemoryManager.memory_monitor` (distributed 2025.9.1,
    worker_memory.py:213) calls `worker.monitor.get_process_memory()` -- process RSS -- and
    compares `memory / self.memory_limit` against `memory.spill` (0.7) and `memory.pause` (0.8),
    while `memory.target` (0.6) compares the MANAGED bytes above.  A leg can therefore sit at
    44 % of target and still cross spill on unmanaged memory, so "the target path was never
    approached" does not imply "the leg never approached spilling".  This polled series is a
    5 s LOWER BOUND and exists as a cross-check; the primary per-worker RSS instrument is
    `harvest_process_history`, which reads dask's own 500 ms history at the end of the run.
    """

    UNREADABLE = -1  # a value is present in principle but could not be read

    def __init__(self, client, n_workers: int, interval: float = 5.0) -> None:
        """Sample every `interval` seconds once start()ed; call stop() to end the thread.

        `n_workers` is the REQUESTED worker count, not one derived from the cluster.
        """
        super().__init__(daemon=True)
        self.client = client
        self.interval = interval
        self.max_disk = 0
        self.unmeasured = False
        self.samples_ok = 0
        self.workers_expected = int(n_workers)
        self.max_managed_worker = 0
        self.max_managed_total = 0
        self.managed_unmeasured = False
        self.managed_samples_ok = 0
        self.max_process_worker = 0
        self.max_process_total = 0
        self.process_unmeasured = False
        self.process_samples_ok = 0
        # NOT `_stop`: that name shadows threading.Thread._stop(), which Thread.join()
        # calls internally, so join() would raise "'Event' object is not callable".
        self._stopped = threading.Event()

    @staticmethod
    def _probe(dask_worker):
        """`{"disk", "managed", "process"}` for this worker; any field may be UNREADABLE.

        `disk` is the bytes currently in the spill directory, `managed` the bytes zict is
        holding in memory -- the quantity `distributed.worker.memory.target` thresholds on,
        per worker -- and `process` this worker's RSS, the quantity
        `distributed.worker.memory.spill` and `.pause` threshold on.  Each carries its own
        UNREADABLE sentinel so that one being unreadable never contaminates the others.

        `worker.data` is a `SpillBuffer` when spilling is on, and its `spilled_total` is a
        `SpilledSize(memory, disk)` namedtuple whose `disk` field is the compressed size
        actually written.  With `--no-spill` it is a plain `dict` with no spill layer at all,
        and 0 is then the true answer rather than a failure to measure.

        Do NOT reach for `worker.data.disk.weight_by_key`: on distributed 2025.9.1
        `worker.data.disk` is a `zict.cache.Cache` and has no such attribute, so that
        expression raises AttributeError and a broad `except` turns it into a silent 0.  That
        is the defect that made every leg of the campaign report `spill 0.00 GB` (D-041).
        """
        unreadable = SpillSampler.UNREADABLE

        # RSS is read FIRST and unconditionally. It comes from the SystemMonitor, not from the
        # data store, so a worker handing us no store must still be able to report its process
        # memory: coupling them made an unreadable `data` silently unmeasure a quantity that was
        # perfectly readable (falsifier, 2026-09-08, finding 6).
        try:
            process = int(dask_worker.monitor.get_process_memory())
        except Exception:
            process = unreadable

        data = getattr(dask_worker, "data", None)
        if data is None:
            # Not "nothing spilled" -- we were handed no store to look at.
            return {"disk": unreadable, "managed": unreadable, "process": process}

        total = getattr(data, "spilled_total", None)
        if total is None:
            # A plain dict (`--no-spill`) has no spill layer, so 0 is the true answer. Anything
            # that DOES have a spill layer but no readable total is a failure, not a zero.
            disk = 0 if isinstance(data, dict) and not hasattr(data, "slow") else unreadable
        else:
            try:
                disk = int(getattr(total, "disk", total))
            except Exception:
                disk = unreadable

        # `data.fast` is the in-memory zict.lru.LRU whose `total_weight` is this worker's
        # managed bytes (verified on distributed 2025.9.1: 4 x 10 MB arrays over two workers
        # reported 20000000 and 40000000).  A plain dict has no `.fast`, and guessing a number
        # for it would be exactly the failure D-041 is about, so it reads UNREADABLE.
        try:
            managed = int(data.fast.total_weight)
        except Exception:
            managed = unreadable

        return {"disk": disk, "managed": managed, "process": process}

    def _latch_all(self) -> None:
        """Mark BOTH quantities unmeasured.

        For a failure of the round trip itself -- it raised, nobody answered, the sample was
        narrower or wider than the cluster, the replies were not readings -- no quantity was
        sampled. Latching only `unmeasured` here would leave the other series claiming a
        coverage the disk series had just rejected on identical evidence.
        """
        self.unmeasured = True
        self.managed_unmeasured = True
        self.process_unmeasured = True

    def run(self) -> None:  # noqa: D102
        while not self._stopped.wait(self.interval):
            try:
                per_worker = self.client.run(self._probe)
            except Exception:
                # A sampler that never sampled must not look like a sampler that saw zero.
                self._latch_all()
                continue
            if not per_worker:
                # No workers answered at all: sum(()) is 0 and would read as a real zero.
                self._latch_all()
                continue
            # A sample that reached only some of the workers under-counts the total exactly
            # like a sentinel does, and the expected width is fixed at construction from the
            # REQUESTED worker count, so a narrow sample cannot satisfy a narrowed expectation.
            if len(per_worker) != self.workers_expected:
                self._latch_all()
                continue
            values = list(per_worker.values())
            if not all(isinstance(v, dict) for v in values):
                # An older `_probe` returning a bare int, or an exception marshalled back as a
                # value. Either way this is not a reading.
                self._latch_all()
                continue

            disk = [v.get("disk") for v in values]
            if not all(_is_count(d) for d in disk):
                # Never sum a sentinel into a total: three workers at -1 and one at +3 would
                # cancel to 0, which is exactly the failure this class exists to prevent.
                self.unmeasured = True
            else:
                self.samples_ok += 1
                self.max_disk = max(self.max_disk, sum(disk))

            # Independent of the disk verdict above, and deliberately so: an unreadable
            # managed figure must not suppress a disk measurement, nor the reverse.
            managed = [v.get("managed") for v in values]
            if not all(_is_count(m) for m in managed):
                self.managed_unmeasured = True
            else:
                self.managed_samples_ok += 1
                self.max_managed_worker = max(self.max_managed_worker, max(managed))
                self.max_managed_total = max(self.max_managed_total, sum(managed))

            # Independent again, for the same reason: RSS is thresholded by `memory.spill`,
            # managed bytes by `memory.target`, and a failure to read one says nothing about
            # the other.  A build that reported one when it had only measured the other is the
            # category error D-042 was amended for.
            process = [v.get("process") for v in values]
            if not all(_is_count(p) for p in process):
                self.process_unmeasured = True
            else:
                self.process_samples_ok += 1
                self.max_process_worker = max(self.max_process_worker, max(process))
                self.max_process_total = max(self.max_process_total, sum(process))

    def stop(self) -> None:  # noqa: D102
        self._stopped.set()


def _prochist_dead() -> dict:
    """Return the process-history fields as they read when nothing was measured.

    One definition, used by `harvest_process_history` on every failure path AND by its caller
    when the harvest itself is abandoned, so that "we did not measure" cannot be written two
    ways -- one of which some later `_measured` clause forgets to reject.
    """
    return {
        "prochist_unmeasured": True,
        "prochist_samples_ok": 0,
        "prochist_workers_sampled": None,
        "prochist_max_worker_bytes": None,
        "prochist_max_worker_fraction": None,
        "prochist_per_worker_max_bytes": None,
        "prochist_worker_limit_bytes": None,
        "prochist_spill_fraction": None,
        "prochist_target_fraction": None,
        "prochist_pause_fraction": None,
        "prochist_covers_whole_run": None,
        "prochist_median_dt_s": None,
        "prochist_span_s": None,
        "prochist_span_s_source": None,
    }


def _worker_process_history(dask_worker):
    """Return this worker's own RSS history, plus the thresholds that history is compared against.

    `SystemMonitor.update` records `get_process_memory()` into `quantities["memory"]` on a
    PeriodicCallback driven by `distributed.admin.system-monitor.interval` (500 ms by default),
    keeping `...system-monitor.log-length` samples (7200, so 3600 s at that cadence).  Reading
    it once at the end of a run therefore yields the per-worker RSS series at dask's NATIVE
    resolution over the whole run -- the instrument `memory.spill` itself effectively uses --
    where a 5 s poller can only offer a lower bound between its samples.

    That holds only while `count <= maxlen`.  Past that the deque has wrapped and the series
    covers the tail of the run alone, so `count` and `maxlen` are both returned and the caller
    decides; inferring "whole run" from a full deque would be exactly the silent-undercount
    failure D-041 and D-042 are about.

    Returns `{"error": ...}` rather than a partial dict if either read fails: a missing series
    must never reach a summary as a zero.
    """
    out = {}
    try:
        monitor = dask_worker.monitor
        out["memory"] = [int(v) for v in monitor.quantities["memory"]]
        out["time"] = [float(v) for v in monitor.quantities["time"]]
        out["count"] = int(monitor.count)
        out["maxlen"] = None if monitor.maxlen is None else int(monitor.maxlen)
    except Exception as exc:
        return {"error": f"monitor unreadable: {type(exc).__name__}"}
    try:
        manager = dask_worker.memory_manager
        out["limit"] = int(manager.memory_limit)
        out["spill_fraction"] = manager.memory_spill_fraction
        out["target_fraction"] = manager.memory_target_fraction
        out["pause_fraction"] = manager.memory_pause_fraction
    except Exception as exc:
        return {"error": f"memory_manager unreadable: {type(exc).__name__}"}
    return out


def harvest_process_history(client, n_workers: int, outdir=None, label: str = "leg"):
    """Collect every worker's RSS history and reduce it to summary fields.

    `n_workers` is the REQUESTED count, for the same reason `SpillSampler` takes it: a reply
    from a strict subset of the cluster under-counts a maximum exactly like an unreadable
    field does, and a width learned from the reply can never fail its own check.

    Every failure path -- the round trip raising, a narrow or wide reply, a worker returning
    an error, an empty series -- yields `prochist_unmeasured True` and `None` byte counts.
    There is deliberately no partial answer: three workers' maxima with the fourth missing
    would print as a per-worker maximum while being a maximum over three quarters of a cluster.

    Returns `(fields, series)`; `series` maps worker address to `(times, rss_bytes)` and is
    written to `<label>_procseries.npz` when `outdir` is given.
    """
    import numpy as np

    dead = _prochist_dead()
    try:
        replies = client.run(_worker_process_history)
    except Exception:
        return dict(dead), {}
    if not replies or len(replies) != int(n_workers):
        return dict(dead, prochist_workers_sampled=len(replies) if replies else 0), {}
    if not all(isinstance(v, dict) and "error" not in v and v.get("memory") for v in replies.values()):
        return dict(dead, prochist_workers_sampled=len(replies)), {}
    per_worker_max = [max(v["memory"]) for v in replies.values()]
    limits = [v["limit"] for v in replies.values()]
    if not all(isinstance(m, int) and m >= 0 for m in per_worker_max) or not all(v > 0 for v in limits):
        return dict(dead, prochist_workers_sampled=len(replies)), {}

    # The fraction is per worker against ITS OWN limit, then maximised -- not the maximum RSS
    # over one worker's limit.  With equal limits the two agree; with unequal ones only the
    # former is the quantity `memory_monitor` computes.
    fractions = [m / lim for m, lim in zip(per_worker_max, limits)]
    dts = []
    for v in replies.values():
        times = v["time"]
        dts.extend(t2 - t1 for t1, t2 in zip(times, times[1:]))
    dts.sort()

    fields = {
        "prochist_unmeasured": False,
        "prochist_samples_ok": min(len(v["memory"]) for v in replies.values()),
        "prochist_workers_sampled": len(replies),
        "prochist_max_worker_bytes": max(per_worker_max),
        "prochist_max_worker_fraction": max(fractions),
        "prochist_per_worker_max_bytes": sorted(per_worker_max),
        "prochist_worker_limit_bytes": sorted({int(v) for v in limits}),
        "prochist_spill_fraction": sorted({v["spill_fraction"] for v in replies.values()}, key=str),
        "prochist_target_fraction": sorted({v["target_fraction"] for v in replies.values()}, key=str),
        "prochist_pause_fraction": sorted({v["pause_fraction"] for v in replies.values()}, key=str),
        # False means the deque wrapped: the maximum is then over the retained tail only, and
        # is a lower bound on the run's maximum rather than the run's maximum.
        "prochist_covers_whole_run": all(v["maxlen"] is None or v["count"] <= v["maxlen"] for v in replies.values()),
        "prochist_median_dt_s": (dts[len(dts) // 2] if dts else None),
        # `covers_whole_run` is `count <= maxlen`, and a worker RESTART resets `count`, so it
        # reads True over a post-restart TAIL. The span does not: it is the wall time the
        # narrowest worker's series actually covers, and a consumer compares it against
        # `elapsed_s` (falsifier, 2026-09-08, finding 13).
        "prochist_span_s": min(max(v["time"]) - min(v["time"]) for v in replies.values()),
        # Names where the span came from, and it is READ by report.py's licence rather than
        # left in a sibling key nothing consumes. A value derived after the fact from a
        # persisted artefact is defensible; a figure that cannot be told apart from one the
        # run itself wrote is not, because the distinction stops travelling with the row.
        "prochist_span_s_source": "run",
    }
    series = {addr: (np.asarray(v["time"], dtype=float), np.asarray(v["memory"], dtype=float)) for addr, v in replies.items()}
    if outdir is not None:
        flat = {}
        for i, (addr, (t, m)) in enumerate(sorted(series.items())):
            flat[f"w{i}_time"] = t
            flat[f"w{i}_rss"] = m
            flat[f"w{i}_addr"] = np.asarray([addr])
        np.savez(Path(outdir) / f"{label}_procseries.npz", **flat)
    return fields, series


class PersistAccountant:
    """Attribute every persisted byte to the marEx source line that requested it.

    Patches the same three entry points the unit tests patch, and additionally rebinds
    modules that did ``from dask import persist`` -- that import binds the *original*
    function into the importing namespace, so patching ``dask.persist`` alone
    under-counts without any sign that it has.
    """

    def __init__(self, marex_root: str, repo_root: str) -> None:
        """Track persisted bytes by source line; call install() to start patching."""
        self.marex_root = marex_root
        self.repo_root = repo_root
        self.by_site: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()
        self._orig: Dict[str, Any] = {}
        self._depth = threading.local()
        self._rebound: list = []

    def _site(self) -> str:
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_filename.startswith(self.marex_root):
                return f"{os.path.relpath(frame.f_code.co_filename, self.repo_root)}:{frame.f_lineno}"
            frame = frame.f_back
        return "<outside marEx>"

    @staticmethod
    def _nbytes(obj) -> int:
        total = 0
        for candidate in (obj,) if not isinstance(obj, (list, tuple)) else obj:
            if hasattr(candidate, "data_vars"):
                total += sum(int(getattr(v.data, "nbytes", 0) or 0) for v in candidate.data_vars.values())
                continue
            total += int(getattr(getattr(candidate, "data", candidate), "nbytes", 0) or 0)
        return total

    def _record(self, site: str, objs) -> None:
        nbytes = sum(self._nbytes(o) for o in objs)
        with self._lock:
            entry = self.by_site.setdefault(site, {"bytes": 0, "calls": 0})
            entry["bytes"] += nbytes
            entry["calls"] += 1

    def install(self) -> "PersistAccountant":
        """Patch the three persist entry points and rebind marEx modules' bare `persist` name."""
        import dask
        import xarray as _xr

        acct = self

        def wrap(orig, is_method):
            def inner(*args, **kwargs):
                depth = getattr(acct._depth, "n", 0)
                acct._depth.n = depth + 1
                try:
                    if depth == 0:
                        acct._record(acct._site(), args[:1] if is_method else args)
                    return orig(*args, **kwargs)
                finally:
                    acct._depth.n = depth

            return inner

        self._orig = {
            "dask.persist": dask.persist,
            "DataArray.persist": _xr.DataArray.persist,
            "Dataset.persist": _xr.Dataset.persist,
        }
        dask.persist = wrap(self._orig["dask.persist"], False)
        _xr.DataArray.persist = wrap(self._orig["DataArray.persist"], True)
        _xr.Dataset.persist = wrap(self._orig["Dataset.persist"], True)

        for modname, mod in list(sys.modules.items()):
            if modname.startswith("marEx") and getattr(mod, "persist", None) is self._orig["dask.persist"]:
                mod.persist = dask.persist
                self._rebound.append(modname)
        return self

    def uninstall(self) -> None:
        """Restore the three persist entry points and every rebound marEx module."""
        import dask
        import xarray as _xr

        if not self._orig:
            return
        dask.persist = self._orig["dask.persist"]
        _xr.DataArray.persist = self._orig["DataArray.persist"]
        _xr.Dataset.persist = self._orig["Dataset.persist"]
        for modname in self._rebound:
            sys.modules[modname].persist = self._orig["dask.persist"]

    def report(self) -> dict:
        """Return the by-site byte/call counts, ranked by bytes descending, plus totals."""
        ranked = dict(sorted(self.by_site.items(), key=lambda kv: -kv[1]["bytes"]))
        return {
            "by_site": ranked,
            "total_bytes": sum(v["bytes"] for v in ranked.values()),
            "total_calls": sum(v["calls"] for v in ranked.values()),
            "marex_bytes": sum(v["bytes"] for s, v in ranked.items() if s != "<outside marEx>"),
        }


class DeadlineExceeded(Exception):
    """The leg exceeded its self-imposed deadline. Says nothing about memory."""


@contextmanager
def deadline(seconds: int):
    """Raise DeadlineExceeded via SIGALRM after `seconds`; a no-op context if seconds is 0."""
    if not seconds:
        yield
        return

    def _fire(signum, frame):
        raise DeadlineExceeded(f"self-imposed deadline of {seconds} s exceeded")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def classify_failure(exc: BaseException, watcher: NannyWatcher) -> str:
    """Name the failure from evidence. 'inconclusive' is a legitimate, reportable answer."""
    from distributed.scheduler import KilledWorker

    memory_evidence = bool(watcher.events)
    if isinstance(exc, DeadlineExceeded):
        return "timeout_with_memory_pressure" if memory_evidence else "timeout_inconclusive"
    if isinstance(exc, (KilledWorker, MemoryError)):
        return "oom"
    text = f"{type(exc).__name__}: {exc}".lower()
    if "killedworker" in text or "memory" in text or "worker process died" in text:
        return "oom"
    if memory_evidence:
        return "oom_probable"
    return "error_unrelated"


# --------------------------------------------------------------------------------------
# How many times did the graph actually run?
# --------------------------------------------------------------------------------------
# `lazy` mode's whole cost model is "the anomaly graph is re-executed once per consumer".
# Nothing in this harness could count that: bytes pinned prove laziness, wall clock is a
# proxy that mixes in scheduling and I/O, and neither is the number the claim states.
#
# `Scheduler.task_prefixes[prefix].state_counts` is that number.  It is CUMULATIVE and it
# outlives the tasks: a prefix keeps counting after its tasks are released and forgotten,
# so the `memory` entry is "how many tasks under this prefix ever completed", summed over
# every `.compute()` in the leg.  Running the same graph three times triples it (verified:
# 16 -> 48 for a `sum` prefix across three computes on a two-worker LocalCluster).
#
# Read the RATIO between modes on a prefix that all three modes share -- the ones that read
# the input store -- not the absolute total: `streaming` adds zarr write/read prefixes that
# `persist` does not have, so the totals are not comparable across modes and the per-prefix
# counts are.
def _scheduler_task_prefix_counts(dask_scheduler):
    """Cumulative completed-task count per prefix, read on the scheduler."""
    out = {}
    for name, prefix in dask_scheduler.task_prefixes.items():
        counts = dict(getattr(prefix, "state_counts", {}) or {})
        out[str(name)] = {str(k): int(v) for k, v in counts.items()}
    return out


def harvest_task_prefix_counts(client) -> dict:
    """Harvest the per-prefix cumulative completion counts, or say it was not measured.

    Returns the `taskcount_*` block for the summary.  A build without the instrument, a
    scheduler that does not answer, and a scheduler that answers with nothing all read
    UNMEASURED -- an empty answer is not a zero, exactly as with the spill sampler.
    """
    dead = {
        "taskcount_by_prefix": None,
        "taskcount_total_completed": None,
        "taskcount_n_prefixes": 0,
        "taskcount_unmeasured": True,
        "taskcount_error": None,
    }
    try:
        by_prefix = client.run_on_scheduler(_scheduler_task_prefix_counts)
    except Exception as exc:  # noqa: BLE001 - a failure to measure is a reportable outcome
        dead["taskcount_error"] = f"{type(exc).__name__}: {exc}"[:400]
        return dead
    if not isinstance(by_prefix, dict) or not by_prefix:
        dead["taskcount_error"] = f"scheduler returned {type(by_prefix).__name__} of length 0"
        return dead
    completed = {k: int(v.get("memory", 0)) for k, v in by_prefix.items()}
    if not any(completed.values()):
        # Every prefix at zero means the counter never incremented; that is the broken-probe
        # shape (D-038), not a run in which no task completed.
        dead["taskcount_by_prefix"] = completed
        dead["taskcount_n_prefixes"] = len(completed)
        dead["taskcount_error"] = "every prefix reports 0 completions"
        return dead
    return {
        "taskcount_by_prefix": completed,
        "taskcount_total_completed": int(sum(completed.values())),
        "taskcount_n_prefixes": len(completed),
        "taskcount_unmeasured": False,
        "taskcount_error": None,
    }


# --------------------------------------------------------------------------------------
# The leg runner
# --------------------------------------------------------------------------------------
def execute(args, meta: dict, work: Callable[[Any], dict]) -> dict:
    """Run one leg end to end, writing a summary whatever happens.

    ``work(client)`` does the actual marEx call and returns any extra fields (event
    counts, output fingerprints) to fold into the summary.
    """
    import numpy as np
    from distributed.diagnostics import MemorySampler

    import marEx

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / f"{args.label}_summary.json"

    marex_root = os.path.dirname(os.path.abspath(marEx.__file__))
    repo_root = os.path.dirname(marex_root)

    summary = {
        "label": args.label,
        "status": "started",
        "outcome": None,
        "compute_mode": args.mode,
        "marex_file": marEx.__file__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        **meta,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    client, cluster_meta = build_cluster(args)
    summary.update(cluster_meta)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    watcher = NannyWatcher().install()
    accountant = PersistAccountant(marex_root, repo_root).install()
    spill = SpillSampler(client, n_workers=args.workers)
    spill.start()
    sampler = MemorySampler()

    extra: dict = {}
    outcome, error = "completed", None
    start = time.perf_counter()
    try:
        with deadline(args.deadline):
            with sampler.sample(args.label):
                extra = work(client) or {}
    except BaseException as exc:  # noqa: BLE001, B036 - the failure IS the measurement here
        outcome = classify_failure(exc, watcher)
        error = f"{type(exc).__name__}: {exc}"[:4000]
        print(f"[{args.label}] FAILED ({outcome}): {error}", flush=True)
    finally:
        elapsed = time.perf_counter() - start
        spill.stop()
        accountant.uninstall()
        watcher.uninstall()

    try:
        series = np.asarray(sampler.to_pandas()).astype(float).ravel()
        peak, mean = float(series.max()), float(series.mean())
        np.save(outdir / f"{args.label}_memseries.npy", series)
    except Exception:
        peak = mean = float("nan")

    # Bounded: a worker that died mid-run can leave `client.run` waiting forever, and an
    # instrument must never be able to cost the leg its wall clock.  A timeout is a failure to
    # measure, which is what `prochist_unmeasured` says.
    try:
        with deadline(60):
            prochist, _ = harvest_process_history(client, args.workers, outdir=outdir, label=args.label)
    except (Exception, DeadlineExceeded):
        # Deliberately NOT BaseException: a KeyboardInterrupt or SystemExit landing inside this
        # 60 s window would otherwise be swallowed and the leg written out as "finished".
        prochist = _prochist_dead()

    # Same bound, same reasoning: an instrument must never cost the leg its wall clock.
    try:
        with deadline(60):
            taskcounts = harvest_task_prefix_counts(client)
    except (Exception, DeadlineExceeded):
        taskcounts = {
            "taskcount_by_prefix": None,
            "taskcount_total_completed": None,
            "taskcount_n_prefixes": 0,
            "taskcount_unmeasured": True,
            "taskcount_error": "harvest exceeded its 60 s bound",
        }

    summary.update(
        status="finished",
        outcome=outcome,
        error=error,
        elapsed_s=elapsed,
        peak_cluster_bytes=peak,
        mean_cluster_bytes=mean,
        spill_max_disk_bytes=None if (spill.unmeasured or spill.samples_ok == 0) else spill.max_disk,
        spill_unmeasured=bool(spill.unmeasured or spill.samples_ok == 0),
        spill_samples_ok=spill.samples_ok,
        spill_workers_sampled=spill.workers_expected,
        managed_max_worker_bytes=None if (spill.managed_unmeasured or spill.managed_samples_ok == 0) else spill.max_managed_worker,
        managed_max_total_bytes=None if (spill.managed_unmeasured or spill.managed_samples_ok == 0) else spill.max_managed_total,
        managed_unmeasured=bool(spill.managed_unmeasured or spill.managed_samples_ok == 0),
        managed_samples_ok=spill.managed_samples_ok,
        process_max_worker_bytes=None if (spill.process_unmeasured or spill.process_samples_ok == 0) else spill.max_process_worker,
        process_max_total_bytes=None if (spill.process_unmeasured or spill.process_samples_ok == 0) else spill.max_process_total,
        process_unmeasured=bool(spill.process_unmeasured or spill.process_samples_ok == 0),
        process_samples_ok=spill.process_samples_ok,
        **prochist,
        **taskcounts,
        nanny_memory_events=len(watcher.events),
        nanny_memory_event_sample=watcher.events[:10],
        persist=accountant.report(),
        **extra,
    )
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    managed_dead = spill.managed_unmeasured or spill.managed_samples_ok == 0
    managed_text = "UNMEASURED" if managed_dead else f"{spill.max_managed_worker / GB:.2f} GB"
    process_dead = spill.process_unmeasured or spill.process_samples_ok == 0
    polled_text = "UNMEASURED" if process_dead else f"{spill.max_process_worker / GB:.2f} GB"
    if prochist["prochist_unmeasured"]:
        proc_text = "UNMEASURED"
    else:
        proc_text = (
            f"{prochist['prochist_max_worker_bytes'] / GB:.2f} GB "
            f"({100 * prochist['prochist_max_worker_fraction']:.1f}% of limit, "
            f"spill at {prochist['prochist_spill_fraction']}, "
            f"whole-run={prochist['prochist_covers_whole_run']}, "
            f"span {prochist['prochist_span_s']:.0f} s vs wall {elapsed:.0f} s, "
            f"n={prochist['prochist_samples_ok']} @ {prochist['prochist_median_dt_s']:.2f} s)"
        )
    if taskcounts["taskcount_unmeasured"]:
        taskcount_text = f"UNMEASURED ({taskcounts['taskcount_error']})"
    else:
        taskcount_text = f"{taskcounts['taskcount_total_completed']} over {taskcounts['taskcount_n_prefixes']} prefixes"
    print(
        f"[{args.label}] {outcome}  wall {elapsed:.1f} s  peak {peak / GB:.1f} GB  "
        f"pinned {accountant.report()['total_bytes'] / GB:.3f} GB  "
        f"spill {'UNMEASURED' if (spill.unmeasured or spill.samples_ok == 0) else f'{spill.max_disk / GB:.2f} GB'}"
        f" (n={spill.samples_ok}/{spill.workers_expected}w)  "
        f"managed/worker {managed_text}"
        f" (n={spill.managed_samples_ok})  "
        f"rss/worker {proc_text}  "
        f"rss/worker-polled {polled_text}"
        f" (n={spill.process_samples_ok})  "
        f"tasks-completed {taskcount_text}  "
        f"nanny-memory-events {len(watcher.events)}",
        flush=True,
    )
    try:
        client.close()
    except Exception:
        pass
    return summary
