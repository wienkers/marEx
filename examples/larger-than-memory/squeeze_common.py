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
    # stays resident.  On the unstructured tracker the same configuration that completes in
    # 54 min at 116.8 GB peak with spilling on was killed at 169.5 GB peak with it off.  Both
    # modes then fail, and a leg where neither mode completes proves nothing about either.
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


class SpillSampler(threading.Thread):
    """Sample bytes spilled to disk. Zero spill plus a low peak is the strong result."""

    def __init__(self, client, interval: float = 5.0) -> None:
        """Sample every `interval` seconds once start()ed; call stop() to end the thread."""
        super().__init__(daemon=True)
        self.client = client
        self.interval = interval
        self.max_disk = 0
        self._stop = threading.Event()

    @staticmethod
    def _probe(dask_worker):
        disk = getattr(getattr(dask_worker, "data", None), "disk", None)
        if disk is None:
            return 0
        try:
            return int(sum(disk.weight_by_key.values()))
        except Exception:
            return 0

    def run(self) -> None:  # noqa: D102
        while not self._stop.wait(self.interval):
            try:
                per_worker = self.client.run(self._probe)
                self.max_disk = max(self.max_disk, sum(int(v) for v in per_worker.values() if isinstance(v, (int, float))))
            except Exception:
                continue

    def stop(self) -> None:  # noqa: D102
        self._stop.set()


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
    spill = SpillSampler(client)
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

    summary.update(
        status="finished",
        outcome=outcome,
        error=error,
        elapsed_s=elapsed,
        peak_cluster_bytes=peak,
        mean_cluster_bytes=mean,
        spill_max_disk_bytes=spill.max_disk,
        nanny_memory_events=len(watcher.events),
        nanny_memory_event_sample=watcher.events[:10],
        persist=accountant.report(),
        **extra,
    )
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(
        f"[{args.label}] {outcome}  wall {elapsed:.1f} s  peak {peak / GB:.1f} GB  "
        f"pinned {accountant.report()['total_bytes'] / GB:.3f} GB  spill {spill.max_disk / GB:.2f} GB  "
        f"nanny-memory-events {len(watcher.events)}",
        flush=True,
    )
    try:
        client.close()
    except Exception:
        pass
    return summary
