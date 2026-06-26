"""
Wall time and memory of a Dask workload, stage by stage.

:class:`ResourceMonitor` wraps blocks of work and records, for each block, the wall time,
the peak memory of the worker processes and of the client, the peak bytes spilled to disk
and the number of worker restarts. A background thread polls the scheduler, so nothing is
added to the task graph.

>>> monitor = ResourceMonitor(client)
>>> with monitor.stage("detect"):
...     extremes_ds.to_zarr(path, mode="w")
>>> monitor.summary()
"""

import os
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

import pandas as pd
import psutil
from dask.distributed import Client

from ..logging_config import get_logger

logger = get_logger(__name__)

_GB = 1e9


def _parse_slurm_mem(value: str) -> Optional[float]:
    """SLURM memory string to bytes. SLURM's M/G/T are binary units, and a bare number is MiB."""
    units = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    value = value.strip().upper()
    if value and value[-1] in units:
        number, scale = value[:-1], units[value[-1]]
    else:
        number, scale = value, units["M"]
    try:
        return float(number) * scale
    except ValueError:
        return None


def _job_memory_bytes() -> Optional[float]:
    """Memory this job may use: the cgroup limit if one is set, else SLURM's grant, else None."""
    try:
        with open("/proc/self/cgroup") as fh:
            rel = next(line.strip().split(":", 2)[2] for line in fh if line.startswith("0::"))
        path = "/sys/fs/cgroup" + rel
        while path.startswith("/sys/fs/cgroup") and len(path) > len("/sys/fs/cgroup"):
            limit_file = os.path.join(path, "memory.max")
            if os.path.exists(limit_file):
                with open(limit_file) as fh:
                    limit = fh.read().strip()
                if limit != "max":
                    return float(limit)
            path = os.path.dirname(path)
    except (OSError, StopIteration, ValueError):
        pass
    if os.environ.get("SLURM_MEM_PER_NODE"):
        return _parse_slurm_mem(os.environ["SLURM_MEM_PER_NODE"])
    if os.environ.get("SLURM_MEM_PER_CPU"):
        per_cpu = _parse_slurm_mem(os.environ["SLURM_MEM_PER_CPU"])
        return per_cpu * float(os.environ.get("SLURM_CPUS_ON_NODE", 1)) if per_cpu else None
    return None


class ResourceMonitor:
    """
    Record wall time and peak memory for named stages of a Dask workload.

    Parameters
    ----------
    client : dask.distributed.Client
        Client of the cluster doing the work.
    interval : float, default=1.0
        Seconds between samples. The peaks are the largest *sampled* values, so a spike
        shorter than this (for example the one that makes the nanny restart a worker) can
        be missed.

    Notes
    -----
    Worker memory is the summed process memory the workers report to the scheduler
    (including unmanaged memory), which is what the nanny compares against
    ``memory_limit``. Client memory is the resident set of this Python process.
    A worker restart shows up as a worker address the monitor has not seen before.

    On construction the monitor logs each worker's effective ``memory_limit``. On HPC
    nodes Dask reads the whole node's memory rather than the job's allocation, so a
    cluster started without an explicit ``memory_limit`` can over-commit; the monitor
    warns when the limits add up to more than the job may use (its cgroup limit, or SLURM's
    grant when no cgroup limit is visible). Spill is the growth of bytes spilled to disk
    during the stage, so data left on disk by an earlier stage does not count.
    """

    def __init__(self, client: Client, interval: float = 1.0) -> None:
        """Read the cluster's shape and memory limits, and warn if they exceed the job's memory."""
        self.client = client
        self.interval = interval
        self._records: List[Dict[str, object]] = []
        self._process = psutil.Process()

        workers = self._workers()
        self.n_workers = len(workers)
        self.threads = sum(w.get("nthreads", 0) for w in workers.values())
        limits = [w.get("memory_limit") or 0 for w in workers.values()]
        self.worker_memory_limit = max(limits) if limits else 0
        self.memory_limit_total = float(sum(limits))
        self.job_memory = _job_memory_bytes()

        logger.info(
            f"Cluster: {self.n_workers} workers, {self.threads} threads, "
            f"memory_limit per worker {sorted({round(v / _GB, 1) for v in limits})} GB "
            f"({self.memory_limit_total / _GB:.0f} GB total)"
        )
        if any(v == 0 for v in limits):
            logger.warning("At least one worker has no memory_limit; Dask will not pause or spill it")
        if self.job_memory and self.memory_limit_total > self.job_memory:
            logger.warning(
                f"Worker memory limits total {self.memory_limit_total / _GB:.0f} GB but the job may use "
                f"{self.job_memory / _GB:.0f} GB: pass memory_limit explicitly when starting the cluster"
            )

    def _workers(self) -> Dict[str, dict]:
        return self.client.scheduler_info(n_workers=-1)["workers"]

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure the enclosed block as one stage called ``name``."""
        peak = {"workers": 0.0, "client": 0.0, "spilled": 0.0}
        start_workers = self._workers()
        seen = set(start_workers)
        spilled_at_start = float(
            sum((w.get("metrics", {}).get("spilled_bytes") or {}).get("disk", 0) for w in start_workers.values())
        )
        n_start = len(seen)
        stop = threading.Event()

        def sample() -> None:
            try:
                workers = self._workers()
            except Exception:  # scheduler busy or closing; skip this sample
                return
            seen.update(workers)
            metrics = [w.get("metrics", {}) for w in workers.values()]
            peak["workers"] = max(peak["workers"], float(sum(m.get("memory", 0) for m in metrics)))
            peak["spilled"] = max(
                peak["spilled"],
                float(sum((m.get("spilled_bytes") or {}).get("disk", 0) for m in metrics)),
            )
            peak["client"] = max(peak["client"], float(self._process.memory_info().rss))

        def run() -> None:
            while not stop.wait(self.interval):
                sample()

        sample()
        thread = threading.Thread(target=run, name=f"ResourceMonitor[{name}]", daemon=True)
        t0 = time.perf_counter()
        thread.start()
        try:
            yield
        finally:
            wall = time.perf_counter() - t0
            stop.set()
            thread.join()
            sample()
            record = {
                "stage": name,
                "wall (min)": wall / 60,
                "peak worker memory (GB)": peak["workers"] / _GB,
                "peak client memory (GB)": peak["client"] / _GB,
                "peak spilled to disk (GB)": max(peak["spilled"] - spilled_at_start, 0.0) / _GB,
                "worker restarts": len(seen) - n_start,
            }
            self._records.append(record)
            print(
                f"{name}: {record['wall (min)']:.1f} min, peak worker memory "
                f"{record['peak worker memory (GB)']:.1f} GB of {self.memory_limit_total / _GB:.0f} GB, "
                f"client {record['peak client memory (GB)']:.1f} GB, spilled "
                f"{record['peak spilled to disk (GB)']:.1f} GB, restarts {record['worker restarts']}"
            )

    def summary(self) -> pd.DataFrame:
        """One row per stage, plus a total row when there is more than one stage."""
        df = pd.DataFrame(self._records)
        if len(df) > 1:
            total = {
                "stage": "total",
                "wall (min)": df["wall (min)"].sum(),
                "peak worker memory (GB)": df["peak worker memory (GB)"].max(),
                "peak client memory (GB)": df["peak client memory (GB)"].max(),
                "peak spilled to disk (GB)": df["peak spilled to disk (GB)"].max(),
                "worker restarts": df["worker restarts"].sum(),
            }
            df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
        return df.set_index("stage").round(2) if len(df) else df
