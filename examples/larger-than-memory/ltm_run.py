#!/usr/bin/env python
"""
Run one rung of the larger-than-memory ladder: detect or track on the first `--years` of the 6-hourly SST record, inside
whatever memory the current job was given, and write a JSON summary of what it cost.

Memory is measured from the job's own control group (Linux cgroup v2), not from Dask: `anon` is the memory the job's
processes actually hold, summed over the client, the workers and everything else in the job. File cache is reported
separately and is not counted, because reading a store larger than memory fills the page cache to the limit whatever
the code does, and the kernel reclaims it on demand. `oom_kill` counts the kernel's out-of-memory kills in the job.

A summary is written before the work starts and again at the end, so a run that is killed still leaves a record.

Usage (inside a job, see ltm.sbatch):
    python ltm_run.py detect --years 10 --mode streaming
    python ltm_run.py track  --years 10 --mode streaming
    SMOKE=1 python ltm_run.py detect --years 2     # plumbing check: 2 years, 1-year baseline
"""

import argparse
import json
import os
import socket
import threading
import time
import traceback
from pathlib import Path

import dask
import pandas as pd
import xarray as xr

import marEx
import marEx.helper as hpc

SMOKE = os.environ.get("SMOKE") == "1"
SCRATCH = Path(os.environ.get("MAREX_SCRATCH", Path("/scratch") / os.environ["USER"][0] / os.environ["USER"]))
STORE = Path(
    os.environ.get("MAREX_LTM_STORE", SCRATCH / "marEx_data" / "input" / "fesom_sst6h" / "fesom_sst_6hourly_1971-2050.zarr")
)
START = "1971-01-01"


def parse_args():
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["detect", "track"])
    p.add_argument("--years", type=int, required=True, help="length of the input record, from --start")
    p.add_argument("--start", default=START, help="first day of the input record")
    p.add_argument("--mode", choices=["streaming", "persist"], default="streaming", help="compute_mode")
    p.add_argument("--window-years", type=int, default=5, help="shifting-baseline length (detect)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--threads", type=int, default=2, help="threads per worker")
    p.add_argument("--memory-limit", default="9GB", help="memory limit per worker")
    p.add_argument(
        "--lat-chunk", type=int, default=103, help="latitude rows per detect input chunk (721 = 7 x 103); 0 keeps space whole"
    )
    p.add_argument(
        "--stall-minutes", type=float, default=0, help="stop the run if the scheduler makes no progress for this long (0: never)"
    )
    p.add_argument("--out-dir", type=Path, default=SCRATCH / "marEx" / "ltm")
    p.add_argument("--label", default=None, help="name of this run (default: <stage>_<years>y_<mode>)")
    args = p.parse_args()
    if SMOKE:
        args.window_years = 1
        args.out_dir = args.out_dir / "smoke"
    args.label = args.label or f"{args.stage}_{args.years}y_{args.mode}"
    return args


class CgroupSampler:
    """Sample the job's cgroup v2 memory counters in a background thread and keep the peaks."""

    def __init__(self, interval=2.0):
        """Find the job's cgroup and prepare the sampling thread (``interval`` seconds between samples)."""
        rel = open("/proc/self/cgroup").read().strip().split("::", 1)[1]
        path = Path("/sys/fs/cgroup" + rel)
        while path.name and not path.name.startswith("job_"):  # climb from the step/task to the whole job
            path = path.parent
        self.path = path if path.name else Path("/sys/fs/cgroup" + rel)
        self.interval = interval
        self.peak = {"anon": 0, "file": 0, "current": 0}
        self.series = []  # (seconds since start, anon bytes)
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read(self):
        stat = dict(line.split() for line in open(self.path / "memory.stat"))
        current = int(open(self.path / "memory.current").read())
        return int(stat["anon"]), int(stat["file"]), current

    def _run(self):
        while not self._stop.is_set():
            anon, file, current = self._read()
            for key, value in (("anon", anon), ("file", file), ("current", current)):
                self.peak[key] = max(self.peak[key], value)
            self.series.append((round(time.time() - self._t0, 1), anon))
            self._stop.wait(self.interval)

    def start(self):
        """Start sampling in the background."""
        self._thread.start()
        return self

    def stop(self):
        """Stop sampling and wait for the thread."""
        self._stop.set()
        self._thread.join()

    def limit(self):
        """Return the job's memory limit in bytes (None if unlimited)."""
        raw = open(self.path / "memory.max").read().strip()
        return None if raw == "max" else int(raw)

    def oom_kills(self):
        """OOM kills in the job so far (cumulative over the cgroup's life: report a difference)."""
        events = dict(line.split() for line in open(self.path / "memory.events"))
        return int(events.get("oom_kill", 0))


def _scheduler_state(dask_scheduler):
    """Tasks held, transitions so far, and per worker: status, tasks processing and executing, process/managed GB."""
    workers = [
        (ws.status.name, len(ws.processing), len(ws.executing), ws.memory.process / 1e9, ws.memory.managed / 1e9)
        for ws in dask_scheduler.workers.values()
    ]
    return len(dask_scheduler.tasks), dask_scheduler.transition_counter, workers


def heartbeat(client, t0, stop, stall_minutes=0, on_stall=None, interval=60):
    """
    Log one progress line a minute: tasks the scheduler holds, and each worker's status, load and memory.

    A worker that stays above Dask's pause threshold stops taking work without logging anything at marEx's log level,
    so the status column is what tells a paused cluster from a slow one. With `stall_minutes`, a run whose scheduler
    makes no transition for that long prints the workers' call stacks and calls `on_stall`.
    """
    import psutil

    me = psutil.Process()
    last_count, last_change = None, time.time()
    while not stop.wait(interval):
        try:
            n_tasks, count, workers = client.run_on_scheduler(_scheduler_state)
            spilled = sum(w["metrics"].get("spilled_bytes", {}).get("disk", 0) for w in client.scheduler_info()["workers"].values())
            per_worker = " ".join(f"{s[:5]} {p}/{e} {m:.1f}/{g:.1f}" for s, p, e, m, g in workers)
            print(
                f"[heartbeat {time.time() - t0:7.0f} s] tasks {n_tasks} transitions {count} | workers "
                f"(status processing/executing process/managed GB) {per_worker} | spilled {spilled / 1e9:.1f} GB | "
                f"client {me.memory_info().rss / 1e9:.1f} GB, cpu {me.cpu_percent():.0f}%",
                flush=True,
            )
            if count != last_count:
                last_count, last_change = count, time.time()
            elif stall_minutes and time.time() - last_change > stall_minutes * 60:
                print(f"[stall] no scheduler transition for {stall_minutes} min; worker call stacks:", flush=True)
                print(json.dumps(client.call_stack(), indent=1, default=str)[:200_000], flush=True)
                on_stall(workers)
                return
        except Exception as exc:  # a heartbeat must never stop the run
            print(f"[heartbeat] {type(exc).__name__}: {exc}", flush=True)


def write(summary, path):
    """Write the summary atomically."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=1, default=str))
    tmp.replace(path)


def open_sst(start, years, lat_chunk=0):
    """Open `years` of the record from `start` in 10-day chunks, in latitude bands of `lat_chunk` rows (0: space whole).

    Returns the field on a regular 6-hourly axis and the number of steps that had to be filled in.
    """
    sst = xr.open_zarr(STORE, chunks={"time": 40, "lat": -1, "lon": -1}).sst
    if SMOKE:
        sst = sst.isel(lat=slice(309, 412))  # plumbing check: one latitude band (12.75 S to 12.75 N)
    if lat_chunk:
        # A local split of each stored chunk: every chunk is still read once. At 6-hourly cadence the climatology has
        # 1464 hour-of-year slots, so one whole-globe tile of it is ~6 GB; latitude bands keep each tile under 1 GB.
        sst = sst.chunk({"lat": lat_chunk})
    steps = pd.date_range(start, pd.Timestamp(start) + pd.DateOffset(years=years), freq="6h", inclusive="left")
    sst = sst.sel(time=slice(steps[0], steps[-1]))
    # The record has a few missing steps (it starts at 06:00, and three 00:00 steps are absent). marEx needs every
    # ocean cell finite at every step, so each missing step is the mean of its 6-hourly neighbours (the next step
    # alone at the start of the record). Each one is its own piece between untouched slices of the store.
    have = pd.DatetimeIndex(sst.time.values)
    missing = steps.difference(have)
    if len(missing):
        six = pd.Timedelta("6h")
        pieces, after = [], None
        for t in missing:
            near = [n for n in (t - six, t + six) if n in have]
            pieces.append(sst.sel(time=slice(after, t - six)))
            pieces.append((sum(sst.sel(time=n, drop=True) for n in near) / len(near)).expand_dims(time=[t]))
            after = t + six
        pieces.append(sst.sel(time=slice(after, None)))
        sst = xr.concat([p for p in pieces if p.sizes["time"]], "time").chunk({"time": 40})
    assert (sst.time.values == steps.values).all() and set(sst.chunks[0][:-1]) <= {40}, sst.chunks[0]
    return sst, len(missing)


def run_detect(args, monitor, staging):
    """Run shifting baseline + hour-of-year percentile on the 6-hourly record."""
    sst, filled = open_sst(args.start, args.years, args.lat_chunk)
    out = args.out_dir / f"extremes_{args.years}y_{args.mode}.zarr"
    with monitor.stage("detect"):
        ds = marEx.preprocess_data(
            sst,
            method_anomaly="shifting_baseline",
            method_extreme="seasonal_percentile",
            threshold_percentile=95,
            window_years=args.window_years,
            smooth_days=21,
            window_days=11,
            dimensions={"time": "time", "x": "lon", "y": "lat"},
            dask_chunks={"time": 40},
            compute_mode=args.mode,
            scratch_dir=str(staging),
        )
        ds[["extreme_events", "mask"]].to_zarr(out, mode="w")
    if args.mode == "streaming":
        marEx.clear_staging(ds)
    res = xr.open_zarr(out)
    return {
        "input_steps": int(sst.sizes["time"]),
        "filled_steps": filled,
        "input_bytes": int(sst.size * 4),
        "output_steps": int(res.sizes["time"]),
        "extreme_cell_steps": int(res.extreme_events.sum().compute()),
        "output": str(out),
    }


def run_track(args, monitor, staging):
    """Run the merging tracker on the detect output of the same rung."""
    src = args.out_dir / f"extremes_{args.years}y_streaming.zarr"
    ds = xr.open_zarr(src, chunks={"time": 40, "lat": -1, "lon": -1})
    out = args.out_dir / f"events_{args.years}y_{args.mode}.zarr"
    mask = ds.mask.where(ds.lat < 85, other=False)  # the grid's anisotropy near the pole biases the tracking
    with monitor.stage("track"):
        tracker = marEx.tracker(
            ds.extreme_events,
            mask,
            grid_resolution=0.25,
            area_filter_absolute=600,
            R_fill=12,
            T_fill=16,  # timesteps: bridges gaps of up to 4 days at 6-hourly cadence
            allow_merging=True,
            overlap_threshold=0.25,
            nn_partitioning=True,
            compute_mode=args.mode,
            temp_dir=str(staging),
        )
        events, merges = tracker.run(return_merges=True)
        events.to_zarr(out, mode="w")
        merges.to_netcdf(out.with_suffix(".merges.nc"), mode="w")
    if args.mode == "streaming":
        marEx.clear_staging(events)
    res = xr.open_zarr(out)
    return {
        "input_steps": int(ds.sizes["time"]),
        "id_field_bytes": int(ds.extreme_events.size * 4),
        "n_events": int(res.attrs.get("N_events_final", -1)),
        "n_merges": int(merges.sizes.get("merge_ID", 0)),
        "output": str(out),
    }


def main():
    """Run one rung and record its cost."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    staging = args.out_dir / "staging" / args.label
    staging.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.label}.json"

    sampler = CgroupSampler().start()
    summary = {
        "label": args.label,
        "stage": args.stage,
        "years": args.years,
        "mode": args.mode,
        "smoke": SMOKE,
        "host": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cgroup": str(sampler.path),
        "job_memory_limit": sampler.limit(),
        "cluster": {"workers": args.workers, "threads": args.threads, "memory_limit": args.memory_limit},
        "outcome": "started",
        "lat_chunk": args.lat_chunk,
    }
    oom_at_start = sampler.oom_kills()
    write(summary, path)

    dask.config.set({"distributed.admin.large-graph-warning-threshold": "1GB"})
    client = hpc.start_local_cluster(
        n_workers=args.workers,
        threads_per_worker=args.threads,
        memory_limit=args.memory_limit,
        scratch_dir=args.out_dir / "dask",
    )
    # Dask does not see the job's limit, so check that every worker got the limit asked for
    limits = sorted({w["memory_limit"] for w in client.scheduler_info()["workers"].values()})
    summary["worker_memory_limits"] = limits
    assert len(limits) == 1 and limits[0] == dask.utils.parse_bytes(args.memory_limit), limits
    monitor = hpc.ResourceMonitor(client)

    t0 = time.time()
    stop = threading.Event()
    finished = threading.Lock()

    def finish():
        """Record wall time, cgroup peaks and the Dask monitor, and write the summary (once, from either thread)."""
        if not finished.acquire(blocking=False):
            return
        stop.set()
        summary["wall_s"] = round(time.time() - t0, 1)
        sampler.stop()
        summary["peak_anon_bytes"] = sampler.peak["anon"]
        summary["peak_file_cache_bytes"] = sampler.peak["file"]
        summary["peak_current_bytes"] = sampler.peak["current"]
        summary["oom_kills"] = sampler.oom_kills() - oom_at_start
        summary["anon_series"] = sampler.series[:: max(1, len(sampler.series) // 2000)]
        summary["monitor"] = monitor.summary().reset_index().to_dict(orient="records")
        write(summary, path)
        print(json.dumps({k: v for k, v in summary.items() if k not in ("anon_series", "traceback")}, indent=1, default=str))

    def on_stall(workers):
        """Finish from this thread and exit: the main thread is stuck waiting on Dask."""
        summary["outcome"] = f"stalled: no scheduler transition for {args.stall_minutes} min"
        summary["workers_at_stall"] = workers
        finish()
        os._exit(3)

    threading.Thread(target=heartbeat, args=(client, t0, stop, args.stall_minutes, on_stall), daemon=True).start()
    try:
        summary["result"] = (run_detect if args.stage == "detect" else run_track)(args, monitor, staging)
        summary["outcome"] = "completed"
    except Exception as exc:  # recorded, then re-raised so the job fails visibly
        summary["outcome"] = f"failed: {type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        raise
    finally:
        finish()
        client.close()


if __name__ == "__main__":
    main()
