#!/usr/bin/env python
"""
Global daily SST (OSTIA): identify marine extremes with marEx.preprocess_data.

Script form of examples/gridded/01_preprocess_extremes.ipynb. Anomalies come from a shifting baseline and the extremes
from a day-of-year percentile (Hobday et al., 2016). The result is written to zarr for run_track.py.

The script is a driver: it launches the Dask workers itself, as short SLURM jobs on the compute partition
(marEx.helper.start_distributed_cluster), so it can run from a login node or inside a small job
(see submit_detect.sh). Numba-compiled kernels, such as those in the extremes histogram, run on those workers.

Usage:
    python run_detect.py --account <slurm_account>
    python run_detect.py --account <slurm_account> --smoke   # five years, small cluster
    python run_detect.py --smoke --local                     # the same, with the workers inside the current job
"""

import argparse
import os
from getpass import getuser
from pathlib import Path

import dask
import xarray as xr

import marEx
import marEx.helper as hpc

SCRATCH = Path("/scratch") / getuser()[0] / getuser()  # Lustre scratch


def parse_args():
    """Parse the command line; defaults reproduce the gridded/01 notebook."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=SCRATCH / "mhws" / "ostia.zarr", help="daily SST zarr store (variable `sst`)")
    p.add_argument("--out-dir", type=Path, default=SCRATCH / "marEx" / "examples" / "gridded", help="output directory")
    p.add_argument("--dask-dir", type=Path, default=SCRATCH / "marEx" / "dask", help="scratch directory for the Dask cluster")
    p.add_argument(
        "--account", default=os.getenv("SLURM_ACCOUNT"), help="SLURM account for the worker jobs (default: $SLURM_ACCOUNT)"
    )
    p.add_argument("--start", default="1982-01-01", help="first day of the input period")
    p.add_argument("--end", default="2022-05-31", help="last day of the input period")
    p.add_argument(
        "--window-years", type=int, default=15, help="shifting-baseline length; the output starts this many years after --start"
    )
    p.add_argument("--n-workers", type=int, default=256, help="Dask workers, single-threaded, each its own slice of a SLURM job")
    p.add_argument("--workers-per-node", type=int, default=32, help="workers per 256 GB node")
    p.add_argument("--runtime", type=int, default=119, help="minutes per worker job; must cover the whole run")
    p.add_argument(
        "--local",
        action="store_true",
        help="start the workers inside this job (start_local_cluster) instead of as SLURM jobs; needs no account",
    )
    p.add_argument("--memory-limit", default="7GB", help="memory per worker with --local")
    p.add_argument("--harmonic", action="store_true", help="also run the harmonic-detrend + global-threshold method")
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "quick check: 5-year period, 2-year baseline, 8 workers, 15-minute worker jobs "
            "(overrides the period and cluster options)"
        ),
    )
    args = p.parse_args()
    if args.smoke:
        args.end = str(int(args.start[:4]) + 4) + "-12-31"
        args.window_years, args.n_workers, args.workers_per_node, args.runtime = 2, 8, 8, 15
    if not args.local and not args.account:
        p.error("a SLURM account is required: pass --account or set $SLURM_ACCOUNT")
    return args


def main():
    """Run shifting-baseline + Hobday detection (and optionally the harmonic method) on a distributed SLURM cluster."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    period = slice(args.start, args.end)
    print(f"Period: {args.start} to {args.end}, window_years={args.window_years}")

    # graphs of a few hundred MB are expected at this scale
    dask.config.set({"distributed.admin.large-graph-warning-threshold": "1GB"})

    if args.local:
        # Dask sizes workers from the whole node's memory, not the job's allocation, so pass memory_limit explicitly
        client = hpc.start_local_cluster(
            n_workers=args.n_workers, threads_per_worker=1, memory_limit=args.memory_limit, scratch_dir=args.dask_dir
        )
    else:
        # Each group of workers_per_node workers is its own SLURM job on the compute partition. The walls cover the whole
        # run: dask's P2P rechunk does not survive a worker leaving mid-shuffle, so workers are neither retired early nor
        # replaced.
        client = hpc.start_distributed_cluster(
            n_workers=args.n_workers,
            workers_per_node=args.workers_per_node,
            runtime=args.runtime,
            node_memory=256,
            scratch_dir=args.dask_dir,
            account=args.account,
        )
        client.wait_for_workers(args.n_workers, timeout=1800)

    # Records wall time, peak memory, spill and worker restarts for each `monitor.stage(...)` block below
    monitor = hpc.ResourceMonitor(client)

    sst = xr.open_zarr(args.input, chunks={"time": 30, "lat": -1, "lon": -1}).sst.sel(time=period)
    print(f"Input: {sst.sizes['time']} days on a {sst.sizes['lat']} x {sst.sizes['lon']} grid")

    with monitor.stage("Detect: shifting baseline + Hobday"):
        extremes_ds = marEx.preprocess_data(
            sst,
            method_anomaly="shifting_baseline",  # anomaly from a backward-looking climatology of the previous `window_years`
            method_extreme="seasonal_percentile",  # day-of-year threshold (Hobday et al., 2016)
            threshold_percentile=95,
            window_years=args.window_years,
            smooth_days=21,  # smoothing applied to the climatology (days)
            window_days=11,  # samples from an 11-day window around each day of year enter its percentile
            dimensions={"time": "time", "x": "lon", "y": "lat"},  # a "y" dimension tells marEx this is a lat/lon grid
            dask_chunks={"time": 25},  # chunking of the output
        )
        extremes_ds.to_zarr(args.out_dir / "extremes_shifting_hobday.zarr", mode="w")
    print(f"Wrote {args.out_dir / 'extremes_shifting_hobday.zarr'} ({extremes_ds.sizes['time']} days)")

    if args.harmonic:
        with monitor.stage("Detect: harmonic detrend + global threshold"):
            extremes_ds = marEx.preprocess_data(
                sst,
                method_anomaly="detrend_harmonic",  # removes the mean, annual & semi-annual harmonics and the trend
                method_extreme="global_percentile",  # one threshold per cell, over all days
                threshold_percentile=95,
                detrend_orders=[1, 2],  # linear + quadratic trend
                dimensions={"time": "time", "x": "lon", "y": "lat"},
                dask_chunks={"time": 25},
            )
            extremes_ds.to_zarr(args.out_dir / "extremes_harmonic_global.zarr", mode="w")
        print(f"Wrote {args.out_dir / 'extremes_harmonic_global.zarr'}")

    monitor.summary()
    client.close()


if __name__ == "__main__":
    main()
