#!/usr/bin/env python
"""
Global daily marine heatwave identification and tracking with marEx.tracker.

Script form of examples/gridded/02_id_track_events.ipynb. Reads the extremes written by run_detect.py, identifies
and tracks events through mergers and splits, and writes the events (zarr) and the merge records (netCDF).
The basic tracker (no merging, no gap filling) is available behind --basic.

Run it inside a SLURM job (submit_track.sh): it starts a local Dask cluster on the job's own node, with
compute_mode="streaming" so the working set is staged on disk in --temp-dir instead of pinned in memory.

Usage:
    python run_track.py
    python run_track.py --smoke   # works on a short run_detect.py --smoke output
"""

import argparse
from getpass import getuser
from pathlib import Path

import dask
import xarray as xr

import marEx
import marEx.helper as hpc

SCRATCH = Path("/scratch") / getuser()[0] / getuser()  # Lustre scratch


def parse_args():
    """Parse the command line; defaults reproduce the gridded/02 notebook."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=SCRATCH / "marEx" / "examples" / "gridded",
        help="where run_detect.py wrote its output; events are written here too",
    )
    p.add_argument("--input", type=Path, default=None, help="extremes zarr (default: <out-dir>/extremes_shifting_hobday.zarr)")
    p.add_argument(
        "--temp-dir", type=Path, default=SCRATCH / "marEx" / "examples" / "tmp", help="staging for compute_mode='streaming'"
    )
    p.add_argument("--dask-dir", type=Path, default=SCRATCH / "marEx" / "dask", help="scratch directory for the Dask cluster")
    p.add_argument("--n-workers", type=int, default=16, help="Dask workers")
    p.add_argument("--threads-per-worker", type=int, default=8, help="threads per worker")
    p.add_argument("--memory-limit", default="14GB", help="memory limit per worker")
    p.add_argument("--basic", action="store_true", help="also run the basic tracker (no merging, no gap filling)")
    p.add_argument("--smoke", action="store_true", help="quick check: 2 workers x 2 threads x 3 GB, first 365 days of the input")
    args = p.parse_args()
    if args.input is None:
        args.input = args.out_dir / "extremes_shifting_hobday.zarr"
    if args.smoke:
        args.n_workers, args.threads_per_worker, args.memory_limit = 2, 2, "3GB"
    return args


def main():
    """Run the merging tracker (and optionally the basic tracker) on a local Dask cluster."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    # graphs of a few hundred MB are expected at this scale
    dask.config.set({"distributed.admin.large-graph-warning-threshold": "1GB"})

    # memory_limit is passed explicitly because on an HPC node Dask sizes workers from the whole node's memory,
    # not from the job's allocation, and without it the workers can claim more than the job was granted.
    client = hpc.start_local_cluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
        scratch_dir=args.dask_dir,
    )

    # Records wall time, peak memory, spill and worker restarts for each `monitor.stage(...)` block below
    monitor = hpc.ResourceMonitor(client)

    # Space stays whole because the morphology and the labelling are global in space. Time is the axis to chunk.
    ds = xr.open_zarr(args.input, chunks={"time": 25, "lat": -1, "lon": -1})
    if args.smoke:
        ds = ds.isel(time=slice(0, 365))
    print(f"Input: {ds.sizes['time']} days")
    # The lat/lon grid's anisotropy near the pole biases identification & tracking, so cells north of 85° are masked out
    mask = ds.mask.where((ds.lat < 85) & (ds.lat > -90), other=False)

    events_file = args.out_dir / "events_merged.zarr"
    with monitor.stage("Track: merging & splitting"):
        tracker = marEx.tracker(
            ds.extreme_events,
            mask,
            grid_resolution=0.25,  # degrees, for object areas
            area_filter_absolute=600,  # drop objects smaller than 600 cells
            R_fill=12,  # fill holes up to 12 cells in radius
            T_fill=4,  # bridge gaps of up to 4 days within one event
            allow_merging=True,  # follow events through mergers and splits, recording the mergers in `merges_ds`
            overlap_threshold=0.25,  # link objects that overlap by at least 25 % of the smaller area
            nn_partitioning=True,  # partition merged objects by nearest parent cell (False: nearest centroid, Sun et al. 2023)
            compute_mode="streaming",
            temp_dir=str(args.temp_dir),
        )
        extreme_events_ds, merges_ds = tracker.run(return_merges=True)
        extreme_events_ds.to_zarr(events_file, mode="w")
        merges_ds.to_netcdf(args.out_dir / "events_merged_merges.nc", mode="w")

    # The staging directory outlives the call: clear it once the output is written, then read the result back from disk
    marEx.clear_staging(extreme_events_ds)
    extreme_events_ds = xr.open_zarr(events_file, chunks={})
    print(f"Wrote {events_file}: {extreme_events_ds.sizes['time']} days, {merges_ds.sizes} merge records")

    if args.basic:
        basic_file = args.out_dir / "events_basic.zarr"
        with monitor.stage("Track: basic"):
            basic_tracker = marEx.tracker(
                # 100-day time chunks (four of the stored 25-day chunks): without merging, events are labelled in 3-D, and the
                # labelling's task graph grows with the number of time chunks, so fewer, longer chunks keep it small
                ds.extreme_events.chunk({"time": 100}),
                mask,
                area_filter_absolute=600,
                R_fill=12,
                T_fill=0,  # no temporal gap filling
                allow_merging=False,  # touching objects share one ID for all time, before and after they touch
                compute_mode="streaming",
                temp_dir=str(args.temp_dir),
            )
            extreme_events_basic_ds = basic_tracker.run()
            extreme_events_basic_ds.to_zarr(basic_file, mode="w")
        marEx.clear_staging(extreme_events_basic_ds)
        print(f"Wrote {basic_file}")

    monitor.summary()
    client.close()


if __name__ == "__main__":
    main()
