#!/usr/bin/env python
"""
Identify and track extreme events with marEx.tracker, as configured by a YAML run file.

Reads a store written by run_detect.py (`track.input`, a name in `output_dir`), and runs one or more tracking
configurations on it. Every key under a run's `kwargs` is passed to the tracker unchanged. A run with
`tracker: regional` uses marEx.regional_tracker (a lat/lon box that does not wrap around the globe). On an
unstructured mesh the neighbours and cell areas that run_detect.py attached to its output are passed automatically.

Usage:
    python run_track.py configs/gridded.yaml
    python run_track.py configs/gridded.yaml --smoke
    python run_track.py configs/gridded.yaml --only merged
"""

import argparse

import dask
import xarray as xr
from _common import load_config, start_cluster

import marEx


def parse_args():
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="YAML run configuration")
    p.add_argument("--smoke", action="store_true", help="apply the config's `smoke:` overrides")
    p.add_argument("--only", nargs="+", metavar="NAME", help="run only these tracking runs (by `name`)")
    return p.parse_args()


def main():
    """Open the detected extremes, start the cluster, run each configured tracker and write its events."""
    args = parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    track = cfg["track"]
    runs = [r for r in track["runs"] if not args.only or r["name"] in args.only]
    temp_dir = cfg["scratch_dir"] / "staging"
    temp_dir.mkdir(parents=True, exist_ok=True)
    dask.config.set(cfg.get("dask_config", {}))

    client, monitor = start_cluster(track["cluster"], cfg["scratch_dir"])

    # The spatial dimension stays whole: the morphology and the labelling are global in space. Chunk time instead.
    ds = xr.open_zarr(cfg["output_dir"] / track["input"], chunks=track.get("chunks", {}))
    if "time" in track:
        ds = ds.sel(time=slice(*track["time"]))
    if "chunks" in track:
        ds = ds.chunk(track["chunks"])  # re-tile from the selection's first step
    print(f"Input: {dict(ds.sizes)}")

    mask = ds.mask
    if "mask_lat" in track:
        # e.g. drop the poles, where a lat/lon grid's anisotropy biases identification and tracking
        lo, hi = track["mask_lat"]
        mask = mask.where((ds.lat > lo) & (ds.lat < hi), other=False)

    grid_kwargs = {}
    if "neighbours" in ds and "cell_areas" in ds:
        grid_kwargs = {"neighbours": ds.neighbours, "cell_areas": ds.cell_areas}

    for run in runs:
        make_tracker = marEx.regional_tracker if run.get("tracker", "global") == "regional" else marEx.tracker
        kwargs = {"compute_mode": "streaming", "temp_dir": str(temp_dir), **grid_kwargs, **run.get("kwargs", {})}
        events_file = cfg["output_dir"] / run["output"]
        merges_file = run.get("merges")
        with monitor.stage(f"Track: {run['name']}"):
            tracker = make_tracker(ds.extreme_events, mask, **kwargs)
            if merges_file:
                events_ds, merges_ds = tracker.run(return_merges=True)
                merges_ds.to_netcdf(cfg["output_dir"] / merges_file, mode="w")
            else:
                events_ds = tracker.run()
            events_ds.to_zarr(events_file, mode="w")
        # The staging directory outlives the call: clear it once the output is written
        if kwargs["compute_mode"] == "streaming":
            marEx.clear_staging(events_ds)
        events_ds = xr.open_zarr(events_file, chunks={})
        print(f"Wrote {events_file}: {events_ds.sizes['time']} time steps")

    monitor.summary()
    client.close()


if __name__ == "__main__":
    main()
