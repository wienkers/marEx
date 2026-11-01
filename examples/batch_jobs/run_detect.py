#!/usr/bin/env python
"""
Identify extremes with marEx.preprocess_data, as configured by a YAML run file.

The configuration names the input, the Dask cluster and one or more detection runs, each written to its own zarr store
in `output_dir` for run_track.py. Every key under a run's `kwargs` is passed to marEx.preprocess_data unchanged, so any
method or parameter of the library can be used without editing this script. configs/ holds worked examples for a
global lat/lon grid (OSTIA), a regional cut and an unstructured ICON mesh.

Usage:
    python run_detect.py configs/gridded.yaml
    python run_detect.py configs/gridded.yaml --smoke    # the config's `smoke:` overrides (short period, small cluster)
    python run_detect.py configs/gridded.yaml --only shifting_hobday
"""

import argparse

import dask
from _common import load_config, open_grid, open_input, start_cluster

import marEx


def parse_args():
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="YAML run configuration")
    p.add_argument("--smoke", action="store_true", help="apply the config's `smoke:` overrides")
    p.add_argument("--only", nargs="+", metavar="NAME", help="run only these detection runs (by `name`)")
    return p.parse_args()


def main():
    """Open the input, start the cluster, run each configured detection and write it to zarr."""
    args = parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    detect = cfg["detect"]
    runs = [r for r in detect["runs"] if not args.only or r["name"] in args.only]
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    dask.config.set(cfg.get("dask_config", {}))

    client, monitor = start_cluster(detect["cluster"], cfg["scratch_dir"])

    data = open_input(cfg["input"])
    print(f"Input: {dict(data.sizes)}, chunks {dict(zip(data.dims, (c[0] for c in data.chunks)))}")

    # An unstructured mesh carries its neighbours and cell areas into the output, where the tracker reads them
    grid_kwargs = {}
    if "grid" in cfg:
        grid_kwargs["neighbours"], grid_kwargs["cell_areas"] = open_grid(cfg["grid"])

    for run in runs:
        out = cfg["output_dir"] / run["output"]
        kwargs = {"dimensions": cfg["dimensions"], **({"coordinates": cfg["coordinates"]} if "coordinates" in cfg else {})}
        kwargs.update(grid_kwargs)
        kwargs.update(run.get("kwargs", {}))
        if kwargs.get("compute_mode") == "streaming":
            kwargs.setdefault("scratch_dir", str(cfg["scratch_dir"] / "staging"))
        with monitor.stage(f"Detect: {run['name']}"):
            extremes_ds = marEx.preprocess_data(data, **kwargs)
            extremes_ds.to_zarr(out, mode="w")
        if kwargs.get("compute_mode") == "streaming":
            marEx.clear_staging(extremes_ds)  # the written store no longer needs the staged intermediates
        print(f"Wrote {out} ({extremes_ds.sizes['time']} time steps)")

    monitor.summary()
    client.close()


if __name__ == "__main__":
    main()
