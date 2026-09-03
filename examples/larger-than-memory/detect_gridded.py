#!/usr/bin/env python
"""Larger-than-memory squeeze: `marEx.preprocess_data` on a structured (lat/lon) grid.

Input is the full 0.25-degree OSTIA record, 14761 days x 720 x 1440 = 61.2 GB of float32.
Under ``persist`` this pins roughly a third of the input and peaks near twice it, so a
cluster of a few tens of GB cannot run it at all; under ``streaming`` the anchors go to a
scratch zarr and only a handful of chunks are ever resident.

Chunking note, and it is the setting that decides whether this runs: the quantile
reductions are global in time and independent in space, so marEx rechunks internally to a
narrow-in-space, full-in-time slab.  The working set of one slab is
``n_time x cells-in-one-INPUT-spatial-chunk x 4 B``, i.e. it is set by the *spatial* chunk
width and not by the time chunk.  Leaving lat/lon whole makes that the entire array and
turns the internal rechunk into an all-to-all transpose -- which does not fail fast, it
thrashes.  That rule was established on a 14.9 M-cell unstructured mesh, and it does **not**
transfer to this grid: at 1.04 M cells, spatially-whole input is what the production
notebook uses and what every previous measurement on this path used.  Splitting lat/lon
into finer tiles here made things strictly worse -- measured 2026-08-26, a 90 x 180 tiling
died in 6 minutes where the spatially-whole version survived, because the histogram's
internal rechunk then has to gather many more pieces.  Chunk the spatial dimension for the
ICON mesh; leave it whole here.

Examples
--------
Feasibility (expect persist to die, streaming to complete)::

    python detect_gridded.py --label f1_persist --mode persist    --workers 6 --mem-per-worker 6GB ...
    python detect_gridded.py --label f1_stream  --mode streaming  --workers 6 --mem-per-worker 6GB ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from squeeze_common import GB, add_common_args, execute

import marEx

DEFAULT_INPUT = "/scratch/b/b382615/mhws/ostia.zarr"


def main() -> None:
    """Run the gridded detect leg: build the input, size the histogram slab, execute."""
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--var", default="sst")
    parser.add_argument("--time-chunk", type=int, default=30)
    parser.add_argument("--lat-chunk", type=int, default=-1, help="-1 leaves latitude whole (see the docstring).")
    parser.add_argument("--lon-chunk", type=int, default=-1)
    parser.add_argument("--method-anomaly", default="shifting_baseline")
    parser.add_argument("--method-extreme", default="seasonal_percentile")
    parser.add_argument("--window-years", type=int, default=15)
    parser.add_argument("--out-time-chunk", type=int, default=25, help="dask_chunks for the OUTPUT dataset.")
    parser.add_argument("--write-output", default="", help="Optional zarr path; also produces a track leg's input.")
    args = parser.parse_args()

    chunks = {"time": args.time_chunk, "lat": args.lat_chunk, "lon": args.lon_chunk}
    da = xr.open_zarr(args.input, chunks=chunks)[args.var]
    if args.nt:
        da = da.isel(time=slice(0, args.nt))

    n_time = int(da.sizes["time"])
    spatial_tile = (da.sizes["lat"] if args.lat_chunk == -1 else args.lat_chunk) * (
        da.sizes["lon"] if args.lon_chunk == -1 else args.lon_chunk
    )

    meta = {
        "leg": "detect_gridded",
        "grid": "structured",
        "input": args.input,
        "n_time": n_time,
        "shape": list(da.shape),
        "dtype": str(da.dtype),
        "input_bytes": int(da.nbytes),
        "input_chunks": {k: int(v) for k, v in chunks.items()},
        "input_chunk_bytes": int(np.prod([c[0] for c in da.chunks]) * da.dtype.itemsize),
        "histogram_slab_bytes": int(n_time * spatial_tile * 4),
        "method_anomaly": args.method_anomaly,
        "method_extreme": args.method_extreme,
        "window_years": args.window_years,
    }
    print(
        f"[{args.label}] input {da.shape} {da.dtype} = {da.nbytes / GB:.1f} GB; "
        f"one full-time spatial slab = {meta['histogram_slab_bytes'] / GB:.2f} GB",
        flush=True,
    )

    def work(client):
        ds = marEx.preprocess_data(
            da,
            method_anomaly=args.method_anomaly,
            method_extreme=args.method_extreme,
            threshold_percentile=95,
            window_years=args.window_years,
            smooth_days=21,
            window_days=11,
            dask_chunks={"time": args.out_time_chunk},
            dimensions={"time": "time", "x": "lon", "y": "lat"},
            compute_mode=args.mode,
            scratch_dir=str(Path(args.scratch) / "staging" / args.label),
            validate=args.validate,
        )
        result = fingerprint(ds)
        if args.write_output:
            ds.to_zarr(args.write_output, mode="w")
            result["written_to"] = args.write_output
            marEx.clear_staging(ds)
        return result

    execute(args, meta, work)


def fingerprint(ds: xr.Dataset) -> dict:
    """Reductions that touch every element, so a lazy mode is genuinely forced to run.

    Deliberately not a `.persist()` of the whole dataset: that would pull a streamed
    result straight back into cluster RAM and defeat the mode under test.
    """
    out = {
        "n_time_out": int(ds.sizes["time"]),
        "n_extreme_cells": int(ds.extreme_events.sum().compute()),
        "anomaly_checksum": float(ds.dat_anomaly.astype("float64").sum().compute()),
    }
    if "thresholds" in ds:
        out["thresholds_checksum"] = float(ds.thresholds.astype("float64").sum().compute())
    return out


if __name__ == "__main__":
    sys.exit(main())
