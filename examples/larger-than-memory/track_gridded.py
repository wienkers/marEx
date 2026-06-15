#!/usr/bin/env python
"""Larger-than-memory squeeze: `marEx.tracker` on a structured (lat/lon) grid.

Unlike ``detect``, the tracker is global *in space*: connected-component labelling and the
dilation matrix span the whole domain, so the spatial dimension must stay whole and time
is the only lever.  What grows with the record length is the whole-field int32 ID array --
3804 days at 0.25 degrees is 15.8 GB of it, and ``persist`` holds several such fields at
once (measured peak 57.0 GB at that length, against streaming's 20.4 GB).

Settings variants exposed here because they change the memory shape, not just the answer:
``--no-allow-merging`` skips the merge loop entirely, and ``--nn-partitioning`` selects the
BFS partition kernel over the centroid method.
"""

import argparse
import sys

import numpy as np
import xarray as xr
from squeeze_common import GB, add_common_args, execute

import marEx

DEFAULT_INPUT = "/scratch/b/b382615/mhws/extremes_binary_gridded_shifting_hobday.zarr"


def main() -> None:
    """Run the gridded track leg: build the input, size the whole-field arrays, execute."""
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--time-chunk", type=int, default=25)
    parser.add_argument("--R-fill", type=int, default=12)
    parser.add_argument("--T-fill", type=int, default=4)
    parser.add_argument("--area-filter-absolute", type=int, default=600)
    parser.add_argument("--overlap-threshold", type=float, default=0.25)
    parser.add_argument("--nn-partitioning", dest="nn", action="store_true", default=True)
    parser.add_argument("--no-nn-partitioning", dest="nn", action="store_false")
    parser.add_argument("--no-allow-merging", dest="merging", action="store_false", default=True)
    parser.add_argument("--temp-dir", required=True, help="Staging root for compute_mode='streaming'.")
    args = parser.parse_args()

    ds = xr.open_zarr(args.input, chunks={"time": args.time_chunk, "lat": -1, "lon": -1})
    if args.nt:
        ds = ds.isel(time=slice(0, args.nt))

    n_time, n_cells = int(ds.sizes["time"]), int(ds.sizes["lat"] * ds.sizes["lon"])
    meta = {
        "leg": "track_gridded",
        "grid": "structured",
        "input": args.input,
        "n_time": n_time,
        "shape": list(ds.extreme_events.shape),
        "input_bytes": int(ds.extreme_events.nbytes),
        "whole_field_int32_bytes": n_time * n_cells * 4,
        "input_chunks": {"time": args.time_chunk, "lat": -1, "lon": -1},
        "input_chunk_bytes": args.time_chunk * n_cells,
        "R_fill": args.R_fill,
        "T_fill": args.T_fill,
        "allow_merging": args.merging,
        "nn_partitioning": args.nn,
        "overlap_threshold": args.overlap_threshold,
    }
    print(
        f"[{args.label}] n_time={n_time} cells={n_cells}; one whole field = "
        f"{n_time * n_cells / GB:.2f} GB as bool, {n_time * n_cells * 4 / GB:.2f} GB as int32",
        flush=True,
    )

    def work(client):
        tracker = marEx.tracker(
            ds.extreme_events,
            ds.mask.where((ds.lat < 85) & (ds.lat > -90), other=False),
            grid_resolution=0.25,
            area_filter_absolute=args.area_filter_absolute,
            R_fill=args.R_fill,
            T_fill=args.T_fill,
            allow_merging=args.merging,
            overlap_threshold=args.overlap_threshold,
            nn_partitioning=args.nn,
            temp_dir=args.temp_dir,
            compute_mode=args.mode,
        )
        return run_and_fingerprint(tracker, args)

    execute(args, meta, work)


def run_and_fingerprint(tracker, args) -> dict:
    """Run the tracker and reduce the ID field without pulling it back into RAM.

    Under streaming the whole field already lives in ``temp_dir``; persisting it here
    would undo the entire mode, so the fingerprints are computed lazily instead.
    """
    if getattr(tracker, "allow_merging", args.merging) and args.merging:
        events_ds, merges_ds = tracker.run(return_merges=True)
    else:
        events_ds, merges_ds = tracker.run(), None

    id_field = events_ds.ID_field.astype(np.int32)
    out = {
        "n_events": int(events_ds.sizes.get("ID", 0)),
        "n_merges": int(merges_ds.sizes.get("merge_ID", 0)) if merges_ds is not None else 0,
        "id_field_sum": int(id_field.sum(dtype=np.int64).compute()),
        "n_nonzero_cells": int((id_field != 0).sum().compute()),
        "max_id": int(id_field.max().compute()),
        "staging_dir": events_ds.attrs.get("marex_staging_dir"),
    }
    if args.mode == "streaming":
        marEx.clear_staging(events_ds)
    return out


if __name__ == "__main__":
    sys.exit(main())
