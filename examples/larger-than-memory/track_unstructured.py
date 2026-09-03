#!/usr/bin/env python
"""Larger-than-memory squeeze: `marEx.tracker` on an unstructured (ICON) mesh.

14,886,338 cells, so one whole int32 ID field is 59.5 MB *per timestep*: 65.3 GB over the
1096-day record.  ``persist`` holds several of those simultaneously and cannot run this at
any single-node budget; the streaming path completed the same record with a measured peak
of 116.8 GB.  This is the leg where the feasibility claim is least ambiguous, and also the
one where no bit-identity reference can exist at full length, because the persist side
produces no field to compare against.

``ncells`` stays whole on purpose -- the tracker is global in space.  Time is the lever,
and the iterative merge algorithm prefers larger time chunks.
"""

import argparse
import sys

import xarray as xr
from squeeze_common import GB, add_common_args, execute
from track_gridded import run_and_fingerprint

import marEx

DEFAULT_INPUT = "/scratch/b/b382615/mhws/extremes_binary_unstruct_shifting_hobday.zarr"


def main() -> None:
    """Run the unstructured track leg: build the input, size the whole-field arrays, execute."""
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--time-chunk", type=int, default=4)
    parser.add_argument("--R-fill", type=int, default=32)
    parser.add_argument("--T-fill", type=int, default=2)
    parser.add_argument("--area-filter-absolute", type=int, default=13500)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--nn-partitioning", dest="nn", action="store_true", default=True)
    parser.add_argument("--no-nn-partitioning", dest="nn", action="store_false")
    parser.add_argument("--no-allow-merging", dest="merging", action="store_false", default=True)
    parser.add_argument("--temp-dir", required=True)
    args = parser.parse_args()

    ds = xr.open_zarr(args.input, chunks={"time": args.time_chunk, "ncells": -1})
    if args.nt:
        ds = ds.isel(time=slice(0, args.nt))

    n_time, n_cells = int(ds.sizes["time"]), int(ds.sizes["ncells"])
    meta = {
        "leg": "track_unstructured",
        "grid": "unstructured",
        "input": args.input,
        "n_time": n_time,
        "ncells": n_cells,
        "shape": list(ds.extreme_events.shape),
        "input_bytes": int(ds.extreme_events.nbytes),
        "whole_field_int32_bytes": n_time * n_cells * 4,
        "input_chunks": {"time": args.time_chunk, "ncells": -1},
        "input_chunk_bytes": args.time_chunk * n_cells,
        "R_fill": args.R_fill,
        "T_fill": args.T_fill,
        "allow_merging": args.merging,
        "nn_partitioning": args.nn,
        "overlap_threshold": args.overlap_threshold,
    }
    print(
        f"[{args.label}] n_time={n_time} ncells={n_cells}; one whole field = "
        f"{n_time * n_cells / GB:.2f} GB as bool, {n_time * n_cells * 4 / GB:.2f} GB as int32",
        flush=True,
    )

    def work(client):
        tracker = marEx.tracker(
            ds.extreme_events,
            ds.mask,
            area_filter_absolute=args.area_filter_absolute,
            R_fill=args.R_fill,
            T_fill=args.T_fill,
            allow_merging=args.merging,
            overlap_threshold=args.overlap_threshold,
            nn_partitioning=args.nn,
            temp_dir=args.temp_dir,
            unstructured_grid=True,
            dimensions={"x": "ncells"},
            coordinates={"x": "lon", "y": "lat"},
            neighbours=ds.neighbours,
            cell_areas=ds.cell_areas,
            compute_mode=args.mode,
        )
        return run_and_fingerprint(tracker, args)

    execute(args, meta, work)


if __name__ == "__main__":
    sys.exit(main())
