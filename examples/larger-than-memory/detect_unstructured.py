#!/usr/bin/env python
"""Larger-than-memory squeeze: `marEx.preprocess_data` on an unstructured (ICON) mesh.

Input is the EERIE ICON-ESM-ER R02B09 ocean mesh, 14,886,338 cells.  The default
eight-year slice (2007 to 2014, 2922 days) is 174 GB of float32 (``n_time x n_cells x 4 B``).
Under ``persist`` the anchors that ``preprocess_data`` keeps are pinned in cluster memory;
under ``streaming`` they are staged to a scratch zarr instead.  At 200 steps the pinned
totals were 31.3 GB (``persist``) and 0.71 GB (``streaming``); the full eight-year leg has not
been run.  ``thresholds`` is *space*-scaled and does not shrink with time chunking.

The catalogue read needs internet (the ``shared`` partition on DKRZ Levante).  Pass
``--input`` to read a pre-staged local zarr instead and the leg becomes network-free; the
grid file (``--grid-file`` or ``$MAREX_ICON_GRID``) is needed either way.

Chunking, as in the gridded leg, is set by the *spatial* width: the notebook's
``ncells=100_000`` bounds one full-time slab at ``n_time x 100_000 x 4 B`` (1.17 GB over
eight years, arithmetic).  Leaving ``ncells`` whole makes it 174 GB and the run thrashes rather than
failing.  The time chunk must stay at or above ``smooth_days``.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from detect_gridded import fingerprint
from squeeze_common import GB, add_common_args, apply_smoke, execute

import marEx

# DKRZ-only resources.  The ICON R02B09 ocean grid file lives on the DKRZ /pool; override it
# with --grid-file or $MAREX_ICON_GRID on another system.  The default input is read through
# the EERIE intake catalogue, whose "dkrz.disk.model-output" key is likewise DKRZ-specific;
# pass --input (a local zarr with the variable on an ``ncells`` dimension) to avoid it.
GRID_FILE = os.environ.get("MAREX_ICON_GRID", "/pool/data/ICON/grids/public/mpim/0016/icon_grid_0016_R02B09_O.nc")
CATALOGUE = os.environ.get(
    "MAREX_EERIE_CATALOGUE", "https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml"
)


def load_from_catalogue(start: str, end: str) -> xr.DataArray:
    """Open the EERIE ICON-ESM-ER hist-1950 2D daily-mean ocean surface field over [start, end]."""
    import intake

    cat = intake.open_catalog(CATALOGUE)
    source = cat["dkrz.disk.model-output"]["icon-esm-er"]["hist-1950"]["v20240618"]["ocean"]["native"]
    return (
        source["2d_daily_mean"](chunks={}).to_dask().to.isel(depth=0).drop_vars({"cell_sea_land_mask"}).sel(time=slice(start, end))
    )


def main() -> None:
    """Run the unstructured detect leg: build the input, size the histogram slab, execute."""
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--input", default="", help="Local zarr instead of the EERIE catalogue (no internet needed).")
    parser.add_argument("--grid-file", default=GRID_FILE, help="ICON grid file with neighbor_cell_index and cell_area.")
    parser.add_argument("--var", default="to")
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default="2014-12-31")
    parser.add_argument("--time-chunk", type=int, default=21, help="Must be >= smooth_days.")
    parser.add_argument("--ncells-chunk", type=int, default=100_000, help="-1 leaves the mesh whole; see the docstring.")
    parser.add_argument("--method-anomaly", default="shifting_baseline")
    parser.add_argument("--method-extreme", default="seasonal_percentile")
    parser.add_argument("--window-years", type=int, default=5)
    parser.add_argument("--out-time-chunk", type=int, default=2)
    parser.add_argument("--write-output", default="")
    args = parser.parse_args()
    apply_smoke(args, nt=365)

    if args.input:
        da = xr.open_zarr(args.input)[args.var]
    else:
        da = load_from_catalogue(args.start, args.end)
    if args.nt:
        da = da.isel(time=slice(0, args.nt))
    da = da.chunk({"time": args.time_chunk, "ncells": args.ncells_chunk})

    grid = xr.open_dataset(args.grid_file, chunks={}).rename({"cell": "ncells"})
    neighbours = grid.neighbor_cell_index.rename({"clat": "lat", "clon": "lon"})
    cell_areas = grid.cell_area.rename({"clat": "lat", "clon": "lon"})

    n_time = int(da.sizes["time"])
    tile = da.sizes["ncells"] if args.ncells_chunk == -1 else args.ncells_chunk
    meta = {
        "leg": "detect_unstructured",
        "grid": "unstructured",
        "input": args.input or CATALOGUE,
        "n_time": n_time,
        "ncells": int(da.sizes["ncells"]),
        "shape": list(da.shape),
        "dtype": str(da.dtype),
        "input_bytes": int(da.nbytes),
        "input_chunks": {"time": args.time_chunk, "ncells": args.ncells_chunk},
        "input_chunk_bytes": int(np.prod([c[0] for c in da.chunks]) * da.dtype.itemsize),
        "histogram_slab_bytes": int(n_time * tile * 4),
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
            neighbours=neighbours,
            cell_areas=cell_areas,
            dimensions={"time": "time", "x": "ncells"},
            coordinates={"time": "time", "x": "lon", "y": "lat"},
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


if __name__ == "__main__":
    sys.exit(main())
