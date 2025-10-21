#!/usr/bin/env python
"""
Global Daily Event Analysis: Marine Heatwave ID & Tracking using MarEx

Performs morphological preprocessing, blob identification, temporal tracking,
and graph reduction to create globally unique IDs for each tracked extreme event.

This script should be run INSIDE a SLURM job (via sbatch submit_track.sh).
It uses a local Dask cluster that utilizes the compute resources allocated to the job.
This ensures that all computation, including Numba JIT-compiled functions, runs on
compute nodes rather than the login node.
"""

import os
from getpass import getuser
from pathlib import Path

import xarray as xr

import marEx
import marEx.helper as hpc


def main():
    """Run MarEx event tracking pipeline on local Dask cluster within SLURM job allocation."""
    # Configuration from environment variables
    n_workers = int(os.getenv("DASK_N_WORKERS", "32"))
    threads_per_worker = int(os.getenv("DASK_THREADS_PER_WORKER", "1"))

    # Tracking configuration
    run_basic = os.getenv("RUN_BASIC_TRACKER", "false").lower() == "true"
    grid_resolution = float(os.getenv("GRID_RESOLUTION", "0.25"))
    area_filter = int(os.getenv("AREA_FILTER", "600"))
    r_fill = int(os.getenv("R_FILL", "12"))
    t_fill = int(os.getenv("T_FILL", "4"))
    overlap_threshold = float(os.getenv("OVERLAP_THRESHOLD", "0.25"))

    # Lustre Scratch Directory
    scratch_dir = Path("/scratch") / getuser()[0] / getuser()

    # Start Local Dask Cluster
    # This cluster runs within the SLURM job allocation, ensuring all computation
    # (including Numba JIT-compiled functions) executes on compute nodes
    print("Starting local Dask cluster within SLURM job allocation...")
    print(f"  Workers: {n_workers}")
    print(f"  Threads per worker: {threads_per_worker}")

    client = hpc.start_local_cluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        scratch_dir=scratch_dir / "clients",
    )

    # Choose optimal chunk size & load preprocessed data
    print("\nLoading preprocessed extremes data...")
    input_file = scratch_dir / "mhws" / "extremes_binary_gridded_shifting_hobday_batch.zarr"
    chunk_size = {"time": 25, "lat": -1, "lon": -1}
    ds = xr.open_zarr(str(input_file), chunks=chunk_size)
    print(f"Data loaded: {ds}")

    # Run ID, Tracking, & Merging
    print("\n" + "=" * 80)
    print("Running Full Tracking with Merge/Split Detection...")
    print("=" * 80)

    tracker = marEx.tracker(
        ds.extreme_events,
        ds.mask.where(
            (ds.lat < 85) & (ds.lat > -90), other=False
        ),  # Modify Mask: Anisotropy of lat/lon grid near poles biases ID & Tracking
        grid_resolution=grid_resolution,  # Grid resolution in degrees, used to calculate object areas on the globe
        area_filter_absolute=area_filter,  # Remove objects smaller than specified cells
        R_fill=r_fill,  # Fill small holes with radius < R_fill _cells_
        T_fill=t_fill,  # Allow gaps of T_fill days and still continue event tracking with same ID
        allow_merging=True,  # Allow extreme events to split/merge. Keeps track of merge events & unique IDs
        overlap_threshold=overlap_threshold,  # Overlap threshold for merging events
        nn_partitioning=True,  # Use new NN method to partition merged children areas
        verbose=True,
    )

    extreme_events_ds, merges_ds = tracker.run(return_merges=True)
    print(f"\nTracking complete: {extreme_events_ds}")
    print(f"Merges dataset: {merges_ds}")

    # Save IDed/Tracked/Merged Events to zarr for efficient parallel I/O
    print("\nSaving tracked events data...")
    output_file = scratch_dir / "mhws" / "extreme_events_merged_gridded_shifting_batch.zarr"
    extreme_events_ds.to_zarr(output_file, mode="w")
    print(f"Events data saved to: {output_file}")

    # Save Merges Dataset to netcdf
    print("Saving merges data...")
    merges_file = scratch_dir / "mhws" / "extreme_events_merged_gridded_shifting_merges_batch.nc"
    merges_ds.to_netcdf(merges_file, mode="w")
    print(f"Merges data saved to: {merges_file}")

    # Optionally run basic tracking for comparison
    if run_basic:
        print("\n" + "=" * 80)
        print("Running Basic Tracking (No Merge/Split, No Temporal Gap Filling)...")
        print("=" * 80)

        tracker_basic = marEx.tracker(
            ds.extreme_events,
            ds.mask.where(
                (ds.lat < 85) & (ds.lat > -90), other=False
            ),  # Modify Mask: Anisotropy of lat/lon grid near poles biases ID & Tracking
            area_filter_absolute=area_filter,  # Remove objects smaller than specified cells
            R_fill=r_fill,  # Fill small holes with radius < R_fill _cells_
            T_fill=0,  # No temporal hole filling
            allow_merging=False,  # Do not allow extreme events to split/merge
        )

        extreme_events_basic_ds = tracker_basic.run()
        print(f"\nBasic tracking complete: {extreme_events_basic_ds}")

        # Save Basic IDed Events to zarr
        print("\nSaving basic tracked events data...")
        basic_output_file = scratch_dir / "mhws" / "extreme_events_basic_gridded_shifting_batch.zarr"
        extreme_events_basic_ds.to_zarr(basic_output_file, mode="w")
        print(f"Basic events data saved to: {basic_output_file}")

    # Close the Dask cluster
    client.close()
    print("\n" + "=" * 80)
    print("Tracking complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
