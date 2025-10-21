#!/usr/bin/env python
"""
Global Daily SST (OSTIA) Analysis: Identifying Marine Extremes with MarEx-Detect

Pre-Process with Shifting-Baseline & Hobday et al. (2016) Definition for Extremes

Run from login node - this script will launch SLURM jobs via Dask distributed cluster

NOTE: This script uses a distributed cluster because detect.py does not contain
Numba JIT-compiled functions that execute locally. Unlike run_track.py, this
approach works correctly for detection/preprocessing tasks. If you prefer to
follow HPC best practices and run everything in SLURM jobs, consider creating
a submit_detect.sh script similar to submit_track.sh.
"""

import os
from getpass import getuser
from pathlib import Path

import xarray as xr

import marEx
import marEx.helper as hpc


def main():
    """Run MarEx preprocessing and extreme detection pipeline on distributed SLURM cluster."""
    # Configuration from environment variables
    n_workers = int(os.getenv("DASK_N_WORKERS", "128"))
    workers_per_node = int(os.getenv("DASK_WORKERS_PER_NODE", "64"))
    runtime = int(os.getenv("DASK_RUNTIME", "39"))
    slurm_account = os.getenv("SLURM_ACCOUNT", "bk1377")

    # Lustre Scratch Directory
    scratch_dir = Path("/scratch") / getuser()[0] / getuser()

    # Start Distributed Dask Cluster on SLURM
    print("Starting distributed Dask cluster on SLURM...")
    print(f"  Workers: {n_workers}")
    print(f"  Workers per node: {workers_per_node}")
    print(f"  Runtime: {runtime} minutes")
    print(f"  SLURM account: {slurm_account}")

    client = hpc.start_distributed_cluster(
        n_workers=n_workers,
        workers_per_node=workers_per_node,
        runtime=runtime,
        scratch_dir=scratch_dir / "clients",
        account=slurm_account,
    )

    # Choose optimal chunk size & load data
    print("Loading data...")
    file_name = scratch_dir / "mhws" / "ostia.zarr"
    chunk_size = {"time": 30, "lat": 360, "lon": 720}
    sst = xr.open_zarr(str(file_name), chunks=chunk_size).sst
    print(f"Data loaded: {sst}")

    # Process Data using MarEx-Detect helper functions:
    print("Processing extremes data...")
    extremes_ds = marEx.preprocess_data(
        sst,
        method_anomaly="shifting_baseline",  # Anomalies from a rolling climatology using previous window_year years
        method_extreme="hobday_extreme",  # Local day-of-year specific thresholds with windowing
        threshold_percentile=95,  # Use the 95th percentile as the extremes threshold
        window_year_baseline=15,
        smooth_days_baseline=21,  # Defines the rolling climatology window (15 years) and smoothing window (21 days)
        window_days_hobday=11,  # Defines the window (11 days) of compiled samples collected for extremes detection
        dimensions={
            "time": "time",
            "x": "lon",
            "y": "lat",
        },  # Define the dimensions of the data -- if 'y' exists, MarEx-Detect knows this is a gridded dataset
        dask_chunks={"time": 25},  # Dask chunks for *output* data
        use_temp_checkpoints=True,  # Enable checkpointing to prevent expensive recomputations
    )
    print(f"Processing complete: {extremes_ds}")

    # Save Extremes Data to zarr for more efficient parallel I/O
    print("Saving extremes data...")
    output_file = scratch_dir / "mhws" / "extremes_binary_gridded_shifting_hobday_batch.zarr"
    extremes_ds.to_zarr(output_file, mode="w")
    print(f"Data saved to: {output_file}")

    # Close the Dask cluster
    client.close()
    print("Processing complete!")


if __name__ == "__main__":
    main()
