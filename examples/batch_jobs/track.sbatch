#!/bin/bash
# SLURM template for run_track.py. Edit the #SBATCH lines for your site, then:
#
#   sbatch --account=<project> track.sbatch configs/gridded.yaml [--smoke] [--only NAME ...]
#
# The tracker runs a local Dask cluster inside this job, so the allocation must hold
# n_workers x memory_limit plus room for the client, which runs the merge loop.
# The configs in configs/ use one whole node (gridded: 16 x 14 GB; unstructured: 8 x 28 GB) or, for
# the regional run, a 64 GB shared job (4 x 12 GB).
#SBATCH --job-name=marEx_track
#SBATCH --partition=compute
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=marEx_track_%j.out
set -euo pipefail

PYTHON="${MAREX_PYTHON:-python}"

ulimit -S -s unlimited
cd "${SLURM_SUBMIT_DIR:-.}"
exec "$PYTHON" -u "${MAREX_BATCH_DIR:-$SLURM_SUBMIT_DIR}/run_track.py" "$@"
