#!/bin/bash
# One rung of the larger-than-memory ladder in a 32 GiB job:
#
#   sbatch --account=<project> --time=08:00:00 ltm.sbatch detect --years 10
#   sbatch --account=<project> --time=08:00:00 ltm.sbatch track  --years 10
#
# The job's memory (--mem) is the system memory the notebook compares the data against. The Dask workers get
# 3 x 9 GB of it (ltm_run.py defaults: 3 workers x 2 threads); the rest is for the client, which runs the tracker's merge loop.
#SBATCH --job-name=marEx_ltm
#SBATCH --partition=shared
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=marEx_ltm_%j.out
set -euo pipefail

PYTHON="${MAREX_PYTHON:-python}"
cd "${SLURM_SUBMIT_DIR:-.}"
exec "$PYTHON" -u "${MAREX_LTM_DIR:-$SLURM_SUBMIT_DIR}/ltm_run.py" "$@"
