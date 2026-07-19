#!/bin/bash
# Submit run_detect.py as a SLURM job.
#
# The script launches its Dask workers as further SLURM jobs, so this job only needs a few cores for the driver.
#
# Usage:
#   ./submit_detect.sh <slurm_account> [run_detect.py options]
#   ./submit_detect.sh <slurm_account> --smoke
#
# Set MAREX_PYTHON to the python of an environment with marEx installed (default: python on PATH).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <slurm_account> [run_detect.py options]" >&2
    exit 2
fi
ACCOUNT="$1"
shift

# Resolve the script directory here, because sbatch runs a spooled copy of the wrapper
HERE="$(cd "$(dirname "$0")" && pwd)"

sbatch --account="$ACCOUNT" \
    --job-name=marEx_detect \
    --partition=shared \
    --cpus-per-task=4 \
    --mem=16G \
    --time=03:00:00 \
    --output="${SLURM_SUBMIT_DIR:-$PWD}/marEx_detect_%j.out" \
    --wrap="ulimit -S -s unlimited; \"${MAREX_PYTHON:-python}\" \"$HERE/run_detect.py\" --account \"$ACCOUNT\" $*"
