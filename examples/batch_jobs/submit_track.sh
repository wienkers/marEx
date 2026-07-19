#!/bin/bash
# Submit run_track.py as a whole-node SLURM job (16 workers x 8 threads x 14 GB, as in the gridded notebook).
#
# Usage:
#   ./submit_track.sh <slurm_account> [run_track.py options]
#   ./submit_track.sh <slurm_account> --basic
#
# Set MAREX_PYTHON to the python of an environment with marEx installed (default: python on PATH).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <slurm_account> [run_track.py options]" >&2
    exit 2
fi
ACCOUNT="$1"
shift

# Resolve the script directory here, because sbatch runs a spooled copy of the wrapper
HERE="$(cd "$(dirname "$0")" && pwd)"

sbatch --account="$ACCOUNT" \
    --job-name=marEx_track \
    --partition=compute \
    --exclusive \
    --mem=0 \
    --time=02:59:00 \
    --output="${SLURM_SUBMIT_DIR:-$PWD}/marEx_track_%j.out" \
    --wrap="ulimit -S -s unlimited; \"${MAREX_PYTHON:-python}\" \"$HERE/run_track.py\" $*"
