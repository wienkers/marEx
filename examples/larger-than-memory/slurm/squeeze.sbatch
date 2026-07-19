#!/bin/bash
# Generic SLURM wrapper for one larger-than-memory squeeze leg.
#
# Partition, --mem, --cpus-per-task and --time are supplied on the sbatch command line so
# that one wrapper serves every leg:
#
#   sbatch --account=$SLURM_ACCOUNT --partition=shared --mem=40G --cpus-per-task=8 \
#          --time=05:59:00 --job-name=g1_persist --output=$LOGDIR/g1_persist-%j.out \
#          --export=ALL,SQUEEZE_SCRIPT=track_gridded.py,SQUEEZE_ARGS="--label g1_persist ..." \
#          slurm/squeeze.sbatch
#
# The account is not set here: pass --account (slurm/submit.sh does).
# SQUEEZE_PYTHON selects the interpreter (default `python` on PATH); SQUEEZE_DIR the script
# directory (default: the directory above this file; sbatch runs a spooled copy of this script,
# so set SQUEEZE_DIR when submitting by hand).
#
# Squeeze legs are small-memory and belong on `shared`, which backfills in seconds.
# Node-scale reference legs go to `compute --exclusive` with a short wall.
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

: "${SQUEEZE_SCRIPT:?set SQUEEZE_SCRIPT}"
: "${SQUEEZE_ARGS:?set SQUEEZE_ARGS}"
SQUEEZE_DIR="${SQUEEZE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SQUEEZE_PYTHON="${SQUEEZE_PYTHON:-${MAREX_PYTHON:-python}}"

cd "$SQUEEZE_DIR"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} mem=${SLURM_MEM_PER_NODE:-?}MB cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "cmd: $SQUEEZE_PYTHON $SQUEEZE_SCRIPT $SQUEEZE_ARGS"

# Keep BLAS single-threaded: the parallelism that matters here is dask's, and an
# oversubscribed node changes the memory picture the leg is trying to measure.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

exec $SQUEEZE_PYTHON "$SQUEEZE_SCRIPT" $SQUEEZE_ARGS
