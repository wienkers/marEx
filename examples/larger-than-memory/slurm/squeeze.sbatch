#!/bin/bash
# Generic SLURM wrapper for one larger-than-memory squeeze leg.
#
# Partition, --mem, --cpus-per-task and --time are supplied on the sbatch command line so
# that one wrapper serves every leg:
#
#   sbatch --partition=shared --mem=40G --cpus-per-task=8 --time=05:59:00 \
#          --job-name=f1_persist --output=$LOGDIR/f1_persist-%j.out \
#          --export=ALL,SQUEEZE_SCRIPT=detect_gridded.py,SQUEEZE_ARGS="--label f1_persist ..." \
#          slurm/squeeze.sbatch
#
# Squeeze legs are small-memory and belong on `shared`, which backfills in seconds.
# Node-scale reference legs go to `compute --exclusive` with a short wall.
#SBATCH --account=bk1377
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

: "${SQUEEZE_SCRIPT:?set SQUEEZE_SCRIPT}"
: "${SQUEEZE_ARGS:?set SQUEEZE_ARGS}"
SQUEEZE_DIR="${SQUEEZE_DIR:-/home/b/b382615/opt/marEx/examples/larger-than-memory}"
SQUEEZE_PYTHON="${SQUEEZE_PYTHON:-/home/b/b382615/opt/anaconda3/envs/super/bin/python}"

cd "$SQUEEZE_DIR"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} mem=${SLURM_MEM_PER_NODE:-?}MB cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "cmd: $SQUEEZE_PYTHON $SQUEEZE_SCRIPT $SQUEEZE_ARGS"

# Keep BLAS single-threaded: the parallelism that matters here is dask's, and an
# oversubscribed node changes the memory picture the leg is trying to measure.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

exec $SQUEEZE_PYTHON "$SQUEEZE_SCRIPT" $SQUEEZE_ARGS
