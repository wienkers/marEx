#!/bin/bash
# SLURM template for run_detect.py. Edit the #SBATCH lines for your site, then:
#
#   sbatch --account=<project> detect.sbatch configs/gridded.yaml [--smoke] [--only NAME ...]
#
# With `cluster.type: slurm` in the config, this job is only the driver: the Dask workers are further SLURM jobs
# (they bill `cluster.account`, else $SLURM_ACCOUNT, else this job's account), so a few cores and 16 GB are enough here.
# With `cluster.type: local` the workers run inside this job: size it to n_workers x memory_limit + a few GB for
# the client (e.g. --cpus-per-task=16 --mem=64G for the gridded smoke).
#SBATCH --job-name=marEx_detect
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=marEx_detect_%j.out
set -euo pipefail

# Activate an environment with marEx installed, e.g.:
#   source /path/to/conda/etc/profile.d/conda.sh && conda activate marex
PYTHON="${MAREX_PYTHON:-python}"

ulimit -S -s unlimited
cd "${SLURM_SUBMIT_DIR:-.}"  # config paths are relative to where sbatch was called
exec "$PYTHON" -u "${MAREX_BATCH_DIR:-$SLURM_SUBMIT_DIR}/run_detect.py" "$@"
