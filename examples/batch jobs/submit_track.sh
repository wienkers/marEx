#!/bin/bash
#SBATCH --job-name=marEx_track
#SBATCH --partition=compute
#SBATCH --cpus-per-task=128
#SBATCH --account=bk1377
#SBATCH --output=slurm_%j.out
#SBATCH --time=02:59:00
#SBATCH --mem=0
#SBATCH --exclusive

# SLURM Batch Script for MarEx Event Tracking
#
# This script submits run_track.py as a SLURM job, ensuring all computation
# (including Numba JIT-compiled functions) runs on allocated compute nodes.
#
# Usage:
#   sbatch submit_track.sh
#
# Configuration via environment variables (optional):
#   export DASK_N_WORKERS=32         # Number of Dask workers
#   export DASK_THREADS_PER_WORKER=1 # Threads per worker
#   export RUN_BASIC_TRACKER=true    # Run basic tracker comparison
#   export GRID_RESOLUTION=0.25      # Grid resolution in degrees
#   export AREA_FILTER=600           # Minimum object area (cells)
#   export R_FILL=12                 # Spatial hole filling radius
#   export T_FILL=4                  # Temporal gap filling (days)
#   export OVERLAP_THRESHOLD=0.25    # Merge overlap threshold
#   sbatch submit_track.sh

# Print job information
echo "========================================="
echo "SLURM Job Information"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "========================================="
echo ""

# Set ulimits for stack size
ulimit -S -s unlimited
ulimit -S -c unlimited

# Initialise conda for bash shell
source $(conda info --base)/etc/profile.d/conda.sh
conda activate super

# Set default environment variables if not already set
export DASK_N_WORKERS=${DASK_N_WORKERS:-32}
export DASK_THREADS_PER_WORKER=${DASK_THREADS_PER_WORKER:-1}

# Print configuration
echo "Configuration:"
echo "  DASK_N_WORKERS: $DASK_N_WORKERS"
echo "  DASK_THREADS_PER_WORKER: $DASK_THREADS_PER_WORKER"
echo "  RUN_BASIC_TRACKER: ${RUN_BASIC_TRACKER:-false}"
echo "  GRID_RESOLUTION: ${GRID_RESOLUTION:-0.25}"
echo "  AREA_FILTER: ${AREA_FILTER:-600}"
echo "  R_FILL: ${R_FILL:-12}"
echo "  T_FILL: ${T_FILL:-4}"
echo "  OVERLAP_THRESHOLD: ${OVERLAP_THRESHOLD:-0.25}"
echo ""

# Run the tracking script
echo "Starting MarEx tracking..."
python run_track.py

# Capture exit status
EXIT_STATUS=$?

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "========================================="

exit $EXIT_STATUS
