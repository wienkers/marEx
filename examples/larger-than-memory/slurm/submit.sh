#!/bin/bash
# Submit one named leg of the larger-than-memory squeeze matrix.
#
#   ./slurm/submit.sh pf_tg_persist        # one leg
#   ./slurm/submit.sh preflight            # the whole pre-flight group
#
# Every leg is named once, here, so the sizing table in README.md and what actually ran
# cannot drift apart.
#
# WHICH LEGS ARE VALIDATED (measured 2026-08-26; see the report):
#   g2_*  gridded track @16 GB   -- THE headline result: persist OOM-killed, streaming completes
#   sc_*  gridded track scaling  -- peak near-flat as n_time doubles
#   u3_*  unstructured @32 GB    -- persist misses a 5 h deadline twice, streaming completes
#   e1_*, f4_*, u2_*, g1_*       -- equivalence and calibration, all completed
# NOT VIABLE, do not submit without rethinking:
#   f1_*  gridded detect @nt=14761 -- detect is compute-bound; nt=2200 alone needs 1.6-2.4 h
#         with ample RAM and times out at 48 GB in BOTH modes. Dropped 2026-08-26.
#   f3_*, v_merge_off, v_fill      -- these read f1_stream's output store, which f1 never wrote.
#         Repoint them at /scratch/b/b382615/mhws/extremes_binary_gridded_shifting_hobday.zarr
#         (nt=3804) to run them, as g2_* already does.  Small squeeze legs go to `shared` (backfills in seconds); node-scale
# legs go to `compute --exclusive` with a short wall.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$(dirname "$HERE")"
OUT="${SQUEEZE_OUT:-/work/bk1377/b382615/marex_fable/measurements/lm}"
SCRATCH="${SQUEEZE_SCRATCH:-/scratch/b/b382615/marEx/lm}"
LOGS="${SQUEEZE_LOGS:-/work/bk1377/b382615/marex_fable/lm_logs}"
JOBS="${SQUEEZE_JOBS:-/home/b/b382615/opt/marEx/.claude/jobs.ndjson}"
mkdir -p "$OUT" "$SCRATCH" "$LOGS"

BASE="--outdir $OUT --scratch $SCRATCH"

submit () {  # submit <label> <partition> <mem> <cpus> <time> <script> <args...>
    local label=$1 part=$2 mem=$3 cpus=$4 wall=$5 script=$6; shift 6
    local extra=""
    [ "$part" = "compute" ] && extra="--exclusive --mem=0"
    # shellcheck disable=SC2086
    local id
    id=$(sbatch --parsable \
        --partition="$part" ${extra:-"--mem=$mem"} --cpus-per-task="$cpus" --time="$wall" \
        --job-name="lm_$label" --output="$LOGS/${label}-%j.out" \
        --export=ALL,SQUEEZE_DIR="$DIR",SQUEEZE_SCRIPT="$script",SQUEEZE_ARGS="--label $label $BASE $*" \
        "$HERE/squeeze.sbatch")
    echo "{\"job_id\":\"$id\",\"log_path\":\"$LOGS/${label}-${id}.out\",\"purpose\":\"marEx larger-than-memory leg $label\"}" >> "$JOBS"
    echo "submitted $label -> $id  ($part, mem=$mem, cpus=$cpus, wall=$wall)"
}

leg () {
case "$1" in
# ---------------------------------------------------------------- pre-flight (cheap) ---
# Purpose: prove each configuration runs AT ALL before spending the squeeze run on it.
# The unstructured detect pre-flight uses detrend_harmonic/global_percentile because the
# shifting/seasonal path needs multiple years of input and would defeat the point of a
# small probe.
pf_dg_persist) submit "$1" shared 60G 10 02:59:00 detect_gridded.py --mode persist   --nt 2200 --window-years 5 --workers 6 --mem-per-worker 8GB ;;
pf_dg_stream)  submit "$1" shared 60G 10 02:59:00 detect_gridded.py --mode streaming --nt 2200 --window-years 5 --workers 6 --mem-per-worker 8GB ;;
pf_dg_lazy)    submit "$1" shared 60G 10 02:59:00 detect_gridded.py --mode lazy      --nt 2200 --window-years 5 --workers 6 --mem-per-worker 8GB ;;
pf_du_persist) submit "$1" shared 80G 12 02:59:00 detect_unstructured.py --mode persist   --nt 200 --method-anomaly detrend_harmonic --method-extreme global_percentile --workers 8 --mem-per-worker 8GB ;;
pf_du_stream)  submit "$1" shared 80G 12 02:59:00 detect_unstructured.py --mode streaming --nt 200 --method-anomaly detrend_harmonic --method-extreme global_percentile --workers 8 --mem-per-worker 8GB ;;
pf_tg_persist) submit "$1" shared 44G  8 02:59:00 track_gridded.py --mode persist   --nt 64 --workers 6 --mem-per-worker 6GB --temp-dir "$SCRATCH/temp_pf_tg_persist" ;;
pf_tg_stream)  submit "$1" shared 44G  8 02:59:00 track_gridded.py --mode streaming --nt 64 --workers 6 --mem-per-worker 6GB --temp-dir "$SCRATCH/temp_pf_tg_stream" ;;
pf_tu_persist) submit "$1" shared 80G 12 02:59:00 track_unstructured.py --mode persist   --nt 64 --workers 8 --mem-per-worker 8GB --temp-dir "$SCRATCH/temp_pf_tu_persist" ;;
pf_tu_stream)  submit "$1" shared 80G 12 02:59:00 track_unstructured.py --mode streaming --nt 64 --workers 8 --mem-per-worker 8GB --temp-dir "$SCRATCH/temp_pf_tu_stream" ;;

# --------------------------------------------------- F1: gridded detect, 61.2 GB in ---
# 6 x 6 GB = 36 GB against a predicted persist peak near 135 GB.
f1_persist)     submit "$1" shared 44G  8 05:59:00 detect_gridded.py --mode persist   --workers 6 --mem-per-worker 6GB --deadline 19800 ;;
f1_persist_r2)  submit "$1" shared 44G  8 05:59:00 detect_gridded.py --mode persist   --workers 6 --mem-per-worker 6GB --deadline 19800 ;;
f1_stream)      submit "$1" shared 44G  8 07:59:00 detect_gridded.py --mode streaming --workers 6 --mem-per-worker 6GB --write-output "$SCRATCH/f1_extremes_gridded.zarr" ;;
f1_lazy)        submit "$1" shared 44G  8 07:59:00 detect_gridded.py --mode lazy      --workers 6 --mem-per-worker 6GB ;;

# ------------------------------------------ F2: unstructured detect, 174 GB in (8 yr) ---
# 8 x 8 GB = 64 GB against a measured persist pinned total of 103.6 GB.
# `shared`, not `compute`: the EERIE catalogue read needs internet.
f2_persist)     submit "$1" shared 72G 12 07:59:00 detect_unstructured.py --mode persist   --workers 8 --mem-per-worker 8GB --deadline 25200 ;;
f2_persist_r2)  submit "$1" shared 72G 12 07:59:00 detect_unstructured.py --mode persist   --workers 8 --mem-per-worker 8GB --deadline 25200 ;;
f2_stream)      submit "$1" shared 72G 12 23:59:00 detect_unstructured.py --mode streaming --workers 8 --mem-per-worker 8GB ;;

# ------------------------------------------------- F3: gridded track on F1's output ---
f3_persist)     submit "$1" shared 72G 12 05:59:00 track_gridded.py --mode persist   --input "$SCRATCH/f1_extremes_gridded.zarr" --workers 8 --mem-per-worker 8GB --temp-dir "$SCRATCH/temp_f3_persist" --deadline 18000 ;;
f3_persist_r2)  submit "$1" shared 72G 12 05:59:00 track_gridded.py --mode persist   --input "$SCRATCH/f1_extremes_gridded.zarr" --workers 8 --mem-per-worker 8GB --temp-dir "$SCRATCH/temp_f3_persist_r2" --deadline 18000 ;;
f3_persist_nospill) submit "$1" shared 72G 12 07:59:00 track_gridded.py --mode persist --no-spill --input "$SCRATCH/f1_extremes_gridded.zarr" --workers 8 --mem-per-worker 8GB --temp-dir "$SCRATCH/temp_f3_nospill" ;;
f3_stream)      submit "$1" shared 72G 12 07:59:00 track_gridded.py --mode streaming --input "$SCRATCH/f1_extremes_gridded.zarr" --workers 8 --mem-per-worker 8GB --temp-dir "$SCRATCH/temp_f3_stream" ;;

# ---------------------------------------- F4: unstructured track, 65.3 GB whole field ---
# Node-scale, so `compute --exclusive` with a short wall rather than a shared-queue wait.
f4_persist)     submit "$1" compute 0 128 02:59:00 track_unstructured.py --mode persist   --workers 16 --mem-per-worker 12GB --temp-dir "$SCRATCH/temp_f4_persist" --deadline 9000 ;;
f4_persist_r2)  submit "$1" compute 0 128 02:59:00 track_unstructured.py --mode persist   --workers 16 --mem-per-worker 12GB --temp-dir "$SCRATCH/temp_f4_persist_r2" --deadline 9000 ;;
f4_stream)      submit "$1" compute 0 128 02:59:00 track_unstructured.py --mode streaming --workers 16 --mem-per-worker 12GB --temp-dir "$SCRATCH/temp_f4_stream" ;;

# --------------------------------------------------------------- equivalence (short) ---
e1_tu_persist)  submit "$1" compute 0 128 02:59:00 track_unstructured.py --mode persist   --nt 256 --workers 16 --mem-per-worker 12GB --temp-dir "$SCRATCH/temp_e1_persist" ;;
e1_tu_stream)   submit "$1" compute 0 128 02:59:00 track_unstructured.py --mode streaming --nt 256 --workers 16 --mem-per-worker 12GB --temp-dir "$SCRATCH/temp_e1_stream" ;;
e2_du_persist)  submit "$1" compute 0 128 05:59:00 detect_unstructured.py --mode persist   --start 2012-01-01 --end 2014-12-31 --window-years 2 --workers 8 --mem-per-worker 16GB ;;
e2_du_stream)   submit "$1" compute 0 128 05:59:00 detect_unstructured.py --mode streaming --start 2012-01-01 --end 2014-12-31 --window-years 2 --workers 8 --mem-per-worker 16GB ;;
e2_du_lazy)     submit "$1" compute 0 128 05:59:00 detect_unstructured.py --mode lazy      --start 2012-01-01 --end 2014-12-31 --window-years 2 --workers 8 --mem-per-worker 16GB ;;
e3_dg_persist)  submit "$1" shared 72G 12 05:59:00 detect_gridded.py --mode persist   --nt 3650 --window-years 5 --workers 8 --mem-per-worker 8GB ;;
e3_dg_stream)   submit "$1" shared 72G 12 05:59:00 detect_gridded.py --mode streaming --nt 3650 --window-years 5 --workers 8 --mem-per-worker 8GB ;;
e3_dg_lazy)     submit "$1" shared 72G 12 05:59:00 detect_gridded.py --mode lazy      --nt 3650 --window-years 5 --workers 8 --mem-per-worker 8GB ;;

# ------------------------------------------------------------- tracker setting variants ---
v_nn_off)       submit "$1" compute 0 128 02:59:00 track_unstructured.py --mode streaming --no-nn-partitioning --workers 16 --mem-per-worker 12GB --temp-dir "$SCRATCH/temp_v_nn_off" ;;
v_merge_off)    submit "$1" shared 72G 12 05:59:00 track_gridded.py --mode streaming --no-allow-merging --input "$SCRATCH/f1_extremes_gridded.zarr" --workers 8 --mem-per-worker 8GB --temp-dir "$SCRATCH/temp_v_merge_off" ;;
v_fill)         submit "$1" shared 72G 12 05:59:00 track_gridded.py --mode streaming --R-fill 24 --T-fill 0 --input "$SCRATCH/f1_extremes_gridded.zarr" --workers 8 --mem-per-worker 8GB --temp-dir "$SCRATCH/temp_v_fill" ;;
*) echo "unknown leg: $1" >&2; return 1 ;;
esac
}

case "${1:-}" in
    preflight)   for l in pf_dg_persist pf_dg_stream pf_dg_lazy pf_du_persist pf_du_stream pf_tg_persist pf_tg_stream pf_tu_persist pf_tu_stream; do leg "$l"; done ;;
    feasibility) for l in f1_persist f1_persist_r2 f1_stream f1_lazy f2_persist f2_persist_r2 f2_stream f4_persist f4_persist_r2 f4_stream; do leg "$l"; done ;;
    equivalence) for l in e1_tu_persist e1_tu_stream e2_du_persist e2_du_stream e2_du_lazy e3_dg_persist e3_dg_stream e3_dg_lazy; do leg "$l"; done ;;
    variants)    for l in v_nn_off v_merge_off v_fill; do leg "$l"; done ;;
    "")          echo "usage: submit.sh <leg|preflight|feasibility|equivalence|variants>" >&2; exit 1 ;;
    *)           leg "$1" ;;
esac
