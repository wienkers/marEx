# Batch Jobs

The gridded tutorial as plain scripts, for running detection and tracking as SLURM jobs without Jupyter.
`run_detect.py` is `gridded/01_preprocess_extremes.ipynb` and `run_track.py` is `gridded/02_id_track_events.ipynb`, with
the same methods, parameters and cluster shapes.

| Script | What it does | Where it runs |
| --- | --- | --- |
| `run_detect.py` | shifting baseline + Hobday percentile on OSTIA daily SST, written to `extremes_shifting_hobday.zarr` (`--harmonic` adds the harmonic-detrend + global-threshold method) | a driver that launches its own Dask workers as SLURM jobs on the compute partition (256 × 7.8 GB by default) |
| `run_track.py` | tracking with merging and splitting, `compute_mode="streaming"`, written to `events_merged.zarr` and `events_merged_merges.nc` (`--basic` adds the basic tracker) | one node, 16 workers × 8 threads × 14 GB |
| `submit_detect.sh` | submits `run_detect.py`; the driver itself needs only 4 cores and 16 GB | `shared` partition |
| `submit_track.sh` | submits `run_track.py` | one whole node |

## Running

```bash
export MAREX_PYTHON=/path/to/env/bin/python   # an environment with marEx installed (default: python on PATH)
./submit_detect.sh <slurm_account>
./submit_track.sh <slurm_account>             # once detection has finished
```

The SLURM account is a required argument. Input and output paths default to `/scratch/<u>/<user>/...` as on Levante, and
every path, the period and the cluster size can be changed with options (`python run_detect.py --help`). `--smoke`
checks the plumbing on a small cluster. With it, detection runs on five years, and tracking on the first year of that output.

```bash
./submit_detect.sh <slurm_account> --smoke
./submit_track.sh <slurm_account> --smoke
```

With `--local`, `run_detect.py` starts its workers inside its own job instead of as separate SLURM jobs (single-threaded,
`--memory-limit` each), and needs no account. Size that allocation yourself, because `submit_detect.sh` asks only for what the driver needs.
`python run_detect.py --smoke --local` runs 8 workers × 7 GB inside one 16-core, 64 GB job.

For timings of the full runs see the stage summaries in the two gridded notebooks.
