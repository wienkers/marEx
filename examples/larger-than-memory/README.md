# Larger-than-Memory Detection and Tracking

`larger_than_memory.ipynb` measures marEx's `compute_mode="streaming"` on a record larger than memory: 6-hourly
global SST on a 0.25° grid (721 x 1440) from the EERIE IFS-FESOM2-SR control run, detected and tracked inside one 32 GiB
batch job with a Dask cluster of 3 workers x 2 threads x 9 GB. Each rung lengthens the record with the same job configuration.

| stage | record | data | data / job | peak memory (anon) | wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| detect | 10 y | 60.7 GB input | 1.8 | 19.4 GiB | 63 min |
| detect | 20 y | 121.3 GB input | 3.5 | 20.6 GiB | 166 min |
| track | 5 y of extremes | 30.3 GB event field | 0.9 | 9.4 GiB | 57 min |
| track | 15 y of extremes | 91.0 GB event field | 2.6 | 9.4 GiB | 139 min |

Peak memory is flat in record length, and a longer record adds wall time (and, for detect, spill to disk). The notebook also compares `compute_mode="persist"` with the same job configuration. With `persist`, detection counted the same extremes at both lengths, but at 20 years it spilled more and ran 15 % longer. Only `streaming` completed the 15-year track.

## Files

| file | what it does |
| --- | --- |
| `larger_than_memory.ipynb` | reads the rung summaries in `ltm_results.json` and plots memory against record length |
| `ltm_run.py` | runs one rung (detect or track, `--years N`, `--mode streaming` or `persist`) and writes a JSON summary of what it cost |
| `ltm.sbatch` | the 32 GiB batch job a rung runs in |

## Running

```bash
sbatch --account=<project> ltm.sbatch detect --years 10 --workers 3 --threads 2 --memory-limit 9GB
sbatch --account=<project> ltm.sbatch track  --years 10 --workers 3 --threads 2 --memory-limit 9GB
SMOKE=1 python ltm_run.py detect --years 2    # one latitude band, minutes: check the plumbing first
```

The tracker reads the detect output of the same length, so run detect first. Detection uses a 5-year shifting baseline, so a
10-year detect rung gives the tracker 5 years of extremes.

Memory is read from the job's cgroup v2 counters. `anon` covers the client, the workers and everything else in the job,
and excludes the page cache, which the kernel reclaims on demand. The summary is written before the work starts and again
at the end, so an interrupted run still leaves a record. `--stall-minutes N` ends a run whose scheduler makes no
progress for N minutes and records it as `stalled`. Keep it at 0 for the tracker, whose merge loop runs on the client.

## Adapting to Another System

| setting | how | default |
| --- | --- | --- |
| input store | `MAREX_LTM_STORE` | `$MAREX_SCRATCH/marEx_data/input/fesom_sst6h/fesom_sst_6hourly_1971-2050.zarr` |
| outputs, staging, spill | `--out-dir` | `$MAREX_SCRATCH/marEx/ltm` |
| Python | `MAREX_PYTHON` | `python` on `PATH` |

Always pass `--memory-limit`, so the workers are sized to the job. A Dask `LocalCluster` otherwise sizes itself from the
whole machine's memory. `ltm_run.py` writes the limit each worker reports
into the summary. Leave room for staging and spill on the output file system: at 20 years detect spilled 138 GB with
`streaming` and 161 GB with `persist`.
