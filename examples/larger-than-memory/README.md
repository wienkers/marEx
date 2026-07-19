# Larger-than-Memory Demonstrations

These scripts test one claim that the unit tests cannot reach:

> A workload that is OOM-killed under `compute_mode="persist"` completes under
> `compute_mode="streaming"` in the same memory allocation.

They are not part of `pytest`. Each leg wants a batch allocation and tens of minutes to hours. The
fast tests (`tests/test_compute_mode.py`, `tests/test_track_compute_mode.py`) check the wiring, count
the bytes actually pinned and compare the modes on small fixtures. What they cannot do is force a
real memory squeeze, so they cannot show feasibility. The user-facing explanation is in the
performance guide of the documentation.

## What Was Measured

The headline result is the gridded tracker on 3804 daily steps of a 720 x 1440 field (the int32
event field alone is 15.8 GB), in a 24 GiB job allocation with a 4 x 4 GB = 16 GB dask budget:

| mode | outcome | peak |
| --- | --- | ---: |
| `persist` | OOM-killed in 5 of 5 runs (SLURM `OUT_OF_MEMORY`) | none (killed) |
| `streaming` | completed in 7 of 7 runs, with the same event and merge counts every time | 6.65 to 7.30 GB |

The peak of the streaming runs is near-flat in series length, for the same allocation and budget:

| n_time | whole int32 field | peak | wall |
| ---: | ---: | ---: | ---: |
| 951 | 3.9 GB | 6.2 GB | 355 s |
| 1902 | 7.9 GB | 6.9 GB | 849 s |
| 3804 | 15.8 GB | 6.65 to 7.30 GB (seven runs) | 1600 to 1676 s |

The field grows 4 times across that span and the peak grows by a small fraction of that. Do not
read a percentage from the table: the seven runs at 3804 steps spread over 0.65 GB, the size of the
growth. The wall column is an order of magnitude only, since the runs were on different nodes and the
event count grows with length.

The same leg across allocations (dask budget held at 4 x 4 GB, one run per cell unless stated):

| allocation | `persist` | `streaming` |
| ---: | --- | --- |
| 15, 16, 20 GiB | not run | completed |
| 24 GiB | OOM-killed, 5 of 5 | completed, 7 of 7 |
| 28 GiB | did not finish: no OOM, stopped by the 12,000 s deadline | not run |
| 40 GiB | completed, 2 of 2, but not cleanly | not run |

This does not show a clean allocation threshold for `persist`. One 40 GiB run froze for about
90 minutes at the same stage where the 28 GiB run stalled, with one worker paused and the others idle,
and then resumed for a reason these runs do not reveal. A second 40 GiB run, with dask's pause
threshold switched off, did not freeze, but its nanny restarted a worker ten times. The 28 GiB row is
unresolved, not a failure. The statement the data support is narrow: at 4 GB per worker, `persist` was
killed in a 24 GiB allocation while `streaming` completed in the same allocation.

The allocation, not the dask budget, is the number to state. `memory_limit` governs the workers, and
under `persist` it is the client process that grows (15.8 GiB at its largest in the 40 GiB run,
against 1.9 GiB for a streaming client).

### The Unstructured Tracker

ICON R02B09 (14.9 M cells), 1096 steps, 4 x 8 GB = 32 GB dask budget on a whole node: `persist` did not
complete within 5 h on either of two replicates, while `streaming` completed in 3 h 50 min with matching
output reductions (4359 events, 9404 merges). This is a wall-clock statement. Both `persist` replicates
were stopped by the harness's own deadline, with no kernel out-of-memory kill (the workers' nanny logged
9 and 10 memory warnings in the two `persist` runs and 11 in the `streaming` run), so it does not show
that `persist` cannot fit. At 16 x 12 GB (192 GB total) `persist` completes the same track in about 72 minutes.
The gridded leg, where the kernel killed `persist`, is what the headline rests on, and the
unstructured result is supporting evidence.

### Detect Does Not Squeeze

On 9.1 GB of input (2200 days), `detect` peaks near 127 GB in both modes while `streaming` pins 35 KB against
`persist`'s 9.95 GB. Its peak is a
transient that does not depend on the mode, so `compute_mode` cannot move it. Use `streaming` for detect
to shrink what is pinned, not to lower the peak.

## Two Gates, Never One Run

| gate | length | cluster | what it shows |
| --- | --- | --- | --- |
| Equivalence | short enough that both modes fit | comfortable | outputs identical across modes |
| Feasibility | long enough that `persist` cannot fit in the allocation | squeezed | `persist` fails, `streaming` completes |

A feasibility leg has no array-level reference by construction: the `persist` side produced no output.
What is compared instead are order-invariant reductions of the result (event count, merge count, sum of
the ID field, number of non-zero cells, maximum ID). The array-level check lives in the test suite, at
fixture scale and zero tolerance.

## Guard Rails

Each of these exists because its absence has produced a wrong answer.

- **The effective per-worker memory limit is asserted.** Dask can read the memory of the whole
  machine and not the cap of the batch job. Without an explicit `memory_limit` the squeeze never binds,
  `persist` completes, and the leg passes for the wrong reason. `build_cluster` asks the workers for
  their limit and exits non-zero rather than run a meaningless test.
- **Failures are classified from evidence.** A wall-clock kill is not proof of an out-of-memory
  condition. The runner looks for `KilledWorker`, `MemoryError` and the nanny's memory warnings, and
  reports `timeout_inconclusive` when none occurred.
- **Pinned bytes are counted.** An array is still a dask collection after `.persist()`, so checking
  for one proves nothing. The accountant wraps the persist entry points and attributes every byte to
  the marEx line that requested it, including modules that did `from dask import persist`, which a
  patch of `dask.persist` alone would miss.
- **Spilling stays enabled.** Switching it off deadlocks the cluster, raises the peak (169.5 GB with
  spilling off against 118.7 GB with it on, on the same workload and budget) and kills both modes. `--no-spill` exists
  only as a labelled control.
- **The spill figure says UNMEASURED when it cannot be read**, and is only printed when every requested
  worker answered. On one 3804-step `streaming` leg it read 0 bytes across 333 samples, each covering all 4 workers.
  That shows the metric reports honestly where nothing should spill. It does not show that nothing could
  have spilled: a spill shorter than the 5 s sampling interval is invisible, and dask's second path to
  disk (the process-memory threshold) was not tracked per worker.
- **A breadcrumb summary is written before the work starts**, so a leg killed by the wall clock still
  records what it attempted.
- **Squeeze by worker count and record length, never by absurd per-worker memory.** Workers keep 4 to
  12 GB so that per-task working sets stay comfortable and only aggregate memory can bind.
- **Never size a squeeze from a measured peak.** Peak memory depends on the room given: the same
  `persist` workload peaked at 13.1 GB with a 16 GB budget and at 22.1 GB with a 32 GB budget. Size it from the arithmetic
  invariant, the whole int32 field `n_time x n_cells x 4 B`.

## Sizing

Byte counts are uncompressed `n_time x n_cells x itemsize`. The slab is the working set of one internal
reduction tile, `n_time x cells in one input spatial chunk x 4 B`.

| leg | data | dimensions | input size | input chunk | slab or whole field |
| --- | --- | --- | ---: | --- | ---: |
| gridded detect | OSTIA SST, first 2200 days | 2200 x 720 x 1440 | 9.1 GB | `{time:30, lat:-1, lon:-1}` | slab 9.1 GB |
| unstructured detect (probe) | ICON-ESM-ER, first 200 days | 200 x 14,886,338 | 11.9 GB | `{time:21, ncells:100_000}` | slab 0.08 GB |
| unstructured detect (`f2_*`, not run) | ICON-ESM-ER, 8 yr | 2922 x 14,886,338 | 174.0 GB | `{time:21, ncells:100_000}` | slab 1.17 GB |
| gridded track | binary extremes, OSTIA | 3804 x 720 x 1440 | 3.9 GB (bool) | `{time:25, lat:-1, lon:-1}` | int32 field 15.8 GB |
| unstructured track | binary extremes, ICON | 1096 x 14,886,338 | 16.3 GB (bool) | `{time:4, ncells:-1}` | int32 field 65.3 GB |

The gridded detect path is a poor squeeze: it is compute-bound and not memory-bound (at 2200 steps on
48 GB both modes were stopped by a 12,000 s deadline), so it says nothing about `compute_mode`. It
is run at 2200 days for equivalence and for the peak figure above. The full eight-year unstructured
detect leg (`f2_*`) has not been run. The gridded-track legs (`g1_*`, `g2_*`, `sc_*`) and the
unstructured-track legs (`u2_*`, `u3_*`, `f4_*`) carry the results above.

## Running

```bash
./slurm/submit.sh preflight      # small probes: does each configuration run at all? Always first.
./slurm/submit.sh headline       # g2_*: persist twice and streaming once, 24 GiB, 4 x 4 GB
./slurm/submit.sh allocation     # the same leg at 15 to 40 GiB
./slurm/submit.sh scaling        # sc_*: streaming peak against series length
./slurm/submit.sh calibration    # g1_* (gridded, 4 x 8 GB) and u2_* (unstructured)
./slurm/submit.sh unstructured   # u3_*: persist twice and streaming once, whole node
./slurm/submit.sh equivalence    # short legs that compare the modes
./slurm/submit.sh variants       # tracker settings
./slurm/submit.sh <leg-name>     # any single leg

python report.py <measurements-dir>
```

`submit.sh` needs `SLURM_ACCOUNT` set (see "Adapting to Another System").

Pre-flight is the cheapest insurance against spending a headline leg on a failure that has nothing to
do with `compute_mode`.

Each leg writes `<label>_summary.json` (dimensions, chunking, cluster budget, the asserted effective
per-worker limit, outcome and failure classification, peak and mean cluster memory, bytes pinned per
marEx source line, spill, nanny events, wall clock and output fingerprints) and a
`<label>_memseries.npy` memory trace. `report.py` collates them into the results table.

## Adapting to Another System

Every path is an argument or an environment variable, and nothing is hardwired to one account.

| setting | how | default |
| --- | --- | --- |
| SLURM account | `SLURM_ACCOUNT` (or `SQUEEZE_ACCOUNT`) | none: `submit.sh` refuses to run without it |
| Python | `MAREX_PYTHON` | `python` on `PATH` |
| input stores | `MAREX_LTM_DATA` directory, or `--input` per script | `/scratch/<initial>/$USER/marEx_data` (OSTIA under `input/ostia/`, detect outputs under `output/detect_binaries/`) |
| summaries, scratch, job logs | `SQUEEZE_OUT`, `SQUEEZE_SCRATCH`, `SQUEEZE_LOGS` | under `/work/$SLURM_ACCOUNT/$USER/marEx` and `/scratch/<initial>/$USER/marEx` |
| ICON grid file | `--grid-file` or `MAREX_ICON_GRID` | the DKRZ `/pool` copy |
| EERIE catalogue | `MAREX_EERIE_CATALOGUE`, or `--input` to skip it | the public catalogue URL (the `dkrz.disk.model-output` entry is DKRZ-only) |

The partitions and per-leg budgets in `slurm/submit.sh` are Levante's. On another cluster, keep the
guard rails, above all the memory-limit assertion, and change the partitions, the input stores and the
budgets.

## Smoke Runs

`SMOKE=1` truncates each script's series (64 steps for the trackers, 365 for detect, unless `--nt`
is given), sets a 10 minute self-deadline and, in `submit.sh`, a 15 minute wall. It checks the plumbing,
the paths and the asserted memory limit, not any result in this document. Give a smoke leg a method
that fits its short series, for example `--method-anomaly detrend_harmonic --method-extreme global_percentile`
for detect.
