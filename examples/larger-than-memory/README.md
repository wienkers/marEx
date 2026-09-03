# Larger-than-memory squeeze demonstrations

These scripts exist to establish one claim that the unit test suite structurally cannot:

> a workload that does **not** complete under `compute_mode="persist"` **does** complete
> under `compute_mode="streaming"`, at the same memory budget.

They are deliberately **not** part of `pytest`. Each leg wants a whole batch allocation and
tens of minutes to hours of wall clock. `tests/test_compute_mode.py` and
`tests/test_track_compute_mode.py` remain the fast gate: they verify configuration wiring,
count the bytes actually pinned, and check cross-mode bit-identity on small fixtures. What
they cannot do is force a real memory squeeze, and so what they cannot show is feasibility.

## Two gates, never one run

Equivalence and feasibility are separate claims and must not share a job:

| gate | length | cluster | what it shows |
| --- | --- | --- | --- |
| **Equivalence** | short enough that both modes fit | comfortable | outputs identical across modes |
| **Feasibility** | long enough that `persist` cannot fit | squeezed | `persist` fails, `streaming` completes |

A feasibility leg has **no bit-identity reference by construction** — the `persist` side
produced no output to compare against. That is a property of the claim, not a gap in the
method, and the report says so rather than implying otherwise.

## Guard rails, and why each one is here

Every one of these exists because its absence has already produced a wrong answer.

- **The effective per-worker memory limit is asserted, not assumed.** Dask reads the host's
  total RAM, not the cgroup the batch system placed the job in. Without an explicit
  `memory_limit`, workers believe they own the whole node, the squeeze never binds,
  `persist` completes, and the leg "passes" for entirely the wrong reason. `build_cluster`
  queries the workers themselves and exits non-zero rather than run a meaningless test.
- **Failures are classified from evidence.** A wall-clock kill is not proof of an OOM, and
  slowness is not proof either. The runner watches for `KilledWorker`, `MemoryError`, and
  the nanny's own memory warnings, and reports `timeout_inconclusive` when none fired.
- **Bytes pinned are counted, not inferred.** An array is *still* a dask collection after
  `.persist()`, so `is_dask_collection` proves nothing. The accountant patches the three
  persist entry points and attributes every byte to the marEx line that requested it —
  including modules that did `from dask import persist`, which binds the original function
  and which a naive patch of `dask.persist` misses silently.
- **Spilling stays ENABLED** (dask's default). Disabling it looks like it would sharpen the
  result; measured, it does the opposite. A worker crossing `pause` can no longer spill back
  down and pauses permanently, so the cluster deadlocks; peak memory *rises*, because what
  would have spilled stays resident (169.5 GB with spilling off versus 118.7 GB with it on, on
  the same run); and it kills both modes, so nothing can be compared. `--no-spill` remains as
  an explicitly labelled control.
- **A breadcrumb summary is written before the work starts**, so a leg killed by the wall
  clock still leaves a record of what it attempted.
- **Squeeze by worker count and record length, never by absurd per-worker RAM.** Per-worker
  memory stays at 6–12 GB so per-task working sets remain comfortable and the only thing
  that can bind is aggregate cluster RAM.

## Sizing table

Byte counts are uncompressed `n_time × n_cells × itemsize`. "Slab" is the working set of one
internal reduction tile, `n_time × cells-in-one-input-spatial-chunk × 4 B` — the quantity that
decides whether `detect` runs at all, and the reason the spatial dimension must be *chunked*
for `detect` and left *whole* for `track`.

### Inputs

| leg | source | dimensions | dtype | input size | input chunk | chunk bytes | slab / whole field |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| **F1** gridded detect | `mhws/ostia.zarr` `sst` | 14761 × 720 × 1440 | f32 | **61.2 GB** | `{time:30, lat:90, lon:180}` | 1.94 MB | slab **0.96 GB** |
| **F2** unstructured detect | EERIE ICON-ESM-ER hist-1950, 8 yr | 2922 × 14,886,338 | f32 | **174.0 GB** | `{time:21, ncells:100_000}` | 8.40 MB | slab **1.17 GB** |
| **F3** gridded track | F1's output (`window_years=15` consumes 15 yr) | 9282 × 720 × 1440 | bool | 9.6 GB | `{time:25, lat:-1, lon:-1}` | 25.9 MB | int32 field **38.5 GB** |
| **F4** unstructured track | `mhws/extremes_binary_unstruct_*` | 1096 × 14,886,338 | bool | 16.3 GB | `{time:4, ncells:-1}` | 59.5 MB | int32 field **65.3 GB** |

The full ICON hist-1950 record is 23741 days, i.e. **1.41 TB**; F2 takes the last eight years
so that a single squeeze leg finishes inside one allocation.

### Feasibility: MEASURED results, not predictions

An earlier version of this file carried *predicted* persist peaks obtained by scaling measured
coefficients. That method is unsound and the predictions were wrong -- **peak memory is
provisioning-dependent**: the same workload peaked 22.1 GB given a 32 GB budget and 57.0 GB
given 96 GB, because dask expands into available memory and releases under pressure. Size a
squeeze from an *arithmetic invariant* instead (the whole int32 field, `n_time x n_cells x 4 B`).

**The headline result -- gridded tracker, nt=3804, whole int32 field 15.8 GB, cluster 4 x 4 GB
= 16 GB:**

| leg | mode | outcome | evidence | peak |
| --- | --- | --- | --- | ---: |
| `g2_persist` | persist | **OOM-KILLED** | SLURM `OUT_OF_MEMORY`, `Detected 1 oom_kill event`, MaxRSS 23.83 GB | - |
| `g2_persist_r2` | persist | **OOM-KILLED** | same, MaxRSS 24.10 GB | - |
| `g2_stream` | streaming | **completed** | output identical to 32 GB and 192 GB runs | 7.3 GB |

**And the property that actually matters -- peak near-flat in series length, same 16 GB budget:**

| n_time | whole int32 field | peak | wall |
| ---: | ---: | ---: | ---: |
| 1902 | 7.9 GB | 6.9 GB | 849 s |
| 3804 | **15.8 GB** | **7.1 GB** | 1658 s |

**Unstructured tracker, nt=1096, cluster 32 GB:** persist failed to complete in 5 h twice
(under memory pressure); streaming completed in 3.84 h with identical output. A weaker result
than the gridded OOM kill, and labelled `timeout_with_memory_pressure` for that reason.

**`detect` does not squeeze, on either grid.** Peak is ~127 GB on a 9.1 GB input in *both*
modes while streaming pins 0.00 GB, so the ceiling is a mode-independent transient and
`compute_mode` cannot move it. See the report and roadmap.

Full results: `docs/superpowers/reports/REPORT_larger_than_memory_squeeze.md`.
Why it is not yet fully larger-than-memory, and what to fix:
`docs/superpowers/specs/2026-08-26-true-larger-than-memory-roadmap.md`.

### Equivalence legs (short, comfortable, only where a reference is missing)

| # | leg | length | cluster | modes |
| --- | --- | ---: | --- | --- |
| E1 | unstructured track | 256 | 16 × 12 GB | persist vs streaming |
| E2 | unstructured detect | 3 yr | 8 × 16 GB | persist vs streaming vs lazy |
| E3 | gridded detect | 3650 | 8 × 8 GB | persist vs streaming vs lazy |

Gridded-track equivalence is already established bit-identically at nt=3804 and is not repeated.

### Tracker setting variants

Carried on the cheapest leg of the right grid type rather than given their own scale run:
`--no-nn-partitioning` (centroid partitioning instead of the BFS kernel), `--no-allow-merging`
(skips the merge loop entirely), and `--R-fill 24 --T-fill 0` (morphology sensitivity).

## Running

```bash
./slurm/submit.sh preflight      # small probes: does each configuration run at all?
./slurm/submit.sh feasibility    # F1-F4, persist twice each
./slurm/submit.sh equivalence    # E1-E3
./slurm/submit.sh variants       # tracker settings
./slurm/submit.sh f1_stream      # or any single leg by name

python report.py /work/bk1377/b382615/marex_fable/measurements/lm
```

Pre-flight first, always. It is the cheapest insurance against burning a headline leg on a
failure that has nothing to do with `compute_mode`.

Each leg writes `<label>_summary.json` (dimensions, chunking, cluster budget, the *asserted*
effective per-worker limit, outcome, failure classification, peak and mean cluster memory,
bytes pinned per marEx source line, spill, nanny events, wall clock, output fingerprints) plus
a `<label>_memseries.npy` memory trace. `report.py` collates them into the results table.

## Adapting to another system

The scripts take every path as an argument; only `slurm/submit.sh` and the `DEFAULT_INPUT`
constants carry DKRZ Levante paths. On another cluster, keep the guard rails — particularly
the memory-limit assertion, which is what makes the result mean anything — and change the
partitions, the input stores, and the per-leg budgets.
