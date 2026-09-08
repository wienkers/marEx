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
- **The spill metric reports UNMEASURED rather than zero when it cannot read the number.**
  Until 2026-09-08 the sampler read an attribute that does not exist on `distributed` 2025.9.1
  and a bare `except` turned the resulting error into `0`, so every leg printed
  `spill 0.00 GB` whether or not anything had spilled -- a fabricated zero that was very nearly
  cited as evidence that streaming never touched disk. It now reads
  `worker.data.spilled_total.disk` and latches an `unmeasured` flag on any failure. Note what
  the number means: `spilled_total` is what is on disk *at that instant*, so the reported
  figure is the peak concurrent total on a 5 s sampling grid, a lower bound on the true peak,
  and never a cumulative "bytes ever spilled".
  Re-measured once the probe worked, the headline gridded-track leg (nt=3804, 4 x 4 GB) reports
  **0 bytes spilled across 324 successful samples**, with the metric flagged as measured: at a
  5 s interval over a 1626 s run that is essentially every interval, so the sampler demonstrably
  ran rather than silently failing. No sample ever caught bytes in the spill directory. That is weaker than "not one byte was written" -- a spill shorter than the
  sampling interval is invisible -- but it is the first spill figure this campaign has produced
  that is a measurement at all. It is one leg at one budget: every other leg predates the fix and
  its spill figure remains *unmeasured*, not zero.
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
| `g2_stream` | streaming | **completed** | five reductions match the 32 GB and 192 GB runs (`id_field_sum` 826033161263, 4388 events, 18712 merges) | 7.3 GB |

**And the property that actually matters -- peak near-flat in series length, same 16 GB budget:**

| n_time | whole int32 field | peak | wall | code |
| ---: | ---: | ---: | ---: | --- |
| 951 | 3.9 GB | 6.2 GB | 355 s | post-fix |
| 1902 | 7.9 GB | 6.9 GB | 849 s | pre-fix |
| 3804 | **15.8 GB** | **7.1 GB** | 1658 s | pre-fix |
| 3804 | **15.8 GB** | **6.8 GB** | 1600 s | post-fix |
| 3804 | **15.8 GB** | **6.7 GB** | 1626 s | post-fix |

Peak grows **7-9 %** while the field it is tracking grows **4x**: that, not the absolute number,
is the larger-than-memory property. Read that figure off the **post-fix** rows only (6.2 GB at
nt=951 against 6.7 and 6.8 GB at nt=3804), which are the like-for-like ones. Comparing across the
`fill_time_gaps` realignment fix instead gives 6.2 -> 7.1 GB, or 14 %, and that number mixes two
different versions of the tracker -- it is quoted here only so the discrepancy is not a surprise.

Wall clock is *not* cleanly linear over the same span -- 2.39x then 1.95x per doubling on the
pre-fix points -- and every row is n=1, taken on three different nodes, over a record whose
event count also grows (1130, 2142, 4388 events), so wall may be tracking work rather than
length. Read the peak column; treat the wall column as an order of magnitude.

Four replicates of the nt=3804 configuration exist and they spread 6.65 / 6.75 / 7.11 / 7.29 GB,
so treat differences below roughly half a gigabyte here as noise rather than signal.

**Unstructured tracker, nt=1096, cluster 4 x 8 GB = 32 GB:** `persist` did not complete
within 5 h on either of two replicates, while `streaming` completed in 3 h 50 min with matching
output reductions.

That is a *wall-clock* statement, and deliberately not more. Both persist replicates were
stopped by this harness's own deadline with roughly an hour of the SLURM wall still unused:
there was no kernel OOM, no SLURM `OUT_OF_MEMORY` and no `KilledWorker`, so "persist cannot
fit here" is **not** something this leg shows. Whether persist would finish given eight hours
is untested. The gridded leg above, where the kernel actually killed persist twice, is what
the headline claim rests on; the unstructured path is supporting evidence.

Nor is there a same-budget equivalence reference: `run_and_fingerprint` calls `clear_staging`,
so the ID field is deleted and no array comparison exists at any budget. What is checked is
five order-invariant reductions (4359 events, 9404 merges, `id_field_sum` 5589043195416,
`n_nonzero_cells` 3341563658, `max_id` 4359), and they match the 72 GB and 192 GB runs -- not
a persist run at 32 GB, which produced no output at all.

**`detect` does not squeeze, on either grid.** Peak is ~127 GB on a 9.1 GB input in *both*
modes while streaming pins 0.00 GB, so the ceiling is a mode-independent transient and
`compute_mode` cannot move it.

Every leg's raw numbers live in its `<label>_summary.json`; `report.py` collates them.

### Equivalence legs (short, comfortable, only where a reference is missing)

| # | leg | length | cluster | modes |
| --- | --- | ---: | --- | --- |
| E1 | unstructured track | 256 | 16 × 12 GB | persist vs streaming |
| E2 | unstructured detect | 3 yr | 8 × 16 GB | persist vs streaming vs lazy |
| E3 | gridded detect | 3650 | 8 × 8 GB | persist vs streaming vs lazy |

Gridded-track equivalence at nt=3804 is carried by the five order-invariant reductions matching
across the 16 / 32 / 192 GB runs, so it is not repeated here. That is reduction equality, not an
array comparison: the streaming staging directory is cleared at the end of each leg, so no
full-field reference survives at squeeze scale. The array-level check lives in the test suite,
at fixture scale and zero tolerance.

### Tracker setting variants

Carried on the cheapest leg of the right grid type rather than given their own scale run:
`--no-nn-partitioning` (centroid partitioning instead of the BFS kernel), `--no-allow-merging`
(skips the merge loop entirely), and `--R-fill 24 --T-fill 0` (morphology sensitivity).

## Running

Not every leg below is a live experiment. **F1 (gridded detect) was dropped**: measured, that
path is compute-bound rather than memory-bound -- both modes hit a 12000 s deadline at nt=2200
on 48 GB, and the specified leg is 6.7x that data -- so squeezing it demonstrates nothing about
`compute_mode`. F3 consumes F1's output store and is therefore blocked by construction, as are
the `v_merge_off` and `v_fill` variants; F2 is independent of F1 (it reads the EERIE catalogue
directly) and is simply unrun.
**F4 and the `g2_*` gridded-track legs are the validated ones**, and they are what the results
above rest on.

```bash
./slurm/submit.sh preflight      # small probes: does each configuration run at all?
./slurm/submit.sh feasibility    # F1-F4 (see the note above: only F4 is live), persist twice each
./slurm/submit.sh equivalence    # E1-E3
./slurm/submit.sh variants       # tracker settings
./slurm/submit.sh f1_stream      # or any single leg by name

python report.py /work/bk1377/b382615/marex_fable/measurements/lm
```

Pre-flight first, always. It is the cheapest insurance against burning a headline leg on a
failure that has nothing to do with `compute_mode`.

Each leg writes `<label>_summary.json` (dimensions, chunking, cluster budget, the *asserted*
effective per-worker limit, outcome, failure classification, peak and mean cluster memory,
bytes pinned per marEx source line, spill (`spill_max_disk_bytes`, `null` when
`spill_unmeasured`), nanny events, wall clock, output fingerprints) plus
a `<label>_memseries.npy` memory trace. `report.py` collates them into the results table.

## Adapting to another system

The scripts take every path as an argument; only `slurm/submit.sh` and the `DEFAULT_INPUT`
constants carry DKRZ Levante paths. On another cluster, keep the guard rails — particularly
the memory-limit assertion, which is what makes the result mean anything — and change the
partitions, the input stores, and the per-leg budgets.
