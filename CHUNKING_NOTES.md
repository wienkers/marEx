# Chunking marEx: rules of thumb, gotchas, and what to try

How to chunk data going into `preprocess_data` (detect) and `tracker` (track), for both
gridded and unstructured grids, and how to think about the chunking that comes back out.

**Every claim here is tagged.** *[measured]* means it comes from an actual run on real data,
with the configuration given. *[reasoned]* means it follows from the code but has not been
measured at scale. Do not promote a *[reasoned]* row to a number you quote.

---

## 1. The one rule, if you read nothing else

> **`detect` wants the spatial dimension CHUNKED. `track` wants it WHOLE.**

They are not inconsistent, they are opposite by nature:

- `detect`'s reductions (climatology, percentile, threshold) are **global in time and
  independent in space**. Every cell's time series is processed on its own, so splitting
  space is free and splitting time is expensive.
- `track`'s core operations — connected-component labelling, the dilation/adjacency matrix,
  merge/split resolution — are **global in space**. An object can span any part of the
  domain, so a spatially split array would cut objects in half.

So the same dataset wants different chunking at the two stages, and **rechunking between
them is correct, not a mistake to be optimised away.**

This is the single most common error. In `examples/unstructured data/`, notebook 01 uses
`{"time": 21, "ncells": 25000}` and notebook 02 uses `{"time": 4, "ncells": -1}`. That
inconsistency is **deliberate and must not be "fixed"**.

---

## 2. `detect` (`preprocess_data`)

### 2.1 The sizing formula that matters

The hobday/quantile reductions rechunk internally to *time-whole* tiles. So the working set
of one task is set by your **input spatial chunk width**, not your time chunk:

```
per-task working set  ≈  n_time × cells-in-one-INPUT-spatial-chunk × 4 B
```

Note `n_time` here is the *whole* series, because the reduction needs it whole. Your input
time chunk does not reduce this — it only changes how much concatenation is needed to build
the tile.

### 2.2 Gridded

| knob | guidance |
| --- | --- |
| spatial (`lat`/`lon`) | **Chunk it.** Aim for tiles of ~10,000–30,000 cells. |
| time | Modest, but **`>= smooth_days_baseline`**. 30–90 is a reasonable band. |
| never | `{"lat": -1, "lon": -1}` on a large grid *with* a long series — see §2.4. |

A 0.25° global grid is 720×1440 = 1.04 M cells. Tiles of 90×180 (16,200 cells) or 60×120
(7,200) both sit in the target band.

*[measured]* OSTIA 0.25°, 20 y subsampled to 180×360, `{time: 73, lat: -1, lon: -1}`,
`shifting_baseline` + `hobday_extreme`, 6 workers × 14 GB × 1 thread: **completed in 321 s**
(`persist`) / **503 s** (`streaming`). At this reduced grid the whole field is only 64,800
cells, so spatially-whole is *within* the target band — that is why it works here and would
not at full resolution.

*[measured]* OSTIA 0.25° full resolution, 29.4 GB input, 8 workers × 12 GB: full pipeline
**2.47 h**.

### 2.3 Unstructured

Same rule, one dimension:

| knob | guidance |
| --- | --- |
| `ncells` | **Chunk it.** ~25,000 cells is the shipped example's value. |
| time | Modest, `>= smooth_days_baseline`. |
| `neighbours`, `cell_areas` | Do **not** constrain input chunking. They are attached to the *output* and rechunked independently. |

*[measured]* ICON R02B09, 14.9 M cells, 8 years (2922 steps), on-disk `{time: 2, ncells: 10M}`:

| input chunks | per-tile working set | outcome |
| --- | --- | --- |
| `{time: 21, ncells: -1}` | 2922 × 14.9 M × 4 B = **174 GB** | thrashed 5 h 40 m, no progress |
| `{time: 21, ncells: 25000}` | 2922 × 25,000 × 4 B = **292 MB** | bounded; what the example ships |

**The failure is a cliff, not a slope.** The bad configuration does not run slowly and
finish — workers pause and resume, "unmanaged memory high" appears, and nothing progresses.

### 2.4 Match the store's layout

Aim for a chunking **reachable from the on-disk chunks by local merges/splits**. A one-pass
jump from a time-chunked store straight to `{time: -1, ...}` makes every output chunk depend
on every input chunk — an all-to-all shuffle.

*[reasoned]* If your store is `{time: N, lat: -1, lon: -1}` (very common) and you ask for
narrow spatial tiles, you are splitting every on-disk chunk many ways *and* merging along
time. Prefer spatial tiles that divide the grid evenly, and a time chunk that is a multiple
of the store's own.

Check what you actually have before guessing:

```python
ds = xr.open_zarr(path, chunks={})
print(dict(ds.sst.sizes), [c[0] for c in ds.sst.chunks], [len(c) for c in ds.sst.chunks])
```

---

## 3. `track` (`tracker` / `regional_tracker`)

| knob | guidance |
| --- | --- |
| spatial (`lat`/`lon` or `ncells`) | **WHOLE — `-1`.** Non-negotiable; see §1. |
| time | **This is your only knob.** Small: 4–25 timesteps. |

*[reasoned]* Because space must stay whole, one chunk is `time_chunk × n_cells`. On a 14.9 M
cell mesh a single timestep is ~60 MB as float32 and ~15 MB as the int32 ID field, so a time
chunk of 4 is ~240 MB. That is why the unstructured example uses `{"time": 4, "ncells": -1}`
while the gridded one can afford more.

### 3.1 The tracker does not stream (yet)

It holds the whole `n_time × space` ID field in memory, so today `n_time` is bounded by RAM.
The merge/split loop itself already works a window at a time, so this is a fixable ceiling
rather than a property of the algorithm — see §5.2.

---

## 4. Output chunking

`preprocess_data(dask_chunks=...)` sets the chunking of the returned dataset. Two things:

- **Chunk the output for the NEXT stage, not the current one.** The output of `detect` is
  usually the input of `track`, which wants space whole. Writing `detect` output with narrow
  spatial chunks means `track` must rechunk it back.
- *[measured]* **`compute_mode="streaming"` changes the returned `dat_anomaly`'s chunking.**
  Staging rechunks to uniform chunks before the zarr write (zarr rejects ragged chunks), so a
  downstream consumer sees a different layout than under `persist`. Values are bit-identical;
  the layout is not.

For zarr output, remember zarr requires **uniform** chunks (a smaller final chunk is allowed).
Several anomaly methods leave ragged chunking behind — `fixed_baseline`'s groupby produces
e.g. `(30, 30, …, 6, 24, 30, …)` along time — and `to_zarr` rejects that outright.

---

## 5. Which combinations handle larger-than-memory data?

| | `detect` (`preprocess_data`) | `track` (`tracker`) |
| --- | --- | --- |
| **gridded** | **Yes** — `compute_mode="streaming"` *[measured]* | **Not yet** — see §5.2 |
| **unstructured** | **Yes, by construction** — same code path *[reasoned: no at-scale streaming run]* | **Not yet** |

For `track`, "larger-than-memory" means **long in time**, not large in space: one spatial
field is always held whole, deliberately (§5.2).

### 5.1 `detect`: solved, with a caveat

`compute_mode` controls every materialisation site:

| mode | peak RAM | anomaly graph runs | disk |
| --- | --- | --- | --- |
| `persist` (default) | ~0.6 × input pinned *[measured, ICON]* | once | none |
| `lazy` | a few chunks *[reasoned]* | 2–3× *[reasoned]* | none |
| `streaming` | a few chunks *[measured]* | once | ~2 × input |

*[measured]* `persist` spilled **9.2–10.5 GB** to disk while `streaming` spilled **0.00 GB**,
in three independent runs across a 5.3× range of per-worker RAM — including one where
`streaming` ran **7× longer** and still spilled nothing.

*[measured]* Output is **bit-identical** between `persist` and `streaming`: `extreme_events`
0 of 354,715,200 elements differing; `thresholds` `max_abs_diff` 0.0 including its NaN mask.

**The caveat that remains for `detect`: the per-task floor.** `streaming` removes the
*pinned* ceiling. It cannot remove the memory one task needs, which is set by the algorithm
and is identical in every mode. If your per-worker `memory_limit` is below that floor, every
mode dies and no setting helps. Size workers against the per-task working set (§2.1) first,
then choose a mode.

### 5.2 `track`: not solved today, but less structural than it looks

**Scope first: space stays whole, by design.** Chunking the spatial dimension in `track`
would mean reworking connected-component labelling and the dilation matrix, and it is not
worth it — a single spatial field is small even at high resolution (0.25° global = 1.04 M
cells ≈ 4 MB as int32; ICON R02B09 = 14.9 M cells ≈ 60 MB). So the target for `track` is
**arbitrarily long in time**, at whatever resolution fits one field. That is the case that
matters in practice.

**The sequential-in-time loop is not the obstacle it appears to be.** A time-sequential
algorithm streams perfectly well provided it holds only a window, and the gridded loop
already does:

- it loads **one time chunk at a time** and processes timesteps within it;
- its scratch list of processed chunks is **flushed and truncated periodically**;
- the state carried between timesteps is the `t-1`/`t-2` slices plus per-event lists that
  grow with the **number of merge events**, not with `n_time × n_cells`.

**What actually caps it is the output accumulator.** Results are written back into a
whole-`n_time × space` dask array that is held in memory and re-persisted as the loop
proceeds. That is a ceiling in `n_time`, and it is a bounded piece of work to replace with
incremental writes to disk — the mechanism already exists in the sibling code path
(`update_object_id_field_zarr`), currently used only on unstructured grids.

Alongside it sit the same input-sized pins `detect` had before `compute_mode`: the
preprocessed binary field, the filled/filtered fields, and the ID field.

So the realistic goal is **"ID field on disk with windowed access"** plus the `compute_mode`
treatment for the remaining pins — not a rewrite of the algorithm.

*[measured]* The practical limit today: unstructured tracking of a 1096-timestep ICON run did
not complete on a single node — merge iteration 1 with 681 objects ran 4 h 22 m and had
processed 256 of 1096 timesteps. Note that run used the *parallel* path, so the existence of
a parallel/zarr code path does **not** by itself mean tracking is larger-than-memory.

### 5.3 How to scale today

**Scale horizontally, not out-of-core**, whenever the data fits in *aggregate* cluster RAM.

*[measured]* On ICON R02B09 (174 GB input) `detect` pinned 103.6 GB, i.e. **~0.6 × input**.
Extrapolating, a **1 TB input needs ~600 GB aggregate** — more than one node, but fine on the
multi-node configuration the example keeps commented out
(`start_distributed_cluster(n_workers=512, workers_per_node=64, node_memory=512)`).

Reach for `streaming` when the data does not fit *anywhere*, not merely when it does not fit
on one node.

---

## 6. Gotchas

**Peak and pinned are different quantities. Do not quote one for the other.**
*[measured]* On ICON, pinned was 0.6 × input. On a gridded OSTIA run, peak was **2.2 ×** input
while pinned was 0.34 ×.

**`P2PConsistencyError: No active shuffle with id=... found` is always a symptom.**
The chain is: an oversized task → worker crosses 95 % → nanny restarts it → the in-flight P2P
shuffle loses its state → the shuffle raises. Look for
`distributed.nanny.memory - WARNING - ... exceeded 95%` **above** the traceback before
believing you have hit a dask bug.

**dask is cgroup-blind under SLURM.** `LocalCluster` sizes workers from the *node's* memory,
not your allocation. Always pass `memory_limit` explicitly, and log the effective value.

**Aggregate RAM does not rescue an oversized task.** A single task larger than one worker's
limit kills the run no matter how many workers you add. Only a bigger *per-worker* limit or a
smaller task helps.

**The SLURM job cgroup also holds the client and scheduler**, not just workers. The task graph
for a large problem can need 10+ GB in the client alone, and that cost is the same in every
`compute_mode`. If a job is OOM-killed, check `sacct MaxRSS` against `ReqMem` before blaming
the thing you were testing.

**Time chunks smaller than `smooth_days_baseline` break the climatology's centred rolling
mean.** `smoothed_rolling_climatology` probes this up front and raises `ConfigurationError`
rather than dying mid-graph.

**Staging survives the call deliberately.** In `streaming` mode the returned Dataset reads
from the staged zarr, so it cannot be deleted on return. Write your output, then
`marEx.clear_staging(ds)`. *[measured]* The `atexit` backstop does **not** run on SIGKILL — a
wall-clock-killed run left **4.8 GB** orphaned — so sweep `scratch_dir` periodically.

---

## 7. What to try, in order, when a run will not complete

1. **Read the actual error's neighbourhood, not the final traceback.** Scroll up for
   `exceeded 95% memory budget`. The last exception is usually a downstream symptom.
2. **Compute the per-task working set** (§2.1) and compare it against one worker's
   `memory_limit` × `threads_per_worker`. This is the single most common miss.
3. **Check `sacct MaxRSS` vs `ReqMem`** to rule out the cgroup rather than the workers.
4. **Inspect the store's on-disk chunks** and see how far your request is from them (§2.4).
5. **Reduce `threads_per_worker` before reducing workers.** Threads multiply concurrent task
   memory within one limit; workers do not.
6. **Only then change `compute_mode`.** It addresses pinned data, which is a different
   problem from an oversized task.

### A counter-intuitive one worth stating explicitly

**Shortening a run does not reliably make it cheaper per task**, so it is a poor debugging
move. Internal tiling divides a fixed element budget by the length of the reduced axis, so a
shorter series yields a *larger* spatial tile — the opposite of the intuition that a smaller
run is a smaller run. Prefer reducing spatial resolution, or the tile size, over reducing the
number of years, and change **one** thing at a time: shrinking a failing run and having it
fail sooner tells you nothing, because you moved two variables.

The tile is now budgeted against **both** sides of the reduction — the slab it reads and the
per-cell output it writes (`n_bins` counts for a histogram, 366 for a per-day-of-year
percentile) — so neither exceeds the budget. Earlier versions bounded only the read, which
meant a series shorter than the per-cell output silently allocated an over-budget task. The
"shorter is not smaller" intuition above still applies to the tile *shape*; the unbounded case
does not.
