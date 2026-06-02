# Why "Small object filtering" is slow — analysis & optimisation

**Scope:** `marEx.tracker` preprocessing, the `log_timing(logger, "Small object filtering")`
block in `marEx/track/tracker.py:770-780` (you see `Starting Small object filtering` then a
long pause). Reproduced on DKRZ Levante against the real dataset
`/scratch/b/b382615/mhws/extremes_binary_gridded_shifting_hobday.zarr`
(9282 daily steps × 720 × 1440, 0.25° global), 32-worker `LocalCluster`, notebook params
(`R_fill=12, T_fill=4, area_filter_absolute=600`). Subset timings at NT=1500 slices;
full run ≈ 6.2×.

> **CORRECTION to the first version of this note.** My initial diagnosis (per-slice
> `regionprops`/`dask_image.label` *max-label pathology* dominating) was measured by feeding
> **raw** `extreme_events` straight into `identify_objects`, which **skipped `fill_holes`**.
> In the real pipeline `fill_holes` runs first and collapses the object count from ~3.88M
> (raw, ~1295/slice) to ~25k (~17/slice), so the max-label pathology **never triggers**.
> The real bottleneck is the **deferred morphology** (`fill_holes`/`fill_time_gaps`), which is
> **compute-bound**. The per-slice label fix still helps (see §3.2) but was not the main thing.

## 1. What actually runs inside "Small object filtering"

Because `fill_holes` is **lazy** (tracker.py:758, logged 0.08s = graph-build only) and the
subsequent `.persist()` is **non-blocking with no `wait()`** (tracker.py:765, logged 0.37s =
submission only), the spatial- and temporal-gap-filling **compute is deferred** and only
forced when `filter_small_objects` → `identify_objects` calls `N_objects.compute()`. So the
"Small object filtering" timed block actually executes **three** operations:

1. **Spatial hole filling** — `fill_holes` (morphology.py:158-223): pad by `diameter=2R=24`
   (`mode="wrap"`), then `dask_image.ndmorph.binary_closing` **then** `binary_opening` with a
   **disk-radius-12 (25×25) structuring element** = 4 morphological passes over 9282×768×1488.
2. **Temporal gap filling** — `fill_time_gaps` (morphology.py:226-298): temporal closing
   (kernel `T_fill+1=5`) **then a second `fill_holes`** with `R_fill//2=6`.
3. **Small-object filtering** — `identify_objects` (`dask_image.label`) + `regionprops` +
   `isin`.

## 2. Measured attribution (NT=1500, 32 workers)

Isolated (forced compute per step with `wait()`):

| Step | Time | Share | Full-run projection |
|---|---:|---:|---:|
| `fill_holes` (spatial morphology) | 70.5s | 47% | ~7.3 min |
| `fill_time_gaps` (temporal + 2nd fill_holes) | 35.3s | 23% | ~3.6 min |
| `filter_small_objects` (label+regionprops+isin) | 44.8s | 30% | ~4.6 min |
| **Total** | **150.6s** | | **~15.5 min** |

End-to-end on the real **fused** (deferred, un-`wait`-ed) graph: the block = **119.8s** for
NT=1500 → **~12 min** at full scale. Matches the observed "~14 min and counting".

**The morphology is ~70% of the block** and was entirely hidden inside the "Small object
filtering" label.

### Why ~1–4 CPUs only
- Binary morphology over ~10 GB bool arrays is **memory-bandwidth-bound**; 32 processes
  contend for bandwidth, so effective parallelism is a few cores.
- `dask_image.label`'s global cross-block relabel phase is **serial** (1 core).
- `fill_holes` emits **~393 tasks/chunk** (23,597 tasks for 60 chunks → ~146k for the full
  run), adding scheduler overhead — but see §3.1: this is *not* the dominant cost.

## 3. Optimisation strategies (benchmarked, NT=1500)

### 3.1 Morphology (~70%) — it is compute-bound, not scheduler-bound

| Approach | Time | Byte-exact vs current? | Speedup |
|---|---:|:--:|---:|
| `dask_image` (current) | 65.7s | (reference) | 1× |
| per-slice `scipy` morphology (`apply_ufunc`, 1 op/slice; the **already-written-but-disabled** `use_dask_morph=False` path) | 48.1s | **yes (diff=0)**, tasks 23,597→677 | 1.4× |
| per-slice **distance-transform** (EDT: dilate=`EDT(~b)²≤R²`, erode=`EDT(b)²>R²`) | 17.8s | no — **0.06%** near lon seam | **3.7×** |

Collapsing the task graph 35× (scipy path) gave only **1.4×** → the cost is the **brute-force
625-element disk SE**, i.e. compute-bound. The EDT is O(N) regardless of radius → 3.7×.

**The EDT "0.06% difference" is the *current code's* bug, not EDT error.** Convergence test
(diff = number of differing pixels):

| Comparison | Diff px | Meaning |
|---|---:|---|
| scipy(2R pad) vs current `dask_image` | **0** | per-slice scipy == current, byte-exact |
| EDT(2R) vs scipy(2R) | 21,432 | boundary band at 2R pad |
| **EDT(4R) vs scipy(4R)** | **0** | **EDT *is* the exact Euclidean-disk closing/opening** |
| scipy(4R) vs scipy(2R) | 23,409 | the current code's under-padding artifact |
| EDT(4R) vs current `dask_image` | 23,409 | EDT(4R) differs from current *only* by that artifact |

The current code pads only `2R=24` while a closing+opening reaches `4R=48`, leaving an
artifact in a 24–48-cell band around the antimeridian. With `4R` padding EDT and scipy agree
bit-for-bit, so **EDT(4R) is both 3.7× faster and more correct** than the current code (it
removes the seam artifact). `np.rint` on squared distances is irrelevant — the diff was never
float rounding. (4R padding is only ~13% more pixels → EDT stays ~3.7×.)

### 3.2 Filter (~30%) — dominated by `dask_image.label` *fixed overhead*, not object count

On hole-filled data (only 25,657 objects):

| Sub-step | Time | Note |
|---|---:|---|
| `dask_image.label` | 40.8s | ~85% of the filter — fixed O(pixels)+global-relabel overhead, *independent of object count* |
| `regionprops` | 5.4s | fast (few objects; max-label pathology absent) |
| `isin`+persist | 1.6s | |
| **per-slice label + `bincount` + keep-LUT** | **1.8s** | **26× faster than the whole filter** |

The per-slice `scipy.ndimage.label` + `np.bincount` fix avoids `dask_image.label`'s ~40s of
fixed overhead. (On hole-filled data it currently differs by 8,723 px / 0.007% from the
`dask_image` wrap-labelling — a lon-seam edge case; it was byte-exact on raw data, so the
seam union-find needs one more reconciliation pass.)

### 3.3 End-to-end (fused, NT=1500)

| Pipeline | Block time | vs baseline |
|---|---:|---:|
| Baseline (current) | 119.8s | 1× |
| EDT morphology + per-slice filter | **44.9s** | **2.7×** (final output diff 0.067%) |

A byte-exact variant (per-slice **scipy** morphology + per-slice filter, with the seam
reconciliation) would land around ~1.7–2× — slower than EDT but with no numerical change.

### 3.4 Free clarity fix (no speed change)
Add `.persist(); wait()` after the spatial- and temporal-fill blocks (or drop the deferred
non-blocking persist) so each step's time is attributed to its own `log_timing` label instead
of all collapsing into "Small object filtering". This alone would have made the bottleneck
obvious.

## 4. Recommendation
1. **Biggest win — speed up `fill_holes`** (the ~70%), compute-bound. The earlier
   speed/exactness fork **dissolves** (§3.1): EDT distance-transform morphology with `4R`
   padding is **3.7× faster and exact** (and fixes the current antimeridian under-padding
   artifact). If exact reproduction of *current* output bit-for-bit is required instead, the
   per-slice `scipy` path (`use_dask_morph=False`, 2R pad) is byte-exact at 1.4× with a 35×
   smaller graph.
2. **Keep & finish the per-slice filter fix** (§3.2): 26× on the filter portion; reconcile the
   0.007% lon-seam edge case to byte-exact (same diagonal-8-connectivity logic already used).
3. **Add `wait()` after the morphology persists** for correct timing attribution.

Combined fused speedup measured at **2.7×** (≈12 min → ≈4.5 min at full scale) using EDT
morphology + per-slice filter; the bulk of the remaining ~45s is still morphology, so further
gains there compound.

## 5. STATUS: implemented & committed
EDT-4R morphology + per-slice filter + `wait()` were implemented and committed (`3dd5af3`,
`clean_up`). Verified on the full 9282-day run: **"Small object filtering" 14 min → 17.5 s**;
total preprocessing ≈ 5.4 min. Test impact (all from the intentional EDT-4R seam change):
golden snapshots + 2 `test_gridded_tracking` reasonable-range bounds regenerated to the new
(correct) behaviour; most notably `total_merges` 13 → 29 because correct antimeridian
seam-closing reshapes seam objects → more temporal overlaps.

---

# Part B — Step 2/3 "Object identification and tracking" (~40 min)

### B0. FULL-SCALE hang (the user's actual symptom): `identify_objects` graph build — FIXED

At the **full 9282-day** scale the run hangs at "Starting Object identification and tracking" for
>30 min on a **single CPU**, before "Finished object identification". A `faulthandler` stack dump
showed the main thread stuck in **graph construction** (not compute):
`dask_image/ndmeasure/__init__.py:374 label → _label.py:170 label_adjacency_graph →
_across_block_label_grouping_delayed → dask.delayed.__call__ → __dask_graph__ → _task_spec.substitute`.
`identify_objects(time_connectivity=False)` calls `dask_image.ndmeasure.label`, which builds a
cross-block adjacency graph via `dask.delayed`; constructing those delayed objects re-materialises
the growing graph per block-pair → **~O(n_time_blocks³)**, single-threaded client-side. Measured
graph-build (no compute): 20 blk=0.81 s, 40 blk=2.37 s, 80 blk=21.4 s. At chunks `time=25` the
9282-day run = **372 blocks** → ~30–40 min. (This was never exposed by the subset profiling below,
which used ≤120 blocks where the cubic build is seconds — the same "subset hides full-scale" trap as
the morphology investigation.) With `time_connectivity=False` there are no real cross-time links, so
the entire adjacency machinery is wasted.

**FIX (implemented, objects.py):** for the structured `time_connectivity=False` branch, label each
2D slice with `scipy.ndimage.label` (8-conn + periodic-lon seam merge) and offset by the running
cumulative object count (mirrors the unstructured branch). No cross-block adjacency → tiny graph.
Measured: **18.7 s** at full 9282 scale (was a ~30–40 min hang); object count **byte-identical** to
`dask_image` (diff=0 on a 1000-slice raw check). End-to-end the change is **bit-for-bit identical**:
the full track test suite — including the bit-exact `test_track_golden` snapshots — passes unchanged
(`cluster_rename_objects_and_props` canonicalises event IDs independent of the intermediate label
numbering), so **no golden regeneration was needed**. `time_connectivity=True` (non-merging) keeps
`dask_image.label` and would hit the same cliff — mitigate with larger time chunks if needed.

### B1. Sequential merge/split loop (next bottleneck, subset-profiled, NOT yet implemented)

After preprocessing + B0, the next bottleneck is `split_and_merge_objects`. Profiled at NT=1000
slices (32 workers):

| Sub-step | Time |
|---|---:|
| `identify_objects` (now per-slice) | ~18 s @ full scale |
| `calculate_object_properties` (area+centroid) | 2.6 s (fine — few objects) |
| **`split_and_merge_objects`** | **719.7 s / 1000 slices** |
| `cluster_rename_objects_and_props` | (memory-heavy; secondary) |

`split_and_merge_objects` is the bottleneck (~0.72 s/timestep → ~1.8 h projected for 9282).
It is the inherently-sequential per-timestep merge/split loop (`merge_split.py:583`). cProfile of
that loop (NT=300, 200 s total):

| Hot spot | tottime | cumtime | Share |
|---|---:|---:|---:|
| **`partition_nn_grid`** (`partitioning.py:163`) | 102.6 s | 102.7 s | **51 %** |
| **`object_properties_chunk`** (per-merge full-slice `regionprops`) | 15.0 s | 36.1 s | **18 %** |
| `nonzero` (masks / regionprops) | 8.3 s | | 4 % |
| xarray `where`/`concat`/`array_eq`/`isinstance` (per-merge `object_props` edits) | ~8 s | ~15 s | ~8 % |
| `check_overlap_slice` / `enforce_overlap_threshold` | ~2 s | ~5 s | ~3 % |

### Recommended optimisations (NOT yet implemented — diagnosis only)
1. **`partition_nn_grid` (51 %).** The numba grid NN search blows up because the search radius
   `max_distance = 3·√(max parent area)` (clamped ≥ 40) becomes huge for large merged blobs, so
   each child pixel scans a large grid neighbourhood of parent points. This is a **nearest-labelled
   -region** assignment: build a parent-label image (parent *i*'s pixels = *i*+1) over the bounding
   box of (child ∪ parents) + margin, call `scipy.ndimage.distance_transform_edt(parent==0,
   return_indices=True)`, and read off the nearest parent label for each child pixel — O(N) over the
   bbox regardless of radius. Handle the periodic-longitude seam by rolling/padding the bbox (as in
   the morphology fix). Expected ~10–50× on this hot spot. Must match `partition_nn_grid`'s
   `max_distance` cutoff + centroid fallback for unreachable pixels; golden test guards correctness.
2. **Per-merge `calculate_object_properties(data_t)` (18 %) + `object_props` edits (~8 %).** Each
   partitioned child recomputes `regionprops` over the *whole* slice and grows `object_props` via
   `xr.concat`/`loc`/`drop_sel` (O(N²) over the run). Compute only the *new children's* area+centroid
   locally from `child_mask_2d` + `new_labels` (already in memory), and accumulate props in a
   numpy/dict structure converted to a Dataset once, instead of per-merge xarray concat.

Combined, these target ~70 % of `split_and_merge_objects`. `cluster_rename_objects_and_props`
(area/centroid + adjacency-ledger recompute) is a separate, memory-heavy step worth profiling next.

## Related (separate)
`calculate_object_properties(..., properties=["area","centroid"])` in the tracking hot loop shares
the regionprops/max-label pattern but here it is cheap (few objects post-filter); not a concern.

## Reproduction scripts (`$CLAUDE_JOB_DIR/tmp/`)
`bench_realpath.py` (per-step attribution), `bench_morph.py` (morphology fork),
`bench_edtexact.py` (EDT boundary diff), `bench_filterdecomp.py` (filter decomposition + label fix),
`bench_e2e.py` (end-to-end fused). Earlier raw-data scripts (`bench_regionprops.py`,
`bench_attrib.py`, `bench_perslice.py`, `bench_seamfix.py`, `bench_full5tuple.py`) measured the
max-label pathology that the real pipeline avoids — kept for the record.
