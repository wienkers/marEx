# Optimising `split_and_merge_objects`: an EDT feature-transform `partition_nn_grid`

## 1. Why — profiling evidence

After the preprocessing fix (`3dd5af3`/`9472f88`) and the full-scale `identify_objects` fix
(`02af763`), the dominant remaining cost in `marEx.tracker` is **`split_and_merge_objects`**
(`marEx/track/merge_split.py:583`), the inherently-sequential per-timestep merge/split loop.

Measured (32-worker `LocalCluster`, DKRZ Levante, real 0.25° global data):

- `split_and_merge_objects` ≈ **719 s / 1000 slices** → **~1.8 h projected at the full 9282-day run**.
  (Each timestep ~0.72 s; the loop is sequential by construction — temporal dependencies.)

`cProfile` of the loop (NT=300, 200 s total, sorted by `tottime`):

| Hot spot | tottime | cumtime | Share |
|---|---:|---:|---:|
| **`partition_nn_grid`** (`partitioning.py:163`) | 102.6 s | 102.7 s | **51 %** |
| **`object_properties_chunk`** (per-merge full-slice `regionprops`, `objects.py:469`) | 15.0 s | 36.1 s | **18 %** |
| `numpy.nonzero` (masks / regionprops) | 8.3 s | | 4 % |
| xarray `where`/`concat`/`array_eq`/`isinstance` (per-merge `object_props` edits) | ~8 s | ~15 s | ~8 % |
| `check_overlap_slice` / `enforce_overlap_threshold` (`overlap.py`) | ~2 s | ~5 s | ~3 % |

`partition_nn_grid` is **51 %** and is the primary target; the per-merge property recompute (18 %)
and the growing-`object_props` xarray edits (~8 %) are clear secondary targets.

---

## 2. Root cause inside `partition_nn_grid`

`partition_nn_grid` (`marEx/track/partitioning.py:163`, numba `@jit(parallel=True)`) assigns every
pixel of a merged **child** blob to the **parent** (at *t-1*) whose nearest pixel is closest, under a
search radius `max_distance`, with a fall-back to the nearest parent *centroid* for pixels that have
no parent pixel within `max_distance`.

Call site (`merge_split.py:791-817`, structured grid):

```python
max_area = np.max(object_props.sel(ID=parent_ids).area.values)
max_distance = int(np.sqrt(max_area) * 3.0)            # 3× the max blob "radius"
new_labels = partition_nn_grid(
    child_mask_2d,           # (ny, nx) bool — the merged child blob
    parent_masks,            # (n_parents, ny, nx) bool — each parent at t-1
    child_ids,               # (n_parents,) int32 — IDs to hand out (child_id + new IDs)
    parent_centroids,        # (n_parents, 2) float
    Nx,
    max_distance=max(max_distance, 40),                # clamp; in cells
    wrap=not regional_mode,                            # periodic longitude
)
```

**The cost:** the algorithm builds a coarse spatial grid index of each parent's pixels and, for every
child pixel, scans the 3×3 neighbourhood of grid cells (cell size `max_distance // 4`) computing
`wrapped_euclidian_distance` to each parent pixel found. Cost ≈
`O(n_child_pixels × n_parents × points_per_3×3_grid_neighbourhood)`. Because
`max_distance = 3·√(max parent area)` **scales with the parent size**, large merged blobs (exactly the
ones that trigger partitioning) make the grid cells huge → each child pixel examines a large
neighbourhood of parent points → ~84 ms/call, 1221 calls/300 slices, **51 %** of the loop.

---

## 3. The fix — exact nearest-parent via a distance-transform feature map

The task "assign each child pixel to the nearest parent pixel" is exactly a **nearest-labelled-region
(feature transform)** problem, solvable in **O(bounding-box area)** *regardless of `max_distance` or
`n_parents`* with `scipy.ndimage.distance_transform_edt(..., return_indices=True)`.

### Algorithm

1. **Bounding box.** Compute the bbox of `child_mask ∪ parent_masks`, expanded by a `max_distance`
   margin (so any parent pixel that could win is included), clamped to the grid in latitude. Work on
   this sub-array only — the whole point is to avoid touching the full 720×1440 field per merge.
2. **Parent-label image.** Allocate `parent_label = np.zeros(bbox, int32)`; for parent `p`
   (`0..n_parents-1`) set its bbox pixels to `p + 1` (0 = background). (If two parents overlap a
   pixel — they shouldn't, parents are disjoint at *t-1* — last-writer is fine.)
3. **Feature transform.**
   ```python
   dist, (iy, ix) = distance_transform_edt(parent_label == 0, return_distances=True, return_indices=True)
   nearest_parent = parent_label[iy, ix] - 1          # parent index for every bbox pixel (-1 where none)
   ```
   `dist[p]` is the exact Euclidean distance to the nearest parent pixel; `(iy, ix)` indexes it.
4. **Assign child pixels.** For child pixels (within the bbox) with `dist <= max_distance`:
   `parent_idx = nearest_parent[child pixel]`. For child pixels with `dist > max_distance` (or no
   parent in the bbox): **centroid fall-back** — nearest of `parent_centroids` by
   `wrapped_euclidian_distance` (reuse the existing helper / a vectorised numpy version).
5. **Output.** Return `child_ids[parent_idx]` ordered by `np.nonzero(child_mask)` — the exact contract
   the caller expects (`temp[child_mask_2d] = new_labels`).

### Periodic-longitude wrap (`wrap=True`, non-regional)

`distance_transform_edt` is **not periodic**. Two correct options (pick per-call):

- **Common case — object away from the seam:** the bbox does not touch both lon edges → plain EDT, no
  wrap needed.
- **Seam-straddling case:** either (a) **roll** the lon axis so the child ∪ parents are contiguous
  (find an unoccupied longitude band, `np.roll` the bbox columns, EDT, roll back), or (b) **pad** the
  lon dimension of the bbox by `max_distance` using periodic (`wrap`) padding, EDT, then trim — the
  same trick used in the EDT morphology fix (`fill_holes`). Rolling is cheaper and exact for objects
  that don't span the full globe; padding is simpler to reason about. Recommend rolling, with a
  guard that falls back to full-width-with-wrap-pad if the object occupies all longitudes.

### Why this is faster

- EDT is `O(bbox area)`, **independent of `max_distance`** — kills the blow-up for large blobs.
- One C-level `distance_transform_edt` call replaces the per-pixel × per-parent grid search.
- No numba compile/parallel overhead per call.
- Expected ~10–50× on `partition_nn_grid`; if the 51 % drops to ~2–5 %, the loop is ~1.8–2× faster
  on its own (before the secondary fixes below).

### Correctness notes (IMPORTANT — not bit-exact)

- **Definition is identical**: "nearest parent = parent owning the closest pixel within
  `max_distance`, else nearest centroid". EDT and the grid search agree everywhere **except at exact
  ties** (a child pixel equidistant to two parents): `scipy` picks one feature deterministically by
  its scan order; the grid search picks the first parent in index order. These can differ for a
  handful of boundary pixels per merge → the partition is *physically equivalent* but **not
  bit-for-bit** the same.
- Consequence: unlike the `identify_objects` fix (which was bit-exact because `cluster_rename`
  canonicalises IDs), this **will shift the bit-exact `test_track_golden` snapshots** and may nudge
  `total_merges` / event counts slightly. Plan to **regenerate the golden baselines** and re-verify
  the `test_gridded_tracking` reasonable-range bounds (same procedure as commit `9472f88`).
- Verify equivalence directly: on a sample of real merges, compare `partition_nn_grid` vs the EDT
  version pixel-by-pixel; expect agreement except a thin boundary set (report the % differing, expect
  ≪ 1 %). Keep the old numba kernel available behind a flag during validation.

---

## 4. Secondary targets (same loop)

### 4a. Per-merge property recompute (18 %)

After each child is partitioned, the loop calls `calculate_object_properties(data_t, ...)`
(`merge_split.py:846`) — a **full-slice `regionprops_table`** — to refresh properties, but only the
newly-created children's `area`/`centroid` are actually needed. Replace with a **local** computation
from `child_mask_2d` + `new_labels` (which are already in memory): `np.bincount` for areas and an
area-weighted centroid (with the same periodic-lon handling as `calculate_centroid`), computed only
for `child_ids`. Eliminates the per-merge full-slice regionprops (`find_objects`, `coords_scaled`,
`image`, `nonzero` in the profile).

### 4b. Growing `object_props` via xarray (~8 %)

`merge_split.py:862-875` updates `object_props` per merge with `object_props.loc[...] = ...`,
`object_props.drop_sel(...)`, and `xr.concat([object_props, ...], dim="ID")`. Repeated `xr.concat`
on a growing `Dataset` is **O(N²)** over the run, plus heavy xarray/`isinstance` overhead per call.
Accumulate new/updated props in plain numpy arrays or a dict keyed by ID and build/refresh the
`object_props` `Dataset` **once** (or in bulk, infrequently) instead of per merge.

### 4c. (Optional) localise the masks

`(data_t == child_id)` and `(data_t_minus_1 == parent_id)` (`merge_split.py:737,802`) scan the full
slice per child/parent. Restricting to the merge bbox (already needed for 3a) trims `numpy.nonzero`
/comparison cost (the 4 % `nonzero` line).

---

## 5. Scope, risk, and test plan

- **Scope:** `marEx/track/partitioning.py::partition_nn_grid` (structured grid) and its call site in
  `merge_split.py`. The unstructured `partition_nn_unstructured` (`partitioning.py:307`) is a
  separate graph-traversal kernel — leave it, or treat as a follow-up.
- **Risk:** intricate, correctness-critical code (the merge/split core). The `test_track_golden`
  snapshot + `test_gridded_tracking` reasonable-range tests are the safety net.
- **Validation steps:**
  1. Stand up the EDT kernel beside `partition_nn_grid`; unit-compare outputs on a battery of
     synthetic merges (1 child / 2–5 parents; seam-straddling; large `max_distance`; pixels beyond
     `max_distance` → centroid fallback). Quantify the boundary-tie disagreement (%).
  2. Benchmark on real data: `cProfile` `split_and_merge_objects` (NT=300) before/after — confirm the
     `partition_nn_grid` share collapses and total loop time drops.
  3. Run the full track suite. Expect `test_track_golden` bit-exact tests to fail (tie-break) →
     **regenerate** `tests/data/track_golden_{events,merges}.nc` and recentre any shifted
     `test_gridded_tracking` reasonable-range bounds (procedure as in commit `9472f88`); confirm
     `total_merges` / `N_events` stay within physically-reasonable ranges (equivalent, not identical).
  4. Full-scale smoke run (9282 days) to confirm `split_and_merge` drops from ~1.8 h to target.
- **Projected impact:** `partition_nn_grid` 51 % → ~few %, plus 4a/4b removing ~25 % → loop roughly
  **3–4× faster** (~1.8 h → ~30 min for the merge/split step). Exact figure to be measured.

## 6. Implementation order

1. EDT `partition_nn_grid` replacement (bbox + feature transform + wrap handling + centroid fallback),
   behind a temporary flag for A/B validation. — biggest win.
2. Local per-merge property computation (4a).
3. Replace per-merge xarray `object_props` edits with bulk numpy accumulation (4b).
4. (Optional) bbox-localise the per-child/parent masks (4c).
5. Regenerate golden baselines + adjust reasonable-range bounds; full-scale smoke test.

## Reproduction
Profiling scripts used for the numbers above lived in the job tmp dir
(`prof_tracking.py`, `cprofile_sam.py`). Re-derive with a `cProfile` of
`tracker.split_and_merge_objects` on a ~300-slice preprocessed subset.
