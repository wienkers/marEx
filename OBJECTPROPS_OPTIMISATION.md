# Scope: `split_and_merge_objects` secondary optimisation (object-property recompute + O(N²) accumulation)

## Status: Fix A + Fix B IMPLEMENTED & VALIDATED. Successor to the EDT `partition_nn_grid` fix (`5d08274`).

- **Fix A** (local per-merge child properties) — committed `bc4e881`. cProfile NT=300: split_and_merge
  112.8s → 79.3s (1.42×); `object_properties_chunk` removed from the profile. Bit-exact (golden
  unchanged).
- **Fix B** (full `ObjectPropsStore`, O(N²) accumulation killed across both loops + `overlap.py`) — this
  change. Gridded golden **bit-exact unchanged**; unstructured path validated; store unit tests added.
  Full-scale 9282-day validation (16×14 GB): `split_and_merge` loop 71.7 → 35.7 min (~2.0×), full
  `run()` 83.5 → 47.6 min (~1.75×); per-chunk time **flat** (≈4–6.6 s/chunk vs the prior 9→16 ramp);
  **behaviourally identical** — N_events=9280, total_merges=37964, 184643 valid IDs all reproduced exactly.

## Context

After the EDT `partition_nn_grid` fix (`5d08274`), `partition_nn_grid` dropped from 51 % → 25 % of the
sequential per-timestep `split_and_merge_objects` loop (`marEx/track/merge_split.py:583`). The
**now-dominant** costs in that loop, from `cProfile` (NT=300, 32-worker `LocalCluster`, total 112.8 s):

| rank | cost | cumtime | share | scaling |
| --- | --- | --- | --- | --- |
| 1 | `object_properties_chunk` (`objects.py:547`) — full-slice `regionprops_table` **per merge** | 33.9 s | **30 %** | ~flat in N |
| 2 | `partition_nn_grid` (already fixed) | 27.9 s | 25 % | flat |
| 3 | `np.nonzero` (mostly the per-object `ids == ID` masks inside #1) | 12.7 s | 11 % | flat (subsumed by #1) |
| 4 | `object_props` xarray accumulation: `xr.concat`/`.sel(ID=)`/`.loc`/`.drop_sel` on the growing Dataset | ~7–10 s | ~8 % | **O(N²)** — grows with N |

Full-scale 9282-day evidence: per-chunk wall time rose 9.7 → 12.5 → 15+ s/chunk and client RSS rose
linearly to ~33 GB — the signature of #4 (the structure grows; you keep re-touching all of it). #1 is the
flat baseline that dominates each chunk; #4 is the scale-dependent growth on top.

## Root causes (code-level)

### A. Per-merge full-slice property recompute (#1, #3)

At every merge, after partitioning the child blob (`merge_split.py:851`):
```python
new_child_props = _objects.calculate_object_properties(data_t, ..., properties=["area", "centroid"])
```
`data_t` is the **entire 2D slice (720×1440)**. Inside (`objects.py:547`), `object_properties_chunk`:
1. runs `regionprops_table(ids, ...)` over the whole slice — properties for **every** object, not just the
   new children;
2. then (`check_centroids`, the default for non-regional) loops over **every** object and computes
   `binary_mask = ids == ID` (a full 720×1440 boolean) + `calculate_centroid` — this is the 12.7 s of
   `np.nonzero`.

The result is used only to read off the handful of new children (`merge_split.py:867,876–878`). The same
full-slice call also fires inside `overlap.consolidate_object_ids` (`overlap.py:~360`).

**Key insight:** the new children are a *partition of the original child blob* (`child_mask_2d`), and their
fresh IDs exist *only* within that blob. So area+centroid for every new child can be computed directly from
the partition pixels already in hand (`np.nonzero(child_mask_2d)` + `new_labels`) — no full-slice
`regionprops`, no per-object full-slice masks.

### B. O(N²) `object_props` accumulation (#4)

`object_props` is an xarray `Dataset` keyed by ID (area, centroid). The hot loop mutates it with label-based
ops whose cost is O(current size), once per merge, while the size grows ~linearly:
- `merge_split.py:877` `object_props = xr.concat([object_props, new_child_props…], dim="ID")` — rebuilds the
  whole Dataset (dominant);
- `:867` `.loc[{"ID": child_id}] = …`, `:870` `.drop_sel(ID=…)`, `:769/782/810` `.sel(ID=parent_ids)`.

Same structure is read/written in `overlap.py`: `enforce_overlap_threshold` (`.sel(ID=…)`,
`set(object_props.ID.values)`) and `consolidate_object_ids` (`.drop_sel`, `.loc`).

## Proposed fixes

### Fix A — local property computation (kills #1+#3; ~30 % flat)

Add a helper that computes area+centroid for a *given set of IDs over their own pixels only*, and use it:
- in the merge loop (`merge_split.py:851`) for `child_ids = [child_id, *new_object_id]`, computed from the
  `child_mask_2d` pixels + `new_labels` (already materialised by the partition step);
- in `overlap.consolidate_object_ids` (`overlap.py:~360`) for the consolidated child id.

Mechanics: `np.add.at`/`np.bincount` over `new_labels` for area (pixel count → ×`cell_area` where the
existing code does); area-weighted mean of pixel coordinates for centroid, reusing `calculate_centroid`
(`objects.py`) for the antimeridian-wrap convention so results match the current definition.

- **Correctness:** area is bit-exact (same pixel set → same count). Centroid matches the current value
  modulo floating-point summation order (regionprops mean vs. local mean), which *may* shift the bit-exact
  golden snapshot by ≤1 pixel at a tie — same situation accepted for the partition fix; regenerate golden +
  verify physical equivalence (event/merge counts).
- **Risk:** medium-low. One new pure helper + two call-site swaps; no change to the `object_props`
  container, so the rest of the loop and `overlap.py` are untouched.

### Fix B — incremental `object_props` store (kills #4, the O(N²))

Replace the xarray `Dataset` used *inside the loop* with O(1)-update structures (dicts or preallocated numpy
arrays keyed by integer ID) for area + centroid; build the final xarray `Dataset` once after the loop.
Threads through `merge_split.py` (loop) **and** `overlap.py` (`enforce_overlap_threshold`,
`consolidate_object_ids`), so the helper signatures change.

- **Risk:** higher — correctness-critical and broader (two modules, the merge/split core). The 4 unit tests
  + golden snapshots + the full-scale smoke are the safety net.
- **Lighter variant B′:** keep xarray but accumulate new-child props in a Python list and `xr.concat` **once
  per chunk** instead of per merge. Removes the per-merge full-Dataset rebuild (the dominant part of #4)
  while leaving the `.sel`/`.loc` lookups; smaller, lower-risk, partial win. Does **not** require touching
  `overlap.py` signatures.

## Recommended path

**Fix A + B′** as the high-value / moderate-risk package: A removes the flat 30 % everywhere; B′ removes the
per-merge concat (the bulk of the O(N²) growth) without the broad `overlap.py` refactor. Reserve full Fix B
for a later pass if the residual `.sel`/`.loc` lookups still dominate at full scale after A+B′.

### DECISION (user, 2026-06-03): **Fix A + full B**, implement **after** the `da.unique` re-verify completes.

Go for the complete O(N²) kill: local property computation (A) **and** replace the in-loop xarray
`object_props` with O(1)-update structures (dict / preallocated numpy arrays keyed by integer ID) for area +
centroid across **both** `merge_split.py` (loop) and `overlap.py` (`enforce_overlap_threshold`,
`consolidate_object_ids`), building the final xarray `Dataset` once after the loop. Higher risk/effort
accepted; the 4 unit tests + golden snapshots + full-scale smoke are the safety net. Sequencing: do code +
unit tests, then run the heavy cProfile + full-scale smoke once the node is free.

## Files

- `marEx/track/objects.py` — add the local area+centroid helper (Fix A).
- `marEx/track/merge_split.py` — swap the per-merge `calculate_object_properties` call (A); batch the
  `object_props` concat per chunk (B′) or replace the container (B).
- `marEx/track/overlap.py` — swap the `consolidate_object_ids` full-slice call (A); only touched further for
  full Fix B.
- `tests/test_track_helpers.py` — unit test the new local-props helper vs. full-slice `regionprops` (A/B
  equivalence on synthetic blobs, incl. antimeridian).
- `tests/data/track_golden_*.nc` + `tests/test_gridded_tracking.py` bounds — regenerate iff the FP-order
  centroid change shifts the snapshot (same procedure as `5d08274`).

## Verification

1. Unit: new local-props helper == full-slice `regionprops` area (exact) + centroid (within FP tol) on
   synthetic blobs, including seam-straddling.
2. A/B: `cProfile` `split_and_merge_objects` on the 300-slice subset — confirm `object_properties_chunk`
   share collapses (30 % → few %) and (with B′/B) the per-chunk time stops rising with N.
3. Golden + suite: `pytest tests/test_track_golden.py tests/test_gridded_tracking.py tests/test_track_helpers.py`;
   regenerate golden only if the centroid FP shift moves it; confirm event/merge counts unchanged.
4. Full-scale 9282-day smoke: confirm the loop time drops and per-chunk time is ~flat (no O(N²) ramp).

## Notes / risk

- Correctness-critical merge/split core; keep changes minimal and gated by golden + full-scale smoke.
- Non-bit-exact centroid tie shifts are expected/accepted (golden regen), as with the partition fix.
- Pre-commit (black/isort/flake8, line-length 132) must pass; do not commit unless asked.
