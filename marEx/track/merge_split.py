"""
MarEx Track: Splitting/merging algorithmic core.

This module holds the heaviest, most numerically sensitive part of the tracker:
the routines that detect and resolve object splits and merges over time and that
relabel local object IDs into the final global event-ID space.

It contains three functions extracted verbatim from the original ``tracker``
methods of the same names:

* :func:`split_and_merge_objects` -- structured-grid (and default) merge/split
  resolution producing the partitioned-merge ledger.
* :func:`split_and_merge_objects_parallel` -- the unstructured-grid parallel
  implementation, with its internal per-chunk closures kept intact.
* :func:`cluster_rename_objects_and_props` -- connected-component clustering of
  overlapping IDs into events, with area/centroid recomputation.

The pervasive ``self.*`` grid/config state the original methods read is threaded
in as explicit arguments. Behaviour and numerics are identical to the original
``tracker`` methods; the tracker now delegates to these functions via thin
method wrappers.
"""

import hashlib
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import dask
import dask.array as da
import numpy as np
import xarray as xr
import zarr
from dask import persist
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..exceptions import TrackingError
from ..logging_config import get_logger
from . import objects as _objects
from . import overlap as _overlap
from .partitioning import (
    partition_centroid_unstructured,
    partition_nn_grid,
    partition_nn_unstructured,
    partition_nn_unstructured_optimised,
    wrapped_euclidian_distance_mask_parallel,
)
from .region_writer import ObjectIDRegionWriter

logger = get_logger(__name__)

# Record widths of the unstructured merge kernel. They size the per-merge arrays only (a few KB
# per chunk); the earlier update-slot bound MAX_MERGES * (MAX_PARENTS - 1) <= 255 no longer exists
# because the kernel writes int32 labels directly. Both are guarded: a child with more accepted
# parents than MAX_PARENTS, or a timestep with more merges than MAX_MERGES, raises TrackingError.
# MAX_PARENTS must stay <= 255: ``parent_masks_uint`` is uint8 with 255 as its "no parent" value.
MAX_MERGES = 64
MAX_PARENTS = 64


def _consolidate_slice(
    ids_prev: NDArray[np.int32],
    ids_cur: NDArray[np.int32],
    area: NDArray[np.float32],
    overlap_threshold: float,
) -> List[int]:
    """Array-only counterpart of ``overlap.consolidate_object_ids`` for the unstructured kernel.

    A parent at ``ids_prev`` that overlaps two or more objects at ``ids_cur`` (each pair at
    ``overlap / min(area) >= overlap_threshold``) hands all of them the smallest child ID, so a
    split keeps one ID per lineage. Parents are visited in ascending ID and the renames compose
    in that order, with the serial path's skip rules: a group whose first child was already
    renamed is skipped, and already-renamed children are not renamed again. ``ids_cur`` is
    relabelled IN PLACE.

    Returns the surviving IDs whose cells grew (empty when nothing changed).
    """
    both = (ids_prev > 0) & (ids_cur > 0)
    if not both.any():
        return []
    prev_fg = ids_prev[both].astype(np.int64)
    cur_fg = ids_cur[both].astype(np.int64)
    max_id = np.int64(max(int(ids_prev.max()), int(ids_cur.max()) + 1))
    pairs, pair_inv = np.unique(prev_fg * max_id + cur_fg, return_inverse=True)
    overlap = np.bincount(pair_inv.reshape(-1), weights=area[both]).astype(np.float32)
    pair_parent = pairs // max_id
    pair_child = pairs % max_id

    def _areas(ids: NDArray[np.int32], wanted: NDArray[np.int64]) -> NDArray[np.float32]:
        fg = ids > 0
        labels, inv = np.unique(ids[fg], return_inverse=True)
        sums = np.bincount(inv.reshape(-1), weights=area[fg]).astype(np.float32)
        return sums[np.searchsorted(labels, wanted)]

    min_area = np.minimum(_areas(ids_prev, pair_parent), _areas(ids_cur, pair_child))
    accepted = overlap / min_area >= overlap_threshold
    pair_parent = pair_parent[accepted]
    pair_child = pair_child[accepted]
    parents, counts = np.unique(pair_parent, return_counts=True)
    splitting = parents[counts > 1]
    if not splitting.size:
        return []

    target: Dict[int, int] = {}  # renamed child -> surviving ID
    grown: List[int] = []
    for parent in splitting:
        children = [int(c) for c in pair_child[pair_parent == parent]]
        first = children[0]
        if first in target:
            continue
        others = [c for c in children[1:] if c not in target]
        if not others:
            continue
        for renamed, survivor in list(target.items()):
            if survivor in others:
                target[renamed] = first
        for c in others:
            target[c] = first
        if first not in grown:
            grown.append(first)
    grown = [g for g in grown if g not in target]
    renamed_ids = np.fromiter(target.keys(), dtype=np.int64, count=len(target))
    survivors = np.fromiter(target.values(), dtype=np.int64, count=len(target))
    order = np.argsort(renamed_ids)
    renamed_ids, survivors = renamed_ids[order], survivors[order]
    cells = np.where(np.isin(ids_cur, renamed_ids))[0]
    ids_cur[cells] = survivors[np.searchsorted(renamed_ids, ids_cur[cells])].astype(ids_cur.dtype)
    return grown


def _anchor_field(obj, label, materialiser):
    """Anchor a whole field read by two or more consumers.

    ``materialiser is None`` keeps the previous behaviour exactly (a plain ``persist``),
    which is what every caller that has not yet been threaded still gets.
    ``preserve_chunks=True`` because the merge loop's end-of-chunk consolidation is
    boundary-dependent, so staging must not move a chunk boundary.
    """
    if materialiser is None:
        return obj.persist()
    return materialiser.stage(obj, label, preserve_chunks=True)


def cluster_rename_objects_and_props(
    object_id_field_unique: xr.DataArray,
    object_props: xr.Dataset,
    overlap_objects_list: NDArray[np.int32],
    merge_events: xr.Dataset,
    unstructured_grid: bool,
    timedim: str,
    timecoord: str,
    timechunks: int,
    ydim: Optional[str],
    xdim: str,
    cell_area: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    regional_mode: bool,
    *,
    materialiser=None,
) -> xr.Dataset:
    """
    Cluster the object pairs and relabel to determine final event IDs.

    Parameters
    ----------
    object_id_field_unique : xarray.DataArray
        Field of unique object IDs. IDs must not be repeated across time.
    object_props : xarray.Dataset
        Properties of each object that also need to be relabeled.
    overlap_objects_list : (N x 2) numpy.ndarray
        Array of object ID pairs that indicate which objects are in the same event.
        The object in the first column precedes the second column in time.
    merge_events : xarray.Dataset
        Information about merge events

    Returns
    -------
    split_merged_events_ds : xarray.Dataset
        Dataset with relabeled events and their properties. ID = 0 indicates no object.
    """
    # Cluster the overlap_pairs into groups of IDs that are actually the same object
    # Get IDs from overlap pairs
    # Step 1: Find all IDs that actually exist in the data
    # (max_ID is taken from the sorted unique IDs computed below rather than from a
    # separate .max() pass over the whole field -- review finding 5.8.)

    # Get unique IDs from overlap list
    if len(overlap_objects_list) > 0:
        overlap_ids = np.unique(overlap_objects_list[:, :2].flatten())
        overlap_ids = overlap_ids[overlap_ids > 0]  # Remove 0 (background)
    else:
        overlap_ids = np.array([], dtype=np.int32)  # pragma: no cover

    # Get unique IDs from object_id_field.
    # Use dask.array.unique (distributed tree-reduction) rather than
    # np.unique(object_id_field_unique.compute().values): the latter materialises the entire
    # global ID field (e.g. ~36 GiB at 9282x720x1440) onto a single worker -> MemoryError at scale.
    # da.unique reduces per-block and returns only the small array of unique IDs (bit-identical).
    field_ids = da.unique(object_id_field_unique.data).compute()
    field_ids = field_ids[field_ids > 0]  # Remove 0 (background)

    # da.unique returns them sorted, so the largest ID present in the field is the last
    # entry -- the same value the separate full-field .max() pass produced (0 for an
    # empty field), for none of the cost.
    max_ID = int(field_ids[-1]) if field_ids.size > 0 else 0

    # Combine and get all valid IDs
    all_valid_ids = np.unique(np.concatenate([overlap_ids, field_ids]))

    logger.info(f"Found {len(all_valid_ids)} valid object IDs (out of max ID {max_ID})")

    # Step 2: Create dense mapping: original_ID -> dense_index
    # This ensures continuous indices for connected_components
    original_to_dense = {int(original_id): dense_idx for dense_idx, original_id in enumerate(all_valid_ids)}
    dense_to_original = {dense_idx: int(original_id) for original_id, dense_idx in original_to_dense.items()}

    n_valid = len(all_valid_ids)

    # Step 3: Convert overlap pairs to dense indices
    if len(overlap_objects_list) > 0:
        # Map to dense indices with one binary search over the sorted ID array instead of
        # a per-pair Python dict lookup across a multi-million-row list (finding 5.9).
        # all_valid_ids is the sorted union that includes every positive entry of this
        # array, so the only rows the dict version dropped were those holding a
        # non-positive (background) ID -- which is exactly what `keep` drops here.
        pairs = overlap_objects_list[:, :2].astype(np.int64)
        keep = (pairs > 0).all(axis=1)
        overlap_pairs_dense = np.searchsorted(all_valid_ids, pairs[keep])

        # Create sparse graph with dense indices
        row_indices, col_indices = overlap_pairs_dense.T
        data = np.ones(len(overlap_pairs_dense), dtype=np.bool_)
        graph = csr_matrix((data, (row_indices, col_indices)), shape=(n_valid, n_valid), dtype=np.bool_)
    else:
        graph = csr_matrix((n_valid, n_valid), dtype=np.bool_)  # pragma: no cover

    # Step 4: Solve for connected components (on dense graph)
    num_components, component_IDs_dense = connected_components(csgraph=graph, directed=False, return_labels=True)

    logger.info(f"Identified {num_components} connected components (events)")

    # Step 5: Create lookup from original IDs to event IDs
    # Event IDs will be continuous: 1, 2, 3, ... num_components
    original_to_event = {}
    for dense_idx, event_id in enumerate(component_IDs_dense):
        original_id = dense_to_original[dense_idx]
        original_to_event[original_id] = event_id + 1  # +1 so events start at 1, not 0

    # Step 6: Create full lookup array for fast remapping
    ID_to_cluster_index_array = np.full(max_ID + 1, 0, dtype=np.int32)  # 0 = background
    for original_id, event_id in original_to_event.items():
        ID_to_cluster_index_array[original_id] = np.int32(event_id)

    # Convert to DataArray for apply_ufunc
    #  N.B.: **Need to pass da into apply_ufunc, otherwise it doesn't manage the memory correctly
    #          with large shared-mem numpy arrays**
    ID_to_cluster_index_da = xr.DataArray(
        ID_to_cluster_index_array,
        dims="ID",
        coords={"ID": np.arange(max_ID + 1, dtype=np.int32)},
    )

    def map_IDs_to_indices(block: NDArray[np.int32], ID_to_cluster_index_array: NDArray[np.int32]) -> NDArray[np.int32]:
        """Map original IDs to cluster indices."""
        mask = block > 0
        new_block = np.zeros_like(block, dtype=np.int32)
        new_block[mask] = ID_to_cluster_index_array[block[mask]]
        return new_block

    # Apply the mapping
    input_dims = [xdim] if unstructured_grid else [ydim, xdim]
    split_merged_relabeled_object_id_field = xr.apply_ufunc(
        map_IDs_to_indices,
        object_id_field_unique,
        ID_to_cluster_index_da,
        input_core_dims=[input_dims, ["ID"]],
        output_core_dims=[input_dims],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32],
    )
    if materialiser is None:
        split_merged_relabeled_object_id_field = split_merged_relabeled_object_id_field.persist()
    else:
        split_merged_relabeled_object_id_field = materialiser.stage(
            split_merged_relabeled_object_id_field, "relabeled_id_field", preserve_chunks=True
        )

    # Relabel the object_props to match the new IDs (and add time dimension)

    max_new_ID = num_components + 1  # New IDs range from 0 to max_new_ID
    new_ids = np.arange(1, max_new_ID + 1, dtype=np.int32)

    # Create new object_props dataset - use dimension coordinate for time data
    time_coord_data = object_id_field_unique.coords[timedim].data
    object_props_extended = xr.Dataset(coords={"ID": new_ids, timecoord: (timedim, time_coord_data)})

    # Create mapping from new IDs to the original IDs _at the corresponding time_
    valid_new_ids = split_merged_relabeled_object_id_field > 0
    # Fill masked points with 0 (not NaN) to keep the ID field integer. These positions are
    # never read: process_timestep indexes original_ids_field only through valid_mask, which
    # is derived from new_ids_field (> 0) at exactly the same points.
    original_ids_field = object_id_field_unique.where(valid_new_ids, 0)
    new_ids_field = split_merged_relabeled_object_id_field.where(valid_new_ids)

    if not unstructured_grid:
        original_ids_field = original_ids_field.stack(z=(ydim, xdim), create_index=False)
        new_ids_field = new_ids_field.stack(z=(ydim, xdim), create_index=False)

    new_id_to_idx = {id_val: idx for idx, id_val in enumerate(new_ids)}

    def process_timestep(orig_ids: NDArray[np.int32], new_ids_t: NDArray[np.int32]) -> NDArray[np.int32]:
        """Process a single timestep to create ID mapping."""
        result = np.zeros(len(new_id_to_idx), dtype=np.int32)

        valid_mask = new_ids_t > 0

        # Get valid points for this timestep
        if not valid_mask.any():
            return result

        orig_valid = orig_ids[valid_mask]
        new_valid = new_ids_t[valid_mask]

        if len(orig_valid) == 0:
            return result

        unique_pairs = np.unique(np.column_stack((orig_valid, new_valid)), axis=0)

        # Create mapping
        for orig_id, new_id in unique_pairs:
            if new_id in new_id_to_idx:
                result[new_id_to_idx[new_id]] = orig_id

        return result

    # Process in parallel
    input_dim = [xdim] if unstructured_grid else ["z"]
    global_id_mapping = (
        xr.apply_ufunc(
            process_timestep,
            original_ids_field,
            new_ids_field,
            input_core_dims=[input_dim, input_dim],
            output_core_dims=[["ID"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int32],
            dask_gufunc_kwargs={"output_sizes": {"ID": len(new_ids)}},
        )
        .assign_coords(ID=new_ids)
        .compute()
    )

    # Store original ID mapping
    object_props_extended["global_ID"] = global_id_mapping
    # Post-condition: Now, e.g. global_id_mapping.sel(ID=10)
    #    --> Given the new ID (10), returns corresponding original_id at every time

    # Transfer all properties from original object_props
    dummy = object_props.isel(ID=0) * np.nan  # Add vale of ID = 0 to this coordinate ID
    object_props = xr.concat([dummy.assign_coords(ID=0), object_props], dim="ID")

    for var_name in object_props.data_vars:
        # Filter global_id_mapping to only include IDs that exist in object_props.
        # Pass an ndarray, not a set: xarray/np.isin against a Python set returns all-False,
        # which silently NaN-ed every transferred property (masked only because area and
        # centroid are recomputed below).
        existing_ids = np.asarray(object_props.ID.values)
        valid_mapping_mask = global_id_mapping.isin(existing_ids)

        # Only select existing IDs
        valid_global_mapping = global_id_mapping.where(valid_mapping_mask, drop=True)

        if len(valid_global_mapping.ID) == 0:
            # No valid IDs - create empty result
            temp = object_props[var_name].isel(ID=slice(0, 0))
        else:
            temp = (
                object_props[var_name]
                .sel(ID=valid_global_mapping.rename({"ID": "new_id"}))
                .drop_vars("ID")
                .rename({"new_id": "ID"})
            )

        if var_name == "ID":
            temp = temp.astype(np.int32)
        else:
            temp = temp.astype(np.float32)

        object_props_extended[var_name] = temp

    # Map the merge_events using the old IDs to be from dimensions (merge_ID, parent_idx)
    #     --> new merge_ledger with dimensions (time, ID, sibling_ID)
    # i.e. for each merge_ID --> merge_parent_IDs   gives the old IDs  --> map to new ID using ID_to_cluster_index_da
    #                   --> merge_time

    old_parent_IDs = xr.where(merge_events.parent_IDs > 0, merge_events.parent_IDs, 0)
    # Guard against ledger parent IDs beyond the final field's max_ID (rare stale entries):
    # map out-of-range IDs to background (0) so .sel does not raise KeyError (§5.6).
    old_parent_IDs = xr.where(old_parent_IDs <= max_ID, old_parent_IDs, 0)
    new_IDs_parents = ID_to_cluster_index_da.sel(ID=old_parent_IDs)
    # Real parents always map to an event ID >= 1; only padded / out-of-range slots yield 0.
    # Map those to -1 so the sentinel matches the merge_ledger fill value (§5.20).
    new_IDs_parents = xr.where(new_IDs_parents > 0, new_IDs_parents, -1)

    # Replace the coordinate merge_ID in new_IDs_parents with merge_time.
    #    merge_events.merge_time gives merge_time for each merge_ID
    new_IDs_parents_t = (
        new_IDs_parents.assign_coords({"merge_time": merge_events.merge_time})
        .drop_vars("ID")
        .swap_dims({"merge_ID": "merge_time"})
        .persist()
    )

    # Map new_IDs_parents_t into a new data array with dimensions time, ID, and sibling_ID
    merge_ledger = (
        xr.full_like(global_id_mapping, fill_value=-1)
        .chunk({timedim: timechunks})
        .expand_dims({"sibling_ID": new_IDs_parents_t.parent_idx.shape[0]})
        .copy()
    )

    # Wrapper for processing/mapping mergers in parallel
    def process_time_group(
        time_block: xr.DataArray,
        IDs_data: NDArray[np.int32],
        IDs_coords: Dict[str, Any],
    ) -> xr.DataArray:
        """Process all mergers for a single block of timesteps.

        Writes are POSITIONAL, never ``.loc`` on the ``ID`` label. ``xr.map_blocks`` hands every
        block the same single-chunk ``ID`` coordinate, so all blocks on a worker share one pandas
        index engine, and its first-use population is not thread-safe: a concurrent
        ``get_indexer`` can read ``is_unique == False`` off a unique index and raise
        ``InvalidIndexError`` (D-079: one block of 73 failed on a full-year ICON run, not
        reproducible from the data). Positions come from ``np.searchsorted`` on the block's own
        ``ID`` values, which are sorted and unique by construction (asserted by the caller).
        """
        result = xr.full_like(time_block, -1)
        values = result.data  # numpy inside map_blocks; written in place
        dims = result.dims
        t_axis, id_axis, sib_axis = dims.index(timedim), dims.index("ID"), dims.index("sibling_ID")
        id_values = np.asarray(time_block["ID"].values)
        n_sib = time_block.sizes["sibling_ID"]

        if timecoord in time_block.coords:
            block_times = np.asarray(time_block.coords[timecoord].values)
        else:
            block_times = np.asarray(time_block[timedim].values)

        merge_times = np.asarray(IDs_coords["merge_time"])
        for t_pos, time_val in enumerate(block_times):
            time_mask = merge_times == time_val
            if not np.any(time_mask):
                continue
            # IDs_data[time_mask] keeps the parent_idx axis: (n_mergers_at_time, n_parent_idx)
            for merger_IDs in IDs_data[time_mask]:
                valid = merger_IDs[merger_IDs > 0]
                if not valid.size:
                    continue
                id_pos = np.searchsorted(id_values, valid)
                if np.any(id_pos >= id_values.size) or np.any(id_values[np.minimum(id_pos, id_values.size - 1)] != valid):
                    raise TrackingError(
                        "Merge ledger parent not in the event ID axis",
                        details=f"time {time_val}: event IDs {valid.tolist()} vs ID axis of {id_values.size}",
                    )
                # Same values the label-based write produced: at (time, ID=valid[i], sibling j) the
                # entry is valid[i] for every j.
                index = [slice(None)] * 3
                index[t_axis] = t_pos
                index[id_axis] = id_pos
                index[sib_axis] = slice(0, n_sib)
                if id_axis < sib_axis:
                    block = np.broadcast_to(valid[:, None], (valid.size, n_sib))
                else:
                    block = np.broadcast_to(valid[None, :], (n_sib, valid.size))
                values[tuple(index)] = block

        return result

    for name in (timedim, "ID"):
        index = merge_ledger.indexes[name]
        if not (index.is_unique and index.is_monotonic_increasing):
            raise TrackingError(
                f"The merge ledger's {name!r} axis must be unique and increasing",
                details=f"{name}: {len(index)} labels, unique={index.is_unique}, increasing={index.is_monotonic_increasing}",
            )

    # Map blocks in parallel
    merge_ledger = xr.map_blocks(
        process_time_group,
        merge_ledger,
        args=(new_IDs_parents_t.values, new_IDs_parents_t.coords),
        template=merge_ledger,
    )

    # Format merge ledger. This is a returned data_var of shape (time, ID, sibling_ID), so
    # it grows quadratically with the series length (1.135 GB at nt=3804, 77 % of everything
    # streaming still pinned). Anchoring it puts those bytes on disk under streaming; in
    # persist mode `stage` is `dask.persist`, byte for byte what this line did before.
    # Staging costs approximately nothing in disk terms. The ledger is overwhelmingly -1
    # fill, and blosc crushes a constant chunk to a few hundred bytes: at the nt=3804 shape,
    # 1.135 GB dense writes as ~5.5 MB of zarr (0.49 %) with that run's real merge count of
    # 18712. The same is true of any events_ds a caller saves. (That figure is modelled, not
    # a du of the staged store -- streaming cleans its staging dir on normal exit, so the
    # A/B could not measure it directly. Sub-1 % is the robust part: a 3000-entry sprinkle
    # gave 8.96 MB, so the exact value moves with merge density and neither is worth quoting
    # to three digits.)
    #
    # So this removes the PIN, not the quadratic growth -- and the growth is a RAM-only
    # concern, confined to persist mode. Do not reach for a sparse representation on
    # disk-size grounds; that was measured and the premise does not hold.
    merge_ledger = merge_ledger.rename("merge_ledger").transpose(timedim, "ID", "sibling_ID")
    merge_ledger = _anchor_field(merge_ledger, "merge_ledger", materialiser)

    # Add start and end time indices for each ID
    valid_presence = object_props_extended["global_ID"] > 0  # i.e. where there is valid data

    object_props_extended["presence"] = valid_presence
    object_props_extended["time_start"] = valid_presence[timecoord][valid_presence.argmax(dim=timedim).astype(np.int32)]
    object_props_extended["time_end"] = valid_presence[timecoord][
        ((valid_presence.sizes[timedim] - 1) - (valid_presence[::-1]).argmax(dim=timedim)).astype(np.int32)
    ]

    # Recompute area & centroid (now that the IDs have been consolidated & merged & made continuous)
    if "area" in object_props_extended.data_vars or "centroid" in object_props_extended.data_vars:
        logger.info("Recalculating area and centroid properties for potentially disjoint events...")

        def calculate_area_centroid_for_slice(
            slice_data: NDArray[np.int32],
            cell_areas_slice: NDArray[np.float32],
            present_mask: NDArray[np.bool_],
            all_event_ids: NDArray[np.int32],
            lat_vals: NDArray[np.float32],
            lon_vals: NDArray[np.float32],
            is_unstructured: bool,
            regional_mode: bool,
        ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
            """
            Calculate area and area-weighted centroid for IDs present at this timestep.
            Returns three arrays with full ID dimension (NaN for absent IDs).

            Parameters
            ----------
            slice_data : array
                Spatial field of event IDs for this timestep
            cell_areas_slice : array
                Spatial field of cell areas
            present_mask : array
                1D boolean array indicating which IDs are present (length = n_IDs)
            all_event_ids : array
                All event IDs (length = n_IDs)
            """
            n_ids = len(all_event_ids)

            # Initialise output arrays with NaN
            areas = np.full(n_ids, np.nan, dtype=np.float32)
            centroid_lats = np.full(n_ids, np.nan, dtype=np.float32)
            centroid_lons = np.full(n_ids, np.nan, dtype=np.float32)

            # Get indices of IDs that are present at this timestep
            present_indices = np.where(present_mask)[0]

            if len(present_indices) == 0:
                return areas, centroid_lats, centroid_lons

            # Group the slice's pixels by ID in one pass, instead of rebuilding a
            # full-slice `slice_data == event_id` mask for every present ID -- that was
            # O(n_present x ny x nx) per timestep, ~1e8-1e9 operations at scale
            # (review finding 5.10).
            #
            # The segments are produced by a *stable* sort of the flat pixel positions, so
            # each ID's pixels arrive in exactly the row-major order np.nonzero gave, and
            # the reductions below run over identical arrays in an identical order. This
            # rewrite is therefore bit-identical, not merely equivalent in real arithmetic:
            # it does not need the Phase-2 float tolerance.
            flat_ids = slice_data.ravel()
            flat_areas = cell_areas_slice.ravel()
            max_id_value = int(all_event_ids.max()) if n_ids > 0 else 0
            id_lookup = np.full(max_id_value + 1, -1, dtype=np.int64)
            in_range = (all_event_ids >= 0) & (all_event_ids <= max_id_value)
            id_lookup[all_event_ids[in_range]] = np.flatnonzero(in_range)

            codes = np.where(
                (flat_ids > 0) & (flat_ids <= max_id_value),
                id_lookup[np.clip(flat_ids, 0, max_id_value)],
                -1,
            )
            pixel_positions = np.flatnonzero(codes >= 0)  # ascending, i.e. row-major
            order = np.argsort(codes[pixel_positions], kind="stable")
            pixel_positions = pixel_positions[order]
            segment_bounds = np.concatenate(([0], np.cumsum(np.bincount(codes[codes >= 0], minlength=n_ids))))

            if is_unstructured:
                # Unstructured grid: area-weighted centroid using spherical geometry

                # Convert to radians for Cartesian calculation
                lat_rad = np.radians(lat_vals)
                lon_rad = np.radians(lon_vals)

                # Process each present ID
                for id_idx in present_indices:
                    cells = pixel_positions[segment_bounds[id_idx] : segment_bounds[id_idx + 1]]

                    if cells.size == 0:
                        continue  # pragma: no cover

                    # Calculate physical area
                    areas_masked = flat_areas[cells]
                    total_area = np.sum(areas_masked)
                    areas[id_idx] = total_area

                    # Calculate area-weighted centroid using spherical geometry
                    cos_lat = np.cos(lat_rad[cells])
                    x = cos_lat * np.cos(lon_rad[cells])
                    y = cos_lat * np.sin(lon_rad[cells])
                    z = np.sin(lat_rad[cells])

                    # Weighted average in Cartesian coordinates
                    weighted_x = np.sum(areas_masked * x)
                    weighted_y = np.sum(areas_masked * y)
                    weighted_z = np.sum(areas_masked * z)

                    # Normalise
                    norm = np.sqrt(weighted_x**2 + weighted_y**2 + weighted_z**2)
                    if norm > 0:
                        weighted_x /= norm
                        weighted_y /= norm
                        weighted_z /= norm

                    # Convert back to lat/lon
                    centroid_lat = np.degrees(np.arcsin(np.clip(weighted_z, -1, 1)))
                    centroid_lon = np.degrees(np.arctan2(weighted_y, weighted_x))

                    # Fix longitude range to [-180, 180]
                    if centroid_lon > 180:
                        centroid_lon -= 360  # pragma: no cover
                    elif centroid_lon < -180:
                        centroid_lon += 360  # pragma: no cover

                    centroid_lats[id_idx] = centroid_lat
                    centroid_lons[id_idx] = centroid_lon
            else:
                # Structured grid: area-weighted centroid with periodic boundary handling
                ny, nx = slice_data.shape

                # Process each present ID
                for id_idx in present_indices:
                    pixels = pixel_positions[segment_bounds[id_idx] : segment_bounds[id_idx + 1]]

                    if pixels.size == 0:
                        continue  # pragma: no cover

                    # Get indices where object exists (row-major, matching np.nonzero)
                    y_indices, x_indices = pixels // nx, pixels % nx

                    # Get cell areas for these indices
                    pixel_areas = flat_areas[pixels]
                    total_area = np.sum(pixel_areas)
                    areas[id_idx] = total_area

                    # Calculate area-weighted y centroid (latitude)
                    centroid_y_pix = np.sum(y_indices * pixel_areas) / total_area

                    # Calculate area-weighted x centroid (longitude) - handle wrapping if needed
                    if not regional_mode:
                        # Check if object is near both edges (wrapping around periodic boundary).
                        # Scale the margin so it never exceeds a quarter of the grid width (a fixed
                        # 100-column margin flags every object on grids with <=200 longitude points).
                        edge_margin = min(100, nx // 4)
                        near_left = np.any(x_indices < edge_margin)
                        near_right = np.any(x_indices >= nx - edge_margin)

                        if near_left and near_right:
                            # Object wraps around - adjust coordinates
                            x_adjusted = x_indices.copy().astype(np.float64)
                            right_side = x_indices > nx / 2
                            x_adjusted[right_side] -= nx

                            # Area-weighted mean with adjusted coordinates
                            centroid_x_pix = np.sum(x_adjusted * pixel_areas) / total_area

                            # Ensure centroid is positive
                            if centroid_x_pix < 0:
                                centroid_x_pix += nx
                        else:
                            # No wrapping - standard area-weighted calculation
                            centroid_x_pix = np.sum(x_indices * pixel_areas) / total_area
                    else:
                        # Regional mode - no wrapping, area-weighted
                        centroid_x_pix = np.sum(x_indices * pixel_areas) / total_area

                    # Convert pixel indices to coordinate values
                    centroid_lat = np.interp(centroid_y_pix, np.arange(len(lat_vals)), lat_vals)
                    centroid_lon = np.interp(centroid_x_pix, np.arange(len(lon_vals)), lon_vals)

                    centroid_lats[id_idx] = centroid_lat
                    centroid_lons[id_idx] = centroid_lon

            return areas, centroid_lats, centroid_lons

        # Prepare spatial dimensions
        spatial_dims = [xdim] if unstructured_grid else [ydim, xdim]

        # Ensure cell_area has correct dimensions for apply_ufunc
        if not unstructured_grid and cell_area.ndim == 1:
            # Broadcast 1D latitude-dependent cell areas to 2D (lat, lon)
            template = split_merged_relabeled_object_id_field.isel({timedim: 0}, drop=True)
            cell_area_broadcast, _ = xr.broadcast(cell_area, template)
        else:
            cell_area_broadcast = cell_area

        # Apply calculation in parallel across time slices
        logger.info("Computing area and centroid properties in parallel...")
        areas_computed, centroid_lats_computed, centroid_lons_computed = xr.apply_ufunc(
            calculate_area_centroid_for_slice,
            split_merged_relabeled_object_id_field,
            cell_area_broadcast,  # Broadcasted to match spatial dimensions
            object_props_extended.presence,  # Boolean mask of which IDs are present at each time
            object_props_extended.ID,
            lat,  # Latitude coordinate values
            lon,  # Longitude coordinate values
            kwargs={"is_unstructured": unstructured_grid, "regional_mode": regional_mode},
            input_core_dims=[
                spatial_dims,
                spatial_dims,
                ["ID"],
                ["ID"],
                [ydim] if not unstructured_grid else [xdim],
                [xdim],
            ],
            output_core_dims=[["ID"], ["ID"], ["ID"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32, np.float32, np.float32],
        )

        results = persist(areas_computed, centroid_lats_computed, centroid_lons_computed)
        areas_computed, centroid_lats_computed, centroid_lons_computed = results

        # Update area with proper dimension ordering (time, ID)
        object_props_extended["area"] = areas_computed.transpose(timedim, "ID")

        # Combine lat/lon centroids along component dimension
        new_centroid = xr.concat([centroid_lats_computed, centroid_lons_computed], dim="component")
        new_centroid = new_centroid.assign_coords(component=[0, 1])

        # Update centroid with proper dimension ordering (component, time, ID)
        object_props_extended["centroid"] = new_centroid.transpose("component", timedim, "ID")

        logger.info("Property recalculation complete.")

    # Combine all components into final dataset
    split_merged_relabeled_events_ds = xr.merge(
        [
            split_merged_relabeled_object_id_field.rename("ID_field"),
            object_props_extended,
            merge_ledger,
        ]
    )

    # Remove the last ID -- it is all 0s (because we added an extra padding one above)
    return split_merged_relabeled_events_ds.isel(ID=slice(0, -1))


def split_and_merge_objects(
    object_id_field_unique: xr.DataArray,
    object_props: xr.Dataset,
    unstructured_grid: bool,
    timedim: str,
    ydim: Optional[str],
    xdim: str,
    cell_area: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    mean_cell_area: float,
    neighbours_int: xr.DataArray,
    nn_partitioning: bool,
    overlap_threshold: float,
    regional_mode: bool,
    *,
    materialiser=None,
    id_field_path=None,
) -> Tuple[xr.DataArray, xr.Dataset, NDArray[np.int32], xr.Dataset]:
    """
    Implement object splitting and merging logic.

    This identifies and processes cases where objects split or merge over time,
    creating new object IDs as needed.

    Parameters
    ----------
    object_id_field_unique : xarray.DataArray
        Field of unique object IDs. IDs are required to be monotonically increasing with time.
    object_props : xarray.Dataset
        Properties of each object

    Returns
    -------
    tuple
        (object_id_field, object_props, overlap_objects_list, merge_events)
    """
    # Replace the ID-indexed object_props Dataset with an O(1) store for the per-timestep loop.
    # The xarray .sel/.loc/.drop_sel/concat that previously mutated object_props per merge cost
    # O(current size), so they grew O(N^2) as objects accumulated. The store makes them O(1); we
    # convert back to a Dataset at the function boundary for cluster_rename_objects_and_props.
    object_props = _objects.ObjectPropsStore.from_dataset(object_props)

    # No up-front overlap pass here: the serial loop below computes overlaps per timestep
    # from the consolidated field, and the full-run list is recomputed after the loop.
    # The result of an up-front pass was persisted and then overwritten unread
    # (review finding 5.7).

    # Initialise merge tracking lists
    merge_times = []  # When the merge occurred
    merge_child_ids = []  # Resulting child ID
    merge_parent_ids = []  # List of parent IDs that merged
    merge_areas = []  # Areas of overlap
    next_new_id = object_props.max_id() + 1  # Start new IDs after highest existing ID

    Nx = object_id_field_unique[xdim].size
    # In streaming mode the accumulator lives on disk. The input field is NOT pinned:
    # each chunk reads its own disjoint time slice, and the upstream object_id_field is
    # already anchored (objects.py's _anchor helper, which calls materialiser.stage), so
    # slice reads are cheap.
    streaming = materialiser is not None and materialiser.is_streaming
    if streaming:
        writer = ObjectIDRegionWriter(object_id_field_unique, id_field_path, timedim)
    else:
        writer = None
        object_id_field_unique = object_id_field_unique.persist()
    updated_chunks = []

    # Process each time chunk with timestep-first approach
    chunk_boundaries = np.cumsum([0] + list(object_id_field_unique.chunks[0]))

    for chunk_idx in range(len(object_id_field_unique.chunks[0])):
        # Extract and load an entire chunk into memory
        chunk_start = chunk_boundaries[chunk_idx]
        chunk_end = chunk_boundaries[chunk_idx + 1]
        # Ensure we don't exceed array bounds
        chunk_end = min(chunk_end, object_id_field_unique.sizes[timedim])

        chunk_data = object_id_field_unique.isel({timedim: slice(chunk_start, chunk_end)}).compute()

        # Process each timestep within chunk sequentially
        for relative_t in range(chunk_data.sizes[timedim]):
            absolute_t = chunk_start + relative_t

            # Get data slices for current timestep
            data_t = chunk_data.isel({timedim: relative_t})

            # Get previous timesteps for consolidation and partitioning
            if relative_t > 1:  # Need both t-1 and t-2 for consolidation
                data_t_minus_2 = chunk_data.isel({timedim: relative_t - 2})
                data_t_minus_1 = chunk_data.isel({timedim: relative_t - 1})
            elif relative_t == 1:  # t-1 is in current chunk, t-2 might be in previous chunk
                data_t_minus_1 = chunk_data.isel({timedim: 0})  # relative_t - 1 = 0
                if updated_chunks:
                    _, _, last_chunk_data = updated_chunks[-1]
                    data_t_minus_2 = last_chunk_data[-1]  # Last timestep from previous chunk
                else:
                    data_t_minus_2 = xr.full_like(data_t, 0)
            else:  # relative_t == 0, get both from previous chunk if available
                if updated_chunks:
                    _, _, last_chunk_data = updated_chunks[-1]
                    if len(last_chunk_data) >= 2:
                        data_t_minus_2 = last_chunk_data[-2]
                        data_t_minus_1 = last_chunk_data[-1]
                    elif len(last_chunk_data) == 1:
                        data_t_minus_2 = xr.full_like(data_t, 0)
                        data_t_minus_1 = last_chunk_data[-1]
                    else:
                        data_t_minus_2 = xr.full_like(data_t, 0)
                        data_t_minus_1 = xr.full_like(data_t, 0)
                else:
                    data_t_minus_2 = xr.full_like(data_t, 0)
                    data_t_minus_1 = xr.full_like(data_t, 0)

            # ID Consolidation of objects at t-1
            if relative_t > 0:  # Only consolidate if we have meaningful t-1 and t-2
                data_t_minus_1, object_props = _overlap.consolidate_object_ids(
                    data_t_minus_2,
                    data_t_minus_1,
                    object_props,
                    absolute_t - 1,
                    unstructured_grid,
                    cell_area,
                    overlap_threshold,
                    lat,
                    lon,
                    timedim,
                    regional_mode,
                    ydim,
                    xdim,
                )

                # Update the chunk with consolidated data whenever t-1 is in current chunk
                chunk_data[{timedim: relative_t - 1}] = data_t_minus_1

            # Normal overlap detection and partitioning (now with consolidated IDs)

            # Calculate overlaps for this timestep
            #   Here, parents are at previous time=t-1 (LHS), children are at current time=t (RHS)
            timestep_overlaps = _overlap.check_overlap_slice(data_t_minus_1.values, data_t.values, unstructured_grid, cell_area)
            timestep_overlaps = _overlap.enforce_overlap_threshold(
                timestep_overlaps, object_props, unstructured_grid, overlap_threshold
            )

            # Iterative processing within timestep=t until convergence
            #  Only modifies data_t, which contains the children to be partitioned/relabelled
            timestep_converged = False
            iteration = 0

            while not timestep_converged and iteration < 10:  # Prevent infinite loops
                # Find merging objects for current timestep
                unique_children, children_counts = np.unique(timestep_overlaps[:, 1], return_counts=True)
                merging_children = unique_children[children_counts > 1]

                if len(merging_children) == 0:
                    timestep_converged = True
                    continue

                # Process all merging objects in this timestep
                #   Parents exist in this timestep, but
                for child_id in merging_children:

                    # Get mask of child object
                    child_mask_2d = (data_t == child_id).values

                    # Find all pairs involving this child
                    child_mask = timestep_overlaps[:, 1] == child_id
                    child_where = np.where(timestep_overlaps[:, 1] == child_id)[0].astype(np.int32)
                    merge_group = timestep_overlaps[child_mask]

                    # Get parent objects (LHS) that overlap with this child object
                    parent_ids = merge_group[:, 0]
                    num_parents = len(parent_ids)

                    # Create new IDs for the other half of the child object & record in the merge ledger
                    new_object_id = np.arange(next_new_id, next_new_id + (num_parents - 1), dtype=np.int32)
                    next_new_id += num_parents - 1

                    # Replace the 2nd+ child in the overlap objects list with the new child ID
                    timestep_overlaps[child_where[1:], 1] = new_object_id
                    child_ids = np.concatenate((np.array([child_id]), new_object_id))

                    # Record merge event - extract time value using dimension name
                    merge_times.append(data_t.coords[timedim].values)
                    merge_child_ids.append(child_ids)
                    merge_parent_ids.append(parent_ids)
                    merge_areas.append(timestep_overlaps[child_mask, 2])

                    # Relabel the Original Child Object ID Field to account for the New ID:
                    # Get parent centroids for partitioning
                    parent_centroids = object_props.centroids(parent_ids)

                    # Partition the child object based on parent associations
                    if nn_partitioning:
                        # Nearest-neighbor partitioning
                        # --> For every (Original) Child Cell in the ID Field, Find the closest (t-1) Parent _Cell_
                        if unstructured_grid:
                            # Prepare parent masks (one broadcast comparison, see below)
                            prev_values = data_t_minus_1.values
                            parent_masks = prev_values[None, :] == np.asarray(parent_ids).reshape(-1, 1)

                            # Calculate maximum search distance
                            max_area = np.max(object_props.areas(parent_ids)) / mean_cell_area
                            max_distance = int(np.sqrt(max_area) * 2.0)

                            # Use optimised unstructured partitioning
                            new_labels = partition_nn_unstructured(
                                child_mask_2d,
                                parent_masks,
                                child_ids,
                                parent_centroids,
                                neighbours_int.values,
                                lat.values,  # Need to pass these as NumPy arrays for JIT compatibility
                                lon.values,
                                max_distance=max(max_distance, 20) * 2,  # Set minimum threshold, in cells
                            )
                        else:
                            # Prepare parent masks for structured grid. One broadcast
                            # comparison against the raw values instead of a Python loop of
                            # per-parent xarray comparisons, each of which built and
                            # materialised its own full-slice DataArray (finding 5.15).
                            prev_values = data_t_minus_1.values
                            parent_masks = prev_values[None, :, :] == np.asarray(parent_ids).reshape(-1, 1, 1)

                            # Calculate maximum search distance
                            max_area = np.max(object_props.areas(parent_ids))
                            max_distance = int(np.sqrt(max_area) * 3.0)  # Use 3x the max blob radius

                            # Use optimised structured grid partitioning
                            new_labels = partition_nn_grid(
                                child_mask_2d,
                                parent_masks,
                                child_ids,
                                parent_centroids,
                                Nx,
                                max_distance=max(max_distance, 40),  # Set minimum threshold, in cells
                                wrap=not regional_mode,  # Turn longitude periodic wrapping off when in regional mode
                            )

                    else:
                        # Centroid-based partitioning
                        # --> For every (Original) Child Cell in the ID Field, Find the closest (t-1) Parent _Centroid_
                        if unstructured_grid:
                            new_labels = partition_centroid_unstructured(
                                child_mask_2d,
                                parent_centroids,
                                child_ids,
                                lat.values,
                                lon.values,
                            )
                        else:
                            # Calculate distances to each parent centroid
                            distances = wrapped_euclidian_distance_mask_parallel(
                                child_mask_2d, parent_centroids, Nx, not regional_mode
                            )

                            # Assign based on closest parent
                            new_labels = child_ids[np.argmin(distances, axis=1).astype(np.int32)]

                    # Update values in data_t and assign the updated slice back to the chunk
                    temp = np.zeros_like(data_t)
                    temp[child_mask_2d] = new_labels
                    data_t = data_t.where(~child_mask_2d, temp)
                    chunk_data[{timedim: relative_t}] = data_t

                    # Update the Properties of the N Children Objects.
                    # The new child IDs exist only within this partitioned child blob, so their
                    # area+centroid can be computed directly from the partition pixels in hand
                    # (child_mask_2d + new_labels) instead of a full-slice regionprops_table per
                    # merge. (Structured grids; the unstructured path keeps the full-slice call.)
                    if unstructured_grid:
                        new_child_props = _objects.calculate_object_properties(
                            data_t,
                            unstructured_grid,
                            lat,
                            lon,
                            cell_area,
                            timedim,
                            regional_mode,
                            ydim,
                            xdim,
                            properties=["area", "centroid"],
                        )
                    else:
                        child_y_idx, child_x_idx = np.nonzero(child_mask_2d)
                        new_child_props = _objects.calculate_partitioned_child_properties(
                            child_y_idx, child_x_idx, new_labels, Nx, regional_mode
                        )

                    # Update the object_props store: (but first, check if the original child still exists)
                    if child_id in new_child_props.ID:
                        # Update existing entry
                        cp = new_child_props.sel(ID=child_id)
                        object_props.set(child_id, cp["area"].values.item(), cp["centroid"].values[0], cp["centroid"].values[1])
                    else:
                        # Delete child_id: The object has split/morphed such that it doesn't get a partition of this child...
                        object_props.drop(child_id)  # N.B.: This means that the IDs are no longer continuous...
                        logger.info(f"Deleted child_id {child_id} because parents have split/morphed")

                    # Add the properties for the N-1 other new child ID
                    new_object_ids_still = new_child_props.ID.where(new_child_props.ID.isin(new_object_id), drop=True).ID
                    for new_id in new_object_ids_still.values:
                        cp = new_child_props.sel(ID=new_id)
                        object_props.set(int(new_id), cp["area"].values.item(), cp["centroid"].values[0], cp["centroid"].values[1])

                    missing_ids = set(new_object_id) - set(new_object_ids_still.values)
                    if len(missing_ids) > 0:
                        logger.warning(
                            f"Missing newly created child_ids {missing_ids} "
                            f"because parents have split/morphed in the meantime..."
                        )

                # After processing all merging objects in this iteration
                # Recalculate overlaps to check for newly viable merges
                timestep_overlaps = _overlap.check_overlap_slice(data_t_minus_1.values, data_t.values, unstructured_grid, cell_area)
                timestep_overlaps = _overlap.enforce_overlap_threshold(
                    timestep_overlaps, object_props, unstructured_grid, overlap_threshold
                )
                iteration += 1

            if iteration == 10:
                logger.warning(f"Resolving mergers at timestep {absolute_t} did not converge after 10 iterations")

        # End-of-chunk consolidation of the last timestep, against the timestep before it. A
        # one-timestep chunk takes that reference from the previous chunk: skipping it left the
        # slice unconsolidated, so the result depended on the time chunking (D-087).
        if chunk_data.sizes[timedim] >= 2 or updated_chunks:

            # Get last and second-to-last timesteps
            last_t_data = chunk_data.isel({timedim: -1})
            if chunk_data.sizes[timedim] >= 2:
                second_last_t_data = chunk_data.isel({timedim: -2})
            else:
                second_last_t_data = updated_chunks[-1][2][-1]

            # Consolidate last timestep using second-to-last as reference
            consolidated_last, object_props = _overlap.consolidate_object_ids(
                second_last_t_data,
                last_t_data,
                object_props,
                chunk_end - 1,
                unstructured_grid,
                cell_area,
                overlap_threshold,
                lat,
                lon,
                timedim,
                regional_mode,
                ydim,
                xdim,
            )

            # Update the last timestep in chunk
            chunk_data[{timedim: -1}] = consolidated_last

        # Store the processed chunk
        updated_chunks.append(
            (
                chunk_start,
                chunk_end,
                chunk_data[: (chunk_end - chunk_start)],
            )
        )

        if chunk_idx % 10 == 0:
            logger.info(f"Processing splitting and merging in chunk {chunk_idx} of {len(object_id_field_unique.chunks[0])}")

            # Periodically flush finished chunks to manage memory. The LAST chunk is
            # always retained: the next chunk reads its final two timesteps as t-1/t-2
            # (see the invariant in region_writer.py). Do not flush it.
            if len(updated_chunks) > 1:
                for start, end, processed_chunk_data in updated_chunks[:-1]:
                    if writer is not None:
                        writer.write(start, end, processed_chunk_data)
                    else:
                        object_id_field_unique[{timedim: slice(start, end)}] = processed_chunk_data
                updated_chunks = updated_chunks[-1:]  # Keep only the last chunk
                if writer is None:
                    object_id_field_unique = object_id_field_unique.persist()

    # Apply final chunk updates
    for start, end, processed_chunk_data in updated_chunks:
        if writer is not None:
            writer.write(start, end, processed_chunk_data)
        else:
            object_id_field_unique[{timedim: slice(start, end)}] = processed_chunk_data
    if writer is not None:
        object_id_field_unique = writer.finalise()
    else:
        object_id_field_unique = object_id_field_unique.persist()

    # Recompute final overlapping objects
    overlap_objects_list = _overlap.find_overlapping_objects(
        object_id_field_unique, timedim, unstructured_grid, ydim, xdim, cell_area
    )
    overlap_objects_list = _overlap.enforce_overlap_threshold(
        overlap_objects_list, object_props, unstructured_grid, overlap_threshold
    )
    logger.info("Finished final overlapping objects search")

    # Check for duplicate children (multiple parents per child)
    if len(overlap_objects_list) > 0:
        child_ids = overlap_objects_list[:, 1]  # RHS column (children)
        unique_children, child_counts = np.unique(child_ids, return_counts=True)

        # Find children with multiple parents
        duplicate_children = unique_children[child_counts > 1]

        # Enhanced validation with comprehensive spatial and temporal information
        if len(duplicate_children) > 0:
            logger.warning(f"There is {len(duplicate_children)} potentially problematic children:")

            # Log problematic child IDs (time info not available at this stage)
            logger.warning(f"Children IDs: {duplicate_children[:10].tolist()}")

            # Detailed analysis of each problematic child
            for child_id in duplicate_children[:5]:  # Limit to first 5 for readability
                # Find all parent-child relationships for this child
                child_relationships = overlap_objects_list[overlap_objects_list[:, 1] == child_id]
                parent_ids = child_relationships[:, 0]
                overlap_areas = child_relationships[:, 2]

                logger.warning(f"\n--- Details for child ID {child_id} ---")
                logger.warning(f"Number of parents: {len(parent_ids)}")
                logger.warning(f"Parent IDs: {parent_ids.tolist()}")
                logger.warning(f"Raw overlap areas: {overlap_areas.tolist()}")

                # Get child object properties if available
                try:
                    if child_id in object_props:
                        child_area = object_props.area(child_id)
                        child_centroid = object_props.centroid(child_id)

                        logger.warning(f"Child total area: {child_area}")
                        logger.warning(f"Child centroid: {child_centroid}")

                        # Calculate overlap fractions for each parent
                        overlap_fractions = []
                        parent_areas = []
                        for i, parent_id in enumerate(parent_ids):
                            if parent_id in object_props:
                                parent_area = object_props.area(parent_id)
                                parent_areas.append(parent_area)

                                # Calculate overlap fraction based on smaller object
                                min_area = min(child_area, parent_area)
                                overlap_fraction = float(overlap_areas[i]) / min_area
                                overlap_fractions.append(overlap_fraction)
                            else:
                                parent_areas.append("N/A")
                                overlap_fractions.append("N/A")

                        logger.warning(f"Parent areas: {parent_areas}")
                        logger.warning(f"Overlap fractions: {overlap_fractions}")

                        # Check for suspicious patterns
                        total_overlap_area = sum(overlap_areas)
                        logger.warning(f"Sum of overlap areas: {total_overlap_area}")
                        logger.warning(f"Sum/Child area ratio: {total_overlap_area/child_area:.3f}")

                        # Flag potential issues
                        valid_fractions = [f for f in overlap_fractions if isinstance(f, (int, float))]
                        if valid_fractions and max(valid_fractions) > 1.0:
                            logger.warning(f"WARNING: Overlap fraction > 1.0 detected (max: {max(valid_fractions):.3f})")
                        if total_overlap_area > child_area * 1.1:  # Allow 10% tolerance
                            logger.warning(
                                f"WARNING: Total overlap exceeds child area by {(total_overlap_area/child_area - 1)*100:.1f}%"
                            )

                    else:
                        logger.warning(f"Child ID {child_id} not found in object_props")

                except Exception as e:
                    logger.warning(f"Error analysing child ID {child_id}: {str(e)}")

                # Try to find timestep information by checking where this child appears
                try:
                    child_timesteps = []
                    for t_idx in range(object_id_field_unique.sizes[timedim]):
                        time_slice = object_id_field_unique.isel({timedim: t_idx})
                        if (time_slice == child_id).any():
                            time_coord = time_slice.coords[timedim].values
                            child_timesteps.append((t_idx, time_coord))

                    if child_timesteps:
                        logger.warning(f"Child appears at timesteps: {child_timesteps}")
                    else:
                        logger.warning("Child timestep information not found")

                except Exception as e:
                    logger.warning(f"Error finding timestep for child ID {child_id}: {str(e)}")

                logger.warning("--- End detailed analysis ---\n")

            # Log summary information as warnings instead of raising error
            logger.warning("=" * 80)
            logger.warning("Tracker Warning: Multiple parents for single child detected after splitting/merging")
            logger.warning(f"Details: {len(duplicate_children)} children have multiple parents")
            logger.warning("Note: This is likely due to consolidation of IDs after splitting/merging")
            logger.warning("      and still is the correct behaviour (as per the tracking overlap logic")
            logger.warning("      applied to disjoint objects that will be grouped together.)")
            logger.warning("=" * 80)
        else:
            logger.info(f"Validation passed: All {len(unique_children)} children have unique parents")
    else:
        logger.info("No overlaps found - validation skipped")

    # Process merge events into a dataset
    # Handle case where there are no merge events
    if merge_parent_ids and merge_child_ids:
        max_parents = max(len(ids) for ids in merge_parent_ids)
        max_children = max(len(ids) for ids in merge_child_ids)
    else:
        max_parents = 1  # Default minimum size
        max_children = 1

    # Convert lists to padded numpy arrays
    parent_ids_array = np.full((len(merge_parent_ids), max_parents), -1, dtype=np.int32)
    child_ids_array = np.full((len(merge_child_ids), max_children), -1, dtype=np.int32)
    # Unstructured merge areas are float32 m^2 that can exceed 2^31; match the parallel path.
    overlap_areas_array = np.full(
        (len(merge_areas), max_parents),
        -1,
        dtype=np.float32 if unstructured_grid else np.int32,
    )

    for i, parents in enumerate(merge_parent_ids):
        parent_ids_array[i, : len(parents)] = parents

    for i, children in enumerate(merge_child_ids):
        child_ids_array[i, : len(children)] = children

    for i, areas in enumerate(merge_areas):
        overlap_areas_array[i, : len(areas)] = areas

    # Create merge events dataset
    merge_events = xr.Dataset(
        {
            "parent_IDs": (("merge_ID", "parent_idx"), parent_ids_array),
            "child_IDs": (("merge_ID", "child_idx"), child_ids_array),
            "overlap_areas": (("merge_ID", "parent_idx"), overlap_areas_array),
            "merge_time": ("merge_ID", merge_times),
            "n_parents": (
                "merge_ID",
                np.array([len(p) for p in merge_parent_ids], dtype=np.int8),
            ),
            "n_children": (
                "merge_ID",
                np.array([len(c) for c in merge_child_ids], dtype=np.int8),
            ),
        },
        attrs={"fill_value": -1},
    )

    # Convert the O(1) store back to the ID-indexed Dataset expected by cluster_rename_objects_and_props.
    object_props = object_props.to_dataset()
    object_props = object_props.persist()

    return (
        object_id_field_unique,
        object_props,
        overlap_objects_list[:, :2],  # Only return first 2 columns (ID pairs)
        merge_events,
    )


def split_and_merge_objects_parallel(
    object_id_field_unique: xr.DataArray,
    object_props: xr.Dataset,
    unstructured_grid: bool,
    timedim: str,
    timecoord: str,
    timechunks: int,
    ydim: Optional[str],
    xdim: str,
    cell_area: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    mean_cell_area: float,
    neighbours_int: xr.DataArray,
    nn_partitioning: bool,
    overlap_threshold: float,
    regional_mode: bool,
    max_iteration: int,
    temp_field_path: str,
    *,
    materialiser=None,
) -> Tuple[xr.DataArray, xr.Dataset, NDArray[np.int32], xr.Dataset]:
    """
    Optimised parallel implementation of object splitting and merging.

    This version is specifically designed for unstructured grids with more efficient
    memory handling and better parallelism than the standard split_and_merge_objects
    method. It processes data in chunks, handles merging events, and efficiently
    updates object IDs.

    Parameters
    ----------
    object_id_field_unique : xarray.DataArray
        Field of unique object IDs
    object_props : xarray.Dataset
        Properties of each object

    Returns
    -------
    tuple
        (object_id_field, object_props, overlap_objects_list, merge_events)
    """
    # -------------------------------------------------------------------------------------
    # Structure (2026-09, Q8). Every time chunk is processed from its PRISTINE input labels
    # with the FINAL labels of the previous chunk's last timestep as its t-1 boundary. A chunk
    # is (re)run whenever that boundary, or the queue of objects handed across it, changed since
    # the chunk last ran; the loop ends when nothing changes. Inside a chunk the kernel is
    # sequential in time against the updated t-1, exactly as a single-chunk run, so the fixpoint
    # equals the single-chunk result whatever the chunking. The previous scheme ran every chunk
    # from the iteration's UPDATED field and deferred cross-chunk cascades: a boundary object was
    # partitioned against stale parents, the re-queue dedup dropped it, and its sibling pieces
    # were never repaired, so events depended on the time chunking (D-082).
    # The accumulator is a zarr copy of the pristine field (``temp_field_path``); a chunk task
    # writes its own region and returns only small results, so no whole-field array is ever
    # held for the update. Records of a re-run chunk replace that chunk's earlier records.
    # -------------------------------------------------------------------------------------
    # Record widths: module-level MAX_MERGES / MAX_PARENTS (read here, at call time).
    id_stride = MAX_MERGES * (MAX_PARENTS - 1)  # IDs a timestep can mint, so ranges never overlap

    if object_id_field_unique.dims != (timedim, xdim):
        object_id_field_unique = object_id_field_unique.transpose(timedim, xdim)
    time_axis = 0
    space_chunks = object_id_field_unique.chunks[1]
    if len(space_chunks) != 1:
        raise TrackingError(
            "The unstructured merge loop needs the spatial dimension in one chunk",
            details=f"{xdim} is split into {len(space_chunks)} chunks",
            suggestions=[f"Rechunk the input with {{'{xdim}': -1}}"],
        )
    chunk_sizes = tuple(int(c) for c in object_id_field_unique.chunks[time_axis])
    if len(chunk_sizes) > 2 and len(set(chunk_sizes[:-1])) != 1:
        raise TrackingError(
            "Time chunks must be uniform except for the last one",
            details=f"time chunk sizes {chunk_sizes}",
            suggestions=[f"Rechunk the input with {{'{timedim}': {max(chunk_sizes)}}}"],
        )
    chunk_starts = np.concatenate([[0], np.cumsum(chunk_sizes)]).astype(int)
    n_chunks = len(chunk_sizes)
    n_time = int(object_id_field_unique.sizes[timedim])

    # -- the kernel -----------------------------------------------------------------------
    def process_chunk(
        chunk_pristine: NDArray[np.int32],
        boundary: Optional[NDArray[np.int32]],
        next_first: Optional[NDArray[np.int32]],
        queue: List[List[int]],
        id_offsets: NDArray[np.int64],
        t0_abs: int,
        lat: NDArray[np.float32],
        lon: NDArray[np.float32],
        area: NDArray[np.float32],
        neighbours_int: NDArray[np.int32],
        zarr_path: str,
        region_dirty: bool,
    ) -> Tuple[List[Tuple[int, NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]], List[int], str, str, int, bool]:
        """Run the split-and-merge kernel over one time chunk from its pristine labels.

        Parameters
        ----------
        chunk_pristine : (n_time_chunk, ncells) int32
            The chunk's labels from the INPUT field (never a previous iteration's output).
        boundary : (ncells,) int32 or None
            The previous chunk's last timestep as it stands in the accumulator (None = first chunk).
        next_first : (ncells,) int32 or None
            The pristine first timestep of the next chunk, for the forward search at the last
            timestep of this chunk (None = last chunk).
        queue : list per timestep of child IDs to examine (initial multi-parent children plus the
            objects forwarded by the previous chunk into this chunk's first timestep).
        id_offsets : (n_time_chunk,) int64
            First new ID each timestep may mint.
        region_dirty : bool
            True when an earlier run of this chunk wrote its region, so the accumulator no longer
            holds the pristine labels and must be rewritten even if this run changes nothing.

        Returns
        -------
        records : list of (absolute t, child_ids, parent_ids, overlap_areas)
        forwarded : sorted list of object IDs at the next chunk's first timestep to examine
        last_hash, boundary_hash : digests of this chunk's final last slice and of the boundary
            it consumed (the driver decides re-runs from them)
        max_parents_seen : largest accepted-parent count in this run of the chunk
        wrote : whether this run wrote the chunk's region
        """
        data = np.array(chunk_pristine, dtype=np.int32, copy=True)
        while data.ndim > 2:
            data = data.squeeze(axis=-1)
        n_t, n_pts = data.shape
        if boundary is None:
            boundary_arr = np.zeros(n_pts, dtype=np.int32)
        else:
            boundary_arr = np.asarray(boundary, dtype=np.int32).reshape(n_pts)
        if next_first is None:
            next_first_arr = np.zeros(n_pts, dtype=np.int32)
        else:
            next_first_arr = np.asarray(next_first, dtype=np.int32).reshape(n_pts)
        boundary_hash = hashlib.blake2b(boundary_arr.tobytes(), digest_size=16).hexdigest()

        lat = np.asarray(lat, dtype=np.float32).reshape(n_pts)
        lon = np.asarray(lon, dtype=np.float32).reshape(n_pts)
        area = np.asarray(area, dtype=np.float32).reshape(n_pts)
        neighbours_int = np.asarray(neighbours_int, dtype=np.int32)
        if neighbours_int.ndim == 2 and neighbours_int.shape[1] != n_pts:
            neighbours_int = neighbours_int.T

        # Cartesian coordinates for the area-weighted parent centroids
        x = (np.cos(np.radians(lat)) * np.cos(np.radians(lon))).astype(np.float32)
        y = (np.cos(np.radians(lat)) * np.sin(np.radians(lon))).astype(np.float32)
        z = np.sin(np.radians(lat)).astype(np.float32)

        merging_objects_list = [list(queue[t]) for t in range(n_t)]
        records = []
        forwarded: List[int] = []
        changed = False
        max_parents_seen = 0

        for t in range(n_t):
            next_new_id = int(id_offsets[t])
            data_m1 = boundary_arr if t == 0 else data[t - 1]
            data_t = data[t]  # a view: in-place writes land in `data`
            data_p1 = data[t + 1] if t < n_t - 1 else next_first_arr
            merges_at_t = 0

            while merging_objects_list[t]:
                child_id = merging_objects_list[t].pop(0)

                child_mask = data_t == child_id
                # Ascending cell indices of the child. Objects are tiny next to the mesh, so
                # testing a candidate parent AT these indices costs O(child) where intersecting
                # two whole-field boolean masks costs O(ncells).
                child_cells = np.where(child_mask)[0].astype(np.int32)
                if child_cells.size == 0:
                    continue

                potential_parents = np.unique(data_m1[child_mask])
                child_area = area[child_mask].sum()
                parent_masks_uint = np.full(n_pts, 255, dtype=np.uint8)
                parent_centroids = np.full((MAX_PARENTS, 2), -1.0e10, dtype=np.float32)
                parent_ids = np.full(MAX_PARENTS, -1, dtype=np.int32)
                parent_areas = np.zeros(MAX_PARENTS, dtype=np.float32)
                overlap_areas = np.zeros(MAX_PARENTS, dtype=np.float32)
                n_parents = 0

                for parent_id in potential_parents[potential_parents > 0]:
                    parent_mask = data_m1 == parent_id
                    overlap_cells = child_cells[parent_mask[child_cells]]
                    if not overlap_cells.size:
                        continue
                    area_0 = area[parent_mask].sum()
                    min_area = np.minimum(area_0, child_area)
                    overlap_area = area[overlap_cells].sum()
                    if overlap_area / min_area < overlap_threshold:
                        continue

                    # Only an ACCEPTED parent can exhaust the fixed-width arrays; a candidate that
                    # fails the threshold above must never trip the guard (2026-08-20).
                    if n_parents >= MAX_PARENTS:  # pragma: no cover
                        raise TrackingError(
                            "Too many parent objects for tracking",
                            details=(
                                f"Child {child_id} at timestep {t0_abs + t} has more than "
                                f"{MAX_PARENTS} parents (limit: {MAX_PARENTS})"
                            ),
                            suggestions=[
                                "Raise MAX_PARENTS in split_and_merge_objects_parallel",
                                "Increase overlap_threshold (weak: wholly-absorbed parents score ~1.0)",
                                "Apply stronger area filtering",
                            ],
                            context={
                                "child_id": int(child_id),
                                "timestep": int(t0_abs + t),
                                "n_parents": int(n_parents),
                                "limit": MAX_PARENTS,
                            },
                        )

                    parent_masks_uint[parent_mask] = n_parents
                    parent_ids[n_parents] = parent_id
                    overlap_areas[n_parents] = overlap_area

                    mask_area = area[parent_mask]
                    weighted = np.array(
                        [
                            np.sum(mask_area * x[parent_mask]),
                            np.sum(mask_area * y[parent_mask]),
                            np.sum(mask_area * z[parent_mask]),
                        ],
                        dtype=np.float32,
                    )
                    norm = np.sqrt(np.sum(weighted * weighted))
                    parent_centroids[n_parents, 0] = np.degrees(np.arcsin(weighted[2] / norm))
                    parent_centroids[n_parents, 1] = np.degrees(np.arctan2(weighted[1], weighted[0]))
                    if parent_centroids[n_parents, 1] > 180:
                        parent_centroids[n_parents, 1] -= 360
                    elif parent_centroids[n_parents, 1] < -180:
                        parent_centroids[n_parents, 1] += 360
                    parent_areas[n_parents] = area_0
                    n_parents += 1

                if n_parents < 2:
                    continue
                max_parents_seen = max(max_parents_seen, n_parents)

                if merges_at_t >= MAX_MERGES:  # pragma: no cover
                    raise TrackingError(
                        "Too many merge operations",
                        details=f"Timestep {t0_abs + t} requires {merges_at_t + 1} merges (limit: {MAX_MERGES})",
                        suggestions=[
                            "Raise MAX_MERGES in split_and_merge_objects_parallel",
                            "Increase area_filter_quartile to reduce small objects",
                        ],
                        context={"timestep": int(t0_abs + t), "merge_count": int(merges_at_t), "limit": MAX_MERGES},
                    )
                merges_at_t += 1

                new_child_ids = np.arange(next_new_id, next_new_id + (n_parents - 1), dtype=np.int32)
                child_ids = np.concatenate((np.array([child_id], dtype=np.int32), new_child_ids))
                next_new_id += n_parents - 1
                records.append(
                    (
                        int(t0_abs + t),
                        child_ids.copy(),
                        parent_ids[:n_parents].copy(),
                        overlap_areas[:n_parents].copy(),
                    )
                )

                if nn_partitioning:
                    max_area = parent_areas[:n_parents].max() / mean_cell_area
                    max_distance = int(np.sqrt(max_area) * 2.0)
                    # No defensive copies: the kernel copies parent_frontiers itself and only
                    # reads child_mask and neighbours_int ((3, ncells) int32 is 178 MB on ICON).
                    new_labels_uint = partition_nn_unstructured_optimised(
                        child_mask,
                        parent_masks_uint,
                        parent_centroids,
                        neighbours_int,
                        lat,
                        lon,
                        max_distance=max(max_distance, 20) * 2,
                    )
                    new_labels = child_ids[new_labels_uint]
                    new_labels_uint = None
                else:
                    new_labels = partition_centroid_unstructured(child_mask, parent_centroids, child_ids, lat, lon)

                data_t[child_mask] = new_labels
                changed = True

                # Forward search: children at t+1 that overlap the new pieces above threshold.
                # Every cell holding a new id was in child_mask, so `child_cells[new_labels ==
                # new_id]` is exactly what `data_t == new_id` would find, ascending, at O(child).
                new_merging_list = []
                for new_id in child_ids:
                    parent_cells = child_cells[new_labels == new_id]
                    if not parent_cells.size:
                        continue
                    area_0 = area[parent_cells].sum()
                    for potential_child in np.unique(data_p1[parent_cells]):
                        if potential_child <= 0:
                            continue
                        potential_child_mask = data_p1 == potential_child
                        area_1 = area[potential_child_mask].sum()
                        min_area = min(area_0, area_1)
                        overlap_cells = parent_cells[potential_child_mask[parent_cells]]
                        overlap_area = area[overlap_cells].sum()
                        if overlap_area / min_area > overlap_threshold:
                            new_merging_list.append(int(potential_child))

                if t < n_t - 1:
                    for new_object_id in new_merging_list:
                        if new_object_id not in merging_objects_list[t + 1]:
                            merging_objects_list[t + 1].append(new_object_id)
                else:
                    for new_object_id in new_merging_list:
                        if new_object_id not in forwarded:
                            forwarded.append(new_object_id)

            # Consolidate t against its final t-1 before t+1 reads it: the serial path's order
            # (it consolidates t-1 against t-2 at the top of step t). Without this a split keeps
            # one ID per piece, and a split that rejoins is re-partitioned and logged as a merge
            # on every later day (D-087). A consolidated object is larger, so it can newly pass
            # the threshold against a child at t+1: queue those children for the parent check.
            grown = _consolidate_slice(data_m1, data_t, area, overlap_threshold)
            if grown:
                changed = True
                grown_cells = np.where(np.isin(data_t, grown))[0]
                for potential_child in np.unique(data_p1[grown_cells]):
                    if potential_child <= 0:
                        continue
                    target_queue = merging_objects_list[t + 1] if t < n_t - 1 else forwarded
                    if int(potential_child) not in target_queue:
                        target_queue.append(int(potential_child))

        # `last_hash` describes `data`, so the accumulator must hold `data`: a re-run that changes
        # nothing still rewrites a region an earlier run of this chunk changed, or the successor
        # reads stale labels whose hash never matches and stays dirty forever.
        if changed or region_dirty:
            zarr.open_group(zarr_path, mode="r+")["temp"][t0_abs : t0_abs + n_t, :] = data
        last_hash = hashlib.blake2b(data[-1].tobytes(), digest_size=16).hexdigest()
        return records, sorted(forwarded), last_hash, boundary_hash, int(max_parents_seen), bool(changed)

    # -- initial queue: children with >= 2 parents on the pristine field ---------------------
    overlap_objects_list = _overlap.find_overlapping_objects(
        object_id_field_unique, timedim, unstructured_grid, ydim, xdim, cell_area
    )
    overlap_objects_list = _overlap.enforce_overlap_threshold(
        overlap_objects_list, _objects.ObjectPropsStore.from_dataset(object_props), unstructured_grid, overlap_threshold
    )
    logger.info("Finished finding overlapping objects")
    unique_children, children_counts = np.unique(overlap_objects_list[:, 1], return_counts=True)
    initial_children = [int(c) for c in unique_children[children_counts > 1]]
    del overlap_objects_list
    global_id_counter = int(object_props.ID.max().item()) + 1
    # Temp IDs are minted from above the range compaction writes into. Every timestep can mint at most
    # `id_stride` IDs, so the compacted range [global_id_counter, global_id_counter + n_minted) never reaches
    # `mint_base`, and a compaction task that dask retries after a worker restart cannot remap an ID it has
    # already compacted (D-093). Compaction keeps sorted order, so the final IDs do not depend on the base.
    mint_base = global_id_counter + n_time * id_stride

    initial_queue: List[List[int]] = [[] for _ in range(n_time)]
    if initial_children:
        time_index_map = _overlap.compute_id_time_dict(
            object_id_field_unique,
            initial_children,
            global_id_counter,
            timedim,
            unstructured_grid,
            ydim,
            xdim,
            # Only the queued IDs are looked up, so restrict the search instead of broadcasting
            # a (time x buffer x max_objects) boolean over every possible ID (review finding 6.8).
            all_objects=False,
        )
        for child in initial_children:
            t_idx = time_index_map.get(child, -1)
            if 0 <= t_idx < n_time:
                initial_queue[t_idx].append(child)
        for t in range(n_time):
            initial_queue[t].sort()
    logger.debug("Finished Mapping Children to Time Indices")

    # -- the accumulator: a zarr copy of the pristine field -------------------------------
    zarr_path = temp_field_path
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    pristine = object_id_field_unique.rename("temp")
    pristine.to_zarr(zarr_path, mode="w")
    pristine_blocks = pristine.data  # (n_chunks, 1) blocks, persisted or staged upstream

    # Static arrays as single-chunk dask arrays: one key each, moved to a worker once, instead
    # of a numpy copy embedded in every chunk task.
    def _as_single_chunk(arr):
        values = arr.data if hasattr(arr, "data") else arr
        if isinstance(values, da.Array):
            return values.rechunk(-1).persist()
        return da.from_array(np.asarray(values), chunks=-1).persist()

    lat_d = _as_single_chunk(lat)
    lon_d = _as_single_chunk(lon)
    area_d = _as_single_chunk(cell_area)
    neighbours_d = _as_single_chunk(neighbours_int)

    def _boundary_slice(t_abs: int) -> NDArray[np.int32]:
        return np.asarray(zarr.open_group(zarr_path, mode="r")["temp"][t_abs, :], dtype=np.int32)

    # -- the loop: run dirty chunks until no boundary or forwarded queue changes -------------
    records_by_chunk: Dict[int, list] = {}
    forwarded_by_chunk: Dict[int, List[int]] = {k: [] for k in range(n_chunks)}
    last_hash: Dict[int, str] = {}
    consumed_boundary: Dict[int, str] = {}
    consumed_forwarded: Dict[int, List[int]] = {}
    max_parents_run = 0
    region_written: set = set()
    dirty = set(range(n_chunks))
    iteration = 0

    while dirty:
        if iteration >= max_iteration:  # pragma: no cover
            raise TrackingError(
                "Maximum iterations reached in tracking algorithm",
                details=f"{len(dirty)} time chunks still changing after {max_iteration} iterations",
                suggestions=["Increase max_iteration parameter", "Increase area_filter_quartile to reduce small objects"],
                context={"max_iteration": max_iteration, "dirty_chunks": sorted(dirty)},
            )
        logger.info(f"Merge loop iteration {iteration + 1}: {len(dirty)} of {n_chunks} time chunks to process")

        # Two phases by chunk parity: a chunk reads its boundary from the accumulator, and no
        # chunk in the same phase writes it, so a read can never see a half-written slice.
        for parity in (0, 1):
            todo = sorted(k for k in dirty if k % 2 == parity)
            if not todo:
                continue
            tasks = []
            for k in todo:
                t0, t1 = int(chunk_starts[k]), int(chunk_starts[k + 1])
                queue_k = [list(initial_queue[t]) for t in range(t0, t1)]
                fwd_in = list(forwarded_by_chunk[k - 1]) if k > 0 else []
                for obj in fwd_in:
                    if obj not in queue_k[0]:
                        queue_k[0].append(obj)
                consumed_forwarded[k] = fwd_in
                boundary = dask.delayed(_boundary_slice)(t0 - 1) if k > 0 else None
                next_first = pristine_blocks.blocks[k + 1, 0][0] if k + 1 < n_chunks else None
                id_offsets = mint_base + np.arange(t0, t1, dtype=np.int64) * id_stride
                tasks.append(
                    dask.delayed(process_chunk)(
                        pristine_blocks.blocks[k, 0],
                        boundary,
                        next_first,
                        queue_k,
                        id_offsets,
                        t0,
                        lat_d,
                        lon_d,
                        area_d,
                        neighbours_d,
                        zarr_path,
                        k in region_written,
                    )
                )
            results = dask.compute(*tasks)
            for k, (recs, fwd, h_last, h_boundary, max_p, wrote) in zip(todo, results):
                if wrote:
                    region_written.add(k)
                else:
                    region_written.discard(k)
                records_by_chunk[k] = recs
                forwarded_by_chunk[k] = fwd
                last_hash[k] = h_last
                consumed_boundary[k] = h_boundary
                max_parents_run = max(max_parents_run, max_p)

        # A chunk must run again when what it consumed from its predecessor is no longer
        # what the predecessor produced.
        next_dirty = set()
        for k in range(1, n_chunks):
            producer_hash = last_hash.get(k - 1)
            if producer_hash is None:
                continue
            if consumed_boundary.get(k) != producer_hash or consumed_forwarded.get(k) != forwarded_by_chunk[k - 1]:
                next_dirty.add(k)
        dirty = next_dirty
        iteration += 1

    n_records = sum(len(r) for r in records_by_chunk.values())
    logger.info(
        f"Merge loop converged after {iteration} iterations: {n_records} merge records, " f"max accepted parents {max_parents_run}"
    )

    # -- compaction: minted IDs become contiguous after the input's maximum ------------------
    all_records = []
    for k in range(n_chunks):
        all_records.extend(records_by_chunk.get(k, []))
    minted = set()
    for _, children, parents, _ in all_records:
        minted.update(int(c) for c in children if c >= global_id_counter)
        minted.update(int(p) for p in parents if p >= global_id_counter)
    temp_sorted = np.array(sorted(minted), dtype=np.int64)
    permanent = np.arange(global_id_counter, global_id_counter + len(temp_sorted), dtype=np.int32)

    def _compact(values: NDArray) -> NDArray:
        values = np.asarray(values)
        mask = values >= global_id_counter
        if not mask.any():
            return values
        out = values.copy()
        out[mask] = permanent[np.searchsorted(temp_sorted, values[mask])]
        return out

    if len(temp_sorted):

        def _compact_region(t0: int, t1: int) -> int:
            group = zarr.open_group(zarr_path, mode="r+")["temp"]
            block = np.asarray(group[t0:t1, :])
            mask = block >= mint_base
            if mask.any():
                block[mask] = permanent[np.searchsorted(temp_sorted, block[mask])]
                group[t0:t1, :] = block
            return int(mask.sum())

        dask.compute(*[dask.delayed(_compact_region)(int(chunk_starts[k]), int(chunk_starts[k + 1])) for k in range(n_chunks)])
        all_records = [(t, _compact(c), _compact(p), a) for t, c, p, a in all_records]

    object_id_field_unique = xr.open_zarr(zarr_path, chunks={timedim: chunk_sizes}).temp
    object_id_field_unique = object_id_field_unique.rename(pristine.name if pristine.name != "temp" else None)

    # -- merge events (same layout as the serial path) ---------------------------------------
    all_records.sort(key=lambda r: r[0])
    times = object_id_field_unique[timecoord].values
    global_child_ids = [r[1] for r in all_records]
    global_parent_ids = [r[2] for r in all_records]
    global_merge_areas = [r[3] for r in all_records]
    global_merge_tidx = np.array([r[0] for r in all_records], dtype=int)

    if global_parent_ids and global_child_ids:
        max_parents = max(len(ids) for ids in global_parent_ids)
        max_children = max(len(ids) for ids in global_child_ids)
    else:
        max_parents = 1
        max_children = 1

    parent_ids_array = np.full((len(global_parent_ids), max_parents), -1, dtype=np.int32)
    child_ids_array = np.full((len(global_child_ids), max_children), -1, dtype=np.int32)
    overlap_areas_array = np.full(
        (len(global_merge_areas), max_parents),
        -1,
        dtype=np.float32 if unstructured_grid else np.int32,
    )
    for i, parents in enumerate(global_parent_ids):
        parent_ids_array[i, : len(parents)] = parents
    for i, children in enumerate(global_child_ids):
        child_ids_array[i, : len(children)] = children
    for i, areas in enumerate(global_merge_areas):
        overlap_areas_array[i, : len(areas)] = areas

    merge_events = xr.Dataset(
        {
            "parent_IDs": (("merge_ID", "parent_idx"), parent_ids_array),
            "child_IDs": (("merge_ID", "child_idx"), child_ids_array),
            "overlap_areas": (("merge_ID", "parent_idx"), overlap_areas_array),
            "merge_time": ("merge_ID", times[global_merge_tidx]),
            "n_parents": (
                "merge_ID",
                np.array([len(p) for p in global_parent_ids], dtype=np.int8),
            ),
            "n_children": (
                "merge_ID",
                np.array([len(c) for c in global_child_ids], dtype=np.int8),
            ),
        },
        attrs={"fill_value": -1},
    )

    object_id_field_unique = _anchor_field(object_id_field_unique, "merged_id_field", materialiser)
    object_props = _objects.calculate_object_properties(
        object_id_field_unique,
        unstructured_grid,
        lat,
        lon,
        cell_area,
        timedim,
        regional_mode,
        ydim,
        xdim,
        properties=["area", "centroid"],
    ).persist(optimize_graph=True)

    overlap_objects_list = _overlap.find_overlapping_objects(
        object_id_field_unique, timedim, unstructured_grid, ydim, xdim, cell_area
    )
    overlap_objects_list = _overlap.enforce_overlap_threshold(
        overlap_objects_list, _objects.ObjectPropsStore.from_dataset(object_props), unstructured_grid, overlap_threshold
    )
    overlap_objects_list = overlap_objects_list[:, :2].astype(np.int32)

    return (
        object_id_field_unique,
        object_props,
        overlap_objects_list,
        merge_events,
    )
