"""
MarEx Track: Overlap and tracking helpers.

Module-level helpers supporting object overlap accounting and tracking. This
module holds the JIT-compiled ``sparse_bool_power`` primitive used for fast
morphological operations on unstructured grids, together with the stateless
overlap/consolidation routines (cross-time overlap detection, overlap-threshold
enforcement, ID consolidation, and the ID/time lookup builder) extracted from
the tracker orchestrator. The tracker config/grid values each method read from
``self`` are threaded in as explicit arguments. Behaviour and numerics are
identical to the original ``tracker`` methods.
"""

import functools
from typing import Dict, List, Tuple, Union

import dask
import numpy as np
import xarray as xr
from dask.base import is_dask_collection
from numba import njit, prange
from numpy.typing import NDArray

from ..logging_config import get_logger

logger = get_logger(__name__)


@njit(fastmath=True, parallel=True)
def sparse_bool_power(
    vec: NDArray[np.bool_],
    sp_data: NDArray[np.bool_],
    indices: NDArray[np.int32],
    indptr: NDArray[np.int32],
    exponent: int,
) -> NDArray[np.bool_]:  # pragma: no cover
    """
    Efficient sparse boolean matrix power operation.

    This function implements a fast sparse matrix power operation for boolean matrices,
    avoiding memory leaks present in scipy+Dask implementations. It's used for efficient
    morphological operations on unstructured grids.

    Parameters
    ----------
    vec : np.ndarray
        Boolean vector to multiply
    sp_data, indices, indptr : np.ndarray
        Sparse matrix in CSR format
    exponent : int
        Number of times to apply the matrix

    Returns
    -------
    np.ndarray
        Result of (sparse_matrix ^ exponent) * vec
    """
    vec = vec.T
    num_rows = indptr.size - 1
    num_cols = vec.shape[1]
    result = vec.copy()

    for _ in range(exponent):
        temp_result = np.zeros((num_rows, num_cols), dtype=np.bool_)

        for i in prange(num_rows):
            for j in range(indptr[i], indptr[i + 1]):
                if sp_data[j]:
                    for k in range(num_cols):
                        if result[indices[j], k]:
                            temp_result[i, k] = True

        result = temp_result

    return result.T


def check_overlap_slice(
    ids_t0: NDArray[np.int32],
    ids_next: NDArray[np.int32],
    unstructured_grid: bool,
    cell_area: xr.DataArray,
) -> NDArray[Union[np.float32, np.int32]]:
    """
    Find overlapping objects between two consecutive time slices.

    Parameters
    ----------
    ids_t0 : numpy.ndarray
        Object IDs at current time
    ids_next : numpy.ndarray
        Object IDs at next time

    Returns
    -------
    numpy.ndarray
        Array of shape (n_overlaps, 3) with [id_t0, id_next, overlap_area]
    """
    # Create masks for valid IDs
    mask_t0 = ids_t0 > 0
    mask_next = ids_next > 0

    # Only process cells where both times have valid IDs
    combined_mask = mask_t0 & mask_next

    if not np.any(combined_mask):
        return np.empty((0, 3), dtype=np.float32 if unstructured_grid else np.int32)

    # Extract the overlapping points
    ids_t0_valid = ids_t0[combined_mask]
    ids_next_valid = ids_next[combined_mask]

    # Create a unique identifier for each pair
    # This is faster than using np.unique with axis=1
    max_id = max(ids_t0.max(), ids_next.max() + 1).astype(np.int64)
    pair_ids = ids_t0_valid.astype(np.int64) * max_id + ids_next_valid.astype(np.int64)

    if unstructured_grid:
        # Get unique pairs and their inverse indices
        unique_pairs, inverse_indices = np.unique(pair_ids, return_inverse=True)
        inverse_indices = inverse_indices.astype(np.int32)  # Ensure int32 for serialisation

        # Sum areas for overlapping cells
        areas_valid = (cell_area.values if isinstance(cell_area, xr.DataArray) else cell_area)[combined_mask]
        areas = np.zeros(len(unique_pairs), dtype=np.float32)
        np.add.at(areas, inverse_indices, areas_valid)
    else:
        # Get unique pairs and their counts (pixel counts)
        unique_pairs, areas = np.unique(pair_ids, return_counts=True)
        areas = areas.astype(np.int32)

    # Convert back to original ID pairs
    id_t0 = (unique_pairs // max_id).astype(np.int32)
    id_next = (unique_pairs % max_id).astype(np.int32)

    # Stack results
    result = np.column_stack((id_t0, id_next, areas))

    return result


def _check_overlap_slice_unstructured(
    ids_t0: NDArray[np.int32], ids_next: NDArray[np.int32], cell_area: NDArray[np.float32]
) -> NDArray[np.float32]:
    """``check_overlap_slice`` on an unstructured grid, with the cell areas as a positional argument."""
    return check_overlap_slice(ids_t0, ids_next, True, cell_area)


def find_overlapping_objects(
    object_id_field: xr.DataArray,
    timedim: str,
    unstructured_grid: bool,
    ydim,
    xdim: str,
    cell_area: xr.DataArray,
) -> NDArray[Union[np.float32, np.int32]]:
    """
    Find all overlapping objects across time.

    Parameters
    ----------
    object_id_field : xarray.DataArray
        Field containing object IDs

    Returns
    -------
    overlap_objects_list_unique_filtered : (N x 3) numpy.ndarray
        Array of object ID pairs that overlap across time, with overlap area
        The object in the first column precedes the second column in time.
        The third column contains:
            * For structured grid: number of overlapping pixels (int32)
            * For unstructured grid: total overlapping area in m^2 (float32)
    """
    # Check just for overlap with next time slice.
    #  Keep a running list of all object IDs that overlap
    object_id_field_next = object_id_field.shift({timedim: -1}, fill_value=0)

    # Calculate overlaps in parallel
    input_dims = [xdim] if unstructured_grid else [ydim, xdim]
    if unstructured_grid:
        # Pass cell_area as a single-chunk core-dimension argument: every overlap task then
        # depends on one shared area key, instead of carrying its own copy of the array
        # inside a functools.partial. Pass the bare array so it is matched to the cells by
        # position, as before, and never aligned on a coordinate.
        area = cell_area.chunk({xdim: -1}).data
        args = (object_id_field, object_id_field_next, area)
        core_dims = [input_dims, input_dims, [xdim]]
        kernel = _check_overlap_slice_unstructured
    else:
        args = (object_id_field, object_id_field_next)
        core_dims = [input_dims, input_dims]
        # The structured branch counts pixels and never reads cell_area.
        kernel = functools.partial(check_overlap_slice, unstructured_grid=False, cell_area=None)
    overlap_object_pairs_list = xr.apply_ufunc(
        kernel,
        *args,
        input_core_dims=core_dims,
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[object],
    ).persist()

    # Concatenate all pairs from different chunks
    all_pairs_with_areas = np.concatenate(overlap_object_pairs_list.values)

    # Get unique pairs and their indices
    unique_pairs, inverse_indices = np.unique(all_pairs_with_areas[:, :2], axis=0, return_inverse=True)
    inverse_indices = inverse_indices.astype(np.int32)  # Ensure int32 for serialisation

    # Take the per-pair MAXIMUM single-boundary overlap, not the sum. Post-tracking, IDs
    # persist across timesteps, so a pair (A, B) recurs at every time boundary they overlap;
    # summing pooled those into a total that, divided by a single-time area downstream,
    # exceeded 1.0 ("overlap fraction > 1.0" warnings) and could pass a pair whose per-boundary
    # overlap never met the threshold. The max is the largest instantaneous overlap.
    output_dtype = np.float32 if unstructured_grid else np.int32
    total_summed_areas = np.zeros(len(unique_pairs), dtype=output_dtype)
    np.maximum.at(total_summed_areas, inverse_indices, all_pairs_with_areas[:, 2])

    # Stack the pairs with their per-pair max overlap area
    overlap_objects_list_unique = np.column_stack((unique_pairs, total_summed_areas))

    return overlap_objects_list_unique


def enforce_overlap_threshold(
    overlap_objects_list: NDArray[Union[np.float32, np.int32]],
    object_props,
    unstructured_grid: bool,
    overlap_threshold: float,
) -> NDArray[Union[np.float32, np.int32]]:
    """
    Filter object pairs based on overlap threshold.

    Parameters
    ----------
    overlap_objects_list : (N x 3) numpy.ndarray
        Array of object ID pairs with overlap area
    object_props : ObjectPropsStore
        O(1) per-ID area/centroid store (membership and ``areas`` lookups used here)

    Returns
    -------
    overlap_objects_list_filtered : (M x 3) numpy.ndarray
        Filtered array of object ID pairs that meet the overlap threshold
    """
    if len(overlap_objects_list) == 0:
        return np.empty((0, 3), dtype=np.float32 if unstructured_grid else np.int32)

    # Filter out overlaps where either ID doesn't exist in object_props. One vectorised
    # membership test over both ID columns rather than a Python comprehension per row --
    # this list runs to millions of rows on a full-length run.
    valid_mask = object_props.contains_many(overlap_objects_list[:, 0]) & object_props.contains_many(overlap_objects_list[:, 1])

    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float32 if unstructured_grid else np.int32)

    valid_overlaps = overlap_objects_list[valid_mask]

    # Calculate overlap fractions
    areas_0 = object_props.areas(valid_overlaps[:, 0])
    areas_1 = object_props.areas(valid_overlaps[:, 1])
    min_areas = np.minimum(areas_0, areas_1)
    overlap_fractions = valid_overlaps[:, 2].astype(float) / min_areas

    if np.any(overlap_fractions > 1.0):
        logger.warning(f"Found {np.sum(overlap_fractions > 1.0)} overlap fractions > 1.0")
        logger.warning(f"Max overlap fraction: {overlap_fractions.max()}")

    # Filter by threshold
    threshold_mask = overlap_fractions >= overlap_threshold
    overlap_objects_list_filtered = valid_overlaps[threshold_mask]

    return overlap_objects_list_filtered


def consolidate_object_ids(
    data_t_minus_2: xr.DataArray,
    data_t_minus_1: xr.DataArray,
    object_props,
    timestep: int,
    unstructured_grid: bool,
    cell_area: xr.DataArray,
    overlap_threshold: float,
    lat: xr.DataArray,
    lon: xr.DataArray,
    timedim: str,
    regional_mode: bool,
    ydim,
    xdim: str,
) -> Tuple[xr.DataArray, xr.Dataset]:
    """
    Consolidate object IDs between t-2 and t-1 to ensure consistent tracking.

    This identifies objects at t-1 that are actually continuations of objects
    from t-2 (but got different IDs due to partitioning) and renames them
    to maintain consistent IDs across timesteps.

    Parameters
    ----------
    data_t_minus_2 : xr.DataArray
        Object field at timestep t-2
    data_t_minus_1 : xr.DataArray
        Object field at timestep t-1 (will be modified)
    object_props : xr.Dataset
        Object properties dataset (will be modified)
    timestep : int
        Current timestep number for logging purposes

    Returns
    -------
    data_t_minus_1_consolidated : xr.DataArray
        Updated t-1 field with consolidated IDs
    object_props_updated : ObjectPropsStore
        The same store, mutated in place (consolidated objects updated, redundant ones dropped)

    Notes
    -----
    - Uses overlap_threshold for determining consolidation eligibility
    - Updates object properties by recalculating for consolidated objects
    - Removes redundant child objects from object_props
    """
    # Imported here to avoid a module-level overlap->objects import edge
    from .objects import calculate_object_properties

    # Find overlaps between t-2 and t-1
    backward_overlaps = check_overlap_slice(data_t_minus_2.values, data_t_minus_1.values, unstructured_grid, cell_area)
    if len(backward_overlaps) == 0:
        return data_t_minus_1, object_props

    backward_overlaps = enforce_overlap_threshold(backward_overlaps, object_props, unstructured_grid, overlap_threshold)
    if len(backward_overlaps) == 0:  # pragma: no cover
        return data_t_minus_1, object_props

    # Find parent IDs that connect to multiple children (partition boundary jumps)
    parent_ids, parent_counts = np.unique(backward_overlaps[:, 0], return_counts=True)
    splitting_parents = parent_ids[parent_counts > 1]

    if len(splitting_parents) == 0:
        return data_t_minus_1, object_props

    # Track ID mappings for logging
    id_mappings = {}  # child_id -> parent_id

    for parent_id in splitting_parents:
        # Skip if parent doesn't exist in properties
        if parent_id not in object_props:
            continue

        # Get all children for this parent
        child_mask = backward_overlaps[:, 0] == parent_id
        children_for_parent = backward_overlaps[child_mask, 1].astype(int)

        # Consolidate all children to use first child_id
        if len(children_for_parent) > 1:
            first_child_id = int(children_for_parent[0])

            # Skip if first child doesn't exist in properties
            if first_child_id not in object_props:
                continue

            # Rename all other children to first_child_id. One vectorised pass over the
            # slice for the whole group rather than a full-slice .where() per child.
            other_children = [int(c) for c in children_for_parent[1:] if int(c) in object_props]
            if other_children:
                values = data_t_minus_1.values
                data_t_minus_1 = data_t_minus_1.copy(
                    data=np.where(np.isin(values, other_children), first_child_id, values).astype(values.dtype)
                )
                for child_id in other_children:
                    # Remove redundant child_id from object_props
                    object_props.drop(child_id)

                    # Track the mapping
                    id_mappings[child_id] = first_child_id

            # Recalculate properties for the consolidated object
            consolidated_mask = data_t_minus_1 == first_child_id
            if consolidated_mask.any():
                # Create temporary field with only this object for property calculation
                temp_field = xr.where(consolidated_mask, first_child_id, 0)
                consolidated_props = calculate_object_properties(
                    temp_field,
                    unstructured_grid=unstructured_grid,
                    lat=lat,
                    lon=lon,
                    cell_area=cell_area,
                    timedim=timedim,
                    regional_mode=regional_mode,
                    ydim=ydim,
                    xdim=xdim,
                    properties=["area", "centroid"],
                )

                if first_child_id in consolidated_props.ID:
                    # Update first child properties with consolidated values
                    cp = consolidated_props.sel(ID=first_child_id)
                    object_props.set(
                        first_child_id,
                        cp["area"].values.item(),
                        cp["centroid"].values[0],
                        cp["centroid"].values[1],
                    )

    return data_t_minus_1, object_props


def _first_time_in_block(block: NDArray[np.int32], t0: int, buffer: int) -> Tuple[NDArray[np.int32], NDArray[np.int64]]:
    """Distinct IDs of a (time, ...) block and the global time index where each first appears in it.

    ID 0 counts as present in a slice that has background, or fewer than ``buffer`` distinct
    IDs, which is where the earlier fixed-width per-slice ID list carried zero padding.
    """
    uniques = []
    for row in block.reshape(block.shape[0], -1):
        u = np.unique(row)
        if u.size < buffer and not np.any(u == 0):
            u = np.concatenate((np.zeros(1, dtype=u.dtype), u))
        uniques.append(u)
    ids = np.concatenate(uniques)
    times = np.concatenate([np.full(u.size, t0 + t, dtype=np.int64) for t, u in enumerate(uniques)])
    distinct, first = np.unique(ids, return_index=True)  # rows are in time order, so this is the earliest
    return distinct, times[first]


def compute_id_time_dict(
    da: xr.DataArray,
    child_objects: Union[List[int], NDArray[np.int32]],
    max_objects: int,
    timedim: str,
    unstructured_grid: bool,
    ydim,
    xdim: str,
    all_objects: bool = True,
) -> Dict[int, int]:
    """
    Generate lookup table mapping object IDs to their time index.

    Parameters
    ----------
    da : xarray.DataArray
        Field of object IDs
    child_objects : list or array
        Object IDs to include in the dictionary
    max_objects : int
        Maximum number of objects
    all_objects : bool, default=True
        Whether to process all objects or just child_objects

    Returns
    -------
    time_index_map : dict
        Dictionary mapping object IDs to time indices
    """
    # Each block reports its distinct IDs and the first time each appears in it; those
    # compact pairs are gathered and reduced to the global first occurrence. Nothing of
    # size (time x objects per slice x queried IDs) is ever built.
    input_dims = [xdim] if unstructured_grid else [ydim, xdim]
    buffer = int(max_objects / da[timedim].shape[0] * 100)
    data = da.transpose(timedim, *input_dims).data
    if is_dask_collection(data) and any(len(c) > 1 for c in data.chunks[1:]):
        # Which IDs a slice holds, and whether it has background, is a whole-slice property.
        raise ValueError(f"compute_id_time_dict needs {input_dims} in a single chunk; got chunks {data.chunks[1:]}")
    if is_dask_collection(data):
        time_starts = np.cumsum((0,) + data.chunks[0])[:-1]
        blocks = data.to_delayed()
        first_time = dask.delayed(_first_time_in_block, pure=True)
        parts = dask.compute(*[first_time(blocks[idx], int(time_starts[idx[0]]), buffer) for idx in np.ndindex(blocks.shape)])
    else:
        parts = [_first_time_in_block(np.asarray(data), 0, buffer)]
    ids = np.concatenate([p[0] for p in parts])
    times = np.concatenate([p[1] for p in parts])
    order = np.lexsort((times, ids))
    seen_ids, first = np.unique(ids[order], return_index=True)
    first_times = times[order][first]

    # Look up the queried IDs. An ID that never appears maps to time index 0, as the
    # earlier argmax over an all-False presence mask did.
    search_ids = np.arange(max_objects, dtype=np.int64) if all_objects else np.asarray(child_objects, dtype=np.int64)
    time_index = np.zeros(len(search_ids), dtype=np.int64)
    if len(seen_ids):
        pos = np.minimum(np.searchsorted(seen_ids, search_ids), len(seen_ids) - 1)
        found = seen_ids[pos] == search_ids
        time_index[found] = first_times[pos[found]]

    # Convert to dictionary for fast lookup
    time_index_map = {int(id_val): int(idx) for id_val, idx in zip(search_ids, time_index)}

    return time_index_map
