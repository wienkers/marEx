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

import gc
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import dask.array as da
import numpy as np
import xarray as xr
from dask import persist
from dask.distributed import wait
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
        """Process all mergers for a single block of timesteps."""
        result = xr.full_like(time_block, -1)

        # Get unique times in this block
        # time_block might not have the coordinate, so get it from the dimension index
        if timecoord in time_block.coords:
            unique_times = np.unique(time_block.coords[timecoord])
        else:
            # Fall back to using the dimension index
            unique_times = np.unique(time_block[timedim])

        for time_val in unique_times:
            # Get IDs for this time
            time_mask = IDs_coords["merge_time"] == time_val
            if not np.any(time_mask):
                continue

            # IDs_data[time_mask] always retains the parent_idx axis, so IDs_at_time is 2D
            # (n_mergers_at_time, n_parent_idx); iterate over each merger's parent vector.
            IDs_at_time = IDs_data[time_mask]

            for merger_IDs in IDs_at_time:
                valid_mask = merger_IDs > 0
                if np.any(valid_mask):
                    expanded_IDs = np.broadcast_to(
                        merger_IDs,
                        (len(time_block.sibling_ID), len(merger_IDs)),
                    )
                    result.loc[{timedim: time_val, "ID": merger_IDs[valid_mask]}] = expanded_IDs[:, valid_mask]

        return result

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

        # End-of-chunk consolidation: consolidate the last timestep if chunk has multiple timesteps
        if chunk_data.sizes[timedim] >= 2:

            # Get last and second-to-last timesteps
            last_t_data = chunk_data.isel({timedim: -1})
            second_last_t_data = chunk_data.isel({timedim: -2})

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
    # Constants for memory allocation
    # Sized from measurement, not judgement: job 27105655 instrumented 5 x 24-timestep
    # windows of the ICON R02B09 store (MAX_PARENTS=100/MAX_MERGES=200 so nothing capped)
    # and saw, over 707 real merges, max n_parents = 10 (99th pct 7) and max merges/timestep
    # = 8 (99th pct 6). Both constants are bounded together by the per-timestep update-id
    # space -- `updates_array` is uint8 with 255 as its "no update" sentinel and
    # `updates_ids` has exactly 255 slots -- giving
    #     MAX_MERGES * (MAX_PARENTS - 1) <= 255
    # 16 x 15 = 240 fits. MAX_MERGES comes DOWN 20 -> 16 (still 2x the measured max of 8)
    # to buy parent slots, because the parent tail is what actually failed: run 27098021
    # died at global t~767 on a 10-parent child. The windowed histogram is a LOWER BOUND on
    # that tail -- a 24-step window restarts the merge loop from a clean `data_m1`, so
    # build-up across a 1096-step run is not reproduced -- which is why the margin over the
    # observed 10 is deliberately wide rather than one or two slots.
    # If a full run still exceeds these, the fix is NOT another bump: it is widening
    # updates_array/updates_ids to uint16, or lowering R_fill.
    # See docs/superpowers/reports/REPORT_max_parents_diagnosis.md.
    MAX_MERGES = 16  # Maximum number of merges per timestep (measured max 8)
    MAX_PARENTS = 16  # Maximum number of parents per merge (measured max 10)
    # NOTE: this also sets MAX_CHILDREN, the width of `child_ids_iter`. Children-per-split
    # was NOT measured; 10 -> 16 only widens that array, so it can add headroom but never
    # remove any.
    MAX_CHILDREN = MAX_PARENTS

    def process_chunk(
        chunk_data_m1_full: NDArray[np.int32],
        chunk_data_p1_full: NDArray[np.int32],
        merging_objects: NDArray[np.int64],
        next_id_start: NDArray[np.int64],
        lat: NDArray[np.float32],
        lon: NDArray[np.float32],
        area: NDArray[np.float32],
        neighbours_int: NDArray[np.int32],
    ) -> Tuple[
        NDArray[np.int32],  # merge_child_ids
        NDArray[np.int32],  # merge_parent_ids
        NDArray[np.float32],  # merge_areas
        NDArray[np.int16],  # merge_counts
        NDArray[np.bool_],  # has_merge
        NDArray[np.uint8],  # updates_array
        NDArray[np.int32],  # updates_ids
        NDArray[np.int32],  # final_merging_objects
    ]:
        """
        Process a single chunk of merging objects.

        This function handles the complex batch processing of splitting and merging objects
        across timesteps within a single chunk. It finds overlapping objects, determines
        parent-child relationships, and creates new IDs as needed.

        Parameters
        ----------
        chunk_data_m1_full : numpy.ndarray
            Data from previous timestep (t-1) and current timestep (t)
        chunk_data_p1_full : numpy.ndarray
            Data from next timestep (t+1)
        merging_objects : (n_time, max_merges) numpy.ndarray
            IDs of objects to process
        next_id_start : (n_time, max_merges) numpy.ndarray
            Starting ID values for new objects
        lat, lon : numpy.ndarray
            Latitude/longitude arrays
        area : numpy.ndarray
            Cell area array
        neighbours_int : numpy.ndarray
            Neighbor connectivity array

        Returns
        -------
        tuple
            Contains merge events, object updates, and newly created objects
        """
        # Fix Broadcasted dimensions of inputs:
        #    Remove extra dimension if present while preserving time chunks
        #    N.B.: This is a weird artefact/choice of xarray apply_ufunc broadcasting...
        #           (i.e. 'nv' dimension gets injected into all the other arrays!)

        # Squeeze only the injected trailing (e.g. 'nv') axes, never the time axis. A blanket
        # .squeeze() drops the time dimension for a size-1 time chunk (n_time % timechunks == 1),
        # after which [0]/[1] index cells instead of the stacked (prev, current) slices and
        # merges in that chunk are silently dropped or crash. Target shape is (2, time, ncells).
        while chunk_data_m1_full.ndim > 3:
            chunk_data_m1_full = chunk_data_m1_full.squeeze(axis=-1)
        # `.astype()` already returns a fresh, writable buffer that does not alias the task
        # input (numpy's `copy` argument defaults to True, even when the dtype is unchanged),
        # so the trailing `.copy()` these three lines used to carry was a second full
        # duplicate of each array, purely transient: two single time slices and one whole
        # chunk, ~358 MB per task on the ICON mesh at timechunks=4. The no-aliasing property
        # the merge loop relies on (it mutates `data_t` in place) is a property of `astype`,
        # not of the dropped copy.
        #
        # `chunk_data_m1_full` is the field shifted forward one step, so its first two TIME
        # entries are (field[t-1], field[t]) for this chunk's first timestep -- which is why
        # both come from the m1 argument and are single slices, not whole chunks.
        chunk_data_m1 = chunk_data_m1_full[0].astype(np.int32)
        chunk_data = chunk_data_m1_full[1].astype(np.int32)
        del chunk_data_m1_full  # Free memory immediately
        chunk_data_p1 = chunk_data_p1_full.astype(np.int32)
        # Remove any singleton dimensions except time and space
        while chunk_data_p1.ndim > 2:
            chunk_data_p1 = chunk_data_p1.squeeze(axis=-1)
        del chunk_data_p1_full

        # Extract and prepare input arrays
        lat = lat.squeeze().astype(np.float32)
        lon = lon.squeeze().astype(np.float32)
        area = area.squeeze().astype(np.float32)
        next_id_start = next_id_start.squeeze()

        # Handle neighbours_int with correct dimensions (nv, ncells)
        neighbours_int = neighbours_int.squeeze()
        if neighbours_int.shape[1] != lat.shape[0]:
            neighbours_int = neighbours_int.T

        # Handle multiple merging objects - ensure proper dimensionality
        merging_objects = merging_objects.squeeze()
        if merging_objects.ndim == 1:
            merging_objects = merging_objects[:, None]  # Add dimension for max_merges

        # Pre-convert lat/lon to Cartesian coordinates for efficiency
        x = (np.cos(np.radians(lat)) * np.cos(np.radians(lon))).astype(np.float32)
        y = (np.cos(np.radians(lat)) * np.sin(np.radians(lon))).astype(np.float32)
        z = np.sin(np.radians(lat)).astype(np.float32)

        # Pre-allocate output arrays
        n_time = chunk_data_p1.shape[0]
        n_points = chunk_data_p1.shape[1]

        merge_child_ids = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.int32)
        merge_parent_ids = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.int32)
        merge_areas = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.float32)
        merge_counts = np.zeros(n_time, dtype=np.int16)  # Number of merges per timestep

        updates_array = np.full((n_time, n_points), 255, dtype=np.uint8)
        updates_ids = np.full((n_time, 255), -1, dtype=np.int32)
        has_merge = np.zeros(n_time, dtype=np.bool_)

        # Prepare merging objects list for each timestep
        merging_objects_list = [list(merging_objects[i][merging_objects[i] > 0]) for i in range(merging_objects.shape[0])]
        final_merging_objects = np.full((n_time, MAX_MERGES), -1, dtype=np.int32)
        final_merge_count = 0

        # Process each timestep
        data_p1 = []
        for t in range(n_time):
            next_new_id = next_id_start[t]  # Use the offset for this timestep

            # Get current time slice data
            if t == 0:
                data_m1 = chunk_data_m1
                data_t = chunk_data
                del chunk_data_m1, chunk_data  # Free memory
            else:
                data_m1 = data_t  # Previous data_t becomes data_m1
                data_t = data_p1  # Previous data_p1 becomes data_t
            data_p1 = chunk_data_p1[t]

            # Process each merging object at this timestep
            while merging_objects_list[t]:
                child_id = merging_objects_list[t].pop(0)

                # Get child mask and identify overlapping parents
                child_mask = data_t == child_id
                # Ascending cell indices of the child. Hoisted here from after the
                # partitioning step, where it used to be recomputed; child_mask is not
                # modified in between. Objects are tiny next to the mesh, so testing a
                # candidate parent AT these indices costs O(child) where intersecting two
                # whole-field boolean masks costs O(ncells).
                child_cells = np.where(child_mask)[0].astype(np.int32)

                # Find parent objects that overlap with this child
                potential_parents = np.unique(data_m1[child_mask])
                # Loop-invariant: the child's own area was recomputed once per candidate
                # parent inside the scan below.
                child_area = area[child_mask].sum()
                parent_iterator = 0
                parent_masks_uint = np.full(n_points, 255, dtype=np.uint8)
                parent_centroids = np.full((MAX_PARENTS, 2), -1.0e10, dtype=np.float32)
                parent_ids = np.full(MAX_PARENTS, -1, dtype=np.int32)
                parent_areas = np.zeros(MAX_PARENTS, dtype=np.float32)
                overlap_areas = np.zeros(MAX_PARENTS, dtype=np.float32)
                n_parents = 0

                # Find all unique parent IDs with significant overlap
                for parent_id in potential_parents[potential_parents > 0]:
                    parent_mask = data_m1 == parent_id
                    # Ascending, and exactly `np.where(parent_mask & child_mask)[0]` -- but
                    # evaluated only at the child's own cells, so it replaces two whole-mesh
                    # boolean intersections (the `any` test and the overlap gather) with two
                    # O(child) lookups.
                    overlap_cells = child_cells[parent_mask[child_cells]]
                    if overlap_cells.size:
                        # Calculate overlap area and check if it's large enough
                        area_0 = area[parent_mask].sum()  # Parent area
                        area_1 = child_area  # Child area
                        min_area = np.minimum(area_0, area_1)
                        overlap_area = area[overlap_cells].sum()

                        # Skip if overlap is below threshold
                        if overlap_area / min_area < overlap_threshold:
                            continue

                        # Only now is this candidate an ACCEPTED parent, so only now can it
                        # exhaust the fixed-width arrays. Checking at the top of the loop
                        # instead -- as this did until 2026-08-20 -- raises whenever
                        # MAX_PARENTS are accepted and ANY further candidate id remains in
                        # `potential_parents`, even though every one of those may be about to
                        # fail the overlap threshold above and the arrays hold indices
                        # 0..MAX_PARENTS-1 exactly. On a basin-scale child `potential_parents`
                        # is mostly such rejects, so that fired on merges the arrays could
                        # hold. See docs/superpowers/reports/REPORT_max_parents_diagnosis.md.
                        if n_parents >= MAX_PARENTS:  # pragma: no cover
                            raise TrackingError(
                                "Too many parent objects for tracking",
                                details=(
                                    f"Child {child_id} at timestep {t} has more than "
                                    f"{MAX_PARENTS} parents (limit: {MAX_PARENTS})"
                                ),
                                suggestions=[
                                    "Raise MAX_PARENTS, honouring MAX_MERGES * (MAX_PARENTS - 1) <= 255",
                                    "Increase overlap_threshold (weak: wholly-absorbed parents score ~1.0)",
                                    "Apply stronger area filtering",
                                ],
                                context={
                                    "child_id": child_id,
                                    "timestep": t,
                                    "n_parents": n_parents,
                                    "limit": MAX_PARENTS,
                                },
                            )

                        # Record parent information
                        parent_masks_uint[parent_mask] = parent_iterator
                        parent_ids[n_parents] = parent_id
                        overlap_areas[n_parents] = overlap_area

                        # Calculate area-weighted centroid for this parent
                        mask_area = area[parent_mask]
                        weighted_coords = np.array(
                            [
                                np.sum(mask_area * x[parent_mask]),
                                np.sum(mask_area * y[parent_mask]),
                                np.sum(mask_area * z[parent_mask]),
                            ],
                            dtype=np.float32,
                        )

                        norm = np.sqrt(np.sum(weighted_coords * weighted_coords))

                        # Convert back to lat/lon
                        parent_centroids[n_parents, 0] = np.degrees(np.arcsin(weighted_coords[2] / norm))
                        parent_centroids[n_parents, 1] = np.degrees(np.arctan2(weighted_coords[1], weighted_coords[0]))

                        # Fix longitude range to [-180, 180]
                        if parent_centroids[n_parents, 1] > 180:
                            parent_centroids[n_parents, 1] -= 360
                        elif parent_centroids[n_parents, 1] < -180:
                            parent_centroids[n_parents, 1] += 360

                        parent_areas[n_parents] = area_0
                        parent_iterator += 1
                        n_parents += 1

                # Need at least 2 parents for merging
                if n_parents < 2:
                    continue

                # Create new IDs for each partition
                new_child_ids = np.arange(next_new_id, next_new_id + (n_parents - 1), dtype=np.int32)
                child_ids = np.concatenate((np.array([child_id]), new_child_ids))

                # Record merge event
                curr_merge_idx = merge_counts[t]
                if curr_merge_idx >= MAX_MERGES:  # pragma: no cover
                    raise TrackingError(
                        "Too many merge operations",
                        details=f"Timestep {t} requires {curr_merge_idx + 1} merges (limit: {MAX_MERGES})",
                        suggestions=[
                            "Increase area_filter_quartile to reduce small objects",
                            "Consider adjusting tracking parameters",
                        ],
                        context={
                            "timestep": t,
                            "merge_count": curr_merge_idx,
                            "limit": MAX_MERGES,
                        },
                    )

                merge_child_ids[t, curr_merge_idx, :n_parents] = child_ids[:n_parents]
                merge_parent_ids[t, curr_merge_idx, :n_parents] = parent_ids[:n_parents]
                merge_areas[t, curr_merge_idx, :n_parents] = overlap_areas[:n_parents]
                merge_counts[t] += 1
                has_merge[t] = True

                # Partition the child object based on parent associations
                if nn_partitioning:
                    # Estimate maximum search distance based on object size
                    max_area = parent_areas.max() / mean_cell_area
                    max_distance = int(np.sqrt(max_area) * 2.0)

                    # Use optimised nearest-neighbor partitioning.
                    #
                    # No defensive copies here: the kernel makes its own working copy of
                    # parent_frontiers (the only array it writes) and merely reads
                    # child_mask and neighbours_int. Copying neighbours_int was by far the
                    # worst of the three -- (3, ncells) int32 is 178 MB on the ICON mesh,
                    # allocated and thrown away on EVERY merge event.
                    new_labels_uint = partition_nn_unstructured_optimised(
                        child_mask,
                        parent_masks_uint,
                        parent_centroids,
                        neighbours_int,
                        lat,
                        lon,
                        max_distance=max(max_distance, 20) * 2,
                    )
                    # Returned 'new_labels_uint' is just the index of the child_ids
                    new_labels = child_ids[new_labels_uint]

                    # Help garbage collection
                    new_labels_uint = None

                else:
                    # Use centroid-based partitioning
                    new_labels = partition_centroid_unstructured(child_mask, parent_centroids, child_ids, lat, lon)

                # Update slice data for subsequent merging in process_chunk
                data_t[child_mask] = new_labels

                # Record which cells get which new IDs for later updates
                spatial_indices_all = child_cells
                child_mask = None  # Free memory
                # No gc.collect() here. CPython frees these arrays by refcount the moment
                # the last name is rebound; a full collection only breaks reference
                # CYCLES, of which this loop creates none. It ran once per merge event and
                # walks the WHOLE process heap each time, so its cost grows with the
                # worker's live object count rather than with anything this loop does.

                # Record update information for each new ID
                for new_id in child_ids[1:]:
                    free_slots = np.where(updates_ids[t] == -1)[0].astype(np.int32)
                    if free_slots.size == 0:  # pragma: no cover
                        # `updates_array` is uint8 with 255 as its "no update" sentinel, so
                        # `updates_ids` has exactly 255 slots per timestep and every new id
                        # minted at this timestep consumes one. Each merge mints
                        # `n_parents - 1`, giving the joint invariant
                        #     MAX_MERGES * (MAX_PARENTS - 1) <= 255
                        # which the two constants must be raised together under. Without
                        # this branch the loop indexes an empty array and dies on a bare
                        # IndexError hours into a run.
                        raise TrackingError(
                            "Exhausted the per-timestep update-id space",
                            details=(
                                f"Timestep {t} minted more than {updates_ids.shape[1]} new IDs; "
                                f"MAX_MERGES={MAX_MERGES} x (MAX_PARENTS-1)={MAX_PARENTS - 1} "
                                f"= {MAX_MERGES * (MAX_PARENTS - 1)}"
                            ),
                            suggestions=[
                                "Lower MAX_MERGES or MAX_PARENTS so their product stays within the slot count",
                                "Widen updates_array to uint16 and updates_ids to match (costs a whole extra "
                                "(time, ncells) byte-field per iteration)",
                            ],
                            context={
                                "timestep": t,
                                "slots": int(updates_ids.shape[1]),
                                "max_merges": MAX_MERGES,
                                "max_parents": MAX_PARENTS,
                            },
                        )
                    update_idx = free_slots[0]  # Next free index in updates_ids
                    updates_ids[t, update_idx] = new_id
                    updates_array[t, spatial_indices_all[new_labels == new_id]] = update_idx

                next_new_id += n_parents - 1

                # Find all child objects in the next timestep that overlap with our newly labeled regions
                new_merging_list = []
                for new_id in child_ids:
                    # Every cell holding new_id is a cell of the child just partitioned:
                    # the new ids are freshly minted, and all cells that held child_id were
                    # in child_mask by construction. So this is the same set `data_t ==
                    # new_id` would find, ascending, without touching the whole field.
                    parent_cells = spatial_indices_all[new_labels == new_id]
                    if parent_cells.size:
                        area_0 = area[parent_cells].sum()
                        potential_children = np.unique(data_p1[parent_cells])

                        for potential_child in potential_children[potential_children > 0]:
                            potential_child_mask = data_p1 == potential_child
                            area_1 = area[potential_child_mask].sum()
                            min_area = min(area_0, area_1)
                            # Ascending, and exactly the cells the whole-mesh
                            # `parent_mask & potential_child_mask` would have selected.
                            overlap_cells = parent_cells[potential_child_mask[parent_cells]]
                            overlap_area = area[overlap_cells].sum()

                            if overlap_area / min_area > overlap_threshold:
                                new_merging_list.append(potential_child)

                # Add newly found merging objects to processing queue
                if t < n_time - 1:
                    # Add to next timestep in this chunk
                    for new_object_id in new_merging_list:
                        if new_object_id not in merging_objects_list[t + 1]:
                            merging_objects_list[t + 1].append(new_object_id)
                else:
                    # Record for next chunk
                    for new_object_id in new_merging_list:
                        # Dedup first: an already-queued object needs no new slot, so it must
                        # not trip the capacity guard.
                        if np.any(final_merging_objects[t][:final_merge_count] == new_object_id):
                            continue
                        if final_merge_count >= MAX_MERGES:  # pragma: no cover
                            raise TrackingError(
                                "Excessive merge operations detected",
                                details=f"Final merge count {final_merge_count + 1} exceeds limit {MAX_MERGES} at timestep {t}",
                                suggestions=[
                                    "Increase area_filter_quartile to reduce small objects",
                                    "Consider adjusting tracking parameters",
                                ],
                                context={
                                    "timestep": t,
                                    "final_merge_count": final_merge_count,
                                    "limit": MAX_MERGES,
                                },
                            )
                        final_merging_objects[t][final_merge_count] = new_object_id
                        final_merge_count += 1

        return (
            merge_child_ids,
            merge_parent_ids,
            merge_areas,
            merge_counts,
            has_merge,
            updates_array,
            updates_ids,
            final_merging_objects,
        )

    def update_object_id_field_inplace(
        object_id_field: xr.DataArray,
        id_lookup: Dict[int, int],
        updates_array: xr.DataArray,
        updates_ids: xr.DataArray,
        has_merge: xr.DataArray,
    ) -> xr.DataArray:  # pragma: no cover
        """
        Update the object field with chunk results using xarray operations.

        This is memory efficient as it avoids creating full copies of the object_id_field.

        Parameters
        ----------
        object_id_field : xarray.DataArray
            The full object field to update
        id_lookup : dict
            Dictionary mapping temporary IDs to new IDs
        updates_array : xarray.DataArray
            Array indicating which spatial indices to update
        updates_ids : xarray.DataArray
            The new IDs to assign to updated indices
        has_merge : xarray.DataArray
            Boolean indicating whether each timestep has merges

        Returns
        -------
        xarray.DataArray
            Updated object field
        """
        # Quick return if no merges to update
        if not has_merge.any():
            return object_id_field

        def update_timeslice(
            data: NDArray[np.int32],
            updates: NDArray[np.uint8],
            update_ids: NDArray[np.int32],
            lookup_values: NDArray[np.int32],
        ) -> NDArray[np.int32]:
            """Process a single timeslice."""
            # Extract valid update IDs
            valid_ids = update_ids[update_ids > -1]
            if len(valid_ids) == 0:
                return data

            # Create result array starting with original values
            result = data.copy()

            # Apply each update
            for idx, update_id in enumerate(valid_ids):
                mask = updates == idx
                if mask.any():
                    result = np.where(mask, lookup_values[update_id], result)

            return result

        # Convert lookup dict to array for vectorized access
        max_id = max(id_lookup.keys()) + 1
        lookup_array = np.full(max_id, -1, dtype=np.int32)
        for temp_id, new_id in id_lookup.items():
            lookup_array[temp_id] = new_id

        # Apply updates in parallel
        result = xr.apply_ufunc(
            update_timeslice,
            object_id_field,
            updates_array,
            updates_ids,
            kwargs={"lookup_values": lookup_array},
            input_core_dims=[[xdim], [xdim], ["update_idx"]],
            output_core_dims=[[xdim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int32],
        )

        return result

    def update_object_id_field_zarr(
        object_id_field: xr.DataArray,
        id_lookup: Dict[int, int],
        updates_array: xr.DataArray,
        updates_ids: xr.DataArray,
        has_merge: xr.DataArray,
    ) -> xr.DataArray:
        """
        Update object field using a temporary zarr store for better memory efficiency.

        This approach minimises memory usage by writing changes directly to disk,
        allowing for more efficient parallel processing of large datasets.

        Parameters
        ----------
        object_id_field : xarray.DataArray
            The object field to update
        id_lookup : dict
            Dictionary mapping temporary IDs to new IDs
        updates_array : xarray.DataArray
            Array indicating which spatial indices to update
        updates_ids : xarray.DataArray
            The new IDs to assign to updated indices
        has_merge : xarray.DataArray
            Boolean indicating whether each timestep has merges

        Returns
        -------
        xarray.DataArray
            Updated object field from zarr store
        """
        # Early return if no merges to save memory
        if not bool(has_merge.any().compute().item()):
            return object_id_field

        zarr_path = temp_field_path

        # Initialise zarr store if needed
        if not os.path.exists(zarr_path):
            object_id_field.name = "temp"
            object_id_field.to_zarr(zarr_path, mode="w")

        def write_receipt(ds_chunk: xr.Dataset) -> xr.DataArray:
            """One byte per timestep, standing in for the chunk this pass just wrote.

            ``update_time_chunk`` is SIDE-EFFECTING: each chunk writes its own zarr region
            and the returned array is thrown away (``del result`` below). Returning the
            field itself made ``result.persist()`` materialise a whole int32 field purely
            to force those writes -- measured at **19.1 GB, 54.8 % of everything this path
            pins**, on the n_time=32 ICON slice. A receipt forces the writes identically,
            because each output block still depends on its own input block, for n_time
            bytes instead of n_time x ncells x 4.

            Must be returned from EVERY branch, including the no-merge early return, or
            ``map_blocks`` raises on a template mismatch for exactly the chunks that skip
            the write.
            """
            return xr.DataArray(
                np.zeros(ds_chunk.sizes[timedim], dtype=np.int8),
                dims=[timedim],
                coords={timecoord: ds_chunk[timecoord].values},
            )

        def update_time_chunk(ds_chunk: xr.Dataset, lookup_dict: Dict[int, int]) -> xr.DataArray:
            """Process a single chunk with optimised memory usage."""
            # Skip processing if no merges in this chunk
            needs_update = bool(ds_chunk["has_merge"].any().compute().item())
            if not needs_update:
                return write_receipt(ds_chunk)

            # Extract data from the chunk
            chunk_data = ds_chunk["object_field"]
            chunk_updates = ds_chunk["updates"]
            chunk_update_ids = ds_chunk["update_ids"]

            # Get zarr region indices
            time_idx_start = int(ds_chunk["time_indices"].values[0])
            time_idx_end = int(ds_chunk["time_indices"].values[-1]) + 1

            updated_chunk = chunk_data.copy()

            # Process each time slice in the chunk
            for t in range(chunk_data.sizes[timedim]):
                # Get update information for this time
                updates_slice = chunk_updates.isel({timedim: t}).values
                update_ids_slice = chunk_update_ids.isel({timedim: t}).values

                # Get valid update IDs
                valid_mask = update_ids_slice > -1
                if not np.any(valid_mask):
                    continue

                valid_ids = update_ids_slice[valid_mask]

                # Get the time slice data and apply updates
                result_slice = updated_chunk.isel({timedim: t})

                for idx, update_id in enumerate(valid_ids):
                    mask = updates_slice == idx
                    if np.any(mask):
                        new_id = lookup_dict.get(int(update_id), update_id)
                        result_slice = xr.where(mask, new_id, result_slice)

                # Store updated slice
                updated_chunk[t] = result_slice

            # Write the updated chunk directly to zarr
            updated_chunk.name = "temp"
            updated_chunk.to_zarr(
                zarr_path,
                region={timedim: slice(time_idx_start, time_idx_end)},
            )

            return write_receipt(ds_chunk)  # Not the field: see write_receipt's docstring

        # Create time indices for slicing
        time_coords = object_id_field[timecoord].values
        time_indices = np.arange(len(time_coords), dtype=np.int32)
        time_index_da = xr.DataArray(time_indices, dims=[timedim], coords={timecoord: time_coords})

        # Create dataset with all necessary components
        ds = xr.Dataset(
            {
                "object_field": object_id_field,
                "updates": updates_array,
                "update_ids": updates_ids,
                "time_indices": time_index_da,
                "has_merge": has_merge,
            }
        ).chunk({timedim: timechunks})

        # Process chunks in parallel. The template is the RECEIPT shape, not the field --
        # this pass exists for its zarr writes, not its return value.
        receipt_template = xr.DataArray(
            np.zeros(len(time_coords), dtype=np.int8),
            dims=[timedim],
            coords={timecoord: time_coords},
        ).chunk({timedim: timechunks})

        result = xr.map_blocks(
            update_time_chunk,
            ds,
            kwargs={"lookup_dict": id_lookup},
            template=receipt_template,
        )

        # Force computation to ensure all writes complete
        result = result.persist()
        wait(result)

        # Release resources
        del result, ds, object_id_field
        gc.collect()

        # Load the updated data from zarr store
        object_id_field_new = xr.open_zarr(zarr_path, chunks={timedim: timechunks}).temp

        return object_id_field_new

    def merge_objects_parallel_iteration(
        object_id_field_unique: xr.DataArray,
        merging_objects: Set[int],
        global_id_counter: int,
        iteration_index: int = 0,
    ) -> Tuple[
        xr.DataArray,  # updated_field
        Tuple[
            NDArray[np.int32],
            NDArray[np.int32],
            NDArray[np.float32],
            NDArray[np.int32],
        ],  # merge_data
        Set[int],  # new_merging_objects
        int,  # updated_counter
    ]:
        """
        Perform a single iteration of the parallel merging process.

        This function handles one complete batch of merging objects across all
        timesteps, updating object IDs and tracking merge events.

        Parameters
        ----------
        object_id_field_unique : xarray.DataArray
            Field of unique object IDs
        merging_objects : set
            Set of object IDs to process in this iteration
        global_id_counter : int
            Current counter for assigning new global IDs

        Returns
        -------
        tuple
            (updated_field, merge_data, new_merging_objects, updated_counter)
        """
        n_time = len(object_id_field_unique[timecoord])

        # Pre-allocate arrays for this iteration
        child_ids_iter = np.full((n_time, MAX_MERGES, MAX_CHILDREN), -1, dtype=np.int32)  # List of child ID arrays for this time
        parent_ids_iter = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.int32)  # List of parent ID arrays for this time
        merge_areas_iter = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.float32)  # List of areas for this time
        merge_counts_iter = np.zeros(n_time, dtype=np.int32)

        # Prepare neighbour information
        neighbours_int_local = neighbours_int.chunk({xdim: -1, "nv": -1})

        logger.info(f"Processing Parallel Iteration {iteration + 1} with {len(merging_objects)} Merging Objects...")

        # Pre-compute the child_time_idx for merging_objects
        time_index_map = _overlap.compute_id_time_dict(
            object_id_field_unique,
            list(merging_objects),
            global_id_counter,
            timedim,
            unstructured_grid,
            ydim,
            xdim,
            # Only the merging IDs are ever looked up below, so restrict the search
            # instead of broadcasting a (time x buffer x max_objects) boolean over every
            # possible ID -- multi-GB chunks for a map of a few hundred entries
            # (review finding 6.8).
            all_objects=False,
        )
        logger.debug("Finished Mapping Children to Time Indices")

        # Bucket the merging objects by time index in a single pass. The previous form
        # rescanned the whole merging set once per timestep, twice over (finding 5.12).
        objects_by_time: List[List[int]] = [[] for _ in range(n_time)]
        for merging_object in merging_objects:
            t_idx = time_index_map.get(merging_object, -1)
            if 0 <= t_idx < n_time:
                objects_by_time[t_idx].append(merging_object)

        # Create uniform array of merging objects for each timestep
        max_merges = max(len(objects_at_t) for objects_at_t in objects_by_time)
        uniform_merging_objects_array = np.zeros((n_time, max_merges), dtype=np.int32)
        for t, objects_at_t in enumerate(objects_by_time):
            if objects_at_t:  # Only fill if there are objects at this time
                uniform_merging_objects_array[t, : len(objects_at_t)] = np.array(objects_at_t, dtype=np.int32)

        # Create DataArrays for parallel processing
        merging_objects_da = xr.DataArray(
            uniform_merging_objects_array,
            dims=[timedim, "merges"],
            coords={timecoord: object_id_field_unique[timecoord]},
        )

        # Calculate ID offsets for each timestep to ensure unique IDs. Stride by the hard
        # worst case (MAX_MERGES merges each spawning up to MAX_PARENTS-1 new IDs) rather
        # than the data-dependent `max_merges * timechunks` (whose initial queue can be as
        # small as 1). A too-small stride lets one timestep's cascade merges overrun into
        # the next timestep's ID range, silently fusing two distinct events under one ID.
        id_stride = MAX_MERGES * (MAX_PARENTS - 1)
        next_id_offsets = np.arange(n_time, dtype=np.int64) * id_stride + global_id_counter
        next_id_offsets_da = xr.DataArray(
            next_id_offsets,
            dims=[timedim],
            coords={timecoord: object_id_field_unique[timecoord]},
        )

        # Create shifted arrays for time connectivity
        object_id_field_unique_p1 = object_id_field_unique.shift({timedim: -1}, fill_value=0)
        object_id_field_unique_m1 = object_id_field_unique.shift({timedim: 1}, fill_value=0)

        # Align chunks for better parallel processing
        object_id_field_unique_m1 = object_id_field_unique_m1.chunk({timedim: timechunks})
        object_id_field_unique_p1 = object_id_field_unique_p1.chunk({timedim: timechunks})
        merging_objects_da = merging_objects_da.chunk({timedim: timechunks})
        next_id_offsets_da = next_id_offsets_da.chunk({timedim: timechunks})

        # Process chunks in parallel
        results = xr.apply_ufunc(
            process_chunk,
            object_id_field_unique_m1,
            object_id_field_unique_p1,
            merging_objects_da,
            next_id_offsets_da,
            lat,
            lon,
            cell_area,
            neighbours_int_local,
            input_core_dims=[
                [xdim],
                [xdim],
                ["merges"],
                [],
                [xdim],
                [xdim],
                [xdim],
                ["nv", xdim],
            ],
            output_core_dims=[
                ["merge", "parent"],
                ["merge", "parent"],
                ["merge", "parent"],
                [],
                [],
                [xdim],
                ["update_idx"],
                ["merge"],
            ],
            output_dtypes=[
                np.int32,
                np.int32,
                np.float32,
                np.int16,
                np.bool_,
                np.uint8,
                np.int32,
                np.int32,
            ],
            dask_gufunc_kwargs={
                "output_sizes": {
                    "merge": MAX_MERGES,
                    "parent": MAX_PARENTS,
                    "update_idx": 255,
                }
            },
            vectorize=False,
            dask="parallelized",
        )

        # Unpack and persist results
        (
            merge_child_ids,
            merge_parent_ids,
            merge_areas,
            merge_counts,
            has_merge,
            updates_array,
            updates_ids,
            final_merging_objects,
        ) = results

        # This persist is LOAD-BEARING FOR CORRECTNESS, not just memory, and must stay
        # unconditional in every mode. These arrays are lazy expressions over
        # `object_id_field_unique`, and `update_object_id_field_zarr` below REWRITES the
        # zarr store that field reads from. Leaving them lazy means they are recomputed
        # after that rewrite, against updated IDs, and the merge ledgers silently change.
        # Routing them through `Materialiser.pin` (a no-op outside persist mode) reproduced
        # exactly that: streaming found 10 events where persist found 11 on the unstructured
        # fixture. They are small -- per-timestep ledgers, a few MB -- so pinning them in
        # every mode costs nothing.
        #
        # `updates_array` is a whole (time, ncells) uint8 field -- 2.382 GB of the 34.8 GB
        # this path pinned at n_time=32 -- so it is ANCHORED separately below, to disk under
        # streaming. Staging it is safe for the same reason the persist is needed: `stage`
        # WRITES it immediately, so it is materialised before the store rewrite, not left
        # lazy over it.
        #
        # It must nonetheless be named in THIS persist call, even though the very next line
        # anchors it again. All eight arrays are `getitem`s on ONE shared blockwise task --
        # the `process_chunk` call. Materialising seven of them lets the scheduler release
        # that shared task, so anchoring the eighth afterwards RE-RUNS `process_chunk` over
        # every time chunk: the whole merge kernel, including the BFS partitioner that is
        # 93 % of this stage's CPU, executed a second time per iteration. Measured, exactly
        # 2x and not a scheduling race: the instrumented ICON runs logged 80 invocations at
        # n_time=32 (5 iterations x 8 time chunks = 40) and 160 at n_time=64 (5 x 16 = 80).
        # Naming it here computes the shared task once and hands `stage` a materialised
        # array to write, which costs one transient whole-field pin (uint8, ~1 GB/worker
        # spread over 16 workers at n_time=1096) and saves an entire second execution.
        #
        # Anchoring inside this call instead is NOT an alternative: `to_zarr(compute=False)`
        # re-optimises its source graph and renames the shared keys, so a deferred write
        # submitted alongside the seven does not share them and still costs 2x. Verified
        # against a counting kernel: persist-seven-then-write 20 calls for 10 chunks,
        # deferred-write-in-one-submission 20, persist-all-then-write 10.
        #
        # The label MUST carry the iteration index. This runs once per merge-loop iteration
        # with a different array each time, and staging writes <label>.zarr with mode="w",
        # so a fixed label would rewrite the store the previous iteration's array is still
        # reading. Materialiser._reject_relabel turns that mistake into an immediate error.
        (
            merge_child_ids,
            merge_parent_ids,
            merge_areas,
            merge_counts,
            has_merge,
            updates_array,
            updates_ids,
            final_merging_objects,
        ) = persist(
            merge_child_ids,
            merge_parent_ids,
            merge_areas,
            merge_counts,
            has_merge,
            updates_array,
            updates_ids,
            final_merging_objects,
        )
        updates_array = _anchor_field(updates_array, f"updates_array_iter{iteration_index}", materialiser)

        # Get time indices where merges occurred
        has_merge = has_merge.compute()
        time_indices = np.where(has_merge)[0].astype(np.int32)

        # Clean up temporary arrays to save memory
        del (
            object_id_field_unique_p1,
            object_id_field_unique_m1,
            merging_objects_da,
            next_id_offsets_da,
        )
        gc.collect()

        logger.debug("Finished Batch Processing Step")

        # ====== Global Consolidation of Data ======

        # 1. Collect all temporary IDs and create global mapping
        all_temp_ids = np.unique(merge_child_ids.where(merge_child_ids >= global_id_counter, other=0).compute().values)
        all_temp_ids = all_temp_ids[all_temp_ids > 0]  # Remove the 0

        if not len(all_temp_ids):  # If no temporary IDs exist
            id_lookup = {}
        else:
            # Create mapping from temporary to permanent IDs
            id_lookup = {
                temp_id: np.int32(new_id)
                for temp_id, new_id in zip(
                    all_temp_ids,
                    range(global_id_counter, global_id_counter + len(all_temp_ids)),
                )
            }
            global_id_counter += len(all_temp_ids)

        logger.debug("Finished Consolidation Step 1: Temporary ID Mapping")

        # 2. Update object ID field with new IDs
        update_on_disk = True  # This is more memory efficient because it refreshes the dask graph every iteration

        if update_on_disk:
            object_id_field_unique = update_object_id_field_zarr(
                object_id_field_unique,
                id_lookup,
                updates_array,
                updates_ids,
                has_merge,
            )
        else:  # pragma: no cover
            object_id_field_unique = update_object_id_field_inplace(
                object_id_field_unique,
                id_lookup,
                updates_array,
                updates_ids,
                has_merge,
            )
            object_id_field_unique = object_id_field_unique.chunk({timedim: timechunks})  # Rechunk to avoid accumulating chunks...

        # Clean up arrays no longer needed
        del updates_array, updates_ids
        gc.collect()

        logger.debug("Finished Consolidation Step 2: Data Field Update")

        # 3. Update merge events
        new_merging_objects = set()
        merge_counts = merge_counts.compute()
        # Materialise the three small persisted ledgers once. Indexing them lazily inside
        # the loop below cost a blocking scheduler round-trip per merge event -- thousands
        # of them per merge-loop iteration for arrays of a few MB (review finding 5.11).
        merge_child_ids_local = merge_child_ids.compute()
        merge_parent_ids_local = merge_parent_ids.compute()
        merge_areas_local = merge_areas.compute()

        for t in time_indices:
            count = int(merge_counts.isel({timedim: t}).item())
            if count > 0:
                merge_counts_iter[t] = count

                # Extract valid IDs and areas for each merge event
                for merge_idx in range(count):
                    # Get child IDs
                    child_ids = merge_child_ids_local.isel({timedim: t, "merge": merge_idx}).values
                    child_ids = child_ids[child_ids >= 0]

                    # Get parent IDs and areas
                    parent_ids = merge_parent_ids_local.isel({timedim: t, "merge": merge_idx}).values
                    areas = merge_areas_local.isel({timedim: t, "merge": merge_idx}).values
                    valid_mask = parent_ids >= 0
                    parent_ids = parent_ids[valid_mask]
                    areas = areas[valid_mask]

                    # Map temporary IDs to permanent IDs
                    mapped_child_ids = [id_lookup.get(int(id_.item()), int(id_.item())) for id_ in child_ids]
                    mapped_parent_ids = [id_lookup.get(int(id_.item()), int(id_.item())) for id_ in parent_ids]

                    # Store in pre-allocated arrays
                    child_ids_iter[t, merge_idx, : len(mapped_child_ids)] = mapped_child_ids
                    parent_ids_iter[t, merge_idx, : len(mapped_parent_ids)] = mapped_parent_ids
                    merge_areas_iter[t, merge_idx, : len(areas)] = areas

        # Process final merging objects for next iteration
        final_merging_objects = final_merging_objects.compute().values
        final_merging_objects = final_merging_objects[final_merging_objects > 0]
        mapped_final_objects = [id_lookup.get(id_, id_) for id_ in final_merging_objects]
        new_merging_objects.update(mapped_final_objects)

        logger.debug("Finished Consolidation Step 3: Merge List Dictionary Consolidation")

        # Clean up memory
        del merge_child_ids, merge_parent_ids, merge_areas, merge_counts, has_merge
        gc.collect()

        return (
            object_id_field_unique,
            (child_ids_iter, parent_ids_iter, merge_areas_iter, merge_counts_iter),
            new_merging_objects,
            global_id_counter,
        )

    # ============================
    # Main Loop for Parallel Merging
    # ============================

    # Find overlapping objects
    overlap_objects_list = _overlap.find_overlapping_objects(
        object_id_field_unique, timedim, unstructured_grid, ydim, xdim, cell_area
    )  # List object pairs that overlap by at least overlap_threshold percent
    # enforce_overlap_threshold consumes an ObjectPropsStore; this (unstructured) path keeps
    # object_props as a Dataset and recomputes it wholesale, so wrap it for the two enforce calls.
    overlap_objects_list = _overlap.enforce_overlap_threshold(
        overlap_objects_list, _objects.ObjectPropsStore.from_dataset(object_props), unstructured_grid, overlap_threshold
    )
    logger.info("Finished finding overlapping objects")

    # Find initial merging objects
    unique_children, children_counts = np.unique(overlap_objects_list[:, 1], return_counts=True)
    merging_objects = set(unique_children[children_counts > 1].astype(np.int32))
    del overlap_objects_list

    # Process chunks iteratively until no new merging objects remain

    iteration = 0
    processed_chunks = set()
    global_id_counter = int(object_props.ID.max().item()) + 1

    # Initialise global merge event tracking
    global_child_ids = []
    global_parent_ids = []
    global_merge_areas = []
    global_merge_tidx = []

    while merging_objects and iteration < max_iteration:
        (
            object_id_field_new,
            merge_data_iter,
            new_merging_objects,
            global_id_counter,
        ) = merge_objects_parallel_iteration(object_id_field_unique, merging_objects, global_id_counter, iteration)
        child_ids_iter, parent_ids_iter, merge_areas_iter, merge_counts_iter = merge_data_iter

        # Consolidate merge events from this iteration
        for t in range(len(merge_counts_iter)):
            count = merge_counts_iter[t]
            if count > 0:
                for merge_idx in range(count):
                    # Extract valid children
                    children = child_ids_iter[t, merge_idx]
                    children = children[children >= 0]

                    # Extract valid parents and areas
                    parents = parent_ids_iter[t, merge_idx]
                    areas = merge_areas_iter[t, merge_idx]
                    valid_mask = parents >= 0
                    parents = parents[valid_mask]
                    areas = areas[valid_mask]

                    # Record valid merge events
                    if len(children) > 0 and len(parents) > 0:
                        global_child_ids.append(children)
                        global_parent_ids.append(parents)
                        global_merge_areas.append(areas)
                        global_merge_tidx.append(t)

        # Prepare for next iteration - only process objects not already handled
        merging_objects = new_merging_objects - processed_chunks
        processed_chunks.update(new_merging_objects)
        iteration += 1

        # Update the object field
        object_id_field_unique = object_id_field_new
        del object_id_field_new

    # Check if we reached maximum iterations
    if iteration == max_iteration:  # pragma: no cover
        raise TrackingError(
            "Maximum iterations reached in tracking algorithm",
            details=f"Algorithm failed to converge after {max_iteration} iterations",
            suggestions=[
                "Increase max_iteration parameter",
                "Increase area_filter_quartile to reduce small objects",
                "Consider adjusting tracking parameters",
            ],
            context={
                "max_iteration": max_iteration,
                "reached_iteration": iteration,
            },
        )

    # Process the collected merge events

    times = object_id_field_unique[timecoord].values

    # Find maximum dimensions for arrays
    # Handle case where there are no merge events
    if global_parent_ids and global_child_ids:
        max_parents = max(len(ids) for ids in global_parent_ids)
        max_children = max(len(ids) for ids in global_child_ids)
    else:
        max_parents = 1  # Default minimum size
        max_children = 1

    # Create padded arrays for merge events
    parent_ids_array = np.full((len(global_parent_ids), max_parents), -1, dtype=np.int32)
    child_ids_array = np.full((len(global_child_ids), max_children), -1, dtype=np.int32)
    overlap_areas_array = np.full(
        (len(global_merge_areas), max_parents),
        -1,
        dtype=np.float32 if unstructured_grid else np.int32,
    )

    # Fill arrays with merge data
    for i, parents in enumerate(global_parent_ids):
        parent_ids_array[i, : len(parents)] = parents

    for i, children in enumerate(global_child_ids):
        child_ids_array[i, : len(children)] = children

    for i, areas in enumerate(global_merge_areas):
        overlap_areas_array[i, : len(areas)] = areas

    # Create merge events dataset
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

    # Recompute object properties and overlaps after all merging. This field has TWO
    # consumers below -- calculate_object_properties and find_overlapping_objects -- so it
    # is an anchor, not a bounded intermediate: leaving it lazy would re-run the whole
    # merge loop's output graph once per consumer. `stage` in persist mode is
    # `obj.persist()`, and `optimize_graph=True` was already dask's default, so the default
    # path is byte-for-byte unchanged.
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

    # Recompute overlaps based on final object configuration
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
