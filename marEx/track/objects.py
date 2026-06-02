"""
MarEx Track: Object identification and property calculation.

Stateless helpers for labelling connected regions and computing per-object
properties (area, centroid), extracted from the tracker orchestrator. The
tracker config/grid values each method read from ``self`` are threaded in as
explicit arguments. Behaviour and numerics are identical to the original
``tracker`` methods.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from dask import persist
from dask_image.ndmeasure import label
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import regionprops_table

from ..exceptions import ConfigurationError
from ..logging_config import get_logger

logger = get_logger(__name__)


def identify_objects(
    data_bin: xr.DataArray,
    time_connectivity: bool,
    unstructured_grid: bool,
    mask: xr.DataArray,
    neighbours_int: Optional[xr.DataArray],
    xdim: str,
    regional_mode: bool,
) -> Tuple[xr.DataArray, None, int]:
    """
    Identify connected regions in binary data.

    Parameters
    ----------
    data_bin : xarray.DataArray
        Binary data to identify objects in
    time_connectivity : bool
        Whether to connect objects across time

    Returns
    -------
    object_id_field : xarray.DataArray
        Field of integer IDs for each object
    None : NoneType
        Placeholder for compatibility with track_objects
    N_objects : int
        Number of objects identified
    """
    if unstructured_grid:
        # The resulting ID field for unstructured grid will start at 0 for each time-slice,
        # which differs from structured grid where IDs are unique across time.

        if time_connectivity:  # pragma: no cover
            raise ConfigurationError(
                "Time connectivity not supported for unstructured grids",
                details="Automatic time connectivity computation requires regular grids",
                suggestions=[
                    "Set time_connectivity=False for unstructured data",
                    "Manually specify connectivity if needed",
                ],
            )

        # Use Union-Find (Disjoint Set Union) clustering for unstructured grid
        def cluster_true_values(arr: NDArray[np.bool_], neighbours_int: NDArray[np.int32]) -> NDArray[np.int32]:
            """Cluster connected True values in binary data on unstructured grid."""
            t, n = arr.shape
            labels = np.full((t, n), -1, dtype=np.int32)

            for i in range(t):
                # Get indices of True values
                true_indices = np.where(arr[i])[0].astype(np.int32)
                mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(true_indices)}

                # Find connected components
                valid_mask = (neighbours_int != -1) & arr[i][neighbours_int]
                row_ind, col_ind = np.where(valid_mask)
                row_ind = row_ind.astype(np.int32)
                col_ind = col_ind.astype(np.int32)

                # Map to compact indices for graph algorithm
                mapped_row_ind = []
                mapped_col_ind = []
                for r, c in zip(neighbours_int[row_ind, col_ind], col_ind):
                    if r in mapping and c in mapping:
                        mapped_row_ind.append(mapping[r])
                        mapped_col_ind.append(mapping[c])

                # Create graph and find connected components
                graph = csr_matrix(
                    (
                        np.ones(len(mapped_row_ind), dtype=np.int32),
                        (mapped_row_ind, mapped_col_ind),
                    ),
                    shape=(len(true_indices), len(true_indices)),
                )
                _, labels_true = connected_components(csgraph=graph, directed=False, return_labels=True)
                labels[i, true_indices] = labels_true

            return labels + 1  # Add 1 so 0 represents no object

        # Apply mask and cluster
        data_bin = data_bin.where(mask, other=False)

        object_id_field = xr.apply_ufunc(
            cluster_true_values,
            data_bin,
            neighbours_int,
            input_core_dims=[[xdim], ["nv", xdim]],
            output_core_dims=[[xdim]],
            output_dtypes=[np.int32],
            dask_gufunc_kwargs={
                "output_sizes": {xdim: data_bin.sizes[xdim]},
            },
            vectorize=False,
            dask="parallelized",
        )

        # Ensure ID = 0 on invalid regions
        object_id_field = object_id_field.where(mask, other=0)
        object_id_field = object_id_field.persist()
        object_id_field = object_id_field.rename("ID_field")
        N_objects = 1  # Placeholder (IDs aren't unique across time)

    else:  # Structured Grid
        # Create connectivity kernel for labeling
        neighbours = np.zeros((3, 3, 3))

        if time_connectivity:
            # ID objects in 3D (i.e. space & time) -- N.B. IDs are unique across time
            neighbours[:, :, :] = 1  # +-1 in time, _and also diagonal in time_ -- i.e. edges can touch
        else:
            # ID objects only in 2D (i.e. space) -- N.B. IDs are _not_ unique across time (i.e. each time starts at 0 again)
            neighbours[1, :, :] = 1  # All 8 neighbours, but ignore time

        # Cluster & label binary data
        # Apply dask-powered ndimage & persist in memory
        if regional_mode:
            object_id_field, N_objects = label(
                data_bin,
                structure=neighbours,
            )
        else:
            object_id_field, N_objects = label(
                data_bin,
                structure=neighbours,
                wrap_axes=(2,),  # Wrap in x-direction !
            )
        results = persist(object_id_field, N_objects)
        object_id_field, N_objects = results

        N_objects = N_objects.compute()

        # Convert to DataArray with same coordinates as input
        object_id_field = (
            xr.DataArray(
                object_id_field,
                coords=data_bin.coords,
                dims=data_bin.dims,
                attrs=data_bin.attrs,
            )
            .rename("ID_field")
            .astype(np.int32)
        )

    return object_id_field, None, N_objects


def calculate_centroid(
    binary_mask: NDArray[np.bool_],
    regional_mode: bool,
    original_centroid: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    Calculate object centroid, handling edge cases for periodic boundaries.

    Parameters
    ----------
    binary_mask : numpy.ndarray
        2D binary array where True indicates the object (dimensions are (y,x))
    original_centroid : tuple, optional
        (y_centroid, x_centroid) from regionprops_table

    Returns
    -------
    tuple
        (y_centroid, x_centroid)
    """
    if regional_mode:  # pragma: no cover
        # We don't need to adjust centroids for periodic boundaries
        return original_centroid

    # Check if object is near either edge of x dimension
    near_left_BC = np.any(binary_mask[:, :100])
    near_right_BC = np.any(binary_mask[:, -100:])

    if original_centroid is None:  # pragma: no cover
        # Calculate y centroid from scratch
        y_indices = np.nonzero(binary_mask)[0]
        y_centroid = np.mean(y_indices)
    else:
        y_centroid = original_centroid[0]

    # If object is near both edges, recalculate x-centroid to handle wrapping
    # N.B.: We calculate _near_ rather than touching, to catch the edge case where the
    # object may be split and straddling the boundary !
    if near_left_BC and near_right_BC:
        # Adjust x coordinates that are near right edge
        x_indices = np.nonzero(binary_mask)[1]
        x_indices_adj = x_indices.copy()
        right_side = x_indices > binary_mask.shape[1] // 2
        x_indices_adj[right_side] -= binary_mask.shape[1]

        x_centroid = np.mean(x_indices_adj)
        if x_centroid < 0:  # Ensure centroid is positive
            x_centroid += binary_mask.shape[1]

    elif original_centroid is None:  # pragma: no cover
        # Calculate x-centroid from scratch
        x_indices = np.nonzero(binary_mask)[1]
        x_centroid = np.mean(x_indices)

    else:
        x_centroid = original_centroid[1]

    return (y_centroid, x_centroid)


def calculate_object_properties(
    object_id_field: xr.DataArray,
    unstructured_grid: bool,
    lat: xr.DataArray,
    lon: xr.DataArray,
    cell_area: xr.DataArray,
    timedim: str,
    regional_mode: bool,
    ydim: Optional[str],
    xdim: str,
    properties: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Calculate properties of objects from ID field.

    Parameters
    ----------
    object_id_field : xarray.DataArray
        Field containing object IDs
    properties : list, optional
        List of properties to calculate (defaults to ['label', 'area'])

    Returns
    -------
    object_props : xarray.Dataset
        Dataset containing calculated properties with 'ID' dimension
    """
    # Set default properties
    if properties is None:
        properties = ["label", "area"]

    # Ensure 'label' is included
    if "label" not in properties:
        properties = ["label"] + properties  # 'label' is actually 'ID' within regionprops

    check_centroids = "centroid" in properties

    if unstructured_grid:
        # Compute properties on unstructured grid

        # Convert lat/lon to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Broadcast coordinate arrays to match object_id_field shape for vectorisation
        lat_rad_broadcast, _ = xr.broadcast(lat_rad, object_id_field)
        lon_rad_broadcast, _ = xr.broadcast(lon_rad, object_id_field)
        cell_area_broadcast, _ = xr.broadcast(cell_area, object_id_field)

        # Calculate buffer size for IDs in chunks
        max_ID = int(object_id_field.max().compute().item()) + 1

        # Handle case where object_id_field may not have time dimension (e.g., single time slice)
        if timedim in object_id_field.dims:
            time_steps = object_id_field.sizes[timedim]
        else:
            # For single time slice, use 1 as time steps
            time_steps = 1

        ID_buffer_size = max(int(max_ID / time_steps) * 4 + 2, max_ID)

        def object_properties_chunk(
            ids: NDArray[np.int32],
            lat: NDArray[np.float32],
            lon: NDArray[np.float32],
            area: NDArray[np.float32],
            buffer_IDs: bool = True,
        ) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
            """
            Calculate object properties for a chunk of data.
            Uses vectorised operations for efficiency.
            """
            # Find valid IDs
            valid_mask = ids > 0
            ids_chunk = np.unique(ids[valid_mask])
            n_ids = len(ids_chunk)

            if n_ids == 0:
                # No objects in this chunk
                if buffer_IDs:
                    result = np.zeros((3, ID_buffer_size), dtype=np.float32)
                    padded_ids = np.zeros(ID_buffer_size, dtype=np.int32)
                    return result, padded_ids
                else:  # pragma: no cover
                    result = np.zeros((3, 0), dtype=np.float32)
                    padded_ids = np.array([], dtype=np.int32)
                    return result, padded_ids

            # Map IDs to consecutive indices
            mapped_indices = np.searchsorted(ids_chunk, ids[valid_mask]).astype(np.int32)

            # Pre-allocate arrays
            areas = np.zeros(n_ids, dtype=np.float32)
            weighted_x = np.zeros(n_ids, dtype=np.float32)
            weighted_y = np.zeros(n_ids, dtype=np.float32)
            weighted_z = np.zeros(n_ids, dtype=np.float32)

            # Convert to Cartesian for centroid calculation
            cos_lat = np.cos(lat[valid_mask])
            x = cos_lat * np.cos(lon[valid_mask])
            y = cos_lat * np.sin(lon[valid_mask])
            z = np.sin(lat[valid_mask])

            # Compute areas
            valid_areas = area[valid_mask]
            np.add.at(areas, mapped_indices, valid_areas)

            # Compute weighted coordinates
            np.add.at(weighted_x, mapped_indices, valid_areas * x)
            np.add.at(weighted_y, mapped_indices, valid_areas * y)
            np.add.at(weighted_z, mapped_indices, valid_areas * z)

            # Clean intermediate arrays
            del x, y, z, cos_lat, valid_areas

            # Normalise vectors
            norm = np.sqrt(weighted_x**2 + weighted_y**2 + weighted_z**2)
            norm = np.where(norm > 0, norm, 1)  # Avoid division by zero

            weighted_x /= norm
            weighted_y /= norm
            weighted_z /= norm

            # Convert back to lat/lon
            centroid_lat = np.degrees(np.arcsin(np.clip(weighted_z, -1, 1)))
            centroid_lon = np.degrees(np.arctan2(weighted_y, weighted_x))

            # Fix longitude range to [-180, 180]
            centroid_lon = np.where(
                centroid_lon > 180.0,
                centroid_lon - 360.0,
                np.where(centroid_lon < -180.0, centroid_lon + 360.0, centroid_lon),
            )

            assert areas.shape == (n_ids,)
            assert centroid_lat.shape == (n_ids,)
            assert centroid_lon.shape == (n_ids,)

            if buffer_IDs:
                # Create padded output arrays
                result = np.zeros((3, ID_buffer_size), dtype=np.float32)
                padded_ids = np.zeros(ID_buffer_size, dtype=np.int32)

                # Fill arrays up to n_ids
                result[0, :n_ids] = areas
                result[1, :n_ids] = centroid_lat
                result[2, :n_ids] = centroid_lon
                padded_ids[:n_ids] = ids_chunk
            else:  # pragma: no cover
                result = np.vstack((areas, centroid_lat, centroid_lon))
                padded_ids = ids_chunk

            return result, padded_ids

        # Process single time or multiple times
        # If time dimension doesn't exist, treat as single time slice
        if timedim not in object_id_field.dims or object_id_field.sizes[timedim] == 1:  # pragma: no cover
            props_np, ids = object_properties_chunk(
                object_id_field.values,
                lat_rad_broadcast.values,
                lon_rad_broadcast.values,
                cell_area_broadcast.values,
                buffer_IDs=False,
            )
            props = xr.DataArray(props_np, dims=["prop", "out_id"])

        else:
            # Process in parallel
            props_buffer, ids_buffer = xr.apply_ufunc(
                object_properties_chunk,
                object_id_field,
                lat_rad_broadcast,
                lon_rad_broadcast,
                cell_area_broadcast,
                input_core_dims=[
                    [xdim],
                    [xdim],
                    [xdim],
                    [xdim],
                ],
                output_core_dims=[["prop", "out_id"], ["out_id"]],
                output_dtypes=[np.float32, np.int32],
                dask_gufunc_kwargs={"output_sizes": {"prop": 3, "out_id": ID_buffer_size}},
                vectorize=True,
                dask="parallelized",
            )
            results = persist(props_buffer, ids_buffer)
            props_buffer, ids_buffer = results
            ids_buffer = ids_buffer.compute().values.reshape(-1)

            # Get valid IDs (non-zero)
            valid_ids_mask = ids_buffer > 0

            # Check if we have any valid IDs before stacking
            if np.any(valid_ids_mask):
                ids = ids_buffer[valid_ids_mask]
                props = props_buffer.stack(combined=(timedim, "out_id")).isel(combined=valid_ids_mask)
            else:  # pragma: no cover
                # No valid IDs found
                ids = np.array([], dtype=np.int32)
                props = xr.DataArray(np.zeros((3, 0), dtype=np.float32), dims=["prop", "out_id"])

        # Create object properties dataset
        if len(ids) > 0:
            object_props = (
                xr.Dataset(
                    {
                        "area": ("out_id", props.isel(prop=0).data),
                        "centroid-0": ("out_id", props.isel(prop=1).data),
                        "centroid-1": ("out_id", props.isel(prop=2).data),
                    },
                    coords={"ID": ("out_id", ids)},
                )
                .set_index(out_id="ID")
                .rename({"out_id": "ID"})
            )
        else:  # pragma: no cover
            # Create empty dataset with correct structure
            object_props = xr.Dataset(
                {
                    "area": ("ID", []),
                    "centroid-0": ("ID", []),
                    "centroid-1": ("ID", []),
                },
                coords={"ID": []},
            )

    else:
        # Structured grid approach
        # N.B.: These operations are simply done on a pixel grid
        #       i.e. with no cartesian conversion
        #       (therefore, polar regions are doubly biased)

        # Define function to calculate properties for each chunk
        def object_properties_chunk(
            ids: NDArray[np.int32],
        ) -> Dict[str, List[Union[int, float]]]:
            """Calculate object properties for a chunk of data."""
            # Use regionprops_table for standard properties
            props_slice = regionprops_table(ids, properties=properties)

            # Handle centroid calculation for objects that wrap around edges
            if check_centroids and not regional_mode and len(props_slice["label"]) > 0:
                # Get original centroids
                centroids = list(zip(props_slice["centroid-0"], props_slice["centroid-1"]))
                centroids_wrapped = []

                # Process each object
                for ID_idx, ID in enumerate(props_slice["label"]):
                    binary_mask = ids == ID
                    centroids_wrapped.append(calculate_centroid(binary_mask, regional_mode, centroids[ID_idx]))

                # Update centroid values
                props_slice["centroid-0"] = [c[0] for c in centroids_wrapped]
                props_slice["centroid-1"] = [c[1] for c in centroids_wrapped]

            return props_slice

        # Process single time or multiple times
        # If time dimension doesn't exist, treat as single time slice
        if timedim not in object_id_field.dims or object_id_field.sizes[timedim] == 1:
            object_props = object_properties_chunk(object_id_field.values)
            object_props = xr.Dataset({key: (["ID"], value) for key, value in object_props.items()})
        else:
            # Run in parallel
            object_props = xr.apply_ufunc(
                object_properties_chunk,
                object_id_field,
                input_core_dims=[[ydim, xdim]],
                output_core_dims=[[]],
                output_dtypes=[object],
                vectorize=True,
                dask="parallelized",
            )

            # Concatenate and convert to dataset
            object_props = xr.concat(
                [xr.Dataset({key: (["ID"], value) for key, value in item.items()}) for item in object_props.values],
                dim="ID",
            )

        # Set ID as coordinate
        object_props = object_props.set_index(ID="label")

    # Combine centroid components into a single variable
    if "centroid" in properties and "centroid-0" in object_props and "centroid-1" in object_props:
        object_props["centroid"] = xr.concat(
            [object_props["centroid-0"], object_props["centroid-1"]],
            dim="component",
        )
        object_props = object_props.drop_vars(["centroid-0", "centroid-1"])

    return object_props
