"""
MarEx Track: Morphological preprocessing.

Stateless helpers for the morphological preprocessing stage -- area
accounting, spatial hole filling, temporal gap filling, Dask-graph refresh and
small-object filtering -- extracted from the tracker orchestrator. The tracker
config/grid values each method read from ``self`` are threaded in as explicit
arguments. Behaviour and numerics are identical to the original ``tracker``
methods.
"""

import gc
from typing import Optional, Tuple

import dask.array as dsa
import numpy as np
import xarray as xr
from dask import persist
from dask_image.ndmorph import binary_closing as binary_closing_dask
from dask_image.ndmorph import binary_opening as binary_opening_dask
from numpy.typing import NDArray
from scipy.ndimage import binary_closing, binary_opening

from ..exceptions import TrackingError
from ..logging_config import get_logger
from .objects import calculate_object_properties, identify_objects
from .overlap import sparse_bool_power

logger = get_logger(__name__)


def compute_area(
    data_bin: xr.DataArray,
    unstructured_grid: bool,
    cell_area: xr.DataArray,
    xdim: str,
    ydim: Optional[str],
) -> xr.DataArray:
    """
    Compute the total area of binary data at each time.

    Parameters
    ----------
    data_bin : xarray.DataArray
        Binary data

    Returns
    -------
    area : xarray.DataArray
        Total area at each time (units: pixels for structured grid, matching cell_area for unstructured)
    """
    if unstructured_grid:
        area = (data_bin * cell_area).sum(dim=[xdim])
    else:
        area = data_bin.sum(dim=[ydim, xdim])

    return area


def fill_holes(
    data_bin: xr.DataArray,
    default_R_fill: int,
    unstructured_grid: bool,
    dilate_sparse,
    xdim: str,
    mask: xr.DataArray,
    regional_mode: bool,
    ydim: Optional[str],
    R_fill: Optional[int] = None,
) -> xr.DataArray:
    """
    Fill holes and gaps using morphological operations.

    This performs closing (dilation followed by erosion) to fill small gaps,
    then opening (erosion followed by dilation) to remove small isolated objects.

    Parameters
    ----------
    data_bin : xarray.DataArray
        Binary data to process
    R_fill : int, optional
        Fill radius override

    Returns
    -------
    data_bin_filled : xarray.DataArray
        Binary data with holes/gaps filled
    """
    if R_fill is None:
        R_fill = default_R_fill

    if unstructured_grid:
        # Process unstructured grid using sparse matrix operations
        # _Put the data into an xarray.DataArray to pass into the apply_ufunc_ -- Needed for correct memory management !
        sp_data = xr.DataArray(dilate_sparse.data, dims="sp_data")
        indices = xr.DataArray(dilate_sparse.indices, dims="indices")
        indptr = xr.DataArray(dilate_sparse.indptr, dims="indptr")

        def binary_open_close(
            bitmap_binary: NDArray[np.bool_],
            sp_data: NDArray[np.bool_],
            indices: NDArray[np.int32],
            indptr: NDArray[np.int32],
            mask: NDArray[np.bool_],
        ) -> NDArray[np.bool_]:
            """
            Binary opening and closing for unstructured grid.
            Uses sparse matrix power operations for efficiency.
            """
            # Closing: Dilation then Erosion (fills small gaps)

            # Dilation
            bitmap_binary = sparse_bool_power(bitmap_binary, sp_data, indices, indptr, R_fill)

            # Set land values to True (to avoid artificially eroding the shore)
            bitmap_binary[:, ~mask] = True

            # Erosion (negated dilation of negated image)
            bitmap_binary = ~sparse_bool_power(~bitmap_binary, sp_data, indices, indptr, R_fill)

            # Opening: Erosion then Dilation (removes small objects)

            # Set land values to True (to avoid artificially eroding the shore)
            bitmap_binary[:, ~mask] = True

            # Erosion
            bitmap_binary = ~sparse_bool_power(~bitmap_binary, sp_data, indices, indptr, R_fill)

            # Dilation
            bitmap_binary = sparse_bool_power(bitmap_binary, sp_data, indices, indptr, R_fill)

            return bitmap_binary

        # Apply the operations
        data_bin = xr.apply_ufunc(
            binary_open_close,
            data_bin,
            sp_data,
            indices,
            indptr,
            mask,
            input_core_dims=[
                [xdim],
                ["sp_data"],
                ["indices"],
                ["indptr"],
                [xdim],
            ],
            output_core_dims=[[xdim]],
            output_dtypes=[np.bool_],
            vectorize=False,
            dask_gufunc_kwargs={
                "output_sizes": {xdim: data_bin.sizes[xdim]},
            },
            dask="parallelized",
        )

    else:
        # Structured grid using dask-powered morphological operations
        use_dask_morph = True

        # Generate structuring element (disk-shaped)
        y, x = np.ogrid[-R_fill : R_fill + 1, -R_fill : R_fill + 1]
        r = x**2 + y**2
        diameter = 2 * R_fill
        se_kernel = r < (R_fill**2) + 1
        mode = "wrap" if not regional_mode else "edge"

        if use_dask_morph:
            # Skip all operations if R_fill is 0
            if R_fill == 0:
                pass  # No morphological operations needed
            else:
                # Pad data to avoid edge effects
                data_bin = data_bin.pad({ydim: diameter, xdim: diameter}, mode=mode)
                data_coords = data_bin.coords
                data_dims = data_bin.dims

                # Apply morphological operations
                data_bin = binary_closing_dask(
                    data_bin.data, structure=se_kernel[np.newaxis, :, :]
                )  # N.B.: There may be a rearing bug in constructing the dask task graph when we
                # extract and then re-imbed the dask array into an xarray DataArray
                data_bin = binary_opening_dask(data_bin, structure=se_kernel[np.newaxis, :, :])

                # Convert back to xarray.DataArray and trim padding
                data_bin = xr.DataArray(data_bin, coords=data_coords, dims=data_dims)
                data_bin = data_bin.isel(
                    {
                        ydim: slice(diameter, -diameter),
                        xdim: slice(diameter, -diameter),
                    }
                )
        else:  # pragma: no cover

            def binary_open_close(
                bitmap_binary: NDArray[np.bool_],
            ) -> NDArray[np.bool_]:
                """Apply binary opening and closing in one function."""
                bitmap_binary_padded = np.pad(
                    bitmap_binary,
                    ((diameter, diameter), (diameter, diameter)),
                    mode=mode,
                )
                s1 = binary_closing(bitmap_binary_padded, se_kernel, iterations=1)
                s2 = binary_opening(s1, se_kernel, iterations=1)
                unpadded = s2[diameter:-diameter, diameter:-diameter]
                return unpadded

            data_bin = xr.apply_ufunc(
                binary_open_close,
                data_bin,
                input_core_dims=[[ydim, xdim]],
                output_core_dims=[[ydim, xdim]],
                output_dtypes=[data_bin.dtype],
                vectorize=True,
                dask="parallelized",
            )

        # Mask out edge features from morphological operations
        data_bin = data_bin.where(mask, drop=False, other=False)

    return data_bin


def fill_time_gaps(
    data_bin: xr.DataArray,
    T_fill: int,
    R_fill: int,
    timedim: str,
    ydim: Optional[str],
    unstructured_grid: bool,
    dilate_sparse,
    xdim: str,
    mask: xr.DataArray,
    regional_mode: bool,
) -> xr.DataArray:
    """
    Fill temporal gaps between objects.

    Performs binary closing (dilation then erosion) along the time dimension
    to fill small time gaps between objects.

    Parameters
    ----------
    data_bin : xarray.DataArray
        Binary data to process

    Returns
    -------
    data_bin_filled : xarray.DataArray
        Binary data with temporal gaps filled
    """
    if T_fill == 0:
        return data_bin

    # Create temporal structuring element
    kernel_size = T_fill + 1  # This will then fill a maximum hole size of T_fill
    time_kernel = np.ones(kernel_size, dtype=bool)

    if ydim is None:
        # Unstructured grid has only 1 additional dimension
        time_kernel = time_kernel[:, np.newaxis]
    else:
        time_kernel = time_kernel[:, np.newaxis, np.newaxis]

    # Pad in time to avoid edge effects
    data_bin = data_bin.pad({timedim: kernel_size}, mode="constant", constant_values=False)

    # Apply temporal closing
    data_bin_dask = data_bin.data
    closed_dask_array = binary_closing_dask(data_bin_dask, structure=time_kernel)

    # Convert back to xarray.DataArray
    data_bin_filled = xr.DataArray(
        closed_dask_array,
        coords=data_bin.coords,
        dims=data_bin.dims,
        attrs=data_bin.attrs,
    )

    # Remove padding
    data_bin_filled = data_bin_filled.isel({timedim: slice(kernel_size, -kernel_size)}).persist()

    # Fill newly-created spatial holes
    data_bin_filled = fill_holes(
        data_bin_filled,
        default_R_fill=R_fill,
        unstructured_grid=unstructured_grid,
        dilate_sparse=dilate_sparse,
        xdim=xdim,
        mask=mask,
        regional_mode=regional_mode,
        ydim=ydim,
        R_fill=R_fill // 2,
    )

    return data_bin_filled


def refresh_dask_graph(data_bin: xr.DataArray, scratch_dir: str) -> xr.DataArray:
    """
    Clear and reset the Dask graph via save/load cycle.

    This is needed to work around a memory leak bug in Dask where
    "Unmanaged Memory" builds up within loops.

    Parameters
    ----------
    data_bin : xarray.DataArray
        Data to refresh

    Returns
    -------
    data_new : xarray.DataArray
        Data with fresh Dask graph
    """
    logger.debug("Refreshing Dask task graph...")

    data_bin.name = "temp"
    data_bin.to_zarr(f"{scratch_dir}/marEx_temp_field.zarr", mode="w")
    del data_bin
    gc.collect()

    data_new = xr.open_zarr(f"{scratch_dir}/marEx_temp_field.zarr", chunks={}).temp
    return data_new


def filter_small_objects(
    data_bin: xr.DataArray,
    unstructured_grid: bool,
    xdim: str,
    use_absolute_filtering: bool,
    area_filter_absolute: int,
    area_filter_quartile: float,
    mask: xr.DataArray,
    neighbours_int: Optional[xr.DataArray],
    regional_mode: bool,
    lat: xr.DataArray,
    lon: xr.DataArray,
    cell_area: xr.DataArray,
    timedim: str,
    ydim: Optional[str],
) -> Tuple[xr.DataArray, float, xr.DataArray, int, int]:
    """
    Remove objects smaller than a threshold area.

    Parameters
    ----------
    data_bin : xarray.DataArray
        Binary data to filter

    Returns
    -------
    data_bin_filtered : xarray.DataArray
        Binary data with small objects removed
    area_threshold : float
        Area threshold used for filtering
    object_areas : xarray.DataArray
        Areas of all objects pre-filtering
    N_objects_prefiltered : int
        Number of objects before filtering
    N_objects_filtered : int
        Number of objects after filtering
    """
    # Cluster & Label Binary Data: Time-independent in 2D (i.e. no time connectivity!)
    object_id_field, _, N_objects_unfiltered = identify_objects(
        data_bin,
        time_connectivity=False,
        unstructured_grid=unstructured_grid,
        mask=mask,
        neighbours_int=neighbours_int,
        xdim=xdim,
        regional_mode=regional_mode,
    )

    if unstructured_grid:
        # Get the maximum ID to dimension arrays
        #  Note: identify_objects() starts at ID=0 for every time slice
        max_ID = int(object_id_field.max().compute().item())

        def count_cluster_sizes(
            object_id_field: NDArray[np.int32],
        ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
            """Count the number of cells in each cluster."""
            unique, counts = np.unique(object_id_field[object_id_field > 0], return_counts=True)
            padded_sizes = np.zeros(max_ID, dtype=np.int32)
            padded_unique = np.zeros(max_ID, dtype=np.int32)
            padded_sizes[: len(counts)] = counts
            padded_unique[: len(counts)] = unique
            return padded_sizes, padded_unique

        # Calculate cluster sizes
        cluster_sizes, unique_cluster_IDs = xr.apply_ufunc(
            count_cluster_sizes,
            object_id_field,
            input_core_dims=[[xdim]],
            output_core_dims=[["ID"], ["ID"]],
            dask_gufunc_kwargs={"output_sizes": {"ID": max_ID}},
            output_dtypes=(np.int32, np.int32),
            vectorize=True,
            dask="parallelized",
        )

        results = persist(cluster_sizes, unique_cluster_IDs)
        cluster_sizes, unique_cluster_IDs = results

        # Pre-filter tiny objects for performance (greatly reduces the size for the percentile calculation)
        if use_absolute_filtering:
            cluster_sizes_filtered_dask = cluster_sizes.where(cluster_sizes > 5).data
        else:
            cluster_sizes_filtered_dask = cluster_sizes.where(cluster_sizes > 50).data
        cluster_areas_mask = dsa.isfinite(cluster_sizes_filtered_dask)
        object_areas = cluster_sizes_filtered_dask[cluster_areas_mask].compute()

        # Filter based on area threshold
        N_objects_unfiltered = len(object_areas)
        if N_objects_unfiltered == 0:  # pragma: no cover
            raise TrackingError(
                "No objects found for area-based filtering",
                details={
                    "objects_count": N_objects_unfiltered,
                    "area_filter_quartile": area_filter_quartile,
                    "grid_type": "unstructured",
                },
                suggestions=[
                    "Check if input data contains any extreme events",
                    "Verify that preprocessing parameters are appropriate",
                    "Consider lowering the extreme threshold percentile",
                ],
            )
        if use_absolute_filtering:
            area_threshold = area_filter_absolute
        else:
            area_threshold = np.percentile(object_areas, area_filter_quartile * 100)
        N_objects_filtered = np.sum(object_areas > area_threshold)

        def filter_area_binary(cluster_IDs_0: NDArray[np.int32], keep_IDs_0: NDArray[np.int32]) -> NDArray[np.bool_]:
            """Keep only clusters above threshold area."""
            keep_IDs_0 = keep_IDs_0[keep_IDs_0 > 0]
            keep_where = np.isin(cluster_IDs_0, keep_IDs_0)
            return keep_where

        # Create filtered binary data
        keep_IDs = xr.where(cluster_sizes > area_threshold, unique_cluster_IDs, 0)

        data_bin_filtered = xr.apply_ufunc(
            filter_area_binary,
            object_id_field,
            keep_IDs,
            input_core_dims=[[xdim], ["ID"]],
            output_core_dims=[[xdim]],
            output_dtypes=[data_bin.dtype],
            vectorize=True,
            dask="parallelized",
        )

        object_areas = cluster_sizes  # Store pre-filtered areas

    else:
        # Structured grid approach

        # Calculate object properties including area
        object_props = calculate_object_properties(
            object_id_field,
            unstructured_grid=unstructured_grid,
            lat=lat,
            lon=lon,
            cell_area=cell_area,
            timedim=timedim,
            regional_mode=regional_mode,
            ydim=ydim,
            xdim=xdim,
        )
        object_areas, object_ids = object_props.area, object_props.ID

        # Calculate area threshold
        if len(object_areas) == 0:  # pragma: no cover
            raise TrackingError(
                "No objects found for area-based filtering",
                details={
                    "objects_count": len(object_areas),
                    "area_filter_quartile": area_filter_quartile,
                    "grid_type": "structured",
                },
                suggestions=[
                    "Check if input data contains any extreme events",
                    "Verify that preprocessing parameters are appropriate",
                    "Consider lowering the extreme threshold percentile",
                ],
            )
        if use_absolute_filtering:
            area_threshold = area_filter_absolute
        else:
            area_threshold = np.percentile(object_areas, area_filter_quartile * 100.0)

        # Keep only objects above threshold
        object_ids_keep = xr.where(object_areas >= area_threshold, object_ids, -1)
        object_ids_keep[0] = -1  # Don't keep ID=0

        # Create filtered binary data
        data_bin_filtered = object_id_field.isin(object_ids_keep)

        # Count objects after filtering
        N_objects_filtered = int(object_ids_keep.where(object_ids_keep > 0).count().item())

    return (
        data_bin_filtered,
        area_threshold,
        object_areas,
        N_objects_unfiltered,
        N_objects_filtered,
    )
