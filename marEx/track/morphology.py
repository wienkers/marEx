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
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as scipy_label

from ..exceptions import TrackingError
from ..logging_config import get_logger

# _EIGHT_CONNECTIVITY and _merge_lon_seam live in objects.py (imported by this module) to avoid
# duplicating the periodic-longitude seam union-find; both the small-object filter (here) and the
# tracking per-slice labeller (objects.identify_objects) use them.
from .objects import _EIGHT_CONNECTIVITY, _merge_lon_seam, identify_objects
from .overlap import sparse_bool_power

logger = get_logger(__name__)


def _hold(obj, materialiser):
    """Pin `obj` -- single consumer, or already anchored upstream.

    A no-op in streaming mode; that is the point. Staging here would write a second
    copy of an array that is already on disk.
    """
    if materialiser is None:
        return obj.persist()
    return materialiser.pin_one(obj)


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
        # Structured grid: exact Euclidean-disk closing + opening via distance transforms.
        #
        # The previous implementation used dask_image.ndmorph.binary_closing/opening with a
        # brute-force disk structuring element (O(N * R_fill^2) per pixel) split into a large
        # per-chunk task graph; profiling showed this morphology dominated tracker preprocessing
        # (deferred into the "Small object filtering" step). The distance-transform formulation
        # is O(N) per 2D slice regardless of R_fill, embarrassingly parallel across time, and a
        # tiny task graph. Padding by 4*R_fill resolves the full closing+opening reach (4*R_fill)
        # exactly at the periodic-longitude seam -- this also removes a latent under-padding
        # artifact in the old code, which padded only 2*R_fill.
        mode = "wrap" if not regional_mode else "edge"

        if R_fill == 0:
            pass  # No morphological operations needed
        else:
            r_squared = R_fill * R_fill
            pad_width = 4 * R_fill

            def edt_close_open(bitmap_binary: NDArray[np.bool_]) -> NDArray[np.bool_]:
                """Euclidean-disk closing (fill gaps) then opening (remove specks) on one slice."""
                # Pad longitude (axis 1) with `mode` (periodic wrap for global grids); always pad
                # latitude (axis 0) with edge. The poles are not periodic, so a single mode="wrap"
                # pad coupled features across the north/south array boundary.
                lon_padded = np.pad(bitmap_binary, ((0, 0), (pad_width, pad_width)), mode=mode)
                padded = np.pad(lon_padded, ((pad_width, pad_width), (0, 0)), mode="edge").astype(bool)

                def dilate(b: NDArray[np.bool_]) -> NDArray[np.bool_]:
                    # True where the nearest foreground pixel is within disk radius R_fill.
                    return distance_transform_edt(~b) ** 2 <= r_squared

                def erode(b: NDArray[np.bool_]) -> NDArray[np.bool_]:
                    # True where the nearest background pixel is beyond disk radius R_fill.
                    return distance_transform_edt(b) ** 2 > r_squared

                closed = erode(dilate(padded))
                opened = dilate(erode(closed))
                return opened[pad_width:-pad_width, pad_width:-pad_width]

            data_bin = xr.apply_ufunc(
                edt_close_open,
                data_bin,
                input_core_dims=[[ydim, xdim]],
                output_core_dims=[[ydim, xdim]],
                output_dtypes=[np.bool_],
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
    *,
    materialiser=None,
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
    # _hold, not _anchor: the only consumer is the fill_holes call immediately below,
    # and the caller stages the RESULT (tracker.py:879). Staging here too would write
    # the whole field to zarr twice.
    data_bin_filled = _hold(
        data_bin_filled.isel({timedim: slice(kernel_size, -kernel_size)}),
        materialiser,
    )

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


def refresh_dask_graph(data_bin: xr.DataArray, zarr_path: str) -> xr.DataArray:
    """
    Clear and reset the Dask graph via save/load cycle.

    This is needed to work around a memory leak bug in Dask where
    "Unmanaged Memory" builds up within loops.

    Parameters
    ----------
    data_bin : xarray.DataArray
        Data to refresh
    zarr_path : str
        Full path of this run's refresh store. Must not be shared with the merge loop's
        store (see ``update_object_id_field_zarr``) or with a concurrent run.

    Returns
    -------
    data_new : xarray.DataArray
        Data with fresh Dask graph. This is a lazy ``open_zarr`` view, so ``zarr_path``
        must outlive the returned array.
    """
    logger.debug("Refreshing Dask task graph...")

    data_bin.name = "temp"
    data_bin.to_zarr(zarr_path, mode="w")
    del data_bin
    gc.collect()

    data_new = xr.open_zarr(zarr_path, chunks={}).temp
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
    *,
    materialiser=None,
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
    if unstructured_grid:
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

        # Pre-filter tiny objects before the percentile calculation. This is a deliberate
        # performance approximation for unstructured (ICON-scale) grids, where computing the
        # quartile over millions of 1-2 cell specks would be prohibitively memory-heavy on the
        # driver. NOTE: it also means "remove the smallest quartile" is evaluated over objects
        # larger than the pre-filter cutoff here, whereas the structured branch uses all objects;
        # the two grid types therefore define the quantile over slightly different populations.
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
        # Keep ties (area == threshold), matching the structured branch and the
        # accepted-area statistic (>= everywhere).
        N_objects_filtered = np.sum(object_areas >= area_threshold)

        def filter_area_binary(cluster_IDs_0: NDArray[np.int32], keep_IDs_0: NDArray[np.int32]) -> NDArray[np.bool_]:
            """Keep only clusters at or above the threshold area."""
            keep_IDs_0 = keep_IDs_0[keep_IDs_0 > 0]
            keep_where = np.isin(cluster_IDs_0, keep_IDs_0)
            return keep_where

        # Create filtered binary data
        keep_IDs = xr.where(cluster_sizes >= area_threshold, unique_cluster_IDs, 0)

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
        # Structured grid: per-2D-slice connected-component labelling + bincount areas.
        #
        # The previous implementation labelled the whole (time, y, x) field with
        # dask_image.ndmeasure.label and then ran skimage.regionprops_table per slice.
        # Profiling showed dask_image.label's global cross-block relabel carries a large
        # fixed overhead that dominates this step even when there are few objects (the
        # object-count-dependent regionprops "max-label" cost is incidental here, because
        # fill_holes already collapses the object count). Labelling each 2D slice
        # independently with scipy.ndimage.label + np.bincount avoids that overhead, is
        # embarrassingly parallel, and uses a tiny task graph. Connectivity is 8-connected
        # in space with periodic-longitude wrap (except in regional mode), matching the old
        # identify_objects(time_connectivity=False, wrap_axes=(2,)). Areas are pixel counts
        # (identical to the old regionprops 'area' on a structured grid).
        area_buffer = 8192  # max objects per 2D slice; overflow is checked below

        def slice_object_areas(bitmap_binary: NDArray[np.bool_]) -> NDArray[np.int64]:
            """Per-object pixel areas for one slice, padded to ``area_buffer`` (0 == empty)."""
            labels, n_labels = scipy_label(bitmap_binary, structure=_EIGHT_CONNECTIVITY)
            if not regional_mode:
                labels = _merge_lon_seam(labels, n_labels)
            counts = np.bincount(labels.ravel())
            areas = counts[1:][counts[1:] > 0]  # drop background; compact post-merge gaps
            out = np.zeros(area_buffer, dtype=np.int64)
            out[: min(len(areas), area_buffer)] = areas[:area_buffer]
            return out

        def slice_areas_and_filter(bitmap_binary: NDArray[np.bool_]) -> Tuple[NDArray[np.bool_], NDArray[np.int64]]:
            """Areas *and* keep-mask from a single labelling pass (absolute-threshold mode)."""
            labels, n_labels = scipy_label(bitmap_binary, structure=_EIGHT_CONNECTIVITY)
            if not regional_mode:
                labels = _merge_lon_seam(labels, n_labels)
            counts = np.bincount(labels.ravel())
            areas = counts[1:][counts[1:] > 0]  # drop background; compact post-merge gaps
            out = np.zeros(area_buffer, dtype=np.int64)
            out[: min(len(areas), area_buffer)] = areas[:area_buffer]
            keep = counts >= area_filter_absolute
            keep[0] = False  # Don't keep background (label 0)
            return keep[labels], out

        # With an absolute threshold the keep-decision is known before the area census, so
        # one labelling pass can produce both. The quartile mode still needs two, because
        # its threshold is a percentile of the census (review finding 6.10). Both outputs
        # are persisted together so the shared pass is actually shared -- computing them
        # separately would label every slice twice again.
        data_bin_filtered = None
        if use_absolute_filtering:
            data_bin_filtered, padded_areas = xr.apply_ufunc(
                slice_areas_and_filter,
                data_bin,
                input_core_dims=[[ydim, xdim]],
                output_core_dims=[[ydim, xdim], ["object_buffer"]],
                dask_gufunc_kwargs={"output_sizes": {"object_buffer": area_buffer}},
                output_dtypes=[np.bool_, np.int64],
                vectorize=True,
                dask="parallelized",
            )
            # Both _hold: data_bin_filtered is staged by the caller (tracker.py:908),
            # and padded_areas is (time, object_buffer), not whole-field.
            if materialiser is None:
                data_bin_filtered, padded_areas = persist(data_bin_filtered, padded_areas)
            else:
                data_bin_filtered, padded_areas = materialiser.pin(data_bin_filtered, padded_areas)
        else:
            padded_areas = xr.apply_ufunc(
                slice_object_areas,
                data_bin,
                input_core_dims=[[ydim, xdim]],
                output_core_dims=[["object_buffer"]],
                dask_gufunc_kwargs={"output_sizes": {"object_buffer": area_buffer}},
                output_dtypes=[np.int64],
                vectorize=True,
                dask="parallelized",
            )
        padded_areas = np.atleast_2d(padded_areas.compute().values)  # (time, area_buffer)
        if np.any(padded_areas[:, -1] != 0):  # pragma: no cover
            raise TrackingError(
                "Per-slice object-area buffer overflow during small-object filtering",
                details={"area_buffer": area_buffer},
                suggestions=["Increase 'area_buffer' in filter_small_objects (structured branch)"],
            )
        object_areas_np = padded_areas[padded_areas > 0]

        N_objects_unfiltered = int(object_areas_np.size)
        if N_objects_unfiltered == 0:  # pragma: no cover
            raise TrackingError(
                "No objects found for area-based filtering",
                details={
                    "objects_count": N_objects_unfiltered,
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
            area_threshold = np.percentile(object_areas_np, area_filter_quartile * 100.0)
        N_objects_filtered = int(np.sum(object_areas_np >= area_threshold))

        keep_threshold = area_threshold

        def slice_filter(bitmap_binary: NDArray[np.bool_]) -> NDArray[np.bool_]:
            """Keep only objects whose pixel area is >= ``keep_threshold`` on one slice."""
            labels, n_labels = scipy_label(bitmap_binary, structure=_EIGHT_CONNECTIVITY)
            if not regional_mode:
                labels = _merge_lon_seam(labels, n_labels)
            counts = np.bincount(labels.ravel())
            keep = counts >= keep_threshold
            keep[0] = False  # Don't keep background (label 0)
            return keep[labels]

        if data_bin_filtered is None:  # quartile mode: the threshold needed the census first
            data_bin_filtered = xr.apply_ufunc(
                slice_filter,
                data_bin,
                input_core_dims=[[ydim, xdim]],
                output_core_dims=[[ydim, xdim]],
                output_dtypes=[np.bool_],
                vectorize=True,
                dask="parallelized",
            )

        object_areas = xr.DataArray(object_areas_np, dims=["ID"])

    return (
        data_bin_filtered,
        area_threshold,
        object_areas,
        N_objects_unfiltered,
        N_objects_filtered,
    )
