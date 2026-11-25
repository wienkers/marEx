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
from math import isqrt
from operator import index
from typing import Optional, Tuple

import dask
import dask.array as dsa
import numpy as np
import xarray as xr
from dask import persist
from numba import njit
from numpy.typing import NDArray
from scipy.ndimage import label as scipy_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..core.encoding import write_zarr
from ..exceptions import TrackingError
from ..logging_config import get_logger

# _EIGHT_CONNECTIVITY and _merge_lon_seam live in objects.py (imported by this module) to avoid
# duplicating the periodic-longitude seam union-find; both the small-object filter (here) and the
# tracking per-slice labeller (objects.identify_objects) use them.
from .objects import _EIGHT_CONNECTIVITY, _merge_lon_seam, identify_objects
from .overlap import sparse_bool_power
from .validation import validate_fill_size

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


@njit(nogil=True)
def _graph_dilation_kind(data, indices, indptr):
    """Identify graphs requiring exact walks, a transpose, or neither."""
    symmetric = True
    for cell in range(indptr.size - 1):
        has_loop = False
        for edge in range(indptr[cell], indptr[cell + 1]):
            if not data[edge]:
                continue
            neighbour = indices[edge]
            if neighbour == cell:
                has_loop = True
            elif symmetric:
                start, stop = indptr[neighbour], indptr[neighbour + 1]
                # Reciprocal triangular meshes have at most four entries per row.
                # Bound the search on other graphs to keep setup linear in the edge count.
                symmetric = False
                if stop - start <= 8:
                    for reverse in range(start, stop):
                        if indices[reverse] == cell and data[reverse]:
                            symmetric = True
                            break
        if not has_loop:
            return 0
    return 2 if symmetric else 1


@njit(nogil=True)
def _graph_close_open(values, data, indices, indptr, wet, radius, result, queue, visited):
    """Apply four bounded dilations with the appropriate foreground polarity."""
    n_frames, n_cells = values.shape
    for frame in range(n_frames):
        for cell in range(n_cells):
            visited[cell] = 0
        for stamp in range(1, 5):
            tail = 0
            for cell in range(n_cells):
                if stamp == 1:
                    seed = values[frame, cell]
                elif stamp == 2:
                    seed = visited[cell] != 1 and wet[cell]
                elif stamp == 3:
                    seed = visited[cell] == 2 and wet[cell]
                else:
                    seed = visited[cell] != 3
                # The wet restriction on each erosion makes land foreground before negation.
                if seed:
                    visited[cell] = stamp
                    queue[tail] = cell
                    tail += 1

            head = 0
            for _ in range(radius):
                end_of_level = tail
                if head == end_of_level or tail == n_cells:
                    break
                while head < end_of_level and tail < n_cells:
                    cell = queue[head]
                    head += 1
                    for edge in range(indptr[cell], indptr[cell + 1]):
                        neighbour = indices[edge]
                        if visited[neighbour] != stamp and data[edge]:
                            visited[neighbour] = stamp
                            queue[tail] = neighbour
                            tail += 1

        for cell in range(n_cells):
            result[frame, cell] = visited[cell] == 4


def _unstructured_fill_block(bitmap, sp_data, indices, indptr, mask, *, radius):
    """Close and open a time chunk using push adjacency when all walks can wait."""
    if radius == 0:
        result = bitmap.copy()
        result[:, ~mask] = True
        return result

    kind = _graph_dilation_kind(sp_data, indices, indptr)
    if kind == 0:
        # Without self-loops, the power counts walks of exactly radius steps.
        bitmap = sparse_bool_power(bitmap, sp_data, indices, indptr, radius)
        bitmap[:, ~mask] = True
        bitmap = ~sparse_bool_power(~bitmap, sp_data, indices, indptr, radius)
        bitmap[:, ~mask] = True
        bitmap = ~sparse_bool_power(~bitmap, sp_data, indices, indptr, radius)
        return sparse_bool_power(bitmap, sp_data, indices, indptr, radius)

    n_cells = indptr.size - 1
    if kind == 1:
        gather = csr_matrix((sp_data, indices, indptr), shape=(n_cells, n_cells))
        push = gather.T.tocsr()
        sp_data, indices, indptr = push.data, push.indices, push.indptr

    result = np.empty(bitmap.shape, dtype=bool)
    queue = np.empty(n_cells, dtype=indices.dtype)
    visited = np.empty(n_cells, dtype=np.uint8)
    _graph_close_open(bitmap, sp_data, indices, indptr, mask, radius, result, queue, visited)
    return result


@njit(nogil=True)
def _dilate_disk_words(source, result, horizontal, scratch, half_widths, first, stop, last_mask):
    """OR disk chords into the valid output rows of a packed longitude grid."""
    radius = half_widths.size - 1
    n_words = source.shape[1]
    result[first:stop] = 0
    expanded = source
    enlarged = horizontal
    other = scratch
    width = 0
    for dy in range(radius, -1, -1):
        half_width = half_widths[dy]
        while width < half_width:
            # Limit the shift to width + 1 to avoid gaps at clipped edges,
            # and to 63 so both shifts fit in a word.
            step = min(half_width - width, width + 1, 63)
            left_shift = np.uint64(step)
            right_shift = np.uint64(64 - step)
            for y in range(first - dy, stop + dy):
                for word in range(n_words):
                    centre = expanded[y, word]
                    spread = centre | (centre << left_shift) | (centre >> left_shift)
                    if word > 0:
                        spread |= expanded[y, word - 1] >> right_shift
                    if word + 1 < n_words:
                        spread |= expanded[y, word + 1] << right_shift
                    enlarged[y, word] = spread
                enlarged[y, n_words - 1] &= last_mask
            width += step
            expanded, enlarged, other = enlarged, other, enlarged

        for y in range(first, stop):
            for word in range(n_words):
                result[y, word] |= expanded[y - dy, word] | expanded[y + dy, word]


def _structured_fill_block(bitmap, *, radius, regional):
    """Close and open each horizontal plane using the same four-radius halo."""
    radius = index(radius)
    ny, nx = bitmap.shape[-2:]
    result = np.empty(bitmap.shape, dtype=bool)
    if result.size == 0:
        return result
    # A change at any outer boundary can travel at most one radius per pass.
    halo = 4 * radius
    ny_padded, nx_padded = ny + 2 * halo, nx + 2 * halo
    mode = "edge" if regional else "wrap"
    planes = bitmap.reshape(-1, ny, nx)
    outputs = result.reshape(-1, ny, nx)
    erosions = (False, True, True, False)

    n_words = (nx_padded + 63) // 64
    shape = ny_padded, n_words
    packed, work, horizontal, scratch = (np.empty(shape, dtype=np.uint64) for _ in range(4))
    half_widths = np.array([isqrt(radius * radius - dy * dy) for dy in range(radius + 1)], dtype=np.int64)
    last_mask = np.uint64((1 << (nx_padded - 64 * (n_words - 1))) - 1)
    packed_bytes = np.zeros((ny_padded, n_words * 8), dtype=np.uint8)
    first_byte, leading_bits = divmod(halo, 8)
    for plane, output in zip(planes, outputs):
        padded = np.pad(np.pad(plane, ((0, 0), (halo, halo)), mode=mode), ((halo, halo), (0, 0)), mode="edge")
        byte_rows = np.packbits(padded.astype(bool, copy=False), axis=1, bitorder="little")
        packed_bytes[:, : byte_rows.shape[1]] = byte_rows
        packed[:] = packed_bytes.view("<u8")
        for step, erosion in enumerate(erosions):
            first, stop = step * radius, ny_padded - step * radius
            if erosion:
                np.bitwise_not(packed[first:stop], out=packed[first:stop])
                packed[first:stop, -1] &= last_mask
            first, stop = first + radius, stop - radius
            _dilate_disk_words(packed, work, horizontal, scratch, half_widths, first, stop, last_mask)
            if erosion:
                np.bitwise_not(work[first:stop], out=work[first:stop])
                work[first:stop, -1] &= last_mask
            packed, work = work, packed
        bytes_in_core_rows = packed[halo : halo + ny].astype("<u8", copy=False).view(np.uint8)
        core_bits = np.unpackbits(bytes_in_core_rows[:, first_byte:], axis=1, count=leading_bits + nx, bitorder="little")
        output[:] = core_bits[:, leading_bits:]
    return result


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
    validate_fill_size(R_fill, "R_fill")

    if unstructured_grid:
        # Keep each CSR vector in a single chunk. Splitting a core dimension
        # forces Dask to join it again and subdivide the time chunks.
        sp_data = xr.DataArray(dilate_sparse.data, dims="sp_data").chunk(-1)
        indices = xr.DataArray(dilate_sparse.indices, dims="indices").chunk(-1)
        indptr = xr.DataArray(dilate_sparse.indptr, dims="indptr").chunk(-1)

        data_bin = xr.apply_ufunc(
            _unstructured_fill_block,
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
            kwargs={"radius": R_fill},
        )

    else:
        if R_fill != 0:
            data_bin = xr.apply_ufunc(
                _structured_fill_block,
                data_bin,
                input_core_dims=[[ydim, xdim]],
                output_core_dims=[[ydim, xdim]],
                output_dtypes=[np.bool_],
                vectorize=False,
                dask="parallelized",
                kwargs={"radius": R_fill, "regional": regional_mode},
            )

        # Mask out edge features from morphological operations
        data_bin = data_bin.where(mask, drop=False, other=False)

    return data_bin


@njit(nogil=True)
def _close_time_runs(values, max_gap):
    """Close bounded false runs in a contiguous (time, cells) block."""
    closed = values.copy()
    n_time, n_cells = values.shape
    # Keep the previous-hit array and the rows being revisited by short fills in cache.
    tile_cells = 8192
    last = np.empty(min(tile_cells, n_cells), dtype=np.int64)
    for start in range(0, n_cells, tile_cells):
        width = min(tile_cells, n_cells - start)
        last[:width] = -1
        for t in range(n_time):
            for j in range(width):
                cell = start + j
                if values[t, cell]:
                    prev = last[j]
                    if prev >= 0 and t - prev - 1 <= max_gap:
                        closed[prev + 1 : t, cell] = True
                    last[j] = t
    return closed


def _close_time_block(block, *, max_gap, time_axis):
    time_first = np.ascontiguousarray(np.moveaxis(block, time_axis, 0))
    if time_first.size == 0:
        return block
    closed = _close_time_runs(time_first.reshape(time_first.shape[0], -1), max_gap)
    return np.moveaxis(closed.reshape(time_first.shape), 0, time_axis)


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

    Fill runs of at most `T_fill` inactive samples bounded by active samples.
    This is the temporal closing with a flat kernel of length `T_fill + 1`
    and false padding at the record ends.

    Parameters
    ----------
    data_bin : xarray.DataArray
        Binary data to process

    Returns
    -------
    data_bin_filled : xarray.DataArray
        Binary data with temporal gaps filled
    """
    validate_fill_size(T_fill, "T_fill")
    validate_fill_size(R_fill, "R_fill")
    if T_fill == 0:
        return data_bin

    # Dask can rebalance time chunks shorter than the halo. Restore the caller's
    # boundaries after closing, including the short final chunk used for Zarr writes.
    time_axis = data_bin.dims.index(timedim)
    input_time_chunks = data_bin.chunks[time_axis] if data_bin.chunks is not None else None
    # Each cell in a fillable run is at most T_fill samples from either bounding hit.
    depth = min(T_fill, max(0, data_bin.sizes[timedim] - 1))
    closed_dask_array = dsa.map_overlap(
        _close_time_block,
        dsa.asarray(data_bin.data, dtype=bool),
        depth={time_axis: depth},
        boundary="none",
        dtype=bool,
        meta=np.empty((0,) * data_bin.ndim, dtype=bool),
        max_gap=T_fill,
        time_axis=time_axis,
    )
    data_bin_filled = xr.DataArray(
        closed_dask_array,
        coords=data_bin.coords,
        dims=data_bin.dims,
        attrs=data_bin.attrs,
    )

    if input_time_chunks is not None and data_bin_filled.chunks is not None:
        if data_bin_filled.chunks[time_axis] != input_time_chunks:
            data_bin_filled = data_bin_filled.chunk({timedim: input_time_chunks})

    # _hold, not _anchor: the only consumer is the fill_holes call immediately below,
    # and the caller stages the RESULT (tracker.py:879). Staging here too would write
    # the whole field to zarr twice.
    data_bin_filled = _hold(data_bin_filled, materialiser)

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
    write_zarr(data_bin, zarr_path, mode="w")
    del data_bin
    gc.collect()

    data_new = xr.open_zarr(zarr_path, chunks={}).temp
    return data_new


def drop_small_components(
    data_bin: xr.DataArray,
    min_cells: int,
    unstructured_grid: bool,
    xdim: str,
    ydim: Optional[str],
    regional_mode: bool,
    neighbours_int: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Drop connected components smaller than ``min_cells`` from each time slice.

    Runs before the morphological closing, so isolated specks cannot act as stepping stones
    that bridge separate objects once they are dilated. ``filter_small_objects`` cannot do
    this: it runs after the closing, when the bridges have already formed.

    Connectivity matches the tracker's own per-slice labelling: 8-connected with the
    longitude seam joined (structured, global), 8-connected without the seam (regional), and
    edge-sharing neighbours (unstructured). The result is lazy and keeps the input chunks.

    Parameters
    ----------
    data_bin : xarray.DataArray
        Binary field, spatially unchunked.
    min_cells : int
        Components with fewer cells than this are set to False. Components of exactly
        ``min_cells`` are kept.

    Returns
    -------
    xarray.DataArray
        Boolean field of the same shape and chunking.
    """
    if unstructured_grid:

        def keep_large(arr: NDArray[np.bool_], nb: NDArray[np.int32]) -> NDArray[np.bool_]:
            out = np.zeros(arr.shape, dtype=bool)
            flat = arr.reshape(-1, arr.shape[-1])
            out_flat = out.reshape(-1, arr.shape[-1])
            for i in range(flat.shape[0]):
                x = flat[i]
                idx = np.flatnonzero(x)
                if idx.size == 0:
                    continue
                cols = nb[:, idx]  # (3, k) neighbour of each True cell, -1 where absent
                rows = np.broadcast_to(np.arange(idx.size), cols.shape)
                ok = cols >= 0
                ok[ok] = x[cols[ok]]
                graph = csr_matrix(
                    (
                        np.ones(int(ok.sum()), dtype=np.int8),
                        (rows[ok], np.searchsorted(idx, cols[ok])),
                    ),
                    shape=(idx.size, idx.size),
                )
                _, labels = connected_components(graph, directed=False)
                keep = np.bincount(labels) >= min_cells
                out_flat[i, idx[keep[labels]]] = True
            return out

        nb = neighbours_int.chunk({d: -1 for d in neighbours_int.dims})
        return xr.apply_ufunc(
            keep_large,
            data_bin,
            nb,
            input_core_dims=[[xdim], ["nv", xdim]],
            output_core_dims=[[xdim]],
            output_dtypes=[np.bool_],
            dask="parallelized",
        )

    def keep_large_slice(bitmap: NDArray[np.bool_]) -> NDArray[np.bool_]:
        labels, n_labels = scipy_label(bitmap, structure=_EIGHT_CONNECTIVITY)
        if not regional_mode:
            labels = _merge_lon_seam(labels, n_labels)
        keep = np.bincount(labels.ravel()) >= min_cells
        keep[0] = False
        return keep[labels]

    return xr.apply_ufunc(
        keep_large_slice,
        data_bin,
        input_core_dims=[[ydim, xdim]],
        output_core_dims=[[ydim, xdim]],
        output_dtypes=[np.bool_],
        vectorize=True,
        dask="parallelized",
    )


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
            materialiser=materialiser,
            # A DISTINCT label: tracker.run_tracking() also calls identify_objects, and on
            # the unstructured path both run in one job. Sharing "object_id_field" would
            # make the second stage overwrite the first's store -- Materialiser now raises
            # on that rather than corrupting, so a shared label would fail the run outright.
            stage_label="filter_object_id_field",
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
        prefilter_cells = 5 if use_absolute_filtering else 50
        cluster_sizes_filtered_dask = cluster_sizes.where(cluster_sizes > prefilter_cells).data
        cluster_areas_mask = dsa.isfinite(cluster_sizes_filtered_dask)
        object_areas = cluster_sizes_filtered_dask[cluster_areas_mask].compute()

        # Filter based on area threshold
        N_objects_unfiltered = len(object_areas)
        if N_objects_unfiltered == 0:
            setting = (
                f"area_filter_absolute={area_filter_absolute}"
                if use_absolute_filtering
                else f"area_filter_quartile={area_filter_quartile}"
            )
            raise TrackingError(
                f"No objects larger than {prefilter_cells} cells to filter on an unstructured grid ({setting})",
                details=(
                    f"Objects of {prefilter_cells} cells or fewer are dropped before the area filter with {setting}, "
                    "and no object is larger."
                ),
                suggestions=[
                    "Check that data_bin contains extreme events inside the mask",
                    "Consider lowering the extreme threshold percentile so that events are larger",
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
        # its threshold is a percentile of the census. Both outputs
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
            # ONE materialisation for both outputs, so the shared labelling pass really is
            # shared. `pin` was wrong here under streaming, where it is a no-op: the
            # census's .compute() below then ran the pass, and the caller's stage of
            # data_bin_filtered ran it a second time over the whole field (measured on the
            # 32-step fixture: 64 slice labellings against persist's 32). `stage_many`
            # writes both into one zarr in a single graph execution, and registers
            # data_bin_filtered under its own name so the caller's stage short-circuits.
            # In persist mode this is dask.persist of the pair -- exactly as before.
            if materialiser is None:
                data_bin_filtered, padded_areas = persist(data_bin_filtered, padded_areas)
            else:
                data_bin_filtered, padded_areas = materialiser.stage_many(
                    {"data_bin_filtered": data_bin_filtered, "padded_areas": padded_areas},
                    "small_object_filter",
                    preserve_chunks=True,
                )
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
        # Drop the padding inside each chunk before gathering, so the driver receives one
        # int64 per object rather than the whole (time, area_buffer) padded census.
        census = padded_areas.data
        object_areas_np, overflow = dask.compute(census[census > 0], (census[..., -1] != 0).any())
        if overflow:  # pragma: no cover
            raise TrackingError(
                "Per-slice object-area buffer overflow during small-object filtering",
                details={"area_buffer": area_buffer},
                suggestions=["Increase 'area_buffer' in filter_small_objects (structured branch)"],
            )

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
