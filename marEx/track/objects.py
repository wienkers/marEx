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
from scipy.ndimage import label as scipy_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import regionprops_table

from ..exceptions import ConfigurationError
from ..logging_config import get_logger

logger = get_logger(__name__)


def _anchor(obj, label, materialiser):
    """Stage `obj` -- it has two or more consumers. Plain persist when unsupplied.

    The `materialiser is None` default keeps every existing caller -- including the
    unstructured path, which Phase 4 does not touch -- on exactly the previous behaviour.
    """
    if materialiser is None:
        return obj.persist()
    return materialiser.stage(obj, label, preserve_chunks=True)


# 8-connectivity structuring element for per-2D-slice connected-component labelling.
_EIGHT_CONNECTIVITY = np.ones((3, 3), dtype=int)


def _merge_lon_seam(labels: NDArray[np.int32], n_labels: int) -> NDArray[np.int32]:
    """
    Union connected-component labels that touch across the periodic-longitude seam.

    ``scipy.ndimage.label`` treats the array as non-periodic, so an object straddling
    the antimeridian is split into a left-edge (column 0) and right-edge (column -1)
    component. This re-joins them with full 8-connectivity across the seam (column 0 of
    row r connects to column -1 of rows r-1, r, r+1), matching the behaviour of
    ``dask_image.ndmeasure.label(..., wrap_axes=(2,))``. Returns ``labels`` with the
    relevant components relabelled to a shared root (label values may become
    non-contiguous; callers using ``np.bincount`` / compaction are unaffected).
    """
    if n_labels == 0:
        return labels
    left = labels[:, 0]
    right = labels[:, -1]
    parent = np.arange(n_labels + 1)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    same = (left > 0) & (right > 0)
    for a, b in zip(left[same], right[same]):
        union(a, b)
    diag_down = (left[1:] > 0) & (right[:-1] > 0)
    for a, b in zip(left[1:][diag_down], right[:-1][diag_down]):
        union(a, b)
    diag_up = (left[:-1] > 0) & (right[1:] > 0)
    for a, b in zip(left[:-1][diag_up], right[1:][diag_up]):
        union(a, b)

    roots = np.array([find(i) for i in range(n_labels + 1)], dtype=labels.dtype)
    return roots[labels]


def identify_objects(
    data_bin: xr.DataArray,
    time_connectivity: bool,
    unstructured_grid: bool,
    mask: xr.DataArray,
    neighbours_int: Optional[xr.DataArray],
    xdim: str,
    regional_mode: bool,
    *,
    materialiser=None,
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

                # Find connected components
                valid_mask = (neighbours_int != -1) & arr[i][neighbours_int]
                row_ind, col_ind = np.where(valid_mask)
                row_ind = row_ind.astype(np.int32)
                col_ind = col_ind.astype(np.int32)

                # Map to compact indices for the graph algorithm. np.where returns
                # true_indices already sorted, so the old-index -> compact-index map is a
                # binary search rather than a Python dict walked once per graph edge (up to
                # 3 x ncells iterations inside the dask task, which dominated unstructured
                # labelling on ICON-size grids -- review finding 6.5).
                edge_rows = neighbours_int[row_ind, col_ind]
                # The neighbour end is True by construction of valid_mask; only the cell end
                # needs testing, which is exactly what the dict membership check did.
                keep = arr[i][col_ind]
                mapped_row_ind = np.searchsorted(true_indices, edge_rows[keep])
                mapped_col_ind = np.searchsorted(true_indices, col_ind[keep])

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

    elif time_connectivity:
        # Structured grid, 3D (space & time) labelling -- IDs unique across time.
        # Genuine cross-time connectivity is required, so dask_image's global relabel is used.
        neighbours = np.zeros((3, 3, 3))
        neighbours[:, :, :] = 1  # +-1 in time, _and also diagonal in time_ -- i.e. edges can touch

        if regional_mode:
            object_id_field, N_objects = label(data_bin, structure=neighbours)
        else:
            object_id_field, N_objects = label(data_bin, structure=neighbours, wrap_axes=(2,))  # Wrap in x-direction !

        def _wrap(field):
            return (
                xr.DataArray(
                    field,
                    coords=data_bin.coords,
                    dims=data_bin.dims,
                    attrs=data_bin.attrs,
                )
                .rename("ID_field")
                .astype(np.int32)
            )

        if materialiser is not None and materialiser.is_streaming:
            # Stage the whole labelled field to disk instead of pinning it in RAM. This
            # is the branch reached only when allow_merging=False and the grid is
            # structured -- the 13th pin site the original Phase 4 profile missed.
            #
            # Staging needs the wrapped DataArray, so wrap first. N_objects is then read
            # back off the STAGED store rather than from `label`'s second return value:
            # that value is a separate node over the same labelling graph, and with the
            # field on disk rather than pinned, computing it would re-run the entire
            # 3D relabel. dask_image.ndmeasure.label numbers objects 1..N contiguously,
            # so max(labels) == N, and an empty field gives 0 == 0.
            object_id_field = _anchor(_wrap(object_id_field), "object_id_field", materialiser)
            N_objects = int(object_id_field.max().compute().item())
        else:
            # persist mode: pin the raw label output and its count together so the one
            # labelling pass serves both, exactly as before Phase 4.
            if materialiser is None:
                results = persist(object_id_field, N_objects)
            else:
                results = materialiser.pin(object_id_field, N_objects)
            object_id_field, N_objects = results
            N_objects = N_objects.compute()
            object_id_field = _wrap(object_id_field)

    else:
        # Structured grid, 2D-per-time-slice labelling (no time connectivity).
        #
        # The connectivity is purely within each 2D slice, so dask_image.ndmeasure.label's
        # cross-block adjacency machinery (label_adjacency_graph) is pure waste here -- and its
        # graph CONSTRUCTION is single-threaded and ~cubic in the number of time-chunks
        # (e.g. ~30-40 min to merely build the graph for a 9282-day, 372-chunk run, before any
        # compute). Instead label each 2D slice independently with scipy.ndimage.label (8-conn,
        # periodic-longitude seam merge except in regional mode) and offset by the running
        # cumulative object count so IDs are globally unique and monotonically increasing with
        # time -- matching the old dask_image(wrap_axes=(2,), time-only-2D) semantics. This is
        # embarrassingly parallel with a tiny task graph (mirrors the unstructured branch).
        # Structured data is ordered (time, ydim, xdim); xdim is known.
        non_x_dims = [d for d in data_bin.dims if d != xdim]
        timedim, ydim = non_x_dims[0], non_x_dims[1]

        def label_slice_2d(binary_2d: NDArray[np.bool_]) -> NDArray[np.int32]:
            """Label one 2D slice; compact local labels to 1..k (0 = background)."""
            labels, n_labels = scipy_label(binary_2d, structure=_EIGHT_CONNECTIVITY)
            if not regional_mode:
                labels = _merge_lon_seam(labels, n_labels)
            if labels.max() == 0:
                return labels.astype(np.int32)
            # Compact (seam-merge leaves gaps) so per-slice max == object count -> valid offsets.
            unique = np.unique(labels)
            remap = np.zeros(int(unique.max()) + 1, dtype=np.int32)
            remap[unique] = np.arange(len(unique), dtype=np.int32)  # background 0 -> 0
            return remap[labels]

        local_ids = xr.apply_ufunc(
            label_slice_2d,
            data_bin,
            input_core_dims=[[ydim, xdim]],
            output_core_dims=[[ydim, xdim]],
            output_dtypes=[np.int32],
            vectorize=True,
            dask="parallelized",
        )

        # Per-slice object counts -> cumulative offset (slice t gets the sum of all earlier counts).
        per_slice_counts = local_ids.max(dim=[ydim, xdim])
        offsets = per_slice_counts.cumsum(timedim).shift({timedim: 1}, fill_value=0)
        object_id_field = xr.where(local_ids > 0, local_ids + offsets, 0).rename("ID_field").astype(np.int32)

        # _anchor, not _hold: this array is read TWICE -- by the .max().compute() on the
        # next line and by the caller via the return. Leaving it unanchored in streaming
        # mode would re-run the whole labelling graph for the second consumer.
        object_id_field = _anchor(object_id_field, "object_id_field", materialiser)
        N_objects = int(object_id_field.max().compute().item())

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

    # Check if object is near either edge of x dimension. Scale the margin so it never
    # exceeds a quarter of the grid width: a fixed 100-column margin on a grid with <=200
    # longitude points marks every object "near both edges" and corrupts mid-domain centroids.
    edge_margin = min(100, binary_mask.shape[1] // 4)
    near_left_BC = np.any(binary_mask[:, :edge_margin])
    near_right_BC = np.any(binary_mask[:, -edge_margin:])

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


def calculate_partitioned_child_properties(
    y_idx: NDArray[np.intp],
    x_idx: NDArray[np.intp],
    new_labels: NDArray[np.int32],
    Nx: int,
    regional_mode: bool,
) -> xr.Dataset:
    """Area + wrap-aware centroid for each label in ``new_labels``, from the child blob's pixels.

    Reproduces ``calculate_object_properties(..., properties=["area", "centroid"])`` on a structured
    grid -- unweighted pixel-count area and unweighted pixel-coordinate centroid, with the same
    antimeridian-wrap convention as :func:`calculate_centroid` -- but computed only from the supplied
    pixels (global ``y_idx``/``x_idx`` coordinates labelled by ``new_labels``) rather than a
    full-slice ``regionprops_table``. Used per merge in ``split_and_merge_objects`` where the freshly
    minted child IDs exist only within the partitioned child blob, so their full-slice properties
    equal these local ones.

    Parameters
    ----------
    y_idx, x_idx : numpy.ndarray
        Global pixel coordinates (rows, columns) of the partitioned child blob.
    new_labels : numpy.ndarray
        Object ID assigned to each pixel (same length and order as ``y_idx``/``x_idx``).
    Nx : int
        Global number of longitude points (slice width); used for the periodic x-wrap.
    regional_mode : bool
        If True, skip the antimeridian adjustment (raw means), matching :func:`calculate_centroid`.

    Returns
    -------
    xarray.Dataset
        ``area`` (dim ``ID``) and ``centroid`` (dims ``component``, ``ID``); ``ID`` is the index
        coordinate. Matches the structure returned by :func:`calculate_object_properties`.
    """
    labels = np.unique(new_labels)
    labels = labels[labels > 0]  # regionprops ignores background (0)
    n = labels.size
    areas = np.empty(n, dtype=np.int64)
    cy = np.empty(n, dtype=np.float64)
    cx = np.empty(n, dtype=np.float64)
    half_Nx = Nx // 2
    for i, lab in enumerate(labels):
        sel = new_labels == lab
        ys = y_idx[sel]
        xs = x_idx[sel]
        areas[i] = ys.size
        cy[i] = ys.mean()
        # Match calculate_centroid: only objects near BOTH x-edges get the wrap adjustment.
        edge_margin = min(100, Nx // 4)
        if not regional_mode and np.any(xs < edge_margin) and np.any(xs >= Nx - edge_margin):
            xs_adj = xs.astype(np.float64)
            xs_adj[xs > half_Nx] -= Nx
            x_centroid = xs_adj.mean()
            if x_centroid < 0:  # keep the centroid in [0, Nx)
                x_centroid += Nx
            cx[i] = x_centroid
        else:
            cx[i] = xs.mean()
    return xr.Dataset(
        {
            "area": ("ID", areas),
            "centroid": (("component", "ID"), np.stack([cy, cx])),
        },
        coords={"ID": labels.astype(np.int32)},
    )


class ObjectPropsStore:
    """O(1) in-memory store of per-object ``area`` + ``centroid``, keyed by integer ID.

    Replaces the per-merge xarray ``Dataset`` mutations (``xr.concat`` / ``.sel`` / ``.loc`` /
    ``.drop_sel``) in ``split_and_merge_objects``, whose cost grew O(N^2) as the object set
    accumulated over the run. Backed by plain dicts, so every lookup / insert / update / delete is
    O(1) (O(k) for a k-ID batch lookup). Build from / convert to the xarray ``Dataset`` that
    :func:`calculate_object_properties` produces at the function boundaries
    (:meth:`from_dataset` / :meth:`to_dataset`).

    Holds ``area`` (pixel count) and ``centroid`` (y, x) -- the structured-grid properties used by
    the merge/split loop and the overlap helpers (``enforce_overlap_threshold``,
    ``consolidate_object_ids``).
    """

    __slots__ = ("_area", "_cy", "_cx", "_sorted_ids")

    def __init__(self, area=None, cy=None, cx=None):
        """Create a store, optionally seeded with ``area``/``cy``/``cx`` dicts (keyed by int ID)."""
        self._area = {} if area is None else area
        self._cy = {} if cy is None else cy
        self._cx = {} if cx is None else cx
        self._sorted_ids = None  # lazily built cache for contains_many; invalidated on mutation

    @classmethod
    def from_dataset(cls, object_props: xr.Dataset) -> "ObjectPropsStore":
        """Build a store from an ``area``/``centroid`` Dataset (ID-indexed, centroid dims component,ID)."""
        ids = object_props["ID"].values
        area = object_props["area"].values
        centroid = object_props["centroid"]
        cy = centroid.isel(component=0).values
        cx = centroid.isel(component=1).values
        return cls(
            area={int(i): a for i, a in zip(ids, area)},
            cy={int(i): float(v) for i, v in zip(ids, cy)},
            cx={int(i): float(v) for i, v in zip(ids, cx)},
        )

    def to_dataset(self) -> xr.Dataset:
        """Convert back to the ID-indexed ``area`` + ``centroid`` Dataset (IDs in sorted order)."""
        ids = np.array(sorted(self._area), dtype=np.int32)
        area = np.array([self._area[int(i)] for i in ids])
        cy = np.array([self._cy[int(i)] for i in ids], dtype=np.float64)
        cx = np.array([self._cx[int(i)] for i in ids], dtype=np.float64)
        return xr.Dataset(
            {
                "area": ("ID", area),
                "centroid": (("component", "ID"), np.stack([cy, cx])),
            },
            coords={"ID": ids},
        )

    def __contains__(self, object_id) -> bool:
        """Whether ``object_id`` is currently present in the store."""
        return int(object_id) in self._area

    def contains_many(self, ids) -> NDArray[np.bool_]:
        """Vectorised membership test: equivalent to ``[i in self for i in ids]``, elementwise.

        Callers filter multi-million-row overlap lists with this; a Python-level ``in`` per
        row dominated ``enforce_overlap_threshold`` at scale (review finding 5.14). The
        sorted key array is cached and invalidated by :meth:`set` / :meth:`drop`.
        """
        query = np.asarray(ids).astype(np.int64)
        if self._sorted_ids is None:
            self._sorted_ids = np.fromiter(self._area.keys(), dtype=np.int64, count=len(self._area))
            self._sorted_ids.sort()
        known = self._sorted_ids
        if known.size == 0 or query.size == 0:
            return np.zeros(query.shape, dtype=bool)
        idx = np.searchsorted(known, query)
        np.clip(idx, 0, known.size - 1, out=idx)
        return known[idx] == query

    def max_id(self) -> int:
        """Largest current object ID (0 if the store is empty)."""
        return max(self._area) if self._area else 0

    def area(self, object_id):
        """Area (pixel count) of a single object ID."""
        return self._area[int(object_id)]

    def centroid(self, object_id) -> NDArray[np.float64]:
        """(y, x) centroid of a single object ID."""
        oid = int(object_id)
        return np.array([self._cy[oid], self._cx[oid]])

    def areas(self, ids) -> NDArray:
        """Areas for a sequence of IDs, order-preserving (like ``object_props['area'].sel(ID=ids)``)."""
        return np.array([self._area[int(i)] for i in ids])

    def centroids(self, ids) -> NDArray[np.float64]:
        """Return an (k, 2) array of (y, x) centroids, matching ``object_props.sel(ID=ids).centroid.values.T``."""
        return np.array([[self._cy[int(i)], self._cx[int(i)]] for i in ids], dtype=np.float64)

    def set(self, object_id, area, cy, cx) -> None:
        """Insert or update the area + (y, x) centroid for ``object_id``."""
        oid = int(object_id)
        if oid not in self._area:
            self._sorted_ids = None  # key set changed
        self._area[oid] = area
        self._cy[oid] = float(cy)
        self._cx[oid] = float(cx)

    def drop(self, object_id) -> None:
        """Remove ``object_id`` from the store (no-op if absent)."""
        oid = int(object_id)
        if oid in self._area:
            self._sorted_ids = None  # key set changed
        self._area.pop(oid, None)
        self._cy.pop(oid, None)
        self._cx.pop(oid, None)


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

        # Per-timestep property buffer. The estimate is 4x the mean number of objects per
        # timestep, plus slack. The old floor was `max_ID`, i.e. the total object count over
        # the whole run, which made these buffers O(time x total objects) and is consistent
        # with the recorded unstructured-tracking OOM (review finding 6.3). The floor is now
        # a constant, so it still gives small runs generous headroom (where max_ID <= 100 it
        # is exactly the old value) without scaling with the length of the run. Overflow is
        # not silent: the `result[0, :n_ids] = areas` fill below raises if a timestep holds
        # more objects than the buffer.
        ID_buffer_size = max(int(max_ID / time_steps) * 4 + 2, min(max_ID, 100))

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
            # Ask regionprops for the bounding box as well, so the antimeridian test below
            # is a column comparison rather than a full-slice mask per object.
            wrap_check = check_centroids and not regional_mode
            props_requested = list(properties)
            bbox_added = wrap_check and "bbox" not in props_requested
            if bbox_added:
                props_requested.append("bbox")

            # Use regionprops_table for standard properties
            props_slice = regionprops_table(ids, properties=props_requested)

            # Handle centroid calculation for objects that wrap around edges
            if wrap_check and len(props_slice["label"]) > 0:
                # Only objects whose bounding box reaches within edge_margin of BOTH x-edges
                # can need the wrap adjustment; calculate_centroid returns the regionprops
                # centroid unchanged for every other object. The bbox is tight, so
                # "has a pixel left of edge_margin" is exactly "bbox min column < margin"
                # -- building an `ids == ID` mask for every object in every timestep just to
                # discover that was the cost here (review finding 6.4).
                nx = ids.shape[1]
                edge_margin = min(100, nx // 4)
                near_left = np.asarray(props_slice["bbox-1"]) < edge_margin
                near_right = np.asarray(props_slice["bbox-3"]) > nx - edge_margin
                wrapping = np.nonzero(near_left & near_right)[0]

                if wrapping.size > 0:
                    centroid_y = np.asarray(props_slice["centroid-0"], dtype=np.float64)
                    centroid_x = np.asarray(props_slice["centroid-1"], dtype=np.float64)
                    for ID_idx in wrapping:
                        binary_mask = ids == props_slice["label"][ID_idx]
                        centroid_y[ID_idx], centroid_x[ID_idx] = calculate_centroid(
                            binary_mask, regional_mode, (centroid_y[ID_idx], centroid_x[ID_idx])
                        )
                    props_slice["centroid-0"] = centroid_y
                    props_slice["centroid-1"] = centroid_x

            if bbox_added:
                for bbox_key in ("bbox-0", "bbox-1", "bbox-2", "bbox-3"):
                    props_slice.pop(bbox_key, None)

            return props_slice

        # Process single time or multiple times
        # If time dimension doesn't exist, treat as single time slice
        if timedim not in object_id_field.dims or object_id_field.sizes[timedim] == 1:
            # Drop a size-1 time axis so object_properties_chunk receives a 2D (ny, nx) slice;
            # a 3D (1, ny, nx) array makes regionprops emit a spurious third centroid component.
            field_slice = object_id_field.isel({timedim: 0}) if timedim in object_id_field.dims else object_id_field
            object_props = object_properties_chunk(field_slice.values)
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
