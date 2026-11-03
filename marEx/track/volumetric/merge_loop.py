"""
MarEx Track: the 4-D split-and-merge loop.

The loop walks a ``(time, z, y, x)`` ID field one time chunk at a time, on the client,
exactly as the 3-D gridded loop does: load one chunk, take ``t-1``/``t-2`` from the
retained previous chunk at a chunk start, consolidate ``t-1`` against ``t-2``, link
``t-1`` to ``t`` by volume overlap, resolve children with several parents, consolidate
the chunk's last timestep, retain the chunk, and flush finished chunks to the output
(zarr regions when streaming; setitem plus a persist every 10 chunks otherwise).

Every slice is handled flat (``ravel()`` of ``(z, y, x)`` in C order), and every
overlap and volume is a float64 sum in a fixed order (see :mod:`.overlap` and
:mod:`.store`), so the result does not depend on the time chunking.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ...exceptions import ConfigurationError, TrackingError
from ...logging_config import get_logger
from ..region_writer import ObjectIDRegionWriter
from .config import VolumetricDims
from .geometry import VerticalGeometry
from .ids import IDAllocator
from .labelling import label_volume
from .overlap import link_mask, volume_overlaps
from .partition import ScaledCoords, partition_child
from .store import ObjectIndex, VolumeStore

logger = get_logger(__name__)

# Cap on merge-resolution passes per timestep. Each pass partitions every child with more
# than one parent; a timestep still unresolved after this many passes raises.
MAX_MERGE_ITERATIONS: int = 100

# Chunks the client can hold at once: streaming keeps the previous chunk while loading the
# current one; persist flushes every 10 chunks, so it can hold 10 finished chunks plus one.
_RETAINED_CHUNKS = {"streaming": 2, "persist": 11}
_PERSIST_FLUSH_EVERY = 10
# Float64 ``(z, y, x)`` arrays the client holds for the whole run: the vertical geometry's layer
# top, bottom, thickness and midpoint depth, and the cell volume. The loop reads the cell volumes
# and the midpoints (through the partition coordinates); the other three stay resident on the
# tracker's geometry.
_STATIC_FLOAT64_PER_VOXEL = 5


@dataclass
class MergeLedger:
    """
    Merges, splits and consolidation renames recorded by the loop.

    Attributes
    ----------
    times : list
        Time coordinate value of each merge.
    parent_ids : list of numpy.ndarray
        int32 parent IDs (at ``t-1``) of each merge.
    child_ids : list of numpy.ndarray
        int32 child IDs (at ``t``): the original child first, then the newly allocated ones.
    overlap_volumes : list of numpy.ndarray
        float64 overlap volume (m^3) of each parent with the child, aligned with ``parent_ids``.
    time_dtype : numpy.dtype, optional
        dtype of ``merge_time`` and ``split_time`` when none was recorded (datetime64[ns] by default).
    renamed : list of (int, int)
        ``(old_id, kept_id)`` for each ID consolidation removed. Resolves ledger IDs that no longer
        exist in the final field.
    split_times, split_parents, split_children : list
        One entry per split: the time of the children, the parent (at the previous timestep) and the
        children's IDs before consolidation renamed them onto the first.
    """

    times: List = field(default_factory=list)
    parent_ids: List[NDArray[np.int32]] = field(default_factory=list)
    child_ids: List[NDArray[np.int32]] = field(default_factory=list)
    overlap_volumes: List[NDArray[np.float64]] = field(default_factory=list)
    time_dtype: Optional[np.dtype] = None
    renamed: List[Tuple[int, int]] = field(default_factory=list)
    split_times: List = field(default_factory=list)
    split_parents: List[int] = field(default_factory=list)
    split_children: List[NDArray[np.int32]] = field(default_factory=list)

    def record_split(self, time_value, parent: int, children: List[int]) -> None:
        """Record that ``parent`` separated into ``children``, the kept ID first."""
        self.split_times.append(time_value)
        self.split_parents.append(parent)
        self.split_children.append(np.asarray(children, dtype=np.int32))

    def to_dataset(self) -> xr.Dataset:
        """
        Return the ledger as a padded dataset.

        Returns
        -------
        xarray.Dataset
            ``parent_IDs`` and ``overlap_volumes`` ``(merge_ID, parent_idx)``, ``child_IDs``
            ``(merge_ID, child_idx)``, ``merge_time``, ``n_parents`` and ``n_children``
            ``(merge_ID,)``; ``split_parent_IDs``, ``split_time`` and ``n_split_children``
            ``(split_ID,)`` and ``split_child_IDs`` ``(split_ID, split_child_idx)``. Padding is -1;
            counts are int32. ``_renamed_from`` and ``_renamed_to`` ``(rename_ID,)`` carry the
            renames to the clusterer, which uses and removes them.
        """
        n = len(self.parent_ids)
        max_parents = max((len(p) for p in self.parent_ids), default=1)
        max_children = max((len(c) for c in self.child_ids), default=1)
        parents = np.full((n, max_parents), -1, dtype=np.int32)
        children = np.full((n, max_children), -1, dtype=np.int32)
        overlaps = np.full((n, max_parents), -1, dtype=np.float64)
        for i, (p, c, v) in enumerate(zip(self.parent_ids, self.child_ids, self.overlap_volumes)):
            parents[i, : len(p)] = p
            children[i, : len(c)] = c
            overlaps[i, : len(v)] = v
        if n:
            merge_time = np.asarray(self.times)
        else:
            merge_time = np.array([], dtype=self.time_dtype if self.time_dtype is not None else "datetime64[ns]")
        n_splits = len(self.split_parents)
        max_split_children = max((c.size for c in self.split_children), default=1)
        split_children = np.full((n_splits, max_split_children), -1, dtype=np.int32)
        for i, c in enumerate(self.split_children):
            split_children[i, : c.size] = c
        if n_splits:
            split_time = np.asarray(self.split_times)
        else:
            split_time = np.array([], dtype=self.time_dtype if self.time_dtype is not None else "datetime64[ns]")
        renamed = np.asarray(self.renamed, dtype=np.int32).reshape(-1, 2)
        return xr.Dataset(
            {
                "parent_IDs": (("merge_ID", "parent_idx"), parents),
                "child_IDs": (("merge_ID", "child_idx"), children),
                "overlap_volumes": (("merge_ID", "parent_idx"), overlaps),
                "merge_time": ("merge_ID", merge_time),
                "n_parents": ("merge_ID", np.array([len(p) for p in self.parent_ids], dtype=np.int32)),
                "n_children": ("merge_ID", np.array([len(c) for c in self.child_ids], dtype=np.int32)),
                "split_parent_IDs": ("split_ID", np.array(self.split_parents, dtype=np.int32)),
                "split_child_IDs": (("split_ID", "split_child_idx"), split_children),
                "split_time": ("split_ID", split_time),
                "n_split_children": ("split_ID", np.array([c.size for c in self.split_children], dtype=np.int32)),
                "_renamed_from": ("rename_ID", renamed[:, 0]),
                "_renamed_to": ("rename_ID", renamed[:, 1]),
            },
            attrs={"fill_value": -1},
        )


@dataclass(frozen=True)
class SplitGeometry:
    """
    What deciding a split needs to label an object's connected pieces.

    Attributes
    ----------
    wet : numpy.ndarray
        Boolean wet mask, ``(z, y, x)``.
    thickness : numpy.ndarray
        Layer thickness in metres, ``(z, y, x)``.
    max_gap_m : float
        Largest gap a vertical link bridges, as in identification.
    wrap_x : bool
        Join across the periodic-longitude seam, as in identification.
    """

    wet: NDArray[np.bool_]
    thickness: NDArray[np.float64]
    max_gap_m: float
    wrap_x: bool

    def separates(self, parent: NDArray[np.bool_], child: NDArray[np.bool_]) -> bool:
        """
        Whether one connected piece of ``parent`` overlaps two or more connected pieces of ``child``.

        Labelling is limited to the bounding box of the two masks, over the whole ``x`` range when
        the seam joins, so its cost follows the object rather than the domain.

        Parameters
        ----------
        parent, child : numpy.ndarray
            Boolean masks, flat or ``(z, y, x)``.

        Returns
        -------
        bool
            True when the parent separated.
        """
        shape = self.wet.shape
        parent, child = parent.reshape(shape), child.reshape(shape)
        occupied = np.nonzero(parent | child)
        if occupied[0].size == 0:
            return False
        box = [slice(int(axis.min()), int(axis.max()) + 1) for axis in occupied]
        if self.wrap_x:
            box[2] = slice(None)
        box = tuple(box)
        wet, thickness = self.wet[box], self.thickness[box]
        parent_pieces, _ = label_volume(parent[box], wet, thickness, self.max_gap_m, self.wrap_x)
        child_pieces, _ = label_volume(child[box], wet, thickness, self.max_gap_m, self.wrap_x)
        both = (parent_pieces > 0) & (child_pieces > 0)
        pairs = np.unique(np.c_[parent_pieces[both], child_pieces[both]], axis=0)
        return pairs.shape[0] > 0 and bool((np.bincount(pairs[:, 0]) >= 2).any())


def merge_loop_client_bytes(n_voxels: int, time_chunk: int, compute_mode: str) -> int:
    """
    Client memory the merge loop needs for chunk buffers and static voxel geometry.

    ``retained_chunks * time_chunk * n_voxels * 4`` (int32 ID chunks) plus
    ``5 * n_voxels * 8``: the five float64 ``(z, y, x)`` arrays of the vertical geometry the
    client holds for the whole run (layer top, bottom, thickness and midpoint depth, and the
    cell volume; the loop reads the volumes and the midpoints). ``retained_chunks`` is 2 for
    ``streaming`` and 11 for ``persist``.

    Not counted: the two boolean wet masks (1 B per voxel each), the one int32 empty slice
    (4 B per voxel), the three float64 horizontal partition coordinates (per column, a factor
    ``nz`` smaller), per-merge partition buffers, and the dask graph.

    Parameters
    ----------
    n_voxels : int
        ``nz * ny * nx``.
    time_chunk : int
        Time chunk width of the ID field.
    compute_mode : {'streaming', 'persist'}
        Tracker compute mode.

    Returns
    -------
    int
        Bytes, as a Python int.
    """
    if compute_mode not in _RETAINED_CHUNKS:
        raise ConfigurationError(
            f"Unknown compute_mode '{compute_mode}' for the 4-D merge loop",
            suggestions=["Use compute_mode='persist' or compute_mode='streaming'"],
            context={"compute_mode": compute_mode, "valid": sorted(_RETAINED_CHUNKS)},
        )
    n_voxels, time_chunk = int(n_voxels), int(time_chunk)
    return _RETAINED_CHUNKS[compute_mode] * time_chunk * n_voxels * 4 + _STATIC_FLOAT64_PER_VOXEL * n_voxels * 8


def consolidate_volumes(
    prev2: NDArray[np.int32],
    prev1: NDArray[np.int32],
    store: VolumeStore,
    cell_volume: NDArray[np.float64],
    threshold: float,
    *,
    ledger: Optional[MergeLedger] = None,
    time_value=None,
    split_geometry: Optional[SplitGeometry] = None,
) -> NDArray[np.int32]:
    """
    Give one ID to the pieces of ``prev1`` that continue a single object of ``prev2``.

    A ``prev2`` object linked to several ``prev1`` objects (a split, or the pieces of an
    earlier partition) keeps the first (lowest) of those IDs; the others are renamed onto
    it and dropped from ``store``, and the kept object's volume is recomputed from the
    renamed slice, so later link decisions see the consolidated volume.

    Parameters
    ----------
    prev2, prev1 : numpy.ndarray
        int32 IDs of the slices at ``t-2`` and ``t-1``, the same shape (flat or ``(z, y, x)``).
    store : VolumeStore
        Object volumes, updated in place.
    cell_volume : numpy.ndarray
        Voxel volumes (m^3), the same number of elements.
    threshold : float
        Overlap fraction above which two objects link.
    ledger : MergeLedger, optional
        Records the renames, and any split when ``split_geometry`` is given.
    time_value : object, optional
        Time coordinate value of ``prev1``, read only with ``ledger``.
    split_geometry : SplitGeometry, optional
        Labels the pieces that decide a split. A rename is not itself a split: the pieces of one
        partition, or an object already in two pieces, also link one parent to several children.

    Returns
    -------
    numpy.ndarray
        The consolidated ``prev1`` (a new array when anything was renamed; ``prev1`` itself otherwise).
    """
    pairs, overlap = volume_overlaps(prev2.ravel(), prev1.ravel(), cell_volume.ravel())
    if pairs.shape[0] == 0:
        return prev1
    linked = pairs[link_mask(pairs, overlap, store, threshold)]
    if linked.shape[0] == 0:
        return prev1

    parent_ids, counts = np.unique(linked[:, 0], return_counts=True)
    splitting = parent_ids[counts > 1]
    if splitting.size == 0:
        return prev1

    out = prev1
    kept: List[int] = []
    renamed_onto: Dict[int, int] = {}
    candidates: List[Tuple[int, List[int]]] = []
    # Parents in ascending order; each sees the store as left by the previous one, as in 3-D.
    for parent in splitting.tolist():
        if parent not in store:
            continue
        children = linked[linked[:, 0] == parent, 1]  # ascending: pairs are sorted
        first = int(children[0])
        if first not in store:
            continue
        others = sorted({int(c) for c in children[1:].tolist() if int(c) in store and int(c) != first})
        if others:
            out = np.where(np.isin(out, others), np.int32(first), out).astype(prev1.dtype, copy=False)
            for oid in others:
                store.drop(oid)
            logger.debug(f"Consolidated {others} into {first} (parent {parent})")
            if ledger is not None:
                ledger.renamed.extend((oid, first) for oid in others)
                renamed_onto.update((oid, first) for oid in others)
                candidates.append((parent, [first, *others]))
        kept.append(first)

    if ledger is not None and split_geometry is not None:
        for parent, children in candidates:
            # A later parent can rename this parent's kept ID in turn, so follow it to the survivor.
            survivor = children[0]
            while survivor in renamed_onto:
                survivor = renamed_onto[survivor]
            if split_geometry.separates(prev2 == parent, out == survivor):
                ledger.record_split(time_value, parent, children)

    # Recompute the kept objects' volumes from the consolidated slice, with the store's own
    # summation so the value is bit-identical to identifying that object directly.
    alive = [k for k in kept if k in store]
    if alive:
        store.update_from_slice(np.where(np.isin(out, alive), out, 0), cell_volume)
    return out


def _linked_pairs(
    prev: NDArray[np.int32],
    cur: NDArray[np.int32],
    store: VolumeStore,
    cell_volume: NDArray[np.float64],
    threshold: float,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Linked ``(prev_id, cur_id)`` pairs and their overlap volumes."""
    pairs, overlap = volume_overlaps(prev, cur, cell_volume)
    keep = link_mask(pairs, overlap, store, threshold)
    return pairs[keep], overlap[keep]


def _resolve_merge(
    slice_t: NDArray[np.int32],
    index_t: ObjectIndex,
    index_prev: ObjectIndex,
    child_id: int,
    parent_ids: NDArray[np.int32],
    overlap_rows: NDArray[np.float64],
    store: VolumeStore,
    allocator: IDAllocator,
    coords: ScaledCoords,
    cell_volume: NDArray[np.float64],
) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
    """
    Partition one merged child among its parents.

    The child keeps its ID for the voxels nearest the first (lowest) parent; every other
    parent's share gets a new ID from ``allocator``. Only the child's voxels are read or
    written: the voxel sets come from the two indexes, never from a full-domain mask.

    Parameters
    ----------
    slice_t : numpy.ndarray
        Flat int32 IDs at ``t``; the child's voxels are relabelled in place.
    index_t : ObjectIndex
        Voxel index of ``slice_t``; updated in place.
    index_prev : ObjectIndex
        Voxel index of the slice at ``t-1``.
    child_id : int
        The merged child.
    parent_ids : numpy.ndarray
        Its linked parents at ``t-1``, ascending.
    overlap_rows : numpy.ndarray
        Overlap volume (m^3) of each parent with the child, aligned with ``parent_ids``.
    store : VolumeStore
        Object volumes; the pieces' volumes are set and a piece with no voxels is dropped.
    allocator : IDAllocator
        Source of the ``len(parent_ids) - 1`` new IDs (the loop passes the block it drew for the
        whole pass, which has the same ``allocate`` method).
    coords : ScaledCoords
        Scaled voxel coordinates.
    cell_volume : numpy.ndarray
        Flat voxel volumes (m^3).

    Returns
    -------
    child_ids : numpy.ndarray
        int32 ``[child_id, *new_ids]``, aligned with ``parent_ids``.
    overlap_volumes : numpy.ndarray
        float64 overlap volumes for the ledger.

    Raises
    ------
    TrackingError
        If the new IDs would exceed the int32 range. Nothing has been modified then.
    """
    parent_ids = np.asarray(parent_ids, dtype=np.int32)
    # Allocate first: an exhausted ID space raises before any field, index or store changes.
    new_ids = allocator.allocate(parent_ids.size - 1)
    child_ids = np.concatenate([np.array([child_id], dtype=np.int32), new_ids]).astype(np.int32, copy=False)

    child_vox = index_t.voxels(child_id)
    parent_vox = [index_prev.voxels(p) for p in parent_ids.tolist()]
    labels = child_ids[partition_child(child_vox, parent_vox, parent_ids, coords)]

    slice_t[child_vox] = labels
    index_t.reassign(child_id, child_vox, labels)

    # Volumes summed over the child's voxels in ascending flat order: the same per-ID order as
    # identifying the pieces directly. A piece that received no voxel is dropped.
    present = np.zeros(child_ids.size, dtype=bool)
    if child_vox.size:
        uniq, inverse = np.unique(labels, return_inverse=True)
        sums = np.bincount(inverse.ravel(), weights=cell_volume[child_vox], minlength=uniq.size)
        for oid, vol in zip(uniq.tolist(), sums.tolist()):
            store.set(oid, vol)
        present = np.isin(child_ids, uniq)
    missing = child_ids[~present].tolist()
    for oid in missing:
        store.drop(oid)
    if missing:
        logger.debug(f"Merge of child {child_id}: pieces {missing} received no voxels and were dropped")
    return child_ids, np.asarray(overlap_rows, dtype=np.float64).copy()


class _IDBlock:
    """IDs already drawn from the allocator, handed out in order through the same ``allocate`` call."""

    def __init__(self, ids: NDArray[np.int32]) -> None:
        self._ids = ids
        self._next = 0

    def allocate(self, n: int) -> NDArray[np.int32]:
        n = int(n)
        if n < 0 or self._next + n > self._ids.size:
            raise TrackingError(
                f"Merge resolution requested {n} IDs from a block with {self._ids.size - self._next} left",
                context={"requested": n, "remaining": self._ids.size - self._next},
            )
        out = self._ids[self._next : self._next + n]
        self._next += n
        return out


def _resolve_merges(
    cur: NDArray[np.int32],
    index_t: ObjectIndex,
    index_prev: ObjectIndex,
    pairs: NDArray[np.int64],
    overlap: NDArray[np.float64],
    merging: NDArray[np.int64],
    store: VolumeStore,
    allocator: IDAllocator,
    coords: ScaledCoords,
    cell_volume: NDArray[np.float64],
    ledger: MergeLedger,
    time_value,
) -> NDArray[np.int32]:
    """
    Partition every child in ``merging`` among its linked parents, in place, and record each merge.

    The new IDs of the whole pass are allocated in one request before any child is touched, so an
    exhausted ID space raises with the slice, indexes, store and ledger unchanged. Children are
    handled in ascending order and take consecutive IDs from that block, which gives the same IDs
    as allocating per child.
    """
    rows_by_child = [pairs[:, 1] == child_id for child_id in merging.tolist()]
    block = _IDBlock(allocator.allocate(sum(int(rows.sum()) - 1 for rows in rows_by_child)))
    for child_id, rows in zip(merging.tolist(), rows_by_child):
        child_ids, overlap_volumes = _resolve_merge(
            cur,
            index_t,
            index_prev,
            child_id,
            pairs[rows, 0].astype(np.int32),
            overlap[rows],
            store,
            block,
            coords,
            cell_volume,
        )
        ledger.times.append(time_value)
        ledger.parent_ids.append(pairs[rows, 0].astype(np.int32))
        ledger.child_ids.append(child_ids)
        ledger.overlap_volumes.append(overlap_volumes)
    return cur


def _resolve_timestep(
    prev: NDArray[np.int32],
    cur: NDArray[np.int32],
    store: VolumeStore,
    allocator: IDAllocator,
    coords: ScaledCoords,
    cell_volume: NDArray[np.float64],
    threshold: float,
    ledger: MergeLedger,
    time_value,
    absolute_t: int,
    max_iterations: int,
) -> NDArray[np.int32]:
    """Resolve merges at one timestep, in place, until no child has more than one linked parent."""
    iteration = 0
    index_prev: Optional[ObjectIndex] = None
    index_t: Optional[ObjectIndex] = None
    while True:
        pairs, overlap = _linked_pairs(prev, cur, store, cell_volume, threshold)
        merging = np.empty(0, dtype=np.int64)
        if pairs.shape[0]:
            children, counts = np.unique(pairs[:, 1], return_counts=True)
            merging = children[counts > 1]
        if merging.size == 0:
            if iteration:
                logger.debug(f"Timestep {absolute_t}: merges resolved in {iteration} iterations")
            return cur
        if iteration >= max_iterations:
            raise TrackingError(
                f"Resolving merges at timestep {absolute_t} did not converge after {max_iterations} iterations",
                details=f"Children still overlapping more than one parent: {merging.tolist()[:20]}",
                suggestions=["Raise overlap_threshold so fewer weak overlaps count as merges"],
                context={"timestep": absolute_t, "unresolved_children": merging.tolist()[:20]},
            )
        if index_t is None:
            # Built once per timestep: t-1 does not change here, and the t index is kept
            # current by each partition.
            index_prev = ObjectIndex.from_slice(prev)
            index_t = ObjectIndex.from_slice(cur)
        cur = _resolve_merges(
            cur, index_t, index_prev, pairs, overlap, merging, store, allocator, coords, cell_volume, ledger, time_value
        )
        iteration += 1


def _check_field(id_field: xr.DataArray, dims: VolumetricDims) -> None:
    if tuple(id_field.dims) != dims.internal_order():
        raise ConfigurationError(
            f"The ID field must have dimensions {dims.internal_order()}, got {tuple(id_field.dims)}",
            suggestions=["Transpose the field to (time, z, y, x) before the merge loop"],
        )
    if id_field.chunks is None:
        raise ConfigurationError(
            "The ID field must be dask-backed and chunked in time",
            suggestions=[f"Chunk the field, e.g. id_field.chunk({{'{dims.time}': 30}})"],
        )
    for axis, name in enumerate(dims.internal_order()[1:], start=1):
        if len(id_field.chunks[axis]) != 1:
            raise ConfigurationError(
                f"The ID field must be a single chunk along '{name}', got {len(id_field.chunks[axis])} chunks",
                details="Labelling and overlaps are global in space, so depth and the horizontal stay whole",
                suggestions=[f"Rechunk with {{'{name}': -1}}"],
            )


def split_and_merge_volumetric(
    id_field: xr.DataArray,
    store: VolumeStore,
    allocator: IDAllocator,
    geom: VerticalGeometry,
    coords: ScaledCoords,
    overlap_threshold: float,
    dims: VolumetricDims,
    *,
    materialiser,
    id_field_path: Optional[str],
    max_iterations: int = MAX_MERGE_ITERATIONS,
    max_gap_m: Optional[float] = None,
    wrap_x: Optional[bool] = None,
) -> Tuple[xr.DataArray, VolumeStore, MergeLedger]:
    """
    Consolidate splits and resolve merges over a 4-D ID field.

    Parameters
    ----------
    id_field : xarray.DataArray
        int32 ``(time, z, y, x)`` IDs, dask-backed, chunked in time only. IDs must increase
        with time (as from :func:`identify_volumes`).
    store : VolumeStore
        Volume of every ID in ``id_field``; updated in place and returned.
    allocator : IDAllocator
        Source of IDs created by merges.
    geom : VerticalGeometry
        Vertical geometry; its ``cell_volume`` weights every overlap.
    coords : ScaledCoords
        Scaled voxel coordinates for partitioning merged children.
    overlap_threshold : float
        Two objects link when their shared volume is at least this fraction of the smaller one.
    dims : VolumetricDims
        Dimension names.
    materialiser : Materialiser or None
        ``streaming`` writes finished chunks to a zarr store; any other mode (or None)
        keeps the field persisted in cluster memory.
    id_field_path : str, optional
        Zarr store for the streamed field. Defaults to ``volume_id_field_merged.zarr`` in the
        materialiser's staging directory. Ignored outside streaming.
    max_iterations : int, default=MAX_MERGE_ITERATIONS
        Merge-resolution passes allowed per timestep before raising.
    max_gap_m : float, optional
        The vertical gap identification bridged. With ``wrap_x``, it lets the loop label pieces
        and record splits. Without both, no split is recorded.
    wrap_x : bool, optional
        Whether identification joined across the periodic-longitude seam.

    Returns
    -------
    id_field : xarray.DataArray
        The processed field (lazy, zarr-backed when streaming; persisted otherwise).
    store : VolumeStore
        The updated store.
    ledger : MergeLedger
        Merge events.

    Raises
    ------
    TrackingError
        If a timestep still has merging children after ``max_iterations`` passes.
    """
    _check_field(id_field, dims)
    timedim = dims.time
    cell_volume = np.ascontiguousarray(geom.cell_volume, dtype=np.float64).ravel()
    spatial_shape = tuple(id_field.shape[1:])
    n_voxels = int(np.prod(spatial_shape))
    if cell_volume.size != n_voxels:
        raise ConfigurationError(
            f"cell_volume has {cell_volume.size} voxels, the ID field {n_voxels} per timestep",
            context={"cell_volume": geom.cell_volume.shape, "id_field": spatial_shape},
        )

    split_geometry = None
    if max_gap_m is not None and wrap_x is not None:
        split_geometry = SplitGeometry(geom.wet, geom.thickness, float(max_gap_m), bool(wrap_x))

    ledger = MergeLedger(time_dtype=id_field[timedim].dtype)
    time_values = id_field[timedim].values

    streaming = materialiser is not None and materialiser.is_streaming
    if streaming:
        if id_field_path is None:
            id_field_path = str(Path(materialiser.staging_dir) / "volume_id_field_merged.zarr")
        writer: Optional[ObjectIDRegionWriter] = ObjectIDRegionWriter(id_field, id_field_path, timedim)
    else:
        writer = None
        id_field = id_field.persist()

    zeros = np.zeros(n_voxels, dtype=np.int32)
    chunk_sizes = list(id_field.chunks[0])
    chunk_bounds = np.cumsum([0] + chunk_sizes)
    n_chunks = len(chunk_sizes)
    # (start, end, (n, z, y, x) values, (n, voxels) flat view of the same buffer)
    updated: List[Tuple[int, int, NDArray[np.int32], NDArray[np.int32]]] = []

    def flush(entries) -> None:
        nonlocal id_field
        for start, end, values, _ in entries:
            if writer is not None:
                writer.write(start, end, id_field.isel({timedim: slice(start, end)}).copy(data=values))
            else:
                id_field[{timedim: slice(start, end)}] = values

    for chunk_idx in range(n_chunks):
        chunk_start, chunk_end = int(chunk_bounds[chunk_idx]), int(chunk_bounds[chunk_idx + 1])
        n = chunk_end - chunk_start
        values = np.ascontiguousarray(id_field.isel({timedim: slice(chunk_start, chunk_end)}).values, dtype=np.int32)
        flat = values.reshape(n, n_voxels)  # a view: edits to flat land in values

        for relative_t in range(n):
            absolute_t = chunk_start + relative_t
            if relative_t > 1:
                prev2, prev1 = flat[relative_t - 2], flat[relative_t - 1]
            elif relative_t == 1:
                prev1 = flat[0]
                prev2 = updated[-1][3][-1] if updated else zeros
            else:
                if updated:
                    last = updated[-1][3]
                    prev2 = last[-2] if last.shape[0] >= 2 else zeros
                    prev1 = last[-1]
                else:
                    prev2 = prev1 = zeros

            # Consolidate t-1 against t-2 whenever t-1 is in this chunk.
            if relative_t > 0:
                flat[relative_t - 1] = consolidate_volumes(
                    prev2,
                    prev1,
                    store,
                    cell_volume,
                    overlap_threshold,
                    ledger=ledger,
                    time_value=time_values[absolute_t - 1],
                    split_geometry=split_geometry,
                )
                prev1 = flat[relative_t - 1]

            flat[relative_t] = _resolve_timestep(
                prev1,
                flat[relative_t],
                store,
                allocator,
                coords,
                cell_volume,
                overlap_threshold,
                ledger,
                time_values[absolute_t],
                absolute_t,
                max_iterations,
            )

        # Consolidate the chunk's last timestep against the one before it; a one-step chunk
        # takes that reference from the previous chunk, so the result is chunking-invariant.
        if n >= 2 or updated:
            reference = flat[-2] if n >= 2 else updated[-1][3][-1]
            flat[-1] = consolidate_volumes(
                reference,
                flat[-1],
                store,
                cell_volume,
                overlap_threshold,
                ledger=ledger,
                time_value=time_values[chunk_end - 1],
                split_geometry=split_geometry,
            )

        updated.append((chunk_start, chunk_end, values, flat))

        if chunk_idx % 10 == 0:
            logger.info(f"4-D split/merge: chunk {chunk_idx} of {n_chunks}")

        # Flush finished chunks; the last one is always retained as the next chunk's t-1/t-2.
        if len(updated) > 1 and (writer is not None or chunk_idx % _PERSIST_FLUSH_EVERY == 0):
            flush(updated[:-1])
            updated = updated[-1:]
            if writer is None:
                id_field = id_field.persist()

        # Drop every name that can still view an older chunk's buffer.
        values = flat = prev1 = prev2 = reference = last = None

    flush(updated)
    updated = []
    if writer is not None:
        id_field = writer.finalise()
    else:
        id_field = id_field.persist()
    return id_field, store, ledger


def final_links(
    id_field: xr.DataArray,
    store: VolumeStore,
    cell_volume: NDArray[np.float64],
    threshold: float,
    timedim: str,
) -> NDArray[np.int64]:
    """
    Every linked ``(id_t, id_t+1)`` pair of a finished ID field.

    One client-side pass, one time chunk at a time, carrying the last slice of each chunk
    into the next. Volumes come from ``store``, so they are the consolidated ones.

    Parameters
    ----------
    id_field : xarray.DataArray
        ``(time, z, y, x)`` IDs, numpy- or dask-backed.
    store : VolumeStore
        Object volumes.
    cell_volume : numpy.ndarray
        Voxel volumes (m^3), ``(z, y, x)``.
    threshold : float
        Overlap fraction above which two objects link.
    timedim : str
        Name of the time dimension (the first dimension).

    Returns
    -------
    numpy.ndarray
        ``(n, 2)`` int64 pairs, ordered by time and then lexicographically.
    """
    if id_field.dims[0] != timedim:
        raise ConfigurationError(f"final_links expects '{timedim}' first, got dims {tuple(id_field.dims)}")
    cv = np.ascontiguousarray(cell_volume, dtype=np.float64).ravel()
    n_voxels = cv.size
    nt = id_field.sizes[timedim]
    chunk_sizes = list(id_field.chunks[0]) if id_field.chunks is not None else [nt]
    out: List[NDArray[np.int64]] = []
    previous: Optional[NDArray] = None
    start = 0
    for size in chunk_sizes:
        values = np.asarray(id_field.isel({timedim: slice(start, start + size)}).values).reshape(size, n_voxels)
        for t in range(size):
            if previous is not None:
                pairs, _ = _linked_pairs(previous, values[t], store, cv, threshold)
                if pairs.shape[0]:
                    out.append(pairs)
            previous = values[t]
        start += size
    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(out, axis=0).astype(np.int64, copy=False)
