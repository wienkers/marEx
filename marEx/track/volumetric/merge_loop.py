"""
MarEx Track: the 4-D split-and-merge loop.

The client walks a ``(time, z, y, x)`` ID field one chunk at a time. It links each
slice to its settled predecessor, resolves merges, consolidates splits, and emits
the boundary's final links. Finished chunks are written to Zarr regions in
streaming mode or assigned and persisted every 10 chunks in persist mode.

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
from .overlap import OverlapTable
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
    time_encoding : dict, optional
        CF calendar and units of the source time coordinate for empty cftime columns.
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
    time_encoding: Dict[str, str] = field(default_factory=dict)

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
        n_splits = len(self.split_parents)
        max_split_children = max((c.size for c in self.split_children), default=1)
        split_children = np.full((n_splits, max_split_children), -1, dtype=np.int32)
        for i, c in enumerate(self.split_children):
            split_children[i, : c.size] = c
        first_time = next(iter(self.times or self.split_times), None)
        calendar = self.time_encoding.get("calendar", getattr(first_time, "calendar", None))
        empty_dtype = object if calendar is not None else self.time_dtype or "datetime64[ns]"
        merge_time = np.asarray(self.times) if n else np.empty(0, dtype=empty_dtype)
        split_time = np.asarray(self.split_times) if n_splits else np.empty(0, dtype=empty_dtype)
        renamed = np.asarray(self.renamed, dtype=np.int32).reshape(-1, 2)
        ds = xr.Dataset(
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
        if calendar is not None:
            for name in ("merge_time", "split_time"):
                if not ds[name].size:
                    # Xarray cannot infer a CF datetime from an empty object array.
                    ds[name].attrs.update(calendar=calendar, units=self.time_encoding.get("units", "days since 1970-01-01"))
                    ds[name].encoding["dtype"] = "int64"
        return ds


@dataclass(frozen=True)
class SplitGeometry:
    """
    What deciding a split needs to label connected pieces.

    Attributes
    ----------
    wet : numpy.ndarray
        Boolean wet mask, ``(z, y, x)``.
    thickness : numpy.ndarray
        Layer thickness in metres, ``(z,)`` or ``(z, y, x)``.
    max_gap_m : float
        Largest gap a vertical link bridges, as in identification.
    wrap_x : bool
        Join across the periodic-longitude seam, as in identification.
    """

    wet: NDArray[np.bool_]
    thickness: NDArray[np.float64]
    max_gap_m: float
    wrap_x: bool

    def water_pieces(self, active: NDArray[np.bool_]) -> NDArray[np.int32]:
        """
        Label the connected pieces of all active water at one timestep, over the whole domain.

        A bounding box would cut the links outside it and invent separations, so this labels every
        voxel. One labelling serves every candidate of a timestep, because renames leave the active
        voxels unchanged.

        Parameters
        ----------
        active : numpy.ndarray
            Boolean mask of every object's voxels, flat or ``(z, y, x)``.

        Returns
        -------
        numpy.ndarray
            int32 labels, ``(z, y, x)``, 0 on background.
        """
        return label_volume(active.reshape(self.wet.shape), self.wet, self.thickness, self.max_gap_m, self.wrap_x)[0]

    def separates(self, parent: NDArray[np.bool_], child: NDArray[np.bool_], water: NDArray[np.int32]) -> bool:
        """
        Whether one connected piece of ``parent`` overlaps ``child`` in two or more pieces of ``water``.

        A connected piece of the child lies in one piece of water, so this counts the same thing as
        the child's pieces in different water and needs no labelling of the child. Pieces joined only
        by another object's voxels share a piece of water and do not count.

        The parent's labelling is limited to its bounding box, over the whole ``x`` range when the
        seam joins, so its cost follows the object rather than the domain.

        Parameters
        ----------
        parent, child : numpy.ndarray
            Boolean masks, flat or ``(z, y, x)``.
        water : numpy.ndarray
            Labels from :meth:`water_pieces` at the child's timestep.

        Returns
        -------
        bool
            True when the parent separated.
        """
        shape = self.wet.shape
        parent, child = parent.reshape(shape), child.reshape(shape)
        occupied = np.nonzero(parent)
        if occupied[0].size == 0:
            return False
        box = [slice(int(axis.min()), int(axis.max()) + 1) for axis in occupied]
        if self.wrap_x:
            box[2] = slice(None)
        box = tuple(box)
        overlap = child[box] & (water[box] > 0)
        if np.unique(water[box][overlap]).size < 2:
            return False
        thickness_box = box[:1] if self.thickness.ndim == 1 else box
        parent_pieces, _ = label_volume(parent[box], self.wet[box], self.thickness[thickness_box], self.max_gap_m, self.wrap_x)
        overlap &= parent_pieces > 0
        pairs = np.unique(np.c_[parent_pieces[overlap], water[box][overlap]], axis=0)
        return pairs.shape[0] > 0 and bool((np.bincount(pairs[:, 0]) >= 2).any())


class _WaterComponents:
    """Pristine identification labels and voxel ownership for the objects that were edited."""

    def __init__(self, labels: NDArray[np.int32]) -> None:
        self.labels = labels
        self._protected = False
        self._sizes: Dict[int, int] = {}
        self._ownership: Dict[int, Dict[int, int]] = {}

    def preserve(self) -> None:
        """Keep the pre-edit labels before the active slice is overwritten."""
        if not self._protected:
            self.labels = self.labels.copy()
            self._protected = True

    def partition(self, voxels: NDArray[np.int64], ids: NDArray, counts: NDArray) -> None:
        """Record each piece's share of a pristine component during merge resolution."""
        # Partitions precede consolidation, so every child here has a single origin.
        origin = int(self.labels[voxels[0]])
        self._sizes.setdefault(origin, voxels.size)
        self._ownership.update({oid: {origin: count} for oid, count in zip(ids.tolist(), counts.tolist())})

    def consolidate(self, kept: int, removed: List[int], index: ObjectIndex) -> None:
        """Combine integer ownership counts while renaming onto the survivor."""
        combined: Dict[int, int] = {}
        for oid in [kept, *removed]:
            membership = self._ownership.pop(oid, None)
            if membership is None:
                count = index.voxels(oid).size
                self._sizes.setdefault(oid, count)
                membership = {oid: count}
            for origin, count in membership.items():
                combined[origin] = combined.get(origin, 0) + count
        self._ownership[kept] = combined

    def certifies_split(self, parent: int, intersection: NDArray[np.int64], current: "_WaterComponents") -> Optional[bool]:
        """Decide from pristine components or defer a partial parent to the geometric predicate."""
        if intersection.size < 2:
            return False
        waters = current.labels[intersection]
        if np.all(waters == waters[0]):
            return False
        # The parent's pieces refine the original components. Deleting an active
        # voxel removes horizontal contacts and adds its thickness to a vertical gap.
        owned = self._ownership.get(parent)
        if owned is None:
            return True
        if len(owned) == 1:
            origin, count = next(iter(owned.items()))
            return True if count == self._sizes[origin] else None

        origins, codes = np.unique(self.labels[intersection], return_inverse=True)
        lower = np.full(origins.size, np.iinfo(waters.dtype).max, dtype=waters.dtype)
        upper = np.zeros_like(lower)
        np.minimum.at(lower, codes, waters)
        np.maximum.at(upper, codes, waters)
        multiple = origins[lower != upper]
        if any(owned[origin] == self._sizes[origin] for origin in multiple.tolist()):
            return True
        return None if multiple.size else False


def merge_loop_client_bytes(
    n_voxels: int, time_chunk: int, compute_mode: str, *, layer_elements: Optional[int] = None, pristine_slices: int = 0
) -> int:
    """
    Client memory the merge loop needs for chunk buffers and static voxel geometry.

    Counts the retained int32 ID chunks, any pristine slices, the dense float64 cell
    volumes and four float64 layer arrays. There are 2 retained chunks in
    ``streaming`` and 11 in ``persist``.

    Not counted: wet masks, horizontal geometry, object and overlap tables, temporary
    working buffers and the Dask graph.

    Parameters
    ----------
    n_voxels : int
        ``nz * ny * nx``.
    time_chunk : int
        Time chunk width of the ID field.
    compute_mode : {'streaming', 'persist'}
        Tracker compute mode.
    layer_elements : int, optional
        Elements per layer array. Defaults to ``n_voxels`` for dense geometry.
    pristine_slices : int, default=0
        Maximum number of extra copies of edited slices used to certify splits.

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
    layer_elements = n_voxels if layer_elements is None else int(layer_elements)
    id_buffers = (_RETAINED_CHUNKS[compute_mode] * time_chunk + pristine_slices) * n_voxels * 4
    return id_buffers + (n_voxels + 4 * layer_elements) * 8


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
    overlaps: Optional[OverlapTable] = None,
    water_components: Optional[Tuple[_WaterComponents, _WaterComponents]] = None,
    inplace: bool = False,
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
    overlaps : OverlapTable, optional
        Active boundary's raw overlaps, updated when objects are renamed.
    water_components : tuple, optional
        Pristine components and ownership of the earlier and later slices.
    inplace : bool, default=False
        Allow editing a writable, C-contiguous slice owned by the caller.

    Returns
    -------
    numpy.ndarray
        The consolidated ``prev1``, copied on the first rename unless ``inplace`` is set.
    """
    if inplace and not prev1.flags.writeable:
        raise ValueError("assignment destination is read-only")
    flat_prev, flat_cur, flat_volume = prev2.ravel(), prev1.ravel(), cell_volume.ravel()
    table = overlaps if overlaps is not None else OverlapTable(flat_prev, flat_cur, flat_volume)
    linked, _ = table.linked(store, threshold)
    if linked.shape[0] == 0:
        return prev1

    parent_ids, counts = np.unique(linked[:, 0], return_counts=True)
    splitting = parent_ids[counts > 1]
    if splitting.size == 0:
        return prev1

    out, flat_out = prev1, None
    kept = set()
    renamed_onto: Dict[int, int] = {}
    candidates: List[Tuple[int, List[int]]] = []
    changed = set()
    index = table.current_index
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
            if flat_out is None:
                direct_edit = inplace and prev1.flags.c_contiguous and prev1.flags.writeable
                out = prev1 if direct_edit else prev1.copy(order="C")
                flat_out = out.ravel()
                if index is None:
                    index = ObjectIndex.from_slice(flat_cur)
                if water_components is not None:
                    water_components[1].preserve()
            if water_components is not None:
                water_components[1].consolidate(first, others, index)
            for oid in others:
                flat_out[index.voxels(oid)] = first
                store.drop(oid)
            index.rename(others, first)
            changed.update([first, *others])
            logger.debug(f"Consolidated {others} into {first} (parent {parent})")
            if ledger is not None:
                ledger.renamed.extend((oid, first) for oid in others)
                renamed_onto.update((oid, first) for oid in others)
                candidates.append((parent, [first, *others]))
        kept.add(first)

    if ledger is not None and split_geometry is not None and candidates:
        water = (
            split_geometry.water_pieces(out != 0)
            if water_components is None
            else water_components[1].labels.reshape(split_geometry.wet.shape)
        )
        for parent, children in candidates:
            # A later parent can rename this parent's kept ID in turn, so follow it to the survivor.
            survivor = children[0]
            while survivor in renamed_onto:
                survivor = renamed_onto[survivor]
            split = None
            if water_components is not None:
                voxels = index.voxels(survivor)
                intersection = voxels[flat_prev[voxels] == parent]
                split = water_components[0].certifies_split(parent, intersection, water_components[1])
            if split is None:
                split = split_geometry.separates(prev2 == parent, out == survivor, water)
            if split:
                ledger.record_split(time_value, parent, children)

    if changed:
        for oid in sorted(kept):
            if oid in store:
                voxels = index.voxels(oid)
                # Bincount sums in ascending voxel order, as in the volume store.
                volume = np.bincount(np.zeros(voxels.size, dtype=np.intp), weights=flat_volume[voxels], minlength=1)
                store.set(oid, volume[0])
        if overlaps is not None:
            overlaps.current_index = index
            overlaps.refresh(flat_prev, flat_out, flat_volume, changed)
        if inplace and out is not prev1:
            prev1[...] = out
            return prev1
    return out


def _resolve_merge(
    slice_t: NDArray[np.int32],
    index_t: ObjectIndex,
    index_prev: ObjectIndex,
    child_id: int,
    parent_ids: NDArray[np.int32],
    store: VolumeStore,
    new_ids: NDArray[np.int32],
    coords: ScaledCoords,
    cell_volume: NDArray[np.float64],
    *,
    components: Optional[_WaterComponents] = None,
) -> NDArray[np.int32]:
    """
    Partition one merged child among its parents.

    The child keeps its ID for the voxels nearest the first (lowest) parent; every other
    parent's share gets a new ID from ``new_ids``. Only the child's voxels are read or
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
    store : VolumeStore
        Object volumes; the pieces' volumes are set and a piece with no voxels is dropped.
    new_ids : numpy.ndarray
        The ``len(parent_ids) - 1`` consecutive IDs allocated for this child.
    coords : ScaledCoords
        Scaled voxel coordinates.
    cell_volume : numpy.ndarray
        Flat voxel volumes (m^3).

    Returns
    -------
    child_ids : numpy.ndarray
        int32 ``[child_id, *new_ids]``, aligned with ``parent_ids``.
    """
    child_ids = np.empty(parent_ids.size, dtype=np.int32)
    child_ids[0], child_ids[1:] = child_id, new_ids

    child_vox = index_t.voxels(child_id)
    parent_vox = [index_prev.voxels(p) for p in parent_ids.tolist()]
    labels = child_ids[partition_child(child_vox, parent_vox, parent_ids, coords)]

    if components is not None:
        components.preserve()
    slice_t[child_vox] = labels
    index_t.reassign(child_id, child_vox, labels)

    # Volumes summed over the child's voxels in ascending flat order: the same per-ID order as
    # identifying the pieces directly. A piece that received no voxel is dropped.
    present = np.zeros(child_ids.size, dtype=bool)
    if child_vox.size:
        uniq, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if components is not None:
            components.partition(child_vox, uniq, counts)
        sums = np.bincount(inverse.ravel(), weights=cell_volume[child_vox], minlength=uniq.size)
        for oid, vol in zip(uniq.tolist(), sums.tolist()):
            store.set(oid, vol)
        present = np.isin(child_ids, uniq)
    missing = child_ids[~present].tolist()
    for oid in missing:
        store.drop(oid)
    if missing:
        logger.debug(f"Merge of child {child_id}: pieces {missing} received no voxels and were dropped")
    return child_ids


def _rows_by_child(pairs: NDArray[np.int64], merging: NDArray[np.int64]) -> List[NDArray[np.int64]]:
    """Return the overlap-table rows of each child in ``merging``, in table order; one sort, O(rows + children) held."""
    order = np.argsort(pairs[:, 1], kind="stable")
    children = pairs[order, 1]
    starts = np.searchsorted(children, merging, side="left")
    stops = np.searchsorted(children, merging, side="right")
    return [order[start:stop] for start, stop in zip(starts.tolist(), stops.tolist())]


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
    *,
    components: Optional[_WaterComponents] = None,
) -> set[int]:
    """
    Partition the merging children in place, record each merge and return all edited IDs.

    The new IDs of the whole pass are allocated in one request before any child is touched, so an
    exhausted ID space raises with the slice, indexes, store and ledger unchanged. Children are
    handled in ascending order and take consecutive IDs from that block, which gives the same IDs
    as allocating per child.
    """
    rows_by_child = _rows_by_child(pairs, merging)
    new_ids = allocator.allocate(sum(rows.size - 1 for rows in rows_by_child))
    changed = set()
    start = 0
    for child_id, rows in zip(merging.tolist(), rows_by_child):
        stop = start + rows.size - 1
        parent_ids = pairs[rows, 0].astype(np.int32)
        overlap_volumes = overlap[rows]
        child_ids = _resolve_merge(
            cur,
            index_t,
            index_prev,
            child_id,
            parent_ids,
            store,
            new_ids[start:stop],
            coords,
            cell_volume,
            components=components,
        )
        ledger.times.append(time_value)
        ledger.parent_ids.append(parent_ids)
        ledger.child_ids.append(child_ids)
        ledger.overlap_volumes.append(overlap_volumes)
        changed.update(child_ids.tolist())
        start = stop
    return changed


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
    *,
    overlaps: Optional[OverlapTable] = None,
    components: Optional[_WaterComponents] = None,
) -> NDArray[np.int32]:
    """Resolve merges at one timestep, in place, until no child has more than one linked parent."""
    iteration = 0
    index_prev: Optional[ObjectIndex] = None
    if overlaps is None:
        overlaps = OverlapTable(prev, cur, cell_volume)
    while True:
        pairs, overlap = overlaps.linked(store, threshold)
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
        if index_prev is None:
            # Built once per timestep: t-1 does not change here, and the t index is kept
            # current by each partition.
            index_prev = ObjectIndex.from_slice(prev)
            overlaps.current_index = ObjectIndex.from_slice(cur)
        changed = _resolve_merges(
            cur,
            overlaps.current_index,
            index_prev,
            pairs,
            overlap,
            merging,
            store,
            allocator,
            coords,
            cell_volume,
            ledger,
            time_value,
            components=components,
        )
        overlaps.refresh(prev, cur, cell_volume, changed)
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
    pristine_labels: bool = False,
    links_out: Optional[List[NDArray[np.int64]]] = None,
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
    pristine_labels : bool, default=False
        Use the input IDs as water-component labels. Only set for the unedited
        output of identification under the supplied gap and seam rules.
    links_out : list, optional
        Append final linked pairs for each nonempty boundary in chronological order.

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
    spatial_shape = id_field.shape[1:]
    if geom.cell_volume.shape != spatial_shape:
        raise ConfigurationError(
            f"cell_volume has shape {geom.cell_volume.shape}, expected {spatial_shape} for the ID field",
            context={"cell_volume": geom.cell_volume.shape, "id_field": spatial_shape},
        )
    cell_volume = np.ascontiguousarray(geom.cell_volume, dtype=np.float64).ravel()
    n_voxels = cell_volume.size

    split_geometry = None
    if max_gap_m is not None and wrap_x is not None:
        split_geometry = SplitGeometry(geom.wet, geom.thickness, float(max_gap_m), bool(wrap_x))

    time = id_field[timedim]
    time_values = time.values
    ledger = MergeLedger(time_dtype=time.dtype)
    if time.dtype.kind == "O":
        if time_values.size and hasattr(time_values[0], "calendar"):
            ledger.time_encoding = {"calendar": time_values[0].calendar}
        elif "calendar" in time.encoding:
            ledger.time_encoding = {"calendar": time.encoding["calendar"]}

    streaming = materialiser is not None and materialiser.is_streaming
    if streaming:
        if id_field_path is None:
            id_field_path = str(Path(materialiser.staging_dir) / "volume_id_field_merged.zarr")
        writer: Optional[ObjectIDRegionWriter] = ObjectIDRegionWriter(id_field, id_field_path, timedim)
    else:
        writer = None
        id_field = id_field.persist()

    chunk_sizes = list(id_field.chunks[0])
    chunk_bounds = np.cumsum([0] + chunk_sizes)
    n_chunks = len(chunk_sizes)
    updated: List[Tuple[int, int, NDArray[np.int32]]] = []
    previous = previous_water = None

    def flush(entries) -> None:
        nonlocal id_field
        for start, end, values in entries:
            if writer is not None:
                writer.write(start, end, id_field.isel({timedim: slice(start, end)}).copy(data=values))
            elif end > start:
                id_field[{timedim: slice(start, end)}] = values

    for chunk_idx in range(n_chunks):
        chunk_start, chunk_end = int(chunk_bounds[chunk_idx]), int(chunk_bounds[chunk_idx + 1])
        n = chunk_end - chunk_start
        values = np.ascontiguousarray(id_field.isel({timedim: slice(chunk_start, chunk_end)}).values, dtype=np.int32)
        flat = values.reshape(n, n_voxels)  # a view: edits to flat land in values

        for relative_t in range(n):
            absolute_t = chunk_start + relative_t
            current = flat[relative_t]
            current_water = _WaterComponents(current) if pristine_labels and split_geometry is not None else None
            if previous is not None:
                overlaps = OverlapTable(previous, current, cell_volume)
                _resolve_timestep(
                    previous,
                    current,
                    store,
                    allocator,
                    coords,
                    cell_volume,
                    overlap_threshold,
                    ledger,
                    time_values[absolute_t],
                    absolute_t,
                    max_iterations,
                    overlaps=overlaps,
                    components=current_water,
                )
                consolidate_volumes(
                    previous,
                    current,
                    store,
                    cell_volume,
                    overlap_threshold,
                    ledger=ledger,
                    time_value=time_values[absolute_t],
                    split_geometry=split_geometry,
                    overlaps=overlaps,
                    water_components=(previous_water, current_water) if current_water is not None else None,
                    inplace=True,
                )
                if links_out is not None:
                    # Both IDs and volumes are final at this boundary. Consolidation
                    # can make formerly subthreshold pairs pass.
                    pairs, _ = overlaps.linked(store, overlap_threshold)
                    if pairs.size:
                        links_out.append(pairs)
                del overlaps
            previous, previous_water = current, current_water

        updated.append((chunk_start, chunk_end, values))

        if chunk_idx % 10 == 0:
            logger.info(f"4-D split/merge: chunk {chunk_idx} of {n_chunks}")

        # Retain the latest chunk for the next boundary.
        if len(updated) > 1 and (writer is not None or chunk_idx % _PERSIST_FLUSH_EVERY == 0):
            flush(updated[:-1])
            updated = updated[-1:]
            if writer is None:
                id_field = id_field.persist()

    flush(updated)
    updated = []
    if writer is not None:
        id_field = writer.finalise()
    else:
        id_field = id_field.persist()
    return id_field, store, ledger
