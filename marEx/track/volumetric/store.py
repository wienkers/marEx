"""
MarEx Track: per-slice object volumes and a voxel index for 4-D tracking.

``VolumeStore`` maps an object ID to its physical volume (m^3). ``ObjectIndex`` maps an
object ID to the flat voxel indices it occupies in one slice, built once by a stable
argsort so later lookups, relabels and merges never rescan the slice.
"""

from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray


def _volumes_by_id(ids: NDArray, cell_volume: NDArray[np.float64]):
    """Return ``(unique foreground ids, summed volume)``, summed in a deterministic order."""
    flat = np.asarray(ids).ravel()
    vol = np.asarray(cell_volume, dtype=np.float64).ravel()
    fg = flat > 0
    if not fg.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    uniq, inverse = np.unique(flat[fg], return_inverse=True)
    sums = np.bincount(inverse.ravel(), weights=vol[fg], minlength=uniq.size)
    return uniq.astype(np.int64), sums


class VolumeStore:
    """Object ID -> volume (m^3, float64)."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._vol: Dict[int, float] = {}

    def set(self, object_id: int, volume: float) -> None:
        """Record ``volume`` for ``object_id``, replacing any earlier value."""
        self._vol[int(object_id)] = float(volume)

    def drop(self, object_id: int) -> None:
        """Forget ``object_id``; a missing ID is ignored."""
        self._vol.pop(int(object_id), None)

    def __contains__(self, object_id) -> bool:
        """Whether ``object_id`` is stored."""
        try:
            return int(object_id) in self._vol
        except (TypeError, ValueError):
            return False

    def __len__(self) -> int:
        """Return the number of stored objects."""
        return len(self._vol)

    def contains_many(self, ids: NDArray) -> NDArray[np.bool_]:
        """Boolean mask of which ``ids`` are stored."""
        arr = np.asarray(ids).ravel()
        return np.fromiter((int(i) in self._vol for i in arr), dtype=bool, count=arr.size)

    def volumes(self, ids: NDArray) -> NDArray[np.float64]:
        """Volumes of ``ids``; raises ``KeyError`` for an ID that is not stored."""
        arr = np.asarray(ids).ravel()
        return np.fromiter((self._vol[int(i)] for i in arr), dtype=np.float64, count=arr.size)

    @classmethod
    def from_slice(cls, ids: NDArray[np.int32], cell_volume: NDArray[np.float64]) -> "VolumeStore":
        """Build a store from a labelled slice and its voxel volumes."""
        store = cls()
        store.update_from_slice(ids, cell_volume)
        return store

    def update_from_slice(self, ids: NDArray[np.int32], cell_volume: NDArray[np.float64]) -> None:
        """Set (or recompute) the volume of every object present in the slice."""
        uniq, sums = _volumes_by_id(ids, cell_volume)
        for oid, vol in zip(uniq.tolist(), sums.tolist()):
            self._vol[oid] = vol


class ObjectIndex:
    """One slice: object ID -> sorted flat voxel indices."""

    def __init__(self, table: Dict[int, NDArray[np.int64]]) -> None:
        """Wrap an ID -> voxel-index table (use :meth:`from_slice`)."""
        self._table = table

    @classmethod
    def from_slice(cls, ids: NDArray[np.int32]) -> "ObjectIndex":
        """Index a labelled slice with one stable argsort of its foreground."""
        flat = np.asarray(ids).ravel()
        fg_idx = np.flatnonzero(flat > 0).astype(np.int64)
        labels = flat[fg_idx]
        order = np.argsort(labels, kind="stable")
        fg_idx = fg_idx[order]
        labels = labels[order]
        table: Dict[int, NDArray[np.int64]] = {}
        if labels.size:
            starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
            stops = np.r_[starts[1:], labels.size]
            for s, e in zip(starts.tolist(), stops.tolist()):
                table[int(labels[s])] = fg_idx[s:e]
        return cls(table)

    def ids(self) -> NDArray[np.int64]:
        """Sorted IDs currently indexed."""
        return np.array(sorted(self._table), dtype=np.int64)

    def voxels(self, object_id: int) -> NDArray[np.int64]:
        """Ascending flat voxel indices of ``object_id`` (empty if absent)."""
        return self._table.get(int(object_id), np.empty(0, dtype=np.int64))

    def reassign(self, object_id: int, flat_idx: NDArray[np.int64], new_labels: NDArray[np.int32]) -> None:
        """
        Give voxels of ``object_id`` new labels.

        Parameters
        ----------
        object_id : int
            Object currently owning ``flat_idx``.
        flat_idx : numpy.ndarray
            Flat voxel indices, a subset of ``voxels(object_id)``.
        new_labels : numpy.ndarray
            New label of each voxel; 0 removes it from the foreground.
        """
        oid = int(object_id)
        idx = np.asarray(flat_idx, dtype=np.int64).ravel()
        lab = np.asarray(new_labels).ravel().astype(np.int64)
        old = self._table.get(oid, np.empty(0, dtype=np.int64))
        remaining = old[~np.isin(old, idx)]
        gathered: Dict[int, list] = {oid: [remaining]}
        for label in np.unique(lab).tolist():
            if label <= 0:
                continue
            gathered.setdefault(label, []).append(idx[lab == label])
        for label, parts in gathered.items():
            if label != oid and label in self._table:
                parts.append(self._table[label])
            merged = np.sort(np.concatenate(parts))
            if merged.size:
                self._table[label] = merged
            else:
                self._table.pop(label, None)

    def rename(self, from_ids: Sequence[int], to_id: int) -> None:
        """Move every voxel of ``from_ids`` onto ``to_id``."""
        to_id = int(to_id)
        parts = []
        for fid in from_ids:
            fid = int(fid)
            if fid in self._table:
                parts.append(self._table[fid] if fid == to_id else self._table.pop(fid))
        if to_id in self._table and not any(p is self._table[to_id] for p in parts):
            parts.append(self._table[to_id])
        if parts:
            self._table[to_id] = np.sort(np.concatenate(parts))
