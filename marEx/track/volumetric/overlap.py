"""
MarEx Track: volume-weighted overlap between two labelled ``(z, y, x)`` slices.

Overlap is measured in physical volume (m^3) rather than voxel count, so a cell in a
thick layer or a large-area column counts for what it is. All sums are float64 and are
taken with ``np.bincount`` over a sorted pair index, so the summation order, and hence
every bit of the result, is deterministic.
"""

from typing import TYPE_CHECKING, Tuple

import numpy as np
from numpy.typing import NDArray

from ...exceptions import TrackingError

if TYPE_CHECKING:
    from .store import VolumeStore

_KEY_LIMIT = 2**62


def volume_overlaps(
    ids_prev: NDArray[np.int32],
    ids_next: NDArray[np.int32],
    cell_volume: NDArray[np.float64],
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Sum the cell volume shared by every pair of objects in two slices.

    Voxels where either ID is 0 (background) are ignored.

    Parameters
    ----------
    ids_prev, ids_next : numpy.ndarray
        Flat 1-D integer labels of the earlier and later slice.
    cell_volume : numpy.ndarray
        Flat 1-D voxel volumes (m^3), the same length.

    Returns
    -------
    pairs : numpy.ndarray
        ``(n, 2)`` int64 array of ``(id_prev, id_next)``, sorted lexicographically.
    overlap : numpy.ndarray
        ``(n,)`` float64 overlap volume of each pair.

    Raises
    ------
    TrackingError
        If the inputs differ in length, or the pair key would exceed ``2**62``.
    """
    a = np.asarray(ids_prev).ravel()
    b = np.asarray(ids_next).ravel()
    v = np.asarray(cell_volume, dtype=np.float64).ravel()
    if not (a.size == b.size == v.size):
        raise TrackingError(
            "Overlap inputs must have the same length",
            context={"ids_prev": a.size, "ids_next": b.size, "cell_volume": v.size},
        )

    keep = (a > 0) & (b > 0)
    if not keep.any():
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    a = a[keep].astype(np.int64)
    b = b[keep].astype(np.int64)
    v = v[keep]

    # Python-int radix so the bound check itself cannot wrap.
    radix = max(int(a.max()), int(b.max())) + 1
    if radix * radix > _KEY_LIMIT:
        raise TrackingError(
            "Event IDs are too large to form a unique overlap key",
            details=f"radix {radix} squared exceeds 2**62",
            suggestions=["Relabel the objects to a compact ID range before linking"],
            context={"radix": radix},
        )

    key = a * radix + b
    unique_keys, inverse = np.unique(key, return_inverse=True)
    overlap = np.bincount(inverse.ravel(), weights=v, minlength=unique_keys.size)
    pairs = np.stack([unique_keys // radix, unique_keys % radix], axis=1).astype(np.int64)
    return pairs, overlap.astype(np.float64, copy=False)


def link_mask(
    pairs: NDArray[np.int64],
    overlap: NDArray[np.float64],
    store: "VolumeStore",
    threshold: float,
) -> NDArray[np.bool_]:
    """
    Flag pairs whose overlap reaches ``threshold`` of the smaller object's volume.

    A pair links when ``overlap / min(V_a, V_b) >= threshold``. Rows where either ID
    is missing from ``store`` are False.

    Parameters
    ----------
    pairs : numpy.ndarray
        ``(n, 2)`` int64 ``(id_prev, id_next)`` rows.
    overlap : numpy.ndarray
        ``(n,)`` float64 overlap volumes.
    store : VolumeStore
        Object volumes.
    threshold : float
        Minimum overlap fraction.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` boolean mask.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    overlap = np.asarray(overlap, dtype=np.float64)
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    present = store.contains_many(pairs[:, 0]) & store.contains_many(pairs[:, 1])
    out = np.zeros(pairs.shape[0], dtype=bool)
    if not present.any():
        return out
    va = store.volumes(pairs[present, 0])
    vb = store.volumes(pairs[present, 1])
    smaller = np.minimum(va, vb)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = overlap[present] / smaller
    out[present] = (smaller > 0) & (fraction >= threshold)
    return out
