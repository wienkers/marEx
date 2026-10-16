"""
Connected-component labelling in space and time with a task graph linear in the number of chunks.

``dask_image.ndmeasure.label`` builds its cross-chunk adjacency graph with a task count that
grows roughly quadratically in the number of time chunks, and its construction time faster
still, so a long record cannot be labelled. ``label_time_connected`` gives identical labels and
count, chunk by chunk, with a fixed number of tasks per chunk:

1. Each original chunk is labelled with ``scipy.ndimage.label``. Only its object count, its
   first and last time planes and its longitude-seam contacts leave the task.
2. Consecutive chunks' facing planes give the cross-chunk contacts. Provisional IDs are the
   local labels offset by the counts of the earlier chunks (C order), exactly as dask-image
   numbers them, and ``scipy.sparse.csgraph.connected_components`` resolves them. It numbers
   components in order of their smallest vertex, so each object gets the ID dask-image gives
   it, and background (vertex 0, never joined) stays 0.
3. Each chunk is labelled again and mapped through its own slice of the result.

The resolving task holds only counts and contact pairs: ``O(objects + contacts)`` integers. The
input is read twice (once per pass), so keep it persisted or staged rather than recomputed.
"""

from typing import List, Tuple

import dask
import dask.array as da
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label as _scipy_label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_FULL_CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)
_INT32_MAX = np.iinfo(np.int32).max


def _contact_pairs(face: NDArray[np.integer]) -> NDArray[np.int64]:
    """Pairs of distinct labels that touch within a two-plane ``face`` (one row per pair, (n, 2))."""
    common = _scipy_label(face > 0, structure=_FULL_CONNECTIVITY)[0].ravel()
    labels = face.ravel()
    fg = common > 0
    if not fg.any():
        return np.empty((0, 2), dtype=np.int64)
    rows = np.unique(np.stack((common[fg], labels[fg]), axis=1).astype(np.int64), axis=0)
    same_group = np.flatnonzero(np.diff(rows[:, 0]) == 0)
    return np.stack((rows[same_group, 1], rows[same_group + 1, 1]), axis=1)


def _label_chunk(block: NDArray[np.bool_], wrap_x: bool) -> Tuple[int, NDArray, NDArray, NDArray[np.int64]]:
    """First pass: object count, first and last time planes, and the chunk's seam contacts."""
    labels, n = _scipy_label(block, structure=_FULL_CONNECTIVITY)
    seam = _contact_pairs(labels[:, :, [0, -1]]) if wrap_x else np.empty((0, 2), dtype=np.int64)
    return int(n), labels[0].copy(), labels[-1].copy(), seam


def _boundary_pairs(first: tuple, second: tuple, wrap_x: bool) -> NDArray[np.int64]:
    """Contacts across the time boundary between two consecutive chunks.

    Pairs are in the first chunk's numbering: the second chunk's labels shifted by the first's count.
    """
    n_first, _, last_plane, _ = first
    _, first_plane, _, _ = second
    shifted = np.where(first_plane > 0, first_plane.astype(np.int64) + n_first, 0)
    face = np.stack((last_plane.astype(np.int64), shifted))
    pairs = [_contact_pairs(face)]
    if wrap_x:  # where the time boundary meets the longitude seam
        pairs.append(_contact_pairs(face[:, :, [0, -1]]))
    return np.concatenate(pairs)


def _resolve(counts: List[int], seams: List[NDArray[np.int64]], boundaries: List[NDArray[np.int64]]) -> tuple:
    """Provisional IDs -> final IDs; returns (final IDs, object count, chunk offsets, chunk counts)."""
    counts = np.array(counts, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    n_provisional = int(counts.sum())
    if n_provisional > _INT32_MAX:
        raise ValueError(
            f"{n_provisional} provisional objects exceed the int32 ID range; an object is counted once per "
            "chunk it spans, so use fewer, larger time chunks"
        )
    edges = [s + offsets[k] for k, s in enumerate(seams)]
    edges += [b + offsets[k] for k, b in enumerate(boundaries)]
    edges = np.concatenate(edges) if edges else np.empty((0, 2), dtype=np.int64)
    graph = coo_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n_provisional + 1, n_provisional + 1)
    ).tocsr()
    n_components, final = connected_components(graph, directed=False)
    return final.astype(np.int32), int(n_components - 1), offsets, counts


def _lut(resolved: tuple, k: int) -> NDArray[np.int32]:
    """Chunk ``k``'s slice of the final IDs, with background first."""
    final, _, offsets, counts = resolved
    return np.concatenate(([0], final[offsets[k] + 1 : offsets[k] + 1 + counts[k]])).astype(np.int32)


def _relabel_chunk(block: NDArray[np.bool_], lut: NDArray[np.int32]) -> NDArray[np.int32]:
    """Second pass: label the chunk again and map it through its slice of the final IDs."""
    return lut[_scipy_label(block, structure=_FULL_CONNECTIVITY)[0]]


def label_time_connected(data: da.Array, wrap_x: bool) -> Tuple[da.Array, da.Array]:
    """
    Label a boolean ``(time, y, x)`` dask array with full 3-D connectivity.

    Equal, value for value, to ``dask_image.ndmeasure.label(data, structure=np.ones((3, 3, 3)),
    wrap_axes=(2,) if wrap_x else None)``. ``data`` must be whole in ``y`` and ``x``.

    Returns
    -------
    labels : dask.array.Array
        int32 labels with ``data``'s chunks.
    n_objects : dask.array.Array
        0-d count of objects.
    """
    if data.ndim != 3 or len(data.chunks[1]) != 1 or len(data.chunks[2]) != 1:
        raise ValueError(f"label_time_connected needs a (time, y, x) array whole in y and x; got chunks {data.chunks}")
    blocks = data.to_delayed(optimize_graph=False).ravel()
    summaries = [dask.delayed(_label_chunk, pure=True)(b, wrap_x) for b in blocks]
    boundary = dask.delayed(_boundary_pairs, pure=True)
    boundaries = [boundary(summaries[k], summaries[k + 1], wrap_x) for k in range(len(blocks) - 1)]
    # Only counts and contact pairs reach the resolving task; the time planes stay with the boundary tasks.
    counts = [s[0] for s in summaries]
    seams = [s[3] for s in summaries]
    resolved = dask.delayed(_resolve, pure=True)(counts, seams, boundaries)
    lut, relabel = dask.delayed(_lut, pure=True), dask.delayed(_relabel_chunk, pure=True)
    ny, nx = data.shape[1], data.shape[2]
    parts = [
        da.from_delayed(relabel(b, lut(resolved, k)), shape=(nt, ny, nx), dtype=np.int32)
        for k, (b, nt) in enumerate(zip(blocks, data.chunks[0]))
    ]
    labels = da.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    n_objects = da.from_delayed(resolved[1], shape=(), dtype=np.int64)
    return labels, n_objects
