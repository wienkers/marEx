"""
Independent reference implementations for volumetric tracking tests.

The graph oracle enumerates voxels and uses explicit edge lists. The distance oracle
uses the stored coordinates directly, and the link oracle scans the finished field.
"""

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from marEx.exceptions import ConfigurationError
from marEx.track.volumetric.overlap import link_mask, volume_overlaps
from marEx.track.volumetric.store import VolumeStore


def _canonical_by_first_voxel(component: NDArray[np.integer]) -> NDArray[np.int32]:
    """Renumber component IDs 1..n in order of first appearance along ``component`` (C-order voxel list)."""
    new_id = {}
    out = np.empty(len(component), dtype=np.int32)
    for i, c in enumerate(component.tolist()):
        if c not in new_id:
            new_id[c] = len(new_id) + 1
        out[i] = new_id[c]
    return out


def voxel_graph_labels(
    active: NDArray[np.bool_],
    wet: NDArray[np.bool_],
    thickness: NDArray[np.float64],
    max_gap_m: float,
    wrap_x: bool,
) -> NDArray[np.int32]:
    """
    Label a ``(z, y, x)`` volume from an explicit voxel graph.

    Foreground voxels are ``active & wet``. Edges:

    - horizontal: the 8 in-plane neighbours on the same level; ``y`` is bounded, ``x`` wraps
      (``x = 0`` and ``x = nx - 1`` are neighbours, diagonals included) when ``wrap_x``;
    - vertical: foreground voxels at levels ``k1 < k2`` of one column, when every cell strictly
      between them is wet and not foreground and the thickness of those cells, summed from the
      top down, is ``<= max_gap_m``.

    Returns int32 labels, 0 on background, components numbered 1..n by first voxel in C order.
    """
    active = np.asarray(active, dtype=bool)
    wet = np.asarray(wet, dtype=bool)
    thickness = np.asarray(thickness, dtype=np.float64)
    if thickness.ndim == 1:
        thickness = np.broadcast_to(thickness[:, None, None], active.shape)
    nz, ny, nx = active.shape
    fg = active & wet

    voxels = np.argwhere(fg)  # C order over (z, y, x)
    index = -np.ones(fg.shape, dtype=np.int64)
    index[fg] = np.arange(len(voxels))
    out = np.zeros(fg.shape, dtype=np.int32)
    if len(voxels) == 0:
        return out

    rows, cols = [], []
    for i, (z, y, x) in enumerate(voxels.tolist()):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if not 0 <= yy < ny:
                    continue
                if wrap_x:
                    xx %= nx
                elif not 0 <= xx < nx:
                    continue
                if fg[z, yy, xx]:
                    rows.append(i)
                    cols.append(int(index[z, yy, xx]))

    for y in range(ny):
        for x in range(nx):
            for k1 in range(nz):
                if not fg[k1, y, x]:
                    continue
                for k2 in range(k1 + 1, nz):
                    if not fg[k2, y, x]:
                        continue
                    eligible = True
                    gap = 0.0
                    for m in range(k1 + 1, k2):
                        if not wet[m, y, x] or fg[m, y, x]:
                            eligible = False
                            break
                        gap += float(thickness[m, y, x])
                    if eligible and gap <= max_gap_m:
                        rows.append(int(index[k1, y, x]))
                        cols.append(int(index[k2, y, x]))

    n = len(voxels)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n, n)).tocsr()
    _, component = connected_components(graph, directed=False)
    out[fg] = _canonical_by_first_voxel(component)
    return out


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
    out = []
    previous = None
    start = 0
    for size in chunk_sizes:
        values = np.asarray(id_field.isel({timedim: slice(start, start + size)}).values).reshape(size, n_voxels)
        for t in range(size):
            if previous is not None:
                pairs, overlap = volume_overlaps(previous, values[t], cv)
                pairs = pairs[link_mask(pairs, overlap, store, threshold)]
                if pairs.shape[0]:
                    out.append(pairs)
            previous = values[t]
        start += size
    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(out, axis=0).astype(np.int64, copy=False)


def _squared_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Squared Euclidean distance over the last axis (length 4), summed in a fixed coordinate order."""
    d = a - b
    return ((d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]) + d[..., 2] * d[..., 2]) + d[..., 3] * d[..., 3]
