"""
Independent reference implementations for volumetric tracking tests.

These are written from the definitions, not from the production kernels: they enumerate
voxels, build explicit edge lists and resolve them with
``scipy.sparse.csgraph.connected_components``. They are slow and only meant for small
synthetic inputs.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


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
