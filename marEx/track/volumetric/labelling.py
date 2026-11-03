"""
MarEx Track: per-timestep labelling for 4-D tracking.

Connected components of a ``(z, y, x)`` extremes field under the 4-D
connectivity rule:

- horizontal: 8-neighbour within each level, plus the periodic-longitude seam
  (column 0 touches column ``nx - 1`` on the same row and both diagonals) when
  ``wrap_x``;
- vertical: same column only. Two active cells of one column join when every cell
  between them is wet and inactive and the thickness of those cells, summed from the
  top down, is ``<= max_gap_m``. Consecutive active levels have gap 0 and always join.
  A dry cell is a barrier: nothing joins across it, and the running gap restarts
  below it.

Only ``active & wet`` cells are foreground; an active flag on a dry cell is ignored.
Cells bridged by the gap rule stay background. Arrays are in internal order:
``(z, y, x)`` with ``z`` ascending positive-down depth.
"""

from typing import Tuple

import numpy as np
import xarray as xr
from numba import njit
from numpy.typing import NDArray
from scipy.ndimage import label as scipy_label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ...exceptions import ConfigurationError
from ...logging_config import get_logger
from ..objects import _EIGHT_CONNECTIVITY, _merge_lon_seam
from .geometry import VerticalGeometry

logger = get_logger(__name__)


@njit(cache=True)
def _column_contacts(
    level_ids: NDArray[np.int64],
    wet: NDArray[np.bool_],
    thickness: NDArray[np.float64],
    max_gap_m: float,
) -> NDArray[np.int64]:
    """
    Vertical contacts between per-level component IDs, one row ``(upper, lower)`` per contact.

    Every column is walked top to bottom, keeping the ID of the last active cell and the
    thickness of the wet inactive cells seen since it, summed in top-to-bottom order. An
    active cell joins the last one when that gap is ``<= max_gap_m``. A dry cell clears
    both. The level loop is outermost so the walk reads memory in C order; each column's
    state lives in ``last`` and ``gap``.
    """
    nz, ny, nx = level_ids.shape
    out = np.empty((max(nz - 1, 0) * ny * nx, 2), dtype=np.int64)
    last = np.zeros((ny, nx), dtype=np.int64)
    gap = np.zeros((ny, nx), dtype=np.float64)
    count = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if not wet[k, j, i]:
                    last[j, i] = 0
                    gap[j, i] = 0.0
                    continue
                cur = level_ids[k, j, i]
                if cur > 0:
                    if last[j, i] > 0 and gap[j, i] <= max_gap_m:
                        out[count, 0] = last[j, i]
                        out[count, 1] = cur
                        count += 1
                    last[j, i] = cur
                    gap[j, i] = 0.0
                else:
                    gap[j, i] += thickness[k, j, i]
    return out[:count].copy()


def _check_max_gap(max_gap_m: float) -> float:
    """Validate the vertical gap threshold (metres, >= 0)."""
    value = float(max_gap_m)
    if not value >= 0.0:  # also rejects NaN
        raise ConfigurationError(
            f"max_vertical_gap_m must be >= 0 metres, got {max_gap_m!r}",
            suggestions=["Use 0 to join only consecutive active levels", "Use a positive gap in metres to bridge thin layers"],
            context={"max_vertical_gap_m": max_gap_m},
        )
    return value


def label_volume(
    active: NDArray[np.bool_],
    wet: NDArray[np.bool_],
    thickness: NDArray[np.float64],
    max_gap_m: float,
    wrap_x: bool,
) -> Tuple[NDArray[np.int32], int]:
    """
    Label the connected components of one ``(z, y, x)`` timestep.

    Parameters
    ----------
    active : numpy.ndarray
        Boolean extremes field, ``(z, y, x)`` in internal (ascending depth) order.
    wet : numpy.ndarray
        Boolean wet mask, same shape. Active dry cells are ignored.
    thickness : numpy.ndarray
        Layer thickness in metres, same shape; read on wet inactive cells only.
    max_gap_m : float
        Largest summed thickness of wet inactive cells that a vertical link may bridge.
        Equality joins.
    wrap_x : bool
        Join across the periodic-longitude seam (global grids; False in regional mode).

    Returns
    -------
    labels : numpy.ndarray
        int32 labels ``1..n``, 0 on background, numbered by first occupied voxel in C
        order over ``(z, y, x)``. With ``nz == 1`` this equals the slice labelling of the 3-D tracker.
    n : int
        Number of objects.
    """
    active = np.asarray(active, dtype=bool)
    if active.ndim != 3 or np.shape(wet) != active.shape or np.shape(thickness) != active.shape:
        raise ConfigurationError(
            "active, wet and thickness must be (z, y, x) arrays of one shape",
            context={"active": active.shape, "wet": np.shape(wet), "thickness": np.shape(thickness)},
        )
    max_gap = _check_max_gap(max_gap_m)
    wet = np.ascontiguousarray(wet, dtype=bool)
    thickness = np.ascontiguousarray(thickness, dtype=np.float64)
    foreground = active & wet

    # Per-level 8-connected labels with distinct temporary IDs: level k's labels are
    # offset by the label counts of the levels above it.
    level_ids = np.zeros(foreground.shape, dtype=np.int64)
    n_provisional = 0
    for k in range(foreground.shape[0]):
        labels_k, n_k = scipy_label(foreground[k], structure=_EIGHT_CONNECTIVITY)
        if n_k == 0:
            continue
        if wrap_x:
            labels_k = _merge_lon_seam(labels_k, n_k)
        level_ids[k] = np.where(labels_k > 0, labels_k.astype(np.int64) + n_provisional, 0)
        n_provisional += n_k
    if n_provisional == 0:
        return np.zeros(foreground.shape, dtype=np.int32), 0

    # Union the vertical contacts. Labels absorbed by the seam union are isolated vertices
    # with no voxels, so the object count comes from the voxels, not from n_components.
    pairs = _column_contacts(level_ids, wet, thickness, max_gap)
    size = n_provisional + 1
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int32), (pairs[:, 0], pairs[:, 1])), shape=(size, size)).tocsr()
    _, component = connected_components(graph, directed=False)

    # Final IDs by first occupied voxel in C order.
    flat = level_ids.ravel()
    voxels = np.flatnonzero(flat)
    voxel_component = component[flat[voxels]]
    present, first = np.unique(voxel_component, return_index=True)
    lut = np.zeros(int(component.max()) + 1, dtype=np.int32)
    lut[present[np.argsort(first, kind="stable")]] = np.arange(1, len(present) + 1, dtype=np.int32)
    labels = np.zeros(flat.size, dtype=np.int32)
    labels[voxels] = lut[voxel_component]
    return labels.reshape(foreground.shape), int(len(present))


def label_volumes_lazy(
    data: xr.DataArray,
    geom: VerticalGeometry,
    max_gap_m: float,
    wrap_x: bool,
    zdim: str,
    ydim: str,
    xdim: str,
) -> xr.DataArray:
    """
    Lazily label every timestep of a 4-D extremes field.

    Applies :func:`label_volume` over the core dimensions ``(zdim, ydim, xdim)``,
    vectorised over the remaining (time) dimension. IDs are local to each timestep
    (``1..n`` per step); making them unique across time is the caller's job.

    Parameters
    ----------
    data : xarray.DataArray
        Boolean extremes field with ``zdim`` in internal (ascending positive-down depth)
        order, i.e. already reordered when ``geom.flip_z``. ``zdim``, ``ydim`` and
        ``xdim`` must each be a single chunk.
    geom : VerticalGeometry
        Supplies ``wet`` and ``thickness``; its ``(z, y, x)`` shape must match ``data``.
    max_gap_m : float
        Vertical gap threshold in metres (see :func:`label_volume`).
    wrap_x : bool
        Join across the periodic-longitude seam.
    zdim, ydim, xdim : str
        Dimension names.

    Returns
    -------
    xarray.DataArray
        int32 per-timestep labels, with ``data``'s dimensions and dimension order, lazy
        when ``data`` is dask-backed.
    """
    core = [zdim, ydim, xdim]
    missing = [d for d in core if d not in data.dims]
    if missing:
        raise ConfigurationError(
            f"data is missing dimension(s) {missing}",
            context={"dims": tuple(data.dims), "expected": tuple(core)},
        )
    shape = tuple(data.sizes[d] for d in core)
    if shape != geom.wet.shape:
        raise ConfigurationError(
            f"data has (z, y, x) shape {shape} but the vertical geometry has {geom.wet.shape}",
            suggestions=["Build the vertical geometry from the same grid as the extremes field"],
            context={"data": shape, "geometry": geom.wet.shape},
        )
    max_gap = _check_max_gap(max_gap_m)

    # Single-chunk dask arguments: passing numpy here would be copied into the graph by
    # dask's from_array, and they carry no time dimension so nothing is broadcast in time.
    whole = {d: -1 for d in core}
    wet = xr.DataArray(geom.wet, dims=core).chunk(whole)
    thickness = xr.DataArray(geom.thickness, dims=core).chunk(whole)

    def _labels_only(active, wet_cells, layer_thickness):
        return label_volume(active, wet_cells, layer_thickness, max_gap, wrap_x)[0]

    labels = xr.apply_ufunc(
        _labels_only,
        data,
        wet,
        thickness,
        input_core_dims=[core, core, core],
        output_core_dims=[core],
        output_dtypes=[np.int32],
        vectorize=True,
        dask="parallelized",
    )
    return labels.transpose(*data.dims)
