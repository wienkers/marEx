"""
MarEx Track: nearest-parent-voxel partition of a merged child.

A merged child is divided between its parents by assigning every child voxel to
the nearest voxel of any parent, under a scaled metric: the horizontal part is the
3-D chord between points on a sphere of radius ``EARTH_RADIUS_M`` divided by ``L_H``,
the vertical part is the layer-midpoint depth divided by ``L_Z``. The chord needs no
wrap-around logic, so the dateline seam and regional domains need no special case.
There is no distance cap: a nearest parent voxel always exists.

Equidistant candidates are resolved exactly, not by whichever point the spatial tree
happens to return: squared distances are recomputed in float64 from the stored scaled
coordinates and a tie (exact equality) goes to the smallest parent ID.
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ...exceptions import ConfigurationError, TrackingError
from ...logging_config import get_logger
from .geometry import EARTH_RADIUS_M

logger = get_logger(__name__)

# Relative widening of the candidate radius so that points at exactly the minimum
# distance are never lost to the tree's own rounding. It only widens the search:
# the winner is always chosen by exact recomputed distances.
_BALL_WIDEN = 1.0e-9


def _positive_finite(value: float, name: str) -> float:
    """Return ``value`` as a float, raising ConfigurationError unless it is positive and finite."""
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ConfigurationError(
            f"{name} must be a positive finite length, got {value!r}",
            suggestions=["Use the grid-derived default scales from the vertical geometry, or pass a positive length in metres"],
            context={name: value},
        )
    return out


def _squared_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Squared Euclidean distance over the last axis (length 4), summed in a fixed coordinate order."""
    d = a - b
    return ((d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]) + d[..., 2] * d[..., 2]) + d[..., 3] * d[..., 3]


class ScaledCoords:
    """
    Scaled 4-D coordinates of the voxels of a structured ``(z, y, x)`` grid.

    A voxel at latitude ``phi``, longitude ``lam`` and layer-midpoint depth ``d`` maps to
    ``(R cos(phi) cos(lam) / L_H, R cos(phi) sin(lam) / L_H, R sin(phi) / L_H, d / L_Z)``.
    Voxels are addressed by their flat index into the C-ordered ``(z, y, x)`` array.

    Parameters
    ----------
    lat_deg, lon_deg : numpy.ndarray
        Latitude and longitude in degrees: ``(y,)`` and ``(x,)``, or both ``(y, x)``.
    midpoint : numpy.ndarray
        Layer-midpoint depth (m, positive down), shape ``(z, y, x)``.
    L_H, L_Z : float
        Horizontal and vertical length scales (m), both positive and finite.
    """

    def __init__(
        self,
        lat_deg: NDArray,
        lon_deg: NDArray,
        midpoint: NDArray[np.float64],
        L_H: float,
        L_Z: float,
    ) -> None:
        """Precompute the horizontal coordinates and store the depth midpoints."""
        self.L_H = _positive_finite(L_H, "L_H")
        self.L_Z = _positive_finite(L_Z, "L_Z")
        midpoint = np.asarray(midpoint, dtype=np.float64)
        if midpoint.ndim != 3:
            raise ConfigurationError(
                f"midpoint must have shape (z, y, x), got {midpoint.shape}",
                context={"shape": midpoint.shape},
            )
        _, ny, nx = midpoint.shape
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        if lat.shape == (ny,):
            lat = lat[:, None]
        elif lat.shape != (ny, nx):
            raise ConfigurationError(f"lat_deg has shape {lat.shape}, expected ({ny},) or ({ny}, {nx})")
        if lon.shape == (nx,):
            lon = lon[None, :]
        elif lon.shape != (ny, nx):
            raise ConfigurationError(f"lon_deg has shape {lon.shape}, expected ({nx},) or ({ny}, {nx})")
        phi = np.deg2rad(np.broadcast_to(lat, (ny, nx)))
        lam = np.deg2rad(np.broadcast_to(lon, (ny, nx)))
        cos_phi = np.cos(phi)
        # Horizontal coordinates are computed once, so a voxel always maps to the same
        # bits no matter which batch or call asks for it.
        self._hx = np.ascontiguousarray((EARTH_RADIUS_M * cos_phi * np.cos(lam)) / self.L_H).reshape(-1)
        self._hy = np.ascontiguousarray((EARTH_RADIUS_M * cos_phi * np.sin(lam)) / self.L_H).reshape(-1)
        self._hz = np.ascontiguousarray((EARTH_RADIUS_M * np.sin(phi)) / self.L_H).reshape(-1)
        self._depth = midpoint.reshape(-1)
        self._ncol = ny * nx

    def points(self, flat_idx: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Return the scaled coordinates of the voxels at ``flat_idx``.

        Parameters
        ----------
        flat_idx : numpy.ndarray
            Flat indices into the C-ordered ``(z, y, x)`` array.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(n, 4)``.
        """
        idx = np.asarray(flat_idx, dtype=np.int64).reshape(-1)
        col = idx % self._ncol
        out = np.empty((idx.size, 4), dtype=np.float64)
        out[:, 0] = self._hx[col]
        out[:, 1] = self._hy[col]
        out[:, 2] = self._hz[col]
        out[:, 3] = self._depth[idx] / self.L_Z
        return out


def partition_child(
    child_flat: NDArray[np.int64],
    parent_flat: Sequence[NDArray[np.int64]],
    parent_ids: NDArray[np.int32],
    coords: ScaledCoords,
    *,
    batch_size: int = 65536,
    leafsize: int = 16,
) -> NDArray[np.intp]:
    """
    Assign every voxel of a merged child to its nearest parent voxel's parent.

    Distances are squared Euclidean distances between scaled coordinates
    (``coords.points``); there is no distance cap. When several parent voxels are
    exactly equidistant, the parent with the smallest ID wins, independently of the
    order of ``parent_flat``, ``leafsize`` and ``batch_size``.

    Parameters
    ----------
    child_flat : numpy.ndarray
        Flat voxel indices of the child.
    parent_flat : sequence of numpy.ndarray
        Flat voxel indices of each parent, aligned with ``parent_ids``.
    parent_ids : numpy.ndarray
        Parent IDs, one per entry of ``parent_flat``.
    coords : ScaledCoords
        Scaled coordinates of the grid voxels.
    batch_size : int, default=65536
        Number of child voxels queried at a time.
    leafsize : int, default=16
        Leaf size of the KD-tree over the parent voxels.

    Returns
    -------
    numpy.ndarray
        ``intp`` array, one entry per child voxel, indexing ``parent_ids``.
    """
    if int(batch_size) < 1:
        raise ConfigurationError(f"batch_size must be at least 1, got {batch_size}")
    if int(leafsize) < 1:
        raise ConfigurationError(f"leafsize must be at least 1, got {leafsize}")
    batch_size, leafsize = int(batch_size), int(leafsize)

    parent_ids = np.asarray(parent_ids)
    if len(parent_flat) != len(parent_ids):
        raise TrackingError(
            f"parent_flat has {len(parent_flat)} entries but parent_ids has {len(parent_ids)}",
            context={"n_parent_flat": len(parent_flat), "n_parent_ids": len(parent_ids)},
        )
    child_flat = np.asarray(child_flat, dtype=np.int64).reshape(-1)
    n_child = child_flat.size

    sizes = np.fromiter((np.asarray(p).size for p in parent_flat), dtype=np.int64, count=len(parent_flat))
    if sizes.sum() == 0:
        raise TrackingError(
            "A merged child has no parent voxels to partition against",
            suggestions=["Check that the parents come from the previous timestep's accepted parent objects"],
            context={"n_parents": len(parent_ids)},
        )
    if n_child == 0:
        return np.empty(0, dtype=np.intp)

    # Parents with no voxel can never win; keep their indices for the final mapping.
    keep = np.flatnonzero(sizes > 0)
    if keep.size == 1:
        return np.full(n_child, keep[0], dtype=np.intp)

    seed_flat = np.concatenate([np.asarray(parent_flat[j], dtype=np.int64).reshape(-1) for j in keep])
    owner = np.repeat(keep, sizes[keep]).astype(np.intp)  # index into parent_ids, per seed voxel
    seed_pts = coords.points(seed_flat)
    seed_id = parent_ids[owner]
    tree = cKDTree(seed_pts, leafsize=leafsize)
    k = 2 if len(seed_pts) >= 2 else 1

    out = np.empty(n_child, dtype=np.intp)
    n_tied = 0
    for start in range(0, n_child, batch_size):
        stop = min(start + batch_size, n_child)
        q = coords.points(child_flat[start:stop])
        if k == 1:
            _, nn = tree.query(q, k=1)
            out[start:stop] = owner[nn]
            continue
        _, nn = tree.query(q, k=2)
        d0 = _squared_distance(q, seed_pts[nn[:, 0]])
        d1 = _squared_distance(q, seed_pts[nn[:, 1]])
        # The tree's distances are not bit-identical to the exact recomputation, so order the
        # pair by the exact values; anything within the widening of a tie goes to the exact search.
        first = np.where(d1 < d0, nn[:, 1], nn[:, 0])
        best = out[start:stop]
        best[:] = owner[first]
        near = d1 <= d0 * (1.0 + _BALL_WIDEN) ** 2
        near_rows = np.flatnonzero(near)
        if near_rows.size == 0:
            continue
        n_tied += near_rows.size
        dmin = np.minimum(d0[near_rows], d1[near_rows])
        radius = np.sqrt(dmin) * (1.0 + _BALL_WIDEN)
        cands = tree.query_ball_point(q[near_rows], radius)
        for row, extra, cand in zip(near_rows, nn[near_rows], cands):
            cand_idx = np.unique(np.concatenate([np.asarray(cand, dtype=np.intp), extra]))
            dist = _squared_distance(q[row][None, :], seed_pts[cand_idx])
            tied = cand_idx[dist == dist.min()]
            winner_id = seed_id[tied].min()
            best[row] = owner[tied[seed_id[tied] == winner_id][0]]
    if n_tied:
        logger.debug(f"Partition: {n_tied} of {n_child} child voxels needed the exact tie search")
    return out
