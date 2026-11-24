"""
MarEx Track: nearest-parent-voxel partition of a merged child.

A merged child is divided between its parents by assigning every child voxel to
the nearest voxel of any parent, under a scaled metric: the horizontal part is the
3-D chord between points on a sphere of radius ``EARTH_RADIUS_M`` divided by ``L_H``,
the vertical part is the layer-midpoint depth divided by ``L_Z``. The chord needs no
wrap-around logic, so the dateline seam and regional domains need no special case.
There is no distance cap: a nearest parent voxel always exists.

Squared distances are evaluated in a fixed float64 order from the stored scaled
coordinates. A tie (exact equality) goes to the smallest parent ID.
"""

from typing import Optional, Sequence

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ...exceptions import ConfigurationError, TrackingError
from .geometry import EARTH_RADIUS_M


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
        Layer-midpoint depth (m, positive down), shape ``(z,)`` or ``(z, y, x)``.
    L_H, L_Z : float
        Horizontal and vertical length scales (m), both positive and finite.
    wet : numpy.ndarray, optional
        Wet mask. For depth-only midpoints, dry voxels read as zero depth.
    """

    def __init__(
        self,
        lat_deg: NDArray,
        lon_deg: NDArray,
        midpoint: NDArray[np.float64],
        L_H: float,
        L_Z: float,
        *,
        wet: Optional[NDArray[np.bool_]] = None,
    ) -> None:
        """Precompute the horizontal coordinates and store the depth midpoints."""
        self.L_H = _positive_finite(L_H, "L_H")
        self.L_Z = _positive_finite(L_Z, "L_Z")
        midpoint = np.asarray(midpoint, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        if midpoint.ndim == 3:
            _, ny, nx = midpoint.shape
        elif midpoint.ndim == 1 and lat.ndim > 0 and lon.ndim > 0:
            ny, nx = lat.shape[0], lon.shape[-1]
        else:
            raise ConfigurationError(
                f"midpoint must have shape (z,) or (z, y, x), got {midpoint.shape}",
                context={"shape": midpoint.shape},
            )
        if midpoint.ndim == 1 and wet is not None and wet.shape != (midpoint.size, ny, nx):
            raise ConfigurationError(f"wet has shape {wet.shape}, expected {(midpoint.size, ny, nx)}")
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
        with np.errstate(over="ignore"):
            self._hx = np.ascontiguousarray((EARTH_RADIUS_M * cos_phi * np.cos(lam)) / self.L_H).reshape(-1)
            self._hy = np.ascontiguousarray((EARTH_RADIUS_M * cos_phi * np.sin(lam)) / self.L_H).reshape(-1)
            self._hz = np.ascontiguousarray((EARTH_RADIUS_M * np.sin(phi)) / self.L_H).reshape(-1)
        self._depth = midpoint
        self._wet = wet
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
        if self._depth.ndim == 1:
            depth = self._depth[idx // self._ncol]
            if self._wet is not None:
                depth = np.where(self._wet.flat[idx], depth, 0.0)
        else:
            depth = self._depth.flat[idx]
        out[:, 3] = depth / self.L_Z
        return out


def _finite_points(coords, flat, role):
    """Gather stored coordinates and reject non-finite values before tree arithmetic."""
    with np.errstate(over="ignore"):
        points = np.ascontiguousarray(coords.points(flat), dtype=np.float64)
    if points.shape != (flat.size, 4):
        raise TrackingError(f"Scaled {role} coordinates have shape {points.shape}, expected {(flat.size, 4)}")
    if not np.isfinite(points).all():
        raise TrackingError(f"Non-finite scaled coordinates among {role} voxels")
    return points


def _seed_columns(coords, seed_flat, points):
    """Return horizontal triples, groups of seed indices and the grid lookup."""
    if not isinstance(coords, ScaledCoords):
        horizontal, column = np.unique(points[:, :3], axis=0, return_inverse=True)
        groups = [[] for _ in horizontal]
        for i, group in enumerate(column.ravel()):
            groups[group].append(i)
        return horizontal, groups, {}

    # Compact only the occupied grid columns in a single pass. Distinct grid columns
    # can have the same horizontal triple and remain distinct leaves of the search.
    by_column = {}
    for i, grid_col in enumerate(seed_flat % coords._ncol):
        by_column.setdefault(grid_col, []).append(i)
    groups = list(by_column.values())
    first = [group[0] for group in groups]
    lookup = {grid_col: i for i, grid_col in enumerate(by_column)}
    return points[first, :3], groups, lookup


def _column_tree(horizontal, leafsize):
    """Build a balanced tree with exact horizontal bounds and contiguous leaf ranges."""
    order = np.arange(len(horizontal), dtype=np.intp)
    # Each median split halves the maximum possible node size, rounding up.
    depth = ((len(order) - 1) // leafsize).bit_length()
    capacity = (2 << depth) - 1
    bounds = np.empty((capacity, 6), dtype=np.float64)
    nodes = np.empty((capacity, 3), dtype=np.intp)
    count = 0

    def build(start, stop):
        nonlocal count
        node = count
        count += 1
        xyz = horizontal[order[start:stop]]
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        bounds[node, :3], bounds[node, 3:] = lo, hi
        if stop - start <= leafsize:
            nodes[node] = (-1, start, stop)
        else:
            axis = (hi - lo).argmax()
            half = (stop - start) // 2
            order[start:stop] = order[start:stop][np.argpartition(xyz[:, axis], half)]
            left = build(start, start + half)
            right = build(start + half, stop)
            nodes[node] = (left, right, -1)
        return node

    # The split heuristic may overflow for opposite extreme finite coordinates.
    # An infinite span is still a valid choice of split axis.
    with np.errstate(over="ignore"):
        build(0, len(order))
    return order, nodes[:count].ravel(), bounds[:count].ravel()


@njit(cache=True, nogil=True)
def _query_columns(points, own_columns, horizontal, spans, depths, owners, nodes, bounds, stack, lower, out):
    """Traverse the tree for a batch, deferring the farther child at each split."""
    # Packed rows: four coordinates per query, three coordinates and two seed
    # offsets per column, three indices and six box coordinates per tree node.
    for row in range(len(out)):
        query = 4 * row
        qx, qy, qz, qd = points[query], points[query + 1], points[query + 2], points[query + 3]
        best_dist, best_owner = np.inf, len(owners)
        own = own_columns[row]
        pending = 1
        stack[0], lower[0] = 0, 0.0
        if own >= 0:
            # Score the query's own column first as a one-column leaf.
            stack[1], lower[1] = -1, 0.0
            pending = 2
        while pending:
            pending -= 1
            if lower[pending] > best_dist:
                continue
            node = stack[pending]
            left = nodes[3 * node] if node >= 0 else -1
            if left >= 0:
                right = nodes[3 * node + 1]
                stack[pending], stack[pending + 1] = right, left
                for i in range(pending, pending + 2):
                    box = 6 * stack[i]
                    h2 = 0.0  # Adding the first square to zero is exact.
                    for axis in range(3):
                        value = points[query + axis]
                        lo, hi = bounds[box + axis], bounds[box + axis + 3]
                        clamped = lo if value < lo else hi if value > hi else value
                        gap = value - clamped
                        h2 += gap * gap
                    # Clamped gaps are no larger than any descendant's gaps. Rounded
                    # squares and fixed-order sums are monotone on these terms.
                    lower[i] = 0.0 if h2 <= 0.0 else np.nextafter(h2, -np.inf)
                if lower[pending] < lower[pending + 1]:
                    stack[pending], stack[pending + 1] = left, right
                    lower[pending], lower[pending + 1] = lower[pending + 1], lower[pending]
                pending += 2
                continue

            if node < 0:
                first, last = own, own + 1
            else:
                first, last = nodes[3 * node + 1], nodes[3 * node + 2]
            for col in range(first, last):
                if node >= 0 and col == own:
                    continue
                xyz = 3 * col
                dx = qx - horizontal[xyz]
                dy = qy - horizontal[xyz + 1]
                dz = qz - horizontal[xyz + 2]
                h2 = (dx * dx + dy * dy) + dz * dz
                bound = 0.0 if h2 <= 0.0 else np.nextafter(h2, -np.inf)
                if bound > best_dist:
                    continue

                start, stop = spans[2 * col], spans[2 * col + 1]
                lo, hi = start, stop
                while lo < hi:
                    mid = lo + ((hi - lo) >> 1)
                    if depths[mid] < qd:
                        lo = mid + 1
                    else:
                        hi = mid

                # The rounded distances never decrease outwards in either direction.
                # Unequal vertical squares can round to the same full distance.
                for side in range(2):
                    step = 2 * side - 1
                    i = lo + side - 1
                    while start <= i < stop:
                        dd = qd - depths[i]
                        dist = h2 + dd * dd
                        if dist > best_dist:
                            break
                        owner = owners[i]
                        if dist < best_dist or owner < best_owner:
                            best_dist = dist
                            best_owner = owner
                        i += step
        out[row] = best_owner


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
    coords : ScaledCoords or object with a points method
        Finite scaled coordinates of the grid voxels.
    batch_size : int, default=65536
        Number of child voxels queried at a time.
    leafsize : int, default=16
        Maximum number of horizontal seed columns in a tree leaf.

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
    # Rank nonempty parents by ID (with original indices as the fallback).
    # The kernel compares ranks, and keep maps them back to input indices.
    keep = np.flatnonzero(sizes > 0)
    keep = keep[np.argsort(parent_ids[keep], kind="stable")]
    seed_flat = np.concatenate([np.asarray(parent_flat[j], dtype=np.int64).reshape(-1) for j in keep])
    points = _finite_points(coords, seed_flat, "parent")
    out = np.empty(n_child, dtype=np.intp)
    if not n_child:
        return out
    if keep.size == 1:
        for start in range(0, n_child, batch_size):
            _finite_points(coords, child_flat[start : start + batch_size], "child")
        out.fill(keep[0])
        return out

    horizontal, groups, lookup = _seed_columns(coords, seed_flat, points)
    owner = np.repeat(np.arange(keep.size, dtype=np.intp), sizes[keep])
    depths = np.empty(len(points), dtype=np.float64)
    owners = np.empty(len(points), dtype=np.intp)
    offsets = np.empty(len(groups) + 1, dtype=np.intp)
    stop = 0
    for col, group in enumerate(groups):
        start, stop = stop, stop + len(group)
        offsets[col] = start
        col_depths = points[group, 3]
        col_owners = owner[group]
        if len(group) > 1:
            vertical_order = np.lexsort((col_owners, col_depths))
            col_depths = col_depths[vertical_order]
            col_owners = col_owners[vertical_order]
        depths[start:stop] = col_depths
        owners[start:stop] = col_owners
    offsets[-1] = stop

    order, nodes, bounds = _column_tree(horizontal, leafsize)
    horizontal = horizontal[order].ravel()
    spans = np.column_stack((offsets[order], offsets[order + 1])).ravel()
    if lookup:
        tree_col = np.empty_like(order)
        tree_col[order] = np.arange(len(order), dtype=np.intp)
        lookup = {grid_col: tree_col[group] for grid_col, group in lookup.items()}

    # Median splits keep the path below 64 entries for index-sized arrays.
    stack = np.empty(64, dtype=np.intp)
    lower = np.empty(64, dtype=np.float64)
    for start in range(0, n_child, batch_size):
        stop = min(start + batch_size, n_child)
        flat = child_flat[start:stop]
        q = _finite_points(coords, flat, "child")
        if lookup:
            own = np.fromiter((lookup.get(col, -1) for col in flat % coords._ncol), dtype=np.intp, count=flat.size)
        else:
            own = np.full(flat.size, -1, dtype=np.intp)
        _query_columns(q.ravel(), own, horizontal, spans, depths, owners, nodes, bounds, stack, lower, out[start:stop])
    return keep[out]
