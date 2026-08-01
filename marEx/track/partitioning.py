"""
MarEx Track: Partitioning and distance helpers.

Module-level, JIT-compiled pure functions for partitioning merged objects among
their parent objects, and for the wrapped (periodic) Euclidian distance
calculations they rely on. These operate on both structured (regular grid) and
unstructured grids and take all inputs explicitly (no shared tracker state).
"""

from typing import Tuple

import numpy as np
from numba import jit, prange
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt


@jit(nopython=True, parallel=True, fastmath=True)
def wrapped_euclidian_distance_mask_parallel(
    mask_values: NDArray[np.bool_],
    parent_centroids_values: NDArray[np.float64],
    Nx: int,
    wrap: bool,
) -> NDArray[np.float64]:  # pragma: no cover
    """
    Optimised function for computing wrapped Euclidean distances.

    Efficiently calculates distances between points in a binary mask and a set of
    centroids, accounting for periodic boundaries in the x dimension.

    Parameters
    ----------
    mask_values : np.ndarray
        2D boolean array where True indicates points to calculate distances for
    parent_centroids_values : np.ndarray
        Array of shape (n_parents, 2) containing (y, x) coordinates of parent centroids
    Nx : int
        Size of the x-dimension for periodic boundary wrapping
    wrap : bool
        Whether to treat x-dimension as periodic and wrap

    Returns
    -------
    distances : np.ndarray
        Array of shape (n_true_points, n_parents) with minimum distances
    """
    n_parents = len(parent_centroids_values)
    half_Nx = Nx / 2

    y_indices, x_indices = np.nonzero(mask_values)
    n_true = len(y_indices)

    distances = np.empty((n_true, n_parents), dtype=np.float64)

    # Precompute for faster access
    parent_y = parent_centroids_values[:, 0]
    parent_x = parent_centroids_values[:, 1]

    # Parallel loop over true positions
    for idx in prange(n_true):
        y, x = y_indices[idx], x_indices[idx]

        # Pre-compute y differences for all parents
        dy = y - parent_y

        # Pre-compute x differences for all parents
        dx = x - parent_x

        # Wrapping correction
        if wrap:
            dx = np.where(dx > half_Nx, dx - Nx, dx)
            dx = np.where(dx < -half_Nx, dx + Nx, dx)

        distances[idx] = np.sqrt(dy * dy + dx * dx)

    return distances


@jit(nopython=True, fastmath=True)
def create_grid_index_arrays(
    points_y: NDArray[np.int32],
    points_x: NDArray[np.int32],
    grid_size: int,
    ny: int,
    nx: int,
) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:  # pragma: no cover
    """
    Create a grid-based spatial index for efficient point lookup.

    This function divides space into a grid and assigns points to grid cells
    for more efficient spatial queries compared to brute force comparisons.

    Parameters
    ----------
    points_y, points_x : np.ndarray
        Coordinates of points to index
    grid_size : int
        Size of each grid cell
    ny, nx : int
        Dimensions of the overall grid

    Returns
    -------
    grid_points : np.ndarray
        3D array mapping grid cells to point indices
    grid_counts : np.ndarray
        2D array with count of points in each grid cell
    """
    n_grids_y = (ny + grid_size - 1) // grid_size
    n_grids_x = (nx + grid_size - 1) // grid_size
    max_points_per_cell = len(points_y)

    grid_points = np.full((n_grids_y, n_grids_x, max_points_per_cell), -1, dtype=np.int32)
    grid_counts = np.zeros((n_grids_y, n_grids_x), dtype=np.int32)

    for idx in range(len(points_y)):
        grid_y = min(points_y[idx] // grid_size, n_grids_y - 1)
        grid_x = min(points_x[idx] // grid_size, n_grids_x - 1)
        count = grid_counts[grid_y, grid_x]
        if count < max_points_per_cell:
            grid_points[grid_y, grid_x, count] = idx
            grid_counts[grid_y, grid_x] += 1

    return grid_points, grid_counts


@jit(nopython=True, fastmath=True)
def wrapped_euclidian_distance_points(
    y1: float, x1: float, y2: float, x2: float, nx: int, half_nx: float, wrap: bool
) -> float:  # pragma: no cover
    """
    Calculate distance with periodic boundary conditions in x dimension.

    Parameters
    ----------
    y1, x1 : float
        Coordinates of first point
    y2, x2 : float
        Coordinates of second point
    nx : int
        Size of x dimension
    half_nx : float
        Half the size of x dimension
    wrap : bool
        Whether to apply periodic boundary conditions in x

    Returns
    -------
    float
        Euclidean distance accounting for periodic boundary in x (or not)
    """
    dy = y1 - y2
    dx = x1 - x2

    if wrap:
        if dx > half_nx:
            dx -= nx
        elif dx < -half_nx:
            dx += nx

    return np.sqrt(dy * dy + dx * dx)


@jit(nopython=True, parallel=True, fastmath=True)
def _partition_nn_grid_gridsearch(
    child_mask: NDArray[np.bool_],
    parent_masks: NDArray[np.bool_],
    child_ids: NDArray[np.int32],
    parent_centroids: NDArray[np.float64],
    Nx: int,
    max_distance: int = 20,
    wrap: bool = True,
) -> NDArray[np.int32]:  # pragma: no cover
    """
    Legacy grid-search nearest-parent partitioner (numba). Retained for A/B
    equivalence testing against the EDT ``partition_nn_grid``; not used in production
    (its O(n_child * n_parents * neighbourhood) cost blows up as ``max_distance`` grows
    with parent size). See ``partition_nn_grid`` for the current implementation.

    This implementation uses spatial indexing and highly-threaded processing
    for efficient distance calculations. The algorithm assigns each point
    in the child object to the closest parent object.

    Parameters
    ----------
    child_mask : np.ndarray
        Binary mask of the child object
    parent_masks : np.ndarray
        List of binary masks for each parent object
    child_ids : np.ndarray
        List of IDs to assign to partitions
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) with parent centroids
    Nx : int
        Size of x dimension for periodic boundaries
    max_distance : int, default=20
        Maximum search distance
    wrap : bool, default=True
        Whether to apply periodic boundary conditions in the x dimension

    Returns
    -------
    new_labels : np.ndarray
        Array containing assigned child_ids for each point
    """
    ny, nx = child_mask.shape
    half_Nx = Nx / 2
    n_parents = len(parent_masks)
    grid_size = max(2, max_distance // 4)

    y_indices, x_indices = np.nonzero(child_mask)
    n_child_points = len(y_indices)

    min_distances = np.full(n_child_points, np.inf)
    parent_assignments = np.zeros(n_child_points, dtype=np.int32)
    found_close = np.zeros(n_child_points, dtype=np.bool_)

    for parent_idx in range(n_parents):
        py, px = np.nonzero(parent_masks[parent_idx])

        if len(py) == 0:  # Skip empty parents
            continue

        # Create grid index for this parent
        n_grids_y = (ny + grid_size - 1) // grid_size
        n_grids_x = (nx + grid_size - 1) // grid_size
        grid_points, grid_counts = create_grid_index_arrays(py, px, grid_size, ny, nx)

        # Process child points in parallel
        for child_idx in prange(n_child_points):
            if found_close[child_idx]:  # Skip if we already found an exact match
                continue

            child_y, child_x = y_indices[child_idx], x_indices[child_idx]
            grid_y = min(child_y // grid_size, n_grids_y - 1)
            grid_x = min(child_x // grid_size, n_grids_x - 1)

            min_dist_to_parent = np.inf

            # Check nearby grid cells
            for dy in range(-1, 2):
                grid_y_check = (grid_y + dy) % n_grids_y

                for dx in range(-1, 2):
                    grid_x_check = (grid_x + dx) % n_grids_x

                    # Process points in this grid cell
                    n_points = grid_counts[grid_y_check, grid_x_check]

                    for p_idx in range(n_points):
                        point_idx = grid_points[grid_y_check, grid_x_check, p_idx]
                        if point_idx == -1:
                            break

                        dist = wrapped_euclidian_distance_points(child_y, child_x, py[point_idx], px[point_idx], Nx, half_Nx, wrap)

                        if dist > max_distance:
                            continue

                        if dist < min_dist_to_parent:
                            min_dist_to_parent = dist

                        if dist < 1e-6:  # Found exact same point (within numerical precision)
                            min_dist_to_parent = dist
                            found_close[child_idx] = True
                            break

                    if found_close[child_idx]:
                        break

                if found_close[child_idx]:
                    break

            # Update assignment if this parent is closer
            if min_dist_to_parent < min_distances[child_idx]:
                min_distances[child_idx] = min_dist_to_parent
                parent_assignments[child_idx] = parent_idx

    # Handle any unassigned points using centroids
    unassigned = min_distances == np.inf
    if np.any(unassigned):
        for child_idx in np.nonzero(unassigned)[0]:
            child_y, child_x = y_indices[child_idx], x_indices[child_idx]
            min_dist = np.inf
            best_parent = 0

            for parent_idx in range(n_parents):
                # Calculate distance to centroid with periodic boundary conditions
                dist = wrapped_euclidian_distance_points(
                    child_y,
                    child_x,
                    parent_centroids[parent_idx, 0],
                    parent_centroids[parent_idx, 1],
                    Nx,
                    half_Nx,
                    wrap,
                )

                if dist < min_dist:
                    min_dist = dist
                    best_parent = parent_idx

            parent_assignments[child_idx] = best_parent

    # Convert from parent indices to child_ids
    new_labels = child_ids[parent_assignments]

    return new_labels


def partition_nn_grid(
    child_mask: NDArray[np.bool_],
    parent_masks: NDArray[np.bool_],
    child_ids: NDArray[np.int32],
    parent_centroids: NDArray[np.float64],
    Nx: int,
    max_distance: int = 20,
    wrap: bool = True,
) -> NDArray[np.int32]:
    """
    Partition a child object among its parents by nearest parent *pixel*.

    Distance-transform (feature-transform) implementation: each child pixel is assigned
    to the parent that owns the nearest pixel, provided that pixel is within
    ``max_distance`` (wrapped-Euclidean, periodic longitude when ``wrap``); child pixels
    with no parent pixel within ``max_distance`` fall back to the nearest parent
    *centroid*. Semantically equivalent to the legacy grid search
    (``_partition_nn_grid_gridsearch``) but O(bounding-box area) regardless of
    ``max_distance`` or the number of parents -- via a single
    ``scipy.ndimage.distance_transform_edt(..., return_indices=True)`` over a sub-array.

    Parameters
    ----------
    child_mask : np.ndarray
        (ny, nx) boolean mask of the merged child object.
    parent_masks : np.ndarray
        (n_parents, ny, nx) boolean masks for each parent object at t-1.
    child_ids : np.ndarray
        (n_parents,) IDs to assign to each partition.
    parent_centroids : np.ndarray
        (n_parents, 2) parent (y, x) centroids (for the fallback).
    Nx : int
        Size of the x dimension (for periodic-longitude distances).
    max_distance : int, default 20
        Maximum nearest-parent-pixel search distance, in cells.
    wrap : bool, default True
        Apply periodic boundary conditions in the x (longitude) dimension.

    Returns
    -------
    new_labels : np.ndarray
        (n_child_pixels,) assigned ``child_ids``, ordered by ``np.nonzero(child_mask)``.

    Notes
    -----
    Not bit-identical to the grid search at exactly-equidistant ties (``scipy``'s
    feature-transform tie-break differs from the grid search's first-parent-in-index
    order); it is physically equivalent (nearest parent = parent owning the nearest pixel).
    """
    ny, nx = child_mask.shape
    y_idx, x_idx = np.nonzero(child_mask)
    n_child = y_idx.size
    if n_child == 0:
        return np.empty(0, dtype=np.int32)

    n_parents = len(parent_masks)
    md = int(max_distance)
    parent_assignments = np.zeros(n_child, dtype=np.int32)
    assigned = np.zeros(n_child, dtype=bool)

    # Union of all parent pixels.
    any_parent = np.zeros((ny, nx), dtype=bool)
    for p in range(n_parents):
        any_parent |= parent_masks[p]

    if any_parent.any():
        # Restrict to a y bounding box (rows of child + parents) expanded by max_distance.
        # Latitude is non-periodic, so a max_distance margin captures every parent that could
        # be the nearest for any child pixel.
        occupied = child_mask | any_parent
        occ_rows = np.nonzero(occupied.any(axis=1))[0]
        y0 = max(int(occ_rows[0]) - md, 0)
        y1 = min(int(occ_rows[-1]) + md + 1, ny)
        occ_cols = np.nonzero(occupied.any(axis=0))[0]
        cmin, cmax = int(occ_cols[0]), int(occ_cols[-1])

        # Wrap only matters when the object lies within max_distance of BOTH antimeridian
        # edges (so a wrapped path could be the nearest). Otherwise -- the common case, and
        # all regional/no-wrap cases -- a tight x bounding box + max_distance margin is exact
        # and far cheaper than padding the full longitude width.
        if wrap and cmin < md and cmax >= nx - md:
            pad = min(md, nx // 2 + 1)
            parent_label = np.zeros((y1 - y0, nx), dtype=np.int32)
            for p in range(n_parents):
                parent_label[parent_masks[p][y0:y1, :]] = p + 1
            parent_label = np.pad(parent_label, ((0, 0), (pad, pad)), mode="wrap")
            child_col = x_idx + pad
        else:
            x0 = max(cmin - md, 0)
            x1 = min(cmax + md + 1, nx)
            parent_label = np.zeros((y1 - y0, x1 - x0), dtype=np.int32)
            for p in range(n_parents):
                parent_label[parent_masks[p][y0:y1, x0:x1]] = p + 1
            child_col = x_idx - x0

        child_row = y_idx - y0
        if parent_label.any():
            # For every pixel: distance to, and index of, the nearest parent pixel.
            dist, (iy, ix) = distance_transform_edt(parent_label == 0, return_distances=True, return_indices=True)
            nearest_parent = parent_label[iy, ix] - 1  # parent index per sub-array pixel
            child_dist = dist[child_row, child_col]
            within = child_dist <= md
            parent_assignments[within] = nearest_parent[child_row, child_col][within]
            assigned[within] = True

    # Centroid fall-back for child pixels with no parent pixel within max_distance
    # (reuses the same wrapped distance-to-centroid kernel as the nn_partitioning=False path).
    if not assigned.all():
        # Only the still-unassigned pixels need centroid distances. Passing the whole child
        # mask computed (n_child_pixels x n_parents) distances and then discarded all but
        # the unassigned rows (review finding 5.21). Both this mask and the kernel enumerate
        # pixels with np.nonzero, so the returned rows line up with ~assigned by construction.
        unassigned_mask = np.zeros_like(child_mask)
        unassigned_mask[y_idx[~assigned], x_idx[~assigned]] = True
        centroid_dist = wrapped_euclidian_distance_mask_parallel(unassigned_mask, parent_centroids, Nx, wrap)
        parent_assignments[~assigned] = np.argmin(centroid_dist, axis=1).astype(np.int32)

    return child_ids[parent_assignments].astype(np.int32)


@jit(nopython=True, fastmath=True)
def partition_nn_unstructured(
    child_mask: NDArray[np.bool_],
    parent_masks: NDArray[np.bool_],
    child_ids: NDArray[np.int32],
    parent_centroids: NDArray[np.float64],
    neighbours_int: NDArray[np.int32],
    lat: NDArray[np.float32],
    lon: NDArray[np.float32],
    max_distance: int = 20,
) -> NDArray[np.int32]:  # pragma: no cover
    """
    Partition a child object on an unstructured grid based on nearest parent points.

    This function implements an efficient algorithm for assigning each cell in a child
    object to the nearest parent object, using graph traversal and spatial distances.
    It is optimised for unstructured grids.

    Parameters
    ----------
    child_mask : np.ndarray
        1D boolean array where True indicates points in the child object
    parent_masks : np.ndarray
        2D boolean array of shape (n_parents, n_points) where True indicates points in each parent object
    child_ids : np.ndarray
        1D array containing the IDs to assign to each partition of the child object
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) containing (lat, lon) coordinates of parent centroids in degrees
    neighbours_int : np.ndarray
        2D array of shape (3, n_points) containing indices of neighboring cells for each point
    lat, lon : np.ndarray
        Latitude/longitude arrays in degrees
    max_distance : int, default=20
        Maximum number of edge hops to search for parent points

    Returns
    -------
    new_labels : np.ndarray
        1D array containing the assigned child_ids for each True point in child_mask
    """
    # Force contiguous arrays in memory for optimal vectorised performance
    child_mask = np.ascontiguousarray(child_mask)
    parent_masks = np.ascontiguousarray(parent_masks)

    n_points = len(child_mask)
    n_parents = len(parent_masks)

    # Pre-allocate arrays. Hop distances are integers in [0, max_distance], so the
    # "not yet reached" sentinel is a finite value one past the maximum rather than
    # np.inf: this function is compiled with fastmath=True, under which LLVM is licensed
    # to assume no infinities and folds ``x == np.inf`` to False. That silently broke the
    # unclaimed-overlap test below -- verified against the same source run unjitted.
    unset_distance = np.int32(max_distance + 1)
    distances = np.full(n_points, unset_distance, dtype=np.int32)
    parent_assignments = np.full(n_points, -1, dtype=np.int32)
    visited = np.zeros((n_parents, n_points), dtype=np.bool_)

    # Explicit per-parent BFS queues. A (parent, point) pair enters its queue at most
    # once, so one row of length n_points per parent is sufficient and the traversal
    # below costs O(visited) instead of re-scanning every parent's whole visited set at
    # every level (review finding 5.16).
    queue = np.empty((n_parents, n_points), dtype=np.int32)
    level_start = np.zeros(n_parents, dtype=np.int32)
    level_end = np.zeros(n_parents, dtype=np.int32)

    # Initialise with direct overlaps
    for parent_idx in range(n_parents):
        for point in range(n_points):
            if parent_masks[parent_idx, point] and child_mask[point]:
                visited[parent_idx, point] = True
                queue[parent_idx, level_end[parent_idx]] = point
                level_end[parent_idx] += 1
                if distances[point] == unset_distance:
                    distances[point] = 0
                    parent_assignments[point] = parent_idx

    # Pre-compute trig values for efficiency
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    cos_lat = np.cos(lat_rad)

    # Graph traversal for remaining points - expanding from parent frontiers
    current_distance = 0
    any_unassigned = np.any(child_mask & (parent_assignments == -1))

    while current_distance < max_distance and any_unassigned:
        current_distance += 1
        updates_made = False

        for parent_idx in range(n_parents):
            # Expand only the level just added. Every neighbour of an earlier level is
            # already visited by this parent, so this reaches exactly the same points.
            start, end = level_start[parent_idx], level_end[parent_idx]
            if start == end:
                continue

            new_end = end
            for k in range(start, end):
                cell = queue[parent_idx, k]
                for i in range(3):  # For each neighbor direction
                    neighbour = neighbours_int[i, cell]
                    if neighbour < 0 or visited[parent_idx, neighbour]:
                        continue
                    visited[parent_idx, neighbour] = True
                    queue[parent_idx, new_end] = neighbour
                    new_end += 1
                    if distances[neighbour] > current_distance:
                        distances[neighbour] = current_distance
                        parent_assignments[neighbour] = parent_idx
                        updates_made = True

            level_start[parent_idx] = end
            level_end[parent_idx] = new_end

        if not updates_made:
            break

        any_unassigned = np.any(child_mask & (parent_assignments == -1))

    # Handle remaining unassigned points using great circle distances
    unassigned_mask = child_mask & (parent_assignments == -1)
    if np.any(unassigned_mask):
        parent_lat_rad = np.deg2rad(parent_centroids[:, 0])
        parent_lon_rad = np.deg2rad(parent_centroids[:, 1])
        cos_parent_lat = np.cos(parent_lat_rad)

        unassigned_points = np.where(unassigned_mask)[0].astype(np.int32)
        for point in unassigned_points:
            # Vectoised haversine calculation
            dlat = parent_lat_rad - lat_rad[point]
            dlon = parent_lon_rad - lon_rad[point]
            a = np.sin(dlat / 2) ** 2 + cos_lat[point] * cos_parent_lat * np.sin(dlon / 2) ** 2
            dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            # np.int32(...) rather than .astype(): numba's np.argmin returns a plain int64
            # scalar, which has no .astype, so the previous form made this whole function
            # fail to compile -- it raised TypingError on every call, taken branch or not.
            # Matches partition_nn_unstructured_optimised, which always had it right.
            parent_assignments[point] = np.int32(np.argmin(dist))

    # Return only the assignments for points in child_mask
    child_points = np.where(child_mask)[0].astype(np.int32)
    return child_ids[parent_assignments[child_points]]


@jit(nopython=True, fastmath=True)
def partition_nn_unstructured_optimised(
    child_mask: NDArray[np.bool_],
    parent_frontiers: NDArray[np.uint8],
    parent_centroids: NDArray[np.float64],
    neighbours_int: NDArray[np.int32],
    lat: NDArray[np.float32],
    lon: NDArray[np.float32],
    max_distance: int = 20,
) -> NDArray[np.uint8]:  # pragma: no cover
    """
    Memory-optimised nearest neighbor partitioning for unstructured grids.

    This version uses more efficient memory management compared to partition_nn_unstructured,
    making it suitable for very large grids. It uses a compact representation of parent
    frontiers to reduce memory usage during graph traversal.

    Parameters
    ----------
    child_mask : np.ndarray
        1D boolean array indicating which cells belong to the child object
    parent_frontiers : np.ndarray
        1D uint8 array with parent indices (255 for unvisited points)
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) containing (lat, lon) coordinates
    neighbours_int : np.ndarray
        2D array of shape (3, n_points) containing indices of neighboring cells
    lat, lon : np.ndarray
        1D arrays of latitude/longitude in degrees
    max_distance : int, default=20
        Maximum number of edge hops to search for parent points

    Returns
    -------
    result : np.ndarray
        1D array containing the assigned parent indices for points in child_mask
    """
    # Create working copies to ensure memory cleanup
    parent_frontiers_working = parent_frontiers.copy()
    child_mask_working = child_mask.copy()

    n_points = len(child_mask_working)
    n_parents = np.max(parent_frontiers_working[parent_frontiers_working < 255]) + 1

    # Explicit BFS frontier queue. Every point is claimed at most once (a claimed point
    # is never revisited), so a single queue of length n_points holds the entire
    # traversal. Expanding only the level just added replaces re-scanning the whole
    # field once per parent per direction per level, which was O(n_parents x
    # max_distance x n_points) -- ~1e9 operations per merge event at ICON scale
    # (review finding 5.16).
    queue = np.empty(n_points, dtype=np.int32)
    n_queued = 0
    # Seed in parent-major order so that, within a level, a lower parent index still
    # claims a contested point first -- the tie-break the scan-based version had.
    for parent_idx in range(n_parents):
        for point in range(n_points):
            if parent_frontiers_working[point] == parent_idx:
                queue[n_queued] = point
                n_queued += 1

    # Graph traversal - expanding frontiers
    current_distance = 0
    any_unassigned = np.any(child_mask_working & (parent_frontiers_working == 255))
    level_start = 0
    level_end = n_queued

    while current_distance < max_distance and any_unassigned:
        current_distance += 1
        updates_made = False

        new_end = level_end
        for k in range(level_start, level_end):
            cell = queue[k]
            parent_idx = parent_frontiers_working[cell]
            for i in range(3):
                neighbour = neighbours_int[i, cell]
                if neighbour < 0 or parent_frontiers_working[neighbour] != 255:
                    continue
                parent_frontiers_working[neighbour] = parent_idx
                queue[new_end] = neighbour
                new_end += 1
                if child_mask_working[neighbour]:
                    updates_made = True

        level_start = level_end
        level_end = new_end

        if not updates_made:
            break

        any_unassigned = np.any(child_mask_working & (parent_frontiers_working == 255))

    # Handle remaining unassigned points using great circle distances
    unassigned_mask = child_mask_working & (parent_frontiers_working == 255)
    if np.any(unassigned_mask):
        # Pre-compute parent coordinates in radians
        parent_lat_rad = np.deg2rad(parent_centroids[:, 0])
        parent_lon_rad = np.deg2rad(parent_centroids[:, 1])
        cos_parent_lat = np.cos(parent_lat_rad)

        # Process each unassigned point
        unassigned_points = np.where(unassigned_mask)[0].astype(np.int32)
        for point in unassigned_points:
            dlat = parent_lat_rad - np.deg2rad(lat[point])
            dlon = parent_lon_rad - np.deg2rad(lon[point])

            a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat[point])) * cos_parent_lat * np.sin(dlon / 2) ** 2
            dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            parent_frontiers_working[point] = np.int32(np.argmin(dist))

    # Extract result for child points only
    result = parent_frontiers_working[child_mask_working].copy()

    # Explicitly clear working arrays to help with memory management
    parent_frontiers_working = None
    child_mask_working = None

    return result


@jit(nopython=True, parallel=True, fastmath=True)
def partition_centroid_unstructured(
    child_mask: NDArray[np.bool_],
    parent_centroids: NDArray[np.float64],
    child_ids: NDArray[np.int32],
    lat: NDArray[np.float32],
    lon: NDArray[np.float32],
) -> NDArray[np.int32]:  # pragma: no cover
    """
    Partition a child object based on closest parent centroids on an unstructured grid.

    This function assigns each cell in the child object to the parent with the closest
    centroid, using great circle distances on a spherical grid.

    Parameters
    ----------
    child_mask : np.ndarray
        1D boolean array indicating which cells belong to the child object
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) containing (lat, lon) coordinates of parent centroids in degrees
    child_ids : np.ndarray
        Array of IDs to assign to each partition of the child object
    lat, lon : np.ndarray
        Latitude/longitude arrays in degrees

    Returns
    -------
    new_labels : np.ndarray
        1D array containing assigned child_ids for cells in child_mask
    """
    n_cells = len(child_mask)
    n_parents = len(parent_centroids)

    # Convert to radians for spherical calculations
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    parent_coords_rad = np.deg2rad(parent_centroids)

    new_labels = np.zeros(n_cells, dtype=child_ids.dtype)

    # Process each child cell in parallel
    for i in prange(n_cells):
        if not child_mask[i]:
            continue

        min_dist = np.inf
        closest_parent = 0

        # Calculate great circle distance to each parent centroid
        for j in range(n_parents):
            dlat = parent_coords_rad[j, 0] - lat_rad[i]
            dlon = parent_coords_rad[j, 1] - lon_rad[i]

            # Use haversine formula for great circle distance
            a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad[i]) * np.cos(parent_coords_rad[j, 0]) * np.sin(dlon / 2) ** 2
            dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            if dist < min_dist:
                min_dist = dist
                closest_parent = j

        new_labels[i] = child_ids[closest_parent]

    return new_labels
