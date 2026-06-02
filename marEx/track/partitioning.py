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
def partition_nn_grid(
    child_mask: NDArray[np.bool_],
    parent_masks: NDArray[np.bool_],
    child_ids: NDArray[np.int32],
    parent_centroids: NDArray[np.float64],
    Nx: int,
    max_distance: int = 20,
    wrap: bool = True,
) -> NDArray[np.int32]:  # pragma: no cover
    """
    Partition a child object based on nearest parent object points.

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

    # Pre-allocate arrays
    distances = np.full(n_points, np.inf, dtype=np.float32)
    parent_assignments = np.full(n_points, -1, dtype=np.int32)
    visited = np.zeros((n_parents, n_points), dtype=np.bool_)

    # Initialise with direct overlaps
    for parent_idx in range(n_parents):
        overlap_mask = parent_masks[parent_idx] & child_mask
        if np.any(overlap_mask):
            visited[parent_idx, overlap_mask] = True
            unclaimed_overlap = distances[overlap_mask] == np.inf
            if np.any(unclaimed_overlap):
                overlap_points = np.where(overlap_mask)[0].astype(np.int32)
                valid_points = overlap_points[unclaimed_overlap]
                distances[valid_points] = 0
                parent_assignments[valid_points] = parent_idx

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
            # Get current frontier points
            frontier_mask = visited[parent_idx]
            if not np.any(frontier_mask):
                continue

            # Process neighbors
            for i in range(3):  # For each neighbor direction
                neighbors = neighbours_int[i, frontier_mask]
                valid_neighbors = neighbors >= 0
                if not np.any(valid_neighbors):
                    continue

                valid_points = neighbors[valid_neighbors]
                unvisited = ~visited[parent_idx, valid_points]
                new_points = valid_points[unvisited]

                if len(new_points) > 0:
                    visited[parent_idx, new_points] = True
                    update_mask = distances[new_points] > current_distance
                    if np.any(update_mask):
                        points_to_update = new_points[update_mask]
                        distances[points_to_update] = current_distance
                        parent_assignments[points_to_update] = parent_idx
                        updates_made = True

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
            parent_assignments[point] = np.argmin(dist).astype(np.int32)

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

    n_parents = np.max(parent_frontiers_working[parent_frontiers_working < 255]) + 1

    # Graph traversal - expanding frontiers
    current_distance = 0
    any_unassigned = np.any(child_mask_working & (parent_frontiers_working == 255))

    while current_distance < max_distance and any_unassigned:
        current_distance += 1
        updates_made = False

        for parent_idx in range(n_parents):
            # Skip if no frontier points for this parent
            if not np.any(parent_frontiers_working == parent_idx):
                continue

            # Process neighbours for current parent's frontier
            for i in range(3):
                neighbors = neighbours_int[i, parent_frontiers_working == parent_idx]
                valid_neighbors = neighbors >= 0

                if not np.any(valid_neighbors):
                    continue

                valid_points = neighbors[valid_neighbors]
                unvisited = parent_frontiers_working[valid_points] == 255

                if not np.any(unvisited):
                    continue

                # Update new frontier points
                new_points = valid_points[unvisited]
                parent_frontiers_working[new_points] = parent_idx

                if np.any(child_mask_working[new_points]):
                    updates_made = True

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
