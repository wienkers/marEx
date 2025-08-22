"""
Unit tests for individual functions in marEx.track module.

Tests core utility functions for marine extreme tracking and partitioning.
Focuses on testing individual function behaviour rather than full pipeline integration.
"""

import numpy as np

import marEx.track as track


class TestWrappedEuclidianParallel:
    """Test wrapped_euclidian_distance_mask_parallel function for distance calculations."""

    def test_wrapped_euclidian_basic(self):
        """Test basic distance calculation without wrapping."""
        # Create simple mask with one point
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True

        # One centroid at (3, 3)
        centroids = np.array([[3.0, 3.0]])

        result = track.wrapped_euclidian_distance_mask_parallel(mask, centroids, Nx=10, wrap=True)

        # Distance should be sqrt((5-3)^2 + (5-3)^2) = sqrt(8) ≈ 2.828
        expected = np.sqrt(8)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], expected, atol=1e-6)

    def test_wrapped_euclidian_with_wrapping(self):
        """Test distance calculation with periodic boundary conditions."""
        # Create mask with point near right edge
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 9] = True  # Point at (5, 9)

        # Centroid at left edge (5, 1)
        centroids = np.array([[5.0, 1.0]])

        result = track.wrapped_euclidian_distance_mask_parallel(mask, centroids, Nx=10, wrap=True)

        # Without wrapping: distance would be sqrt((5-5)^2 + (9-1)^2) = 8
        # With wrapping: distance should be sqrt((5-5)^2 + (1-9+10)^2) = sqrt(4) = 2
        expected = 2.0
        assert np.isclose(result[0, 0], expected, atol=1e-6)

    def test_wrapped_euclidian_multiple_points(self):
        """Test distance calculation with multiple points and centroids."""
        # Create mask with multiple points
        mask = np.zeros((10, 10), dtype=bool)
        mask[2, 2] = True
        mask[8, 8] = True

        # Multiple centroids
        centroids = np.array([[1.0, 1.0], [7.0, 7.0]])

        result = track.wrapped_euclidian_distance_mask_parallel(mask, centroids, Nx=10, wrap=True)

        assert result.shape == (2, 2)  # 2 points, 2 centroids

        # Point (2,2) should be closer to centroid (1,1)
        assert result[0, 0] < result[0, 1]

        # Point (8,8) should be closer to centroid (7,7)
        assert result[1, 1] < result[1, 0]

    def test_wrapped_euclidian_edge_cases(self):
        """Test edge cases for wrapped distance calculation."""
        # Test with empty mask
        mask = np.zeros((5, 5), dtype=bool)
        centroids = np.array([[2.0, 2.0]])

        result = track.wrapped_euclidian_distance_mask_parallel(mask, centroids, Nx=5, wrap=True)
        assert result.shape == (0, 1)

        # Test with point at same location as centroid
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        centroids = np.array([[2.0, 2.0]])

        result = track.wrapped_euclidian_distance_mask_parallel(mask, centroids, Nx=5, wrap=True)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 0.0, atol=1e-6)


class TestCalculateWrappedDistance:
    """Test wrapped_euclidian_distance_points function for single distance calculations."""

    def test_wrapped_euclidian_distance_points_basic(self):
        """Test basic distance calculation without wrapping."""
        distance = track.wrapped_euclidian_distance_points(3.0, 4.0, 0.0, 0.0, nx=10, half_nx=5.0, wrap=False)

        # Distance should be sqrt((3-0)^2 + (4-0)^2) = 5.0
        expected = 5.0
        assert np.isclose(distance, expected, atol=1e-6)

    def test_wrapped_euclidian_distance_points_x_wrapping(self):
        """Test distance calculation with x-axis wrapping."""
        # Point at x=9, centroid at x=1, in 10-wide grid
        # Normal distance would be 8, wrapped should be 2
        distance = track.wrapped_euclidian_distance_points(0.0, 9.0, 0.0, 1.0, nx=10, half_nx=5.0, wrap=True)

        expected = 2.0  # Wrapped distance
        assert np.isclose(distance, expected, atol=1e-6)

    def test_wrapped_euclidian_distance_points_negative_wrapping(self):
        """Test distance calculation with negative x wrapping."""
        # Point at x=1, centroid at x=9, in 10-wide grid
        distance = track.wrapped_euclidian_distance_points(0.0, 1.0, 0.0, 9.0, nx=10, half_nx=5.0, wrap=True)

        expected = 2.0  # Wrapped distance (same as above, symmetric)
        assert np.isclose(distance, expected, atol=1e-6)

    def test_wrapped_euclidian_distance_points_no_y_wrapping(self):
        """Test that y-axis doesn't wrap (only x-axis should wrap)."""
        # Large y difference should not wrap
        distance = track.wrapped_euclidian_distance_points(0.0, 0.0, 9.0, 0.0, nx=10, half_nx=5.0, wrap=True)

        expected = 9.0  # No wrapping in y direction
        assert np.isclose(distance, expected, atol=1e-6)

    def test_wrapped_euclidian_distance_points_exact_half(self):
        """Test distance calculation at exactly half the grid width."""
        # At exactly half the grid width, should not wrap
        distance = track.wrapped_euclidian_distance_points(0.0, 0.0, 0.0, 5.0, nx=10, half_nx=5.0, wrap=True)

        expected = 5.0  # Should not wrap at exactly half
        assert np.isclose(distance, expected, atol=1e-6)


class TestCreateGridIndexArrays:
    """Test create_grid_index_arrays function for spatial indexing."""

    def test_create_grid_index_basic(self):
        """Test basic grid index creation."""
        # Points in a 10x10 grid with grid_size=5
        points_y = np.array([1, 6, 8], dtype=np.int32)
        points_x = np.array([2, 3, 7], dtype=np.int32)

        grid_points, grid_counts = track.create_grid_index_arrays(points_y, points_x, grid_size=5, ny=10, nx=10)

        # Should create 2x2 grid (10/5 = 2)
        assert grid_points.shape == (2, 2, 3)  # 2x2 grid, max 3 points
        assert grid_counts.shape == (2, 2)

        # Check that points are assigned to correct grid cells
        # Point (1,2) should be in grid cell (0,0)
        # Point (6,3) should be in grid cell (1,0)
        # Point (8,7) should be in grid cell (1,1)
        assert grid_counts[0, 0] == 1  # One point in cell (0,0)
        assert grid_counts[1, 0] == 1  # One point in cell (1,0)
        assert grid_counts[1, 1] == 1  # One point in cell (1,1)
        assert grid_counts[0, 1] == 0  # No points in cell (0,1)

    def test_create_grid_index_boundary_cases(self):
        """Test grid index creation with boundary cases."""
        # Points exactly at boundaries
        points_y = np.array([0, 4, 5, 9], dtype=np.int32)
        points_x = np.array([0, 4, 5, 9], dtype=np.int32)

        grid_points, grid_counts = track.create_grid_index_arrays(points_y, points_x, grid_size=5, ny=10, nx=10)

        # Points at (0,0) and (4,4) should be in grid cell (0,0)
        # Points at (5,5) and (9,9) should be in grid cell (1,1)
        assert grid_counts[0, 0] == 2
        assert grid_counts[1, 1] == 2
        assert grid_counts[0, 1] == 0
        assert grid_counts[1, 0] == 0

    def test_create_grid_index_empty(self):
        """Test grid index creation with no points."""
        points_y = np.array([], dtype=np.int32)
        points_x = np.array([], dtype=np.int32)

        grid_points, grid_counts = track.create_grid_index_arrays(points_y, points_x, grid_size=5, ny=10, nx=10)

        assert grid_points.shape == (2, 2, 0)
        assert grid_counts.shape == (2, 2)
        assert np.all(grid_counts == 0)

    def test_create_grid_index_overflow_protection(self):
        """Test that grid index handles points at exact grid boundaries."""
        # Point exactly at the edge of the grid
        points_y = np.array([9], dtype=np.int32)
        points_x = np.array([9], dtype=np.int32)

        grid_points, grid_counts = track.create_grid_index_arrays(points_y, points_x, grid_size=5, ny=10, nx=10)

        # Should be placed in grid cell (1,1), not cause overflow
        assert grid_counts[1, 1] == 1
        assert grid_points[1, 1, 0] == 0  # First (and only) point index


class TestSparseBoolPower:
    """Test sparse_bool_power function for sparse matrix operations."""

    def test_sparse_bool_power_identity(self):
        """Test sparse boolean power with identity matrix."""
        # Create identity matrix
        n = 5
        data = np.ones(n, dtype=bool)
        indices = np.arange(n, dtype=np.int32)
        indptr = np.arange(n + 1, dtype=np.int32)

        # Test vector
        vec = np.array([True, False, True, False, True], dtype=bool)[np.newaxis, :]

        # Identity matrix to any power should return original vector
        for exponent in [1, 2, 3]:
            result = track.sparse_bool_power(vec, data, indices, indptr, exponent)
            assert np.array_equal(result, vec)

    def test_sparse_bool_power_simple_graph(self):
        """Test sparse boolean power with simple connectivity graph."""
        # Create simple 3-node linear chain: 0-1-2
        data = np.array([True, True, True, True], dtype=bool)
        indices = np.array([1, 0, 2, 1], dtype=np.int32)  # Connections: 0->1, 1->0, 1->2, 2->1
        indptr = np.array([0, 1, 3, 4], dtype=np.int32)  # Node 0: 1 connection, Node 1: 2 connections, Node 2: 1 connection

        # Start with activation at node 0
        vec = np.array([True, False, False], dtype=bool)[np.newaxis, :]

        # After 1 step: should activate node 1
        result1 = track.sparse_bool_power(vec, data, indices, indptr, 1)
        expected1 = np.array([False, True, False], dtype=bool)[np.newaxis, :]
        assert np.array_equal(result1, expected1)

        # After 2 steps: should activate nodes 0 and 2
        result2 = track.sparse_bool_power(vec, data, indices, indptr, 2)
        expected2 = np.array([True, False, True], dtype=bool)[np.newaxis, :]
        assert np.array_equal(result2, expected2)

    def test_sparse_bool_power_exponent_zero(self):
        """Test sparse boolean power with exponent zero (should return original vector)."""
        # Simple matrix (doesn't matter what it is)
        data = np.array([True, True], dtype=bool)
        indices = np.array([1, 0], dtype=np.int32)
        indptr = np.array([0, 1, 2], dtype=np.int32)

        vec = np.array([True, False], dtype=bool)[np.newaxis, :]

        # Exponent 0 should return original vector
        result = track.sparse_bool_power(vec, data, indices, indptr, 0)
        assert np.array_equal(result, vec)

    def test_sparse_bool_power_multiple_vectors(self):
        """Test sparse boolean power with multiple input vectors."""
        # Identity matrix
        n = 3
        data = np.ones(n, dtype=bool)
        indices = np.arange(n, dtype=np.int32)
        indptr = np.arange(n + 1, dtype=np.int32)

        # Multiple vectors
        vec = np.array(
            [[True, False, False], [False, True, False], [False, False, True]],
            dtype=bool,
        )

        # Identity matrix should preserve all vectors
        result = track.sparse_bool_power(vec, data, indices, indptr, 1)
        assert np.array_equal(result, vec)

    def test_sparse_bool_power_disconnected_components(self):
        """Test sparse boolean power with disconnected graph components."""
        # Create two disconnected pairs: 0-1 and 2-3
        data = np.array([True, True, True, True], dtype=bool)
        indices = np.array([1, 0, 3, 2], dtype=np.int32)
        indptr = np.array([0, 1, 2, 3, 4], dtype=np.int32)

        # Activate nodes 0 and 2
        vec = np.array([True, False, True, False], dtype=bool)[np.newaxis, :]

        # After 1 step: should activate nodes 1 and 3
        result = track.sparse_bool_power(vec, data, indices, indptr, 1)
        expected = np.array([False, True, False, True], dtype=bool)[np.newaxis, :]
        assert np.array_equal(result, expected)


class TestPartitionNNValidation:
    """Test validation and edge cases for partition_nn_grid function."""

    def test_partition_nn_grid_basic(self):
        """Test basic nearest neighbor partitioning."""
        # Create simple child mask
        child_mask = np.zeros((10, 10), dtype=bool)
        child_mask[5, 5] = True  # Single point

        # Create two parent masks
        parent_mask1 = np.zeros((10, 10), dtype=bool)
        parent_mask1[3, 3] = True  # Closer parent

        parent_mask2 = np.zeros((10, 10), dtype=bool)
        parent_mask2[8, 8] = True  # Farther parent

        parent_masks = np.array([parent_mask1, parent_mask2])
        child_ids = np.array([100, 200], dtype=np.int32)
        parent_centroids = np.array([[3.0, 3.0], [8.0, 8.0]])

        result = track.partition_nn_grid(child_mask, parent_masks, child_ids, parent_centroids, Nx=10)

        # Child point at (5,5) should be assigned to closer parent (ID 100)
        assert len(result) == 1
        assert result[0] == 100

    def test_partition_nn_grid_wrapping(self):
        """Test nearest neighbor partitioning with periodic boundaries."""
        child_mask = np.zeros((10, 10), dtype=bool)
        child_mask[5, 9] = True  # Point near right edge

        # Parent near left edge should be closer due to wrapping
        parent_mask1 = np.zeros((10, 10), dtype=bool)
        parent_mask1[5, 1] = True  # Distance 2 with wrapping

        # Parent in middle should be farther
        parent_mask2 = np.zeros((10, 10), dtype=bool)
        parent_mask2[5, 5] = True  # Distance 4 without wrapping

        parent_masks = np.array([parent_mask1, parent_mask2])
        child_ids = np.array([100, 200], dtype=np.int32)
        parent_centroids = np.array([[5.0, 1.0], [5.0, 5.0]])

        result = track.partition_nn_grid(child_mask, parent_masks, child_ids, parent_centroids, Nx=10)

        # Should choose wrapped parent (ID 100)
        assert result[0] == 100

    def test_partition_nn_grid_empty_parents(self):
        """Test partition behaviour with empty parent masks."""
        child_mask = np.zeros((5, 5), dtype=bool)
        child_mask[2, 2] = True

        # One empty parent, one with points
        parent_mask1 = np.zeros((5, 5), dtype=bool)  # Empty
        parent_mask2 = np.zeros((5, 5), dtype=bool)
        parent_mask2[1, 1] = True

        parent_masks = np.array([parent_mask1, parent_mask2])
        child_ids = np.array([100, 200], dtype=np.int32)
        parent_centroids = np.array([[0.0, 0.0], [1.0, 1.0]])

        result = track.partition_nn_grid(child_mask, parent_masks, child_ids, parent_centroids, Nx=5)

        # Should fall back to centroid-based assignment
        assert result[0] in [100, 200]  # Should get one of the IDs

    def test_partition_nn_grid_max_distance(self):
        """Test that max_distance parameter works correctly."""
        child_mask = np.zeros((20, 20), dtype=bool)
        child_mask[10, 10] = True

        # Parent very far away
        parent_mask = np.zeros((20, 20), dtype=bool)
        parent_mask[0, 0] = True  # Distance > 10

        parent_masks = np.array([parent_mask])
        child_ids = np.array([100], dtype=np.int32)
        parent_centroids = np.array([[0.0, 0.0]])

        result = track.partition_nn_grid(child_mask, parent_masks, child_ids, parent_centroids, Nx=20, max_distance=5)

        # Should still assign to the only available parent (fallback to centroid)
        assert result[0] == 100


class TestDistanceCalculationValidation:
    """Test validation of distance calculation edge cases."""

    def test_wrapped_distance_symmetry(self):
        """Test that wrapped distance is symmetric."""
        # Test multiple point pairs
        test_cases = [
            (0, 0, 5, 5),
            (0, 9, 0, 1),  # Wrapping case
            (3, 2, 7, 8),
        ]

        for y1, x1, y2, x2 in test_cases:
            dist1 = track.wrapped_euclidian_distance_points(y1, x1, y2, x2, nx=10, half_nx=5.0, wrap=True)
            dist2 = track.wrapped_euclidian_distance_points(y2, x2, y1, x1, nx=10, half_nx=5.0, wrap=True)

            assert np.isclose(dist1, dist2, atol=1e-10), f"Distance not symmetric for ({y1},{x1}) and ({y2},{x2})"

    def test_wrapped_distance_triangle_inequality(self):
        """Test that wrapped distance satisfies triangle inequality."""
        # Test points
        points = [(0, 0), (0, 5), (5, 5), (0, 9), (0, 1)]

        for i, (y1, x1) in enumerate(points):
            for j, (y2, x2) in enumerate(points):
                for k, (y3, x3) in enumerate(points):
                    if i != j and j != k and i != k:
                        d12 = track.wrapped_euclidian_distance_points(y1, x1, y2, x2, nx=10, half_nx=5.0, wrap=True)
                        d23 = track.wrapped_euclidian_distance_points(y2, x2, y3, x3, nx=10, half_nx=5.0, wrap=True)
                        d13 = track.wrapped_euclidian_distance_points(y1, x1, y3, x3, nx=10, half_nx=5.0, wrap=True)

                        # Triangle inequality: d13 <= d12 + d23
                        assert d13 <= d12 + d23 + 1e-10, f"Triangle inequality violated for points {i},{j},{k}"

    def test_wrapped_distance_minimum_value(self):
        """Test that wrapped distance gives minimum possible distance."""
        # Point at (0, 9) should be distance 1 from (0, 0) in a 10-wide grid
        dist = track.wrapped_euclidian_distance_points(0, 9, 0, 0, nx=10, half_nx=5.0, wrap=True)
        assert np.isclose(dist, 1.0, atol=1e-10)

        # Point at (0, 6) should be distance 4 from (0, 0) (not wrapped)
        dist = track.wrapped_euclidian_distance_points(0, 6, 0, 0, nx=10, half_nx=5.0, wrap=True)
        assert np.isclose(dist, 4.0, atol=1e-10)

        # Point at (0, 4) should be distance 4 from (0, 0) (not wrapped)
        dist = track.wrapped_euclidian_distance_points(0, 4, 0, 0, nx=10, half_nx=5.0, wrap=True)
        assert np.isclose(dist, 4.0, atol=1e-10)
