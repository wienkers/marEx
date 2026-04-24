"""
Unit tests for individual functions in marEx.track module.

Tests core utility functions for marine extreme tracking and partitioning.
Focuses on testing individual function behaviour rather than full pipeline integration.
"""

import numpy as np
import xarray as xr

import marEx.track as track
from marEx.track.objects import ObjectPropsStore, calculate_object_properties, calculate_partitioned_child_properties


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

    def test_partition_nn_grid_matches_legacy_gridsearch(self):
        """The EDT partition_nn_grid is physically equivalent to the legacy grid search.

        Both assign each child pixel to the parent owning the nearest pixel (within
        max_distance, else nearest centroid), so over a randomized battery -- including
        seam-straddling cases -- the assignments must agree everywhere except a small set
        of (near-)equidistant boundary pixels, where scipy's feature-transform tie-break
        differs from the grid search's first-parent-in-index-order. A large disagreement
        would indicate a real bug, not a tie.
        """
        from marEx.track.partitioning import _partition_nn_grid_gridsearch as legacy

        rng = np.random.default_rng(2024)
        ny, nx = 50, 100

        def make_blob(cy, cx, r):
            yy, xx = np.ogrid[:ny, :nx]
            dx = np.abs(xx - cx)
            dx = np.minimum(dx, nx - dx)  # wrapped longitude distance
            return ((yy - cy) ** 2 + dx**2) <= r * r

        total = 0
        diff = 0
        trials = 0
        for trial in range(60):
            n_parents = int(rng.integers(2, 6))
            # Half the trials place the child near the antimeridian seam.
            seam = trial % 2 == 0
            base_x = 0 if seam else int(rng.integers(20, 80))
            centres, parent_masks = [], []
            for _ in range(n_parents):
                cy = int(rng.integers(8, ny - 8))
                cx = int((base_x + rng.integers(-12, 13)) % nx)
                centres.append((cy, cx))
                parent_masks.append(make_blob(cy, cx, int(rng.integers(2, 5))))
            parent_masks = np.array(parent_masks)
            if not parent_masks.any(axis=(1, 2)).all():
                continue  # skip trials with an empty parent (covered elsewhere)
            parent_centroids = np.array([[float(c[0]), float(c[1])] for c in centres])
            ccy = int(np.mean([c[0] for c in centres]))
            child_mask = make_blob(ccy, base_x, int(rng.integers(8, 14)))
            if not child_mask.any():
                continue
            child_ids = np.arange(100, 100 + n_parents, dtype=np.int32)
            md = 60
            new_edt = track.partition_nn_grid(
                child_mask, parent_masks, child_ids, parent_centroids, Nx=nx, max_distance=md, wrap=True
            )
            new_leg = legacy(child_mask, parent_masks, child_ids, parent_centroids, nx, md, True)
            assert new_edt.shape == new_leg.shape
            total += new_edt.size
            diff += int(np.sum(new_edt != new_leg))
            trials += 1

        assert trials >= 20, "Too few valid trials generated"
        frac = diff / max(total, 1)
        assert frac < 0.02, f"EDT vs grid-search disagreement {frac:.3%} exceeds 2% (only equidistant ties expected)"


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


class TestPartitionedChildProperties:
    """Test calculate_partitioned_child_properties against the full-slice regionprops oracle.

    The helper must reproduce calculate_object_properties(..., properties=["area","centroid"]) on a
    structured grid (unweighted pixel-count area + unweighted pixel-coordinate centroid with the same
    antimeridian-wrap convention), but computed only from a label set's own pixels rather than a
    full-slice regionprops_table per merge.
    """

    @staticmethod
    def _oracle(ids):
        """Full-slice structured calculate_object_properties for a 2D label array."""
        field = xr.DataArray(ids, dims=["lat", "lon"])
        ds = calculate_object_properties(
            field,
            unstructured_grid=False,
            lat=None,
            lon=None,
            cell_area=None,
            timedim="time",
            regional_mode=False,
            ydim="lat",
            xdim="lon",
            properties=["area", "centroid"],
        )
        return ds.sortby("ID")

    @staticmethod
    def _helper(ids, regional_mode=False):
        y_idx, x_idx = np.nonzero(ids)
        new_labels = ids[y_idx, x_idx]
        ds = calculate_partitioned_child_properties(y_idx, x_idx, new_labels, Nx=ids.shape[1], regional_mode=regional_mode)
        return ds.sortby("ID")

    def _assert_matches_oracle(self, ids):
        oracle = self._oracle(ids)
        helper = self._helper(ids)
        # Same set of IDs
        np.testing.assert_array_equal(helper.ID.values, oracle.ID.values)
        # Area: exact (pixel count)
        np.testing.assert_array_equal(helper.area.values.astype(np.int64), oracle.area.values.astype(np.int64))
        # Centroid: identical to floating-point tolerance (summation order)
        max_diff = float(np.max(np.abs(helper.centroid.values - oracle.centroid.values)))
        assert max_diff < 1e-6, f"centroid diff {max_diff}"

    def test_simple_blobs(self):
        """Interior blobs away from any edge."""
        ids = np.zeros((30, 60), dtype=np.int32)
        ids[5:10, 5:12] = 7
        ids[20:25, 30:40] = 13
        ids[15:18, 45:50] = 21
        self._assert_matches_oracle(ids)

    def test_edge_touching_blobs(self):
        """Blobs touching only one x-edge must NOT trigger wrap adjustment."""
        ids = np.zeros((30, 60), dtype=np.int32)
        ids[5:10, 0:8] = 3  # left edge only
        ids[20:25, 52:60] = 9  # right edge only
        self._assert_matches_oracle(ids)

    def test_antimeridian_straddling_blob(self):
        """A blob with pixels near BOTH x-edges must wrap-adjust the x-centroid like the oracle."""
        ids = np.zeros((20, 200), dtype=np.int32)
        # Pixels in the first and last few columns (within the 100-col edge band) -> straddle
        ids[8:12, 0:5] = 4
        ids[8:12, 196:200] = 4
        # An ordinary interior blob alongside it
        ids[2:6, 90:100] = 8
        self._assert_matches_oracle(ids)

    def test_regional_mode_no_wrap(self):
        """regional_mode=True: raw means, no antimeridian adjustment."""
        ids = np.zeros((20, 200), dtype=np.int32)
        ids[8:12, 0:5] = 4
        ids[8:12, 196:200] = 4
        oracle = self._oracle(ids)  # regional_mode handled below
        # Oracle in regional mode:
        field = xr.DataArray(ids, dims=["lat", "lon"])
        oracle_reg = calculate_object_properties(
            field,
            unstructured_grid=False,
            lat=None,
            lon=None,
            cell_area=None,
            timedim="time",
            regional_mode=True,
            ydim="lat",
            xdim="lon",
            properties=["area", "centroid"],
        ).sortby("ID")
        helper_reg = self._helper(ids, regional_mode=True).sortby("ID")
        np.testing.assert_array_equal(helper_reg.ID.values, oracle_reg.ID.values)
        np.testing.assert_array_equal(helper_reg.area.values.astype(np.int64), oracle_reg.area.values.astype(np.int64))
        max_diff = float(np.max(np.abs(helper_reg.centroid.values - oracle_reg.centroid.values)))
        assert max_diff < 1e-6, f"regional centroid diff {max_diff}"
        # Sanity: the straddling blob's raw x-mean is mid-grid (~100), not wrap-adjusted
        assert abs(float(oracle.centroid.sel(ID=4).values[1]) - float(oracle_reg.centroid.sel(ID=4).values[1])) > 50


class TestObjectPropsStore:
    """Test the O(1) ObjectPropsStore that replaces the in-loop xarray object_props Dataset."""

    @staticmethod
    def _dataset(ids, areas, cys, cxs):
        return xr.Dataset(
            {
                "area": ("ID", np.array(areas)),
                "centroid": (("component", "ID"), np.array([cys, cxs], dtype=np.float64)),
            },
            coords={"ID": np.array(ids, dtype=np.int32)},
        )

    def test_roundtrip_dataset(self):
        """from_dataset -> to_dataset preserves IDs (sorted), areas, centroids."""
        ds = self._dataset([5, 2, 9], [100, 50, 7], [1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        store = ObjectPropsStore.from_dataset(ds)
        out = store.to_dataset()
        np.testing.assert_array_equal(out.ID.values, np.array([2, 5, 9], dtype=np.int32))
        # Aligned by ID, values must match the input
        ds_sorted = ds.sortby("ID")
        np.testing.assert_array_equal(out.area.values.astype(np.int64), ds_sorted.area.values.astype(np.int64))
        np.testing.assert_allclose(out.centroid.values, ds_sorted.centroid.values)

    def test_lookups_match_sel(self):
        """areas()/centroids()/centroid() match xarray .sel(ID=...) semantics (order-preserving)."""
        ds = self._dataset([5, 2, 9], [100, 50, 7], [1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        store = ObjectPropsStore.from_dataset(ds)
        ids = np.array([9, 5])
        np.testing.assert_array_equal(store.areas(ids), ds["area"].sel(ID=ids).values)
        # centroids() must equal object_props.sel(ID=ids).centroid.values.T  (shape (k, 2), [:,0]=y)
        np.testing.assert_allclose(store.centroids(ids), ds["centroid"].sel(ID=ids).values.T)
        np.testing.assert_allclose(store.centroid(2), ds["centroid"].sel(ID=2).values)
        assert store.area(5) == 100
        assert store.max_id() == 9
        assert 2 in store and 999 not in store

    def test_set_drop_add(self):
        """set updates/inserts, drop removes; max_id and membership track correctly."""
        ds = self._dataset([1, 2], [10, 20], [0.0, 0.0], [0.0, 0.0])
        store = ObjectPropsStore.from_dataset(ds)
        store.set(2, 99, 5.0, 6.0)  # update existing
        assert store.area(2) == 99
        np.testing.assert_allclose(store.centroid(2), [5.0, 6.0])
        store.set(50, 7, 1.0, 2.0)  # insert new
        assert 50 in store and store.max_id() == 50
        store.drop(1)  # delete
        assert 1 not in store
        np.testing.assert_array_equal(store.to_dataset().ID.values, np.array([2, 50], dtype=np.int32))
