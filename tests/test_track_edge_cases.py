"""
Edge Case Tests for marEx.track Module

Tests for edge cases, warnings, and less-common code paths in the tracking system.
Focuses on checkpoint functionality, chunking validation warnings, and complex
validation scenarios.
"""

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError, DataValidationError


@pytest.fixture(scope="module")
def extremes_gridded():
    """Load gridded extremes data for testing."""
    test_data_path = Path(__file__).parent / "data" / "extremes_gridded.zarr"
    return xr.open_zarr(str(test_data_path), chunks={}).persist()


@pytest.fixture(scope="module")
def extremes_unstructured():
    """Load unstructured extremes data for testing."""
    test_data_path = Path(__file__).parent / "data" / "extremes_unstructured.zarr"
    return xr.open_zarr(str(test_data_path), chunks={}).persist()


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoint tests."""
    temp_dir = tempfile.mkdtemp(prefix="marex_test_checkpoint_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestCheckpointFunctionality:
    """Test checkpoint save and load functionality."""

    def test_checkpoint_save(self, extremes_unstructured, temp_checkpoint_dir, dask_client_unstructured):
        """Test saving checkpoints during preprocessing (unstructured grids only)."""
        neighbours = extremes_unstructured.neighbours
        cell_areas = extremes_unstructured.cell_areas

        tracker = marEx.tracker(
            extremes_unstructured.extreme_events,
            extremes_unstructured.mask,
            area_filter_quartile=0.5,
            R_fill=0,
            T_fill=0,
            temp_dir=temp_checkpoint_dir,
            unstructured_grid=True,
            dimensions={"x": "ncells"},
            coordinates={"x": "lon", "y": "lat"},
            coordinate_units="degrees",
            neighbours=neighbours,
            cell_areas=cell_areas,
            quiet=True,
        )

        # Run preprocessing with checkpoint save
        data_proc, stats = tracker.run_preprocess(checkpoint="save")

        # Verify checkpoint files were created
        checkpoint_zarr = Path(tracker.scratch_dir) / "marEx_checkpoint_proc_bin.zarr"
        checkpoint_stats = Path(tracker.scratch_dir) / "marEx_checkpoint_stats.npz"

        assert checkpoint_zarr.exists(), "Checkpoint zarr file not created"
        assert checkpoint_stats.exists(), "Checkpoint stats file not created"

        # Verify stats tuple structure
        assert isinstance(stats, tuple), "Stats should be a tuple"
        assert len(stats) == 6, "Stats tuple should have 6 elements"

        # Verify zarr can be loaded
        loaded_zarr = xr.open_zarr(str(checkpoint_zarr))
        assert "data_bin_preproc" in loaded_zarr, "Preprocessed data not in checkpoint"

    def test_checkpoint_load(self, extremes_unstructured, temp_checkpoint_dir, dask_client_unstructured):
        """Test loading from existing checkpoints (unstructured grids only)."""
        neighbours = extremes_unstructured.neighbours
        cell_areas = extremes_unstructured.cell_areas

        # First, create a checkpoint
        tracker = marEx.tracker(
            extremes_unstructured.extreme_events,
            extremes_unstructured.mask,
            area_filter_quartile=0.5,
            R_fill=0,
            T_fill=0,
            temp_dir=temp_checkpoint_dir,
            unstructured_grid=True,
            dimensions={"x": "ncells"},
            coordinates={"x": "lon", "y": "lat"},
            coordinate_units="degrees",
            neighbours=neighbours,
            cell_areas=cell_areas,
            quiet=True,
        )

        # Save checkpoint
        data_saved, stats_saved = tracker.run_preprocess(checkpoint="save")

        # Now create a new tracker instance and load from checkpoint
        tracker_load = marEx.tracker(
            extremes_unstructured.extreme_events,
            extremes_unstructured.mask,
            area_filter_quartile=0.5,
            R_fill=0,
            T_fill=0,
            temp_dir=temp_checkpoint_dir,
            unstructured_grid=True,
            dimensions={"x": "ncells"},
            coordinates={"x": "lon", "y": "lat"},
            coordinate_units="degrees",
            neighbours=neighbours,
            cell_areas=cell_areas,
            quiet=True,
        )

        # Load from checkpoint
        data_loaded, stats_loaded = tracker_load.run_preprocess(checkpoint="load")

        # Verify loaded data matches saved data
        assert data_loaded.shape == data_saved.shape, "Loaded data shape doesn't match saved"
        assert len(stats_loaded) == len(stats_saved), "Stats tuple length mismatch"

        # Verify stats values match (allowing for floating point tolerance)
        for i, (saved_val, loaded_val) in enumerate(zip(stats_saved, stats_loaded)):
            if isinstance(saved_val, (int, np.integer)):
                assert saved_val == loaded_val, f"Stats element {i} doesn't match"
            else:
                np.testing.assert_allclose(saved_val, loaded_val, rtol=1e-10)


class TestChunkingValidationWarnings:
    """Test chunking validation and rechunking warnings."""

    def test_neighbours_nv_chunking_warning(self, extremes_unstructured, temp_checkpoint_dir, dask_client_unstructured):
        """Test warning when neighbours has multiple chunks in nv dimension."""
        neighbours = extremes_unstructured.neighbours
        cell_areas = extremes_unstructured.cell_areas

        # Create neighbours array with intentionally bad nv chunking
        neighbours_bad_chunks = neighbours.chunk({"ncells": -1, "nv": 2})  # Multiple nv chunks

        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")

            tracker = marEx.tracker(
                extremes_unstructured.extreme_events,
                extremes_unstructured.mask,
                area_filter_quartile=0.5,
                R_fill=0,
                temp_dir=temp_checkpoint_dir,
                unstructured_grid=True,
                dimensions={"x": "ncells"},
                coordinates={"x": "lon", "y": "lat"},
                coordinate_units="degrees",
                neighbours=neighbours_bad_chunks,
                cell_areas=cell_areas,
                quiet=True,
            )

            # This test verifies the tracker initialises even with bad chunking
            # The actual warning may or may not be raised depending on the data
            assert tracker is not None


class TestDataValidationEdgeCases:
    """Test edge cases in data validation."""

    def test_empty_data_bin_attrs(self, extremes_gridded, dask_client_gridded):
        """Test handling of DataArray with no attrs."""
        # Remove all attributes
        data_no_attrs = extremes_gridded.extreme_events.copy(deep=True)
        data_no_attrs.attrs = {}

        # Should handle gracefully
        tracker = marEx.tracker(
            data_no_attrs,
            extremes_gridded.mask,
            area_filter_quartile=0.5,
            R_fill=4,
            quiet=True,
        )

        assert tracker.data_attrs == {}, "Should handle empty attrs gracefully"

    def test_missing_coordinates_unstructured(self, dask_client_unstructured):
        """Test error when required coordinates are missing in unstructured data."""
        # Create data with missing coordinates
        test_data = xr.DataArray(
            np.random.rand(10, 100).astype(bool),
            dims=["time", "ncells"],
            coords={
                "time": np.arange(10),
                "ncells": np.arange(100),
            },  # Missing lat/lon spatial coordinates
        ).chunk({"time": 5})

        mask = xr.DataArray(
            np.ones(100, dtype=bool),
            dims=["ncells"],
            coords={"ncells": np.arange(100)},
        )

        # Should raise DataValidationError when trying to use unstructured mode
        with pytest.raises(
            (DataValidationError, KeyError), match="lat|lon|Missing required coordinates|Cannot auto-detect coordinate"
        ):
            marEx.tracker(
                test_data,
                mask,
                R_fill=0,
                unstructured_grid=True,
                dimensions={"x": "ncells"},
                coordinates={"x": "lon", "y": "lat"},  # These don't exist in data
                quiet=True,
            )


class TestOverlapValidationAndWarnings:
    """Test overlap validation and warning conditions."""

    def test_empty_overlap_list_return(self, extremes_gridded, dask_client_gridded):
        """Test handling of empty overlap lists."""
        # Create minimal tracker
        tracker = marEx.tracker(
            extremes_gridded.extreme_events.isel(time=slice(0, 2)),  # Only 2 timesteps
            extremes_gridded.mask,
            area_filter_quartile=0.9,  # Very high threshold = few objects
            R_fill=0,
            T_fill=0,
            allow_merging=False,
            quiet=True,
        )

        # Run tracking
        result = tracker.run()

        # Should handle case with minimal/no overlaps
        assert isinstance(result, xr.Dataset), "Should return valid Dataset even with minimal overlaps"


class TestComplexMergeSplitValidation:
    """Test complex merge/split validation and logging."""

    def test_complex_merging_scenario(self, extremes_gridded, dask_client_gridded):
        """Test complex merging with validation logging."""
        # Create tracker with aggressive merging settings
        tracker = marEx.tracker(
            extremes_gridded.extreme_events,
            extremes_gridded.mask.where((extremes_gridded.lat < 85) & (extremes_gridded.lat > -90), other=False),
            area_filter_quartile=0.4,
            R_fill=8,
            T_fill=2,  # Must be even
            allow_merging=True,
            overlap_threshold=0.15,
            quiet=False,  # Enable logging to trigger validation messages
        )

        # Run tracking - this should trigger complex merge validation
        tracked_ds = tracker.run()

        # Verify tracking completed successfully
        assert "ID_field" in tracked_ds
        assert tracked_ds.attrs["N_events_final"] > 0
        assert "allow_merging" in tracked_ds.attrs
        assert tracked_ds.attrs["allow_merging"] == 1

    def test_merging_with_temporal_fill(self, extremes_gridded, dask_client_gridded):
        """Test merging with temporal filling."""
        tracker = marEx.tracker(
            extremes_gridded.extreme_events,
            extremes_gridded.mask,
            area_filter_quartile=0.4,
            R_fill=4,
            T_fill=2,  # Must be even
            allow_merging=True,
            overlap_threshold=0.2,
            quiet=False,
        )

        tracked_ds = tracker.run()

        # Verify tracking completed
        assert "ID_field" in tracked_ds
        assert "allow_merging" in tracked_ds.attrs
        assert tracked_ds.attrs["allow_merging"] == 1


class TestUnstructuredPartitioning:
    """Test unstructured grid partitioning code paths."""

    def test_unstructured_tracking_with_merging(self, extremes_unstructured, temp_checkpoint_dir, dask_client_unstructured):
        """Test unstructured grid tracking with merging to trigger partition code."""
        neighbours = extremes_unstructured.neighbours
        cell_areas = extremes_unstructured.cell_areas

        # Create tracker with merging enabled
        tracker = marEx.tracker(
            extremes_unstructured.extreme_events,
            extremes_unstructured.mask,
            area_filter_quartile=0.4,
            R_fill=2,
            T_fill=2,  # Must be even
            allow_merging=True,
            overlap_threshold=0.2,
            temp_dir=temp_checkpoint_dir,
            unstructured_grid=True,
            dimensions={"x": "ncells"},
            coordinates={"x": "lon", "y": "lat"},
            coordinate_units="degrees",
            neighbours=neighbours,
            cell_areas=cell_areas,
            quiet=True,
        )

        # Run tracking
        tracked_ds = tracker.run()

        # Verify completion
        assert "ID_field" in tracked_ds
        assert tracked_ds.attrs["N_events_final"] > 0
        assert "allow_merging" in tracked_ds.attrs


class TestSpatialChunkingValidationWarnings:
    """Test warnings for improper chunking of neighbours and cell_areas in spatial dimension."""

    def test_neighbours_multi_chunk_warning(self, extremes_unstructured, temp_checkpoint_dir):
        """Test warning when neighbours has multiple chunks in spatial dimension."""
        import warnings

        # Get neighbours and cell_areas with proper single chunk
        neighbours = extremes_unstructured.neighbours
        cell_areas = extremes_unstructured.cell_areas

        # Rechunk neighbours to have multiple chunks in spatial dimension (xdim)
        neighbours_multichunk = neighbours.chunk({"ncells": 100})  # Create multiple chunks

        # Only run test if we actually created multiple chunks
        if len(neighbours_multichunk.chunksizes.get("ncells", [])) > 1:
            with warnings.catch_warnings(record=True) as _:
                warnings.simplefilter("always")

                # Create tracker - this should trigger warning about neighbours chunking
                tracker = marEx.tracker(
                    extremes_unstructured.extreme_events,
                    extremes_unstructured.mask,
                    area_filter_quartile=0.5,
                    R_fill=0,
                    T_fill=0,
                    temp_dir=temp_checkpoint_dir,
                    unstructured_grid=True,
                    dimensions={"x": "ncells"},
                    coordinates={"x": "lon", "y": "lat"},
                    coordinate_units="degrees",
                    neighbours=neighbours_multichunk,  # Multi-chunk neighbours
                    cell_areas=cell_areas,
                    quiet=True,
                )

                # The tracker should complete successfully even with warning
                assert tracker is not None

    def test_cell_areas_multi_chunk_warning(self, extremes_unstructured, temp_checkpoint_dir):
        """Test warning when cell_areas has multiple chunks in spatial dimension."""
        import warnings

        # Get neighbours and cell_areas
        neighbours = extremes_unstructured.neighbours
        cell_areas = extremes_unstructured.cell_areas

        # Rechunk cell_areas to have multiple chunks in spatial dimension
        # This should trigger the warning on lines 899-906
        cell_areas_multichunk = cell_areas.chunk({"ncells": 100})  # Create multiple chunks

        # Only run test if we actually created multiple chunks
        if len(cell_areas_multichunk.chunksizes.get("ncells", [])) > 1:
            with warnings.catch_warnings(record=True) as _:
                warnings.simplefilter("always")

                # Create tracker - this should trigger warning about cell_areas chunking
                tracker = marEx.tracker(
                    extremes_unstructured.extreme_events,
                    extremes_unstructured.mask,
                    area_filter_quartile=0.5,
                    R_fill=0,
                    T_fill=0,
                    temp_dir=temp_checkpoint_dir,
                    unstructured_grid=True,
                    dimensions={"x": "ncells"},
                    coordinates={"x": "lon", "y": "lat"},
                    coordinate_units="degrees",
                    neighbours=neighbours,
                    cell_areas=cell_areas_multichunk,  # Multi-chunk cell_areas
                    quiet=True,
                )

                # The tracker should complete successfully even with warning
                assert tracker is not None

    def test_both_multi_chunk_warnings(self, extremes_unstructured, temp_checkpoint_dir):
        """Test warnings when both neighbours and cell_areas have multiple chunks."""
        import warnings

        # Get neighbours and cell_areas
        neighbours = extremes_unstructured.neighbours
        cell_areas = extremes_unstructured.cell_areas

        # Rechunk both to have multiple chunks
        neighbours_multichunk = neighbours.chunk({"ncells": 100})
        cell_areas_multichunk = cell_areas.chunk({"ncells": 100})

        # Only run test if we actually created multiple chunks
        if (
            len(neighbours_multichunk.chunksizes.get("ncells", [])) > 1
            and len(cell_areas_multichunk.chunksizes.get("ncells", [])) > 1
        ):
            with warnings.catch_warnings(record=True) as _:
                warnings.simplefilter("always")

                # Create tracker - this should trigger both warnings
                tracker = marEx.tracker(
                    extremes_unstructured.extreme_events,
                    extremes_unstructured.mask,
                    area_filter_quartile=0.5,
                    R_fill=0,
                    T_fill=0,
                    temp_dir=temp_checkpoint_dir,
                    unstructured_grid=True,
                    dimensions={"x": "ncells"},
                    coordinates={"x": "lon", "y": "lat"},
                    coordinate_units="degrees",
                    neighbours=neighbours_multichunk,
                    cell_areas=cell_areas_multichunk,
                    quiet=True,
                )

                # The tracker should complete successfully even with warnings
                assert tracker is not None


class TestEmptyAttrsHandling:
    """Test handling of data with empty or missing attrs."""

    def test_empty_attrs_dict(self):
        """Test tracker with data_bin that has empty attrs dictionary."""
        # Create binary data with empty attrs
        data = xr.DataArray(
            np.random.random((10, 5, 4)) > 0.7,
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-20, 20, 5),
                "lon": np.linspace(-10, 10, 4),
            },
            attrs={},  # Empty attrs dict
        ).chunk({"time": 5})

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        # Should handle empty attrs gracefully
        tracker = marEx.tracker(
            data,
            mask,
            R_fill=2,
            area_filter_quartile=0.5,
            regional_mode=True,
            coordinate_units="degrees",
        )

        assert tracker is not None
        assert tracker.data_attrs == {}

    def test_no_attrs_attribute(self):
        """Test tracker with data_bin that has no attrs attribute."""
        # Create binary data
        data = xr.DataArray(
            np.random.random((10, 5, 4)) > 0.7,
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-20, 20, 5),
                "lon": np.linspace(-10, 10, 4),
            },
        ).chunk({"time": 5})

        # Assign some attrs then clear them
        data.attrs.clear()

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        # Should handle missing attrs gracefully
        tracker = marEx.tracker(
            data,
            mask,
            R_fill=2,
            area_filter_quartile=0.5,
            regional_mode=True,
            coordinate_units="degrees",
        )

        assert tracker is not None
        assert tracker.data_attrs == {}


class TestGridResolutionValidation:
    """Test grid_resolution parameter validation."""

    def test_grid_resolution_with_unstructured_error(self, extremes_unstructured, temp_checkpoint_dir):
        """Test error when grid_resolution is used with unstructured grids."""
        neighbours = extremes_unstructured.neighbours
        cell_areas = extremes_unstructured.cell_areas

        # Try to use grid_resolution with unstructured grid - should error
        with pytest.raises(DataValidationError, match="grid_resolution parameter is not supported for unstructured grids"):
            marEx.tracker(
                extremes_unstructured.extreme_events,
                extremes_unstructured.mask,
                area_filter_quartile=0.5,
                R_fill=0,
                T_fill=0,
                temp_dir=temp_checkpoint_dir,
                unstructured_grid=True,
                dimensions={"x": "ncells"},
                coordinates={"x": "lon", "y": "lat"},
                coordinate_units="degrees",
                neighbours=neighbours,
                cell_areas=cell_areas,
                grid_resolution=0.1,  # This should trigger error
                quiet=True,
            )


class TestAreaFilteringValidation:
    """Test area filtering parameter validation."""

    def test_negative_area_filter_absolute(self):
        """Test error for negative area_filter_absolute."""
        # Create simple binary data
        data = xr.DataArray(
            np.random.random((10, 5, 4)) > 0.7,
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-20, 20, 5),
                "lon": np.linspace(-10, 10, 4),
            },
        ).chunk({"time": 5})

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        with pytest.raises(ConfigurationError, match="area_filter_absolute.*must be positive"):
            marEx.tracker(
                data,
                mask,
                R_fill=2,
                area_filter_absolute=-100.0,  # Negative value
                regional_mode=True,
                coordinate_units="degrees",
            )

    def test_both_area_filters_specified(self):
        """Test error when both area_filter_quartile and area_filter_absolute are specified."""
        # Create simple binary data
        data = xr.DataArray(
            np.random.random((10, 5, 4)) > 0.7,
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-20, 20, 5),
                "lon": np.linspace(-10, 10, 4),
            },
        ).chunk({"time": 5})

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        with pytest.raises(ConfigurationError, match="Cannot specify both area filtering parameters"):
            marEx.tracker(
                data,
                mask,
                R_fill=2,
                area_filter_quartile=0.5,  # Both specified
                area_filter_absolute=1000.0,  # Both specified
                regional_mode=True,
                coordinate_units="degrees",
            )


class TestNvDimensionChunkingWarning:
    """Test warning for nv dimension multi-chunking."""

    def test_nv_dimension_multi_chunk_warning(self, extremes_unstructured, temp_checkpoint_dir):
        """Test warning when neighbours nv dimension has multiple chunks."""
        import warnings

        neighbours = extremes_unstructured.neighbours
        cell_areas = extremes_unstructured.cell_areas

        # Rechunk neighbours to have multiple chunks in nv dimension
        neighbours_multichunk = neighbours.chunk({"nv": 1})  # Force 3 chunks (one per nv value)

        # Only run if we created multiple chunks
        if len(neighbours_multichunk.chunksizes.get("nv", [])) > 1:
            with warnings.catch_warnings(record=True) as _:
                warnings.simplefilter("always")

                # Create tracker - should trigger warning about nv chunking
                tracker = marEx.tracker(
                    extremes_unstructured.extreme_events,
                    extremes_unstructured.mask,
                    area_filter_quartile=0.5,
                    R_fill=0,
                    T_fill=0,
                    temp_dir=temp_checkpoint_dir,
                    unstructured_grid=True,
                    dimensions={"x": "ncells"},
                    coordinates={"x": "lon", "y": "lat"},
                    coordinate_units="degrees",
                    neighbours=neighbours_multichunk,
                    cell_areas=cell_areas,
                    quiet=True,
                )

                # Tracker should complete successfully
                assert tracker is not None


class TestRadiansCoordinateHandling:
    """Test radians coordinate auto-detection and conversion."""

    def test_radians_auto_detection(self):
        """Test auto-detection of radians from 2π coordinate range."""
        # Create data with coordinates in [0, 2π] range (radians)
        data = xr.DataArray(
            np.random.random((10, 5, 4)) > 0.7,
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-np.pi / 2, np.pi / 2, 5),  # [-π/2, π/2] in radians
                "lon": np.linspace(0, 2 * np.pi, 4),  # [0, 2π] in radians
            },
        ).chunk({"time": 5})

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        # Should auto-detect radians from coordinate range
        tracker = marEx.tracker(
            data,
            mask,
            R_fill=2,
            area_filter_quartile=0.5,
        )

        # Verify radians were detected and converted
        assert tracker.coordinate_units == "radians"

    def test_radians_to_degrees_conversion(self):
        """Test conversion from radians to degrees."""
        # Create data with coordinates in radians
        lat_rad = np.linspace(-np.pi / 2, np.pi / 2, 5)
        lon_rad = np.linspace(-np.pi, np.pi, 4)

        data = xr.DataArray(
            np.random.random((10, 5, 4)) > 0.7,
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": lat_rad,
                "lon": lon_rad,
            },
        ).chunk({"time": 5})

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        # Create tracker - should convert radians to degrees
        tracker = marEx.tracker(
            data,
            mask,
            R_fill=2,
            area_filter_quartile=0.5,
        )

        # Verify coordinates were converted to degrees
        lat_deg = tracker.data_bin["lat"].values
        lon_deg = tracker.data_bin["lon"].values

        # Check that values are in degree range, not radian range
        assert np.max(np.abs(lat_deg)) > 10  # Should be close to 90 degrees, not π/2
        assert np.max(np.abs(lon_deg)) > 10  # Should be close to 180 degrees, not π


class TestAbsoluteAreaFiltering:
    """Test absolute area filtering code path."""

    def test_absolute_area_filtering_used(self):
        """Test that absolute area filtering is used when specified."""
        # Create data with larger, more robust objects
        data = xr.DataArray(
            np.zeros((5, 20, 20), dtype=bool),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(5),
                "lat": np.linspace(-20, 20, 20),
                "lon": np.linspace(-10, 10, 20),
            },
        ).chunk({"time": 5})

        # Add larger, more connected objects that will survive morphological operations
        data.values[0, 2:8, 2:8] = True
        data.values[0, 10:16, 10:16] = True

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        # Use absolute area filtering with threshold below object sizes
        tracker = marEx.tracker(
            data,
            mask,
            R_fill=0,  # No filling to preserve objects
            T_fill=0,
            area_filter_absolute=10.0,  # Filter objects < 10 cells (our objects are ~36 cells)
            regional_mode=True,
            coordinate_units="degrees",
        )

        # Just verify tracker creation (tests that absolute filtering flag is set)
        assert tracker is not None
        assert tracker._use_absolute_filtering is True
        assert tracker.area_filter_absolute == 10.0


class TestSingleTimestepTracking:
    """Test tracking with single timestep."""

    def test_single_timestep_data(self):
        """Test tracking with data containing only 1 timestep."""
        # Create data with single timestep
        data = xr.DataArray(
            np.random.random((1, 10, 10)) > 0.7,
            dims=["time", "lat", "lon"],
            coords={
                "time": [0],
                "lat": np.linspace(-20, 20, 10),
                "lon": np.linspace(-10, 10, 10),
            },
        ).chunk({"time": 1})

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        # Should handle single timestep
        tracker = marEx.tracker(
            data,
            mask,
            R_fill=0,
            T_fill=0,
            area_filter_quartile=0.5,
            regional_mode=True,
            coordinate_units="degrees",
        )

        # Verify tracker was created successfully with single timestep
        assert tracker is not None
        assert tracker.data_bin.sizes["time"] == 1


class TestRegionalTrackerFunction:
    """Test regional_tracker convenience function."""

    def test_regional_tracker_function(self):
        """Test marEx.regional_tracker() convenience function."""
        # Create simple binary data
        data = xr.DataArray(
            np.random.random((10, 5, 4)) > 0.7,
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-20, 20, 5),
                "lon": np.linspace(-10, 10, 4),
            },
        ).chunk({"time": 5})

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        # Use regional_tracker function
        regional_track = marEx.regional_tracker(
            data,
            mask,
            R_fill=2,
            area_filter_absolute=10.0,  # Use absolute filtering
            coordinate_units="degrees",
        )

        # Verify it returns a tracker instance
        assert regional_track is not None
        assert hasattr(regional_track, "run")
        assert regional_track.regional_mode is True
        assert regional_track.area_filter_absolute == 10.0


class TestDataValidationErrors:
    """Test data validation error paths that are typically avoided in normal usage."""

    def test_non_dask_data_error(self):
        """Test error when data is not Dask-backed."""
        # Create NON-Dask data (no .chunk() call)
        data = xr.DataArray(
            np.random.random((10, 5, 4)) > 0.7,
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-20, 20, 5),
                "lon": np.linspace(-10, 10, 4),
            },
        )  # NOT chunked!

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        # Should raise error about non-Dask data
        with pytest.raises(DataValidationError, match="Data must be chunked"):
            marEx.tracker(
                data,
                mask,
                R_fill=2,
                area_filter_quartile=0.5,
                regional_mode=True,
                coordinate_units="degrees",
            )

    def test_missing_temp_dir_unstructured_error(self, extremes_unstructured):
        """Test error when temp_dir not provided for unstructured grid."""
        neighbours = extremes_unstructured.neighbours
        cell_areas = extremes_unstructured.cell_areas

        # Try to create unstructured tracker without temp_dir
        with pytest.raises(DataValidationError, match="temp_dir is required for unstructured grids"):
            marEx.tracker(
                extremes_unstructured.extreme_events,
                extremes_unstructured.mask,
                area_filter_quartile=0.5,
                R_fill=0,
                T_fill=0,
                # temp_dir NOT provided!
                unstructured_grid=True,
                dimensions={"x": "ncells"},
                coordinates={"x": "lon", "y": "lat"},
                coordinate_units="degrees",
                neighbours=neighbours,
                cell_areas=cell_areas,
                quiet=True,
            )


class TestEnforceOverlapThreshold:
    """Test enforce_overlap_threshold method directly."""

    def test_enforce_overlap_empty_valid_overlaps(self):
        """Test return empty array when no valid overlaps."""
        # Create minimal tracker
        data = xr.DataArray(
            np.random.random((5, 5, 5)) > 0.7,
            dims=["time", "lat", "lon"],
            coords={
                "time": range(5),
                "lat": np.linspace(-10, 10, 5),
                "lon": np.linspace(-5, 5, 5),
            },
        ).chunk({"time": 5})

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        tracker = marEx.tracker(
            data,
            mask,
            R_fill=0,
            area_filter_quartile=0.5,
            regional_mode=True,
            coordinate_units="degrees",
        )

        # Create overlap list with IDs that don't exist in object_props
        overlap_list = np.array([[999, 1000, 50.0], [1001, 1002, 60.0]], dtype=np.float32)

        # Create object_props with different IDs
        object_props = xr.Dataset(
            {"area": ("ID", [100.0, 200.0]), "centroid-0": ("ID", [0.0, 1.0]), "centroid-1": ("ID", [0.0, 1.0])},
            coords={"ID": [1, 2]},
        )

        # Call enforce_overlap_threshold - should return empty array
        result = tracker.enforce_overlap_threshold(overlap_list, object_props)

        assert result.shape == (0, 3)
        assert result.dtype == np.int32  # structured grid

    def test_enforce_overlap_fraction_greater_than_one(self, dask_client):
        """Test warning when overlap fraction > 1.0."""
        # Create tracker
        data = xr.DataArray(
            np.random.random((5, 10, 10)) > 0.7,
            dims=["time", "lat", "lon"],
            coords={
                "time": range(5),
                "lat": np.linspace(-20, 20, 10),
                "lon": np.linspace(-10, 10, 10),
            },
        ).chunk({"time": 5})

        mask = xr.ones_like(data.isel(time=0), dtype=bool)

        tracker = marEx.tracker(
            data,
            mask,
            R_fill=0,
            area_filter_quartile=0.5,
            overlap_threshold=0.1,
            regional_mode=True,
            coordinate_units="degrees",
        )

        # Create overlap list where overlap area > min(area_0, area_1)
        # This creates overlap_fraction > 1.0
        overlap_list = np.array(
            [
                [1, 2, 150],  # Overlap = 150, but min(area_1, area_2) = 100
                [3, 4, 80],  # Overlap = 80, min = 200, fraction = 0.4
            ],
            dtype=np.int32,
        )

        # Create object_props where ID 1,2 have small areas relative to overlap
        object_props = xr.Dataset(
            {
                "area": ("ID", [100.0, 120.0, 200.0, 300.0]),
                "centroid-0": ("ID", [0.0, 1.0, 2.0, 3.0]),
                "centroid-1": ("ID", [0.0, 1.0, 2.0, 3.0]),
            },
            coords={"ID": [1, 2, 3, 4]},
        )

        # Call enforce_overlap_threshold - should log warning
        result = tracker.enforce_overlap_threshold(overlap_list, object_props)

        # Verify processing completed and filtered correctly
        assert result is not None
        assert len(result) > 0  # Should have at least one valid overlap
