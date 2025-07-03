"""
Tests for unstructured tracking functionality.

This module tests the tracking of marine extreme events on unstructured grids,
including data validation and tracker initialisation tests.

NOTE: Sparse matrix construction issues have been fixed. Most tracking functionality
tests are currently skipped due to remaining issues in object properties calculation
(array broadcasting errors). The tests serve as a framework for when these issues are resolved.

Working tests:
- test_unstructured_data_validation: Validates input data structure
- test_unstructured_tracker_initialisation: Tests sparse matrix construction (now working)

Skipped tests:
- All actual tracking tests due to array shape mismatch in object properties calculation
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx

from .conftest import assert_count_in_reasonable_range, assert_reasonable_bounds


class TestUnstructuredTracking:
    """Test event tracking functionality for unstructured data."""

    @classmethod
    def setup_class(cls):
        """Load test data for all tests."""
        test_data_path = Path(__file__).parent / "data" / "extremes_unstructured.zarr"
        cls.extremes_data = xr.open_zarr(str(test_data_path), chunks={}).persist()

        # Standard chunk size for tracking (spatial dimensions must be contiguous)
        cls.chunk_size = {"time": 2, "ncells": -1}

        # Define dimensions for unstructured data
        cls.dimensions = {
            "time": "time",
            "x": "ncells",  # No 'y' indicates unstructured grid
        }

        # Create a temporary directory for unstructured processing
        cls.temp_dir = tempfile.mkdtemp(prefix="marex_unstructured_test_")

    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory."""
        if hasattr(cls, "temp_dir") and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)

    def test_unstructured_tracker_initialisation(self, dask_client):
        """Test that unstructured tracker initialisation succeeds."""
        # Previously unstructured tracking had issues with sparse matrix construction
        # This has now been fixed
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            R_fill=2,  # Smaller radius to reduce complexity
            area_filter_quartile=0.8,  # Higher filtering to reduce number of objects
            temp_dir=self.temp_dir,
            T_fill=0,  # No temporal filling for basic test
            allow_merging=False,
            unstructured_grid=True,
            dimensions={"x": "ncells"},  # Must specify the name of the spatial dimension
            coordinates={"x": "lon", "y": "lat"},
            regional_mode=False,
            coordinate_units="degrees",
            quiet=True,
            neighbours=self.extremes_data.neighbours,
            cell_areas=self.extremes_data.cell_areas,
        )

        # Verify tracker was created successfully
        assert tracker is not None
        assert hasattr(tracker, "dilate_sparse")  # Sparse matrix should be constructed

    @pytest.mark.slow
    def test_basic_unstructured_tracking(self, dask_client):
        """Test basic tracking on unstructured grid without merging/splitting."""
        # Skip this test for now due to object properties calculation issues
        # TEMPORARILY ENABLED TO DEBUG ARRAY SHAPE ISSUE
        # pytest.skip("Skipping unstructured tracking due to array broadcasting errors in object properties calculation")

        # Create tracker with basic settings for unstructured data
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            R_fill=2,  # Reduced spatial fill for test data
            area_filter_quartile=0.9,  # Very high filtering to reduce objects
            temp_dir=self.temp_dir,  # Temporary directory for processing
            T_fill=0,  # No temporal filling for basic test
            allow_merging=False,
            unstructured_grid=True,  # Enable unstructured grid mode
            dimensions={"x": "ncells"},  # Must specify the name of the spatial dimension
            coordinates={
                "x": "lon",
                "y": "lat",
            },  # Coordinate mapping for unstructured grid
            regional_mode=False,  # Disable regional mode (not yet implemented)
            coordinate_units="degrees",  # Specify coordinate units
            quiet=True,  # Suppress output for tests
            # Provide unstructured grid information
            neighbours=self.extremes_data.neighbours,
            cell_areas=self.extremes_data.cell_areas,
        )

        # Run tracking
        tracked_ds = tracker.run()

        # Verify output structure
        assert isinstance(tracked_ds, xr.Dataset)
        assert "ID_field" in tracked_ds.data_vars

        # Verify dimensions for unstructured data
        assert "time" in tracked_ds.ID_field.dims
        assert "ncells" in tracked_ds.ID_field.dims

        # Verify attributes are set
        assert "N_events_final" in tracked_ds.attrs
        assert "allow_merging" in tracked_ds.attrs
        assert tracked_ds.attrs["allow_merging"] == 0
        assert "R_fill" in tracked_ds.attrs
        assert "T_fill" in tracked_ds.attrs

        # Verify ID field contains reasonable values
        max_id = int(tracked_ds.ID_field.max())
        assert max_id >= 0, "Invalid ID field values"  # Allow 0 if no events found

        if max_id > 0:  # Only check if events were found
            assert max_id == tracked_ds.attrs["N_events_final"], "Max ID doesn't match reported event count"

        # Verify that background is labeled as 0
        assert int(tracked_ds.ID_field.min()) == 0

    def test_unstructured_data_validation(self, dask_client):
        """Test data validation for unstructured grids."""
        # Test that the data has the expected structure
        assert "extreme_events" in self.extremes_data.data_vars
        assert "mask" in self.extremes_data.data_vars
        assert "neighbours" in self.extremes_data.data_vars
        assert "cell_areas" in self.extremes_data.data_vars

        # Test dimensions
        assert self.extremes_data.extreme_events.dims == ("time", "ncells")
        assert self.extremes_data.mask.dims == ("ncells",)
        assert self.extremes_data.neighbours.dims == ("nv", "ncells")
        assert self.extremes_data.cell_areas.dims == ("ncells",)

        # Test coordinate presence
        assert "lat" in self.extremes_data.coords
        assert "lon" in self.extremes_data.coords
        assert self.extremes_data.lat.dims == ("ncells",)
        assert self.extremes_data.lon.dims == ("ncells",)

        # Test that data is boolean for extreme_events and mask
        assert self.extremes_data.extreme_events.dtype == bool
        assert self.extremes_data.mask.dtype == bool

    @pytest.mark.slow
    def test_advanced_unstructured_tracking_with_merging(self, dask_client):
        """Test advanced tracking with temporal filling and merging enabled on unstructured grid."""
        # Array broadcasting issue has been fixed

        # Create tracker with advanced settings
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            R_fill=3,
            area_filter_quartile=0.5,
            temp_dir=self.temp_dir,
            T_fill=2,  # Allow 2-day gaps
            allow_merging=True,
            overlap_threshold=0.5,
            nn_partitioning=True,
            unstructured_grid=True,  # Enable unstructured grid mode
            dimensions={"x": "ncells"},  # Must specify the name of the spatial dimension
            coordinates={
                "x": "lon",
                "y": "lat",
            },  # Coordinate mapping for unstructured grid
            regional_mode=False,  # Disable regional mode (not yet implemented)
            coordinate_units="degrees",  # Specify coordinate units
            quiet=True,
            # Provide unstructured grid information
            neighbours=self.extremes_data.neighbours,
            cell_areas=self.extremes_data.cell_areas,
        )

        # Run tracking with merge information
        tracked_ds, merges_ds = tracker.run(return_merges=True)

        # Verify main output structure
        assert isinstance(tracked_ds, xr.Dataset)
        assert "ID_field" in tracked_ds.data_vars
        assert "global_ID" in tracked_ds.data_vars
        assert "area" in tracked_ds.data_vars
        assert "centroid" in tracked_ds.data_vars
        assert "presence" in tracked_ds.data_vars
        assert "time_start" in tracked_ds.data_vars
        assert "time_end" in tracked_ds.data_vars
        assert "merge_ledger" in tracked_ds.data_vars

        # Verify merge dataset structure
        assert isinstance(merges_ds, xr.Dataset)
        assert "parent_IDs" in merges_ds.data_vars
        assert "child_IDs" in merges_ds.data_vars
        assert "overlap_areas" in merges_ds.data_vars
        assert "merge_time" in merges_ds.data_vars
        assert "n_parents" in merges_ds.data_vars
        assert "n_children" in merges_ds.data_vars

        # Verify advanced tracking attributes
        assert tracked_ds.attrs["allow_merging"] == 1
        assert tracked_ds.attrs["T_fill"] == 2
        assert "total_merges" in tracked_ds.attrs

        # Verify ID dimension consistency
        n_events = tracked_ds.sizes["ID"]
        assert n_events == tracked_ds.attrs["N_events_final"]

        # Verify that time_start <= time_end for all events
        valid_events = tracked_ds.presence.any(dim="time").compute()  # Compute the boolean mask
        for event_id in tracked_ds.ID[valid_events]:
            start_time = tracked_ds.time_start.sel(ID=event_id)
            end_time = tracked_ds.time_end.sel(ID=event_id)
            assert start_time <= end_time, f"Event {event_id} has start_time > end_time"

        # Assert tracking statistics are within reasonable bounds
        assert_reasonable_bounds(
            tracked_ds.attrs["preprocessed_area_fraction"],
            1.0,
            tolerance_relative=0.3,
        )
        assert_count_in_reasonable_range(tracked_ds.attrs["N_objects_prefiltered"], 200, tolerance=100)
        assert_count_in_reasonable_range(tracked_ds.attrs["N_objects_filtered"], 100, tolerance=75)
        assert_count_in_reasonable_range(tracked_ds.attrs["N_events_final"], 20, tolerance=10)
        assert_count_in_reasonable_range(tracked_ds.attrs["total_merges"], 15, tolerance=15)

    @pytest.mark.slow
    def test_unstructured_tracking_data_consistency(self, dask_client):
        """Test that unstructured tracking produces consistent data structures."""
        # Array broadcasting issue has been fixed
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            R_fill=3,
            area_filter_quartile=0.5,
            temp_dir=self.temp_dir,
            T_fill=2,
            allow_merging=True,
            unstructured_grid=True,  # Enable unstructured grid mode
            dimensions={"x": "ncells"},  # Must specify the name of the spatial dimension
            coordinates={
                "x": "lon",
                "y": "lat",
            },  # Coordinate mapping for unstructured grid
            regional_mode=False,  # Disable regional mode (not yet implemented)
            coordinate_units="degrees",  # Specify coordinate units
            quiet=True,
            # Provide unstructured grid information
            neighbours=self.extremes_data.neighbours,
            cell_areas=self.extremes_data.cell_areas,
        )

        tracked_ds = tracker.run()

        # Test that presence matches where global_ID is non-zero
        presence_from_global_id = tracked_ds.global_ID != 0
        assert tracked_ds.presence.equals(presence_from_global_id), "Presence doesn't match global_ID"

        # Test that area is positive where events are present
        present_events = tracked_ds.presence
        areas_np = tracked_ds.area.values
        areas_at_present = areas_np[present_events.values]
        assert (areas_at_present > 0).all(), "Some events have non-positive area"

        # Test that centroids are within reasonable bounds for unstructured data
        lat_centroids = tracked_ds.centroid.sel(component=0).values
        lon_centroids = tracked_ds.centroid.sel(component=1).values

        present_lat_centroids = lat_centroids[present_events.values]
        present_lon_centroids = lon_centroids[present_events.values]

        # Ignore nan values
        valid_lat = ~np.isnan(present_lat_centroids)
        valid_lon = ~np.isnan(present_lon_centroids)

        # Centroids should be within data bounds
        lat_min, lat_max = float(self.extremes_data.lat.min()), float(self.extremes_data.lat.max())
        lon_min, lon_max = float(self.extremes_data.lon.min()), float(self.extremes_data.lon.max())

        assert (present_lat_centroids[valid_lat] >= lat_min).all(), "Some centroids below lat bounds"
        assert (present_lat_centroids[valid_lat] <= lat_max).all(), "Some centroids above lat bounds"
        assert (present_lon_centroids[valid_lon] >= lon_min).all(), "Some centroids below lon bounds"
        assert (present_lon_centroids[valid_lon] <= lon_max).all(), "Some centroids above lon bounds"

        # Assert tracking statistics are within reasonable bounds
        assert_reasonable_bounds(
            tracked_ds.attrs["preprocessed_area_fraction"],
            1.0,
            tolerance_relative=0.3,
        )
        assert_count_in_reasonable_range(tracked_ds.attrs["N_objects_prefiltered"], 200, tolerance=100)
        assert_count_in_reasonable_range(tracked_ds.attrs["N_objects_filtered"], 100, tolerance=75)
        assert_count_in_reasonable_range(tracked_ds.attrs["N_events_final"], 5, tolerance=10)

    @pytest.mark.slow
    @pytest.mark.slow
    def test_unstructured_different_filtering_parameters(self, dask_client):
        """Test unstructured tracking with different area filtering parameters."""
        # Array broadcasting issue has been fixed
        # Use full dataset - subset too small to form trackable objects

        # Test with minimal filtering (quartile = 0.1)
        tracker_low_filter = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            R_fill=2,
            area_filter_quartile=0.1,
            temp_dir=self.temp_dir,
            T_fill=0,
            allow_merging=False,
            unstructured_grid=True,  # Enable unstructured grid mode
            dimensions={"x": "ncells"},  # Must specify the name of the spatial dimension
            coordinates={
                "x": "lon",
                "y": "lat",
            },  # Coordinate mapping for unstructured grid
            regional_mode=False,  # Disable regional mode (not yet implemented)
            coordinate_units="degrees",  # Specify coordinate units
            quiet=True,
            neighbours=self.extremes_data.neighbours,
            cell_areas=self.extremes_data.cell_areas,
        )

        # Test with aggressive filtering (quartile = 0.9)
        tracker_high_filter = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            R_fill=2,
            area_filter_quartile=0.9,
            temp_dir=self.temp_dir,
            T_fill=0,
            allow_merging=False,
            unstructured_grid=True,  # Enable unstructured grid mode
            dimensions={"x": "ncells"},  # Must specify the name of the spatial dimension
            coordinates={
                "x": "lon",
                "y": "lat",
            },  # Coordinate mapping for unstructured grid
            regional_mode=False,  # Disable regional mode (not yet implemented)
            coordinate_units="degrees",  # Specify coordinate units
            quiet=True,
            neighbours=self.extremes_data.neighbours,
            cell_areas=self.extremes_data.cell_areas,
        )

        tracked_low_filter = tracker_low_filter.run()
        tracked_high_filter = tracker_high_filter.run()

        # Higher filtering should result in fewer events
        n_events_low_filter = tracked_low_filter.attrs["N_events_final"]
        n_events_high_filter = tracked_high_filter.attrs["N_events_final"]

        assert n_events_high_filter <= n_events_low_filter, "High filtering should produce fewer or equal events"

        # Both should have valid ID fields
        assert int(tracked_low_filter.ID_field.max()) > 0
        assert int(tracked_high_filter.ID_field.max()) >= 0  # Could be 0 if all events filtered out

        # Assert tracking statistics are within reasonable bounds for low filter case
        assert_reasonable_bounds(
            tracked_low_filter.attrs["preprocessed_area_fraction"],
            1.0,
            tolerance_relative=0.4,
        )
        # Expected ranges for full dataset (732 timesteps × 1000 cells)
        assert_count_in_reasonable_range(tracked_low_filter.attrs["N_objects_prefiltered"], 800, tolerance=100)
        assert_count_in_reasonable_range(tracked_low_filter.attrs["N_objects_filtered"], 800, tolerance=100)
        assert_count_in_reasonable_range(tracked_low_filter.attrs["N_events_final"], 100, tolerance=50)

    @pytest.mark.slow
    def test_unstructured_temporal_gap_filling(self, dask_client):
        """Test that temporal gap filling works correctly on unstructured grids."""
        # Array broadcasting issue has been fixed
        # Use full dataset - subset too small to form trackable objects

        # Test with no gap filling
        tracker_no_gaps = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            R_fill=2,
            area_filter_quartile=0.5,
            temp_dir=self.temp_dir,
            T_fill=0,
            allow_merging=False,
            unstructured_grid=True,  # Enable unstructured grid mode
            dimensions={"x": "ncells"},  # Must specify the name of the spatial dimension
            coordinates={
                "x": "lon",
                "y": "lat",
            },  # Coordinate mapping for unstructured grid
            regional_mode=False,  # Disable regional mode (not yet implemented)
            coordinate_units="degrees",  # Specify coordinate units
            quiet=True,
            neighbours=self.extremes_data.neighbours,
            cell_areas=self.extremes_data.cell_areas,
        )

        # Test with gap filling
        tracker_with_gaps = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            R_fill=2,
            area_filter_quartile=0.5,
            temp_dir=self.temp_dir,
            T_fill=4,  # Allow 4-day gaps
            allow_merging=False,
            unstructured_grid=True,  # Enable unstructured grid mode
            dimensions={"x": "ncells"},  # Must specify the name of the spatial dimension
            coordinates={
                "x": "lon",
                "y": "lat",
            },  # Coordinate mapping for unstructured grid
            regional_mode=False,  # Disable regional mode (not yet implemented)
            coordinate_units="degrees",  # Specify coordinate units
            quiet=True,
            neighbours=self.extremes_data.neighbours,
            cell_areas=self.extremes_data.cell_areas,
        )

        tracked_no_gaps = tracker_no_gaps.run()
        tracked_with_gaps = tracker_with_gaps.run()

        # Gap filling should typically result in fewer total events (some are merged)
        # but longer individual events
        n_events_no_gaps = tracked_no_gaps.attrs["N_events_final"]
        n_events_with_gaps = tracked_with_gaps.attrs["N_events_final"]

        # Both should produce valid results
        assert n_events_no_gaps > 0, "No gap filling should produce some events"
        assert n_events_with_gaps > 0, "Gap filling should produce some events"

        # Verify T_fill attribute is correctly set
        assert tracked_no_gaps.attrs["T_fill"] == 0
        assert tracked_with_gaps.attrs["T_fill"] == 4

        # Assert tracking statistics are within reasonable bounds
        assert_reasonable_bounds(
            tracked_no_gaps.attrs["preprocessed_area_fraction"],
            1.0,
            tolerance_relative=0.4,
        )
        # Expected ranges for full dataset (732 timesteps × 1000 cells)
        assert_count_in_reasonable_range(tracked_no_gaps.attrs["N_objects_prefiltered"], 200, tolerance=100)
        assert_count_in_reasonable_range(tracked_no_gaps.attrs["N_objects_filtered"], 100, tolerance=75)
        assert_count_in_reasonable_range(tracked_no_gaps.attrs["N_events_final"], 40, tolerance=20)

    def test_unstructured_grid_requirements(self, dask_client):
        """Test that unstructured tracking properly validates grid requirements."""
        # Test now enabled with array broadcasting fix
        # Test that tracking fails gracefully without neighbors information
        with pytest.raises((ValueError, TypeError, AttributeError)):
            tracker = marEx.tracker(
                self.extremes_data.extreme_events,
                self.extremes_data.mask,
                R_fill=2,
                area_filter_quartile=0.5,
                temp_dir=self.temp_dir,
                T_fill=0,
                allow_merging=False,
                unstructured_grid=True,  # Enable unstructured grid mode
                dimensions={"x": "ncells"},  # Must specify the name of the spatial dimension
                coordinates={"x": "lon", "y": "lat"},
                regional_mode=False,
                coordinate_units="degrees",
                quiet=True,
                # Missing neighbours and cell_areas - should cause error
            )
            tracker.run()

    @pytest.mark.slow
    def test_unstructured_centroid_calculation(self, dask_client):
        """Test that centroids are calculated correctly for unstructured data."""
        # Array broadcasting issue has been fixed
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            R_fill=2,
            area_filter_quartile=0.6,  # Higher filtering for cleaner test
            temp_dir=self.temp_dir,
            T_fill=0,
            allow_merging=True,
            unstructured_grid=True,  # Enable unstructured grid mode
            dimensions={"x": "ncells"},  # Must specify the name of the spatial dimension
            coordinates={
                "x": "lon",
                "y": "lat",
            },  # Coordinate mapping for unstructured grid
            regional_mode=False,  # Disable regional mode (not yet implemented)
            coordinate_units="degrees",  # Specify coordinate units
            quiet=True,
            neighbours=self.extremes_data.neighbours,
            cell_areas=self.extremes_data.cell_areas,
        )

        tracked_ds = tracker.run()

        # Get events that are present
        present_events = tracked_ds.presence.any(dim="time")
        n_present_events = present_events.sum().compute().item()

        assert n_present_events > 0, "No events found for centroid testing"

        # Check that centroids are calculated for present events
        for event_id in tracked_ds.ID[present_events.compute()]:
            event_presence = tracked_ds.presence.sel(ID=event_id)
            event_centroids = tracked_ds.centroid.sel(ID=event_id)

            # Where the event is present, centroids should not be NaN
            present_times = event_presence
            if present_times.any():
                lat_centroids = event_centroids.sel(component=0).where(present_times)
                lon_centroids = event_centroids.sel(component=1).where(present_times)

                # At least some centroids should be non-NaN where event is present
                assert not np.isnan(lat_centroids).all(), f"All lat centroids are NaN for event {event_id}"
                assert not np.isnan(lon_centroids).all(), f"All lon centroids are NaN for event {event_id}"

    @pytest.mark.slow
    def test_unstructured_tracking_memory_efficiency(self, dask_client):
        """Test that unstructured tracking handles memory efficiently."""
        # Array broadcasting issue has been fixed
        # Use smaller chunks to test memory management
        extremes_rechunked = self.extremes_data.extreme_events.chunk({"time": 1, "ncells": 500})
        mask_rechunked = self.extremes_data.mask.chunk({"ncells": 500})

        tracker = marEx.tracker(
            extremes_rechunked,
            mask_rechunked,
            R_fill=2,
            area_filter_quartile=0.7,  # Higher filtering to reduce memory usage
            temp_dir=self.temp_dir,
            T_fill=2,  # Must be even for temporal symmetry
            allow_merging=False,
            unstructured_grid=True,  # Enable unstructured grid mode
            dimensions={"x": "ncells"},  # Must specify the name of the spatial dimension
            coordinates={
                "x": "lon",
                "y": "lat",
            },  # Coordinate mapping for unstructured grid
            regional_mode=False,  # Disable regional mode (not yet implemented)
            coordinate_units="degrees",  # Specify coordinate units
            quiet=True,
            neighbours=self.extremes_data.neighbours,
            cell_areas=self.extremes_data.cell_areas,
        )

        # This should complete without memory errors
        tracked_ds = tracker.run()

        # Basic validation
        assert isinstance(tracked_ds, xr.Dataset)
        assert "ID_field" in tracked_ds.data_vars
        assert tracked_ds.attrs["N_events_final"] >= 0

    def test_custom_dimension_names_unstructured_tracking(self, dask_client):
        """Test unstructured tracking with custom dimension and coordinate names for both allow_merging options."""
        # Use a much smaller subset for custom dimension tests: 50 timesteps and 200 cells
        # This reduces computational load while still testing the functionality
        subset_data = self.extremes_data.isel(time=slice(0, 50), ncells=slice(0, 200))

        # For unstructured grids, we need to adjust the neighbours array to match the subset
        # Set any neighbour indices >= 200 to -1 (invalid) to avoid index out of bounds
        subset_neighbours = subset_data.neighbours.where(subset_data.neighbours < 200, -1)
        subset_data = subset_data.assign(neighbours=subset_neighbours)

        # Rename dimensions to: "t" and "cell"
        # Rename coordinates to: "latitude", "longitude"
        renamed_data = subset_data.rename({"time": "t", "ncells": "cell"}).rename({"lat": "latitude", "lon": "longitude"})

        # Test 1: Tracking with allow_merging=False
        tracker_no_merge = marEx.tracker(
            renamed_data.extreme_events,
            renamed_data.mask,
            R_fill=2,
            area_filter_quartile=0.8,
            temp_dir=self.temp_dir,
            T_fill=0,  # No temporal filling for basic test
            allow_merging=False,
            unstructured_grid=True,
            dimensions={"time": "t", "x": "cell"},  # Must specify the name of the spatial dimension
            coordinates={"x": "longitude", "y": "latitude"},  # Use custom coordinate names
            regional_mode=False,
            coordinate_units="degrees",
            quiet=True,
            neighbours=renamed_data.neighbours,
            cell_areas=renamed_data.cell_areas,
        )

        # Verify tracker was created successfully with custom dimension and coordinate names (no merging)
        assert tracker_no_merge is not None
        assert hasattr(tracker_no_merge, "dilate_sparse")

        # Note: Actual tracking execution is currently skipped due to unstructured tracking issues
        # This test validates tracker initialisation with custom dimensions/coordinates

        # Test 2: Tracking with allow_merging=True
        tracker_with_merge = marEx.tracker(
            renamed_data.extreme_events,
            renamed_data.mask,
            R_fill=2,
            area_filter_quartile=0.8,
            temp_dir=self.temp_dir,
            T_fill=2,  # Allow temporal filling
            allow_merging=True,
            overlap_threshold=0.5,
            nn_partitioning=True,
            unstructured_grid=True,
            dimensions={"time": "t", "x": "cell"},  # Must specify the name of the spatial dimension
            coordinates={"x": "longitude", "y": "latitude"},  # Use custom coordinate names
            regional_mode=False,
            coordinate_units="degrees",
            quiet=True,
            neighbours=renamed_data.neighbours,
            cell_areas=renamed_data.cell_areas,
        )

        # Verify tracker was created successfully with custom dimension and coordinate names (with merging)
        assert tracker_with_merge is not None
        assert hasattr(tracker_with_merge, "dilate_sparse")

        # Test 3: Verify different configurations were applied correctly
        # Check internal attributes to verify configuration differences
        assert hasattr(tracker_no_merge, "T_fill")
        assert hasattr(tracker_with_merge, "T_fill")
        assert hasattr(tracker_no_merge, "allow_merging")
        assert hasattr(tracker_with_merge, "allow_merging")

        # Note: The following would be the full tracking test when unstructured tracking is fixed:
        #
        # # Run tracking without merging
        # tracked_ds_no_merge = tracker_no_merge.run()
        #
        # # Verify output structure (no merging)
        # assert isinstance(tracked_ds_no_merge, xr.Dataset)
        # assert "ID_field" in tracked_ds_no_merge.data_vars
        # assert "t" in tracked_ds_no_merge.ID_field.dims
        # assert "cell" in tracked_ds_no_merge.ID_field.dims
        # assert tracked_ds_no_merge.attrs["allow_merging"] == 0
        #
        # # Run tracking with merging
        # tracked_ds_with_merge = tracker_with_merge.run()
        #
        # # Verify output structure (with merging)
        # assert isinstance(tracked_ds_with_merge, xr.Dataset)
        # assert "ID_field" in tracked_ds_with_merge.data_vars
        # assert "global_ID" in tracked_ds_with_merge.data_vars
        # assert "area" in tracked_ds_with_merge.data_vars
        # assert "centroid" in tracked_ds_with_merge.data_vars
        # assert "presence" in tracked_ds_with_merge.data_vars
        # assert "time_start" in tracked_ds_with_merge.data_vars
        # assert "time_end" in tracked_ds_with_merge.data_vars
        # assert "merge_ledger" in tracked_ds_with_merge.data_vars
        # assert "t" in tracked_ds_with_merge.ID_field.dims
        # assert "cell" in tracked_ds_with_merge.ID_field.dims
        # assert tracked_ds_with_merge.attrs["allow_merging"] == 1
        # assert tracked_ds_with_merge.attrs["T_fill"] == 2
        # assert "total_merges" in tracked_ds_with_merge.attrs

    def test_custom_dimension_names_comparison_with_original(self, dask_client):
        """Test that custom dimension names produce equivalent results to original dimension names."""
        # Use the same smaller subset as in the custom dimension test
        subset_data = self.extremes_data.isel(time=slice(0, 50), ncells=slice(0, 200))

        # For unstructured grids, we need to adjust the neighbours array to match the subset
        # Set any neighbour indices >= 200 to -1 (invalid) to avoid index out of bounds
        subset_neighbours = subset_data.neighbours.where(subset_data.neighbours < 200, -1)
        subset_data = subset_data.assign(neighbours=subset_neighbours)

        # Test 1: Original dimension names
        tracker_original = marEx.tracker(
            subset_data.extreme_events,
            subset_data.mask,
            R_fill=2,
            area_filter_quartile=0.8,
            temp_dir=self.temp_dir,
            T_fill=0,  # No temporal filling for basic test
            allow_merging=False,
            unstructured_grid=True,
            dimensions={"time": "time", "x": "ncells"},  # Original dimension names
            coordinates={"x": "lon", "y": "lat"},  # Original coordinate names
            regional_mode=False,
            coordinate_units="degrees",
            quiet=True,
            neighbours=self.extremes_data.neighbours,
            cell_areas=self.extremes_data.cell_areas,
        )

        # Test 2: Custom dimension names (renamed data)
        renamed_data = subset_data.rename({"time": "t", "ncells": "cell"}).rename({"lat": "latitude", "lon": "longitude"})

        tracker_custom = marEx.tracker(
            renamed_data.extreme_events,
            renamed_data.mask,
            R_fill=2,
            area_filter_quartile=0.8,
            temp_dir=self.temp_dir,
            T_fill=0,  # No temporal filling for basic test
            allow_merging=False,
            unstructured_grid=True,
            dimensions={"time": "t", "x": "cell"},  # Custom dimension names
            coordinates={"x": "longitude", "y": "latitude"},  # Custom coordinate names
            regional_mode=False,
            coordinate_units="degrees",
            quiet=True,
            neighbours=renamed_data.neighbours,
            cell_areas=renamed_data.cell_areas,
        )

        # Verify both trackers were created successfully
        assert tracker_original is not None
        assert tracker_custom is not None
        assert hasattr(tracker_original, "dilate_sparse")
        assert hasattr(tracker_custom, "dilate_sparse")

        # Test that configuration parameters are the same
        assert tracker_original.R_fill == tracker_custom.R_fill
        assert tracker_original.area_filter_quartile == tracker_custom.area_filter_quartile
        assert tracker_original.T_fill == tracker_custom.T_fill
        assert tracker_original.allow_merging == tracker_custom.allow_merging

        # Note: Full comparison would require running the tracking and comparing results
        # This test validates that both configurations can be created successfully
        # and have equivalent parameters
