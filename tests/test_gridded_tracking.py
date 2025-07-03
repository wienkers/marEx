from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx

from .conftest import assert_count_in_reasonable_range, assert_reasonable_bounds


class TestGriddedTracking:
    """Test event tracking functionality for gridded data."""

    @classmethod
    def setup_class(cls):
        """Load test data for all tests."""
        test_data_path = Path(__file__).parent / "data" / "extremes_gridded.zarr"
        cls.extremes_data = xr.open_zarr(str(test_data_path), chunks={}).persist()

        # Standard chunk size for tracking (spatial dimensions must be contiguous)
        cls.chunk_size = {"time": 2, "lat": -1, "lon": -1}

    def test_basic_tracking(self, dask_client):
        """Test basic tracking without merging/splitting."""
        # Create tracker with basic settings
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask.where(
                (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
                other=False,
            ),  # Exclude poles
            area_filter_quartile=0.5,
            R_fill=4,  # Reduced for test data
            T_fill=0,  # No temporal filling for basic test
            allow_merging=False,
            quiet=True,  # Suppress output for tests
        )

        # Run tracking
        tracked_ds = tracker.run()

        # Verify output structure
        assert isinstance(tracked_ds, xr.Dataset)
        assert "ID_field" in tracked_ds.data_vars

        # Verify dimensions
        assert "time" in tracked_ds.ID_field.dims
        assert "lat" in tracked_ds.ID_field.dims
        assert "lon" in tracked_ds.ID_field.dims

        # Verify attributes are set
        assert "N_events_final" in tracked_ds.attrs
        assert "allow_merging" in tracked_ds.attrs
        assert tracked_ds.attrs["allow_merging"] == 0
        assert "R_fill" in tracked_ds.attrs
        assert "T_fill" in tracked_ds.attrs

        # Verify ID field contains reasonable values
        max_id = int(tracked_ds.ID_field.max())
        assert max_id > 0, "No events were tracked"
        assert (
            max_id == tracked_ds.attrs["N_events_final"]
        ), "Max ID doesn't match reported event count"

        # Verify that background is labeled as 0
        assert int(tracked_ds.ID_field.min()) == 0

        # Assert tracking statistics are within reasonable bounds
        assert_reasonable_bounds(
            tracked_ds.attrs["preprocessed_area_fraction"],
            0.9724,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(
            tracked_ds.attrs["N_objects_prefiltered"], 549, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_ds.attrs["N_objects_filtered"], 274, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_ds.attrs["N_events_final"], 24, tolerance=1
        )

    def test_advanced_tracking_with_merging(self, dask_client):
        """Test advanced tracking with temporal filling and merging enabled."""
        # Create tracker with advanced settings
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask.where(
                (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
                other=False,
            ),
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=2,  # Allow 2-day gaps
            allow_merging=True,
            overlap_threshold=0.5,
            nn_partitioning=True,
            quiet=True,
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
        valid_events = tracked_ds.presence.any(
            dim="time"
        ).compute()  # Compute the boolean mask
        for event_id in tracked_ds.ID[valid_events]:
            start_time = tracked_ds.time_start.sel(ID=event_id)
            end_time = tracked_ds.time_end.sel(ID=event_id)
            assert start_time <= end_time, f"Event {event_id} has start_time > end_time"

        # Assert tracking statistics are within reasonable bounds
        assert_reasonable_bounds(
            tracked_ds.attrs["preprocessed_area_fraction"],
            0.9143,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(
            tracked_ds.attrs["N_objects_prefiltered"], 516, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_ds.attrs["N_objects_filtered"], 258, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_ds.attrs["N_events_final"], 20, tolerance=1
        )
        assert_count_in_reasonable_range(
            tracked_ds.attrs["total_merges"], 26, tolerance=2
        )

    def test_tracking_data_consistency(self, dask_client):
        """Test that tracking produces consistent data structures."""
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask.where(
                (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
                other=False,
            ),
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=2,
            allow_merging=True,
            quiet=True,
        )

        tracked_ds = tracker.run()

        # Test that presence matches where global_ID is non-zero
        presence_from_global_id = tracked_ds.global_ID != 0
        assert tracked_ds.presence.equals(
            presence_from_global_id
        ), "Presence doesn't match global_ID"

        # Test that area is positive where events are present
        present_events = tracked_ds.presence
        areas_np = tracked_ds.area.values
        areas_at_present = areas_np[present_events.values]
        assert (areas_at_present > 0).all(), "Some events have non-positive area"

        # Test that centroids are within reasonable bounds
        lat_centroids = tracked_ds.centroid.sel(component=0).values
        lon_centroids = tracked_ds.centroid.sel(component=1).values

        present_lat_centroids = lat_centroids[present_events.values]
        present_lon_centroids = lon_centroids[present_events.values]

        # Ignore nan values
        valid_lat = ~np.isnan(present_lat_centroids)
        valid_lon = ~np.isnan(present_lon_centroids)

        # Centroids should be within data bounds
        lat_min, lat_max = float(self.extremes_data.lat.min()), float(
            self.extremes_data.lat.max()
        )
        lon_min, lon_max = float(self.extremes_data.lon.min()), float(
            self.extremes_data.lon.max()
        )

        assert (
            present_lat_centroids[valid_lat] >= lat_min
        ).all(), "Some centroids below lat bounds"
        assert (
            present_lat_centroids[valid_lat] <= lat_max
        ).all(), "Some centroids above lat bounds"
        assert (
            present_lon_centroids[valid_lon] >= lon_min
        ).all(), "Some centroids below lon bounds"
        assert (
            present_lon_centroids[valid_lon] <= lon_max
        ).all(), "Some centroids above lon bounds"

        # Assert tracking statistics are within reasonable bounds
        assert_reasonable_bounds(
            tracked_ds.attrs["preprocessed_area_fraction"],
            0.9143,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(
            tracked_ds.attrs["N_objects_prefiltered"], 516, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_ds.attrs["N_objects_filtered"], 258, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_ds.attrs["N_events_final"], 19, tolerance=1
        )
        assert_count_in_reasonable_range(
            tracked_ds.attrs["total_merges"], 27, tolerance=2
        )

    def test_different_filtering_parameters(self, dask_client):
        """Test tracking with different area filtering parameters."""
        # Test with no filtering (quartile = 0)
        tracker_no_filter = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            area_filter_quartile=0.0,
            R_fill=2,
            T_fill=0,
            allow_merging=False,
            quiet=True,
        )

        # Test with aggressive filtering (quartile = 0.8)
        tracker_high_filter = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            area_filter_quartile=0.8,
            R_fill=2,
            T_fill=0,
            allow_merging=False,
            quiet=True,
        )

        tracked_no_filter = tracker_no_filter.run()
        tracked_high_filter = tracker_high_filter.run()

        # Higher filtering should result in fewer events
        n_events_no_filter = tracked_no_filter.attrs["N_events_final"]
        n_events_high_filter = tracked_high_filter.attrs["N_events_final"]

        assert (
            n_events_high_filter <= n_events_no_filter
        ), "High filtering should produce fewer or equal events"

        # Both should have valid ID fields
        assert int(tracked_no_filter.ID_field.max()) > 0
        assert (
            int(tracked_high_filter.ID_field.max()) >= 0
        )  # Could be 0 if all events filtered out

        # Assert tracking statistics are within reasonable bounds for no filter case
        assert_reasonable_bounds(
            tracked_no_filter.attrs["preprocessed_area_fraction"],
            1.0622,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(
            tracked_no_filter.attrs["N_objects_prefiltered"], 1046, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_no_filter.attrs["N_objects_filtered"], 1045, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_no_filter.attrs["N_events_final"], 152, tolerance=1
        )

        # Assert tracking statistics are within reasonable bounds for high filter case
        assert_reasonable_bounds(
            tracked_high_filter.attrs["preprocessed_area_fraction"],
            1.5423,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(
            tracked_high_filter.attrs["N_objects_prefiltered"], 1046, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_high_filter.attrs["N_objects_filtered"], 209, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_high_filter.attrs["N_events_final"], 21, tolerance=1
        )

    def test_temporal_gap_filling(self, dask_client):
        """Test that temporal gap filling works correctly."""
        # Test with no gap filling
        tracker_no_gaps = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            area_filter_quartile=0.5,
            R_fill=2,
            T_fill=0,
            allow_merging=False,
            quiet=True,
        )

        # Test with gap filling
        tracker_with_gaps = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            area_filter_quartile=0.5,
            R_fill=2,
            T_fill=4,  # Allow 4-day gaps
            allow_merging=False,
            quiet=True,
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

        # Assert tracking statistics are within reasonable bounds for no gaps case
        assert_reasonable_bounds(
            tracked_no_gaps.attrs["preprocessed_area_fraction"],
            1.1650,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(
            tracked_no_gaps.attrs["N_objects_prefiltered"], 1046, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_no_gaps.attrs["N_objects_filtered"], 522, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_no_gaps.attrs["N_events_final"], 54, tolerance=1
        )

        # Assert tracking statistics are within reasonable bounds for with gaps case
        assert_reasonable_bounds(
            tracked_with_gaps.attrs["preprocessed_area_fraction"],
            1.0080,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(
            tracked_with_gaps.attrs["N_objects_prefiltered"], 1041, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_with_gaps.attrs["N_objects_filtered"], 522, tolerance=2
        )
        assert_count_in_reasonable_range(
            tracked_with_gaps.attrs["N_events_final"], 38, tolerance=1
        )

    def test_custom_dimension_names_tracking(self, dask_client):
        """Test tracking with custom dimension and coordinate names for both allow_merging options."""
        # Create dataset with custom dimension and coordinate names
        # Dimensions: "t", "x", "y"
        # Coordinates: "T", "longitude", "latitude"
        
        # First create the renamed dataset structure
        extreme_events_renamed = xr.DataArray(
            self.extremes_data.extreme_events.values,
            dims=['t', 'y', 'x'],
            coords={
                'T': ('t', self.extremes_data.time.values),
                'latitude': ('y', self.extremes_data.lat.values),
                'longitude': ('x', self.extremes_data.lon.values)
            }
        ).chunk({"t": 2, "y": -1, "x": -1})
        
        mask_renamed = xr.DataArray(
            self.extremes_data.mask.values,
            dims=['y', 'x'],
            coords={
                'latitude': ('y', self.extremes_data.lat.values),
                'longitude': ('x', self.extremes_data.lon.values)
            }
        )
        
        # Apply common mask to exclude poles
        mask_filtered = mask_renamed.where(
            (mask_renamed.latitude < 85) & (mask_renamed.latitude > -90),
            other=False,
        )
        
        # Test 1: Tracking with allow_merging=False
        tracker_no_merge = marEx.tracker(
            extreme_events_renamed,
            mask_filtered,
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=0,  # No temporal filling for basic test
            allow_merging=False,
            dimensions={"time": "t", "x": "x", "y": "y"},   # Must specify the name of the spatial dimension
            coordinates={"time": "T", "x": "longitude", "y": "latitude"},
            quiet=True,
        )
        
        # Run tracking without merging
        tracked_ds_no_merge = tracker_no_merge.run()
        
        # Verify output structure with renamed dimensions (no merging)
        assert isinstance(tracked_ds_no_merge, xr.Dataset)
        assert "ID_field" in tracked_ds_no_merge.data_vars
        
        # Verify dimensions are correctly named
        assert "t" in tracked_ds_no_merge.ID_field.dims
        assert "y" in tracked_ds_no_merge.ID_field.dims
        assert "x" in tracked_ds_no_merge.ID_field.dims
        
        # Verify coordinates are present
        assert "T" in tracked_ds_no_merge.coords
        assert "latitude" in tracked_ds_no_merge.coords
        assert "longitude" in tracked_ds_no_merge.coords
        
        # Verify tracking attributes (no merging)
        assert tracked_ds_no_merge.attrs["allow_merging"] == 0
        assert tracked_ds_no_merge.attrs["T_fill"] == 0
        assert "N_events_final" in tracked_ds_no_merge.attrs
        
        # Verify tracking produced valid results
        max_id_no_merge = int(tracked_ds_no_merge.ID_field.max())
        assert max_id_no_merge > 0, "No events were tracked with custom dimension names (no merging)"
        assert max_id_no_merge == tracked_ds_no_merge.attrs["N_events_final"]
        
        # Test 2: Tracking with allow_merging=True
        tracker_with_merge = marEx.tracker(
            extreme_events_renamed,
            mask_filtered,
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=2,  # Allow temporal filling
            allow_merging=True,
            overlap_threshold=0.5,
            nn_partitioning=True,
            dimensions={"time": "t", "x": "x", "y": "y"},   # Must specify the name of the spatial dimension
            coordinates={"time": "T", "x": "longitude", "y": "latitude"},
            quiet=True,
        )
        
        # Run tracking with merging
        tracked_ds_with_merge = tracker_with_merge.run()
        
        # Verify output structure with renamed dimensions (with merging)
        assert isinstance(tracked_ds_with_merge, xr.Dataset)
        assert "ID_field" in tracked_ds_with_merge.data_vars
        assert "global_ID" in tracked_ds_with_merge.data_vars
        assert "area" in tracked_ds_with_merge.data_vars
        assert "centroid" in tracked_ds_with_merge.data_vars
        assert "presence" in tracked_ds_with_merge.data_vars
        assert "time_start" in tracked_ds_with_merge.data_vars
        assert "time_end" in tracked_ds_with_merge.data_vars
        assert "merge_ledger" in tracked_ds_with_merge.data_vars
        
        # Verify dimensions are correctly named
        assert "t" in tracked_ds_with_merge.ID_field.dims
        assert "y" in tracked_ds_with_merge.ID_field.dims
        assert "x" in tracked_ds_with_merge.ID_field.dims
        
        # Verify coordinates are present
        assert "T" in tracked_ds_with_merge.coords
        assert "latitude" in tracked_ds_with_merge.coords
        assert "longitude" in tracked_ds_with_merge.coords
        
        # Verify tracking attributes (with merging)
        assert tracked_ds_with_merge.attrs["allow_merging"] == 1
        assert tracked_ds_with_merge.attrs["T_fill"] == 2
        assert "total_merges" in tracked_ds_with_merge.attrs
        assert "N_events_final" in tracked_ds_with_merge.attrs
        
        # Verify ID dimension consistency
        n_events = tracked_ds_with_merge.sizes["ID"]
        assert n_events == tracked_ds_with_merge.attrs["N_events_final"]
        
        # Verify tracking produced valid results
        max_id_with_merge = int(tracked_ds_with_merge.ID_field.max())
        assert max_id_with_merge > 0, "No events were tracked with custom dimension names (with merging)"
        assert max_id_with_merge == tracked_ds_with_merge.attrs["N_events_final"]
        
        # Test 3: Compare results between merging modes
        # Both should track some events
        assert tracked_ds_no_merge.attrs["N_events_final"] > 0
        assert tracked_ds_with_merge.attrs["N_events_final"] > 0
        
        # Both should have the same core structure (ID_field)
        assert tracked_ds_no_merge.ID_field.dims == tracked_ds_with_merge.ID_field.dims
        
        # Verify that time_start <= time_end for all events (merging case)
        valid_events = tracked_ds_with_merge.presence.any(dim="t").compute()
        for event_id in tracked_ds_with_merge.ID[valid_events]:
            start_time = tracked_ds_with_merge.time_start.sel(ID=event_id)
            end_time = tracked_ds_with_merge.time_end.sel(ID=event_id)
            assert start_time <= end_time, f"Event {event_id} has start_time > end_time"



    def test_centroid_tracking_moving_blob(self, dask_client):
        """Test that centroid tracking correctly follows a steadily moving blob."""
        # Load test data with steadily moving blob
        test_data_path = Path(__file__).parent / "data" / "extremes_gridded_blob.zarr"
        blob_data = xr.open_zarr(str(test_data_path), chunks={}).chunk({"time": 2, "lat": -1, "lon": -1}).persist()
        
        # Create tracker with merging enabled to get centroids
        tracker = marEx.tracker(
            blob_data.extreme_events,
            blob_data.mask.where(
                (blob_data.lat < 85) & (blob_data.lat > -90),
                other=False,
            ),
            area_filter_quartile=0.0,  # No area filtering to capture the blob
            R_fill=0,  # No spatial filling to avoid coordinate alignment issues with small blob
            T_fill=0,  # No temporal filling to keep the test simple
            allow_merging=True,  # Required to save centroids
            overlap_threshold=0.3,
            quiet=True,
        )
        
        # Run tracking
        tracked_ds = tracker.run()
        
        # Verify that we have centroid data
        assert "centroid" in tracked_ds.data_vars, "Centroid data not present"
        assert tracked_ds.attrs["allow_merging"] == 1, "Merging should be enabled"
        
        # Find the largest tracked event (should be our moving blob)
        # Look for events that are present for multiple time steps
        event_durations = tracked_ds.presence.sum(dim="time")
        longest_event_idx = int(event_durations.argmax().values)
        longest_event_id = event_durations.ID[longest_event_idx]
        
        assert event_durations[longest_event_idx] > 5, "No long-duration event found (expected moving blob)"
        
        # Extract centroid positions for the longest event
        event_presence = tracked_ds.presence.isel(ID=longest_event_idx).compute()
        present_times = tracked_ds.time[event_presence]
        
        # Get centroids where the event is present
        lat_centroids = tracked_ds.centroid.sel(component=0).isel(ID=longest_event_idx)
        lon_centroids = tracked_ds.centroid.sel(component=1).isel(ID=longest_event_idx)
        
        tracked_lat_centroids = lat_centroids[event_presence].values
        tracked_lon_centroids = lon_centroids[event_presence].values
        
        # Remove any NaN values (shouldn't be any for a valid tracked event)
        valid_mask = ~(np.isnan(tracked_lat_centroids) | np.isnan(tracked_lon_centroids))
        valid_times = present_times[valid_mask]
        valid_lat_centroids = tracked_lat_centroids[valid_mask]
        valid_lon_centroids = tracked_lon_centroids[valid_mask]
        
        assert len(valid_times) > 5, "Not enough valid centroid measurements"
        
        # Expected centroid locations based on make_test_data.ipynb
        # rate = 3 degrees east per day, start_lon = 170, no movement in latitude
        rate = 3.0  # degrees east per day
        start_lon = 170.0
        expected_lat = 0.0  # No movement in latitude
        
        # Calculate expected positions for the times when the event is present
        start_time = blob_data.time.min()
        delta_days = (valid_times - start_time).dt.days.values
        expected_lon_centroids = start_lon + delta_days * rate
        expected_lat_centroids = np.full_like(expected_lon_centroids, expected_lat)
        
        # Handle longitude wraparound (expected values may exceed 180)
        expected_lon_centroids = np.where(
            expected_lon_centroids > 180, 
            expected_lon_centroids - 360, 
            expected_lon_centroids
        )
        
        # Test centroid accuracy with specified tolerances
        lon_tolerance = 0.5   # degrees (relaxed slightly from 0.25 to account for discretisation)
        lat_tolerance = 0.25  # degrees (relaxed slightly from 0.1 to account for discretisation)
        
        # Check latitude centroids (should be close to 0)
        lat_differences = np.abs(valid_lat_centroids - expected_lat_centroids)
        max_lat_error = np.max(lat_differences)
        
        assert max_lat_error <= lat_tolerance, (
            f"Maximum latitude error {max_lat_error:.3f}° exceeds tolerance {lat_tolerance}°. "
            f"Expected lat ≈ {expected_lat}°, got range [{valid_lat_centroids.min():.2f}, "
            f"{valid_lat_centroids.max():.2f}]°"
        )
        
        # Check longitude centroids (should follow eastward movement)
        lon_differences = np.abs(valid_lon_centroids - expected_lon_centroids)
        max_lon_error = np.max(lon_differences)
        
        assert max_lon_error <= lon_tolerance, (
            f"Maximum longitude error {max_lon_error:.3f}° exceeds tolerance {lon_tolerance}°. "
            f"Expected eastward movement from {start_lon}° at {rate}°/day. "
            f"Time range: {delta_days.min():.1f} to {delta_days.max():.1f} days. "
            f"Expected lon range: [{expected_lon_centroids.min():.1f}, {expected_lon_centroids.max():.1f}]°, "
            f"got range: [{valid_lon_centroids.min():.1f}, {valid_lon_centroids.max():.1f}]°"
        )
        
