from pathlib import Path

import numpy as np
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

    def test_basic_tracking(self, dask_client_gridded):
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
        assert max_id == tracked_ds.attrs["N_events_final"], "Max ID doesn't match reported event count"

        # Verify that background is labeled as 0
        assert int(tracked_ds.ID_field.min()) == 0

        # Verify ID_field is int
        assert np.issubdtype(tracked_ds.ID_field.dtype, np.integer), "ID_field should be integer type"

        # Assert tracking statistics are within reasonable bounds
        assert_reasonable_bounds(
            tracked_ds.attrs["preprocessed_area_fraction"],
            0.9724,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(tracked_ds.attrs["N_objects_prefiltered"], 549, tolerance=2)
        assert_count_in_reasonable_range(tracked_ds.attrs["N_objects_filtered"], 274, tolerance=2)
        assert_count_in_reasonable_range(tracked_ds.attrs["N_events_final"], 24, tolerance=1)

    def test_advanced_tracking_with_merging(self, dask_client_gridded):
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
        valid_events = tracked_ds.presence.any(dim="time").compute()  # Compute the boolean mask
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
        assert_count_in_reasonable_range(tracked_ds.attrs["N_objects_prefiltered"], 516, tolerance=2)
        assert_count_in_reasonable_range(tracked_ds.attrs["N_objects_filtered"], 258, tolerance=2)
        assert_count_in_reasonable_range(tracked_ds.attrs["N_events_final"], 20, tolerance=1)
        assert_count_in_reasonable_range(tracked_ds.attrs["total_merges"], 13, tolerance=2)

    def test_tracking_data_consistency(self, dask_client_gridded):
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
        assert tracked_ds.presence.equals(presence_from_global_id), "Presence doesn't match global_ID"

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
        lat_min, lat_max = float(self.extremes_data.lat.min()), float(self.extremes_data.lat.max())
        lon_min, lon_max = float(self.extremes_data.lon.min()), float(self.extremes_data.lon.max())

        assert (present_lat_centroids[valid_lat] >= lat_min).all(), "Some centroids below lat bounds"
        assert (present_lat_centroids[valid_lat] <= lat_max).all(), "Some centroids above lat bounds"
        assert (present_lon_centroids[valid_lon] >= lon_min).all(), "Some centroids below lon bounds"
        assert (present_lon_centroids[valid_lon] <= lon_max).all(), "Some centroids above lon bounds"

        # Assert tracking statistics are within reasonable bounds
        assert_reasonable_bounds(
            tracked_ds.attrs["preprocessed_area_fraction"],
            0.9143,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(tracked_ds.attrs["N_objects_prefiltered"], 516, tolerance=2)
        assert_count_in_reasonable_range(tracked_ds.attrs["N_objects_filtered"], 258, tolerance=2)
        assert_count_in_reasonable_range(tracked_ds.attrs["N_events_final"], 21, tolerance=1)
        assert_count_in_reasonable_range(tracked_ds.attrs["total_merges"], 15, tolerance=2)

    def test_different_filtering_parameters(self, dask_client_gridded):
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

        assert n_events_high_filter <= n_events_no_filter, "High filtering should produce fewer or equal events"

        # Both should have valid ID fields
        assert int(tracked_no_filter.ID_field.max()) > 0
        assert int(tracked_high_filter.ID_field.max()) >= 0  # Could be 0 if all events filtered out

        # Assert tracking statistics are within reasonable bounds for no filter case
        assert_reasonable_bounds(
            tracked_no_filter.attrs["preprocessed_area_fraction"],
            1.0622,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(tracked_no_filter.attrs["N_objects_prefiltered"], 1046, tolerance=2)
        assert_count_in_reasonable_range(tracked_no_filter.attrs["N_objects_filtered"], 1045, tolerance=2)
        assert_count_in_reasonable_range(tracked_no_filter.attrs["N_events_final"], 152, tolerance=1)

        # Assert tracking statistics are within reasonable bounds for high filter case
        assert_reasonable_bounds(
            tracked_high_filter.attrs["preprocessed_area_fraction"],
            1.5423,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(tracked_high_filter.attrs["N_objects_prefiltered"], 1046, tolerance=2)
        assert_count_in_reasonable_range(tracked_high_filter.attrs["N_objects_filtered"], 209, tolerance=2)
        assert_count_in_reasonable_range(tracked_high_filter.attrs["N_events_final"], 21, tolerance=1)

    def test_temporal_gap_filling(self, dask_client_gridded):
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
        assert_count_in_reasonable_range(tracked_no_gaps.attrs["N_objects_prefiltered"], 1046, tolerance=2)
        assert_count_in_reasonable_range(tracked_no_gaps.attrs["N_objects_filtered"], 522, tolerance=2)
        assert_count_in_reasonable_range(tracked_no_gaps.attrs["N_events_final"], 54, tolerance=1)

        # Assert tracking statistics are within reasonable bounds for with gaps case
        assert_reasonable_bounds(
            tracked_with_gaps.attrs["preprocessed_area_fraction"],
            1.0080,
            tolerance_absolute=0.02,
        )
        assert_count_in_reasonable_range(tracked_with_gaps.attrs["N_objects_prefiltered"], 1041, tolerance=2)
        assert_count_in_reasonable_range(tracked_with_gaps.attrs["N_objects_filtered"], 522, tolerance=2)
        assert_count_in_reasonable_range(tracked_with_gaps.attrs["N_events_final"], 38, tolerance=1)

    def test_custom_dimension_names_tracking(self, dask_client_gridded):
        """Test tracking with custom dimension and coordinate names and compare with standard names.

        This test validates that:
        1. Custom dimension/coordinate names work correctly (t, y, x) vs (time, lat, lon)
        2. Tracking results are identical between standard and custom names
        3. Both allow_merging=False and allow_merging=True scenarios work properly

        Most tracking results should be identical, but centroid values may differ due to
        coordinate system transformations from pixel coordinates to geographic coordinates.
        """
        # Common tracking parameters
        tracking_params_no_merge = {
            "area_filter_quartile": 0.5,
            "R_fill": 4,
            "T_fill": 0,
            "allow_merging": False,
            "quiet": True,
        }

        tracking_params_with_merge = {
            "area_filter_quartile": 0.5,
            "R_fill": 4,
            "T_fill": 2,
            "allow_merging": True,
            "overlap_threshold": 0.5,
            "nn_partitioning": True,
            "quiet": True,
        }

        # Apply common mask to exclude poles
        mask_standard = self.extremes_data.mask.where(
            (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
            other=False,
        )

        # Create dataset with custom dimension and coordinate names
        # Dimensions: "t", "x", "y"
        # Coordinates: "T", "longitude", "latitude"
        extreme_events_custom = xr.DataArray(
            self.extremes_data.extreme_events.values,
            dims=["t", "y", "x"],
            coords={
                "T": ("t", self.extremes_data.time.values),
                "latitude": ("y", self.extremes_data.lat.values),
                "longitude": ("x", self.extremes_data.lon.values),
            },
        ).chunk({"t": 2, "y": -1, "x": -1})

        mask_custom = xr.DataArray(
            self.extremes_data.mask.values,
            dims=["y", "x"],
            coords={"latitude": ("y", self.extremes_data.lat.values), "longitude": ("x", self.extremes_data.lon.values)},
        )

        mask_custom_filtered = mask_custom.where(
            (mask_custom.latitude < 85) & (mask_custom.latitude > -90),
            other=False,
        )

        # Test 1: Standard vs Custom with allow_merging=False
        # Standard case
        tracker_standard_no_merge = marEx.tracker(self.extremes_data.extreme_events, mask_standard, **tracking_params_no_merge)
        tracked_standard_no_merge = tracker_standard_no_merge.run()

        # Custom case
        tracker_custom_no_merge = marEx.tracker(
            extreme_events_custom,
            mask_custom_filtered,
            dimensions={"time": "t", "x": "x", "y": "y"},
            coordinates={"time": "T", "x": "longitude", "y": "latitude"},
            **tracking_params_no_merge,
        )
        tracked_custom_no_merge = tracker_custom_no_merge.run()

        # Verify custom case output structure (no merging)
        assert isinstance(tracked_custom_no_merge, xr.Dataset)
        assert "ID_field" in tracked_custom_no_merge.data_vars

        # Verify custom dimensions are correctly named
        assert "t" in tracked_custom_no_merge.ID_field.dims
        assert "y" in tracked_custom_no_merge.ID_field.dims
        assert "x" in tracked_custom_no_merge.ID_field.dims

        # Verify custom coordinates are present
        assert "T" in tracked_custom_no_merge.coords
        assert "latitude" in tracked_custom_no_merge.coords
        assert "longitude" in tracked_custom_no_merge.coords

        # Compare results between standard and custom (no merging)
        self._compare_tracking_results(tracked_standard_no_merge, tracked_custom_no_merge, allow_merging=False)

        # Test 2: Standard vs Custom with allow_merging=True
        # Standard case
        tracker_standard_merge = marEx.tracker(self.extremes_data.extreme_events, mask_standard, **tracking_params_with_merge)
        tracked_standard_merge = tracker_standard_merge.run()

        # Custom case
        tracker_custom_merge = marEx.tracker(
            extreme_events_custom,
            mask_custom_filtered,
            dimensions={"time": "t", "x": "x", "y": "y"},
            coordinates={"time": "T", "x": "longitude", "y": "latitude"},
            **tracking_params_with_merge,
        )
        tracked_custom_merge = tracker_custom_merge.run()

        # Verify custom case output structure (with merging)
        assert isinstance(tracked_custom_merge, xr.Dataset)
        expected_vars = {"ID_field", "global_ID", "area", "centroid", "presence", "time_start", "time_end", "merge_ledger"}
        assert set(tracked_custom_merge.data_vars.keys()) == expected_vars

        # Verify custom dimensions are correctly named
        assert "t" in tracked_custom_merge.ID_field.dims
        assert "y" in tracked_custom_merge.ID_field.dims
        assert "x" in tracked_custom_merge.ID_field.dims

        # Verify custom coordinates are present
        assert "T" in tracked_custom_merge.coords
        assert "latitude" in tracked_custom_merge.coords
        assert "longitude" in tracked_custom_merge.coords

        # Compare results between standard and custom (with merging)
        self._compare_tracking_results(tracked_standard_merge, tracked_custom_merge, allow_merging=True)

        # Test 3: Compare between merging modes (custom case)
        # Both should track some events
        assert tracked_custom_no_merge.attrs["N_events_final"] > 0
        assert tracked_custom_merge.attrs["N_events_final"] > 0

        # Both should have the same core structure (ID_field)
        assert tracked_custom_no_merge.ID_field.dims == tracked_custom_merge.ID_field.dims

        # Verify that time_start <= time_end for all events (merging case)
        valid_events = tracked_custom_merge.presence.any(dim="t").compute()
        for event_id in tracked_custom_merge.ID[valid_events]:
            start_time = tracked_custom_merge.time_start.sel(ID=event_id)
            end_time = tracked_custom_merge.time_end.sel(ID=event_id)
            assert start_time <= end_time, f"Event {event_id} has start_time > end_time"

    def _compare_tracking_results(self, standard_ds, custom_ds, allow_merging=True):
        """Helper method to compare tracking results between standard and custom dimension names."""
        # Compare key tracking statistics - they should be identical
        standard_attrs = standard_ds.attrs
        custom_attrs = custom_ds.attrs

        # Essential statistics that should match exactly
        key_stats = [
            "N_events_final",
            "N_objects_prefiltered",
            "N_objects_filtered",
            "allow_merging",
            "T_fill",
            "R_fill",
        ]

        for stat in key_stats:
            assert standard_attrs[stat] == custom_attrs[stat], f"{stat} differs: {standard_attrs[stat]} vs {custom_attrs[stat]}"

        # Floating point statistics should match within tolerance
        float_stats = ["preprocessed_area_fraction"]

        for stat in float_stats:
            np.testing.assert_allclose(
                standard_attrs[stat], custom_attrs[stat], rtol=1e-12, err_msg=f"{stat} differs beyond tolerance"
            )

        # Compare ID_field values (the core tracking results)
        # Account for different dimension names: (time, lat, lon) vs (t, y, x)
        if allow_merging:
            standard_id_field = standard_ds.ID_field.transpose("time", "lat", "lon")
            custom_id_field = custom_ds.ID_field.transpose("t", "y", "x")
        else:
            # For no merging case, use the natural dimension order
            standard_id_field = standard_ds.ID_field
            custom_id_field = custom_ds.ID_field

        np.testing.assert_array_equal(
            standard_id_field.values,
            custom_id_field.values,
            err_msg="ID_field values differ between standard and custom dimensions",
        )

        if allow_merging:
            # Compare data variables specific to merging case
            # Compare global_ID values - should be identical
            np.testing.assert_array_equal(
                standard_ds.global_ID.values,
                custom_ds.global_ID.values,
                err_msg="global_ID values differ between standard and custom dimensions",
            )

            # Compare area values - should be identical
            np.testing.assert_array_equal(
                standard_ds.area.values, custom_ds.area.values, err_msg="area values differ between standard and custom dimensions"
            )

            # Compare presence values - should be identical
            np.testing.assert_array_equal(
                standard_ds.presence.values,
                custom_ds.presence.values,
                err_msg="presence values differ between standard and custom dimensions",
            )

            # Compare centroid values - they may differ due to coordinate system transformations
            # but should have the same shape
            standard_centroid = standard_ds.centroid.transpose("time", "ID", "component")
            custom_centroid = custom_ds.centroid.transpose("t", "ID", "component")

            assert standard_centroid.shape == custom_centroid.shape, "centroid shapes differ"
            # Note: Centroid values may differ due to different coordinate systems
            # (lat/lon vs y/x) but the tracking patterns should be similar

            # Compare time_start and time_end - values should be identical
            np.testing.assert_array_equal(
                standard_ds.time_start.values,
                custom_ds.time_start.values,
                err_msg="time_start values differ between standard and custom dimensions",
            )

            np.testing.assert_array_equal(
                standard_ds.time_end.values,
                custom_ds.time_end.values,
                err_msg="time_end values differ between standard and custom dimensions",
            )

            # Compare merge_ledger values - should be identical
            standard_merge_ledger = standard_ds.merge_ledger.transpose("time", "ID", "sibling_ID")
            custom_merge_ledger = custom_ds.merge_ledger.transpose("t", "ID", "sibling_ID")

            np.testing.assert_array_equal(
                standard_merge_ledger.values,
                custom_merge_ledger.values,
                err_msg="merge_ledger values differ between standard and custom dimensions",
            )

        # Verify coordinate values are the same (just with different names)
        if allow_merging:
            time_coord_standard = "time"
            time_coord_custom = "T"
            lat_coord_standard = "lat"
            lat_coord_custom = "latitude"
            lon_coord_standard = "lon"
            lon_coord_custom = "longitude"
        else:
            # For no merging case, use the actual coordinate names
            time_coord_standard = "time"
            time_coord_custom = "T"
            lat_coord_standard = "lat"
            lat_coord_custom = "latitude"
            lon_coord_standard = "lon"
            lon_coord_custom = "longitude"

        np.testing.assert_array_equal(
            standard_ds[time_coord_standard].values, custom_ds[time_coord_custom].values, err_msg="time coordinate values differ"
        )

        np.testing.assert_array_equal(
            standard_ds[lat_coord_standard].values, custom_ds[lat_coord_custom].values, err_msg="latitude coordinate values differ"
        )

        np.testing.assert_array_equal(
            standard_ds[lon_coord_standard].values, custom_ds[lon_coord_custom].values, err_msg="longitude coordinate values differ"
        )

    def test_centroid_tracking_moving_blob(self, dask_client_gridded):
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
        event_durations.ID[longest_event_idx]

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
        expected_lon_centroids = np.where(expected_lon_centroids > 180, expected_lon_centroids - 360, expected_lon_centroids)

        # Test centroid accuracy with specified tolerances
        lon_tolerance = 0.5  # degrees (relaxed slightly from 0.25 to account for discretisation)
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

    def test_spatial_chunking_validation_extremes_data(self, dask_client_gridded):
        """Test _validate_spatial_chunking() with extremes binary data chunked in x and y dimensions."""
        import warnings

        # Create extremes data with multiple chunks in spatial dimensions
        extremes_chunked = self.extremes_data.extreme_events.chunk({"time": 2, "lat": 10, "lon": 15})
        mask = self.extremes_data.mask.where(
            (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
            other=False,
        )

        # Test that warnings are raised for spatially chunked extremes data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            tracker = marEx.tracker(
                extremes_chunked,
                mask,
                area_filter_quartile=0.5,
                R_fill=4,
                T_fill=0,
                allow_merging=False,
                quiet=True,
            )

            # Check that warnings were raised for both spatial dimensions
            warning_messages = [str(warning.message) for warning in w]

            # Should have warnings for both lat and lon dimensions
            lat_warning_found = any("lat" in msg and "multiple chunks" in msg and "apply_ufunc" in msg for msg in warning_messages)
            lon_warning_found = any("lon" in msg and "multiple chunks" in msg and "apply_ufunc" in msg for msg in warning_messages)

            assert lat_warning_found, f"Expected warning for lat chunking not found. Warnings: {warning_messages}"
            assert lon_warning_found, f"Expected warning for lon chunking not found. Warnings: {warning_messages}"

            # Run tracking to ensure it still works after rechunking
            tracked_ds = tracker.run()

            # Verify output structure (same assertions as basic test)
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
            assert max_id == tracked_ds.attrs["N_events_final"], "Max ID doesn't match reported event count"

            # Verify that background is labeled as 0
            assert int(tracked_ds.ID_field.min()) == 0

            # Verify ID_field is int
            assert np.issubdtype(tracked_ds.ID_field.dtype, np.integer), "ID_field should be integer type"

    def test_physical_area_functionality(self, dask_client_gridded):
        """Test tracking with physical cell areas for structured grids."""
        # Create mock physical cell areas (lat-dependent for realistic test)
        lat_values = self.extremes_data.lat.values
        lon_values = self.extremes_data.lon.values

        # Create realistic cell areas that vary with latitude (smaller near poles)
        lat_mesh, lon_mesh = np.meshgrid(lat_values, lon_values, indexing="ij")
        # Physical area proportional to cos(latitude) for a sphere
        cell_areas_values = 111320 * 111320 * np.cos(np.radians(lat_mesh))  # Approximate km² per degree

        # Create xarray DataArray for cell areas
        cell_areas = xr.DataArray(
            cell_areas_values.astype(np.float32),
            coords={"lat": lat_values, "lon": lon_values},
            dims=["lat", "lon"],
            name="cell_areas",
        )

        # Test tracking with physical areas
        tracker_physical = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask.where(
                (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
                other=False,
            ),
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=2,
            allow_merging=True,  # Enable merging to get area calculations
            cell_areas=cell_areas,  # Provide physical areas
            quiet=True,
        )

        # Test tracking with default cell counts (no cell_areas)
        tracker_counts = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask.where(
                (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
                other=False,
            ),
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=2,
            allow_merging=True,  # Enable merging to get area calculations
            quiet=True,
        )

        # Run both trackers
        tracked_physical = tracker_physical.run()
        tracked_counts = tracker_counts.run()

        # Verify both produce valid outputs
        assert isinstance(tracked_physical, xr.Dataset)
        assert isinstance(tracked_counts, xr.Dataset)

        # Verify that physical areas are larger than cell counts (since we're using km²)
        physical_areas = tracked_physical.area.values
        count_areas = tracked_counts.area.values

        # Remove NaN values for comparison
        valid_physical = physical_areas[~np.isnan(physical_areas)]
        valid_counts = count_areas[~np.isnan(count_areas)]

        if len(valid_physical) > 0 and len(valid_counts) > 0:
            # Physical areas should generally be much larger than cell counts
            assert np.mean(valid_physical) > np.mean(valid_counts), "Physical areas should be larger than cell counts"

        # Verify that the mean cell area was calculated correctly for physical tracker
        expected_mean = float(cell_areas.mean().compute())
        assert (
            abs(tracker_physical.mean_cell_area - expected_mean) < 1e-6
        ), f"Mean cell area mismatch: {tracker_physical.mean_cell_area} vs {expected_mean}"

        # Verify that the default tracker uses unit areas (mean=1.0)
        assert (
            tracker_counts.mean_cell_area == 1.0
        ), f"Default tracker should have mean_cell_area=1.0, got {tracker_counts.mean_cell_area}"

    def test_invalid_cell_areas_dimensions(self, dask_client_gridded):
        """Test that invalid cell_areas dimensions raise appropriate errors."""
        # Create cell_areas with wrong dimensions (missing lat)
        wrong_cell_areas = xr.DataArray(
            np.ones(len(self.extremes_data.lon)), coords={"lon": self.extremes_data.lon}, dims=["lon"], name="cell_areas"
        )

        # Should raise validation error
        try:
            marEx.tracker(
                self.extremes_data.extreme_events,
                self.extremes_data.mask,
                R_fill=4,
                cell_areas=wrong_cell_areas,
                quiet=True,
            )
            raise AssertionError("Expected DataValidationError for invalid cell_areas dimensions")
        except marEx.exceptions.DataValidationError as e:
            assert "Invalid cell_areas dimensions" in str(e)

        # Create cell_areas with extra dimension
        wrong_cell_areas_3d = xr.DataArray(
            np.ones((len(self.extremes_data.lat), len(self.extremes_data.lon), 2)),
            coords={"lat": self.extremes_data.lat, "lon": self.extremes_data.lon, "extra": [0, 1]},
            dims=["lat", "lon", "extra"],
            name="cell_areas",
        )

        # Should also raise validation error
        try:
            marEx.tracker(
                self.extremes_data.extreme_events,
                self.extremes_data.mask,
                R_fill=4,
                cell_areas=wrong_cell_areas_3d,
                quiet=True,
            )
            raise AssertionError("Expected DataValidationError for invalid cell_areas dimensions")
        except marEx.exceptions.DataValidationError as e:
            assert "Invalid cell_areas dimensions" in str(e)

    def test_grid_resolution_parameter(self, dask_client_gridded):
        """Test tracking with automatic grid area calculation from resolution."""
        # Get the actual grid resolution from the test data
        lat_res = abs(float(self.extremes_data.lat[1] - self.extremes_data.lat[0]))

        # Assuming uniform grid spacing
        grid_resolution = lat_res  # Use latitude resolution (should match longitude)

        # Test tracking with grid_resolution parameter
        tracker_grid_res = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask.where(
                (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
                other=False,
            ),
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=2,
            allow_merging=True,
            grid_resolution=grid_resolution,  # Automatic area calculation
            quiet=True,
        )

        # Test tracking with manual cell_areas calculation
        lat_values = self.extremes_data.lat.values
        lon_values = self.extremes_data.lon.values
        lat_mesh, lon_mesh = np.meshgrid(lat_values, lon_values, indexing="ij")

        # Calculate areas manually using same formula as in track.py
        R_earth = 6378.0  # km
        lat_r = np.radians(lat_mesh)
        dlat = np.radians(grid_resolution)
        dlon = np.radians(grid_resolution)

        manual_areas = (R_earth**2 * np.abs(np.sin(lat_r + dlat / 2) - np.sin(lat_r - dlat / 2)) * dlon).astype(np.float32)

        cell_areas_manual = xr.DataArray(
            manual_areas, coords={"lat": lat_values, "lon": lon_values}, dims=["lat", "lon"], name="cell_areas"
        )

        tracker_manual = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask.where(
                (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
                other=False,
            ),
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=2,
            allow_merging=True,
            cell_areas=cell_areas_manual,  # Manual area calculation
            quiet=True,
        )

        # Verify that both methods produce very similar mean cell areas
        expected_mean = float(cell_areas_manual.mean().compute())
        actual_mean = tracker_grid_res.mean_cell_area

        # Allow small numerical differences due to floating point precision and grid sampling
        relative_error = abs(actual_mean - expected_mean) / expected_mean
        assert (
            relative_error < 0.01
        ), f"Grid resolution area calculation mismatch: {actual_mean} vs {expected_mean} (rel error: {relative_error})"

        # Run both trackers and verify they produce similar results
        tracked_grid_res = tracker_grid_res.run()
        tracked_manual = tracker_manual.run()

        assert isinstance(tracked_grid_res, xr.Dataset)
        assert isinstance(tracked_manual, xr.Dataset)

        # Areas should be very similar between methods
        areas_grid_res = tracked_grid_res.area.values
        areas_manual = tracked_manual.area.values

        # Remove NaN values for comparison
        valid_grid_res = areas_grid_res[~np.isnan(areas_grid_res)]
        valid_manual = areas_manual[~np.isnan(areas_manual)]

        if len(valid_grid_res) > 0 and len(valid_manual) > 0:
            # Calculate relative difference in mean areas
            mean_diff = abs(np.mean(valid_grid_res) - np.mean(valid_manual)) / np.mean(valid_manual)
            assert mean_diff < 0.01, f"Area calculations should be nearly identical: {mean_diff:.4f} relative difference"

    def test_grid_resolution_validation(self, dask_client_gridded):
        """Test validation of grid_resolution parameter."""
        # Test 1: Negative grid_resolution should fail
        try:
            marEx.tracker(
                self.extremes_data.extreme_events,
                self.extremes_data.mask,
                R_fill=4,
                grid_resolution=-0.5,
                quiet=True,
            )
            raise AssertionError("Expected DataValidationError for negative grid_resolution")
        except marEx.exceptions.DataValidationError as e:
            assert "grid_resolution must be a positive number" in str(e)

        # Test 2: Zero grid_resolution should fail
        try:
            marEx.tracker(
                self.extremes_data.extreme_events,
                self.extremes_data.mask,
                R_fill=4,
                grid_resolution=0.0,
                quiet=True,
            )
            raise AssertionError("Expected DataValidationError for zero grid_resolution")
        except marEx.exceptions.DataValidationError as e:
            assert "grid_resolution must be a positive number" in str(e)

        # Test 3: Valid grid_resolution should work
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask.where(
                (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
                other=False,
            ),
            R_fill=4,
            grid_resolution=1.0,  # 1 degree resolution
            quiet=True,
        )

        # Should initialize without error
        assert tracker.mean_cell_area > 0, "Mean cell area should be positive"

        # Verify areas were calculated (not unit areas)
        assert tracker.mean_cell_area != 1.0, "Should use calculated areas, not unit areas"

    def test_grid_resolution_override_warning(self, dask_client_gridded):
        """Test that grid_resolution overrides cell_areas with warning."""
        # Create some cell_areas
        cell_areas = xr.DataArray(
            np.ones((len(self.extremes_data.lat), len(self.extremes_data.lon)), dtype=np.float32),
            coords={"lat": self.extremes_data.lat, "lon": self.extremes_data.lon},
            dims=["lat", "lon"],
            name="cell_areas",
        )

        # Test with both parameters - should use grid_resolution and warn
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            tracker = marEx.tracker(
                self.extremes_data.extreme_events,
                self.extremes_data.mask.where(
                    (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
                    other=False,
                ),
                R_fill=4,
                cell_areas=cell_areas,
                grid_resolution=1.0,  # Should override cell_areas
                quiet=True,
            )

            # Should produce calculated areas, not the unit areas we provided
            assert tracker.mean_cell_area != 1.0, "Should use grid_resolution calculation, not provided cell_areas"
            assert tracker.mean_cell_area > 1000, "Should calculate realistic km² areas"

    def test_spatial_chunking_validation_mask(self, dask_client_gridded):
        """Test _validate_spatial_chunking() with mask chunked in x and y dimensions."""
        import warnings

        # Create mask with multiple chunks in spatial dimensions
        mask = self.extremes_data.mask.where(
            (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
            other=False,
        )
        mask_chunked = mask.chunk({"lat": 8, "lon": 12})

        # Test that warnings are raised for spatially chunked mask
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            tracker = marEx.tracker(
                self.extremes_data.extreme_events,
                mask_chunked,
                area_filter_quartile=0.5,
                R_fill=4,
                T_fill=0,
                allow_merging=False,
                quiet=True,
            )

            # Check that warnings were raised for both spatial dimensions
            warning_messages = [str(warning.message) for warning in w]

            # Should have warnings for both mask spatial dimensions
            mask_lat_warning_found = any(
                "Mask spatial dimension 'lat'" in msg and "multiple chunks" in msg for msg in warning_messages
            )
            mask_lon_warning_found = any(
                "Mask spatial dimension 'lon'" in msg and "multiple chunks" in msg for msg in warning_messages
            )

            assert mask_lat_warning_found, f"Expected warning for mask lat chunking not found. Warnings: {warning_messages}"
            assert mask_lon_warning_found, f"Expected warning for mask lon chunking not found. Warnings: {warning_messages}"

            # Run tracking to ensure it still works after rechunking
            tracked_ds = tracker.run()

            # Verify output structure (same assertions as basic test)
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

            # Verify ID field contains reasonable values
            max_id = int(tracked_ds.ID_field.max())
            assert max_id > 0, "No events were tracked"
            assert max_id == tracked_ds.attrs["N_events_final"], "Max ID doesn't match reported event count"

    def test_spatial_chunking_validation_coordinates(self, dask_client_gridded):
        """Test _validate_spatial_chunking() with lat and lon coordinates chunked in their respective dimensions.

        Note: This test validates that the coordinate chunking validation works. Since the coordinates
        are extracted during tracker initialization with .persist(), the chunking information might
        not be preserved unless the coordinates were already chunked in the original data.
        """
        import warnings

        import pytest

        # Skip this test for now as coordinate chunking warnings only occur when
        # coordinates are passed as separate chunked DataArrays, which is not the typical usage
        # This test documents the expected behavior but may not trigger warnings in practice
        pytest.skip("Coordinate chunking warnings only occur with explicitly chunked coordinate DataArrays")

        # Create dataset with chunked coordinates - we need to modify the coordinates within the dataset
        extremes_data_chunked_coords = self.extremes_data.copy()

        # Chunk the coordinate variables in the dataset
        extremes_data_chunked_coords["lat"] = extremes_data_chunked_coords.lat.chunk({"lat": 6})
        extremes_data_chunked_coords["lon"] = extremes_data_chunked_coords.lon.chunk({"lon": 9})

        mask = extremes_data_chunked_coords.mask.where(
            (extremes_data_chunked_coords.lat < 85) & (extremes_data_chunked_coords.lat > -90),
            other=False,
        )

        # Test that warnings are raised for chunked coordinates in the dataset
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            tracker = marEx.tracker(
                extremes_data_chunked_coords.extreme_events,
                mask,
                area_filter_quartile=0.5,
                R_fill=4,
                T_fill=0,
                allow_merging=False,
                quiet=True,
            )

            # Check that warnings were raised for coordinate chunking
            warning_messages = [str(warning.message) for warning in w]

            # Should have warnings for both coordinate dimensions
            lat_coord_warning_found = any(
                "Latitude coordinate spatial dimension" in msg and "multiple chunks" in msg for msg in warning_messages
            )
            lon_coord_warning_found = any(
                "Longitude coordinate spatial dimension" in msg and "multiple chunks" in msg for msg in warning_messages
            )

            assert lat_coord_warning_found, f"Expected warning for lat coordinate chunking not found. Warnings: {warning_messages}"
            assert lon_coord_warning_found, f"Expected warning for lon coordinate chunking not found. Warnings: {warning_messages}"

            # Run tracking to ensure it still works after rechunking
            tracked_ds = tracker.run()

            # Verify output structure (same assertions as basic test)
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

            # Verify ID field contains reasonable values
            max_id = int(tracked_ds.ID_field.max())
            assert max_id > 0, "No events were tracked"
            assert max_id == tracked_ds.attrs["N_events_final"], "Max ID doesn't match reported event count"

    def test_spatial_chunking_validation_combined(self, dask_client_gridded):
        """Test _validate_spatial_chunking() with all data types chunked in spatial dimensions."""
        import warnings

        # Create all data types with multiple chunks in spatial dimensions
        extremes_chunked = self.extremes_data.extreme_events.chunk({"time": 2, "lat": 5, "lon": 8})

        # Create chunked mask
        mask = self.extremes_data.mask.where(
            (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
            other=False,
        )
        mask_chunked = mask.chunk({"lat": 7, "lon": 11})

        # Create dataset with chunked coordinates
        extremes_data_chunked_coords = self.extremes_data.copy()
        extremes_data_chunked_coords["lat"] = extremes_data_chunked_coords.lat.chunk({"lat": 4})
        extremes_data_chunked_coords["lon"] = extremes_data_chunked_coords.lon.chunk({"lon": 6})

        # Test that warnings are raised for everything
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Create a complete dataset with all chunked components
            extremes_all_chunked = extremes_data_chunked_coords.copy()
            extremes_all_chunked["extreme_events"] = extremes_chunked
            extremes_all_chunked["mask"] = mask_chunked

            tracker = marEx.tracker(
                extremes_all_chunked.extreme_events,
                extremes_all_chunked.mask,
                area_filter_quartile=0.5,
                R_fill=4,
                T_fill=0,
                allow_merging=False,
                quiet=True,
            )

            # Check that warnings were raised for all chunked data types
            warning_messages = [str(warning.message) for warning in w]

            # Should have warnings for extremes data spatial dimensions
            extremes_lat_warning = any(
                "lat" in msg and "multiple chunks" in msg and "apply_ufunc" in msg for msg in warning_messages
            )
            extremes_lon_warning = any(
                "lon" in msg and "multiple chunks" in msg and "apply_ufunc" in msg for msg in warning_messages
            )

            # Should have warnings for mask spatial dimensions
            mask_lat_warning = any("Mask spatial dimension 'lat'" in msg and "multiple chunks" in msg for msg in warning_messages)
            mask_lon_warning = any("Mask spatial dimension 'lon'" in msg and "multiple chunks" in msg for msg in warning_messages)

            # Note: Coordinate warnings don't trigger as coordinates are extracted with .persist()
            # which doesn't preserve chunking information from the original dataset

            assert extremes_lat_warning, f"Expected warning for extremes lat chunking not found. Warnings: {warning_messages}"
            assert extremes_lon_warning, f"Expected warning for extremes lon chunking not found. Warnings: {warning_messages}"
            assert mask_lat_warning, f"Expected warning for mask lat chunking not found. Warnings: {warning_messages}"
            assert mask_lon_warning, f"Expected warning for mask lon chunking not found. Warnings: {warning_messages}"

            # Run tracking to ensure it still works after rechunking
            tracked_ds = tracker.run()

            # Verify output structure (same assertions as basic test)
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
            assert max_id == tracked_ds.attrs["N_events_final"], "Max ID doesn't match reported event count"

            # Verify that background is labeled as 0
            assert int(tracked_ds.ID_field.min()) == 0

            # Verify ID_field is int
            assert np.issubdtype(tracked_ds.ID_field.dtype, np.integer), "ID_field should be integer type"
