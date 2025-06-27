import pytest
import xarray as xr
import numpy as np
from pathlib import Path
import marEx


class TestGriddedTracking:
    """Test event tracking functionality for gridded data."""
    
    @classmethod
    def setup_class(cls):
        """Load test data for all tests."""
        test_data_path = Path(__file__).parent / "data" / "extremes_gridded.zarr"
        cls.extremes_data = xr.open_zarr(str(test_data_path), chunks={}).persist()
        
        # Standard chunk size for tracking (spatial dimensions must be contiguous)
        cls.chunk_size = {'time': 2, 'lat': -1, 'lon': -1}
    
    def test_basic_tracking(self, dask_client):
        """Test basic tracking without merging/splitting."""
        # Create tracker with basic settings
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask.where(
                (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90), 
                other=False
            ),  # Exclude poles
            area_filter_quartile=0.5,
            R_fill=4,  # Reduced for test data
            T_fill=0,  # No temporal filling for basic test
            allow_merging=False,
            verbosity=0  # Suppress output for tests
        )
        
        # Run tracking
        tracked_ds = tracker.run()
        
        # Verify output structure
        assert isinstance(tracked_ds, xr.Dataset)
        assert 'ID_field' in tracked_ds.data_vars
        
        # Verify dimensions 
        assert 'time' in tracked_ds.ID_field.dims
        assert 'lat' in tracked_ds.ID_field.dims
        assert 'lon' in tracked_ds.ID_field.dims
        
        # Verify attributes are set
        assert 'N_events_final' in tracked_ds.attrs
        assert 'allow_merging' in tracked_ds.attrs
        assert tracked_ds.attrs['allow_merging'] == 0
        assert 'R_fill' in tracked_ds.attrs
        assert 'T_fill' in tracked_ds.attrs
        
        # Verify ID field contains reasonable values
        max_id = int(tracked_ds.ID_field.max())
        assert max_id > 0, "No events were tracked"
        assert max_id == tracked_ds.attrs['N_events_final'], "Max ID doesn't match reported event count"
        
        # Verify that background is labeled as 0
        assert int(tracked_ds.ID_field.min()) == 0
        
        # Assert tracking statistics bounds
        assert abs(tracked_ds.attrs['preprocessed_area_fraction'] - 0.9724) < 0.02
        assert abs(tracked_ds.attrs['N_objects_prefiltered'] - 549) < 2
        assert abs(tracked_ds.attrs['N_objects_filtered'] - 274) < 2
        assert abs(tracked_ds.attrs['N_events_final'] - 24) < 1
    
    def test_advanced_tracking_with_merging(self, dask_client):
        """Test advanced tracking with temporal filling and merging enabled."""
        # Create tracker with advanced settings
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask.where(
                (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90), 
                other=False
            ),
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=2,  # Allow 2-day gaps
            allow_merging=True,
            overlap_threshold=0.5,
            nn_partitioning=True,
            verbosity=0
        )
        
        # Run tracking with merge information
        tracked_ds, merges_ds = tracker.run(return_merges=True)
        
        # Verify main output structure
        assert isinstance(tracked_ds, xr.Dataset)
        assert 'ID_field' in tracked_ds.data_vars
        assert 'global_ID' in tracked_ds.data_vars
        assert 'area' in tracked_ds.data_vars
        assert 'centroid' in tracked_ds.data_vars
        assert 'presence' in tracked_ds.data_vars
        assert 'time_start' in tracked_ds.data_vars
        assert 'time_end' in tracked_ds.data_vars
        assert 'merge_ledger' in tracked_ds.data_vars
        
        # Verify merge dataset structure
        assert isinstance(merges_ds, xr.Dataset)
        assert 'parent_IDs' in merges_ds.data_vars
        assert 'child_IDs' in merges_ds.data_vars
        assert 'overlap_areas' in merges_ds.data_vars
        assert 'merge_time' in merges_ds.data_vars
        assert 'n_parents' in merges_ds.data_vars
        assert 'n_children' in merges_ds.data_vars
        
        # Verify advanced tracking attributes
        assert tracked_ds.attrs['allow_merging'] == 1
        assert tracked_ds.attrs['T_fill'] == 2
        assert 'total_merges' in tracked_ds.attrs
        
        # Verify ID dimension consistency
        n_events = tracked_ds.sizes['ID']
        assert n_events == tracked_ds.attrs['N_events_final']
        
        # Verify that time_start <= time_end for all events
        valid_events = tracked_ds.presence.any(dim='time').compute()  # Compute the boolean mask
        for event_id in tracked_ds.ID[valid_events]:
            start_time = tracked_ds.time_start.sel(ID=event_id)
            end_time = tracked_ds.time_end.sel(ID=event_id)
            assert start_time <= end_time, f"Event {event_id} has start_time > end_time"
        
        # Assert tracking statistics bounds
        assert abs(tracked_ds.attrs['preprocessed_area_fraction'] - 0.9143) < 0.02
        assert abs(tracked_ds.attrs['N_objects_prefiltered'] - 516) < 2
        assert abs(tracked_ds.attrs['N_objects_filtered'] - 258) < 2
        assert abs(tracked_ds.attrs['N_events_final'] - 20) < 1
        assert abs(tracked_ds.attrs['total_merges'] - 26) < 2
    
    def test_tracking_data_consistency(self, dask_client):
        """Test that tracking produces consistent data structures."""
        tracker = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask.where(
                (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90), 
                other=False
            ),
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=2,
            allow_merging=True,
            verbosity=0
        )
        
        tracked_ds = tracker.run()
        
        # Test that presence matches where global_ID is non-zero
        presence_from_global_id = (tracked_ds.global_ID != 0)
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
        
        # Assert tracking statistics bounds
        assert abs(tracked_ds.attrs['preprocessed_area_fraction'] - 0.9143) < 0.02
        assert abs(tracked_ds.attrs['N_objects_prefiltered'] - 516) < 2
        assert abs(tracked_ds.attrs['N_objects_filtered'] - 258) < 2
        assert abs(tracked_ds.attrs['N_events_final'] - 19) < 1
        assert abs(tracked_ds.attrs['total_merges'] - 27) < 2
    
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
            verbosity=0
        )
        
        # Test with aggressive filtering (quartile = 0.8)  
        tracker_high_filter = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            area_filter_quartile=0.8,
            R_fill=2,
            T_fill=0,
            allow_merging=False,
            verbosity=0
        )
        
        tracked_no_filter = tracker_no_filter.run()
        tracked_high_filter = tracker_high_filter.run()
        
        # Higher filtering should result in fewer events
        n_events_no_filter = tracked_no_filter.attrs['N_events_final']
        n_events_high_filter = tracked_high_filter.attrs['N_events_final']
        
        assert n_events_high_filter <= n_events_no_filter, "High filtering should produce fewer or equal events"
        
        # Both should have valid ID fields
        assert int(tracked_no_filter.ID_field.max()) > 0
        assert int(tracked_high_filter.ID_field.max()) >= 0  # Could be 0 if all events filtered out
        
        # Assert tracking statistics bounds for no filter case
        assert abs(tracked_no_filter.attrs['preprocessed_area_fraction'] - 1.0622) < 0.02
        assert abs(tracked_no_filter.attrs['N_objects_prefiltered'] - 1046) < 2
        assert abs(tracked_no_filter.attrs['N_objects_filtered'] - 1045) < 2
        assert abs(tracked_no_filter.attrs['N_events_final'] - 152) < 1
        
        # Assert tracking statistics bounds for high filter case
        assert abs(tracked_high_filter.attrs['preprocessed_area_fraction'] - 1.5423) < 0.02
        assert abs(tracked_high_filter.attrs['N_objects_prefiltered'] - 1046) < 2
        assert abs(tracked_high_filter.attrs['N_objects_filtered'] - 209) < 2
        assert abs(tracked_high_filter.attrs['N_events_final'] - 21) < 1
    
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
            verbosity=0
        )
        
        # Test with gap filling
        tracker_with_gaps = marEx.tracker(
            self.extremes_data.extreme_events,
            self.extremes_data.mask,
            area_filter_quartile=0.5,
            R_fill=2,
            T_fill=4,  # Allow 4-day gaps
            allow_merging=False,
            verbosity=0
        )
        
        tracked_no_gaps = tracker_no_gaps.run()
        tracked_with_gaps = tracker_with_gaps.run()
        
        # Gap filling should typically result in fewer total events (some are merged)
        # but longer individual events
        n_events_no_gaps = tracked_no_gaps.attrs['N_events_final']
        n_events_with_gaps = tracked_with_gaps.attrs['N_events_final']
        
        # Both should produce valid results
        assert n_events_no_gaps > 0, "No gap filling should produce some events"
        assert n_events_with_gaps > 0, "Gap filling should produce some events"
        
        # Verify T_fill attribute is correctly set
        assert tracked_no_gaps.attrs['T_fill'] == 0
        assert tracked_with_gaps.attrs['T_fill'] == 4
        
        # Assert tracking statistics bounds for no gaps case
        assert abs(tracked_no_gaps.attrs['preprocessed_area_fraction'] - 1.1650) < 0.02
        assert abs(tracked_no_gaps.attrs['N_objects_prefiltered'] - 1046) < 2
        assert abs(tracked_no_gaps.attrs['N_objects_filtered'] - 522) < 2
        assert abs(tracked_no_gaps.attrs['N_events_final'] - 54) < 1
        
        # Assert tracking statistics bounds for with gaps case
        assert abs(tracked_with_gaps.attrs['preprocessed_area_fraction'] - 1.0080) < 0.02
        assert abs(tracked_with_gaps.attrs['N_objects_prefiltered'] - 1041) < 2
        assert abs(tracked_with_gaps.attrs['N_objects_filtered'] - 522) < 2
        assert abs(tracked_with_gaps.attrs['N_events_final'] - 38) < 1