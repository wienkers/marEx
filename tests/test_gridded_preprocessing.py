import pytest
import xarray as xr
import numpy as np
from pathlib import Path
import marEx


class TestGriddedPreprocessing:
    """Test preprocessing functionality for gridded data using test datasets."""
    
    @classmethod
    def setup_class(cls):
        """Load test data for all tests."""
        test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        ds = xr.open_zarr(str(test_data_path), chunks={}).persist()
        cls.sst_data = ds.to
        
        # Define standard dimensions for gridded data
        cls.dimensions = {
            'time': 'time',
            'xdim': 'lon', 
            'ydim': 'lat'
        }
        
        # Standard dask chunks for output
        cls.dask_chunks = {'time': 25}
    
    def test_shifting_baseline_hobday_extreme(self):
        """Test preprocessing with shifting_baseline + hobday_extreme combination."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly='shifting_baseline',
            method_extreme='hobday_extreme',
            threshold_percentile=95,
            window_year_baseline=5,  # Reduced for test data
            smooth_days_baseline=11,  # Reduced for test data
            window_days_hobday=5,  # Reduced for test data
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks
        )
        
        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert 'extreme_events' in extremes_ds.data_vars
        assert 'dat_detrend' in extremes_ds.data_vars
        assert 'thresholds' in extremes_ds.data_vars
        assert 'mask' in extremes_ds.data_vars
        
        # Verify attributes
        assert extremes_ds.attrs['method_anomaly'] == 'shifting_baseline'
        assert extremes_ds.attrs['method_extreme'] == 'hobday_extreme'
        assert extremes_ds.attrs['threshold_percentile'] == 95
        
        # Verify data types
        assert extremes_ds.extreme_events.dtype == bool
        assert extremes_ds.dat_detrend.dtype == np.float32
        
        # Verify dimensions
        assert 'time' in extremes_ds.extreme_events.dims
        assert 'lat' in extremes_ds.extreme_events.dims
        assert 'lon' in extremes_ds.extreme_events.dims
        assert 'dayofyear' in extremes_ds.thresholds.dims
        
        # Verify reasonable extreme event frequency (should be close to 5% for 95th percentile)
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        assert 0.03 < extreme_frequency < 0.07, f"Extreme frequency {extreme_frequency} outside expected range"
    
    def test_detrended_baseline_global_extreme(self):
        """Test preprocessing with detrended_baseline + global_extreme combination."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly='detrended_baseline',
            method_extreme='global_extreme',
            threshold_percentile=95,
            detrend_orders=[1, 2],
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks
        )
        
        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert 'extreme_events' in extremes_ds.data_vars
        assert 'dat_detrend' in extremes_ds.data_vars
        assert 'thresholds' in extremes_ds.data_vars
        assert 'mask' in extremes_ds.data_vars
        
        # Verify attributes
        assert extremes_ds.attrs['method_anomaly'] == 'detrended_baseline'
        assert extremes_ds.attrs['method_extreme'] == 'global_extreme'
        assert extremes_ds.attrs['threshold_percentile'] == 95
        
        # Verify data types
        assert extremes_ds.extreme_events.dtype == bool
        assert extremes_ds.dat_detrend.dtype == np.float32
        
        # Verify dimensions
        assert 'time' in extremes_ds.extreme_events.dims
        assert 'lat' in extremes_ds.extreme_events.dims
        assert 'lon' in extremes_ds.extreme_events.dims
        
        # For global_extreme, thresholds should be 2D (lat, lon) not 3D with dayofyear
        assert 'dayofyear' not in extremes_ds.thresholds.dims
        assert 'lat' in extremes_ds.thresholds.dims
        assert 'lon' in extremes_ds.thresholds.dims
        
        # Verify reasonable extreme event frequency
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        assert 0.03 < extreme_frequency < 0.07, f"Extreme frequency {extreme_frequency} outside expected range"
    
    def test_output_consistency(self):
        """Test that both preprocessing methods produce consistent output structures."""
        # Run both methods
        shifting_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly='shifting_baseline',
            method_extreme='hobday_extreme',
            threshold_percentile=95,
            window_year_baseline=5,
            smooth_days_baseline=21,
            window_days_hobday=11,
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks
        )
        
        detrended_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly='detrended_baseline', 
            method_extreme='global_extreme',
            threshold_percentile=95,
            detrend_orders=[1, 2],
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks
        )
        
        # Both should have the same core data variables
        core_vars = ['extreme_events', 'dat_detrend', 'mask']
        for var in core_vars:
            assert var in shifting_ds.data_vars
            assert var in detrended_ds.data_vars
        
        # Both should have mask with same spatial shape
        assert shifting_ds.mask.shape == detrended_ds.mask.shape
        
        # Extreme events should have consistent spatial dimensions
        assert shifting_ds.extreme_events.dims[-2:] == detrended_ds.extreme_events.dims[-2:]
        
        # Both should have consistent coordinate structure (lat, lon)
        assert 'lat' in shifting_ds.coords
        assert 'lon' in shifting_ds.coords
        assert 'lat' in detrended_ds.coords  
        assert 'lon' in detrended_ds.coords