from pathlib import Path

import numpy as np
import xarray as xr

import marEx

from .conftest import assert_percentile_frequency


class TestGriddedPreprocessing:
    """Test preprocessing functionality for gridded data using test datasets."""

    @classmethod
    def setup_class(cls):
        """Load test data for all tests."""
        test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        cls.sst_data = xr.open_zarr(str(test_data_path), chunks={}).to.persist()

        # Artificially make some masked NaN data in the 2nd lat and 2nd lon point
        cls.sst_data = cls.sst_data.where(
            ~((cls.sst_data.lat == cls.sst_data.lat[1]) & (cls.sst_data.lon == cls.sst_data.lon[1])), np.nan
        )

        # Define standard dimensions for gridded data
        cls.dimensions = {"time": "time", "x": "lon", "y": "lat"}

        # Standard dask chunks for output
        cls.dask_chunks = {"time": 25}

    def test_shifting_baseline_hobday_extreme(self):
        """Test preprocessing with shifting_baseline + hobday_extreme combination."""
        window_year_baseline = 5

        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="shifting_baseline",
            method_extreme="hobday_extreme",
            threshold_percentile=95,
            window_year_baseline=window_year_baseline,  # Reduced for test data
            smooth_days_baseline=11,  # Reduced for test data
            window_days_hobday=3,  # Reduced for test data
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert "extreme_events" in extremes_ds.data_vars
        assert "dat_anomaly" in extremes_ds.data_vars
        assert "thresholds" in extremes_ds.data_vars
        assert "mask" in extremes_ds.data_vars

        # Verify attributes
        assert extremes_ds.attrs["method_anomaly"] == "shifting_baseline"
        assert extremes_ds.attrs["method_extreme"] == "hobday_extreme"
        assert extremes_ds.attrs["threshold_percentile"] == 95

        # Verify data types
        assert extremes_ds.extreme_events.dtype == bool
        assert extremes_ds.dat_anomaly.dtype == np.float32

        # Verify dimensions
        assert "time" in extremes_ds.extreme_events.dims
        assert "lat" in extremes_ds.extreme_events.dims
        assert "lon" in extremes_ds.extreme_events.dims
        assert "dayofyear" in extremes_ds.thresholds.dims

        # Verify time dimension: shifting_baseline should reduce time by window_year_baseline
        input_time_size = self.sst_data.sizes["time"]
        output_time_size = extremes_ds.sizes["time"]

        expected_time_reduction = window_year_baseline * 365  # Approximate daily reduction
        assert (
            output_time_size < input_time_size
        ), f"Output time size ({output_time_size}) should be less than input ({input_time_size}) for shifting_baseline"
        # Allow some flexibility due to leap years and exact windowing
        time_reduction = input_time_size - output_time_size
        assert (
            abs(time_reduction - expected_time_reduction) <= 10
        ), f"Time reduction ({time_reduction}) should be approximately {expected_time_reduction} days"

        # Verify reasonable extreme event frequency (should be close to 5% for 95th percentile)
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        print(f"Exact extreme_frequency for shifting_baseline + hobday_extreme: {extreme_frequency}")
        assert_percentile_frequency(extreme_frequency, 95, description="shifting_baseline + hobday_extreme")

    def test_detrend_harmonic_global_extreme(self):
        """Test preprocessing with detrend_harmonic + global_extreme combination."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_harmonic",
            method_extreme="global_extreme",
            threshold_percentile=95,
            detrend_orders=[1, 2],
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert "extreme_events" in extremes_ds.data_vars
        assert "dat_anomaly" in extremes_ds.data_vars
        assert "thresholds" in extremes_ds.data_vars
        assert "mask" in extremes_ds.data_vars

        # Verify attributes
        assert extremes_ds.attrs["method_anomaly"] == "detrend_harmonic"
        assert extremes_ds.attrs["method_extreme"] == "global_extreme"
        assert extremes_ds.attrs["threshold_percentile"] == 95

        # Verify data types
        assert extremes_ds.extreme_events.dtype == bool
        assert extremes_ds.dat_anomaly.dtype == np.float32

        # Verify dimensions
        assert "time" in extremes_ds.extreme_events.dims
        assert "lat" in extremes_ds.extreme_events.dims
        assert "lon" in extremes_ds.extreme_events.dims

        # For global_extreme, thresholds should be 2D (lat, lon) not 3D with dayofyear
        assert "dayofyear" not in extremes_ds.thresholds.dims
        assert "lat" in extremes_ds.thresholds.dims
        assert "lon" in extremes_ds.thresholds.dims

        # Verify time dimension: detrend_harmonic should preserve all time steps
        input_time_size = self.sst_data.sizes["time"]
        output_time_size = extremes_ds.sizes["time"]
        assert (
            output_time_size == input_time_size
        ), f"Output time size ({output_time_size}) should equal input ({input_time_size}) for detrend_harmonic"

        # Verify reasonable extreme event frequency
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        print(f"Exact extreme_frequency for detrend_harmonic + global_extreme: {extreme_frequency}")
        assert_percentile_frequency(extreme_frequency, 95, description="detrend_harmonic + global_extreme")

    def test_output_consistency(self):
        """Test that all preprocessing methods produce consistent output structures."""
        # Run all methods
        shifting_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="shifting_baseline",
            method_extreme="hobday_extreme",
            threshold_percentile=95,
            window_year_baseline=5,
            smooth_days_baseline=21,
            window_days_hobday=11,
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        detrended_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_harmonic",
            method_extreme="global_extreme",
            threshold_percentile=95,
            detrend_orders=[1, 2],
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        fixed_detrended_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_fixed_baseline",
            method_extreme="hobday_extreme",
            threshold_percentile=95,
            detrend_orders=[1],
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        fixed_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="fixed_baseline",
            method_extreme="hobday_extreme",
            threshold_percentile=95,
            detrend_orders=[1],
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # All should have the same core data variables
        core_vars = ["extreme_events", "dat_anomaly", "mask"]
        for var in core_vars:
            assert var in shifting_ds.data_vars
            assert var in detrended_ds.data_vars
            assert var in fixed_detrended_ds.data_vars
            assert var in fixed_ds.data_vars

        # All should have mask with same spatial shape
        datasets = [shifting_ds, detrended_ds, fixed_detrended_ds, fixed_ds]
        mask_shapes = [ds.mask.shape for ds in datasets]
        assert all(shape == mask_shapes[0] for shape in mask_shapes)

        # Extreme events should have consistent spatial dimensions
        spatial_dims = [ds.extreme_events.dims[-2:] for ds in datasets]
        assert all(dims == spatial_dims[0] for dims in spatial_dims)

    def test_std_normalise_detrend_harmonic(self):
        """Test preprocessing with std_normalise=True for detrend_harmonic method."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_harmonic",
            method_extreme="global_extreme",
            threshold_percentile=95,
            std_normalise=True,
            detrend_orders=[1, 2],
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Verify output structure includes standardised data
        assert isinstance(extremes_ds, xr.Dataset)
        assert "extreme_events" in extremes_ds.data_vars
        assert "dat_anomaly" in extremes_ds.data_vars
        assert "thresholds" in extremes_ds.data_vars
        assert "mask" in extremes_ds.data_vars

        # Check for additional variables when std_normalise=True
        assert "dat_stn" in extremes_ds.data_vars, "dat_stn should be present when std_normalise=True"
        assert "STD" in extremes_ds.data_vars, "STD should be present when std_normalise=True"
        assert "extreme_events_stn" in extremes_ds.data_vars, "extreme_events_stn should be present when std_normalise=True"
        assert "thresholds_stn" in extremes_ds.data_vars, "thresholds_stn should be present when std_normalise=True"

        # Verify attributes
        assert extremes_ds.attrs["method_anomaly"] == "detrend_harmonic"
        assert extremes_ds.attrs["method_extreme"] == "global_extreme"
        assert extremes_ds.attrs["threshold_percentile"] == 95
        assert extremes_ds.attrs["std_normalise"] is True

        # Verify data types
        assert extremes_ds.extreme_events.dtype == bool
        assert extremes_ds.extreme_events_stn.dtype == bool
        assert extremes_ds.dat_anomaly.dtype == np.float32
        assert extremes_ds.dat_stn.dtype == np.float32
        assert extremes_ds.STD.dtype == np.float32

        # Verify dimensions
        assert "time" in extremes_ds.dat_stn.dims
        assert "lat" in extremes_ds.dat_stn.dims
        assert "lon" in extremes_ds.dat_stn.dims

        # STD should have dayofyear dimension but not time
        assert "dayofyear" in extremes_ds.STD.dims
        assert "lat" in extremes_ds.STD.dims
        assert "lon" in extremes_ds.STD.dims
        assert "time" not in extremes_ds.STD.dims

        # Verify reasonable extreme event frequency for both regular and standardised
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        extreme_frequency_stn = float(extremes_ds.extreme_events_stn.mean())
        print(f"Extreme frequency (regular): {extreme_frequency}")
        print(f"Extreme frequency (standardised): {extreme_frequency_stn}")

        # Both should be close to 5% for 95th percentile
        assert_percentile_frequency(
            extreme_frequency,
            95,
            description="Regular extreme events (std_normalise=True)",
        )
        assert_percentile_frequency(
            extreme_frequency_stn,
            95,
            description="Standardised extreme events (std_normalise=True)",
        )

    def test_shifting_baseline_hobday_extreme_exact_percentile(self):
        """Test preprocessing with shifting_baseline + hobday_extreme combination and exact_percentile=True."""
        window_year_baseline = 5
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="shifting_baseline",
            method_extreme="hobday_extreme",
            threshold_percentile=95,
            method_percentile="exact",
            window_year_baseline=window_year_baseline,  # Reduced for test data
            smooth_days_baseline=5,  # Reduced for test data
            window_days_hobday=3,  # Reduced for test data
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert "extreme_events" in extremes_ds.data_vars
        assert "dat_anomaly" in extremes_ds.data_vars
        assert "thresholds" in extremes_ds.data_vars
        assert "mask" in extremes_ds.data_vars

        # Verify attributes
        assert extremes_ds.attrs["method_anomaly"] == "shifting_baseline"
        assert extremes_ds.attrs["method_extreme"] == "hobday_extreme"
        assert extremes_ds.attrs["threshold_percentile"] == 95
        assert extremes_ds.attrs["method_percentile"] == "exact"

        # Verify data types
        assert extremes_ds.extreme_events.dtype == bool
        assert extremes_ds.dat_anomaly.dtype == np.float32

        # Verify dimensions
        assert "time" in extremes_ds.extreme_events.dims
        assert "lat" in extremes_ds.extreme_events.dims
        assert "lon" in extremes_ds.extreme_events.dims
        assert "dayofyear" in extremes_ds.thresholds.dims

        # Verify time dimension: shifting_baseline should reduce time by window_year_baseline
        input_time_size = self.sst_data.sizes["time"]
        output_time_size = extremes_ds.sizes["time"]
        expected_time_reduction = window_year_baseline * 365  # Approximate daily reduction
        assert (
            output_time_size < input_time_size
        ), f"Output time size ({output_time_size}) should be less than input ({input_time_size}) for shifting_baseline"
        # Allow some flexibility due to leap years and exact windowing
        time_reduction = input_time_size - output_time_size
        assert (
            abs(time_reduction - expected_time_reduction) <= 10
        ), f"Time reduction ({time_reduction}) should be approximately {expected_time_reduction} days"

        # Verify reasonable extreme event frequency (should be close to 5% for 95th percentile)
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        print(f"Exact extreme_frequency for shifting_baseline + hobday_extreme (exact_percentile=True): {extreme_frequency}")
        assert_percentile_frequency(
            extreme_frequency,
            95,
            description="shifting_baseline + hobday_extreme (exact_percentile=True)",
        )

    def test_detrend_harmonic_global_extreme_exact_percentile(self):
        """Test preprocessing with detrend_harmonic + global_extreme combination and exact_percentile=True."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_harmonic",
            method_extreme="global_extreme",
            threshold_percentile=95,
            method_percentile="exact",
            detrend_orders=[1, 2],
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert "extreme_events" in extremes_ds.data_vars
        assert "dat_anomaly" in extremes_ds.data_vars
        assert "thresholds" in extremes_ds.data_vars
        assert "mask" in extremes_ds.data_vars

        # Verify attributes
        assert extremes_ds.attrs["method_anomaly"] == "detrend_harmonic"
        assert extremes_ds.attrs["method_extreme"] == "global_extreme"
        assert extremes_ds.attrs["threshold_percentile"] == 95
        assert extremes_ds.attrs["method_percentile"] == "exact"

        # Verify data types
        assert extremes_ds.extreme_events.dtype == bool
        assert extremes_ds.dat_anomaly.dtype == np.float32

        # Verify dimensions
        assert "time" in extremes_ds.extreme_events.dims
        assert "lat" in extremes_ds.extreme_events.dims
        assert "lon" in extremes_ds.extreme_events.dims

        # For global_extreme, thresholds should be 2D (lat, lon) not 3D with dayofyear
        assert "dayofyear" not in extremes_ds.thresholds.dims
        assert "lat" in extremes_ds.thresholds.dims
        assert "lon" in extremes_ds.thresholds.dims

        # Verify time dimension: detrend_harmonic should preserve all time steps
        input_time_size = self.sst_data.sizes["time"]
        output_time_size = extremes_ds.sizes["time"]
        assert (
            output_time_size == input_time_size
        ), f"Output time size ({output_time_size}) should equal input ({input_time_size}) for detrend_harmonic"

        # Verify reasonable extreme event frequency
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        print(f"Exact extreme_frequency for detrend_harmonic + global_extreme (exact_percentile=True): {extreme_frequency}")
        # For exact_percentile=True, we expect very precise adherence to the percentile
        assert_percentile_frequency(
            extreme_frequency,
            95,
            tolerance_std=1.0,
            description="detrend_harmonic + global_extreme (exact_percentile=True)",
        )

    def test_custom_dimension_names(self):
        """Test preprocessing with custom dimension and coordinate names for both detect methods."""
        # Create dataset with custom dimension and coordinate names
        # Dimensions: "t", "x", "y"
        # Coordinates: "T", "longitude", "latitude"
        da_renamed = xr.DataArray(
            self.sst_data.values,
            dims=["t", "y", "x"],
            coords={
                "T": ("t", self.sst_data.time.values),
                "latitude": ("y", self.sst_data.lat.values),
                "longitude": ("x", self.sst_data.lon.values),
            },
            attrs=self.sst_data.attrs,
        )
        # Rechunk the data
        da_renamed = da_renamed.chunk({"t": 25})

        # Define custom dimensions and coordinates mapping
        custom_dimensions = {"time": "t", "x": "x", "y": "y"}
        custom_coordinates = {"time": "T", "x": "longitude", "y": "latitude"}

        # Test 1: detrend_harmonic + global_extreme
        extremes_ds_detrended = marEx.preprocess_data(
            da_renamed,
            method_anomaly="detrend_harmonic",
            method_extreme="global_extreme",
            threshold_percentile=95,
            detrend_orders=[1, 2],
            dimensions=custom_dimensions,
            coordinates=custom_coordinates,
            dask_chunks={"t": 25},
        )

        # Verify output structure for detrend_harmonic method
        assert isinstance(extremes_ds_detrended, xr.Dataset)
        assert "extreme_events" in extremes_ds_detrended.data_vars
        assert "dat_anomaly" in extremes_ds_detrended.data_vars
        assert "thresholds" in extremes_ds_detrended.data_vars
        assert "mask" in extremes_ds_detrended.data_vars

        # Verify dimensions are correctly named
        assert "t" in extremes_ds_detrended.extreme_events.dims
        assert "y" in extremes_ds_detrended.extreme_events.dims
        assert "x" in extremes_ds_detrended.extreme_events.dims

        # Verify coordinates are present
        assert "T" in extremes_ds_detrended.coords
        assert "latitude" in extremes_ds_detrended.coords
        assert "longitude" in extremes_ds_detrended.coords

        # Verify attributes for detrend_harmonic
        assert extremes_ds_detrended.attrs["method_anomaly"] == "detrend_harmonic"
        assert extremes_ds_detrended.attrs["method_extreme"] == "global_extreme"

        # For global_extreme, thresholds should be 2D (y, x) not 3D with dayofyear
        assert "dayofyear" not in extremes_ds_detrended.thresholds.dims
        assert "y" in extremes_ds_detrended.thresholds.dims
        assert "x" in extremes_ds_detrended.thresholds.dims

        # Verify reasonable extreme event frequency
        extreme_frequency_detrended = float(extremes_ds_detrended.extreme_events.mean())
        print(f"extreme_frequency for detrend_harmonic + global_extreme: {extreme_frequency_detrended}")
        assert_percentile_frequency(
            extreme_frequency_detrended,
            95,
            description="detrend_harmonic + global_extreme",
        )

        # Test 2: shifting_baseline + hobday_extreme
        extremes_ds_shifting = marEx.preprocess_data(
            da_renamed,
            method_anomaly="shifting_baseline",
            method_extreme="hobday_extreme",
            threshold_percentile=95,
            window_year_baseline=5,  # Reduced for test data
            smooth_days_baseline=11,  # Reduced for test data
            window_days_hobday=3,  # Reduced for test data
            dimensions=custom_dimensions,
            coordinates=custom_coordinates,
            dask_chunks={"t": 25},
        )

        # Verify output structure for shifting_baseline method
        assert isinstance(extremes_ds_shifting, xr.Dataset)
        assert "extreme_events" in extremes_ds_shifting.data_vars
        assert "dat_anomaly" in extremes_ds_shifting.data_vars
        assert "thresholds" in extremes_ds_shifting.data_vars
        assert "mask" in extremes_ds_shifting.data_vars

        # Verify dimensions are correctly named
        assert "t" in extremes_ds_shifting.extreme_events.dims
        assert "y" in extremes_ds_shifting.extreme_events.dims
        assert "x" in extremes_ds_shifting.extreme_events.dims

        # Verify coordinates are present
        assert "T" in extremes_ds_shifting.coords
        assert "latitude" in extremes_ds_shifting.coords
        assert "longitude" in extremes_ds_shifting.coords

        # Verify attributes for shifting_baseline
        assert extremes_ds_shifting.attrs["method_anomaly"] == "shifting_baseline"
        assert extremes_ds_shifting.attrs["method_extreme"] == "hobday_extreme"

        # For hobday_extreme, thresholds should have dayofyear dimension
        assert "dayofyear" in extremes_ds_shifting.thresholds.dims
        assert "y" in extremes_ds_shifting.thresholds.dims
        assert "x" in extremes_ds_shifting.thresholds.dims

        # Verify time dimension: shifting_baseline should reduce time
        input_time_size = da_renamed.sizes["t"]
        output_time_size = extremes_ds_shifting.sizes["t"]
        assert output_time_size < input_time_size, "shifting_baseline should reduce time dimension"

        # Verify reasonable extreme event frequency
        extreme_frequency_shifting = float(extremes_ds_shifting.extreme_events.mean())
        print(f"extreme_frequency for shifting_baseline + hobday_extreme: {extreme_frequency_shifting}")
        assert_percentile_frequency(
            extreme_frequency_shifting,
            95,
            description="shifting_baseline + hobday_extreme",
        )

        # Test 3: Verify both methods produce consistent core structure
        core_vars = ["extreme_events", "dat_anomaly", "mask"]
        for var in core_vars:
            assert var in extremes_ds_detrended.data_vars
            assert var in extremes_ds_shifting.data_vars

        # Both should have same coordinate structure
        assert "T" in extremes_ds_detrended.coords
        assert "latitude" in extremes_ds_detrended.coords
        assert "longitude" in extremes_ds_detrended.coords
        assert "T" in extremes_ds_shifting.coords
        assert "latitude" in extremes_ds_shifting.coords
        assert "longitude" in extremes_ds_shifting.coords

    def test_hobday_extreme_exact_percentile(self):
        """Test Hobday extreme method with exact percentile."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_harmonic",
            method_extreme="hobday_extreme",
            method_percentile="exact",
            threshold_percentile=95,
            window_days_hobday=11,
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert "extreme_events" in extremes_ds.data_vars
        assert "thresholds" in extremes_ds.data_vars
        assert extremes_ds.extreme_events.dtype == bool

        # Verify thresholds have correct dimensions (day-of-year specific)
        assert "dayofyear" in extremes_ds.thresholds.dims
        assert extremes_ds.thresholds.dims == ("dayofyear", "lat", "lon")

        # Verify some extremes were identified
        assert extremes_ds.extreme_events.sum() > 0

        # Verify threshold values are reasonable
        threshold_values = extremes_ds.thresholds.values
        finite_mask = np.isfinite(threshold_values)
        assert finite_mask.sum() > 0  # Should have some finite values
        assert np.all(threshold_values[finite_mask] > 0)  # Finite values should be positive

        # Verify attributes
        assert extremes_ds.attrs["method_percentile"] == "exact"

    def test_fixed_baseline_global_extreme(self):
        """Test preprocessing with fixed_baseline + global_extreme combination."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="fixed_baseline",
            method_extreme="global_extreme",
            threshold_percentile=95,
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert "extreme_events" in extremes_ds.data_vars
        assert "dat_anomaly" in extremes_ds.data_vars
        assert "thresholds" in extremes_ds.data_vars
        assert "mask" in extremes_ds.data_vars

        # Verify attributes
        assert extremes_ds.attrs["method_anomaly"] == "fixed_baseline"
        assert extremes_ds.attrs["method_extreme"] == "global_extreme"
        assert extremes_ds.attrs["threshold_percentile"] == 95

        # Verify data types
        assert extremes_ds.extreme_events.dtype == bool
        assert extremes_ds.dat_anomaly.dtype == np.float32

        # Fixed_baseline should preserve all time steps (unlike shifting_baseline)
        input_time_size = self.sst_data.sizes["time"]
        output_time_size = extremes_ds.sizes["time"]
        assert (
            output_time_size == input_time_size
        ), f"fixed_baseline should preserve all time steps: {output_time_size} == {input_time_size}"

        # Verify global extreme thresholds are 2D (lat, lon)
        assert "dayofyear" not in extremes_ds.thresholds.dims
        assert "lat" in extremes_ds.thresholds.dims
        assert "lon" in extremes_ds.thresholds.dims

        # Verify reasonable extreme event frequency
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        assert_percentile_frequency(extreme_frequency, 95, description="fixed_baseline + global_extreme")

    def test_fixed_baseline_hobday_extreme(self):
        """Test preprocessing with fixed_baseline + hobday_extreme combination."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="fixed_baseline",
            method_extreme="hobday_extreme",
            threshold_percentile=95,
            window_days_hobday=11,
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert "extreme_events" in extremes_ds.data_vars
        assert "thresholds" in extremes_ds.data_vars

        # Fixed_baseline should preserve all time steps
        input_time_size = self.sst_data.sizes["time"]
        output_time_size = extremes_ds.sizes["time"]
        assert output_time_size == input_time_size

        # Verify hobday extreme thresholds are 3D (dayofyear, lat, lon)
        assert "dayofyear" in extremes_ds.thresholds.dims
        assert "lat" in extremes_ds.thresholds.dims
        assert "lon" in extremes_ds.thresholds.dims

        # Verify extreme frequency
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        assert_percentile_frequency(extreme_frequency, 95, description="fixed_baseline + hobday_extreme")

    def test_detrend_fixed_baseline_global_extreme(self):
        """Test preprocessing with fixed_detrend_harmonic + global_extreme combination."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_fixed_baseline",
            method_extreme="global_extreme",
            threshold_percentile=95,
            detrend_orders=[1, 2],  # Linear and quadratic trends
            force_zero_mean=True,
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert "extreme_events" in extremes_ds.data_vars
        assert "dat_anomaly" in extremes_ds.data_vars
        assert "thresholds" in extremes_ds.data_vars

        # Verify attributes
        assert extremes_ds.attrs["method_anomaly"] == "detrend_fixed_baseline"
        assert extremes_ds.attrs["detrend_orders"] == [1, 2]
        assert extremes_ds.attrs["force_zero_mean"] is True

        # Should preserve all time steps
        input_time_size = self.sst_data.sizes["time"]
        output_time_size = extremes_ds.sizes["time"]
        assert output_time_size == input_time_size

        # Verify anomalies have approximately zero mean (due to force_zero_mean=True)
        anomaly_mean = float(extremes_ds.dat_anomaly.mean())
        assert abs(anomaly_mean) < 0.01, f"Expected near-zero mean, got {anomaly_mean}"

        # Verify extreme frequency
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        assert_percentile_frequency(extreme_frequency, 95, description="fixed_detrend_harmonic + global_extreme")

    def test_detrend_fixed_baseline_different_orders(self):
        """Test detrend_fixed_baseline with different polynomial orders."""
        # Test linear only
        result_linear = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_fixed_baseline",
            detrend_orders=[1],  # Linear only
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Test quadratic only
        result_quadratic = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_fixed_baseline",
            detrend_orders=[2],  # Quadratic only
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Test cubic
        result_cubic = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_fixed_baseline",
            detrend_orders=[1, 2, 3],  # Linear + quadratic + cubic
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # All should work and produce reasonable results
        for result, name in [(result_linear, "linear"), (result_quadratic, "quadratic"), (result_cubic, "cubic")]:
            assert isinstance(result, xr.Dataset)
            extreme_frequency = float(result.extreme_events.mean())
            assert 0.025 < extreme_frequency < 0.075, f"{name} detrending produced unreasonable frequency: {extreme_frequency}"

    def test_detrend_fixed_baseline_force_zero_mean(self):
        """Test detrend_fixed_baseline with force_zero_mean parameter."""
        # Test with force_zero_mean=True (default)
        result_true = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_fixed_baseline",
            force_zero_mean=True,
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # Test with force_zero_mean=False
        result_false = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrend_fixed_baseline",
            force_zero_mean=False,
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
        )

        # force_zero_mean=True should have very small mean
        mean_true = float(result_true.dat_anomaly.mean())
        assert abs(mean_true) < 0.01, f"force_zero_mean=True should give near-zero mean, got {mean_true}"

        # force_zero_mean=False may have non-zero mean
        mean_false = float(result_false.dat_anomaly.mean())
        # We can't assert much about this, but at least check it's a valid number
        assert isinstance(mean_false, float)  # Just check it's a valid number

    def test_with_all_extreme_methods(self):
        """Test that new anomaly methods work with both extreme detection methods."""
        # Test all combinations
        combinations = [
            ("fixed_baseline", "global_extreme"),
            ("fixed_baseline", "hobday_extreme"),
            ("detrend_fixed_baseline", "global_extreme"),
            ("detrend_fixed_baseline", "hobday_extreme"),
            ("shifting_baseline", "global_extreme"),
            ("shifting_baseline", "hobday_extreme"),
            ("detrend_harmonic", "global_extreme"),
            ("detrend_harmonic", "hobday_extreme"),
        ]

        for method_anomaly, method_extreme in combinations:
            result = marEx.preprocess_data(
                self.sst_data,
                method_anomaly=method_anomaly,
                method_extreme=method_extreme,
                threshold_percentile=95,
                detrend_orders=[1] if "detrended" in method_anomaly else None,
                window_days_hobday=11 if method_extreme == "hobday_extreme" else None,
                dimensions=self.dimensions,
                dask_chunks=self.dask_chunks,
            )

            assert isinstance(result, xr.Dataset)
            assert "extreme_events" in result.data_vars
            assert result.attrs["method_anomaly"] == method_anomaly
            assert result.attrs["method_extreme"] == method_extreme

            # Verify reasonable extreme frequency
            extreme_frequency = float(result.extreme_events.mean())
            assert (
                0.025 < extreme_frequency < 0.075
            ), f"{method_anomaly}+{method_extreme} produced unreasonable frequency: {extreme_frequency}"
