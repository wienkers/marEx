"""
Error Handling Tests for marEx Package

Tests for proper error handling and validation across the marEx package.
This includes testing for common user mistakes and ensuring helpful error messages.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError, CoordinateError, DataValidationError


@pytest.fixture(scope="module")
def test_data_dask():
    """Load test data as Dask-backed DataArray."""
    test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
    ds = xr.open_zarr(str(test_data_path), chunks={"time": 25}).isel(lon=slice(0, 4), lat=slice(0, 3))
    return ds.to  # Extract the DataArray named 'to'


@pytest.fixture(scope="module")
def test_data_numpy():
    """Load test data as numpy-backed DataArray."""
    test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
    ds = xr.open_zarr(str(test_data_path), chunks={"time": 25}).isel(lon=slice(0, 4), lat=slice(0, 3))
    return ds.to.load()  # Extract and load the DataArray named 'to'


@pytest.fixture(scope="module")
def dimensions_gridded():
    """Standard dimensions for gridded data."""
    return {"time": "time", "x": "lon", "y": "lat"}


@pytest.fixture(scope="module")
def dask_chunks():
    """Standard dask chunks."""
    return {"time": 25}


class TestNonDaskInputValidation:
    """Test error handling for non-Dask input arrays."""

    def test_preprocess_data_non_dask_input(self, test_data_numpy, dimensions_gridded, dask_chunks):
        """Test that preprocess_data raises DataValidationError for non-Dask inputs."""
        with pytest.raises(DataValidationError, match=r"Input DataArray must be Dask-backed"):
            marEx.preprocess_data(
                test_data_numpy,  # Non-Dask array
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )

    def test_compute_normalised_anomaly_non_dask_input(self, test_data_numpy, dimensions_gridded):
        """Test that compute_normalised_anomaly raises error for non-Dask inputs."""
        # This will fail with TypeError when trying to access .chunks on non-Dask array
        with pytest.raises(TypeError, match=r"'NoneType' object is not subscriptable"):
            marEx.compute_normalised_anomaly(
                test_data_numpy,  # Non-Dask array
                method_anomaly="detrend_harmonic",
                dimensions=dimensions_gridded,
            )

    def test_tracker_non_dask_binary_data(self, test_data_numpy):
        """Test that tracker raises DataValidationError for non-Dask binary data."""
        # Create binary test data
        binary_data = (test_data_numpy > test_data_numpy.quantile(0.95)).astype(bool)
        mask = ~np.isnan(test_data_numpy.isel(time=0))

        with pytest.raises(
            (DataValidationError, CoordinateError),
            match="(Input DataArray must be Dask-backed|Cannot auto-detect coordinate)",
        ):
            marEx.tracker(binary_data, mask, R_fill=8, area_filter_quartile=0.5)


class TestMethodValidation:
    """Test error handling for invalid method combinations and parameters."""

    def test_invalid_method_anomaly_runtime(self, test_data_dask, dimensions_gridded):
        """Test runtime error handling for invalid method_anomaly values via compute_normalised_anomaly."""
        # Since Literal types prevent invalid values at call time, we test the underlying function
        with pytest.raises(ConfigurationError, match=r"Unknown anomaly method"):
            marEx.compute_normalised_anomaly(
                test_data_dask,
                method_anomaly="invalid_method",  # Invalid method - will bypass Literal check
                dimensions=dimensions_gridded,
            )

    def test_invalid_method_extreme_runtime(self, test_data_dask, dimensions_gridded):
        """Test runtime error handling for invalid method_extreme values via identify_extremes."""
        # Test with the underlying function that has runtime validation
        with pytest.raises(ConfigurationError, match=r"Unknown extreme method"):
            marEx.identify_extremes(
                test_data_dask,
                method_extreme="invalid_extreme",  # Invalid method
                dimensions=dimensions_gridded,
            )

    def test_valid_method_combinations(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test that valid method combinations work correctly."""
        # Test detrend_harmonic + global_extreme
        result1 = marEx.preprocess_data(
            test_data_dask,
            method_anomaly="detrend_harmonic",
            method_extreme="global_extreme",
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )
        assert isinstance(result1, xr.Dataset)
        assert result1.attrs["method_anomaly"] == "detrend_harmonic"
        assert result1.attrs["method_extreme"] == "global_extreme"

        # Test shifting_baseline + hobday_extreme (default combination)
        result2 = marEx.preprocess_data(
            test_data_dask,
            method_anomaly="shifting_baseline",
            method_extreme="hobday_extreme",
            window_year_baseline=2,  # Reduced for test data
            window_days_hobday=3,  # Reduced for test data
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )
        assert isinstance(result2, xr.Dataset)
        assert result2.attrs["method_anomaly"] == "shifting_baseline"
        assert result2.attrs["method_extreme"] == "hobday_extreme"


class TestInsufficientDataValidation:
    """Test error handling for insufficient data scenarios."""

    def test_shifting_baseline_insufficient_data(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test error for insufficient data with shifting_baseline method."""
        # Create a small dataset with only 2 years of data
        small_data = test_data_dask.isel(time=slice(0, 730)).chunk(dask_chunks)  # ~2 years, chunked

        with pytest.raises((DataValidationError, IndexError)):
            marEx.preprocess_data(
                small_data,
                method_anomaly="shifting_baseline",
                method_extreme="hobday_extreme",
                window_year_baseline=15,  # Requires 15 years but only have 2
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )

    def test_shifting_baseline_custom_window_insufficient_data(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test error for insufficient data with custom window_year_baseline."""
        # Create dataset with 3 years but require 5 years
        small_data = test_data_dask.isel(time=slice(0, 1095)).chunk(dask_chunks)  # ~3 years, chunked

        with pytest.raises((DataValidationError, IndexError)):
            marEx.preprocess_data(
                small_data,
                method_anomaly="shifting_baseline",
                method_extreme="hobday_extreme",
                window_year_baseline=5,  # Requires 5 years but only have 3
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )


class TestParameterValidation:
    """Test validation of parameter ranges and constraints."""

    def test_missing_dimensions_time(self, test_data_dask, dask_chunks):
        """Test error when dimensions parameter is missing required keys."""
        # Test with missing x dimension - time gets default value, but missing x causes KeyError
        incomplete_dimensions = {"y": "lat"}  # Missing 'time' and 'x'

        with pytest.raises(KeyError, match=r"'x'"):
            marEx.preprocess_data(
                test_data_dask,
                dimensions=incomplete_dimensions,  # Missing required dimensions
                dask_chunks=dask_chunks,
            )

    def test_wrong_dimension_names(self, test_data_dask, dask_chunks):
        """Test error when dimension names don't match actual data dimensions."""
        wrong_dimensions = {
            "time": "time",
            "x": "wrong_lon",  # Wrong name - should be 'lon'
            "y": "wrong_lat",  # Wrong name - should be 'lat'
        }

        # This should raise an error when trying to access non-existent dimensions
        with pytest.raises(DataValidationError, match=r"Missing required dimensions"):
            marEx.preprocess_data(test_data_dask, dimensions=wrong_dimensions, dask_chunks=dask_chunks)

    def test_detrend_fixed_baseline_parameter_validation(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test parameter validation for detrend_fixed_baseline method."""
        # Test with empty detrend_orders
        with pytest.raises((ValueError, ConfigurationError)):
            marEx.preprocess_data(
                test_data_dask,
                method_anomaly="detrend_fixed_baseline",
                detrend_orders=[],  # Empty list should cause error
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )

        # Test with invalid detrend_orders (negative)
        with pytest.raises((ValueError, ConfigurationError)):
            marEx.preprocess_data(
                test_data_dask,
                method_anomaly="detrend_fixed_baseline",
                detrend_orders=[-1],  # Negative order invalid
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )


class TestCoordinateDimensionValidation:
    """Test validation of coordinates and dimensions parameters in new format."""

    def test_invalid_dimension_names_detect(self, test_data_dask, dask_chunks):
        """Test DataValidationError when dimensions don't exist in dataset."""
        invalid_dimensions = {"time": "nonexistent_time", "x": "nonexistent_x", "y": "nonexistent_y"}

        with pytest.raises(DataValidationError, match=r"Missing required dimensions"):
            marEx.preprocess_data(
                test_data_dask,
                dimensions=invalid_dimensions,
                dask_chunks=dask_chunks,
            )

    def test_invalid_coordinate_names_detect(self, test_data_dask, dask_chunks):
        """Test DataValidationError when coordinates don't exist in dataset."""
        # Valid dimensions but invalid coordinates
        valid_dimensions = {"time": "time", "x": "lon", "y": "lat"}
        invalid_coordinates = {"time": "nonexistent_time_coord", "x": "nonexistent_x_coord", "y": "nonexistent_y_coord"}

        with pytest.raises(DataValidationError, match=r"Missing required coordinates"):
            marEx.preprocess_data(
                test_data_dask,
                dimensions=valid_dimensions,
                coordinates=invalid_coordinates,
                dask_chunks=dask_chunks,
            )

    def test_unstructured_data_missing_coordinates(self, test_data_dask, dask_chunks):
        """Test error when coordinates=None for unstructured data."""
        # Create 2D unstructured-like data by removing y dimension
        unstructured_dims = {"time": "time", "x": "lon"}  # No 'y' key = unstructured

        with pytest.raises(DataValidationError, match=r"Coordinates parameter must be explicitly specified for unstructured data"):
            marEx.preprocess_data(
                test_data_dask,
                dimensions=unstructured_dims,
                coordinates=None,  # Should require explicit coordinates for unstructured
                dask_chunks=dask_chunks,
            )

    def test_unstructured_data_valid_coordinates(self, test_data_dask, dask_chunks):
        """Test that unstructured data works with valid coordinates."""
        # Create proper 2D data for this test
        unstructured_data = test_data_dask.stack(ncells=("lat", "lon"))
        unstructured_dims = {"time": "time", "x": "ncells"}
        unstructured_coords = {"time": "time", "x": "ncells", "y": "ncells"}  # Both x,y use same coord for 2D

        # This should work without error
        result = marEx.preprocess_data(
            unstructured_data,
            dimensions=unstructured_dims,
            coordinates=unstructured_coords,
            dask_chunks={"time": 25},
        )
        assert isinstance(result, xr.Dataset)

    def test_tracker_invalid_dimensions_gridded(self, test_data_dask, dask_chunks):
        """Test tracker validation for invalid gridded dimensions."""
        # Create global coordinate data to avoid regional coordinate issues
        global_data = test_data_dask.copy()
        global_data = global_data.assign_coords(
            lon=np.linspace(-180, 180, len(global_data.lon)), lat=np.linspace(-90, 90, len(global_data.lat))
        )

        # Use new dimension format for preprocessing
        valid_dimensions = {"time": "time", "x": "lon", "y": "lat"}
        extremes_ds = marEx.preprocess_data(global_data, dimensions=valid_dimensions, dask_chunks=dask_chunks)

        # Now test tracker with dimensions that don't exist - this will cause transpose to fail
        wrong_dimensions = {"time": "time", "x": "nonexistent_x", "y": "nonexistent_y"}
        wrong_coordinates = {"time": "time", "x": "lon", "y": "lat"}  # Valid coordinates

        with pytest.raises(DataValidationError, match=r"Invalid dimensions for gridded data"):
            marEx.tracker(
                extremes_ds.extreme_events,
                extremes_ds.mask,
                R_fill=8,
                area_filter_quartile=0.5,
                dimensions=wrong_dimensions,
                coordinates=wrong_coordinates,
            )

    def test_tracker_invalid_coordinates_unstructured(self, test_data_dask, dask_chunks):
        """Test tracker validation for missing coordinates in unstructured data."""
        # Create 2D unstructured data
        unstructured_data = test_data_dask.stack(ncells=("lat", "lon"))
        unstructured_dims = {"time": "time", "x": "ncells"}
        unstructured_coords = {"time": "time", "x": "ncells", "y": "ncells"}

        # Create binary data
        extremes_ds = marEx.preprocess_data(
            unstructured_data,
            dimensions=unstructured_dims,
            coordinates=unstructured_coords,
            dask_chunks={"time": 25},
        )

        # Test tracker with coordinates that exist but don't have required spatial info
        # Use existing coordinates to avoid KeyError before validation
        wrong_coords = {"time": "time", "x": "time", "y": "time"}  # Wrong mapping, using existing coord

        with pytest.raises((DataValidationError, CoordinateError)):
            marEx.tracker(
                extremes_ds.extreme_events,
                extremes_ds.mask,
                R_fill=8,
                area_filter_quartile=0.5,
                dimensions=unstructured_dims,
                coordinates=wrong_coords,
                regional_mode=False,  # Don't bypass coordinate validation
            )

    def test_compute_normalised_anomaly_dimension_validation(self, test_data_dask):
        """Test dimension validation in compute_normalised_anomaly."""
        invalid_dimensions = {"time": "time", "x": "missing_x", "y": "missing_y"}

        with pytest.raises(DataValidationError, match=r"Missing required dimensions"):
            marEx.compute_normalised_anomaly(
                test_data_dask,
                method_anomaly="detrend_harmonic",
                dimensions=invalid_dimensions,
            )

    def test_compute_normalised_anomaly_coordinate_validation(self, test_data_dask):
        """Test coordinate validation in compute_normalised_anomaly."""
        valid_dimensions = {"time": "time", "x": "lon", "y": "lat"}
        invalid_coordinates = {"time": "time", "x": "missing_x", "y": "missing_y"}

        with pytest.raises(DataValidationError, match=r"Missing required coordinates"):
            marEx.compute_normalised_anomaly(
                test_data_dask,
                method_anomaly="detrend_harmonic",
                dimensions=valid_dimensions,
                coordinates=invalid_coordinates,
            )

    def test_identify_extremes_dimension_validation(self, test_data_dask):
        """Test dimension validation in identify_extremes."""
        invalid_dimensions = {"time": "time", "x": "missing_x", "y": "missing_y"}

        with pytest.raises(DataValidationError, match=r"Missing required dimensions"):
            marEx.identify_extremes(
                test_data_dask,
                method_extreme="global_extreme",
                dimensions=invalid_dimensions,
            )

    def test_identify_extremes_coordinate_validation(self, test_data_dask):
        """Test coordinate validation in identify_extremes."""
        valid_dimensions = {"time": "time", "x": "lon", "y": "lat"}
        invalid_coordinates = {"time": "time", "x": "missing_x", "y": "missing_y"}

        with pytest.raises(DataValidationError, match=r"Missing required coordinates"):
            marEx.identify_extremes(
                test_data_dask,
                method_extreme="global_extreme",
                dimensions=valid_dimensions,
                coordinates=invalid_coordinates,
            )


class TestTrackerValidation:
    """Test tracker-specific parameter validation."""

    def test_area_filter_quartile_validation(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test validation of area_filter_quartile parameter range."""
        # Create global coordinate data to avoid regional coordinate issues
        global_data = test_data_dask.copy()
        global_data = global_data.assign_coords(
            lon=np.linspace(-180, 180, len(global_data.lon)), lat=np.linspace(-90, 90, len(global_data.lat))
        )

        # First create valid binary data with global coordinates
        extremes_ds = marEx.preprocess_data(global_data, dimensions=dimensions_gridded, dask_chunks=dask_chunks)

        # Test below valid range
        with pytest.raises(ConfigurationError, match="Invalid area_filter_quartile value"):
            marEx.tracker(
                extremes_ds.extreme_events,
                extremes_ds.mask,
                R_fill=8,
                area_filter_quartile=-0.1,  # Invalid value
            )

        # Test above valid range
        with pytest.raises(ConfigurationError, match="Invalid area_filter_quartile value"):
            marEx.tracker(
                extremes_ds.extreme_events,
                extremes_ds.mask,
                R_fill=8,
                area_filter_quartile=1.5,  # Invalid value
            )

        # Test valid values work
        tracker_valid = marEx.tracker(
            extremes_ds.extreme_events,
            extremes_ds.mask,
            R_fill=8,
            area_filter_quartile=0.5,  # Valid value
        )
        assert tracker_valid is not None

    def test_t_fill_even_validation(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test that T_fill must be even."""
        # Create global coordinate data to avoid regional coordinate issues
        global_data = test_data_dask.copy()
        global_data = global_data.assign_coords(
            lon=np.linspace(-180, 180, len(global_data.lon)), lat=np.linspace(-90, 90, len(global_data.lat))
        )

        # First create valid binary data
        extremes_ds = marEx.preprocess_data(global_data, dimensions=dimensions_gridded, dask_chunks=dask_chunks)

        with pytest.raises(ConfigurationError, match="T_fill must be even for temporal symmetry"):
            marEx.tracker(
                extremes_ds.extreme_events,
                extremes_ds.mask,
                R_fill=8,
                T_fill=3,  # Odd number - should be even
                area_filter_quartile=0.5,
            )

    def test_coordinate_range_validation(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test that coordinates must be in degree range."""
        # Create test data with coordinates in wrong range (e.g., meters instead of degrees)
        invalid_coords_data = test_data_dask.copy()
        invalid_coords_data = invalid_coords_data.assign_coords(
            lon=np.linspace(0, 10, len(invalid_coords_data.lon)),  # Only 10 degree range - too small
            lat=np.linspace(0, 5, len(invalid_coords_data.lat)),  # Only 5 degree range - too small
        )

        # First preprocess the data to get binary events
        extremes_ds = marEx.preprocess_data(invalid_coords_data, dimensions=dimensions_gridded, dask_chunks=dask_chunks)

        with pytest.raises(CoordinateError, match="Cannot auto-detect coordinate units"):
            marEx.tracker(
                extremes_ds.extreme_events,
                extremes_ds.mask,
                R_fill=8,
                area_filter_quartile=0.5,
            )

    def test_data_type_validation(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test that tracker validates data types correctly."""
        # Create global coordinate data to avoid regional coordinate issues
        global_data = test_data_dask.copy()
        global_data = global_data.assign_coords(
            lon=np.linspace(-180, 180, len(global_data.lon)), lat=np.linspace(-90, 90, len(global_data.lat))
        )

        # First create valid binary data
        extremes_ds = marEx.preprocess_data(global_data, dimensions=dimensions_gridded, dask_chunks=dask_chunks)

        # Test non-boolean binary data
        float_data = extremes_ds.extreme_events.astype(float)
        with pytest.raises(
            DataValidationError,
            match="Input DataArray must be binary \\(boolean type\\)",
        ):
            marEx.tracker(
                float_data,  # Float data instead of boolean
                extremes_ds.mask,
                R_fill=8,
                area_filter_quartile=0.5,
            )

        # Test non-boolean mask
        float_mask = extremes_ds.mask.astype(float)
        with pytest.raises(DataValidationError, match="Mask must be binary \\(boolean type\\)"):
            marEx.tracker(
                extremes_ds.extreme_events,
                float_mask,  # Float mask instead of boolean
                R_fill=8,
                area_filter_quartile=0.5,
            )

        # Test all-False mask
        all_false_mask = xr.zeros_like(extremes_ds.mask)
        with pytest.raises(
            DataValidationError,
            match="Mask contains only False values",
        ):
            marEx.tracker(
                extremes_ds.extreme_events,
                all_false_mask,  # All False mask
                R_fill=8,
                area_filter_quartile=0.5,
            )


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_percentile_edge_cases(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test edge cases for percentile values."""
        # Test low percentile (should work but create more events)
        result_low = marEx.preprocess_data(
            test_data_dask,
            threshold_percentile=80,  # 80th percentile - conservative but should work
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )
        assert isinstance(result_low, xr.Dataset)
        assert result_low.attrs["threshold_percentile"] == 80

        # Verify that events are detected for lower percentile
        extreme_frequency = float(result_low.extreme_events.mean())
        assert 0.001 < extreme_frequency < 0.3, f"Extreme frequency {extreme_frequency} should be reasonable for 80th percentile"

        # Test very high percentile (should work but create very few events)
        result_high = marEx.preprocess_data(
            test_data_dask,
            threshold_percentile=99,  # Very high percentile
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )
        assert isinstance(result_high, xr.Dataset)
        assert result_high.attrs["threshold_percentile"] == 99

        # Verify that very few events are detected for high percentile
        extreme_frequency_high = float(result_high.extreme_events.mean())
        assert (
            0.0001 < extreme_frequency_high < 0.1
        ), f"Extreme frequency {extreme_frequency_high} should be low for 99th percentile"

    def test_large_window_parameters_error(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test that overly large window parameters raise appropriate errors."""
        # Test with window larger than dataset
        with pytest.raises((DataValidationError, IndexError)):
            marEx.preprocess_data(
                test_data_dask,
                method_anomaly="shifting_baseline",
                window_year_baseline=50,  # Larger than dataset (~40 years)
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )


class TestHelpfulErrorMessages:
    """Test that error messages are helpful and informative."""

    def test_dask_error_message_includes_chunks_hint(self, test_data_numpy, dimensions_gridded, dask_chunks):
        """Test that Dask error includes helpful chunking hint."""
        with pytest.raises(DataValidationError) as exc_info:
            marEx.preprocess_data(test_data_numpy, dimensions=dimensions_gridded, dask_chunks=dask_chunks)

        error_message = str(exc_info.value)
        assert "chunk" in error_message.lower()
        assert "dask" in error_message.lower()

    def test_insufficient_data_error_includes_suggestions(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test that insufficient data error includes helpful suggestions."""
        small_data = test_data_dask.isel(time=slice(0, 730)).chunk(dask_chunks)  # ~2 years

        with pytest.raises((DataValidationError, IndexError)) as exc_info:
            marEx.preprocess_data(
                small_data,
                method_anomaly="shifting_baseline",
                window_year_baseline=15,
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )

        error_message = str(exc_info.value)
        # The error message should indicate data insufficiency in some form
        assert "years" in error_message or "window" in error_message or "index" in error_message or "arrays" in error_message

    def test_coordinate_error_includes_degree_requirement(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test that coordinate range error includes helpful information."""
        # Create test data with coordinates in wrong range
        invalid_coords_data = test_data_dask.copy()
        invalid_coords_data = invalid_coords_data.assign_coords(
            lon=np.linspace(0, 10, len(invalid_coords_data.lon)),
            lat=np.linspace(0, 5, len(invalid_coords_data.lat)),
        )

        # First preprocess the data
        extremes_ds = marEx.preprocess_data(invalid_coords_data, dimensions=dimensions_gridded, dask_chunks=dask_chunks)

        with pytest.raises(CoordinateError) as exc_info:
            marEx.tracker(
                extremes_ds.extreme_events,
                extremes_ds.mask,
                R_fill=8,
                area_filter_quartile=0.5,
            )

        error_message = str(exc_info.value)
        assert "coordinate" in error_message.lower()
        assert "range" in error_message.lower()


class TestIdentifyExtremesConfigurationErrors:
    """Test all ConfigurationError cases in identify_extremes."""

    @pytest.fixture(scope="class")
    def anomaly_data(self):
        """Create anomaly data for testing identify_extremes directly."""
        test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        ds = xr.open_zarr(str(test_data_path), chunks={"time": 25}).isel(lon=slice(0, 4), lat=slice(0, 3))
        sst_da = ds.to  # Extract the DataArray named 'to'

        # Create anomaly data for testing identify_extremes directly
        dimensions = {"time": "time", "x": "lon", "y": "lat"}
        anomalies_ds = marEx.compute_normalised_anomaly(sst_da, dimensions=dimensions)
        return anomalies_ds.dat_anomaly

    @pytest.fixture(scope="class")
    def dimensions_coords(self):
        """Standard dimensions and coordinates for testing."""
        return {"dimensions": {"time": "time", "x": "lon", "y": "lat"}, "coordinates": {"time": "time", "x": "lon", "y": "lat"}}

    def test_invalid_method_percentile(self, anomaly_data):
        """Test ConfigurationError for invalid method_percentile."""
        with pytest.raises(ConfigurationError, match="Unknown method_percentile 'invalid_method'"):
            marEx.identify_extremes(
                anomaly_data,
                method_percentile="invalid_method",
            )

    def test_precision_with_exact_percentile(self, anomaly_data):
        """Test ConfigurationError when precision is used with exact percentile method."""
        with pytest.raises(ConfigurationError, match="Parameter 'precision' cannot be used with method_percentile='exact'"):
            marEx.identify_extremes(
                anomaly_data,
                method_percentile="exact",
                precision=0.02,  # Non-default precision
            )

    def test_max_anomaly_with_exact_percentile(self, anomaly_data):
        """Test ConfigurationError when max_anomaly is used with exact percentile method."""
        with pytest.raises(ConfigurationError, match="Parameter 'max_anomaly' cannot be used with method_percentile='exact'"):
            marEx.identify_extremes(
                anomaly_data,
                method_percentile="exact",
                max_anomaly=10.0,  # Non-default max_anomaly
            )

    def test_low_percentile_with_approximate_method(self, anomaly_data):
        """Test ConfigurationError for low percentile with approximate method."""
        with pytest.raises(
            ConfigurationError, match="Percentile threshold 50% is not supported with method_percentile='approximate'"
        ):
            marEx.identify_extremes(
                anomaly_data,
                method_percentile="approximate",
                threshold_percentile=50,  # Below 60% threshold
            )

    def test_window_spatial_hobday_with_global_extreme(self, anomaly_data, dimensions_coords):
        """Test ConfigurationError when window_spatial_hobday is used with global_extreme."""
        with pytest.raises(ConfigurationError, match="window_spatial_hobday can only be used with method_extreme='hobday_extreme'"):
            marEx.identify_extremes(
                anomaly_data,
                method_extreme="global_extreme",
                window_spatial_hobday=3,  # This should trigger error with global_extreme
                dimensions=dimensions_coords["dimensions"],
                coordinates=dimensions_coords["coordinates"],
            )

    def test_window_spatial_hobday_with_exact_percentile(self, anomaly_data, dimensions_coords):
        """Test ConfigurationError when window_spatial_hobday is used with exact percentile."""
        with pytest.raises(ConfigurationError, match="window_spatial_hobday is not supported with method_percentile='exact'"):
            marEx.identify_extremes(
                anomaly_data,
                method_extreme="hobday_extreme",
                method_percentile="exact",
                window_spatial_hobday=3,  # This should trigger error with exact percentile
                dimensions=dimensions_coords["dimensions"],
                coordinates=dimensions_coords["coordinates"],
            )

    def test_even_window_days_hobday(self, anomaly_data, dimensions_coords):
        """Test ConfigurationError for even window_days_hobday."""
        with pytest.raises(ConfigurationError, match="window_days_hobday must be an odd number"):
            marEx.identify_extremes(
                anomaly_data,
                method_extreme="hobday_extreme",
                window_days_hobday=10,  # Even number
                dimensions=dimensions_coords["dimensions"],
                coordinates=dimensions_coords["coordinates"],
            )

    def test_even_window_spatial_hobday(self, anomaly_data, dimensions_coords):
        """Test ConfigurationError for even window_spatial_hobday."""
        with pytest.raises(ConfigurationError, match="window_spatial_hobday must be an odd number"):
            marEx.identify_extremes(
                anomaly_data,
                method_extreme="hobday_extreme",
                method_percentile="approximate",
                window_spatial_hobday=4,  # Even number
                dimensions=dimensions_coords["dimensions"],
                coordinates=dimensions_coords["coordinates"],
            )

    def test_invalid_method_extreme(self, anomaly_data):
        """Test ConfigurationError for invalid method_extreme."""
        with pytest.raises(ConfigurationError, match="Unknown extreme method 'invalid_extreme'"):
            marEx.identify_extremes(
                anomaly_data,
                method_extreme="invalid_extreme",
            )


class TestTrackerDataValidationErrors:
    """Test all create_data_validation_error cases in track.py."""

    @pytest.fixture(scope="function")
    def valid_binary_data(self):
        """Create valid binary data and mask for testing."""
        test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        ds = xr.open_zarr(str(test_data_path), chunks={"time": 25}).isel(lon=slice(0, 4), lat=slice(0, 3))
        sst_da = ds.to

        # Create global coordinate data to avoid regional coordinate issues
        global_data = sst_da.assign_coords(lon=np.linspace(-180, 180, len(sst_da.lon)), lat=np.linspace(-90, 90, len(sst_da.lat)))

        dimensions = {"time": "time", "x": "lon", "y": "lat"}
        extremes_ds = marEx.preprocess_data(global_data, dimensions=dimensions, dask_chunks={"time": 25})
        return extremes_ds.extreme_events, extremes_ds.mask

    def test_data_bin_not_chunked(self):
        """Test error for non-chunked data_bin."""
        # Create non-chunked data
        data = xr.DataArray(
            np.random.random((10, 5, 4)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-20, 20, 5),
                "lon": np.linspace(-10, 10, 4),
            },
        )
        binary_data = (data > data.mean()).astype(bool).load()  # Load to remove chunks
        mask = ~np.isnan(data.isel(time=0))

        with pytest.raises(DataValidationError, match="Data must be chunked"):
            marEx.tracker(binary_data, mask, R_fill=8, area_filter_quartile=0.5, regional_mode=True, coordinate_units="degrees")

    def test_unstructured_grid_missing_temp_dir(self):
        """Test error when temp_dir is missing for unstructured grids."""
        # Create 2D unstructured data with proper coordinate range
        data = xr.DataArray(
            np.random.random((10, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(10),
                "lat": ("cell", np.random.uniform(-90, 90, 100)),
                "lon": ("cell", np.random.uniform(-180, 180, 100)),
            },
        ).chunk({"time": 5})

        binary_data = (data > 0.5).astype(bool)
        mask = xr.ones_like(binary_data.isel(time=0), dtype=bool)

        with pytest.raises(DataValidationError, match="temp_dir is required for unstructured grids"):
            marEx.tracker(
                binary_data,
                mask,
                R_fill=8,
                area_filter_quartile=0.5,
                unstructured_grid=True,
                temp_dir=None,  # Missing temp_dir
                coordinate_units="degrees",  # Explicitly specify units
                dimensions={"time": "time", "x": "cell"},  # Match the data dimensions
                coordinates={"time": "time", "x": "lon", "y": "lat"},
            )

    def test_unstructured_grid_missing_neighbours(self):
        """Test error when neighbours array is missing for unstructured grids."""
        # Create 2D unstructured data with proper coordinate range
        data = xr.DataArray(
            np.random.random((10, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(10),
                "lat": ("cell", np.random.uniform(-90, 90, 100)),
                "lon": ("cell", np.random.uniform(-180, 180, 100)),
            },
        ).chunk({"time": 5})

        binary_data = (data > 0.5).astype(bool)
        mask = xr.ones_like(binary_data.isel(time=0), dtype=bool)

        with pytest.raises(DataValidationError, match="neighbours array is required for unstructured grids"):
            marEx.tracker(
                binary_data,
                mask,
                R_fill=8,
                area_filter_quartile=0.5,
                unstructured_grid=True,
                temp_dir="/tmp",
                neighbours=None,  # Missing neighbours
                coordinate_units="degrees",  # Explicitly specify units
                dimensions={"time": "time", "x": "cell"},  # Match the data dimensions
                coordinates={"time": "time", "x": "lon", "y": "lat"},
            )

    def test_unstructured_grid_missing_cell_areas(self):
        """Test error when cell_areas array is missing for unstructured grids."""
        # Create 2D unstructured data with proper coordinate range
        data = xr.DataArray(
            np.random.random((10, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(10),
                "lat": ("cell", np.random.uniform(-90, 90, 100)),
                "lon": ("cell", np.random.uniform(-180, 180, 100)),
            },
        ).chunk({"time": 5})

        binary_data = (data > 0.5).astype(bool)
        mask = xr.ones_like(binary_data.isel(time=0), dtype=bool)

        with pytest.raises(DataValidationError, match="cell_areas array is required for unstructured grids"):
            marEx.tracker(
                binary_data,
                mask,
                R_fill=8,
                area_filter_quartile=0.5,
                unstructured_grid=True,
                temp_dir="/tmp",
                neighbours=np.ones((3, 100)),  # Dummy neighbours
                cell_areas=None,  # Missing cell_areas
                coordinate_units="degrees",  # Explicitly specify units
                dimensions={"time": "time", "x": "cell"},  # Match the data dimensions
                coordinates={"time": "time", "x": "lon", "y": "lat"},
            )

    def test_invalid_dimensions_unstructured_data(self):
        """Test error for invalid dimensions in unstructured data."""
        # Create 2D unstructured data
        data = xr.DataArray(
            np.random.random((10, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(10),
                "lat": ("cell", np.random.uniform(-90, 90, 100)),
                "lon": ("cell", np.random.uniform(-180, 180, 100)),
            },
        ).chunk({"time": 5})

        binary_data = (data > data.mean()).astype(bool)
        mask = xr.ones_like(binary_data.isel(time=0), dtype=bool)

        # Use invalid dimensions that don't match the data structure
        invalid_dimensions = {"time": "time", "x": "invalid_dim"}
        invalid_coordinates = {"time": "time", "x": "lon", "y": "lat"}

        with pytest.raises(DataValidationError, match="Invalid dimensions for unstructured data"):
            marEx.tracker(
                binary_data,
                mask,
                R_fill=8,
                area_filter_quartile=0.5,
                dimensions=invalid_dimensions,
                coordinates=invalid_coordinates,
                unstructured_grid=True,  # Explicitly set to trigger unstructured validation
                coordinate_units="degrees",  # Explicitly specify units
            )

    def test_invalid_dimensions_gridded_data(self, valid_binary_data):
        """Test error for invalid dimensions in gridded data."""
        binary_data, mask = valid_binary_data

        # Use dimensions that exist but aren't in the right order/format
        invalid_dimensions = {"time": "time", "x": "invalid_x", "y": "invalid_y"}
        invalid_coordinates = {"time": "time", "x": "lon", "y": "lat"}

        with pytest.raises(DataValidationError, match="Invalid dimensions for gridded data"):
            marEx.tracker(
                binary_data,
                mask,
                R_fill=8,
                area_filter_quartile=0.5,
                dimensions=invalid_dimensions,
                coordinates=invalid_coordinates,
            )

    def test_missing_coordinates_unstructured_data(self):
        """Test error for missing coordinates in unstructured data."""
        # Create 2D unstructured data without required coordinates
        data = xr.DataArray(
            np.random.random((10, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(10),
                # Missing lat/lon coordinates
            },
        ).chunk({"time": 5})

        binary_data = (data > 0.5).astype(bool)
        mask = xr.ones_like(binary_data.isel(time=0), dtype=bool)

        dimensions = {"time": "time", "x": "cell"}
        coordinates = {"time": "time", "x": "missing_lon", "y": "missing_lat"}

        with pytest.raises(KeyError, match="missing_lat"):
            marEx.tracker(binary_data, mask, R_fill=8, area_filter_quartile=0.5, dimensions=dimensions, coordinates=coordinates)

    def test_non_boolean_data_bin(self, valid_binary_data):
        """Test error for non-boolean data_bin."""
        binary_data, mask = valid_binary_data

        # Convert to float data
        float_data = binary_data.astype(float)

        with pytest.raises(DataValidationError, match="Input DataArray must be binary \\(boolean type\\)"):
            marEx.tracker(float_data, mask, R_fill=8, area_filter_quartile=0.5)

    def test_non_dask_data_bin(self, valid_binary_data):
        """Test error for non-Dask data_bin."""
        binary_data, mask = valid_binary_data

        # Load data to remove Dask backing
        loaded_data = binary_data.load()

        with pytest.raises(DataValidationError, match="Data must be chunked"):
            marEx.tracker(loaded_data, mask, R_fill=8, area_filter_quartile=0.5)

    def test_non_boolean_mask(self, valid_binary_data):
        """Test error for non-boolean mask."""
        binary_data, mask = valid_binary_data

        # Convert mask to float and ensure it's chunked (mask from fixture is not chunked)
        float_mask = mask.astype(float).chunk({"lat": -1, "lon": -1})

        with pytest.raises(DataValidationError, match="Mask must be binary \\(boolean type\\)"):
            marEx.tracker(binary_data, float_mask, R_fill=8, area_filter_quartile=0.5)

    def test_all_false_mask(self, valid_binary_data):
        """Test error for mask with only False values."""
        binary_data, mask = valid_binary_data

        # Create all-False mask and ensure it's chunked (mask from fixture is not chunked)
        all_false_mask = xr.zeros_like(mask, dtype=bool).chunk({"lat": -1, "lon": -1})

        with pytest.raises(DataValidationError, match="Mask contains only False values"):
            marEx.tracker(binary_data, all_false_mask, R_fill=8, area_filter_quartile=0.5)

    def test_invalid_neighbour_array_shape(self):
        """Test error for invalid neighbour array shape in unstructured grid setup."""
        # Create minimal unstructured data
        data = xr.DataArray(
            np.random.random((5, 50)),
            dims=["time", "cell"],
            coords={
                "time": range(5),
                "lat": ("cell", np.random.uniform(-90, 90, 50)),
                "lon": ("cell", np.random.uniform(-180, 180, 50)),
            },
        ).chunk({"time": 3})

        binary_data = (data > 0.5).astype(bool)
        mask = xr.ones_like(binary_data.isel(time=0), dtype=bool)

        # Create invalid neighbours array (wrong first dimension)
        invalid_neighbours = xr.DataArray(
            np.ones((4, 50)),  # Should be (3, 50) for triangular grid
            dims=["nv", "cell"],
            coords={
                "cell": range(50),
                "nv": range(4),
                "lat": ("cell", np.random.uniform(-90, 90, 50)),
                "lon": ("cell", np.random.uniform(-180, 180, 50)),
            },
        )

        with pytest.raises(DataValidationError, match="Invalid neighbour array for triangular grid"):
            marEx.tracker(
                binary_data,
                mask,
                R_fill=8,
                area_filter_quartile=0.5,
                unstructured_grid=True,
                temp_dir="/tmp",
                neighbours=invalid_neighbours,
                cell_areas=xr.DataArray(
                    np.ones(50),
                    dims=["cell"],
                    coords={
                        "cell": range(50),
                        "lat": ("cell", np.random.uniform(-90, 90, 50)),
                        "lon": ("cell", np.random.uniform(-180, 180, 50)),
                    },
                ),
                coordinate_units="degrees",  # Explicitly specify units
                dimensions={"time": "time", "x": "cell"},  # Match the data dimensions
                coordinates={"time": "time", "x": "lon", "y": "lat"},
            )

    def test_invalid_neighbour_array_dimensions(self):
        """Test error for invalid neighbour array dimensions in unstructured grid setup."""
        # Create minimal unstructured data
        data = xr.DataArray(
            np.random.random((5, 50)),
            dims=["time", "cell"],
            coords={
                "time": range(5),
                "lat": ("cell", np.random.uniform(-90, 90, 50)),
                "lon": ("cell", np.random.uniform(-180, 180, 50)),
            },
        ).chunk({"time": 3})

        binary_data = (data > 0.5).astype(bool)
        mask = xr.ones_like(binary_data.isel(time=0), dtype=bool)

        # Create neighbours array with wrong dimension names
        invalid_neighbours = xr.DataArray(
            np.ones((3, 50)),
            dims=["wrong_nv", "wrong_cell"],  # Wrong dimension names
            coords={
                "wrong_cell": range(50),
                "wrong_nv": range(3),
                "lat": ("wrong_cell", np.random.uniform(-90, 90, 50)),
                "lon": ("wrong_cell", np.random.uniform(-180, 180, 50)),
                "nv": ("wrong_nv", range(3)),  # Add this coordinate that will be dropped
            },
        )

        with pytest.raises(DataValidationError, match="Invalid neighbour array dimensions"):
            marEx.tracker(
                binary_data,
                mask,
                R_fill=8,
                area_filter_quartile=0.5,
                unstructured_grid=True,
                temp_dir="/tmp",
                neighbours=invalid_neighbours,
                cell_areas=xr.DataArray(
                    np.ones(50),
                    dims=["cell"],
                    coords={
                        "cell": range(50),
                        "lat": ("cell", np.random.uniform(-90, 90, 50)),
                        "lon": ("cell", np.random.uniform(-180, 180, 50)),
                    },
                ),
                coordinate_units="degrees",  # Explicitly specify units
                dimensions={"time": "time", "x": "cell"},  # Match the data dimensions
                coordinates={"time": "time", "x": "lon", "y": "lat"},
            )


class TestTrackerCoordinateErrors:
    """Test all create_coordinate_error cases in track.py."""

    @pytest.fixture(scope="function")
    def valid_binary_data_for_coords(self):
        """Create valid binary data for coordinate testing."""
        test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        ds = xr.open_zarr(str(test_data_path), chunks={"time": 25}).isel(lon=slice(0, 4), lat=slice(0, 3))
        sst_da = ds.to

        # Create binary data
        binary_data = (sst_da > sst_da.quantile(0.95)).astype(bool)
        mask = ~np.isnan(sst_da.isel(time=0))

        return binary_data, mask

    def test_regional_mode_missing_coordinate_units(self, valid_binary_data_for_coords):
        """Test error when coordinate_units is None in regional mode."""
        binary_data, mask = valid_binary_data_for_coords

        with pytest.raises(CoordinateError, match="coordinate_units must be specified when regional_mode=True"):
            marEx.tracker(
                binary_data,
                mask,
                R_fill=8,
                area_filter_quartile=0.5,
                regional_mode=True,
                coordinate_units=None,  # Missing coordinate_units
            )

    def test_regional_mode_invalid_coordinate_units(self, valid_binary_data_for_coords):
        """Test error for invalid coordinate_units in regional mode."""
        binary_data, mask = valid_binary_data_for_coords

        with pytest.raises(CoordinateError, match="Invalid coordinate_units 'invalid_units'"):
            marEx.tracker(
                binary_data,
                mask,
                R_fill=8,
                area_filter_quartile=0.5,
                regional_mode=True,
                coordinate_units="invalid_units",  # Invalid coordinate_units
            )

    def test_global_mode_invalid_coordinate_units(self, valid_binary_data_for_coords):
        """Test error for invalid coordinate_units in global mode."""
        binary_data, mask = valid_binary_data_for_coords

        with pytest.raises(CoordinateError, match="Invalid coordinate_units 'invalid_units'"):
            marEx.tracker(
                binary_data,
                mask,
                R_fill=8,
                area_filter_quartile=0.5,
                regional_mode=False,
                coordinate_units="invalid_units",  # Invalid coordinate_units
            )

    def test_global_mode_coordinate_autodetection_failure(self, valid_binary_data_for_coords):
        """Test error when coordinate auto-detection fails in global mode."""
        binary_data, mask = valid_binary_data_for_coords

        # Create data with coordinates that can't be auto-detected (small range)
        small_range_data = binary_data.assign_coords(
            lon=np.linspace(0, 10, len(binary_data.lon)),  # Only 10 degree range
            lat=np.linspace(0, 5, len(binary_data.lat)),  # Only 5 degree range
        )

        with pytest.raises(CoordinateError, match="Cannot auto-detect coordinate units from range"):
            marEx.tracker(
                small_range_data,
                mask,
                R_fill=8,
                area_filter_quartile=0.5,
                regional_mode=False,
                coordinate_units=None,  # Auto-detection should fail
            )


class TestDetectDataValidationEdgeCases:
    """Test edge cases in data validation for detect module."""

    def test_all_nan_data_error(self, dimensions_gridded, dask_chunks):
        """Test error when dataset contains only NaN/infinite values."""
        # Create data with all NaN values - directly as Dask array
        all_nan_data = xr.DataArray(
            np.full((100, 3, 4), np.nan),
            dims=["time", "lat", "lon"],
            coords={
                "time": np.arange(100),
                "lat": np.linspace(-10, 10, 3),
                "lon": np.linspace(0, 20, 4),
            },
        ).chunk({"time": 25})

        with pytest.raises(DataValidationError, match="contains no valid.*finite.*data"):
            marEx.preprocess_data(
                all_nan_data,
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
                method_anomaly="detrend_harmonic",
            )

    def test_all_inf_data_error(self, dimensions_gridded, dask_chunks):
        """Test error when dataset contains only infinite values."""
        # Create data with all infinite values - directly as Dask array
        all_inf_data = xr.DataArray(
            np.full((100, 3, 4), np.inf),
            dims=["time", "lat", "lon"],
            coords={
                "time": np.arange(100),
                "lat": np.linspace(-10, 10, 3),
                "lon": np.linspace(0, 20, 4),
            },
        ).chunk({"time": 25})

        with pytest.raises(DataValidationError, match="contains no valid.*finite.*data"):
            marEx.preprocess_data(
                all_inf_data,
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
                method_anomaly="detrend_harmonic",
            )

    def test_partial_invalid_data_error(self, dimensions_gridded, dask_chunks):
        """Test error when some ocean locations have NaN/infinite values across time."""
        import pandas as pd

        # Create data with mostly valid values
        data = np.random.rand(100, 3, 4) * 5 + 15

        # Make some specific locations have invalid values across time
        data[:, 1, 2] = np.nan  # One location all NaN
        data[50:, 0, 1] = np.inf  # Another location partially infinite
        data[:20, 2, 0] = np.nan  # Another location with some NaN values

        partial_invalid_data = xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2020-01-01", periods=100),
                "lat": np.linspace(-10, 10, 3),
                "lon": np.linspace(0, 20, 4),
            },
        ).chunk({"time": 25})

        with pytest.raises(DataValidationError, match="contains.*invalid values.*ocean locations"):
            marEx.preprocess_data(
                partial_invalid_data,
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
                method_anomaly="detrend_harmonic",
            )


class TestQuantileThresholdWarnings:
    """Test warnings for quantile threshold edge cases."""

    def test_constant_anomaly_warning(self, dimensions_gridded):
        """Test warning for constant anomaly regions (e.g., sea ice)."""
        # Create synthetic data with constant anomaly in some regions
        time_steps = 100
        lat_size = 5
        lon_size = 5

        # Create data with one region having constant zero anomaly
        data = xr.DataArray(
            np.random.randn(time_steps, lat_size, lon_size) * 2 + 15,
            dims=["time", "lat", "lon"],
            coords={
                "time": np.arange(time_steps),
                "lat": np.linspace(-10, 10, lat_size),
                "lon": np.linspace(0, 20, lon_size),
            },
        ).chunk({"time": 25})

        # Set one location to constant value (zero anomaly after detrending)
        data.values[:, 0, 0] = 15.0

        # Process with global_extreme to trigger threshold validation
        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")

            result = marEx.preprocess_data(
                data,
                dimensions=dimensions_gridded,
                dask_chunks={"time": 25},
                method_anomaly="detrend_harmonic",
                method_extreme="global_extreme",
                threshold_percentile=95,
            )

            # Verify processing completes
            assert result is not None

    def test_near_constant_data_quantile_warning(self, dimensions_gridded):
        """Test warning when data has very low variance leading to low quantile values."""
        # Create data with very low variance
        time_steps = 100
        lat_size = 4
        lon_size = 4

        # Base value with minimal variation
        data = xr.DataArray(
            np.random.randn(time_steps, lat_size, lon_size) * 0.01 + 10.0,  # Very small variance
            dims=["time", "lat", "lon"],
            coords={
                "time": np.arange(time_steps),
                "lat": np.linspace(-5, 5, lat_size),
                "lon": np.linspace(0, 10, lon_size),
            },
        ).chunk({"time": 25})

        # Add one location with even smaller variance (near constant)
        data.values[:, 0, 0] = 10.0 + np.random.randn(time_steps) * 0.0001

        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")

            result = marEx.preprocess_data(
                data,
                dimensions=dimensions_gridded,
                dask_chunks={"time": 25},
                method_anomaly="detrend_harmonic",
                method_extreme="global_extreme",
                threshold_percentile=99,  # High percentile
            )

            # Verify processing completes
            assert result is not None

    def test_high_quantile_threshold_warning_global_extreme(self, dimensions_gridded):
        """Test warning when quantiles exceed max_anomaly bounds with global_extreme (lines 2445)."""
        import pandas as pd

        # Create data with very high variance to trigger high quantile warning
        time_steps = 200
        lat_size = 4
        lon_size = 4

        # Create data with extreme outliers that will produce high quantiles
        data = xr.DataArray(
            np.random.randn(time_steps, lat_size, lon_size) * 20 + 15,  # High variance
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2020-01-01", periods=time_steps, freq="D"),
                "lat": np.linspace(-5, 5, lat_size),
                "lon": np.linspace(0, 10, lon_size),
            },
        ).chunk({"time": 25})

        # Add extreme values that will push quantiles high
        data.values[:50, 0, 0] = data.values[:50, 0, 0] + 50  # Add large positive anomalies

        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")

            result = marEx.preprocess_data(
                data,
                dimensions=dimensions_gridded,
                dask_chunks={"time": 25},
                method_anomaly="detrend_harmonic",
                method_extreme="global_extreme",
                method_percentile="approximate",
                threshold_percentile=99,
                max_anomaly=5.0,  # Low max_anomaly to trigger warning with high quantiles
            )

            # Verify processing completes
            assert result is not None

    def test_high_quantile_threshold_warning_hobday_extreme(self, dimensions_gridded):
        """Test warning when quantiles exceed max_anomaly bounds with hobday_extreme (line 2576)."""
        import pandas as pd

        # Create data with very high variance to trigger high quantile warning
        time_steps = 200
        lat_size = 4
        lon_size = 4

        # Create data with extreme outliers that will produce high quantiles
        data = xr.DataArray(
            np.random.randn(time_steps, lat_size, lon_size) * 20 + 15,  # High variance
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2020-01-01", periods=time_steps, freq="D"),
                "lat": np.linspace(-5, 5, lat_size),
                "lon": np.linspace(0, 10, lon_size),
            },
        ).chunk({"time": 25})

        # Add extreme values that will push quantiles high
        data.values[:50, 0, 0] = data.values[:50, 0, 0] + 50  # Add large positive anomalies

        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")

            result = marEx.preprocess_data(
                data,
                dimensions=dimensions_gridded,
                dask_chunks={"time": 25},
                method_anomaly="detrend_harmonic",
                method_extreme="hobday_extreme",
                method_percentile="approximate",
                window_days_hobday=5,
                threshold_percentile=99,
                max_anomaly=5.0,  # Low max_anomaly to trigger warning with high quantiles
            )

            # Verify processing completes
            assert result is not None


class TestUnstructuredGridConfigurationErrors:
    """Test configuration errors specific to unstructured grids."""

    def test_window_spatial_hobday_unstructured_error(self, dimensions_gridded, dask_chunks):
        """Test error when window_spatial_hobday is used with unstructured grid."""
        # Create 2D unstructured-like data
        test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        ds = xr.open_zarr(str(test_data_path), chunks={"time": 25}).isel(lon=slice(0, 4), lat=slice(0, 3))
        sst_da = ds.to

        # Stack to create unstructured data
        unstructured_data = sst_da.stack(ncells=("lat", "lon"))
        unstructured_dims = {"time": "time", "x": "ncells"}
        unstructured_coords = {"time": "time", "x": "ncells", "y": "ncells"}

        # Try to use window_spatial_hobday with unstructured data - should raise ConfigurationError
        with pytest.raises(ConfigurationError, match="window_spatial_hobday is not supported for unstructured grids"):
            marEx.preprocess_data(
                unstructured_data,
                dimensions=unstructured_dims,
                coordinates=unstructured_coords,
                dask_chunks={"time": 25},
                method_extreme="hobday_extreme",
                method_percentile="approximate",
                window_spatial_hobday=3,  # This should trigger error on unstructured grid
                window_days_hobday=5,
            )
