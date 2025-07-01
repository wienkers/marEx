"""
Error Handling Tests for marEx Package

Tests for proper error handling and validation across the marEx package.
This includes testing for common user mistakes and ensuring helpful error messages.
"""

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import (
    ConfigurationError,
    CoordinateError,
    DataValidationError,
    ProcessingError,
    TrackingError,
)


@pytest.fixture(scope="module")
def test_data_dask():
    """Load test data as Dask-backed DataArray."""
    test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
    ds = xr.open_zarr(str(test_data_path), chunks={"time": 25})
    return ds.to


@pytest.fixture(scope="module")
def test_data_numpy():
    """Load test data as numpy-backed DataArray."""
    test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
    ds = xr.open_zarr(str(test_data_path), chunks={"time": 25})
    return ds.to.load()


@pytest.fixture(scope="module")
def dimensions_gridded():
    """Standard dimensions for gridded data."""
    return {"time": "time", "xdim": "lon", "ydim": "lat"}


@pytest.fixture(scope="module")
def dask_chunks():
    """Standard dask chunks."""
    return {"time": 25}


class TestNonDaskInputValidation:
    """Test error handling for non-Dask input arrays."""

    def test_preprocess_data_non_dask_input(
        self, test_data_numpy, dimensions_gridded, dask_chunks
    ):
        """Test that preprocess_data raises DataValidationError for non-Dask inputs."""
        with pytest.raises(
            DataValidationError, match=r"Input DataArray must be Dask-backed"
        ):
            marEx.preprocess_data(
                test_data_numpy,  # Non-Dask array
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )

    def test_compute_normalised_anomaly_non_dask_input(
        self, test_data_numpy, dimensions_gridded
    ):
        """Test that compute_normalised_anomaly raises error for non-Dask inputs."""
        # This will fail with TypeError when trying to access .chunks on non-Dask array
        with pytest.raises(TypeError, match=r"'NoneType' object is not subscriptable"):
            marEx.compute_normalised_anomaly(
                test_data_numpy,  # Non-Dask array
                method_anomaly="detrended_baseline",
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

    def test_valid_method_combinations(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
        """Test that valid method combinations work correctly."""
        # Test detrended_baseline + global_extreme (default combination)
        result1 = marEx.preprocess_data(
            test_data_dask,
            method_anomaly="detrended_baseline",
            method_extreme="global_extreme",
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )
        assert isinstance(result1, xr.Dataset)
        assert result1.attrs["method_anomaly"] == "detrended_baseline"
        assert result1.attrs["method_extreme"] == "global_extreme"

        # Test shifting_baseline + hobday_extreme
        result2 = marEx.preprocess_data(
            test_data_dask,
            method_anomaly="shifting_baseline",
            method_extreme="hobday_extreme",
            window_year_baseline=5,  # Reduced for test data
            window_days_hobday=5,  # Reduced for test data
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )
        assert isinstance(result2, xr.Dataset)
        assert result2.attrs["method_anomaly"] == "shifting_baseline"
        assert result2.attrs["method_extreme"] == "hobday_extreme"


class TestInsufficientDataValidation:
    """Test error handling for insufficient data scenarios."""

    def test_shifting_baseline_insufficient_data(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
        """Test error for insufficient data with shifting_baseline method."""
        # Create a small dataset with only 2 years of data
        small_data = test_data_dask.isel(time=slice(0, 730)).chunk(
            dask_chunks
        )  # ~2 years, chunked

        with pytest.raises((DataValidationError, IndexError)):
            marEx.preprocess_data(
                small_data,
                method_anomaly="shifting_baseline",
                method_extreme="hobday_extreme",
                window_year_baseline=15,  # Requires 15 years but only have 2
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )

    def test_shifting_baseline_custom_window_insufficient_data(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
        """Test error for insufficient data with custom window_year_baseline."""
        # Create dataset with 3 years but require 5 years
        small_data = test_data_dask.isel(time=slice(0, 1095)).chunk(
            dask_chunks
        )  # ~3 years, chunked

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
        # Test with missing time dimension
        incomplete_dimensions = {"xdim": "lon", "ydim": "lat"}  # Missing 'time'

        with pytest.raises(KeyError, match=r"'time'"):
            marEx.preprocess_data(
                test_data_dask,
                dimensions=incomplete_dimensions,  # Missing time dimension
                dask_chunks=dask_chunks,
            )

    def test_wrong_dimension_names(self, test_data_dask, dask_chunks):
        """Test error when dimension names don't match actual data dimensions."""
        wrong_dimensions = {
            "time": "time",
            "xdim": "wrong_lon",  # Wrong name - should be 'lon'
            "ydim": "wrong_lat",  # Wrong name - should be 'lat'
        }

        # This should raise an error when trying to access non-existent dimensions
        with pytest.raises((KeyError, ValueError)):
            marEx.preprocess_data(
                test_data_dask, dimensions=wrong_dimensions, dask_chunks=dask_chunks
            )


class TestTrackerValidation:
    """Test tracker-specific parameter validation."""

    def test_area_filter_quartile_validation(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
        """Test validation of area_filter_quartile parameter range."""
        # First create valid binary data
        extremes_ds = marEx.preprocess_data(
            test_data_dask, dimensions=dimensions_gridded, dask_chunks=dask_chunks
        )

        # Test below valid range
        with pytest.raises(
            ConfigurationError, match="Invalid area_filter_quartile value"
        ):
            marEx.tracker(
                extremes_ds.extreme_events,
                extremes_ds.mask,
                R_fill=8,
                area_filter_quartile=-0.1,  # Invalid value
                regional_mode=True,  # Bypass coordinate validation
                coordinate_units="degrees",
            )

        # Test above valid range
        with pytest.raises(
            ConfigurationError, match="Invalid area_filter_quartile value"
        ):
            marEx.tracker(
                extremes_ds.extreme_events,
                extremes_ds.mask,
                R_fill=8,
                area_filter_quartile=1.5,  # Invalid value
                regional_mode=True,  # Bypass coordinate validation
                coordinate_units="degrees",
            )

        # Test valid values work
        tracker_valid = marEx.tracker(
            extremes_ds.extreme_events,
            extremes_ds.mask,
            R_fill=8,
            area_filter_quartile=0.5,  # Valid value
            regional_mode=True,  # Bypass coordinate validation
            coordinate_units="degrees",
        )
        assert tracker_valid is not None

    def test_t_fill_even_validation(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
        """Test that T_fill must be even."""
        # First create valid binary data
        extremes_ds = marEx.preprocess_data(
            test_data_dask, dimensions=dimensions_gridded, dask_chunks=dask_chunks
        )

        with pytest.raises(
            ConfigurationError, match="T_fill must be even for symmetry"
        ):
            marEx.tracker(
                extremes_ds.extreme_events,
                extremes_ds.mask,
                R_fill=8,
                T_fill=3,  # Odd number - should be even
                area_filter_quartile=0.5,
                regional_mode=True,  # Bypass coordinate validation
                coordinate_units="degrees",
            )

    def test_coordinate_range_validation(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
        """Test that coordinates must be in degree range."""
        # Create test data with coordinates in wrong range (e.g., meters instead of degrees)
        invalid_coords_data = test_data_dask.copy()
        invalid_coords_data = invalid_coords_data.assign_coords(
            lon=np.linspace(
                0, 10, len(invalid_coords_data.lon)
            ),  # Only 10 degree range - too small
            lat=np.linspace(
                0, 5, len(invalid_coords_data.lat)
            ),  # Only 5 degree range - too small
        )

        # First preprocess the data to get binary events
        extremes_ds = marEx.preprocess_data(
            invalid_coords_data, dimensions=dimensions_gridded, dask_chunks=dask_chunks
        )

        with pytest.raises(
            CoordinateError, match="Cannot auto-detect coordinate units"
        ):
            marEx.tracker(
                extremes_ds.extreme_events,
                extremes_ds.mask,
                R_fill=8,
                area_filter_quartile=0.5,
            )

    def test_data_type_validation(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
        """Test that tracker validates data types correctly."""
        # First create valid binary data
        extremes_ds = marEx.preprocess_data(
            test_data_dask, dimensions=dimensions_gridded, dask_chunks=dask_chunks
        )

        # Test non-boolean binary data
        float_data = extremes_ds.extreme_events.astype(float)
        with pytest.raises(
            DataValidationError,
            match="The input DataArray must be binary \\(boolean type\\)",
        ):
            marEx.tracker(
                float_data,  # Float data instead of boolean
                extremes_ds.mask,
                R_fill=8,
                area_filter_quartile=0.5,
                regional_mode=True,  # Bypass coordinate validation
                coordinate_units="degrees",
            )

        # Test non-boolean mask
        float_mask = extremes_ds.mask.astype(float)
        with pytest.raises(
            DataValidationError, match="The mask must be binary \\(boolean type\\)"
        ):
            marEx.tracker(
                extremes_ds.extreme_events,
                float_mask,  # Float mask instead of boolean
                R_fill=8,
                area_filter_quartile=0.5,
                regional_mode=True,  # Bypass coordinate validation
                coordinate_units="degrees",
            )

        # Test all-False mask
        all_false_mask = xr.zeros_like(extremes_ds.mask)
        with pytest.raises(
            DataValidationError,
            match="Mask contains only False values. It should indicate valid regions with True",
        ):
            marEx.tracker(
                extremes_ds.extreme_events,
                all_false_mask,  # All False mask
                R_fill=8,
                area_filter_quartile=0.5,
                regional_mode=True,  # Bypass coordinate validation
                coordinate_units="degrees",
            )


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_percentile_edge_cases(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
        """Test edge cases for percentile values."""
        # Test very low percentile (should work but create many events)
        result_low = marEx.preprocess_data(
            test_data_dask,
            threshold_percentile=1,  # Very low percentile
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )
        assert isinstance(result_low, xr.Dataset)
        assert result_low.attrs["threshold_percentile"] == 1

        # Verify that ~99% of data is above 1st percentile
        extreme_frequency = float(result_low.extreme_events.mean())
        assert (
            0.95 < extreme_frequency < 1.0
        ), f"Extreme frequency {extreme_frequency} not near 0.99 for 1st percentile"

        # Test very high percentile (should work but create very few events)
        result_high = marEx.preprocess_data(
            test_data_dask,
            threshold_percentile=99,  # Very high percentile
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )
        assert isinstance(result_high, xr.Dataset)
        assert result_high.attrs["threshold_percentile"] == 99

        # Verify that ~1% of data is above 99th percentile
        extreme_frequency_high = float(result_high.extreme_events.mean())
        assert (
            0.005 < extreme_frequency_high < 0.02
        ), f"Extreme frequency {extreme_frequency_high} not near 0.01 for 99th percentile"

    def test_unusual_but_valid_parameters(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
        """Test unusual but technically valid parameter combinations."""
        # Test very small window parameters
        result_small = marEx.preprocess_data(
            test_data_dask,
            method_anomaly="shifting_baseline",
            window_year_baseline=1,  # Very small window
            smooth_days_baseline=1,  # Very small smoothing
            window_days_hobday=1,  # Very small hobday window
            method_extreme="hobday_extreme",
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )
        assert isinstance(result_small, xr.Dataset)
        assert result_small.attrs["window_year_baseline"] == 1
        assert result_small.attrs["smooth_days_baseline"] == 1
        assert result_small.attrs["window_days_hobday"] == 1

    def test_large_window_parameters_error(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
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

    def test_dask_error_message_includes_chunks_hint(
        self, test_data_numpy, dimensions_gridded, dask_chunks
    ):
        """Test that Dask error includes helpful chunking hint."""
        with pytest.raises(DataValidationError) as exc_info:
            marEx.preprocess_data(
                test_data_numpy, dimensions=dimensions_gridded, dask_chunks=dask_chunks
            )

        error_message = str(exc_info.value)
        assert "chunk" in error_message.lower()
        assert "dask" in error_message.lower()

    def test_insufficient_data_error_includes_suggestions(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
        """Test that insufficient data error includes helpful suggestions."""
        small_data = test_data_dask.isel(time=slice(0, 730)).chunk(
            dask_chunks
        )  # ~2 years

        with pytest.raises(DataValidationError) as exc_info:
            marEx.preprocess_data(
                small_data,
                method_anomaly="shifting_baseline",
                window_year_baseline=15,
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )

        error_message = str(exc_info.value)
        assert "years" in error_message
        assert (
            "reduce window_year_baseline" in error_message
            or "more data" in error_message
            or "Insufficient data" in error_message
        )

    def test_coordinate_error_includes_degree_requirement(
        self, test_data_dask, dimensions_gridded, dask_chunks
    ):
        """Test that coordinate range error includes helpful information."""
        # Create test data with coordinates in wrong range
        invalid_coords_data = test_data_dask.copy()
        invalid_coords_data = invalid_coords_data.assign_coords(
            lon=np.linspace(0, 10, len(invalid_coords_data.lon)),
            lat=np.linspace(0, 5, len(invalid_coords_data.lat)),
        )

        # First preprocess the data
        extremes_ds = marEx.preprocess_data(
            invalid_coords_data, dimensions=dimensions_gridded, dask_chunks=dask_chunks
        )

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
