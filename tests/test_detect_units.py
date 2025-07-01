"""
Unit tests for individual functions in marEx.detect module.

Tests core utility functions for marine extreme detection preprocessing.
Focuses on testing individual function behaviour rather than full pipeline integration.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx.detect as detect
import marEx.exceptions


class TestAddDecimalYear:
    """Test add_decimal_year function for decimal year calculation."""

    def test_add_decimal_year_basic(self):
        """Test basic decimal year calculation for known dates."""
        # Create test data with known dates
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        da = xr.DataArray(
            np.random.randn(len(dates)), coords={"time": dates}, dims=["time"]
        )

        result = detect.add_decimal_year(da, "time")

        # Check that decimal_year coordinate was added
        assert "decimal_year" in result.coords
        assert len(result.decimal_year) == len(dates)

        # Test specific known values
        decimal_years = result.decimal_year.values

        # January 1st should be exactly 2020.0
        assert np.isclose(decimal_years[0], 2020.0, atol=1e-6)

        # December 31st should be close to 2021.0 (2020 was a leap year)
        assert np.isclose(decimal_years[-1], 2020.0 + 365 / 366, atol=1e-6)

        # June 1st (roughly mid-year)
        june_1_idx = (dates.month == 6) & (dates.day == 1)
        june_1_decimal = decimal_years[june_1_idx][0]
        assert 2020.4 < june_1_decimal < 2020.5

    def test_add_decimal_year_leap_year(self):
        """Test decimal year calculation for leap year."""
        # Test leap year (2020)
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        da = xr.DataArray(
            np.random.randn(len(dates)), coords={"time": dates}, dims=["time"]
        )

        result = detect.add_decimal_year(da, "time")
        decimal_years = result.decimal_year.values

        # In leap year, Feb 29 should exist and be counted
        feb_29_exists = any((dates.month == 2) & (dates.day == 29))
        assert feb_29_exists

        # December 31st should be 365/366 through the year
        assert np.isclose(decimal_years[-1], 2020.0 + 365 / 366, atol=1e-6)

    def test_add_decimal_year_non_leap_year(self):
        """Test decimal year calculation for non-leap year."""
        # Test non-leap year (2021)
        dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
        da = xr.DataArray(
            np.random.randn(len(dates)), coords={"time": dates}, dims=["time"]
        )

        result = detect.add_decimal_year(da, "time")
        decimal_years = result.decimal_year.values

        # December 31st should be 364/365 through the year
        assert np.isclose(decimal_years[-1], 2021.0 + 364 / 365, atol=1e-6)

    def test_add_decimal_year_custom_dim(self):
        """Test add_decimal_year with custom dimension name."""
        dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        da = xr.DataArray(
            np.random.randn(len(dates)),
            coords={"custom_time": dates},
            dims=["custom_time"],
        )

        result = detect.add_decimal_year(da, "custom_time")

        assert "decimal_year" in result.coords
        assert len(result.decimal_year) == len(dates)
        assert np.all(result.decimal_year >= 2020.0)
        assert np.all(result.decimal_year < 2020.1)

    def test_add_decimal_year_single_date(self):
        """Test decimal year calculation for single date."""
        da = xr.DataArray(
            [1.0], coords={"time": [pd.Timestamp("2020-07-01")]}, dims=["time"]
        )

        result = detect.add_decimal_year(da, "time")

        # July 1st in leap year (182 days into year / 366 total)
        expected = 2020.0 + 182 / 366
        assert np.isclose(result.decimal_year.values[0], expected, atol=1e-6)


class TestComputeHistogramQuantile1D:
    """Test compute_histogram_quantile_1d function for quantile calculation."""

    def test_histogram_quantile_basic(self):
        """Test basic quantile calculation with known distribution."""
        # Create normally distributed data
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        da = xr.DataArray(
            data[np.newaxis, :, np.newaxis],  # Add singleton dimensions
            dims=["time", "x", "y"],
            coords={"time": [0], "x": np.arange(1000), "y": [0]},
            name="test_data",
        )

        # Test 95th percentile
        result = detect.compute_histogram_quantile_1d(da, 0.95, dim="x")

        # Compare with numpy percentile
        expected = np.percentile(data, 95)
        assert np.isclose(result.values[0, 0], expected, atol=0.1)

    def test_histogram_quantile_multiple_quantiles(self):
        """Test multiple quantile values."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        da = xr.DataArray(
            data[np.newaxis, :, np.newaxis],
            dims=["time", "x", "y"],
            coords={"time": [0], "x": np.arange(1000), "y": [0]},
            name="test_data",
        )

        quantiles = [0.5, 0.9, 0.95, 0.99]

        for q in quantiles:
            result = detect.compute_histogram_quantile_1d(da, q, dim="x")
            expected = np.percentile(data, q * 100)
            assert np.isclose(result.values[0, 0], expected, atol=0.2)

    def test_histogram_quantile_extreme_values(self):
        """Test quantile calculation with extreme values."""
        # Data with some extreme positive values (common in marine heatwave detection)
        data = np.concatenate(
            [
                np.random.normal(0, 1, 900),  # Normal background
                np.random.uniform(3, 5, 100),  # Extreme values
            ]
        )

        da = xr.DataArray(
            data[np.newaxis, :, np.newaxis],
            dims=["time", "x", "y"],
            coords={"time": [0], "x": np.arange(1000), "y": [0]},
            name="test_data",
        )

        # Test 95th percentile should be in the extreme range
        result = detect.compute_histogram_quantile_1d(da, 0.95, dim="x")

        # Should be above normal values but reasonable
        assert result.values[0, 0] > 2.0
        assert result.values[0, 0] < 6.0

    def test_histogram_quantile_custom_bins(self):
        """Test quantile calculation with custom bin edges."""
        np.random.seed(42)
        data = np.random.uniform(-2, 3, 1000)  # Uniform distribution

        da = xr.DataArray(
            data[np.newaxis, :, np.newaxis],
            dims=["time", "x", "y"],
            coords={"time": [0], "x": np.arange(1000), "y": [0]},
            name="test_data",
        )

        # Custom bin edges
        bin_edges = np.linspace(-3, 4, 100)

        result = detect.compute_histogram_quantile_1d(
            da, 0.5, dim="x", bin_edges=bin_edges
        )

        # Median of uniform distribution should be near 0.5
        expected = np.percentile(data, 50)
        assert np.isclose(result.values[0, 0], expected, atol=0.1)

    def test_histogram_quantile_edge_cases(self):
        """Test edge cases for quantile calculation."""
        # Test with constant data
        da_const = xr.DataArray(
            np.full((1, 100, 1), 2.5),
            dims=["time", "x", "y"],
            coords={"time": [0], "x": np.arange(100), "y": [0]},
            name="test_data",
        )

        result = detect.compute_histogram_quantile_1d(da_const, 0.95, dim="x")
        assert np.isclose(result.values[0, 0], 2.5, atol=0.01)

        # Test with very small dataset
        da_small = xr.DataArray(
            np.array([[[1.0], [2.0], [3.0]]]).transpose(2, 1, 0),
            dims=["time", "x", "y"],
            coords={"time": [0], "x": [0, 1, 2], "y": [0]},
            name="test_data",
        )

        result = detect.compute_histogram_quantile_1d(da_small, 0.5, dim="x")
        assert np.isclose(result.values[0, 0], 2.0, atol=0.1)


class TestGetPreprocessingSteps:
    """Test _get_preprocessing_steps function for metadata generation."""

    def test_detrended_baseline_steps(self):
        """Test preprocessing steps for detrended baseline method."""
        steps = detect._get_preprocessing_steps(
            method_anomaly="detrended_baseline",
            method_extreme="global_extreme",
            std_normalise=False,
            detrend_orders=[1, 2],
            window_year_baseline=15,
            smooth_days_baseline=21,
            window_days_hobday=11,
        )

        expected_steps = [
            "Removed polynomial trend orders=[1, 2] & seasonal cycle",
            "Global percentile threshold applied to all days",
        ]

        assert steps == expected_steps

    def test_detrended_baseline_with_std_normalise(self):
        """Test preprocessing steps with standardisation."""
        steps = detect._get_preprocessing_steps(
            method_anomaly="detrended_baseline",
            method_extreme="global_extreme",
            std_normalise=True,
            detrend_orders=[1],
            window_year_baseline=15,
            smooth_days_baseline=21,
            window_days_hobday=11,
        )

        expected_steps = [
            "Removed polynomial trend orders=[1] & seasonal cycle",
            "Normalised by 30-day rolling STD",
            "Global percentile threshold applied to all days",
        ]

        assert steps == expected_steps

    def test_shifting_baseline_steps(self):
        """Test preprocessing steps for shifting baseline method."""
        steps = detect._get_preprocessing_steps(
            method_anomaly="shifting_baseline",
            method_extreme="hobday_extreme",
            std_normalise=False,
            detrend_orders=[1],
            window_year_baseline=10,
            smooth_days_baseline=15,
            window_days_hobday=5,
        )

        expected_steps = [
            "Rolling climatology using 10 years",
            "Smoothed with 15-day window",
            "Day-of-year thresholds with 5 day window",
        ]

        assert steps == expected_steps

    def test_all_combinations(self):
        """Test all valid method combinations."""
        # Test all four valid combinations
        combinations = [
            ("detrended_baseline", "global_extreme"),
            ("detrended_baseline", "hobday_extreme"),
            ("shifting_baseline", "global_extreme"),
            ("shifting_baseline", "hobday_extreme"),
        ]

        for method_anomaly, method_extreme in combinations:
            steps = detect._get_preprocessing_steps(
                method_anomaly=method_anomaly,
                method_extreme=method_extreme,
                std_normalise=False,
                detrend_orders=[1],
                window_year_baseline=15,
                smooth_days_baseline=21,
                window_days_hobday=11,
            )

            # Should always have at least 2 steps
            assert len(steps) >= 2

            # Should contain method-specific keywords
            steps_text = " ".join(steps)
            if method_anomaly == "detrended_baseline":
                assert "polynomial trend" in steps_text
            else:
                assert "climatology" in steps_text

            if method_extreme == "global_extreme":
                assert "Global percentile" in steps_text
            else:
                assert "Day-of-year" in steps_text


class TestValidationFunctions:
    """Test input validation and error handling."""

    def test_invalid_method_anomaly(self):
        """Test error handling for invalid anomaly method."""
        da = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2020-01-01", periods=10),
                "lat": np.arange(5),
                "lon": np.arange(5),
            },
        ).chunk({"time": 5})

        with pytest.raises(
            marEx.exceptions.ConfigurationError, match="Unknown anomaly method"
        ):
            detect.compute_normalised_anomaly(da, method_anomaly="invalid_method")

    def test_invalid_method_extreme(self):
        """Test error handling for invalid extreme method."""
        da = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2020-01-01", periods=10),
                "lat": np.arange(5),
                "lon": np.arange(5),
            },
        ).chunk({"time": 5})

        with pytest.raises(
            marEx.exceptions.ConfigurationError, match="Unknown extreme method"
        ):
            detect.identify_extremes(da, method_extreme="invalid_method")

    def test_non_dask_array_error(self):
        """Test error when non-Dask array is provided to preprocess_data."""
        da = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2020-01-01", periods=10),
                "lat": np.arange(5),
                "lon": np.arange(5),
            },
        )  # Not chunked = not Dask-backed

        with pytest.raises(
            marEx.exceptions.DataValidationError,
            match="Input DataArray must be Dask-backed",
        ):
            detect.preprocess_data(da)


class TestRollingHistogramQuantile:
    """Test _rolling_histogram_quantile helper function."""

    def test_rolling_histogram_basic(self):
        """Test basic rolling histogram quantile calculation."""
        # Create simple histogram data (365 days, 10 bins)
        np.random.seed(42)
        hist_chunk = np.random.randint(0, 100, size=(365, 10))
        bin_centers = np.linspace(-2, 3, 10)

        result = detect._rolling_histogram_quantile(
            hist_chunk, window_days_hobday=11, q=0.95, bin_centers=bin_centers
        )

        # Check output shape
        assert result.shape == (365,)
        assert result.dtype == np.float32

        # Check reasonable output range
        assert np.all(result >= bin_centers[0])
        assert np.all(result <= bin_centers[-1])

    def test_rolling_histogram_edge_wrapping(self):
        """Test that rolling window properly wraps around year boundaries."""
        # Create histogram with distinct pattern at year boundaries
        hist_chunk = np.ones((365, 10), dtype=np.float64)

        # Make January 1st (day 0) and December 31st (day 364) have high values in last bin
        hist_chunk[0, -1] = 1000  # Jan 1st
        hist_chunk[364, -1] = 1000  # Dec 31st

        bin_centers = np.linspace(0, 10, 10)

        result = detect._rolling_histogram_quantile(
            hist_chunk, window_days_hobday=11, q=0.95, bin_centers=bin_centers
        )

        # Due to wrapping, both Jan 1st and Dec 31st should see influence from each other
        # The 95th percentile near these dates should be higher than in the middle of year
        jan_1_quantile = result[0]
        mid_year_quantile = result[182]  # Around July
        dec_31_quantile = result[364]

        assert jan_1_quantile > mid_year_quantile
        assert dec_31_quantile > mid_year_quantile

    def test_rolling_histogram_realistic_input(self):
        """Test rolling histogram with realistic input dimensions."""
        # Test with standard yearly data (365 days) and reasonable bin count
        np.random.seed(42)
        hist_chunk = np.random.randint(0, 100, size=(365, 20)).astype(np.float64)
        bin_centers = np.linspace(-2, 3, 20)

        # Test 95th percentile calculation
        result = detect._rolling_histogram_quantile(
            hist_chunk, window_days_hobday=11, q=0.95, bin_centers=bin_centers
        )

        # Basic validation of output shape and reasonable values
        assert len(result) == 365
        assert result.dtype == np.float32
        assert np.all(result >= bin_centers[0])
        assert np.all(result <= bin_centers[-1])
        assert np.all(np.isfinite(result))
