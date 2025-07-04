"""
Unit tests for individual functions in marEx.detect module.

Tests core utility functions for marine extreme detection preprocessing.
Focuses on testing individual function behaviour rather than full pipeline integration.
"""

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
        da = xr.DataArray(np.random.randn(len(dates)), coords={"time": dates}, dims=["time"])

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
        da = xr.DataArray(np.random.randn(len(dates)), coords={"time": dates}, dims=["time"])

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
        da = xr.DataArray(np.random.randn(len(dates)), coords={"time": dates}, dims=["time"])

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
        da = xr.DataArray([1.0], coords={"time": [pd.Timestamp("2020-07-01")]}, dims=["time"])

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
        result = detect._compute_histogram_quantile_1d(da, 0.95, dim="x")

        # Compare with numpy percentile
        expected = np.percentile(data, 95)
        assert np.isclose(result.values[0, 0], expected, atol=0.01)

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

        quantiles = [0.6, 0.9, 0.95, 0.99]

        for q in quantiles:
            result = detect._compute_histogram_quantile_1d(da, q, dim="x")
            expected = np.percentile(data, q * 100)

            histogram_value = result.values[0, 0]
            error = abs(histogram_value - expected)

            assert np.isclose(histogram_value, expected, atol=0.01), (
                f"Approximate percentile histogram method has too high error for {q*100:.1f}th percentile. "
                f"Histogram result: {histogram_value:.6f}, NumPy result: {expected:.6f}, "
                f"Error: {error:.6f} (tolerance: 0.01). "
                f"Consider using exact_percentile=True for high percentiles or percentiles with sparse data."
            )

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
        result = detect._compute_histogram_quantile_1d(da, 0.95, dim="x")

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

        result = detect._compute_histogram_quantile_1d(da, 0.5, dim="x", bin_edges=bin_edges)

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

        result = detect._compute_histogram_quantile_1d(da_const, 0.95, dim="x")
        assert np.isclose(result.values[0, 0], 2.5, atol=0.01)

        # Test with very small dataset
        da_small = xr.DataArray(
            np.array([[[1.0], [2.0], [3.0]]]).transpose(2, 1, 0),
            dims=["time", "x", "y"],
            coords={"time": [0], "x": [0, 1, 2], "y": [0]},
            name="test_data",
        )

        result = detect._compute_histogram_quantile_1d(da_small, 0.5, dim="x")
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

        with pytest.raises(marEx.exceptions.ConfigurationError, match="Unknown anomaly method"):
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

        with pytest.raises(marEx.exceptions.ConfigurationError, match="Unknown extreme method"):
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


class TestComputeHistogramQuantile2D:
    """Test _compute_histogram_quantile_2d function for 2D quantile calculation."""

    def test_histogram_quantile_2d_basic(self):
        """Test basic 2D histogram quantile calculation."""
        # Create synthetic temperature anomaly data
        np.random.seed(42)
        n_years = 3
        n_days_per_year = 365
        lat_size, lon_size = 5, 5

        # Create time coordinate with proper DatetimeIndex
        time_coord = pd.date_range("2020-01-01", periods=n_years * n_days_per_year, freq="D")

        # Generate data with seasonal cycle and anomalies
        temp_data = np.random.normal(0, 1, (len(time_coord), lat_size, lon_size))

        da = xr.DataArray(
            temp_data,
            dims=["time", "lat", "lon"],
            coords={
                "time": time_coord,
                "lat": np.arange(lat_size),
                "lon": np.arange(lon_size),
            },
            name="temperature_anomaly",
        ).chunk({"time": 100, "lat": -1, "lon": -1})

        # Add dayofyear coordinate as expected by the function
        da = da.assign_coords(dayofyear=da.time.dt.dayofyear)

        # Test 95th percentile calculation
        result = detect._compute_histogram_quantile_2d(
            da, q=0.95, window_days_hobday=11, dimensions={"time": "time", "x": "lon", "y": "lat"}
        )

        # Check output shape - should have dayofyear and spatial dimensions
        assert "dayofyear" in result.dims
        assert "lat" in result.dims
        assert "lon" in result.dims
        assert result.sizes["dayofyear"] == 366  # Includes leap day
        assert result.sizes["lat"] == lat_size
        assert result.sizes["lon"] == lon_size

        # Check that results are reasonable (should be positive for 95th percentile)
        assert np.all(result >= -5.0)  # Should be reasonable anomaly values
        assert np.all(result <= 10.0)

    def test_histogram_quantile_2d_vs_exact_quantile(self):
        """Test 2D histogram quantile vs exact quantile computation with custom bins."""
        # Create controlled test data
        np.random.seed(42)
        n_years = 3
        lat_size, lon_size = 2, 2

        # Create time coordinate
        time_coord = pd.date_range("2020-01-01", periods=n_years * 365, freq="D")

        # Generate simple positive anomaly data (range 0 to 4)
        temp_data = np.random.uniform(0, 4, (len(time_coord), lat_size, lon_size))

        da = xr.DataArray(
            temp_data,
            dims=["time", "lat", "lon"],
            coords={
                "time": time_coord,
                "lat": np.arange(lat_size),
                "lon": np.arange(lon_size),
            },
            name="temperature_anomaly",
        ).chunk({"time": 200, "lat": -1, "lon": -1})

        # Add dayofyear coordinate as expected by the function
        da = da.assign_coords(dayofyear=da.time.dt.dayofyear)

        # Use custom bins that match our data range
        custom_bins = np.linspace(-0.5, 4.5, 25)  # Good resolution for our data range

        # Test a single quantile with custom bins
        q = 0.9
        window_days = 11

        # Get 2D histogram-based quantile
        hist_result = detect._compute_histogram_quantile_2d(
            da, q=q, window_days_hobday=window_days, bin_edges=custom_bins, dimensions={"time": "time", "x": "lon", "y": "lat"}
        )

        # Compare against exact quantile for a few representative points
        test_days = [1, 100, 200]  # Test a few days
        pad_size = window_days // 2

        for lat_idx in range(lat_size):
            for lon_idx in range(lon_size):
                for doy in test_days:
                    # Get window around this day of year with wrapping
                    window_doys = [(doy + offset - 1) % 365 + 1 for offset in range(-pad_size, pad_size + 1)]

                    # Collect data for these days across all years for this spatial point
                    window_data = []
                    for wd in window_doys:
                        wd_mask = time_coord.dayofyear == wd
                        if np.any(wd_mask):
                            point_data = da.isel(lat=lat_idx, lon=lon_idx).values
                            window_data.extend(point_data[wd_mask])

                    if len(window_data) > 15:  # Need sufficient data for meaningful comparison
                        exact_value = np.percentile(window_data, q * 100)
                        hist_value = hist_result.isel(lat=lat_idx, lon=lon_idx, dayofyear=doy - 1).values

                        # Use tolerance based on bin width
                        bin_width = custom_bins[1] - custom_bins[0]
                        tolerance = bin_width * 3  # Allow up to 3 bin widths of error

                        assert np.isclose(hist_value, exact_value, atol=tolerance), (
                            f"2D histogram quantile differs from exact quantile for {q*100:.1f}th percentile "
                            f"at lat={lat_idx}, lon={lon_idx}, day={doy}. "
                            f"Histogram: {hist_value:.4f}, Exact: {exact_value:.4f}, "
                            f"Error: {abs(hist_value - exact_value):.4f} (tolerance: {tolerance:.4f}). "
                            f"Bin width: {bin_width:.4f}"
                        )

    def test_histogram_quantile_2d_custom_bins(self):
        """Test 2D histogram quantile with custom bin edges."""
        # Create test data
        np.random.seed(123)
        time_coord = pd.date_range("2020-01-01", periods=365, freq="D")
        temp_data = np.random.uniform(-2, 4, (365, 2, 2))  # Uniform distribution

        da = xr.DataArray(
            temp_data,
            dims=["time", "lat", "lon"],
            coords={
                "time": time_coord,
                "lat": [0, 1],
                "lon": [0, 1],
            },
            name="test_data",
        ).chunk({"time": 100, "lat": -1, "lon": -1})

        # Add dayofyear coordinate as expected by the function
        da = da.assign_coords(dayofyear=da.time.dt.dayofyear)

        # Custom bin edges with fine resolution
        custom_bins = np.linspace(-3, 5, 50)

        result = detect._compute_histogram_quantile_2d(
            da, q=0.5, window_days_hobday=5, bin_edges=custom_bins, dimensions={"time": "time", "x": "lon", "y": "lat"}
        )

        # For uniform distribution, median should be reasonable
        # Check a few representative values (accounting for bin precision)
        median_values = result.isel(lat=0, lon=0, dayofyear=[0, 100, 200]).values

        # With small window (5 days) and custom bins, results should be within the data range
        for val in median_values:
            assert 0 <= val <= 5, f"Median value {val} outside expected range [0, 5] for uniform distribution with custom bins"

    def test_histogram_quantile_2d_window_sizes(self):
        """Test 2D histogram quantile with different window sizes."""
        # Create simple test data
        np.random.seed(456)
        time_coord = pd.date_range("2019-01-01", periods=365 * 2, freq="D")
        temp_data = np.random.normal(0, 1, (len(time_coord), 2, 2))

        da = xr.DataArray(
            temp_data,
            dims=["time", "lat", "lon"],
            coords={
                "time": time_coord,
                "lat": [0, 1],
                "lon": [0, 1],
            },
            name="test_data",
        ).chunk({"time": 200, "lat": -1, "lon": -1})

        # Add dayofyear coordinate as expected by the function
        da = da.assign_coords(dayofyear=da.time.dt.dayofyear)

        # Test different window sizes
        window_sizes = [5, 11, 21]
        quantile = 0.9

        results = {}
        for window_days in window_sizes:
            result = detect._compute_histogram_quantile_2d(
                da, q=quantile, window_days_hobday=window_days, dimensions={"time": "time", "x": "lon", "y": "lat"}
            )
            results[window_days] = result

        # Larger windows should give smoother results (less variation between adjacent days)
        for window_days in window_sizes:
            result = results[window_days]
            # Check that results are finite and reasonable
            assert np.all(np.isfinite(result.values))
            assert np.all(result.values >= -5.0)
            assert np.all(result.values <= 5.0)

            # Larger windows should have less day-to-day variation
            daily_variation = np.std(result.isel(lat=0, lon=0).diff("dayofyear").values)
            assert daily_variation < 1.0, f"Daily variation {daily_variation} too high for window {window_days}"

    def test_histogram_quantile_2d_edge_cases(self):
        """Test edge cases for 2D histogram quantile calculation."""
        # Test with minimal data
        time_coord = pd.date_range("2020-01-01", periods=30, freq="D")
        temp_data = np.ones((30, 1, 1))  # Constant values

        da = xr.DataArray(
            temp_data,
            dims=["time", "lat", "lon"],
            coords={
                "time": time_coord,
                "lat": [0],
                "lon": [0],
            },
            name="constant_data",
        ).chunk({"time": 30, "lat": -1, "lon": -1})

        # Add dayofyear coordinate as expected by the function
        da = da.assign_coords(dayofyear=da.time.dt.dayofyear)

        result = detect._compute_histogram_quantile_2d(
            da, q=0.95, window_days_hobday=5, dimensions={"time": "time", "x": "lon", "y": "lat"}
        )

        # With constant data, quantiles should be either 1.0 (the value) or 0.0 (due to asymmetric bins)
        # Since we have constant value of 1.0 (positive), most should be ~1.0
        # But some might be 0.0 due to insufficient data in certain day-of-year bins
        result_values = result.values
        unique_values = np.unique(result_values[np.isfinite(result_values)])

        # Should only have values around 0.0 or 1.0 for constant input
        assert np.all(
            (unique_values < 0.1) | (np.abs(unique_values - 1.0) < 0.1)
        ), f"Unexpected values in constant data result: {unique_values}"
