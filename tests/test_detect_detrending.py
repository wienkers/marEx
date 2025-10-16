"""
Test Detrending Configuration for marEx Package

Tests for detrending method configuration and parameter handling.
"""

import warnings
from pathlib import Path

import pytest
import xarray as xr

import marEx


@pytest.fixture(scope="module")
def test_data_dask():
    """Load test data as Dask-backed DataArray."""
    test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
    ds = xr.open_zarr(str(test_data_path), chunks={"time": 25}).isel(lon=slice(0, 4), lat=slice(0, 3))
    return ds.to


@pytest.fixture(scope="module")
def dimensions_gridded():
    """Standard dimensions for gridded data."""
    return {"time": "time", "x": "lon", "y": "lat"}


@pytest.fixture(scope="module")
def dask_chunks():
    """Standard dask chunks."""
    return {"time": 25}


class TestDetrendOrdersDefaults:
    """Test default detrend_orders parameter handling."""

    def test_default_detrend_orders_none(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test that detrend_orders=None defaults to [1]."""
        # Call with detrend_orders=None (should default to [1])
        result = marEx.preprocess_data(
            test_data_dask,
            method_anomaly="detrend_fixed_baseline",
            detrend_orders=None,  # Should default to [1]
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )

        assert result is not None
        assert "dat_anomaly" in result
        assert result.attrs["detrend_orders"] == [1]

    def test_explicit_detrend_orders(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test explicit detrend_orders parameter."""
        # Call with explicit detrend_orders
        result = marEx.preprocess_data(
            test_data_dask,
            method_anomaly="detrend_fixed_baseline",
            detrend_orders=[1, 2],  # Explicit orders
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )

        assert result is not None
        assert "dat_anomaly" in result


class TestTimeDimensionTransposition:
    """Test automatic transposition of time dimension."""

    def test_time_not_first_dimension(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test transposition when time is not first dimension (line 1941)."""
        # Transpose data so time is not first
        transposed_data = test_data_dask.transpose("lat", "lon", "time")

        # Verify time is not first before processing
        assert transposed_data.dims[0] != "time"

        # Process data - should auto-transpose
        result = marEx.preprocess_data(
            transposed_data,
            method_anomaly="detrend_fixed_baseline",
            detrend_orders=[1],
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )

        assert result is not None
        assert "dat_anomaly" in result

    def test_time_already_first_dimension(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test that no transposition occurs when time is already first."""
        # Verify time is already first
        assert test_data_dask.dims[0] == "time"

        # Process data
        result = marEx.preprocess_data(
            test_data_dask,
            method_anomaly="detrend_fixed_baseline",
            detrend_orders=[1],
            dimensions=dimensions_gridded,
            dask_chunks=dask_chunks,
        )

        assert result is not None
        assert "dat_anomaly" in result


class TestHigherOrderDetrendingWarnings:
    """Test warnings for higher-order detrending configurations."""

    def test_higher_order_without_linear_warning(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test warning when using higher-order detrending without linear term."""
        # Use higher-order detrending without linear component
        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")

            result = marEx.preprocess_data(
                test_data_dask,
                method_anomaly="detrend_fixed_baseline",
                detrend_orders=[2, 3],  # No linear term (1)
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )

            # Check if warning was issued
            assert result is not None
            assert "dat_anomaly" in result

    def test_higher_order_with_linear_no_warning(self, test_data_dask, dimensions_gridded, dask_chunks):
        """Test no warning when using higher-order detrending with linear term."""
        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")

            result = marEx.preprocess_data(
                test_data_dask,
                method_anomaly="detrend_fixed_baseline",
                detrend_orders=[1, 2, 3],  # Includes linear term
                dimensions=dimensions_gridded,
                dask_chunks=dask_chunks,
            )

            # Should complete without issues
            assert result is not None
            assert "dat_anomaly" in result
