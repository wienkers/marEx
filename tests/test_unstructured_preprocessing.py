from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx

from .conftest import assert_percentile_frequency


class TestUnstructuredPreprocessing:
    """Test preprocessing functionality for unstructured data using test datasets."""

    @classmethod
    def setup_class(cls):
        """Load test data for all tests."""
        test_data_path = Path(__file__).parent / "data" / "sst_unstructured.zarr"
        ds = xr.open_zarr(str(test_data_path), chunks={}).persist()
        cls.sst_data = ds.to  # Extract the DataArray named 'to'

        # Define standard dimensions for unstructured data (no ydim)
        cls.dimensions = {
            "time": "time",
            "xdim": "ncells",  # Note: no 'ydim' indicates unstructured grid
        }

        # Standard dask chunks for output
        cls.dask_chunks = {"time": 25}

        # Mock neighbors and cell areas for unstructured grid
        # In real usage these would come from grid files
        ncells = cls.sst_data.sizes.get("ncells", cls.sst_data.sizes.get("cell", 1000))
        cls.mock_neighbours = xr.DataArray(
            np.random.randint(0, ncells, (3, ncells)), dims=["nv", "ncells"]
        )
        cls.mock_cell_areas = xr.DataArray(
            np.ones(ncells) * 1000.0, dims=["ncells"]  # Mock cell areas in m²
        )

    def test_shifting_baseline_hobday_extreme_unstructured(self):
        """Test preprocessing with shifting_baseline + hobday_extreme for unstructured grid."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="shifting_baseline",
            method_extreme="hobday_extreme",
            threshold_percentile=95,
            window_year_baseline=5,  # Reduced for test data
            smooth_days_baseline=11,  # Reduced for test data
            window_days_hobday=5,  # Reduced for test data
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
            neighbours=self.mock_neighbours,
            cell_areas=self.mock_cell_areas,
        )

        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert "extreme_events" in extremes_ds.data_vars
        assert "dat_anomaly" in extremes_ds.data_vars
        assert "thresholds" in extremes_ds.data_vars
        assert "mask" in extremes_ds.data_vars

        # Verify unstructured-specific variables
        assert "neighbours" in extremes_ds.data_vars
        assert "cell_areas" in extremes_ds.data_vars

        # Verify attributes
        assert extremes_ds.attrs["method_anomaly"] == "shifting_baseline"
        assert extremes_ds.attrs["method_extreme"] == "hobday_extreme"
        assert extremes_ds.attrs["threshold_percentile"] == 95

        # Verify data types
        assert extremes_ds.extreme_events.dtype == bool
        assert extremes_ds.dat_anomaly.dtype == np.float32

        # Verify dimensions for unstructured grid
        cell_dim = "ncells" if "ncells" in extremes_ds.dims else "cell"
        assert "time" in extremes_ds.extreme_events.dims
        assert cell_dim in extremes_ds.extreme_events.dims
        assert "dayofyear" in extremes_ds.thresholds.dims
        assert cell_dim in extremes_ds.thresholds.dims

        # Verify no lat/lon dimensions (since it's unstructured)
        assert "lat" not in extremes_ds.extreme_events.dims
        assert "lon" not in extremes_ds.extreme_events.dims

        # Verify reasonable extreme event frequency
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        print(
            f"Exact extreme_frequency for shifting_baseline + hobday_extreme (unstructured): {extreme_frequency}"
        )
        assert_percentile_frequency(
            extreme_frequency,
            95,
            description="shifting_baseline + hobday_extreme (unstructured)",
        )

    def test_detrended_baseline_global_extreme_unstructured(self):
        """Test preprocessing with detrended_baseline + global_extreme for unstructured grid."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrended_baseline",
            method_extreme="global_extreme",
            threshold_percentile=95,
            detrend_orders=[1, 2],
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
            neighbours=self.mock_neighbours,
            cell_areas=self.mock_cell_areas,
        )

        # Verify output structure
        assert isinstance(extremes_ds, xr.Dataset)
        assert "extreme_events" in extremes_ds.data_vars
        assert "dat_anomaly" in extremes_ds.data_vars
        assert "thresholds" in extremes_ds.data_vars
        assert "mask" in extremes_ds.data_vars

        # Verify unstructured-specific variables
        assert "neighbours" in extremes_ds.data_vars
        assert "cell_areas" in extremes_ds.data_vars

        # Verify attributes
        assert extremes_ds.attrs["method_anomaly"] == "detrended_baseline"
        assert extremes_ds.attrs["method_extreme"] == "global_extreme"
        assert extremes_ds.attrs["threshold_percentile"] == 95

        # Verify data types
        assert extremes_ds.extreme_events.dtype == bool
        assert extremes_ds.dat_anomaly.dtype == np.float32

        # Verify dimensions for unstructured grid
        cell_dim = "ncells" if "ncells" in extremes_ds.dims else "cell"
        assert "time" in extremes_ds.extreme_events.dims
        assert cell_dim in extremes_ds.extreme_events.dims

        # For global_extreme, thresholds should be 1D (cells) not 2D with dayofyear
        assert "dayofyear" not in extremes_ds.thresholds.dims
        assert cell_dim in extremes_ds.thresholds.dims

        # Verify reasonable extreme event frequency
        extreme_frequency = float(extremes_ds.extreme_events.mean())
        print(
            f"Exact extreme_frequency for detrended_baseline + global_extreme (unstructured): {extreme_frequency}"
        )
        assert_percentile_frequency(
            extreme_frequency,
            95,
            description="detrended_baseline + global_extreme (unstructured)",
        )

    def test_unstructured_grid_detection(self):
        """Test that the function correctly detects unstructured vs gridded data."""
        # Test with unstructured dimensions (no ydim)
        unstructured_dims = {"time": "time", "xdim": "ncells"}

        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrended_baseline",
            method_extreme="global_extreme",
            threshold_percentile=95,
            dimensions=unstructured_dims,
            dask_chunks=self.dask_chunks,
            neighbours=self.mock_neighbours,
            cell_areas=self.mock_cell_areas,
        )

        # Should have neighbors and cell_areas for unstructured grid
        assert "neighbours" in extremes_ds.data_vars
        assert "cell_areas" in extremes_ds.data_vars

        # Should not have regular lat/lon structure
        cell_dim = "ncells" if "ncells" in extremes_ds.dims else "cell"
        assert cell_dim in extremes_ds.extreme_events.dims
        assert len(extremes_ds.extreme_events.dims) == 2  # time + cell dimension only

    def test_unstructured_consistency(self):
        """Test that both preprocessing methods produce consistent output for unstructured grids."""
        # Run both methods
        shifting_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="shifting_baseline",
            method_extreme="hobday_extreme",
            threshold_percentile=95,
            window_year_baseline=5,  # Reduced for test data
            smooth_days_baseline=11,  # Reduced for test data
            window_days_hobday=5,  # Reduced for test data
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
            neighbours=self.mock_neighbours,
            cell_areas=self.mock_cell_areas,
        )

        detrended_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrended_baseline",
            method_extreme="global_extreme",
            threshold_percentile=95,
            detrend_orders=[1, 2],
            dimensions=self.dimensions,
            dask_chunks=self.dask_chunks,
            neighbours=self.mock_neighbours,
            cell_areas=self.mock_cell_areas,
        )

        # Both should have the same core data variables
        core_vars = [
            "extreme_events",
            "dat_anomaly",
            "mask",
            "neighbours",
            "cell_areas",
        ]
        for var in core_vars:
            assert var in shifting_ds.data_vars
            assert var in detrended_ds.data_vars

        # Both should have mask with same spatial shape
        assert shifting_ds.mask.shape == detrended_ds.mask.shape

        # Both should have same cell dimension structure
        cell_dim = "ncells" if "ncells" in shifting_ds.dims else "cell"
        assert cell_dim in shifting_ds.extreme_events.dims
        assert cell_dim in detrended_ds.extreme_events.dims

        # Both should have consistent neighbours and cell_areas
        assert shifting_ds.neighbours.shape == detrended_ds.neighbours.shape
        assert shifting_ds.cell_areas.shape == detrended_ds.cell_areas.shape
