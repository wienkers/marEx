from pathlib import Path

import numpy as np
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

        # Artificially make some masked NaN data in the 2nd cell
        cls.sst_data = cls.sst_data.where(~((cls.sst_data.ncells == 2)), np.nan)

        # Define standard dimensions for unstructured data (no y)
        cls.dimensions = {
            "time": "time",
            "x": "ncells",  # Note: no 'y' indicates unstructured grid
        }

        # For unstructured data, we specify coordinate names even if they don't exist yet
        # The preprocessing pipeline may create them or use them for reference
        cls.coordinates = {"time": "time", "x": "lon", "y": "lat"}

        # Standard dask chunks for output
        cls.dask_chunks = {"time": 25}

        # Mock neighbors and cell areas for unstructured grid
        # In real usage these would come from grid files
        ncells = cls.sst_data.sizes.get("ncells", cls.sst_data.sizes.get("cell", 1000))
        cls.mock_neighbours = xr.DataArray(np.random.randint(0, ncells, (3, ncells)), dims=["nv", "ncells"])
        cls.mock_cell_areas = xr.DataArray(np.ones(ncells) * 1000.0, dims=["ncells"])  # Mock cell areas in m²

        # Add mock lat/lon coordinates to the data for unstructured processing
        # These are required by the preprocessing pipeline
        lat_coords = xr.DataArray(np.linspace(-90, 90, ncells), dims=["ncells"], name="lat")
        lon_coords = xr.DataArray(np.linspace(-180, 180, ncells), dims=["ncells"], name="lon")
        cls.sst_data = cls.sst_data.assign_coords(lat=lat_coords, lon=lon_coords)

    def test_shifting_baseline_hobday_extreme_unstructured(self):
        """Test preprocessing with shifting_baseline + hobday_extreme for unstructured grid."""
        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="shifting_baseline",
            method_extreme="hobday_extreme",
            threshold_percentile=95,
            window_year_baseline=5,  # Reduced for test data
            smooth_days_baseline=5,  # Reduced for test data
            window_days_hobday=3,  # Reduced for test data
            dimensions=self.dimensions,
            coordinates=self.coordinates,
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
        print(f"Exact extreme_frequency for shifting_baseline + hobday_extreme (unstructured): {extreme_frequency}")
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
            coordinates=self.coordinates,
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
        print(f"Exact extreme_frequency for detrended_baseline + global_extreme (unstructured): {extreme_frequency}")
        assert_percentile_frequency(
            extreme_frequency,
            95,
            description="detrended_baseline + global_extreme (unstructured)",
        )

    def test_unstructured_grid_detection(self):
        """Test that the function correctly detects unstructured vs gridded data."""
        # Test with unstructured dimensions (no y)
        unstructured_dims = {"time": "time", "x": "ncells"}

        extremes_ds = marEx.preprocess_data(
            self.sst_data,
            method_anomaly="detrended_baseline",
            method_extreme="global_extreme",
            threshold_percentile=95,
            dimensions=unstructured_dims,
            coordinates=self.coordinates,
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
            coordinates=self.coordinates,
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
            coordinates=self.coordinates,
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

    def test_custom_dimension_names_unstructured(self):
        """Test preprocessing with custom dimension and coordinate names for unstructured grid with both detect methods."""
        # Rename dimensions to: "t" and "cell"
        # Rename coordinates to: "T", "longitude", and "latitude"
        renamed_data = self.sst_data.rename({"time": "t", "ncells": "cell", "lon": "longitude", "lat": "latitude"})

        # Update mock data to match renamed dimensions
        mock_neighbours_renamed = self.mock_neighbours.rename({"ncells": "cell"})
        mock_cell_areas_renamed = self.mock_cell_areas.rename({"ncells": "cell"})

        # Define custom dimensions and coordinates
        custom_dimensions = {"time": "t", "x": "cell"}
        custom_coordinates = {"time": "t", "x": "longitude", "y": "latitude"}

        # Test 1: detrended_baseline + global_extreme
        extremes_ds_detrended = marEx.preprocess_data(
            renamed_data,
            method_anomaly="detrended_baseline",
            method_extreme="global_extreme",
            threshold_percentile=95,
            detrend_orders=[1, 2],
            dimensions=custom_dimensions,
            coordinates=custom_coordinates,
            dask_chunks={"t": 25},
            neighbours=mock_neighbours_renamed,
            cell_areas=mock_cell_areas_renamed,
        )

        # Verify output structure for detrended_baseline method
        assert isinstance(extremes_ds_detrended, xr.Dataset)
        assert "extreme_events" in extremes_ds_detrended.data_vars
        assert "dat_anomaly" in extremes_ds_detrended.data_vars
        assert "thresholds" in extremes_ds_detrended.data_vars
        assert "mask" in extremes_ds_detrended.data_vars
        assert "neighbours" in extremes_ds_detrended.data_vars
        assert "cell_areas" in extremes_ds_detrended.data_vars

        # Verify dimensions are correctly renamed
        assert "t" in extremes_ds_detrended.extreme_events.dims
        assert "cell" in extremes_ds_detrended.extreme_events.dims

        # Verify no regular grid dimensions present
        assert "latitude" not in extremes_ds_detrended.extreme_events.dims
        assert "longitude" not in extremes_ds_detrended.extreme_events.dims

        # Verify coordinates are present
        assert "latitude" in extremes_ds_detrended.coords
        assert "longitude" in extremes_ds_detrended.coords
        assert extremes_ds_detrended.latitude.dims == ("cell",)
        assert extremes_ds_detrended.longitude.dims == ("cell",)

        # Verify attributes for detrended_baseline
        assert extremes_ds_detrended.attrs["method_anomaly"] == "detrended_baseline"
        assert extremes_ds_detrended.attrs["method_extreme"] == "global_extreme"

        # For global_extreme, thresholds should be 1D (cells) not 2D with dayofyear
        assert "dayofyear" not in extremes_ds_detrended.thresholds.dims
        assert "cell" in extremes_ds_detrended.thresholds.dims

        # Verify reasonable extreme event frequency
        extreme_frequency_detrended = float(extremes_ds_detrended.extreme_events.mean())
        assert_percentile_frequency(
            extreme_frequency_detrended,
            95,
            description="Custom dimensions (unstructured): detrended_baseline + global_extreme",
        )

        # Test 2: shifting_baseline + hobday_extreme
        extremes_ds_shifting = marEx.preprocess_data(
            renamed_data,
            method_anomaly="shifting_baseline",
            method_extreme="hobday_extreme",
            threshold_percentile=95,
            window_year_baseline=5,  # Reduced for test data
            smooth_days_baseline=11,  # Reduced for test data
            window_days_hobday=5,  # Reduced for test data
            dimensions=custom_dimensions,
            coordinates=custom_coordinates,
            dask_chunks={"t": 25},
            neighbours=mock_neighbours_renamed,
            cell_areas=mock_cell_areas_renamed,
        )

        # Verify output structure for shifting_baseline method
        assert isinstance(extremes_ds_shifting, xr.Dataset)
        assert "extreme_events" in extremes_ds_shifting.data_vars
        assert "dat_anomaly" in extremes_ds_shifting.data_vars
        assert "thresholds" in extremes_ds_shifting.data_vars
        assert "mask" in extremes_ds_shifting.data_vars
        assert "neighbours" in extremes_ds_shifting.data_vars
        assert "cell_areas" in extremes_ds_shifting.data_vars

        # Verify dimensions are correctly renamed
        assert "t" in extremes_ds_shifting.extreme_events.dims
        assert "cell" in extremes_ds_shifting.extreme_events.dims

        # Verify no regular grid dimensions present
        assert "latitude" not in extremes_ds_shifting.extreme_events.dims
        assert "longitude" not in extremes_ds_shifting.extreme_events.dims

        # Verify coordinates are present
        assert "latitude" in extremes_ds_shifting.coords
        assert "longitude" in extremes_ds_shifting.coords
        assert extremes_ds_shifting.latitude.dims == ("cell",)
        assert extremes_ds_shifting.longitude.dims == ("cell",)

        # Verify attributes for shifting_baseline
        assert extremes_ds_shifting.attrs["method_anomaly"] == "shifting_baseline"
        assert extremes_ds_shifting.attrs["method_extreme"] == "hobday_extreme"

        # For hobday_extreme, thresholds should have dayofyear dimension
        assert "dayofyear" in extremes_ds_shifting.thresholds.dims
        assert "cell" in extremes_ds_shifting.thresholds.dims

        # Verify time dimension: shifting_baseline should reduce time
        input_time_size = renamed_data.sizes["t"]
        output_time_size = extremes_ds_shifting.sizes["t"]
        assert output_time_size < input_time_size, "shifting_baseline should reduce time dimension"

        # Verify reasonable extreme event frequency
        extreme_frequency_shifting = float(extremes_ds_shifting.extreme_events.mean())
        assert_percentile_frequency(
            extreme_frequency_shifting,
            95,
            description="Custom dimensions (unstructured): shifting_baseline + hobday_extreme",
        )

        # Test 3: Verify both methods produce consistent core structure
        core_vars = ["extreme_events", "dat_anomaly", "mask", "neighbours", "cell_areas"]
        for var in core_vars:
            assert var in extremes_ds_detrended.data_vars
            assert var in extremes_ds_shifting.data_vars

        # Both should have same cell dimension structure
        assert "cell" in extremes_ds_detrended.extreme_events.dims
        assert "cell" in extremes_ds_shifting.extreme_events.dims

        # Both should have consistent neighbours and cell_areas
        assert extremes_ds_detrended.neighbours.shape == extremes_ds_shifting.neighbours.shape
        assert extremes_ds_detrended.cell_areas.shape == extremes_ds_shifting.cell_areas.shape

        # Both should have same coordinate structure
        assert "latitude" in extremes_ds_detrended.coords
        assert "longitude" in extremes_ds_detrended.coords
        assert "latitude" in extremes_ds_shifting.coords
        assert "longitude" in extremes_ds_shifting.coords
