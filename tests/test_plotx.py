from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.plotX import PlotConfig, PlotXAccessor, _detect_grid_type, specify_grid
from marEx.plotX.base import PlotterBase
from marEx.plotX.gridded import GriddedPlotter
from marEx.plotX.unstructured import UnstructuredPlotter


class TestPlotConfig:
    """Test PlotConfig dataclass functionality."""

    def test_default_config_creation(self):
        """Test PlotConfig with default values."""
        config = PlotConfig()

        assert config.title is None
        assert config.var_units == ""
        assert config.issym is False
        assert config.cmap is None
        assert config.cperc == [4, 96]
        assert config.clim is None
        assert config.show_colorbar is True
        assert config.grid_lines is True
        assert config.grid_labels is False
        assert config.dimensions == {"time": "time", "y": "lat", "x": "lon"}
        assert config.norm is None
        assert config.plot_IDs is False
        assert config.extend == "both"

    def test_custom_config_creation(self):
        """Test PlotConfig with custom values."""
        config = PlotConfig(
            title="Test Plot",
            var_units="°C",
            issym=True,
            cmap="viridis",
            cperc=[5, 95],
            clim=(0, 10),
            show_colorbar=False,
            grid_lines=False,
            grid_labels=True,
            plot_IDs=True,
            extend="neither",
        )

        assert config.title == "Test Plot"
        assert config.var_units == "°C"
        assert config.issym is True
        assert config.cmap == "viridis"
        assert config.cperc == [5, 95]
        assert config.clim == (0, 10)
        assert config.show_colorbar is False  # Should be False due to plot_IDs=True
        assert config.grid_lines is False
        assert config.grid_labels is True
        assert config.plot_IDs is True
        assert config.extend == "neither"

    def test_plot_ids_disables_colorbar(self):
        """Test that plot_IDs=True disables colorbar."""
        config = PlotConfig(plot_IDs=True, show_colorbar=True)
        assert config.show_colorbar is False


class TestGridDetection:
    """Test grid type detection functionality."""

    def test_detect_gridded_data(self):
        """Test detection of gridded data structure."""
        # Create gridded data with lat/lon as dimensions
        data = xr.DataArray(
            np.random.random((10, 5, 8)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-90, 90, 5),
                "lon": np.linspace(-180, 180, 8),
            },
        )

        assert _detect_grid_type(data) == "gridded"

    def test_detect_unstructured_data(self):
        """Test detection of unstructured data structure."""
        # Create unstructured data with lat/lon as coordinates but not dimensions
        data = xr.DataArray(
            np.random.random((10, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(10),
                "lat": ("cell", np.random.uniform(-90, 90, 100)),
                "lon": ("cell", np.random.uniform(-180, 180, 100)),
            },
        )

        assert _detect_grid_type(data) == "unstructured"

    def test_detect_gridded_fallback(self):
        """Test fallback to gridded when structure is ambiguous."""
        # Create data without clear lat/lon structure
        data = xr.DataArray(
            np.random.random((10, 5, 8)), dims=["time", "lat", "lon"], coords={"time": range(10), "lat": range(5), "lon": range(8)}
        )

        assert _detect_grid_type(data) == "gridded"


class TestPlotterRegistration:
    """Test plotter registration and selection."""

    def test_register_gridded_plotter(self):
        """Test registration of gridded plotter."""
        data = xr.DataArray(
            np.random.random((10, 5, 8)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-90, 90, 5),
                "lon": np.linspace(-180, 180, 8),
            },
        )

        accessor = PlotXAccessor(data)
        plotter = accessor()
        assert isinstance(plotter, GriddedPlotter)
        assert plotter.da is data

    def test_register_unstructured_plotter(self):
        """Test registration of unstructured plotter."""
        data = xr.DataArray(
            np.random.random((10, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(10),
                "lat": ("cell", np.random.uniform(-90, 90, 100)),
                "lon": ("cell", np.random.uniform(-180, 180, 100)),
            },
        )

        # Need to specify custom dimensions since test data uses 'cell' not 'ncells'
        custom_dims = {"time": "time", "x": "cell"}
        custom_coords = {"time": "time", "x": "lon", "y": "lat"}

        accessor = PlotXAccessor(data)
        plotter = accessor(dimensions=custom_dims, coordinates=custom_coords)
        assert isinstance(plotter, UnstructuredPlotter)
        assert plotter.da is data

    def test_specify_grid_override(self):
        """Test that specify_grid overrides automatic detection."""
        # Create gridded data
        data = xr.DataArray(
            np.random.random((10, 5, 8)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-90, 90, 5),
                "lon": np.linspace(-180, 180, 8),
            },
        )

        # Specify as unstructured
        specify_grid(grid_type="unstructured")

        # Need to specify custom dimensions for gridded->unstructured override
        custom_dims = {"time": "time", "x": "lon"}  # No y for unstructured
        custom_coords = {"time": "time", "x": "lon", "y": "lat"}

        # The accessor should use the specified grid type (unstructured) despite detection
        accessor = PlotXAccessor(data)
        plotter = accessor(dimensions=custom_dims, coordinates=custom_coords)
        assert isinstance(plotter, UnstructuredPlotter)

        # Reset global state
        specify_grid(grid_type=None)

    def test_xarray_accessor_registration(self):
        """Test that plotX accessor is properly registered."""
        data = xr.DataArray(
            np.random.random((10, 5, 8)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-90, 90, 5),
                "lon": np.linspace(-180, 180, 8),
            },
        )

        # Check that accessor exists and returns proper accessor
        assert hasattr(data, "plotX")
        accessor = data.plotX
        assert isinstance(accessor, PlotXAccessor)

        # Check that calling the accessor returns proper plotter
        plotter = data.plotX()
        assert isinstance(plotter, GriddedPlotter)


class TestSpecifyGrid:
    """Test grid specification functionality."""

    def test_specify_grid_valid_types(self):
        """Test specify_grid with valid grid types."""
        specify_grid(grid_type="gridded")
        specify_grid(grid_type="unstructured")
        specify_grid(grid_type="GRIDDED")  # Should work with uppercase
        specify_grid(grid_type="UNSTRUCTURED")

        # Reset
        specify_grid(grid_type=None)

    def test_specify_grid_invalid_type(self):
        """Test specify_grid with invalid grid type."""
        from marEx.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Invalid grid type specification"):
            specify_grid(grid_type="invalid")

    def test_specify_grid_with_paths(self):
        """Test specify_grid with file paths."""
        tgrid_path = Path("/tmp/test_tgrid.nc")
        ckdtree_path = Path("/tmp/test_ckdtree")

        specify_grid(grid_type="unstructured", fpath_tgrid=tgrid_path, fpath_ckdtree=ckdtree_path)

        # Create test data
        data = xr.DataArray(
            np.random.random((10, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(10),
                "lat": ("cell", np.random.uniform(-90, 90, 100)),
                "lon": ("cell", np.random.uniform(-180, 180, 100)),
            },
        )

        # Need to specify custom dimensions since test data uses 'cell' not 'ncells'
        custom_dims = {"time": "time", "x": "cell"}
        custom_coords = {"time": "time", "x": "lon", "y": "lat"}

        accessor = PlotXAccessor(data)
        plotter = accessor(dimensions=custom_dims, coordinates=custom_coords)
        assert isinstance(plotter, UnstructuredPlotter)
        assert str(plotter.fpath_tgrid) == str(tgrid_path)
        assert str(plotter.fpath_ckdtree) == str(ckdtree_path)

        # Reset
        specify_grid(grid_type=None)


class TestPlotterBase:
    """Test PlotterBase functionality."""

    def setup_method(self):
        """Create test data for each test."""
        self.gridded_data = xr.DataArray(
            np.random.random((10, 5, 8)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(10),
                "lat": np.linspace(-90, 90, 5),
                "lon": np.linspace(-180, 180, 8),
            },
        )

    def test_plotter_base_initialisation(self):
        """Test PlotterBase initialisation."""

        # Create a concrete subclass for testing
        class TestPlotter(PlotterBase):
            def plot(self, ax, cmap="viridis", clim=None, norm=None):
                return ax, MagicMock()

        plotter = TestPlotter(self.gridded_data)
        assert plotter.da is self.gridded_data
        assert plotter.dimensions == {"time": "time", "y": "lat", "x": "lon"}
        assert plotter.coordinates == {"time": "time", "y": "lat", "x": "lon"}
        assert plotter._land is not None
        assert plotter._coastlines is not None

    def test_plotter_base_custom_dimensions(self):
        """Test PlotterBase with custom dimensions and coordinates."""

        class TestPlotter(PlotterBase):
            def plot(self, ax, cmap="viridis", clim=None, norm=None):
                return ax, MagicMock()

        custom_dims = {"time": "time", "x": "lon", "y": "lat"}
        custom_coords = {"time": "time", "x": "lon", "y": "lat"}

        plotter = TestPlotter(self.gridded_data, dimensions=custom_dims, coordinates=custom_coords)
        assert plotter.da is self.gridded_data
        assert plotter.dimensions == custom_dims
        assert plotter.coordinates == custom_coords
        assert plotter._land is not None
        assert plotter._coastlines is not None

    def test_clim_robust_symmetric(self):
        """Test robust color limit calculation for symmetric data."""

        class TestPlotter(PlotterBase):
            def plot(self, ax, cmap="viridis", clim=None, norm=None):
                return ax, MagicMock()

        plotter = TestPlotter(self.gridded_data)

        # Test with symmetric data
        data = np.array([-5, -2, 0, 2, 5])
        clim = plotter.clim_robust(data, issym=True, percentiles=[10, 90])

        assert len(clim) == 2
        assert clim[0] == -clim[1]  # Should be symmetric
        assert clim[1] > 0

    def test_clim_robust_asymmetric(self):
        """Test robust color limit calculation for asymmetric data."""

        class TestPlotter(PlotterBase):
            def plot(self, ax, cmap="viridis", clim=None, norm=None):
                return ax, MagicMock()

        plotter = TestPlotter(self.gridded_data)

        # Test with asymmetric data
        data = np.array([0, 1, 2, 3, 4, 5])
        clim = plotter.clim_robust(data, issym=False, percentiles=[10, 90])

        assert len(clim) == 2
        assert clim[0] < clim[1]
        assert clim[0] >= 0  # Should respect lower bound

    def test_clim_robust_with_zero_percentile(self):
        """Test color limit calculation with zero percentile."""

        class TestPlotter(PlotterBase):
            def plot(self, ax, cmap="viridis", clim=None, norm=None):
                return ax, MagicMock()

        plotter = TestPlotter(self.gridded_data)

        data = np.array([1, 2, 3, 4, 5])
        clim = plotter.clim_robust(data, issym=False, percentiles=[0, 90])

        assert clim[0] == 0  # Should be set to 0 when percentile is 0
        assert clim[1] > 0

    def test_setup_plot_params(self):
        """Test plot parameter setup."""

        class TestPlotter(PlotterBase):
            def plot(self, ax, cmap="viridis", clim=None, norm=None):
                return ax, MagicMock()

        plotter = TestPlotter(self.gridded_data)

        # Should not raise any errors
        plotter.setup_plot_params()

    def test_setup_id_plot_params(self):
        """Test ID plot parameter setup."""

        class TestPlotter(PlotterBase):
            def plot(self, ax, cmap="viridis", clim=None, norm=None):
                return ax, MagicMock()

        # Create data with integer IDs
        id_data = xr.DataArray(
            np.array([[[1, 2, 0], [3, 1, 0]], [[2, 3, 1], [0, 2, 3]]]),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(2),
                "lat": np.linspace(-90, 90, 2),
                "lon": np.linspace(-180, 180, 3),
            },
        )

        plotter = TestPlotter(id_data)
        cmap, norm, var_units = plotter.setup_id_plot_params()

        assert var_units == "ID"
        assert cmap is not None
        assert norm is not None

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.axes")
    def test_setup_axes_without_ax(self, mock_axes, mock_figure):
        """Test axes setup when no axes provided."""

        class TestPlotter(PlotterBase):
            def plot(self, ax, cmap="viridis", clim=None, norm=None):
                return ax, MagicMock()

        plotter = TestPlotter(self.gridded_data)

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_axes.return_value = mock_ax

        fig, ax = plotter._setup_axes()

        mock_figure.assert_called_once()
        mock_axes.assert_called_once()
        assert fig == mock_fig
        assert ax == mock_ax

    def test_setup_axes_with_ax(self):
        """Test axes setup when axes provided."""

        class TestPlotter(PlotterBase):
            def plot(self, ax, cmap="viridis", clim=None, norm=None):
                return ax, MagicMock()

        plotter = TestPlotter(self.gridded_data)

        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_ax.get_figure.return_value = mock_fig

        fig, ax = plotter._setup_axes(ax=mock_ax)

        assert fig == mock_fig
        assert ax == mock_ax
        mock_ax.get_figure.assert_called_once()


class TestPlotXWithTestData:
    """Test plotX functionality using actual test data."""

    @classmethod
    def setup_class(cls):
        """Load test data for all tests."""
        # Reset global grid state to avoid pollution from other tests
        specify_grid(grid_type=None, fpath_tgrid=None, fpath_ckdtree=None)

        test_data_path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        if test_data_path.exists():
            ds = xr.open_zarr(str(test_data_path), chunks={})
            # Extract the 'to' data variable which should be gridded
            cls.sst_data = ds.to
        else:
            # Create minimal test data if file doesn't exist
            cls.sst_data = xr.DataArray(
                np.random.random((10, 5, 8)),
                dims=["time", "lat", "lon"],
                coords={
                    "time": range(10),
                    "lat": np.linspace(-90, 90, 5),
                    "lon": np.linspace(-180, 180, 8),
                },
            )

    def test_plotx_accessor_works(self):
        """Test that plotX accessor works with test data."""
        plotter = self.sst_data.plotX()
        assert isinstance(plotter, GriddedPlotter)
        assert plotter.da is self.sst_data

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.axes")
    def test_single_plot_creation(self, mock_axes, mock_figure):
        """Test single plot creation."""
        config = PlotConfig(title="Test Plot")

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_axes.return_value = mock_ax

        plotter = self.sst_data.plotX()

        # Mock the plot method to avoid actual plotting based on plotter type
        with patch.object(type(plotter), "plot", return_value=(mock_ax, MagicMock())):
            fig, ax, im = plotter.single_plot(config)

            assert fig == mock_fig
            assert ax == mock_ax
            mock_ax.set_title.assert_called_once_with("Test Plot", size=12)


class TestCustomDimensions:
    """Test custom dimensions and coordinates functionality."""

    def test_custom_dimensions_gridded(self):
        """Test custom dimensions with gridded data."""
        data = xr.DataArray(
            np.random.random((3, 4, 5)),
            dims=["t", "latitude", "longitude"],
            coords={
                "t": range(3),
                "latitude": np.linspace(-20, 20, 4),
                "longitude": np.linspace(-25, 25, 5),
            },
        )

        custom_dims = {"time": "t", "x": "longitude", "y": "latitude"}
        custom_coords = {"time": "t", "x": "longitude", "y": "latitude"}

        accessor = PlotXAccessor(data)
        plotter = accessor(dimensions=custom_dims, coordinates=custom_coords)

        assert isinstance(plotter, GriddedPlotter)
        assert plotter.dimensions == custom_dims
        assert plotter.coordinates == custom_coords
        assert plotter.da is data

    def test_custom_dimensions_unstructured(self):
        """Test custom dimensions with unstructured data."""
        data = xr.DataArray(
            np.random.random((3, 100)),
            dims=["t", "cells"],
            coords={
                "t": range(3),
                "x_coord": (["cells"], np.random.rand(100)),
                "y_coord": (["cells"], np.random.rand(100)),
            },
        )

        custom_dims = {"time": "t", "x": "cells"}
        custom_coords = {"time": "t", "x": "x_coord", "y": "y_coord"}

        accessor = PlotXAccessor(data)
        plotter = accessor(dimensions=custom_dims, coordinates=custom_coords)

        assert isinstance(plotter, UnstructuredPlotter)
        assert plotter.dimensions == custom_dims
        assert plotter.coordinates == custom_coords
        assert plotter.da is data

    def test_custom_detection_logic(self):
        """Test grid type detection with custom dimensions."""
        # Test that custom dimensions are used for detection
        data = xr.DataArray(
            np.random.random((3, 4, 5)),
            dims=["t", "spatial_y", "spatial_x"],
            coords={
                "t": range(3),
                "spatial_y": np.linspace(-20, 20, 4),
                "spatial_x": np.linspace(-25, 25, 5),
            },
        )

        # With custom dimensions mapping y dimension
        custom_dims = {"time": "t", "x": "spatial_x", "y": "spatial_y"}
        grid_type = _detect_grid_type(data, dimensions=custom_dims)
        assert grid_type == "gridded"

        # Without y dimension mapping (should be unstructured)
        custom_dims_no_y = {"time": "t", "x": "spatial_x"}
        grid_type = _detect_grid_type(data, dimensions=custom_dims_no_y)
        assert grid_type == "unstructured"

    def test_xarray_accessor_with_custom_dims(self):
        """Test xarray accessor with custom dimensions."""
        data = xr.DataArray(
            np.random.random((3, 4, 5)),
            dims=["t", "y_custom", "x_custom"],
            coords={
                "t": range(3),
                "y_custom": np.linspace(-20, 20, 4),
                "x_custom": np.linspace(-25, 25, 5),
            },
        )

        custom_dims = {"time": "t", "x": "x_custom", "y": "y_custom"}
        custom_coords = {"time": "t", "x": "x_custom", "y": "y_custom"}

        # Test accessor with custom parameters
        plotter = data.plotX(dimensions=custom_dims, coordinates=custom_coords)
        assert isinstance(plotter, GriddedPlotter)
        assert plotter.dimensions == custom_dims
        assert plotter.coordinates == custom_coords

    def test_plotconfig_with_custom_params(self):
        """Test PlotConfig with custom dimensions and coordinates."""
        custom_dims = {"time": "time_var", "x": "x_var", "y": "y_var"}
        custom_coords = {"time": "time_coord", "x": "x_coord", "y": "y_coord"}

        config = PlotConfig(
            title="Custom Config Test",
            dimensions=custom_dims,
            coordinates=custom_coords,
            issym=True,
        )

        assert config.dimensions == custom_dims
        assert config.coordinates == custom_coords
        assert config.title == "Custom Config Test"
        assert config.issym is True


class TestErrorHandling:
    """Test error handling for invalid dimensions and coordinates."""

    def test_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        data = xr.DataArray(
            np.random.random((3, 4, 5)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(3),
                "lat": np.linspace(-20, 20, 4),
                "lon": np.linspace(-25, 25, 5),
            },
        )

        invalid_dims = {"time": "invalid_time", "x": "invalid_x", "y": "invalid_y"}

        accessor = PlotXAccessor(data)
        with pytest.raises(marEx.VisualisationError):
            accessor(dimensions=invalid_dims)

    def test_invalid_coordinates(self):
        """Test error handling for invalid coordinates."""
        data = xr.DataArray(
            np.random.random((3, 4, 5)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(3),
                "lat": np.linspace(-20, 20, 4),
                "lon": np.linspace(-25, 25, 5),
            },
        )

        invalid_coords = {"time": "invalid_time", "x": "invalid_x", "y": "invalid_y"}

        accessor = PlotXAccessor(data)
        with pytest.raises(marEx.VisualisationError):
            accessor(coordinates=invalid_coords)

    def test_partial_invalid_dimensions(self):
        """Test error handling when only some dimensions are invalid."""
        data = xr.DataArray(
            np.random.random((3, 4, 5)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(3),
                "lat": np.linspace(-20, 20, 4),
                "lon": np.linspace(-25, 25, 5),
            },
        )

        # Mix of valid and invalid dimensions
        mixed_dims = {"time": "time", "x": "invalid_x", "y": "lat"}

        accessor = PlotXAccessor(data)
        with pytest.raises(marEx.VisualisationError):
            accessor(dimensions=mixed_dims)


class TestBackwardCompatibility:
    """Test backward compatibility of the updated plotX functionality."""

    def test_default_accessor_call(self):
        """Test that default accessor call still works."""
        data = xr.DataArray(
            np.random.random((3, 4, 5)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(3),
                "lat": np.linspace(-20, 20, 4),
                "lon": np.linspace(-25, 25, 5),
            },
        )

        # Default call should work as before
        plotter = data.plotX()
        assert isinstance(plotter, GriddedPlotter)
        assert plotter.dimensions == {"time": "time", "y": "lat", "x": "lon"}
        assert plotter.coordinates == {"time": "time", "y": "lat", "x": "lon"}

    def test_accessor_convenience_methods(self):
        """Test that accessor convenience methods exist."""
        data = xr.DataArray(
            np.random.random((3, 4, 5)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(3),
                "lat": np.linspace(-20, 20, 4),
                "lon": np.linspace(-25, 25, 5),
            },
        )

        accessor = data.plotX

        # Check that convenience methods exist
        assert hasattr(accessor, "single_plot")
        assert hasattr(accessor, "multi_plot")
        assert hasattr(accessor, "animate")
        assert callable(accessor.single_plot)
        assert callable(accessor.multi_plot)
        assert callable(accessor.animate)

    def test_plotconfig_default_compatibility(self):
        """Test that PlotConfig defaults are backward compatible."""
        config = PlotConfig()

        # Default dimensions and coordinates should match original behavior
        expected_defaults = {"time": "time", "y": "lat", "x": "lon"}
        assert config.dimensions == expected_defaults
        assert config.coordinates == expected_defaults

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.axes")
    def test_single_plot_with_defaults(self, mock_axes, mock_figure):
        """Test that single_plot works with default parameters."""
        data = xr.DataArray(
            np.random.random((3, 4, 5)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(3),
                "lat": np.linspace(-20, 20, 4),
                "lon": np.linspace(-25, 25, 5),
            },
        )

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_axes.return_value = mock_ax

        config = PlotConfig(title="Backward Compatibility Test")
        plotter = data.plotX()

        # Mock the plot method to avoid actual plotting
        with patch.object(type(plotter), "plot", return_value=(mock_ax, MagicMock())):
            fig, ax, im = plotter.single_plot(config)

            assert fig == mock_fig
            assert ax == mock_ax
            mock_ax.set_title.assert_called_once_with("Backward Compatibility Test", size=12)


class TestGriddedPlotterCoverage:
    """Test GriddedPlotter functionality with detailed coverage."""

    def setup_method(self):
        """Create test data for each test."""
        # Create gridded test data
        self.gridded_data = xr.DataArray(
            np.random.random((5, 4, 8)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(5),
                "lat": np.linspace(-20, 20, 4),
                "lon": np.linspace(-180, 180, 8),
            },
        )

        # Create global data that needs longitude wrapping
        self.global_data = xr.DataArray(
            np.random.random((2, 3, 6)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(2),
                "lat": np.linspace(-90, 90, 3),
                "lon": np.array([0, 60, 120, 180, 240, 300]),  # Global coverage
            },
        )

    def test_wrap_lon_global_data(self):
        """Test wrap_lon with global data that needs wrapping."""
        plotter = GriddedPlotter(self.global_data)

        wrapped = plotter.wrap_lon(self.global_data)

        # Should add one more longitude point
        assert wrapped.shape[2] == self.global_data.shape[2] + 1
        assert wrapped.dims == self.global_data.dims

        # Check that new longitude is added correctly
        original_lons = self.global_data.lon.values
        wrapped_lons = wrapped.lon.values
        assert len(wrapped_lons) == len(original_lons) + 1
        assert wrapped_lons[-1] == original_lons[0] + 360

    @patch("marEx.plotX.gridded.logger")
    @patch("marEx.plotX.gridded.log_timing")
    def test_plot_basic_functionality(self, mock_log_timing, mock_logger):
        """Test basic plot functionality."""
        plotter = GriddedPlotter(self.gridded_data)

        # Mock matplotlib components
        mock_ax = MagicMock()
        mock_im = MagicMock()
        mock_ax.pcolormesh.return_value = mock_im

        # Mock log_timing context manager
        mock_log_timing.return_value.__enter__ = MagicMock()
        mock_log_timing.return_value.__exit__ = MagicMock()

        # Mock ccrs import
        with patch("marEx.plotX.gridded.ccrs") as mock_ccrs:
            mock_ccrs.PlateCarree.return_value = "mock_transform"

            # Select single time slice to avoid squeeze operation
            single_time_data = self.gridded_data.isel(time=0)
            plotter.da = single_time_data

            result_ax, result_im = plotter.plot(mock_ax)

            # Verify results
            assert result_ax is mock_ax
            assert result_im is mock_im

            # Verify pcolormesh was called
            mock_ax.pcolormesh.assert_called_once()

            # Verify logging
            mock_logger.debug.assert_called()

    @patch("marEx.plotX.gridded.logger")
    @patch("marEx.plotX.gridded.log_timing")
    def test_plot_with_clim_parameter(self, mock_log_timing, mock_logger):
        """Test plot with color limits specified."""
        plotter = GriddedPlotter(self.gridded_data)

        mock_ax = MagicMock()
        mock_im = MagicMock()
        mock_ax.pcolormesh.return_value = mock_im

        # Mock log_timing context manager
        mock_log_timing.return_value.__enter__ = MagicMock()
        mock_log_timing.return_value.__exit__ = MagicMock()

        with patch("marEx.plotX.gridded.ccrs") as mock_ccrs:
            mock_ccrs.PlateCarree.return_value = "mock_transform"

            single_time_data = self.gridded_data.isel(time=0)
            plotter.da = single_time_data

            result_ax, result_im = plotter.plot(mock_ax, clim=(0, 1))

            # Verify pcolormesh was called with vmin/vmax
            call_args = mock_ax.pcolormesh.call_args
            kwargs = call_args[1]
            assert "vmin" in kwargs
            assert "vmax" in kwargs
            assert kwargs["vmin"] == 0
            assert kwargs["vmax"] == 1


class TestUnstructuredPlotterCoverage:
    """Test UnstructuredPlotter functionality with detailed coverage."""

    def setup_method(self):
        """Create test data for each test."""
        # Create unstructured test data
        self.unstructured_data = xr.DataArray(
            np.random.random((5, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(5),
                "lat": ("cell", np.random.uniform(-90, 90, 100)),
                "lon": ("cell", np.random.uniform(-180, 180, 100)),
            },
        )

        self.custom_dims = {"time": "time", "x": "cell"}
        self.custom_coords = {"time": "time", "x": "lon", "y": "lat"}

    def test_unstructured_plotter_initialization(self):
        """Test UnstructuredPlotter initialization."""
        plotter = UnstructuredPlotter(self.unstructured_data, dimensions=self.custom_dims, coordinates=self.custom_coords)

        assert plotter.da is self.unstructured_data
        assert plotter.dimensions == self.custom_dims
        assert plotter.coordinates == self.custom_coords

    def test_specify_grid_method(self):
        """Test specify_grid method with various inputs."""
        plotter = UnstructuredPlotter(self.unstructured_data, dimensions=self.custom_dims, coordinates=self.custom_coords)

        # Test with string paths
        tgrid_path = "/path/to/triangulation.nc"
        ckdtree_path = "/path/to/ckdtree"

        plotter.specify_grid(fpath_tgrid=tgrid_path, fpath_ckdtree=ckdtree_path)

        assert str(plotter.fpath_tgrid).replace("\\", "/") == tgrid_path
        assert str(plotter.fpath_ckdtree).replace("\\", "/") == ckdtree_path

    def test_specify_grid_with_path_objects(self):
        """Test specify_grid method with Path objects."""
        from pathlib import Path

        plotter = UnstructuredPlotter(self.unstructured_data, dimensions=self.custom_dims, coordinates=self.custom_coords)

        # Test with Path objects
        tgrid_path = Path("/path/to/triangulation.nc")
        ckdtree_path = Path("/path/to/ckdtree")

        plotter.specify_grid(fpath_tgrid=tgrid_path, fpath_ckdtree=ckdtree_path)

        assert plotter.fpath_tgrid == tgrid_path
        assert plotter.fpath_ckdtree == ckdtree_path

    def test_specify_grid_partial_paths(self):
        """Test specify_grid with only some paths specified."""
        plotter = UnstructuredPlotter(self.unstructured_data, dimensions=self.custom_dims, coordinates=self.custom_coords)

        # Test with only tgrid path
        tgrid_path = "/path/to/triangulation.nc"
        plotter.specify_grid(fpath_tgrid=tgrid_path)

        assert str(plotter.fpath_tgrid).replace("\\", "/") == tgrid_path
        assert plotter.fpath_ckdtree is None

    def test_specify_grid_none_values(self):
        """Test specify_grid with None values."""
        plotter = UnstructuredPlotter(self.unstructured_data, dimensions=self.custom_dims, coordinates=self.custom_coords)

        # Set some paths first
        plotter.specify_grid(fpath_tgrid="/some/path", fpath_ckdtree="/another/path")

        # Then clear them with None
        plotter.specify_grid(fpath_tgrid=None, fpath_ckdtree=None)

        assert plotter.fpath_tgrid is None
        assert plotter.fpath_ckdtree is None


class TestUnstructuredUtilityFunctions:
    """Test unstructured module utility functions."""

    def test_clear_cache_function(self):
        """Test clear_cache function."""
        from marEx.plotX.unstructured import _GRID_CACHE, clear_cache

        # Add some dummy data to cache
        _GRID_CACHE["triangulation"]["test_key"] = "test_value"
        _GRID_CACHE["ckdtree"]["test_key"] = "test_value"

        assert len(_GRID_CACHE["triangulation"]) > 0
        assert len(_GRID_CACHE["ckdtree"]) > 0

        # Clear cache
        clear_cache()

        assert len(_GRID_CACHE["triangulation"]) == 0
        assert len(_GRID_CACHE["ckdtree"]) == 0

    def test_load_triangulation_missing_variables(self):
        """Test _load_triangulation with missing required variables."""
        from marEx.exceptions import DataValidationError
        from marEx.plotX.unstructured import _load_triangulation

        # Mock xarray.open_dataset to return dataset without required variables
        mock_dataset = MagicMock()
        mock_dataset.variables = {"other_var": MagicMock()}  # Missing required vars

        with patch("marEx.plotX.unstructured.xr.open_dataset", return_value=mock_dataset):
            with pytest.raises(DataValidationError, match="Invalid triangulation grid file format"):
                _load_triangulation("/fake/path.nc")

    def test_load_triangulation_success(self):
        """Test _load_triangulation with valid data."""
        from marEx.plotX.unstructured import _load_triangulation, clear_cache

        # Clear cache first
        clear_cache()

        # Mock valid dataset
        mock_dataset = MagicMock()
        mock_dataset.variables = {
            "vertex_of_cell": MagicMock(),
            "clon": MagicMock(),
            "clat": MagicMock(),
        }
        mock_dataset.vertex_of_cell.values = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]).T
        mock_dataset.clon.values = np.array([0, 30, 60])
        mock_dataset.clat.values = np.array([0, 10, 20])
        mock_dataset.close = MagicMock()

        with patch("marEx.plotX.unstructured.xr.open_dataset", return_value=mock_dataset):
            with patch("marEx.plotX.unstructured.Triangulation") as mock_triangulation:
                mock_triang = MagicMock()
                mock_triangulation.return_value = mock_triang

                result = _load_triangulation("/test/path.nc")

                assert result is mock_triang
                mock_dataset.close.assert_called_once()
                mock_triangulation.assert_called_once()

    def test_load_triangulation_cached(self):
        """Test _load_triangulation returns cached result."""
        from marEx.plotX.unstructured import _GRID_CACHE, _load_triangulation, clear_cache

        clear_cache()

        test_path = "/test/path.nc"
        cached_triangulation = MagicMock()

        # Pre-populate cache
        _GRID_CACHE["triangulation"][test_path] = cached_triangulation

        result = _load_triangulation(test_path)

        assert result is cached_triangulation

    def test_load_ckdtree_file_not_found(self):
        """Test _load_ckdtree with missing file."""
        from marEx.exceptions import DataValidationError
        from marEx.plotX.unstructured import _load_ckdtree

        with patch("marEx.plotX.unstructured.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_ckdtree_file = MagicMock()
            mock_ckdtree_file.exists.return_value = False
            mock_path.__truediv__.return_value = mock_ckdtree_file
            mock_path_class.return_value = mock_path

            with pytest.raises(DataValidationError, match="KDTree file not found"):
                _load_ckdtree("/test/path", 0.25)

    def test_load_ckdtree_success(self):
        """Test _load_ckdtree with valid data."""
        from marEx.plotX.unstructured import _load_ckdtree, clear_cache

        clear_cache()

        mock_dataset = MagicMock()
        mock_dataset.ickdtree_c.values = np.array([0, 1, 2, 3])
        mock_dataset.lon.values = np.array([0, 30, 60])
        mock_dataset.lat.values = np.array([-30, 0, 30])
        mock_dataset.close = MagicMock()

        with patch("marEx.plotX.unstructured.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_ckdtree_file = MagicMock()
            mock_ckdtree_file.exists.return_value = True
            mock_path.__truediv__.return_value = mock_ckdtree_file
            mock_path_class.return_value = mock_path

            with patch("marEx.plotX.unstructured.xr.open_dataset", return_value=mock_dataset):
                result = _load_ckdtree("/test/path", 0.25)

                assert "indices" in result
                assert "lon" in result
                assert "lat" in result
                assert np.array_equal(result["indices"], mock_dataset.ickdtree_c.values)
                mock_dataset.close.assert_called_once()

    def test_load_ckdtree_cached(self):
        """Test _load_ckdtree returns cached result."""
        from marEx.plotX.unstructured import _GRID_CACHE, _load_ckdtree, clear_cache

        clear_cache()

        test_path = "/test/path"
        test_res = 0.25
        cache_key = (test_path, test_res)
        cached_data = {"indices": np.array([0, 1, 2]), "lon": np.array([0, 30]), "lat": np.array([0, 30])}

        # Pre-populate cache
        _GRID_CACHE["ckdtree"][cache_key] = cached_data

        result = _load_ckdtree(test_path, test_res)

        assert result is cached_data


class TestPlotXImportErrorHandling:
    """Test behavior when plotting dependencies are not available."""

    def test_gridded_without_plotting_deps(self):
        """Test GriddedPlotter when plotting dependencies are missing."""
        gridded_data = xr.DataArray(
            np.random.random((5, 4, 8)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(5),
                "lat": np.linspace(-20, 20, 4),
                "lon": np.linspace(-180, 180, 8),
            },
        )

        with patch("marEx.plotX.gridded.HAS_PLOTTING_DEPS", False):
            # Should still be able to create plotter
            plotter = GriddedPlotter(gridded_data)
            assert plotter is not None

    def test_unstructured_without_plotting_deps(self):
        """Test UnstructuredPlotter when plotting dependencies are missing."""
        unstructured_data = xr.DataArray(
            np.random.random((5, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(5),
                "lat": ("cell", np.random.uniform(-90, 90, 100)),
                "lon": ("cell", np.random.uniform(-180, 180, 100)),
            },
        )

        custom_dims = {"time": "time", "x": "cell"}
        custom_coords = {"time": "time", "x": "lon", "y": "lat"}

        with patch("marEx.plotX.unstructured.HAS_PLOTTING_DEPS", False):
            # Should still be able to create plotter
            plotter = UnstructuredPlotter(unstructured_data, dimensions=custom_dims, coordinates=custom_coords)
            assert plotter is not None


class TestPlotXEdgeCases:
    """Test edge cases and error conditions in plotX modules."""

    def test_gridded_wrap_lon_edge_cases(self):
        """Test wrap_lon method edge cases."""
        # Test with exactly 360-degree span
        exact_global_data = xr.DataArray(
            np.random.random((2, 3, 4)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(2),
                "lat": np.linspace(-90, 90, 3),
                "lon": np.array([0, 120, 240, 359.9]),  # Almost exactly 360 degrees
            },
        )

        plotter = GriddedPlotter(exact_global_data)
        wrapped = plotter.wrap_lon(exact_global_data)

        # Should add longitude wrapping
        assert wrapped.shape[2] == exact_global_data.shape[2] + 1

    def test_gridded_wrap_lon_no_wrapping(self):
        """Test wrap_lon when no wrapping is needed."""
        regional_data = xr.DataArray(
            np.random.random((2, 3, 4)),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(2),
                "lat": np.linspace(-90, 90, 3),
                "lon": np.array([0, 30, 60, 90]),  # Regional data
            },
        )

        plotter = GriddedPlotter(regional_data)
        wrapped = plotter.wrap_lon(regional_data)

        # Should not add longitude wrapping
        assert wrapped.shape[2] == regional_data.shape[2]

    def test_unstructured_interpolate_with_ckdtree_no_path(self):
        """Test _interpolate_with_ckdtree when no ckdtree path is set."""
        from marEx.exceptions import VisualisationError

        unstructured_data = xr.DataArray(
            np.random.random((5, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(5),
                "lat": ("cell", np.random.uniform(-90, 90, 100)),
                "lon": ("cell", np.random.uniform(-180, 180, 100)),
            },
        )

        custom_dims = {"time": "time", "x": "cell"}
        custom_coords = {"time": "time", "x": "lon", "y": "lat"}

        plotter = UnstructuredPlotter(unstructured_data, dimensions=custom_dims, coordinates=custom_coords)
        plotter.fpath_ckdtree = None  # Ensure no path is set

        test_data = np.array([10, 20, 30])

        with pytest.raises(VisualisationError, match="KDTree path not specified"):
            plotter._interpolate_with_ckdtree(test_data, res=0.3)

    def test_unstructured_interpolate_with_ckdtree_success(self):
        """Test _interpolate_with_ckdtree with valid data."""
        from pathlib import Path

        unstructured_data = xr.DataArray(
            np.random.random((5, 100)),
            dims=["time", "cell"],
            coords={
                "time": range(5),
                "lat": ("cell", np.random.uniform(-90, 90, 100)),
                "lon": ("cell", np.random.uniform(-180, 180, 100)),
            },
        )

        custom_dims = {"time": "time", "x": "cell"}
        custom_coords = {"time": "time", "x": "lon", "y": "lat"}

        plotter = UnstructuredPlotter(unstructured_data, dimensions=custom_dims, coordinates=custom_coords)
        plotter.fpath_ckdtree = Path("/test/ckdtree")

        # Mock ckdtree data
        mock_ckdt_data = {
            "indices": np.array([0, 1, 2, 0, 1, 2]),  # 6 indices for 2x3 grid
            "lon": np.array([0, 30, 60]),  # 3 longitudes
            "lat": np.array([-30, 30]),  # 2 latitudes
        }

        test_data = np.array([10, 20, 30])  # 3 data points
        expected_grid_shape = (2, 3)  # lat x lon

        with patch("marEx.plotX.unstructured._load_ckdtree", return_value=mock_ckdt_data):
            with patch("marEx.plotX.unstructured.np.meshgrid") as mock_meshgrid:
                mock_lon_2d = np.ones(expected_grid_shape)
                mock_lat_2d = np.ones(expected_grid_shape)
                mock_meshgrid.return_value = (mock_lon_2d, mock_lat_2d)

                grid_lon, grid_lat, grid_data = plotter._interpolate_with_ckdtree(test_data, res=0.3)

                assert grid_lon.shape == expected_grid_shape
                assert grid_lat.shape == expected_grid_shape
                assert grid_data.shape == expected_grid_shape


class TestPlotConfigurationEdgeCases:
    """Test edge cases in plot configuration and customisation."""

    def test_plot_ids_configuration(self):
        """Test plotting with plot_IDs=True configuration."""
        # Create data with tracked IDs
        data = xr.DataArray(
            np.array([[[0, 1, 2], [1, 2, 0], [2, 0, 1]]]),
            dims=["time", "lat", "lon"],
            coords={
                "time": [0],
                "lat": np.linspace(-10, 10, 3),
                "lon": np.linspace(0, 20, 3),
            },
        )

        config = PlotConfig(plot_IDs=True, show_colorbar=True)

        plotter = GriddedPlotter(data)
        fig, ax, im = plotter.single_plot(config)

        # Verify plot was created
        assert fig is not None
        assert ax is not None
        assert im is not None

        # Verify show_colorbar was set to False by plot_IDs
        assert config.show_colorbar is False

    def test_custom_cmap_parameter(self):
        """Test plotting with custom colormap."""
        data = xr.DataArray(
            np.random.randn(1, 3, 3),
            dims=["time", "lat", "lon"],
            coords={
                "time": [0],
                "lat": np.linspace(-10, 10, 3),
                "lon": np.linspace(0, 20, 3),
            },
        )

        config = PlotConfig(cmap="plasma")
        plotter = GriddedPlotter(data)

        fig, ax, im = plotter.single_plot(config)

        assert fig is not None

    def test_custom_norm_parameter(self):
        """Test plotting with custom norm."""
        from matplotlib.colors import LogNorm

        data = xr.DataArray(
            np.abs(np.random.randn(1, 3, 3)) + 1,  # Positive values for LogNorm
            dims=["time", "lat", "lon"],
            coords={
                "time": [0],
                "lat": np.linspace(-10, 10, 3),
                "lon": np.linspace(0, 20, 3),
            },
        )

        config = PlotConfig(norm=LogNorm(vmin=0.1, vmax=10), clim=(0.1, 10))
        plotter = GriddedPlotter(data)

        fig, ax, im = plotter.single_plot(config)

        assert fig is not None

    def test_show_colorbar_false(self):
        """Test plotting with show_colorbar=False."""
        data = xr.DataArray(
            np.random.randn(1, 3, 3),
            dims=["time", "lat", "lon"],
            coords={
                "time": [0],
                "lat": np.linspace(-10, 10, 3),
                "lon": np.linspace(0, 20, 3),
            },
        )

        config = PlotConfig(show_colorbar=False)
        plotter = GriddedPlotter(data)

        fig, ax, im = plotter.single_plot(config)

        assert fig is not None

    def test_custom_var_units(self):
        """Test plotting with custom var_units."""
        data = xr.DataArray(
            np.random.randn(1, 3, 3),
            dims=["time", "lat", "lon"],
            coords={
                "time": [0],
                "lat": np.linspace(-10, 10, 3),
                "lon": np.linspace(0, 20, 3),
            },
        )

        config = PlotConfig(var_units="Test Units [°C]", show_colorbar=True)
        plotter = GriddedPlotter(data)

        fig, ax, im = plotter.single_plot(config)

        assert fig is not None

    def test_data_without_time_dimension(self):
        """Test plotting data without a time dimension."""
        # Create 2D data (no time dimension)
        data = xr.DataArray(
            np.random.randn(5, 8),
            dims=["lat", "lon"],
            coords={
                "lat": np.linspace(-10, 10, 5),
                "lon": np.linspace(0, 20, 8),
            },
        )

        config = PlotConfig()
        plotter = GriddedPlotter(data)

        # Should handle data without time dimension
        fig, ax, im = plotter.single_plot(config)

        assert fig is not None


class TestTitleGeneration:
    """Test title generation for different dimension types."""

    def test_title_generation_time_dimension(self):
        """Test title generation for time dimension."""
        import pandas as pd

        data = xr.DataArray(
            np.random.randn(3, 4, 5),
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "lat": np.linspace(-10, 10, 4),
                "lon": np.linspace(0, 20, 5),
            },
        )

        plotter = GriddedPlotter(data)
        title = plotter._get_title(time_index=0, col_name="time")

        # Should return formatted date string
        assert "2020" in title

    def test_title_generation_non_time_dimension(self):
        """Test title generation for non-time dimensions."""
        import pandas as pd

        # Create data with time and depth dimensions
        data = xr.DataArray(
            np.random.randn(3, 2, 4, 5),
            dims=["time", "depth", "lat", "lon"],
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "depth": [0, 10],
                "lat": np.linspace(-10, 10, 4),
                "lon": np.linspace(0, 20, 5),
            },
        )

        plotter = GriddedPlotter(data)

        # Test title for depth dimension (not the time dimension)
        title = plotter._get_title(time_index=0, col_name="depth")

        # Should return dimension=value format for non-time dimensions
        assert "depth" in title
        assert "=" in title

    def test_title_generation_custom_dimension(self):
        """Test title generation with a different value dimension."""
        import pandas as pd

        data = xr.DataArray(
            np.random.randn(3, 2, 4, 5),
            dims=["time", "level", "lat", "lon"],
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "level": [100, 200],
                "lat": np.linspace(-10, 10, 4),
                "lon": np.linspace(0, 20, 5),
            },
        )

        plotter = GriddedPlotter(data)

        # Test with level dimension
        title = plotter._get_title(time_index=0, col_name="level")

        assert "level" in title
        assert "=" in title


class TestGriddedPlotterEdgeCases:
    """Test edge cases specific to gridded plotter."""

    def test_gridded_plotter_with_minimal_data(self):
        """Test gridded plotter with minimal 2x2 grid."""
        data = xr.DataArray(
            np.array([[[1, 2], [3, 4]]]),
            dims=["time", "lat", "lon"],
            coords={
                "time": [0],
                "lat": [0, 1],
                "lon": [0, 1],
            },
        )

        config = PlotConfig()
        plotter = GriddedPlotter(data)

        fig, ax, im = plotter.single_plot(config)

        assert fig is not None

    def test_gridded_plotter_symmetric_colorscale(self):
        """Test gridded plotter with symmetric colorscale (issym=True)."""
        data = xr.DataArray(
            np.array([[[-5, -2, 0], [2, 4, 6]]]),
            dims=["time", "lat", "lon"],
            coords={
                "time": [0],
                "lat": [0, 1],
                "lon": [0, 1, 2],
            },
        )

        config = PlotConfig(issym=True)
        plotter = GriddedPlotter(data)

        fig, ax, im = plotter.single_plot(config)

        assert fig is not None
