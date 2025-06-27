import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
from pathlib import Path

import marEx
from marEx.plotX import (
    PlotConfig,
    _detect_grid_type,
    register_plotter,
    specify_grid,
)
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
        assert config.dimensions == {"time": "time", "ydim": "lat", "xdim": "lon"}
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
        data = xr.DataArray(np.random.random((10, 5, 8)), dims=["time", "y", "x"])

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

        plotter = register_plotter(data)
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

        plotter = register_plotter(data)
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

        with pytest.warns(UserWarning, match="Specified grid type"):
            plotter = register_plotter(data)
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

        # Check that accessor exists and returns proper plotter
        assert hasattr(data, "plotX")
        plotter = data.plotX
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
        with pytest.raises(ValueError, match="grid_type must be either"):
            specify_grid(grid_type="invalid")

    def test_specify_grid_with_paths(self):
        """Test specify_grid with file paths."""
        tgrid_path = Path("/tmp/test_tgrid.nc")
        ckdtree_path = Path("/tmp/test_ckdtree")

        specify_grid(
            grid_type="unstructured", fpath_tgrid=tgrid_path, fpath_ckdtree=ckdtree_path
        )

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

        plotter = register_plotter(data)
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

    def test_plotter_base_initialization(self):
        """Test PlotterBase initialization."""

        # Create a concrete subclass for testing
        class TestPlotter(PlotterBase):
            def plot(self, ax, cmap="viridis", clim=None, norm=None):
                return ax, MagicMock()

        plotter = TestPlotter(self.gridded_data)
        assert plotter.da is self.gridded_data
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
        plotter = self.sst_data.plotX
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

        plotter = self.sst_data.plotX

        # Mock the plot method to avoid actual plotting based on plotter type
        with patch.object(type(plotter), "plot", return_value=(mock_ax, MagicMock())):
            fig, ax, im = plotter.single_plot(config)

            assert fig == mock_fig
            assert ax == mock_ax
            mock_ax.set_title.assert_called_once_with("Test Plot", size=12)
