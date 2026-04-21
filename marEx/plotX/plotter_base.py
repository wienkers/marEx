"""Base plotter class for the plotX visualisation system.

Provides ``PlotterBase``, the common infrastructure subclassed by the gridded
and unstructured plotters: parameter setup, axes/map features, colorbars,
single/multi panel plots, and animation (delegated to ``animation._animate``).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..logging_config import get_logger
from .animation import _animate
from .config import PlotConfig
from .dependencies import _check_plotting_dependencies
from .validation import _validate_coordinates_exist, _validate_dimensions_exist

# Get module logger
logger = get_logger(__name__)

# Handle optional dependencies for plotting
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    ccrs = None
    cfeature = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    Axes = None
    ScalarMappable = None
    Colorbar = None
    BoundaryNorm = None
    ListedColormap = None
    Normalize = None
    Figure = None


class PlotterBase:
    """Base class for all plotters providing common functionality.

    This class provides the core infrastructure for plotting marine extreme event data,
    including parameter setup, map features, colorbars, and animation capabilities.
    """

    def __init__(
        self,
        xarray_obj: xr.DataArray,
        dimensions: Optional[Dict[str, str]] = None,
        coordinates: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialise the plotter with data and coordinate mappings.

        Args:
            xarray_obj: The data to plot
            dimensions: Mapping of conceptual to actual dimension names
            coordinates: Mapping of conceptual to actual coordinate names
        """
        _check_plotting_dependencies()
        self.da = xarray_obj

        # Set default dimensions and coordinates if not provided
        if dimensions is None:
            dimensions = {"time": "time", "y": "lat", "x": "lon"}
        if coordinates is None:
            coordinates = {"time": "time", "y": "lat", "x": "lon"}

        self.dimensions = dimensions
        self.coordinates = coordinates

        # Validate dimensions and coordinates exist in the data
        _validate_dimensions_exist(self.da, self.dimensions)
        _validate_coordinates_exist(self.da, self.coordinates)

        # Cache common features
        self._land = cfeature.LAND.with_scale("50m")
        self._coastlines = cfeature.COASTLINE.with_scale("50m")

    def _setup_common_params(self, config: PlotConfig) -> Tuple[
        Union[str, ListedColormap],
        Optional[Union[BoundaryNorm, Normalize]],
        Optional[Tuple[float, float]],
        str,
        str,
    ]:
        """Centralise common parameter setup"""
        self.setup_plot_params()

        if config.plot_IDs:
            cmap, norm, var_units = self.setup_id_plot_params(config.cmap)
            clim = None
            extend = "neither"
            self.da = self.da.where(self.da > 0)  # Fill value to NaN (get rid of 0s)
        else:
            if config.cmap is None:
                cmap = "RdBu_r" if config.issym else "viridis"
            else:
                cmap = config.cmap
            norm = config.norm
            if config.clim is None and norm is None:
                # Sample data to avoid loading entire time series into memory
                time_dim = self.dimensions.get("time", "time")
                if time_dim in self.da.dims:
                    sampled_da = self.da.isel({time_dim: slice(None, None, 10)})
                else:
                    sampled_da = self.da
                clim = self.clim_robust(sampled_da.values, config.issym, config.cperc)
            else:
                clim = config.clim
            var_units = config.var_units
            extend = config.extend

        return cmap, norm, clim, var_units, extend

    def _setup_axes(self, ax: Optional[Axes] = None, projection: Optional[Any] = None) -> Tuple[Figure, Axes]:
        """Create or use existing axes with projection"""
        if ax is None:
            # Use provided projection or default to Robinson
            proj = projection if projection is not None else ccrs.Robinson()
            fig = plt.figure(figsize=(7, 5))
            ax = plt.axes(projection=proj)
        else:
            fig = ax.get_figure()
        return fig, ax

    def _add_map_features(self, ax: Axes, grid_lines: bool = True, grid_labels: bool = True) -> None:
        """Add common map features to the plot"""
        ax.add_feature(self._land, facecolor="darkgrey", zorder=2)
        ax.add_feature(self._coastlines, linewidth=0.5, zorder=3)
        if grid_lines:
            ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=grid_labels,
                linewidth=1,
                color="gray",
                alpha=0.5,
                linestyle="--",
                zorder=4,
            )

    def _setup_colorbar(
        self,
        fig: Figure,
        im: Union[ScalarMappable, Any],
        show_colorbar: bool,
        var_units: str,
        extend: str = "both",
        position: Optional[List[float]] = None,
    ) -> Optional[Colorbar]:
        """Set up colorbar with common parameters"""
        if not show_colorbar:
            return None

        if position is not None:
            # For column plots
            cbar_ax = fig.add_axes(position)
            cb = fig.colorbar(im, cax=cbar_ax, extend=extend)
        else:
            # For single plots
            cb = plt.colorbar(im, shrink=0.6, ax=plt.gca(), extend=extend)

        if var_units:
            cb.ax.set_ylabel(var_units, fontsize=10)
        cb.ax.tick_params(labelsize=10)
        return cb

    def _get_title(self, time_index: int, col_name: str, dimensions: Optional[Dict[str, str]] = None) -> str:
        """Generate appropriate title based on dimension"""
        if dimensions is None:
            dimensions = self.dimensions

        if col_name == dimensions["time"]:
            time_coord = self.coordinates.get("time", "time")
            return f"{self.da[time_coord].isel({col_name: time_index}).dt.strftime('%Y-%m-%d').values}"
        return f"{col_name}={self.da[col_name].isel({col_name: time_index}).values}"

    def single_plot(self, config: PlotConfig, ax: Optional[Axes] = None) -> Tuple[Figure, Axes, Any]:
        """Make a single plot with given configuration"""
        cmap, norm, clim, var_units, extend = self._setup_common_params(config)

        fig, ax = self._setup_axes(ax, config.projection)

        # Call implementation-specific plot function
        ax, im = self.plot(ax=ax, cmap=cmap, clim=clim, norm=norm)

        if config.title:
            ax.set_title(config.title, size=12)

        self._setup_colorbar(fig, im, config.show_colorbar, var_units, extend)
        self._add_map_features(ax, config.grid_lines, config.grid_labels)

        return fig, ax, im

    def multi_plot(
        self, config: PlotConfig, col: str = "time", col_wrap: int = 3
    ) -> Tuple[Figure, NDArray[Any]]:  # pragma: no cover
        """Make wrapped subplots with given configuration"""
        npanels = self.da[col].size
        nrows = int(np.ceil(npanels / col_wrap))
        ncols = min(npanels, col_wrap)

        cmap, norm, clim, var_units, extend = self._setup_common_params(config)

        fig = plt.figure(figsize=(6 * ncols, 3 * nrows))
        axes = fig.subplots(nrows, ncols, subplot_kw={"projection": config.projection}).flatten()

        # Create a single plotter instance to be reused
        base_plotter = type(self)(self.da)
        for attr in ["fpath_tgrid", "fpath_ckdtree"]:
            if hasattr(self, attr):
                setattr(base_plotter, attr, getattr(self, attr))

        for i, ax in enumerate(axes):
            if i < npanels:
                title = self._get_title(i, col, config.dimensions)

                # Create new config for individual panel
                panel_config = PlotConfig(
                    title=title,
                    cmap=cmap,
                    clim=clim,
                    show_colorbar=False,
                    grid_labels=False,
                    norm=norm,
                    plot_IDs=False,
                    extend=extend,
                    dimensions=config.dimensions,
                    coordinates=config.coordinates,
                    projection=config.projection,
                )

                # Update data in base plotter instead of creating new instance
                base_plotter.da = self.da.isel({col: i})

                # Plot individual panel using the same plotter instance
                base_plotter.single_plot(panel_config, ax=ax)
            else:
                fig.delaxes(ax)

        # Add single colorbar for all panels
        if config.show_colorbar:
            fig.subplots_adjust(right=0.9)
            if norm is None and clim is not None:
                # Create a proper norm from clim
                from matplotlib.colors import Normalize

                norm = Normalize(vmin=clim[0], vmax=clim[1])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            self._setup_colorbar(fig, sm, True, var_units, extend, position=[0.92, 0.15, 0.02, 0.7])

        return fig, axes

    def animate(
        self,
        config: PlotConfig,
        plot_dir: Union[str, Path] = "./",
        file_name: Optional[str] = None,
        centroids: Optional[xr.DataArray] = None,
        object_ids: Optional[xr.DataArray] = None,
    ) -> Optional[str]:  # pragma: no cover
        """Create an animation from time series data

        Args:
            config: Plot configuration (including framerate for animation, default 10 fps)
            plot_dir: Directory to save animation files
            file_name: Name for the output animation file
            centroids: Optional DataArray containing centroid data with dimensions (component, time, ID)
            object_ids: Optional DataArray containing object ID field with integers > 0 for drawing contour outlines
        """
        return _animate(self, config, plot_dir, file_name, centroids, object_ids)

    def clim_robust(self, data: NDArray[Any], issym: bool, percentiles: Optional[List[int]] = None) -> NDArray[np.float64]:
        """Compute robust colour limits from data percentiles."""
        if percentiles is None:
            percentiles = [2, 98]
        clim = np.nanpercentile(data, percentiles)

        if issym:
            clim = np.abs(clim).max()
            clim = np.array([-clim, clim])
        elif percentiles[0] == 0:
            clim = np.array([0, clim[1]])

        return clim

    def setup_plot_params(self) -> None:
        """Set up common plotting parameters"""
        plt.rc("text", usetex=False)
        plt.rc("font", family="serif")

    def setup_id_plot_params(self, cmap: Optional[Union[str, ListedColormap]] = None) -> Tuple[ListedColormap, BoundaryNorm, str]:
        """Set up parameters for plotting IDs"""
        # Use min=1 and max from data without computing all unique values
        max_id = int(self.da.max().values)
        bounds = np.arange(1, max_id + 2) - 0.5
        n_bins = len(bounds) - 1

        if cmap is None:
            np.random.seed(42)
            cmap = ListedColormap(np.random.random(size=(n_bins, 3)))

        norm = BoundaryNorm(bounds, cmap.N)
        return cmap, norm, "ID"

    def plot(
        self,
        ax: Axes,
        cmap: Union[str, ListedColormap] = "viridis",
        clim: Optional[Tuple[float, float]] = None,
        norm: Optional[Union[BoundaryNorm, Normalize]] = None,
    ) -> Tuple[Axes, Any]:
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement plot method")
