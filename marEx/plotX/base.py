"""Base classes and utilities for the plotX visualisation system.

This module provides the core infrastructure for plotting marine extreme event data,
supporting both structured and unstructured grids with comprehensive configuration
and animation capabilities.
"""

import gc
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..exceptions import DependencyError, VisualisationError
from ..logging_config import configure_logging, get_logger

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

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


def _check_plotting_dependencies() -> None:
    """Check if plotting dependencies are available and raise informative error if not."""
    from .._dependencies import require_dependencies

    require_dependencies(["matplotlib", "cartopy"], "Plotting functionality")


@dataclass
class PlotConfig:
    """Configuration class for plot parameters

    Attributes:
        title: Plot title
        var_units: Variable units for colorbar label
        issym: Whether data is symmetric (centers colormap at 0)
        cmap: Colormap name or ListedColormap object
        cperc: Percentile range for automatic color limits [min, max]
        clim: Manual color limits (vmin, vmax)
        show_colorbar: Whether to display colorbar
        grid_lines: Whether to display grid lines
        grid_labels: Whether to display grid labels
        dimensions: Mapping of conceptual to actual dimension names
        coordinates: Mapping of conceptual to actual coordinate names
        norm: Custom normalization (BoundaryNorm or Normalize)
        plot_IDs: Whether to plot object IDs with random colors
        extend: Colorbar extension ('neither', 'both', 'min', 'max')
        verbose: Enable verbose logging
        quiet: Enable quiet logging
        projection: Cartopy projection for map plots
        framerate: Frames per second for animations (default 10)
    """

    title: Optional[str] = None
    var_units: str = ""
    issym: bool = False
    cmap: Optional[Union[str, ListedColormap]] = None
    cperc: List[int] = None
    clim: Optional[Tuple[float, float]] = None
    show_colorbar: bool = True
    grid_lines: bool = True
    grid_labels: bool = False
    dimensions: Dict[str, str] = None
    coordinates: Dict[str, str] = None
    norm: Optional[Union[BoundaryNorm, Normalize]] = None
    plot_IDs: bool = False
    extend: str = "both"
    verbose: Optional[bool] = None
    quiet: Optional[bool] = None
    projection: Optional[Any] = None
    framerate: int = 10

    def __post_init__(self) -> None:
        """Initialise default values and configure logging."""
        if self.cperc is None:
            self.cperc = [4, 96]
        if self.dimensions is None:
            self.dimensions = {"time": "time", "y": "lat", "x": "lon"}
        if self.coordinates is None:
            self.coordinates = {"time": "time", "y": "lat", "x": "lon"}
        if self.plot_IDs:
            self.show_colorbar = False
        if self.projection is None:
            self.projection = ccrs.Robinson()

        # Configure logging if verbose/quiet parameters are provided
        if self.verbose is not None or self.quiet is not None:
            configure_logging(verbose=self.verbose, quiet=self.quiet)


def _validate_dimensions_exist(da: xr.DataArray, dimensions: Dict[str, str]) -> None:
    """Validate that required dimensions exist in the dataset. Time dimension is optional."""
    missing_dims = []
    for concept_dim, actual_dim in dimensions.items():
        # Time dimension is optional for plotting - only spatial dimensions are required
        if concept_dim == "time" and actual_dim not in da.dims:
            continue
        if actual_dim not in da.dims:
            missing_dims.append(f"'{actual_dim}' (for {concept_dim})")

    if missing_dims:
        available_dims = list(da.dims)
        raise VisualisationError(
            f"Missing required dimensions: {', '.join(missing_dims)}",
            details=f"Dataset has dimensions: {available_dims}",
            suggestions=[
                "Check dimension names in your data",
                "Update the 'dimensions' parameter to match your data structure",
                f"Available dimensions: {available_dims}",
            ],
            context={
                "missing_dimensions": missing_dims,
                "available_dimensions": available_dims,
                "provided_dimensions": dimensions,
            },
        )


def _validate_coordinates_exist(da: xr.DataArray, coordinates: Dict[str, str]) -> None:
    """Validate that required coordinates exist in the dataset. Time coordinate is optional."""
    missing_coords = []
    for concept_coord, actual_coord in coordinates.items():
        # Time coordinate is optional for plotting - only spatial coordinates are required
        if concept_coord == "time" and actual_coord not in da.coords:
            continue
        if actual_coord not in da.coords:
            missing_coords.append(f"'{actual_coord}' (for {concept_coord})")

    if missing_coords:
        available_coords = list(da.coords)
        raise VisualisationError(
            f"Missing required coordinates: {', '.join(missing_coords)}",
            details=f"Dataset has coordinates: {available_coords}",
            suggestions=[
                "Check coordinate names in your data",
                "Update the 'coordinates' parameter to match your data structure",
                f"Available coordinates: {available_coords}",
            ],
            context={
                "missing_coordinates": missing_coords,
                "available_coordinates": available_coords,
                "provided_coordinates": coordinates,
            },
        )


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
        # Check if PIL is available for image processing
        from .._dependencies import require_dependencies

        require_dependencies(["pillow"], "Animation functionality")

        # Check if ffmpeg is installed
        if shutil.which("ffmpeg") is None:
            warnings.warn(
                "ffmpeg executable not found in system PATH. Cannot create animation.\n"
                "Please install ffmpeg using one of the following methods:\n"
                "  - Linux: sudo apt install ffmpeg (Ubuntu/Debian) or sudo yum install ffmpeg (CentOS/RHEL)\n"
                "  - Conda: conda install -c conda-forge ffmpeg\n"
                "Alternatively, use matplotlib for animation in Jupyter notebooks.",
                stacklevel=2,
            )
            return None

        plot_dir = Path(plot_dir)
        plot_dir.mkdir(exist_ok=True)

        # Use dask's scratch directory for temporary frames
        dask_temp = dask.config.get("temporary-directory", default=None)
        if dask_temp is None:
            dask_temp = tempfile.gettempdir()
        temp_dir = Path(tempfile.mkdtemp(prefix="marex_animate_", dir=dask_temp))

        if not file_name:
            file_name = f"movie_{self.da.name}.mp4"

        output_file = plot_dir / f"{file_name}.mp4"

        # Set up plotting parameters
        cmap, norm, clim, var_units, extend = self._setup_common_params(config)

        plot_params = {
            "cmap": cmap,
            "norm": norm,
            "clim": clim,
            "var_units": var_units,
            "extend": extend,
            "show_colorbar": config.show_colorbar,
            "grid_labels": config.grid_labels,
        }

        # Set up grid information if needed
        grid_info = None
        if hasattr(self, "fpath_tgrid") or hasattr(self, "fpath_ckdtree"):
            grid_info = {
                "type": "unstructured",
                "tgrid_path": getattr(self, "fpath_tgrid", None),
                "ckdtree_path": getattr(self, "fpath_ckdtree", None),
                "res": 0.3,
            }

        # Generate frames using dask for parallel processing
        delayed_tasks = []
        time_dim = config.dimensions["time"] if config.dimensions else "time"
        time_coord = config.coordinates.get("time", time_dim) if config.coordinates else time_dim

        # Use provided centroids or None if not provided
        centroid_data = centroids

        for time_ind in range(len(self.da[time_dim])):
            data_slice = self.da.isel({time_dim: time_ind})

            # Create fresh copy of plot_params for this frame to avoid shared references
            frame_params = plot_params.copy()
            frame_params["time_str"] = str(self.da[time_coord].isel({time_dim: time_ind}).dt.strftime("%Y-%m-%d").values)

            # Extract centroids for this time step if available
            if centroid_data is not None:
                try:
                    centroids_time = centroid_data.isel({time_dim: time_ind})
                    frame_params["centroids"] = centroids_time
                except Exception:
                    frame_params["centroids"] = None
            else:
                frame_params["centroids"] = None

            # Extract object IDs for this time step if available
            if object_ids is not None:
                try:
                    object_ids_time = object_ids.isel({time_dim: time_ind})
                    frame_params["object_ids"] = object_ids_time
                except Exception:
                    frame_params["object_ids"] = None
            else:
                frame_params["object_ids"] = None

            delayed_tasks.append(make_frame(data_slice, time_ind, temp_dir, frame_params, grid_info))

        # Process frames in batches to manage memory efficiently
        batch_size = 200
        filenames = []
        for i in range(0, len(delayed_tasks), batch_size):
            batch = delayed_tasks[i : i + batch_size]
            batch_results = dask.compute(*batch)
            filenames.extend(batch_results)
            # Force garbage collection between batches to release memory
            gc.collect()

        filenames = sorted(filenames, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        # Create movie using ffmpeg
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-threads",
                "0",
                "-framerate",
                str(config.framerate),
                "-i",
                str(temp_dir / "time_%04d.jpg"),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_file),
            ],
            check=True,
        )

        # Clean up temporary frames directory
        shutil.rmtree(temp_dir)

        return str(output_file)

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


@dask.delayed
def make_frame(
    data_slice: xr.DataArray,
    time_ind: int,
    temp_dir: Path,
    plot_params: Dict[str, Any],
    grid_info: Optional[Dict[str, Any]] = None,
) -> str:  # pragma: no cover
    """Create a single frame for movies - minimise memory usage with dask

    Args:
        data_slice: The data for this specific frame
        time_ind: Frame index
        temp_dir: Directory for temporary files
        plot_params: Dict containing plotting parameters
        grid_info: Dict containing grid paths and settings for unstructured data
    """
    # Set up plotting parameters
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes(projection=ccrs.Robinson())

    data_slice_np = data_slice.values

    # Set up plot kwargs
    plot_kwargs = {
        "transform": ccrs.PlateCarree(),
        "cmap": plot_params["cmap"],
        "shading": "auto",
    }

    if plot_params.get("norm") is not None:
        plot_kwargs["norm"] = plot_params["norm"]
    elif plot_params.get("clim") is not None:
        plot_kwargs["vmin"] = plot_params["clim"][0]
        plot_kwargs["vmax"] = plot_params["clim"][1]

    # Handle different grid types
    if grid_info and grid_info.get("type") == "unstructured":
        try:
            from .unstructured import _load_ckdtree, _load_triangulation
        except ImportError as e:
            raise DependencyError(
                "Unstructured plotting dependencies missing",
                details=str(e),
                suggestions=[
                    "Install plotting dependencies: pip install marEx[plot]",
                    "Check that scipy and matplotlib are available",
                    "Verify unstructured grid support is properly installed",
                ],
                context={"missing_dependency": str(e), "plot_type": "unstructured"},
            )

        if grid_info.get("ckdtree_path"):
            # Use cached ckdtree data
            ckdt_data = _load_ckdtree(grid_info["ckdtree_path"], grid_info.get("res", 0.3))
            grid_data = data_slice_np[ckdt_data["indices"]].reshape(ckdt_data["lat"].size, ckdt_data["lon"].size)
            grid_data = np.ma.masked_invalid(grid_data)
            im = ax.pcolormesh(ckdt_data["lon"], ckdt_data["lat"], grid_data, **plot_kwargs)
        elif grid_info.get("tgrid_path"):
            # Use triangulation
            triang = _load_triangulation(grid_info["tgrid_path"])
            data_masked = np.ma.masked_invalid(data_slice_np)
            im = ax.tripcolor(triang, data_masked, **plot_kwargs)
    else:
        # Regular grid plotting
        lat = data_slice.lat.values
        lon = data_slice.lon.values
        im = ax.pcolormesh(lon, lat, data_slice_np, **plot_kwargs)

    time_str = plot_params.get("time_str", f"Frame {time_ind}")
    ax.set_title(time_str, size=12)

    # Plot object ID contours if available
    object_ids_data = plot_params.get("object_ids")
    if object_ids_data is not None:
        try:
            object_ids_np = object_ids_data.values
            # Create binary mask where object IDs > 0
            object_mask = object_ids_np > 0

            if np.any(object_mask):
                # Handle different grid types for contouring
                if grid_info and grid_info.get("type") == "unstructured":
                    # For unstructured grids, we need to handle contouring differently
                    # This is more complex and may require interpolation to regular grid
                    pass
                else:
                    # Regular grid plotting - use lat/lon coordinates
                    lat = data_slice.lat.values
                    lon = data_slice.lon.values

                    # Draw contours around object boundaries (treating all IDs > 0 the same)
                    ax.contour(
                        lon,
                        lat,
                        object_mask.astype(float),
                        levels=[0.5],
                        colors=["white"],
                        linewidths=1.5,
                        transform=ccrs.PlateCarree(),
                        zorder=6,
                    )
        except Exception:
            # Silently skip object ID contouring if any error occurs
            pass

    # Plot centroids if available
    centroids = plot_params.get("centroids")
    if centroids is not None:
        try:
            # Get unique object IDs present in this frame
            unique_ids = np.unique(data_slice_np)
            unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)

            if len(unique_ids) > 0:
                # Extract centroid coordinates for present objects
                # centroids shape: (component, ID) where component 0=lat, 1=lon
                centroids_np = centroids.values

                # Find which IDs have valid centroids
                valid_centroids = []
                for obj_id in unique_ids:
                    try:
                        # Find ID index in centroids
                        id_idx = np.where(centroids.ID.values == obj_id)[0]
                        if len(id_idx) > 0:
                            idx = id_idx[0]
                            lat_centroid = centroids_np[0, idx]  # component 0 = latitude
                            lon_centroid = centroids_np[1, idx]  # component 1 = longitude

                            # Check if centroid is valid (not NaN)
                            if not (np.isnan(lat_centroid) or np.isnan(lon_centroid)):
                                valid_centroids.append((lon_centroid, lat_centroid))
                    except (IndexError, KeyError):
                        continue

                # Plot centroids as scatter points
                if valid_centroids:
                    centroid_lons, centroid_lats = zip(*valid_centroids)
                    ax.scatter(
                        centroid_lons,
                        centroid_lats,
                        c="black",
                        s=20,
                        marker="o",
                        edgecolors="white",
                        linewidth=1.5,
                        transform=ccrs.PlateCarree(),
                        zorder=5,  # Plot above data but below grid lines
                        alpha=0.8,
                    )
        except Exception:
            # Silently skip centroid plotting if any error occurs
            pass

    if plot_params.get("show_colorbar"):
        cb = plt.colorbar(im, shrink=0.6, ax=ax, extend=plot_params.get("extend", "both"))
        if plot_params.get("var_units"):
            cb.ax.set_ylabel(plot_params["var_units"], fontsize=10)
        cb.ax.tick_params(labelsize=10)

    land = cfeature.LAND.with_scale("50m")
    coastlines = cfeature.COASTLINE.with_scale("50m")
    ax.add_feature(land, facecolor="darkgrey", zorder=2)
    ax.add_feature(coastlines, linewidth=0.5, zorder=3)
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=plot_params.get("grid_labels", False),
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
        zorder=4,
    )

    # Save and process frame
    filename = f"time_{time_ind:04d}.jpg"
    temp_file = temp_dir / f"temp_{filename}"
    fig.savefig(str(temp_file), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Ensure dimensions are even for video encoding
    image = Image.open(str(temp_file))
    width, height = image.size
    new_width = width - (width % 2)
    new_height = height - (height % 2)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    image.save(str(temp_dir / filename))
    temp_file.unlink()

    return filename
