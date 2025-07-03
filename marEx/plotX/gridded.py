from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr

try:
    import cartopy.crs as ccrs
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import BoundaryNorm, Normalize

    HAS_PLOTTING_DEPS = True
except ImportError:
    # These will be checked in the base class
    ccrs = None
    Axes = None
    QuadMesh = None
    BoundaryNorm = None
    Normalize = None
    HAS_PLOTTING_DEPS = False

from ..logging_config import get_logger, log_timing
from .base import PlotterBase

# Get module logger
logger = get_logger(__name__)


class GriddedPlotter(PlotterBase):
    def __init__(
        self,
        xarray_obj: xr.DataArray,
        dimensions: Optional[Dict[str, str]] = None,
        coordinates: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(xarray_obj, dimensions, coordinates)

    def wrap_lon(self, data: xr.DataArray, dimensions: Optional[Dict[str, str]] = None) -> xr.DataArray:
        """Handle periodic boundary in longitude by adding a column of data."""
        if dimensions is None:
            dimensions = self.dimensions

        x_dim = dimensions["x"]
        x_coord = self.coordinates["x"]
        lon = data[x_coord]

        # Check if we're dealing with global data that needs wrapping
        lon_spacing = np.diff(lon)[0]
        if abs(360 - (lon.max() - lon.min())) < 2 * lon_spacing:
            # Add a column at lon=360 that equals the data at lon=0
            new_lon = np.append(lon, lon[0] + 360)
            wrapped_data = xr.concat([data, data.isel({x_dim: 0})], dim=x_dim)
            wrapped_data[x_coord] = new_lon
            return wrapped_data
        return data

    def plot(
        self,
        ax: Axes,
        cmap: Union[str, Any] = "viridis",
        clim: Optional[Tuple[float, float]] = None,
        norm: Optional[Union[BoundaryNorm, Normalize]] = None,
    ) -> Tuple[Axes, QuadMesh]:
        """Implement plotting for gridded (i.e. regular grid) data."""
        logger.debug(f"Plotting gridded data with shape {self.da.shape}")

        with log_timing(logger, "Gridded plot rendering", show_progress=True):
            data = self.wrap_lon(self.da)

            # Ensure data has only required dimensions for plotting
            time_dim = self.dimensions.get("time", "time")
            if time_dim in data.dims and len(data[time_dim]) == 1:
                data = data.squeeze(dim=time_dim)  # Remove time dimension if it's singular

            plot_kwargs = {
                "transform": ccrs.PlateCarree(),
                "cmap": cmap,
                "shading": "auto",
            }

            if norm is not None:
                plot_kwargs["norm"] = norm
            elif clim is not None:
                plot_kwargs["vmin"] = clim[0]
                plot_kwargs["vmax"] = clim[1]

            x_coord = self.coordinates["x"]
            y_coord = self.coordinates["y"]
            lons = data[x_coord].values
            lats = data[y_coord].values
            values = data.values

            logger.debug(f"Rendering plot with {len(lons)} x {len(lats)} grid points")
            # imshow has some dimension issues with cartopy...
            im = ax.pcolormesh(lons, lats, values, **plot_kwargs)

        return ax, im
