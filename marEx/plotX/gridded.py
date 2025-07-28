"""
Gridded data visualisation module for regular rectangular grids.

Provides specialised plotting capabilities for structured oceanographic data
with lat/lon coordinates on regular grids (3D arrays: time, lat, lon).
"""

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
    """Plotter for structured oceanographic data on regular rectangular grids."""

    def __init__(
        self,
        xarray_obj: xr.DataArray,
        dimensions: Optional[Dict[str, str]] = None,
        coordinates: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialise GriddedPlotter."""
        super().__init__(xarray_obj, dimensions, coordinates)

    def wrap_lon(self, data: xr.DataArray) -> xr.DataArray:
        """Handle periodic boundary in longitude by adding a column of data."""
        lon = data[self.dimensions["x"]]

        # Check if we're dealing with global data that needs wrapping
        lon_spacing = np.diff(lon)[0]
        if abs(360 - (lon.max() - lon.min())) < 2 * lon_spacing:
            # Add a column at lon=360 that equals the data at lon=0
            new_lon = np.append(lon, lon[0] + 360)
            wrapped_data = xr.concat([data, data.isel({self.dimensions["x"]: 0})], dim=self.dimensions["x"])
            wrapped_data[self.dimensions["x"]] = new_lon
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

            # Ensure data has only required dimensions for imshow
            if self.dimensions["time"] in data.dims and len(data[self.dimensions["time"]]) == 1:
                data = data.squeeze(dim=self.dimensions["time"])  # Remove time dimension if singular

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

            lons = data[self.dimensions["x"]].values
            lats = data[self.dimensions["y"]].values
            values = data.values

            logger.debug(f"Rendering plot with {len(lons)} x {len(lats)} grid points")
            # imshow has some dimension issues with cartopy...
            im = ax.pcolormesh(lons, lats, values, **plot_kwargs)

        return ax, im
