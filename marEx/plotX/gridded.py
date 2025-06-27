import numpy as np
from numpy.typing import NDArray
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
import cartopy.crs as ccrs
from typing import Dict, Tuple, Optional, Union, Any
from matplotlib.colors import Normalize, BoundaryNorm

from .base import PlotterBase

class GriddedPlotter(PlotterBase):
    def __init__(self, xarray_obj: xr.DataArray) -> None:
        super().__init__(xarray_obj)
    
    def wrap_lon(
        self, 
        data: xr.DataArray, 
        dimensions: Dict[str, str]
    ) -> xr.DataArray:
        """Handle periodic boundary in longitude by adding a column of data."""
        lon = data[dimensions['xdim']]
        
        # Check if we're dealing with global data that needs wrapping
        lon_spacing = np.diff(lon)[0]
        if abs(360 - (lon.max() - lon.min())) < 2 * lon_spacing:
            # Add a column at lon=360 that equals the data at lon=0
            new_lon = np.append(lon, lon[0] + 360)
            wrapped_data = xr.concat([data, data.isel({dimensions['xdim']: 0})], 
                                   dim=dimensions['xdim'])
            wrapped_data[dimensions['xdim']] = new_lon
            return wrapped_data
        return data

    def plot(
        self, 
        ax: Axes, 
        cmap: Union[str, Any] = 'viridis', 
        clim: Optional[Tuple[float, float]] = None, 
        norm: Optional[Union[BoundaryNorm, Normalize]] = None
    ) -> Tuple[Axes, QuadMesh]:
        """Implement plotting for gridded (i.e. regular grid) data."""
        dimensions = {'ydim': 'lat', 'xdim': 'lon'}
        data = self.wrap_lon(self.da, dimensions)
        
         # Ensure data has only required dimensions for imshow
        if 'time' in data.dims and len(data.time) == 1:
            data = data.squeeze(dim='time')  # Remove time dimension if it's singular
        
        plot_kwargs = {
            'transform': ccrs.PlateCarree(),
            'cmap': cmap,
            'shading': 'auto'
        }
        
        if norm is not None:
            plot_kwargs['norm'] = norm
        elif clim is not None:
            plot_kwargs['vmin'] = clim[0]
            plot_kwargs['vmax'] = clim[1]
        
        lons = data[dimensions['xdim']].values
        lats = data[dimensions['ydim']].values
        values = data.values
        
        # imshow has some dimension issues with cartopy...
        im = ax.pcolormesh(lons, lats, values, **plot_kwargs)
        
        return ax, im