import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from .base import PlotterBase

class StructuredPlotter(PlotterBase):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
    
    def wrap_lon(self, data, dimensions):
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

    def plot(self, ax, cmap='viridis', clim=None, norm=None):
        """Implement plotting for structured (regular grid) data."""
        dimensions = {'ydim': 'lat', 'xdim': 'lon'}
        data = self.wrap_lon(self.da, dimensions)
        
        plot_kwargs = {
            'transform': ccrs.PlateCarree(),
            'cmap': cmap,
            'interpolation': 'nearest',
            'extent': [
                data[dimensions['xdim']].min(), 
                data[dimensions['xdim']].max(),
                data[dimensions['ydim']].min(), 
                data[dimensions['ydim']].max()
            ],
            'origin': 'lower'
        }
        
        if norm is not None:
            plot_kwargs['norm'] = norm
        elif clim is not None:
            plot_kwargs['vmin'] = clim[0]
            plot_kwargs['vmax'] = clim[1]
            
        im = ax.imshow(data.values, **plot_kwargs)
        
        return ax, im