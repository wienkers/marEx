import xarray as xr
import numpy as np
import dask

import matplotlib.pylab as plt
import matplotlib.cm as cm
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def clim_robust(data, issym, percentiles=[2, 98]):
    clim = np.nanpercentile(data, percentiles)
    
    if issym:
        clim = np.abs(clim).max()
        clim = np.array([-clim, clim])
    elif percentiles[0] == 0:
        clim = np.array([0, clim[1]])

    return clim


@xr.register_dataarray_accessor('plotter')
class PlotterAccessor:
    def __init__(self, xarray_obj):
        self.da = xarray_obj
        
        # Cache common features
        self._land = cfeature.LAND.with_scale('50m')
        self._coastlines = cfeature.COASTLINE.with_scale('50m')

        
    def pplot(self, title = None, var_units='', issym = False, cmap = None, cperc = [4, 96], clim = None, show_colorbar=True, ax = None, grid_labels = True, dimensions={'time':'time','ydim':'lat','xdim':'lon'}):
        '''Make pretty plots.'''
        
        plt.rc('text', usetex=False)  # Use built-in math text rendering
        plt.rc('font', family='serif')
        
        if cmap is None:
            cmap = 'RdBu_r' if issym else 'viridis'
        
        if ax is None:
            fig = plt.figure(figsize=(7, 5))
            ax = plt.axes(projection=ccrs.Robinson())
        else:
            fig = ax.get_figure()
        
        
        cl = clim_robust(self.da, issym, cperc) if clim is None else clim
        
        data = self.wrap_lon(self.da, dimensions)
        
        regular_grid = True
        
        if not regular_grid:  # Use pcolormesh 
            im = ax.pcolormesh(data[dimensions['xdim']],data[dimensions['ydim']],data.values,
                        vmin=cl[0],vmax=cl[1],cmap=cmap,
                        transform=ccrs.PlateCarree())
        else:  # imshow is much faster...
            extent = [data[dimensions['xdim']].min(), data[dimensions['xdim']].max(),
                     data[dimensions['ydim']].min(), data[dimensions['ydim']].max()]
            im = ax.imshow(data.values, origin='lower', extent=extent,
                          transform=ccrs.PlateCarree(),
                          vmin=cl[0], vmax=cl[1], cmap=cmap,
                          interpolation='nearest')
        
        if title is not None: ax.set_title(title,size=12)
        
        if show_colorbar:
            cb = plt.colorbar(im, shrink=0.6, ax=ax, extend='both')
            cb.ax.set_ylabel(f'{var_units}', fontsize=10)
            cb.ax.tick_params(labelsize=10)

        ax.add_feature(self._land, facecolor='darkgrey')
        ax.add_feature(self._coastlines, linewidth=0.5)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=grid_labels,
                        linewidth=1, color='gray', alpha=0.5, linestyle='--')
        
        return fig, ax

    def pplot_col(self, col='time', col_wrap=3, var_units='', issym = False, cmap = None, cperc = [4, 96], clim = None, show_colorbar=True, ax = None, dimensions={'time':'time','ydim':'lat','xdim':'lon'}):
        '''Make pretty wrapped subplots.'''
        
        plt.rc('text', usetex=False)
        
        npanels = self.da[col].size
        nrows = int(np.ceil(npanels/col_wrap))
        ncols = min(npanels, col_wrap)
        
        fig = plt.figure(figsize=(6*ncols, 3*nrows))
        axes = fig.subplots(nrows, ncols, subplot_kw={'projection': ccrs.Robinson()}).flatten()
        
        for i, ax in enumerate(axes):
            if i < npanels:
                if col == dimensions['time']:
                    title = f"{self.da[col].isel({col: i}).time.dt.strftime('%Y-%m-%d').values}"
                else:
                    title = f"{col}={self.da[col].isel({col: i}).values}"
                self.da.isel({col: i}).plotter.pplot(title = title, var_units=var_units, issym = issym, cmap = cmap, cperc = cperc, clim = clim, show_colorbar = show_colorbar, grid_labels = False, ax = ax, dimensions=dimensions)
            else:
                fig.delaxes(ax)
        
        return fig, axes
    
    
    def wrap_lon(self, data, dimensions):
        '''Handle periodic boundary in longitude by adding a column of data.'''
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