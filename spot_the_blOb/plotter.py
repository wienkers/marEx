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
        self._obj = xarray_obj

        
    def pplot(self, title = None, var_units='', issym = False, cmap = None, cperc = [4, 96], clim = None, ax = None, dimensions={'time':'time','ydim':'lat','xdim':'lon'}):
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
        
        
        cl = clim_robust(self, issym, cperc) if clim is None else clim
        im = ax.pcolormesh(self[dimensions['xdim']],self[dimensions['ydim']],self.values,
                    vmin=cl[0],vmax=cl[1],cmap=cmap,
                    transform=ccrs.PlateCarree())
        if title is not None: ax.set_title(title,size=14)
        cb = plt.colorbar(im,shrink=0.6,ax=ax,extend='both')
        cb.ax.set_ylabel(f'{var_units}',fontsize=10)
        cb.ax.tick_params(labelsize=10)
        ax.add_feature(cfeature.LAND,facecolor='darkgrey')
        ax.coastlines()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=1, color='gray', alpha=0.5, linestyle='--')
        
        return fig, ax

    def pplot_col(self, col='time', col_wrap=3, var_units='', issym = False, cmap = None, cperc = [4, 96], clim = None, ax = None, dimensions={'time':'time','ydim':'lat','xdim':'lon'}):
        '''Make pretty wrapped subplots.'''
        
        plt.rc('text', usetex=False)
        
        npanels = self[col].size
        nrows = int(np.ceil(npanels/col_wrap))
        ncols = min(npanels, col_wrap)
        
        fig = plt.figure(figsize=(4*ncols, 3*nrows))
        axes = fig.subplots(nrows, ncols, subplot_kw={'projection': ccrs.Robinson()}).flatten()
        
        for i, ax in enumerate(axes):
            if i < npanels:
                self.isel({col: i}).pplot(title = f"{col}={self[col].isel({col: i}).values()}", var_units=var_units, issym = issym, cmap = cmap, cperc = cperc, clim = clim, ax = ax, dimensions=dimensions)
            else:
                fig.delaxes(ax)
        
        return fig, axes
        