import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap, BoundaryNorm, SymLogNorm
import dask
from pathlib import Path
from PIL import Image
import subprocess


from .base import PlotterBase


class UnstructuredPlotter(PlotterBase):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
        # Cache the grid path - this should be set by the user or configured globally
        self.fpath_tgrid = None
        self.fpath_ckdtree = None
        
    def set_grid_path(self, fpath_tgrid, fpath_ckdtree):
        """Set the path to the ICON grid file."""
        self.fpath_tgrid = str(fpath_tgrid)
        self.fpath_ckdtree = str(fpath_ckdtree)
    
    def pplot(self, title=None, var_units='', issym=False, cmap=None, cperc=[4, 96], 
             clim=None, show_colorbar=True, ax=None, grid_labels=True, 
             dimensions={'time':'time','xdim':'ncells'},
             norm=None, plot_IDs=False, extend='both', res=0.3):
        """Make pretty plots for unstructured (ICON) data."""
        self.setup_plot_params()
        
        if plot_IDs:
            unique_values = np.unique(self.da.values[~np.isnan(self.da.values)])
            unique_values = unique_values[unique_values > 0]
            bounds = np.arange(unique_values.min(), unique_values.max() + 2) - 0.5
            n_bins = len(bounds) - 1
            
            if cmap is None:
                np.random.seed(42)
                cmap = ListedColormap(np.random.random(size=(n_bins, 3)))
            norm = BoundaryNorm(bounds, cmap.N)
            clim = [bounds[0], bounds[-1]]
            var_units = 'ID'
        else:
            if cmap is None:
                cmap = 'RdBu_r' if issym else 'viridis'
            if clim is None and norm is None:
                clim = self.clim_robust(self.da, issym, cperc)
        
        if ax is None:
            fig = plt.figure(figsize=(7, 5))
            ax = plt.axes(projection=ccrs.Robinson())
        else:
            fig = ax.get_figure()
            
        # Create colourbar
        cb = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, shrink=0.6, extend=extend)
        
        # Plot using pyicon
        self.plot(ax=ax, cax=cb.ax, res=res, 
                         cmap=cmap, clim=clim)
        
        if title is not None:
            ax.set_title(title, size=12)
        
        if show_colorbar:
            if var_units is not None:
                cb.ax.set_ylabel(f'{var_units}', fontsize=10)
            cb.ax.tick_params(labelsize=10)

        ax.add_feature(self._land, facecolor='darkgrey')
        ax.add_feature(self._coastlines, linewidth=0.5)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=grid_labels,
                         linewidth=1, color='gray', alpha=0.5, linestyle='--')
        
        return fig, ax
    
    def pplot_col(self, col='time', col_wrap=3, var_units='', issym=False, cmap=None, 
                 cperc=[4, 96], clim=None, show_colorbar=True,
                 dimensions={'time':'time','ydim':'lat','xdim':'lon'},
                 plot_IDs=False, extend='both'):
        """Make pretty wrapped subplots for unstructured data."""
        self.setup_plot_params()
        
        npanels = self.da[col].size
        nrows = int(np.ceil(npanels/col_wrap))
        ncols = min(npanels, col_wrap)
        
        if plot_IDs:
            unique_values = np.unique(self.da.values[~np.isnan(self.da.values)])
            unique_values = unique_values[unique_values > 0]
            bounds = np.arange(unique_values.min(), unique_values.max() + 2) - 0.5
            n_bins = len(bounds) - 1
            
            if cmap is None:
                np.random.seed(42)
                cmap = ListedColormap(np.random.random(size=(n_bins, 3)))
            norm = BoundaryNorm(bounds, cmap.N)
            clim = [bounds[0], bounds[-1]]
            var_units = 'ID'
        else:
            norm = None
            if clim is None:
                clim = self.clim_robust(self.da.values, issym, cperc)
        
        fig = plt.figure(figsize=(6*ncols, 3*nrows))
        axes = fig.subplots(nrows, ncols, subplot_kw={'projection': ccrs.Robinson()}).flatten()
        
        for i, ax in enumerate(axes):
            if i < npanels:
                if col == dimensions['time']:
                    title = f"{self.da[col].isel({col: i}).time.dt.strftime('%Y-%m-%d').values}"
                else:
                    title = f"{col}={self.da[col].isel({col: i}).values}"
                    
                self.da.isel({col: i}).plotter.pplot(
                    title=title, var_units=var_units, issym=issym, 
                    cmap=cmap, cperc=cperc, clim=clim, show_colorbar=False,
                    grid_labels=False, dimensions=dimensions, 
                    norm=norm, plot_IDs=plot_IDs, extend=extend
                )
            else:
                fig.delaxes(ax)
        
        if show_colorbar:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cbar_ax, extend=extend)
            cb.ax.set_ylabel(var_units, fontsize=10)
            cb.ax.tick_params(labelsize=10)
        
        return fig, axes
    
    def pplot_mov(self, plot_dir='./', var_units='', issym=False, cmap=None, 
                 cperc=[4, 96], clim=None, show_colorbar=True,
                 dimensions={'time':'time','ydim':'lat','xdim':'lon'},
                 plot_IDs=False, extend='both'):
        """Create an animation from time series data."""
        self.setup_plot_params()
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(exist_ok=True)
        
        if plot_IDs:
            unique_values = np.unique(self.da.values[~np.isnan(self.da.values)])
            unique_values = unique_values[unique_values > 0]
            bounds = np.arange(unique_values.min(), unique_values.max() + 2) - 0.5
            n_bins = len(bounds) - 1
            
            if cmap is None:
                np.random.seed(42)
                cmap = ListedColormap(np.random.random(size=(n_bins, 3)))
            norm = BoundaryNorm(bounds, cmap.N)
            clim = [bounds[0], bounds[-1]]
            var_units = 'ID'
            show_colorbar = False
        else:
            norm = None
            if clim is None:
                clim = self.clim_robust(self.da.values, issym, cperc)
        
        temp_dir = plot_dir / "blobs_seq"
        temp_dir.mkdir(exist_ok=True)
        output_file = plot_dir / f"movie_{self.da.name}.mp4"
        
        delayed_tasks = []
        for time_ind in range(len(self.da[dimensions['time']])):
            da_slice = self.da.isel({dimensions['time']: time_ind})
            delayed_tasks.append(
                make_frame(
                    da_slice, time_ind, var_units, show_colorbar, 
                    cmap, norm, clim, dimensions, temp_dir, 
                    self.fpath_tgrid, extend
                )
            )
        
        filenames = dask.compute(*delayed_tasks)
        filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        subprocess.run([
            'ffmpeg', '-y', '-threads', '0', '-framerate', '10',
            '-i', str(temp_dir / 'time_%04d.jpg'),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(output_file)
        ], check=True)
        
        return str(output_file)

    
    def plot(self, ax, cax, cbar_pos='vertical', res=0.3, cmap='viridis',
         clim=None, extend='both', 
         transform=ccrs.PlateCarree(), mask_data=True, logplot=False, 
         coastlines_color='k', land_facecolor='darkgrey', use_regular_grid=True):
        """
        Plot unstructured data matching pyicon's implementation.
        
        Parameters
        ----------
        data : xarray.DataArray
            Data to plot
        ax : matplotlib.axes.Axes
            Axes to plot on
        cax : matplotlib.axes.Axes
            Axes for colorbar
        cbar_pos : str
            Position of colorbar ('vertical' or 'horizontal')
        res : float
            Resolution parameter (not used when fpath_ckdtree is provided)
        cmap : str or matplotlib.colors.Colormap
            Colormap to use
        clim : tuple
            Color limits (min, max)
        fpath_tgrid : str
            Path to triangulation file
        fpath_ckdtree : str
            Path to pre-computed KDTree indices file
        extend : str
            How to extend the colorbar
        transform : cartopy.crs
            Coordinate transform for plotting
        mask_data : bool
            Whether to mask data where values are 0
        logplot : bool
            Whether to use log scale for colors
        coastlines_color : str
            Color of coastlines
        land_facecolor : str
            Color of land
        use_regular_grid : bool
            Whether to interpolate to a regular grid (True) or use triangulation (False)
        """
        
        data = self.da
        
        # Handle data masking
        if mask_data:
            data = data.where(data != 0)
            if isinstance(data.values, np.ma.MaskedArray):
                data_values = data.values.filled(np.nan)
            else:
                data_values = data.values

        if use_regular_grid:
            if self.fpath_ckdtree is None:
                raise ValueError("fpath_ckdtree is required when use_regular_grid=True")
            # Interpolate using pre-computed KDTree indices
            grid_lon, grid_lat, grid_data = interpolate_with_ckdtree(
                data_values, self.fpath_ckdtree, res)
            x, y, plot_data = grid_lon, grid_lat, grid_data
        else:
            # Use triangulation from file
            if self.fpath_tgrid is None:
                raise ValueError("fpath_tgrid is required when use_regular_grid=False")
            
            grid_data = xr.open_dataset(self.fpath_tgrid)
            # Extract triangulation vertices - convert to 0-based indexing
            triangles = grid_data.vertex_of_cell.values.T - 1
            # Create matplotlib triangulation object
            x = Triangulation(grid_data.clon.values, 
                            grid_data.clat.values,
                            triangles)
            y = None
            plot_data = data_values
            
        # Do the actual plotting using shade function
        hm = shade(x, y, plot_data, ax=ax, cax=cax, clim=clim, cmap=cmap,
                extend=extend, transform=transform, logplot=logplot,
                use_regular_grid=use_regular_grid)

        # Add map features if using cartopy
        if isinstance(ax.projection, ccrs.Projection):
            ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=land_facecolor, zorder=2)
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor=coastlines_color, linewidth=0.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                            linewidth=1, color='gray', alpha=0.5, linestyle='--')

        

        return ax, hm[1]  # Return ax and colorbar



def interpolate_with_ckdtree(data, fpath_ckdtree, res, gname='r2b9_oce_r0004'):
    """
    Interpolate unstructured data using pre-computed KDTree indices.
    
    Parameters
    ----------
    data : np.ndarray
        Data to interpolate
    fpath_ckdtree : str
        Path to file containing pre-computed KDTree indices
    """
    # Load the ckdtree indices and grid information
    ckdtree_file = fpath_ckdtree + '/rectgrids/' + f'{gname}_res{res:3.2f}_180W-180E_90S-90N.nc'
    ds_ckdt = xr.open_dataset(ckdtree_file)
    
    # Get the KDTree indices
    indices = ds_ckdt.ickdtree_c.values
    
    # Get the regular grid coordinates
    grid_lon = ds_ckdt.lon.values
    grid_lat = ds_ckdt.lat.values
    
    # Create meshgrid for plotting
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
    
    # Use indices to create interpolated data
    grid_data = data[indices].reshape(grid_lat.size, grid_lon.size)
    
    return grid_lon_2d, grid_lat_2d, grid_data

def shade(x, y, datai, ax, cax, clim=None, cmap='viridis', extend='both',
          transform=None, logplot=False, rasterized=True, use_regular_grid=False):
    """
    Core shading function matching pyicon's implementation
    """
    # Handle data masking and log transform
    data = datai.copy()
    data = np.ma.masked_invalid(data)
    if logplot and isinstance(data, np.ma.MaskedArray):
        data[data <= 0.0] = np.ma.masked
        data = np.ma.log10(data)
    elif logplot and not isinstance(data, np.ma.MaskedArray):
        data[data <= 0.0] = np.nan
        data = np.log10(data)

    # Handle color limits
    if clim is None:
        clim = [data.min(), data.max()]
    elif isinstance(clim, str) and clim == 'sym':
        clim = np.abs(data).max()
    clim = np.array(clim)
    if clim.size == 1:
        clim = np.array([-1, 1]) * clim
    if clim[0] is None:
        clim[0] = data.min()
    if clim[1] is None:
        clim[1] = data.max()

    # Handle colormap
    if (clim[0] == -clim[1]) and cmap == 'auto':
        cmap = 'RdBu_r'
    elif cmap == 'auto':
        cmap = 'RdYlBu_r'
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Plot using either pcolormesh (regular grid) or tripcolor (triangulation)
    kwargs = {}
    if transform is not None:
        kwargs['transform'] = transform

    if use_regular_grid:
        # For regular grid data
        hm = ax.pcolormesh(x, y, data,
                          cmap=cmap,
                          vmin=clim[0], vmax=clim[1],
                          rasterized=rasterized,
                          shading='auto',
                          **kwargs)
    else:
        # For triangulation data
        hm = ax.tripcolor(x, y, data,
                         cmap=cmap,
                         vmin=clim[0], vmax=clim[1],
                         rasterized=rasterized,
                         **kwargs)

    # Create colorbar
    cb = plt.colorbar(mappable=hm, cax=cax, extend=extend)
    cb.solids.set_edgecolor("face")
    try:
        cb.formatter.set_powerlimits((-3, 3))
    except:
        pass
    
    return [hm, cb]

    
@dask.delayed
def make_frame(da_slice, time_ind, var_units, show_colorbar, cmap, norm, clim, 
               dimensions, temp_dir, fpath_tgrid, extend='both'):
    """Standalone function to create a single frame for movies - dask compatible."""
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes(projection=ccrs.Robinson())
    
    # Create colorbar
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, shrink=0.6, extend=extend)
    
    # Plot using pyicon
    da_slice.pyic.plot(ax=ax, cax=cb.ax, cbar_pos='vertical', res=0.1, 
                      cmap=cmap, clim=clim, fpath_tgrid=fpath_tgrid)
    
    ax.set_title(f"{da_slice.time.dt.strftime('%Y-%m-%d').values}", size=12)
    
    if show_colorbar:
        if var_units is not None:
            cb.ax.set_ylabel(f'{var_units}', fontsize=10)
        cb.ax.tick_params(labelsize=10)
    
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='darkgrey')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    filename = f'time_{time_ind:04d}.jpg'
    temp_file = temp_dir / f'temp_{filename}'
    fig.savefig(str(temp_file), dpi=300, bbox_inches='tight')
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