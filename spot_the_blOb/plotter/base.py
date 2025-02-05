import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import dask
from pathlib import Path
from PIL import Image
import subprocess
from matplotlib.colors import ListedColormap, BoundaryNorm

class PlotterBase:
    def __init__(self, xarray_obj):
        self.da = xarray_obj
        
        # Cache common features
        self._land = cfeature.LAND.with_scale('50m')
        self._coastlines = cfeature.COASTLINE.with_scale('50m')
    
    def clim_robust(self, data, issym, percentiles=[2, 98]):
        """Base method for computing colour limits"""
        clim = np.nanpercentile(data, percentiles)
        
        if issym:
            clim = np.abs(clim).max()
            clim = np.array([-clim, clim])
        elif percentiles[0] == 0:
            clim = np.array([0, clim[1]])

        return clim
    
    def setup_plot_params(self):
        """Set up common plotting parameters"""
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')

    def setup_id_plot_params(self, cmap=None):
        """Set up parameters for plotting IDs."""
        unique_values = np.unique(self.da.values[~np.isnan(self.da.values)])
        unique_values = unique_values[unique_values > 0]  # Exclude 0 and negative values
        bounds = np.arange(unique_values.min(), unique_values.max() + 2) - 0.5
        n_bins = len(bounds) - 1
        
        if cmap is None:
            np.random.seed(42)  # for reproducibility
            cmap = ListedColormap(np.random.random(size=(n_bins, 3)))
        
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap, norm, 'ID'

    def pplot(self, title=None, var_units='', issym=False, cmap=None, cperc=[4, 96], 
             clim=None, show_colorbar=True, ax=None, grid_labels=True, 
             dimensions={'time':'time','ydim':'lat','xdim':'lon'},
             norm=None, plot_IDs=False, extend='both'):
        """Make pretty plots for data."""
        self.setup_plot_params()
        
        if plot_IDs:
            cmap, norm, var_units = self.setup_id_plot_params(cmap)
            clim = None
            extend = 'neither'
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
        
        # Call the implementation-specific plot function
        ax, cb = self.plot(ax=ax, cmap=cmap, clim=clim, norm=norm, show_colorbar=show_colorbar, extend=extend)
        
        if title is not None:
            ax.set_title(title, size=12)
        
        if show_colorbar:
            if var_units is not None:
                cb.ax.set_ylabel(f'{var_units}', fontsize=10)
            cb.ax.tick_params(labelsize=10)

        ax.add_feature(self._land, facecolor='darkgrey', zorder=2)
        ax.add_feature(self._coastlines, linewidth=0.5, zorder=3)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=grid_labels,
                         linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder=4)
        
        return fig, ax

    def pplot_col(self, col='time', col_wrap=3, var_units='', issym=False, cmap=None, 
                 cperc=[4, 96], clim=None, show_colorbar=True,
                 dimensions={'time':'time','ydim':'lat','xdim':'lon'},
                 plot_IDs=False, extend='both'):
        """Make pretty wrapped subplots."""
        self.setup_plot_params()
        
        npanels = self.da[col].size
        nrows = int(np.ceil(npanels/col_wrap))
        ncols = min(npanels, col_wrap)
        
        if plot_IDs:
            cmap, norm, var_units = self.setup_id_plot_params(cmap)
            clim = None
            extend = 'neither'
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
                    
                self.da.isel({col: i}).plotter.pplot(ax=ax, 
                    title=title, var_units=var_units, issym=issym, 
                    cmap=cmap, cperc=cperc, clim=clim, show_colorbar=False, extend=extend,
                    grid_labels=False, dimensions=dimensions, 
                    norm=norm, plot_IDs=False # Don't re-globalise colorbar norm !
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
            cmap, norm, var_units = self.setup_id_plot_params(cmap)
            clim = None
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
                self._make_frame(
                    da_slice, time_ind, var_units, show_colorbar, 
                    cmap, norm, clim, dimensions, temp_dir, extend
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

    def plot(self, ax, cmap, clim, norm):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement plot()")



@dask.delayed
def make_frame(da_slice, time_ind, var_units, show_colorbar, cmap, norm, clim, 
               dimensions, temp_dir, extend='both'):
    """Create a single frame for movies - dask compatible."""
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes(projection=ccrs.Robinson())

    # Create basic features that would normally be cached
    land = cfeature.LAND.with_scale('50m')
    coastlines = cfeature.COASTLINE.with_scale('50m')

    # Get plotter instance for this slice
    ax, cb = da_slice.plotter.plot(ax=ax, cmap=cmap, clim=clim, norm=norm)
    
    ax.set_title(f"{da_slice.time.dt.strftime('%Y-%m-%d').values}", size=12)
    
    if show_colorbar:
        if var_units is not None:
            cb.ax.set_ylabel(f'{var_units}', fontsize=10)
        cb.ax.tick_params(labelsize=10)
    
    ax.add_feature(land, facecolor='darkgrey')
    ax.add_feature(coastlines, linewidth=0.5)
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