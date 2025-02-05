from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask
from PIL import Image
import subprocess
from matplotlib.colors import ListedColormap, BoundaryNorm
from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Tuple

@dataclass
class PlotConfig:
    """Configuration class for plot parameters"""
    title: Optional[str] = None
    var_units: str = ''
    issym: bool = False
    cmap: Optional[str] = None
    cperc: List[int] = None
    clim: Optional[Tuple[float, float]] = None
    show_colorbar: bool = True
    grid_lines: bool = True
    grid_labels: bool = False
    dimensions: Dict[str, str] = None
    norm: Optional[object] = None
    plot_IDs: bool = False
    extend: str = 'both'
    
    def __post_init__(self):
        if self.cperc is None:
            self.cperc = [4, 96]
        if self.dimensions is None:
            self.dimensions = {'time': 'time', 'ydim': 'lat', 'xdim': 'lon'}
        if self.plot_IDs:
            self.show_colorbar = False

class PlotterBase:
    def __init__(self, xarray_obj):
        self.da = xarray_obj
        
        # Cache common features
        self._land = cfeature.LAND.with_scale('50m')
        self._coastlines = cfeature.COASTLINE.with_scale('50m')
    
    def _setup_common_params(self, config: PlotConfig) -> Tuple:
        """Centralise common parameter setup"""
        self.setup_plot_params()
        
        if config.plot_IDs:
            cmap, norm, var_units = self.setup_id_plot_params(config.cmap)
            clim = None
            extend = 'neither'
        else:
            if config.cmap is None:
                cmap = 'RdBu_r' if config.issym else 'viridis'
            else:
                cmap = config.cmap
            norm = config.norm
            if config.clim is None and norm is None:
                clim = self.clim_robust(self.da.values, config.issym, config.cperc)
            else:
                clim = config.clim
            var_units = config.var_units
            extend = config.extend
                
        return cmap, norm, clim, var_units, extend

    def _setup_axes(self, ax=None):
        """Create or use existing axes with projection"""
        if ax is None:
            fig = plt.figure(figsize=(7, 5))
            ax = plt.axes(projection=ccrs.Robinson())
        else:
            fig = ax.get_figure()
        return fig, ax

    def _add_map_features(self, ax, grid_lines=True, grid_labels=True):
        """Add common map features to the plot"""
        ax.add_feature(self._land, facecolor='darkgrey', zorder=2)
        ax.add_feature(self._coastlines, linewidth=0.5, zorder=3)
        if grid_lines:
            ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=grid_labels,
                        linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder=4)

    def _setup_colorbar(self, fig, im, show_colorbar, var_units, extend='both', 
                       position=None):
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

    def _get_title(self, time_index, col_name, dimensions):
        """Generate appropriate title based on dimension"""
        if col_name == dimensions['time']:
            return f"{self.da[col_name].isel({col_name: time_index}).time.dt.strftime('%Y-%m-%d').values}"
        return f"{col_name}={self.da[col_name].isel({col_name: time_index}).values}"


    def single_plot(self, config: PlotConfig, ax=None):
        """Make a single plot with given configuration"""
        cmap, norm, clim, var_units, extend = self._setup_common_params(config)
        
        fig, ax = self._setup_axes(ax)
        
        # Call implementation-specific plot function
        ax, im = self.plot(ax=ax, cmap=cmap, clim=clim, norm=norm)
        
        if config.title:
            ax.set_title(config.title, size=12)
        
        self._setup_colorbar(fig, im, config.show_colorbar, var_units, extend)
        self._add_map_features(ax, config.grid_lines, config.grid_labels)
        
        return fig, ax

    def multi_plot(self, config: PlotConfig, col='time', col_wrap=3):
        """Make wrapped subplots with given configuration"""
        npanels = self.da[col].size
        nrows = int(np.ceil(npanels/col_wrap))
        ncols = min(npanels, col_wrap)
        
        cmap, norm, clim, var_units, extend = self._setup_common_params(config)
        
        fig = plt.figure(figsize=(6*ncols, 3*nrows))
        axes = fig.subplots(nrows, ncols, 
                          subplot_kw={'projection': ccrs.Robinson()}).flatten()
        
        # Create a single plotter instance to be reused
        base_plotter = type(self)(self.da)
        for attr in ['fpath_tgrid', 'fpath_ckdtree']:
            if hasattr(self, attr):
                setattr(base_plotter, attr, getattr(self, attr))
        
        for i, ax in enumerate(axes):
            if i < npanels:
                title = self._get_title(i, col, config.dimensions)
                
                # Create new config for individual panel
                panel_config = PlotConfig(
                    title=title, cmap=cmap, clim=clim,
                    show_colorbar=False, grid_labels=False,
                    norm=norm, plot_IDs=False, extend=extend
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
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            self._setup_colorbar(fig, sm, True, var_units, extend,
                               position=[0.92, 0.15, 0.02, 0.7])
        
        return fig, axes
    

    def animate(self, config: PlotConfig, plot_dir='./'):
        """Create an animation from time series data"""
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(exist_ok=True)
        temp_dir = plot_dir / "blobs_seq"
        temp_dir.mkdir(exist_ok=True)
        output_file = plot_dir / f"movie_{self.da.name}.mp4"
        
        # Set up plotting parameters
        cmap, norm, clim, var_units, extend = self._setup_common_params(config)
        
        plot_params = {
            'cmap': cmap,
            'norm': norm,
            'clim': clim,
            'var_units': var_units,
            'extend': extend,
            'show_colorbar': config.show_colorbar
        }
        
        # Set up grid information if needed
        grid_info = None
        if hasattr(self, 'fpath_tgrid') or hasattr(self, 'fpath_ckdtree'):
            grid_info = {
                'type': 'unstructured',
                'tgrid_path': getattr(self, 'fpath_tgrid', None),
                'ckdtree_path': getattr(self, 'fpath_ckdtree', None),
                'res': 0.3
            }
        
        # Generate frames using dask for parallel processing
        delayed_tasks = []
        for time_ind in range(len(self.da[config.dimensions['time']])):
            data_slice = self.da.isel({config.dimensions['time']: time_ind}).values
            plot_params['time_str'] = str(
                self.da[config.dimensions['time']]
                .isel({config.dimensions['time']: time_ind})
                .dt.strftime('%Y-%m-%d').values
            )
            delayed_tasks.append(
                make_frame(data_slice, time_ind, temp_dir, plot_params, grid_info)
            )
        
        filenames = dask.compute(*delayed_tasks)
        filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Create movie using ffmpeg
        subprocess.run([
            'ffmpeg', '-y', '-threads', '0', '-framerate', '10',
            '-i', str(temp_dir / 'time_%04d.jpg'),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(output_file)
        ], check=True)
        
        return str(output_file)


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
        """Set up parameters for plotting IDs"""
        unique_values = np.unique(self.da.values[~np.isnan(self.da.values)])
        unique_values = unique_values[unique_values > 0]
        bounds = np.arange(unique_values.min(), unique_values.max() + 2) - 0.5
        n_bins = len(bounds) - 1
        
        if cmap is None:
            np.random.seed(42)
            cmap = ListedColormap(np.random.random(size=(n_bins, 3)))
        
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap, norm, 'ID'



@dask.delayed
def make_frame(data_slice, time_ind, temp_dir, plot_params, grid_info=None):
    """Create a single frame for movies - minimise memory usage with dask
    
    Args:
        data_slice: The data for this specific frame
        time_ind: Frame index
        temp_dir: Directory for temporary files
        plot_params: Dict containing plotting parameters
        grid_info: Dict containing grid paths and settings for unstructured data
    """
    # Set up plotting parameters
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    
    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes(projection=ccrs.Robinson())
    
    # Set up plot kwargs
    plot_kwargs = {
        'transform': ccrs.PlateCarree(),
        'cmap': plot_params['cmap'],
        'shading': 'auto'
    }
    
    if plot_params.get('norm'):
        plot_kwargs['norm'] = plot_params['norm']
    elif plot_params.get('clim'):
        plot_kwargs['vmin'] = plot_params['clim'][0]
        plot_kwargs['vmax'] = plot_params['clim'][1]
    
    # Handle different grid types
    if grid_info and grid_info.get('type') == 'unstructured':
        from .unstructured import _load_ckdtree, _load_triangulation
        if grid_info.get('ckdtree_path'):
            # Use cached ckdtree data
            ckdt_data = _load_ckdtree(grid_info['ckdtree_path'], grid_info.get('res', 0.3))
            grid_data = data_slice[ckdt_data['indices']].reshape(
                ckdt_data['lat'].size,
                ckdt_data['lon'].size
            )
            grid_data = np.ma.masked_invalid(grid_data)
            im = ax.pcolormesh(
                ckdt_data['lon'], ckdt_data['lat'],
                grid_data, **plot_kwargs
            )
        elif grid_info.get('tgrid_path'):
            # Use triangulation
            triang = _load_triangulation(grid_info['tgrid_path'])
            data_masked = np.ma.masked_invalid(data_slice)
            im = ax.tripcolor(triang, data_masked, **plot_kwargs)
    else:
        # Regular grid plotting
        im = ax.imshow(data_slice, **plot_kwargs)
    
    time_str = plot_params.get('time_str', f'Frame {time_ind}')
    ax.set_title(time_str, size=12)
    
    if plot_params.get('show_colorbar'):
        cb = plt.colorbar(im, shrink=0.6, ax=ax, extend=plot_params.get('extend', 'both'))
        if plot_params.get('var_units'):
            cb.ax.set_ylabel(plot_params['var_units'], fontsize=10)
        cb.ax.tick_params(labelsize=10)
    
    land = cfeature.LAND.with_scale('50m')
    coastlines = cfeature.COASTLINE.with_scale('50m')
    ax.add_feature(land, facecolor='darkgrey', zorder=2)
    ax.add_feature(coastlines, linewidth=0.5, zorder=3)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder=4)
    
    # Save and process frame
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