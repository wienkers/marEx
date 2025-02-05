import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.tri import Triangulation
from pathlib import Path

from .base import PlotterBase

# Global cache for grid data
_GRID_CACHE = {
    'triangulation': {},  # key: grid_path, value: triangulation object
    'ckdtree': {}         # key: (ckdtree_path, res), value: ckdtree data
}

def clear_cache():
    """Clear the global grid cache."""
    _GRID_CACHE['triangulation'].clear()
    _GRID_CACHE['ckdtree'].clear()

def _load_triangulation(fpath_tgrid):
    """Load and cache triangulation data globally."""
    fpath_tgrid = str(fpath_tgrid)  # Convert Path to string for dict key
    if fpath_tgrid not in _GRID_CACHE['triangulation']:
        # Only load required variables
        grid_data = xr.open_dataset(
            fpath_tgrid,
            chunks={},  # Load as numpy array
            drop_variables=[v for v in xr.open_dataset(fpath_tgrid).variables 
                          if v not in ['vertex_of_cell', 'clon', 'clat']]
        )
        # Extract triangulation vertices - convert to 0-based indexing
        triangles = grid_data.vertex_of_cell.values.T - 1
        # Create matplotlib triangulation object
        _GRID_CACHE['triangulation'][fpath_tgrid] = Triangulation(
            grid_data.clon.values,
            grid_data.clat.values,
            triangles
        )
        grid_data.close()
        
    return _GRID_CACHE['triangulation'][fpath_tgrid]

def _load_ckdtree(fpath_ckdtree, res):
    """Load and cache ckdtree data globally for a specific resolution."""
    cache_key = (str(fpath_ckdtree), res)  # Convert Path to string for dict key
    
    if cache_key not in _GRID_CACHE['ckdtree']:
        # Format resolution string to match file naming
        ckdtree_file = Path(fpath_ckdtree) / f"res{res:3.2f}.nc"
        
        if not ckdtree_file.exists():
            raise ValueError(f"No ckdtree file found for resolution {res}")
            
        ds_ckdt = xr.open_dataset(ckdtree_file)
        _GRID_CACHE['ckdtree'][cache_key] = {
            'indices': ds_ckdt.ickdtree_c.values,
            'lon': ds_ckdt.lon.values,
            'lat': ds_ckdt.lat.values
        }
        ds_ckdt.close()
        
    return _GRID_CACHE['ckdtree'][cache_key]

class UnstructuredPlotter(PlotterBase):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
        # Paths
        self.fpath_tgrid = None
        self.fpath_ckdtree = None
        
    def specify_grid(self, fpath_tgrid=None, fpath_ckdtree=None):
        """Set the path to the unstructured grid files.
        
        Optional Parameters
        ----------
        fpath_tgrid : str or Path
            Path to triangulation grid file
        fpath_ckdtree : str or Path
            Path to directory containing ckdtree files
            Expected structure: fpath_ckdtree/rectgrids/res{resolution}.nc
        """
        self.fpath_tgrid = Path(fpath_tgrid)
        self.fpath_ckdtree = Path(fpath_ckdtree)

    def plot(self, ax, cmap='viridis', clim=None, norm=None, show_colorbar=True, extend='both'):
        """Implement plotting for unstructured data."""
        
        if self.fpath_ckdtree is not None:
            # Interpolate using pre-computed KDTree indices
            grid_lon, grid_lat, grid_data = self._interpolate_with_ckdtree(self.da.values, res=0.3)
            
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
            
            # Mask NaNs
            grid_data = np.ma.masked_invalid(grid_data)
            
            im = ax.pcolormesh(grid_lon, grid_lat, grid_data, **plot_kwargs)
            
        else:
            # Use triangulation from file if available
            if self.fpath_tgrid is None:
                raise ValueError("Either fpath_tgrid or fpath_ckdtree must be provided")
            
            triang = _load_triangulation(self.fpath_tgrid)
            
            plot_kwargs = {
                'transform': ccrs.PlateCarree(),
                'cmap': cmap
            }
            
            if norm is not None:
                plot_kwargs['norm'] = norm
            elif clim is not None:
                plot_kwargs['vmin'] = clim[0]
                plot_kwargs['vmax'] = clim[1]
            
            # Mask NaNs
            native_data = self.da.copy()
            native_data = np.ma.masked_invalid(native_data)
            
            im = ax.tripcolor(triang, native_data, **plot_kwargs)
        
        cb = plt.colorbar(im, shrink=0.6, ax=ax, extend=extend) if show_colorbar else None
        
        return ax, cb

    def _interpolate_with_ckdtree(self, data, res):
        """
        Interpolate unstructured data using pre-computed KDTree indices.
        
        Parameters
        ----------
        data : np.ndarray
            Data to interpolate
        res : float
            Resolution for interpolation grid (e.g., 0.02, 0.1, 0.3, 1.0)
        """
        if self.fpath_ckdtree is None:
            raise ValueError("ckdtree path not specified")
            
        # Load or get cached ckdtree data
        ckdt_data = _load_ckdtree(self.fpath_ckdtree, res)
        
        # Create meshgrid for plotting
        grid_lon_2d, grid_lat_2d = np.meshgrid(ckdt_data['lon'], ckdt_data['lat'])
        
        # Use indices to create interpolated data
        grid_data = data[ckdt_data['indices']].reshape(
            ckdt_data['lat'].size,
            ckdt_data['lon'].size
        )
        
        return grid_lon_2d, grid_lat_2d, grid_data