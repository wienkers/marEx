from .base import PlotterBase
from .structured import StructuredPlotter
from .unstructured import UnstructuredPlotter, clear_cache
import xarray as xr

# Global variable to store the grid path
_fpath_tgrid = None
_fpath_ckdtree = None

def marEx_plotter(data_type='structured'):
    """
    Factory function to return the appropriate plotter based on data type.
    
    Args:
        data_type (str): Either 'structured' or 'unstructured'
    
    Returns:
        Appropriate plotter class
    """
    if data_type.lower() == 'unstructured':
        return UnstructuredPlotter
    return StructuredPlotter

def register_plotter(xarray_obj):
    """
    Determine the appropriate plotter to use based on the data structure.
    This function is called automatically by xarray's accessor system.
    
    The determination is based on how lat/lon coordinates are structured:
    - Unstructured: lat and lon exist as 1D coordinates but not as dimensions
    - Structured: lat and lon exist as both coordinates and dimensions
    
    Returns:
        Appropriate plotter instance for the data structure
    """
    has_lat_lon_coords = 'lat' in xarray_obj.coords and 'lon' in xarray_obj.coords
    has_lat_lon_dims = 'lat' in xarray_obj.dims and 'lon' in xarray_obj.dims
    
    # For unstructured data, lat/lon are coordinates but not dimensions
    is_unstructured = (has_lat_lon_coords and not has_lat_lon_dims)
    
    plotter_class = marEx_plotter('unstructured' if is_unstructured else 'structured')
    
    # Set grid path if available
    plotter = plotter_class(xarray_obj)
    if is_unstructured and _fpath_tgrid is not None and _fpath_ckdtree is not None:
        plotter.specify_grid(fpath_tgrid=_fpath_tgrid, fpath_ckdtree=_fpath_ckdtree)
    
    return plotter

def specify_grid(fpath_tgrid=None, fpath_ckdtree=None):
    """
    Set the global unstructured grid path that will be used by all unstructured plotters.
    
    Args:
        fpath_tgrid: Path to the triangulation grid file
        fpath_ckdtree: Path to the pre-computed KDTree indices directory
    """
    global _fpath_tgrid, _fpath_ckdtree
    _fpath_tgrid = str(fpath_tgrid)
    _fpath_ckdtree = str(fpath_ckdtree)

# Register the accessor
xr.register_dataarray_accessor('plotter')(register_plotter)