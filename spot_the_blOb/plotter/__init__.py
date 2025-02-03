from .base import PlotterBase
from .structured import StructuredPlotter
from .unstructured import UnstructuredPlotter
import xarray as xr

# Global variable to store the grid path
_UNSTRUCT_GRID_PATH = None

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
    if is_unstructured and _UNSTRUCT_GRID_PATH is not None:
        plotter.set_grid_path(_UNSTRUCT_GRID_PATH)
    
    return plotter


def set_grid_path(path):
    """
    Set the global ICON grid path that will be used by all unstructured plotters.
    
    Args:
        path: Path to the ICON grid file
    """
    global _UNSTRUCT_GRID_PATH
    _UNSTRUCT_GRID_PATH = path


# Register the accessor
xr.register_dataarray_accessor('plotter')(register_plotter)
