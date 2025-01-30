from .base import PlotterBase
from .structured import StructuredPlotter
from .unstructured import UnstructuredPlotter
import xarray as xr

def get_plotter(data_type='structured'):
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

# Register the appropriate plotter based on data structure
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
    
    plotter_class = get_plotter('unstructured' if is_unstructured else 'structured')
    return plotter_class(xarray_obj)


# Register the accessor
xr.register_dataarray_accessor('plotter')(register_plotter)
