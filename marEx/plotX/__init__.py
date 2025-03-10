from .base import PlotterBase, PlotConfig
from .gridded import GriddedPlotter
from .unstructured import UnstructuredPlotter, clear_cache
import xarray as xr
import warnings

# Global variables to store grid information
_fpath_tgrid = None
_fpath_ckdtree = None
_grid_type = None

def _detect_grid_type(xarray_obj):
    """
    Deduce grid type based on coordinate structure.
    
    Returns:
        str: 'gridded' or 'unstructured'
    """
    has_lat_lon_coords = 'lat' in xarray_obj.coords and 'lon' in xarray_obj.coords
    has_lat_lon_dims = 'lat' in xarray_obj.dims and 'lon' in xarray_obj.dims
    
    # For unstructured data, lat/lon are coordinates but not dimensions
    return 'unstructured' if (has_lat_lon_coords and not has_lat_lon_dims) else 'gridded'

def register_plotter(xarray_obj):
    """
    Determine the appropriate plotter to use based on the data structure.
    This function is called automatically by xarray's accessor system.
    
    First checks if grid type was specified via specify_grid(),
    then falls back to coordinate-based detection if needed.
    
    Returns:
        Appropriate plotter instance for the data structure
    """
    global _grid_type
    
    # Determine grid type
    detected_type = _detect_grid_type(xarray_obj)
    
    # If grid type was explicitly specified, check for consistency
    if _grid_type is not None:
        if _grid_type != detected_type:
            warnings.warn(
                f"Specified grid type '{_grid_type}' differs from detected type '{detected_type}' "
                f"based on coordinate structure. Using specified type '{_grid_type}'."
            )
        final_type = _grid_type
    else:
        final_type = detected_type
    
    # Create appropriate plotter
    plotter_class = UnstructuredPlotter if final_type.lower() == 'unstructured' else GriddedPlotter
    plotter = plotter_class(xarray_obj)
    
    # Set grid path if available for unstructured grids
    if final_type == 'unstructured' and _fpath_tgrid is not None and _fpath_ckdtree is not None:
        plotter.specify_grid(fpath_tgrid=_fpath_tgrid, fpath_ckdtree=_fpath_ckdtree)
    
    return plotter

def specify_grid(grid_type=None, fpath_tgrid=None, fpath_ckdtree=None):
    """
    Set the global grid specification that will be used by all plotters.
    
    Args:
        grid_type: str, either 'gridded' or 'unstructured'.
                  If specified, this will be used as the primary method
                  to determine grid type.
        fpath_tgrid: Path to the triangulation grid file
        fpath_ckdtree: Path to the pre-computed KDTree indices directory
    """
    global _fpath_tgrid, _fpath_ckdtree, _grid_type
    
    if grid_type is not None and grid_type.lower() not in ['gridded', 'unstructured']:
        raise ValueError("grid_type must be either 'gridded' or 'unstructured'")
    
    _fpath_tgrid = str(fpath_tgrid) if fpath_tgrid else None
    _fpath_ckdtree = str(fpath_ckdtree) if fpath_ckdtree else None
    _grid_type = grid_type.lower() if grid_type else None

# Register the accessor
xr.register_dataarray_accessor('plotX')(register_plotter)