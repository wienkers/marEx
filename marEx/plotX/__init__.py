"""
MarEx-PlotX: Marine Extremes Visualisation Module

Comprehensive visualisation module for marine extreme events supporting both
structured (regular grid) and unstructured oceanographic data.
Provides automated grid detection and specialised plotting capabilities optimised
for each data structure type.

Core capabilities:

* Polymorphic plotting with automatic grid type detection
* Single plot generation with customisable styling and projections
* Multi-panel plotting for comparative analysis
* Animation generation for temporal visualisation
* Memory-efficient handling of large oceanographic datasets

Supported data formats:

* Structured data: 3D arrays (time, lat, lon) for regular rectangular grids
* Unstructured data: 2D arrays (time, cell) for irregular triangular meshes
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import xarray as xr

from ..exceptions import ConfigurationError
from ..logging_config import get_logger
from .base import PlotConfig
from .gridded import GriddedPlotter
from .unstructured import UnstructuredPlotter

# Get module logger
logger = get_logger(__name__)

# Global variables to store grid information
_fpath_tgrid: Optional[str] = None
_fpath_ckdtree: Optional[str] = None
_grid_type: Optional[str] = None


def _detect_grid_type(
    xarray_obj: Union[xr.Dataset, xr.DataArray],
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
) -> str:
    """
    Deduce grid type based on coordinate structure and dimension mapping.

    Args:
        xarray_obj: The xarray object to analyze
        dimensions: Optional dimension mapping
        coordinates: Optional coordinate mapping

    Returns:
        str: 'gridded' or 'unstructured'
    """
    # Use default mappings if not provided
    if dimensions is None:
        dimensions = {"time": "time", "y": "lat", "x": "lon"}
    if coordinates is None:
        coordinates = {"time": "time", "y": "lat", "x": "lon"}

    # Check if y dimension exists (key indicator for gridded vs unstructured)
    has_y_dim = "y" in dimensions and dimensions["y"] in xarray_obj.dims

    # For coordinate checking, use the coordinate mapping
    x_coord = coordinates.get("x", "lon")
    y_coord = coordinates.get("y", "lat")
    has_spatial_coords = x_coord in xarray_obj.coords and y_coord in xarray_obj.coords

    # Gridded: has both x and y dimensions
    # Unstructured: has only x dimension (typically for cells) but x,y coordinates
    grid_type = "gridded" if has_y_dim else "unstructured"

    logger.debug(f"Detected grid type: {grid_type} (y_dim: {has_y_dim}, spatial_coords: {has_spatial_coords})")
    return grid_type


class PlotXAccessor:
    """Xarray accessor for plotX functionality with support for custom dimensions and coordinates."""

    def __init__(self, xarray_obj: xr.DataArray):
        """Initialise the PlotXAccessor."""
        self._obj = xarray_obj

    def __call__(
        self,
        dimensions: Optional[Dict[str, str]] = None,
        coordinates: Optional[Dict[str, str]] = None,
    ) -> Union["GriddedPlotter", "UnstructuredPlotter"]:
        """
        Create a plotter instance with optional custom dimensions and coordinates.

        Args:
            dimensions: Optional mapping of conceptual dimensions to actual dimension names
            coordinates: Optional mapping of conceptual coordinates to actual coordinate names

        Returns:
            Appropriate plotter instance for the data structure
        """
        # Note: _grid_type is accessed globally

        # Determine grid type
        detected_type = _detect_grid_type(self._obj, dimensions, coordinates)

        # If grid type was explicitly specified, check for consistency
        if _grid_type is not None:
            if _grid_type != detected_type:  # pragma: no cover
                logger.warning(
                    f"Specified grid type '{_grid_type}' differs from detected type '{detected_type}' "
                    f"based on coordinate structure. Using specified type '{_grid_type}'"
                )
                warnings.warn(
                    f"Specified grid type '{_grid_type}' differs from detected type '{detected_type}' "
                    f"based on coordinate structure. Using specified type '{_grid_type}'.",
                    stacklevel=2,
                )
            final_type = _grid_type
        else:
            final_type = detected_type

        logger.debug(f"Creating {final_type} plotter")

        # Create appropriate plotter
        plotter_class = UnstructuredPlotter if final_type.lower() == "unstructured" else GriddedPlotter
        plotter = plotter_class(self._obj, dimensions, coordinates)

        # Set grid path if available for unstructured grids
        if final_type == "unstructured" and _fpath_tgrid is not None and _fpath_ckdtree is not None:
            logger.debug("Setting grid paths for unstructured plotter")
            # Type check to ensure we have an UnstructuredPlotter before calling specify_grid
            if isinstance(plotter, UnstructuredPlotter):
                plotter.specify_grid(fpath_tgrid=_fpath_tgrid, fpath_ckdtree=_fpath_ckdtree)

        return plotter

    # Also provide methods that work with default parameters for backward compatibility
    def single_plot(self, config: PlotConfig, **kwargs):  # pragma: no cover
        """Create a single plot with default dimension detection."""
        plotter = self()
        return plotter.single_plot(config, **kwargs)

    def multi_plot(self, config: PlotConfig, **kwargs):  # pragma: no cover
        """Create multiple plots with default dimension detection."""
        plotter = self()
        return plotter.multi_plot(config, **kwargs)

    def animate(self, config: PlotConfig, **kwargs):  # pragma: no cover
        """Create animation with default dimension detection."""
        plotter = self()
        return plotter.animate(config, **kwargs)


def specify_grid(
    grid_type: Optional[str] = None,
    fpath_tgrid: Optional[Union[str, Path]] = None,
    fpath_ckdtree: Optional[Union[str, Path]] = None,
) -> None:
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

    if grid_type is not None and grid_type.lower() not in ["gridded", "unstructured"]:
        logger.error(f"Invalid grid_type: {grid_type}")
        raise ConfigurationError(
            "Invalid grid type specification",
            details=f"Provided grid_type '{grid_type}' is not supported",
            suggestions=[
                "Use 'gridded' for regular lat/lon grids",
                "Use 'unstructured' for triangular/irregular meshes",
            ],
            context={
                "provided_type": grid_type,
                "valid_types": ["gridded", "unstructured"],
            },
        )

    logger.info(f"Setting global grid specification: type={grid_type}, tgrid={fpath_tgrid}, ckdtree={fpath_ckdtree}")

    _fpath_tgrid = str(fpath_tgrid) if fpath_tgrid else None
    _fpath_ckdtree = str(fpath_ckdtree) if fpath_ckdtree else None
    _grid_type = grid_type.lower() if grid_type else None


# Register the accessor
xr.register_dataarray_accessor("plotX")(PlotXAccessor)
