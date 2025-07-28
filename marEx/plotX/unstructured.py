"""
Unstructured data visualisation module for irregular meshes.

Provides specialised plotting capabilities for irregular oceanographic data
on triangular grids (2D arrays: time, cell) with triangulation support.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray

try:
    import cartopy.crs as ccrs
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh, TriMesh
    from matplotlib.colors import BoundaryNorm, Normalize
    from matplotlib.tri import Triangulation

    HAS_PLOTTING_DEPS = True
except ImportError:
    # These will be checked in the base class
    ccrs = None
    Axes = None
    QuadMesh = None
    TriMesh = None
    BoundaryNorm = None
    Normalize = None
    Triangulation = None
    HAS_PLOTTING_DEPS = False

from ..exceptions import DataValidationError, VisualisationError
from .base import PlotterBase

# Global cache for grid data
_GRID_CACHE: Dict[str, Dict[Union[str, Tuple[str, float]], Any]] = {
    "triangulation": {},  # key: grid_path, value: triangulation object
    "ckdtree": {},  # key: (ckdtree_path, res), value: ckdtree data
}


def clear_cache() -> None:
    """Clear the global grid cache."""
    _GRID_CACHE["triangulation"].clear()
    _GRID_CACHE["ckdtree"].clear()


def _load_triangulation(fpath_tgrid: Union[str, Path]) -> Triangulation:
    """Load and cache triangulation data globally."""
    fpath_tgrid = str(fpath_tgrid)  # Convert Path to string for dict key
    if fpath_tgrid not in _GRID_CACHE["triangulation"]:
        # Only load required variables
        grid_data = xr.open_dataset(
            fpath_tgrid,
            chunks={},
            drop_variables=[v for v in xr.open_dataset(fpath_tgrid).variables if v not in ["vertex_of_cell", "clon", "clat"]],
        )

        if "vertex_of_cell" not in grid_data.variables or "clon" not in grid_data.variables or "clat" not in grid_data.variables:
            raise DataValidationError(
                "Invalid triangulation grid file format",
                details="Missing required variables for triangulation",
                suggestions=[
                    "Ensure grid file contains 'vertex_of_cell', 'clon', and 'clat' variables",
                    "Check grid file format and variable names",
                    "Verify unstructured grid file is properly formatted",
                ],
                context={
                    "required_vars": ["vertex_of_cell", "clon", "clat"],
                    "available_vars": list(grid_data.variables.keys()),
                },
            )

        # Extract triangulation vertices - convert to 0-based indexing
        triangles = grid_data.vertex_of_cell.values.T - 1
        # Create matplotlib triangulation object
        _GRID_CACHE["triangulation"][fpath_tgrid] = Triangulation(grid_data.clon.values, grid_data.clat.values, triangles)
        grid_data.close()

    return _GRID_CACHE["triangulation"][fpath_tgrid]


def _load_ckdtree(fpath_ckdtree: Union[str, Path], res: float) -> Dict[str, NDArray[Any]]:
    """Load and cache ckdtree data globally."""
    cache_key = (str(fpath_ckdtree), res)  # Convert Path to string for dict key

    if cache_key not in _GRID_CACHE["ckdtree"]:
        # Format resolution string to match file naming
        ckdtree_file = Path(fpath_ckdtree) / f"res{res:3.2f}.nc"

        if not ckdtree_file.exists():
            raise DataValidationError(
                "KDTree file not found",
                details=f"Expected file at {ckdtree_file} for resolution {res}",
                suggestions=[
                    "Check that the ckdtree path is correct",
                    "Verify the resolution value matches available files",
                    "Ensure ckdtree data files are available",
                ],
                context={"expected_file": str(ckdtree_file), "resolution": res},
            )

        ds_ckdt = xr.open_dataset(ckdtree_file)
        _GRID_CACHE["ckdtree"][cache_key] = {
            "indices": ds_ckdt.ickdtree_c.values,
            "lon": ds_ckdt.lon.values,
            "lat": ds_ckdt.lat.values,
        }
        ds_ckdt.close()

    return _GRID_CACHE["ckdtree"][cache_key]


class UnstructuredPlotter(PlotterBase):
    """Plotter for unstructured oceanographic data on triangular meshes."""

    def __init__(
        self,
        xarray_obj: xr.DataArray,
        dimensions: Optional[Dict[str, str]] = None,
        coordinates: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialise UnstructuredPlotter."""
        super().__init__(xarray_obj, dimensions, coordinates)

        from . import _fpath_ckdtree, _fpath_tgrid

        self.fpath_tgrid: Optional[Path] = _fpath_tgrid
        self.fpath_ckdtree: Optional[Path] = _fpath_ckdtree

    def specify_grid(
        self,
        fpath_tgrid: Optional[Union[str, Path]] = None,
        fpath_ckdtree: Optional[Union[str, Path]] = None,
    ) -> None:
        """Set the path to the unstructured grid files."""
        self.fpath_tgrid = Path(fpath_tgrid) if fpath_tgrid else None
        self.fpath_ckdtree = Path(fpath_ckdtree) if fpath_ckdtree else None

    def plot(
        self,
        ax: Axes,
        cmap: Union[str, Any] = "viridis",
        clim: Optional[Tuple[float, float]] = None,
        norm: Optional[Union[BoundaryNorm, Normalize]] = None,
    ) -> Tuple[Axes, Union[TriMesh, QuadMesh]]:
        """Implement plotting for unstructured data."""
        if self.fpath_ckdtree is not None:
            # Interpolate using pre-computed KDTree indices
            grid_lon, grid_lat, grid_data = self._interpolate_with_ckdtree(self.da.values, res=0.3)

            plot_kwargs = {
                "transform": ccrs.PlateCarree(),
                "cmap": cmap,
                "shading": "auto",
            }

            if norm is not None:
                plot_kwargs["norm"] = norm
            elif clim is not None:
                plot_kwargs["vmin"] = clim[0]
                plot_kwargs["vmax"] = clim[1]

            # Mask NaNs
            grid_data = np.ma.masked_invalid(grid_data)
            im = ax.pcolormesh(grid_lon, grid_lat, grid_data, **plot_kwargs)

        else:
            # Use triangulation from file if available
            if self.fpath_tgrid is None:
                raise VisualisationError(
                    "Missing grid specification for unstructured plot",
                    details="Unstructured plotting requires either triangulation or ckdtree data",
                    suggestions=[
                        "Provide fpath_tgrid for triangulation-based plotting",
                        "Provide fpath_ckdtree for interpolated regular grid plotting",
                        "Use specify_grid() to set global grid paths",
                    ],
                )

            triang = _load_triangulation(self.fpath_tgrid)

            plot_kwargs = {"transform": ccrs.PlateCarree(), "cmap": cmap}

            if norm is not None:
                plot_kwargs["norm"] = norm
            elif clim is not None:
                plot_kwargs["vmin"] = clim[0]
                plot_kwargs["vmax"] = clim[1]

            # Mask NaNs
            native_data = self.da.copy()
            native_data = np.ma.masked_invalid(native_data)

            im = ax.tripcolor(triang, native_data, **plot_kwargs)

        return ax, im

    def _interpolate_with_ckdtree(
        self, data: NDArray[Any], res: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[Any]]:
        """Interpolate unstructured data using pre-computed KDTree indices."""
        if self.fpath_ckdtree is None:
            raise VisualisationError(
                "KDTree path not specified",
                details="KDTree plotting method requires ckdtree_path parameter",
                suggestions=[
                    "Provide fpath_ckdtree parameter",
                    "Use specify_grid() to set global ckdtree path",
                    "Consider using triangulation method instead",
                ],
            )

        # Load or get cached ckdtree data
        ckdt_data = _load_ckdtree(self.fpath_ckdtree, res)

        # Create meshgrid for plotting
        grid_lon_2d, grid_lat_2d = np.meshgrid(ckdt_data["lon"], ckdt_data["lat"])

        # Use indices to create interpolated data
        grid_data = data[ckdt_data["indices"]].reshape(ckdt_data["lat"].size, ckdt_data["lon"].size)

        return grid_lon_2d, grid_lat_2d, grid_data
