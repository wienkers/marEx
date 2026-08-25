"""
Infrastructure shared across the marEx analysis packages.

``core`` holds everything that ``anomaly``, ``extremes`` and ``track`` all need
and none of them owns: the materialisation policy behind ``compute_mode``, input
validation, the dimension and time-axis contracts, and output serialisation
helpers. Nothing in this package imports from an analysis package, which is what
keeps the dependency graph acyclic.
"""

from .attrs import make_netcdf_safe_attrs
from .compute_mode import ComputeMode, Materialiser, clear_staging, create_staging_dir
from .dimensions import DimSpec, extra_dims, horizontal_dims, resolve_dims, spatial_chunks, spatial_dims, tile_spatial_chunks
from .time_axis import DAILY_CYCLE, SeasonalCycle, add_decimal_year, infer_cycle, resolve_cycle
from .validation import _infer_dims_coords, _validate_coordinates_exist, _validate_data_values, _validate_dimensions_exist

__all__ = [
    "ComputeMode",
    "Materialiser",
    "DimSpec",
    "resolve_dims",
    "horizontal_dims",
    "extra_dims",
    "spatial_dims",
    "spatial_chunks",
    "tile_spatial_chunks",
    "clear_staging",
    "create_staging_dir",
    "add_decimal_year",
    "SeasonalCycle",
    "infer_cycle",
    "resolve_cycle",
    "DAILY_CYCLE",
    "make_netcdf_safe_attrs",
    "_infer_dims_coords",
    "_validate_dimensions_exist",
    "_validate_coordinates_exist",
    "_validate_data_values",
]
