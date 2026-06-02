"""Dimension and coordinate validation helpers for the plotX visualisation system."""

from typing import Dict

import xarray as xr

from ..exceptions import VisualisationError


def _validate_dimensions_exist(da: xr.DataArray, dimensions: Dict[str, str]) -> None:
    """Validate that required dimensions exist in the dataset. Time dimension is optional."""
    missing_dims = []
    for concept_dim, actual_dim in dimensions.items():
        # Time dimension is optional for plotting - only spatial dimensions are required
        if concept_dim == "time" and actual_dim not in da.dims:
            continue
        if actual_dim not in da.dims:
            missing_dims.append(f"'{actual_dim}' (for {concept_dim})")

    if missing_dims:
        available_dims = list(da.dims)
        raise VisualisationError(
            f"Missing required dimensions: {', '.join(missing_dims)}",
            details=f"Dataset has dimensions: {available_dims}",
            suggestions=[
                "Check dimension names in your data",
                "Update the 'dimensions' parameter to match your data structure",
                f"Available dimensions: {available_dims}",
            ],
            context={
                "missing_dimensions": missing_dims,
                "available_dimensions": available_dims,
                "provided_dimensions": dimensions,
            },
        )


def _validate_coordinates_exist(da: xr.DataArray, coordinates: Dict[str, str]) -> None:
    """Validate that required coordinates exist in the dataset. Time coordinate is optional."""
    missing_coords = []
    for concept_coord, actual_coord in coordinates.items():
        # Time coordinate is optional for plotting - only spatial coordinates are required
        if concept_coord == "time" and actual_coord not in da.coords:
            continue
        if actual_coord not in da.coords:
            missing_coords.append(f"'{actual_coord}' (for {concept_coord})")

    if missing_coords:
        available_coords = list(da.coords)
        raise VisualisationError(
            f"Missing required coordinates: {', '.join(missing_coords)}",
            details=f"Dataset has coordinates: {available_coords}",
            suggestions=[
                "Check coordinate names in your data",
                "Update the 'coordinates' parameter to match your data structure",
                f"Available coordinates: {available_coords}",
            ],
            context={
                "missing_coordinates": missing_coords,
                "available_coordinates": available_coords,
                "provided_coordinates": coordinates,
            },
        )
