"""
Input validation helpers shared across marEx.

Provides dimension, coordinate, and data-value validators used by the anomaly and
extremes packages. This module is a leaf in the dependency graph: it imports only
the package-level exception and logging utilities and never depends on a sibling
analysis package.
"""

from typing import Dict, Optional, Tuple

import dask
import numpy as np
import xarray as xr

from ..exceptions import create_data_validation_error
from ..logging_config import get_logger
from .dimensions import COORDINATE_KEYS

# Get module logger
logger = get_logger(__name__)


def _validate_dimensions_exist(da: xr.DataArray, dimensions: Dict[str, str]) -> None:
    """
    Validate that all specified dimensions exist in the dataset.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array to validate
    dimensions : dict
        Mapping of conceptual dimensions to actual dimension names

    Raises
    ------
    DataValidationError
        If any specified dimension does not exist in the dataset
    """
    missing_dims = []
    for concept_dim, actual_dim in dimensions.items():
        if actual_dim not in da.dims:
            missing_dims.append(f"'{actual_dim}' (for {concept_dim})")

    if missing_dims:
        available_dims = list(da.dims)
        raise create_data_validation_error(
            f"Missing required dimensions: {', '.join(missing_dims)}",
            details=f"Dataset has dimensions: {available_dims}",
            suggestions=[
                "Check dimension names in your data",
                "Update the 'dimensions' parameter to match your data structure",
                f"Available dimensions: {available_dims}",
            ],
            data_info={
                "missing_dimensions": missing_dims,
                "available_dimensions": available_dims,
                "provided_dimensions": dimensions,
            },
        )


def _validate_coordinates_exist(da: xr.DataArray, coordinates: Dict[str, str]) -> None:
    """
    Validate that all specified coordinates exist in the dataset.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array to validate
    coordinates : dict
        Mapping of conceptual coordinates to actual coordinate names

    Raises
    ------
    DataValidationError
        If any specified coordinate does not exist in the dataset
    """
    missing_coords = []
    for concept_coord, actual_coord in coordinates.items():
        if actual_coord not in da.coords:
            missing_coords.append(f"'{actual_coord}' (for {concept_coord})")

    if missing_coords:
        available_coords = list(da.coords.keys())
        raise create_data_validation_error(
            f"Missing required coordinates: {', '.join(missing_coords)}",
            details=f"Dataset has coordinates: {available_coords}",
            suggestions=[
                "Check coordinate names in your data",
                "Update the 'coordinates' parameter to match your data structure",
                f"Available coordinates: {available_coords}",
            ],
            data_info={
                "missing_coordinates": missing_coords,
                "available_coordinates": available_coords,
                "provided_coordinates": coordinates,
            },
        )


def _infer_dims_coords(
    da: xr.DataArray, dimensions: Optional[Dict[str, str]], coordinates: Optional[Dict[str, str]]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Determine full set of dimensions and coordinates for the DataArray.
    Sets default (standard) dimension and coordinate names if unspecified.

    This function ensures the dimensions dictionary includes required keys and coordinates
    are properly set based on data structure. It validates that all specified dimensions
    and coordinates exist in the dataset.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array to infer dimensions and coordinates for
    dimensions : dict
        Mapping of conceptual dimensions to actual dimension names
    coordinates : dict, optional
        Mapping of conceptual coordinates to actual coordinate names

    Returns
    -------
    tuple
        Tuple of (dimensions, coordinates) dictionaries with defaults applied

    Raises
    ------
    DataValidationError
        If any specified dimension or coordinate does not exist in the dataset
    """
    if dimensions is None:
        dimensions = {"time": "time", "x": "lon", "y": "lat"}

    if "time" not in dimensions:
        dimensions = {"time": "time", **dimensions}  # Permit partial default dimensions --> "time"

    # A partial ``dimensions`` mapping must still name the horizontal axis, unless the
    # field is a bare 1-D time series and has none to name. Without this check a mapping
    # that omits 'x' -- a typo, or {"y": "lat"} -- is not an error at all: every unnamed
    # dimension is picked up as an *extra* dim by ``core.dimensions.extra_dims`` and
    # carried through as a broadcast axis, so the run completes with the horizontal axis
    # silently demoted to a depth-like one. Checked before the coordinates branch below,
    # which already assumes dimensions['x'] when it is present.
    non_time_dims = [str(d) for d in da.dims if str(d) != dimensions["time"]]
    if "x" not in dimensions and non_time_dims:
        logger.error("Dimensions mapping does not name the horizontal 'x' dimension")
        raise create_data_validation_error(
            "Dimensions mapping is missing the required 'x' entry",
            details=(
                "'x' names the horizontal dimension (the cell dimension of an unstructured "
                f"mesh, or the longitude axis of a grid). The data carries {non_time_dims} "
                "beyond time, so one of them must be named as 'x'; the rest are detected "
                "automatically and carried through as extra broadcast axes."
            ),
            suggestions=[
                f"Add an 'x' entry, e.g. dimensions={{'time': '{dimensions['time']}', 'x': '{non_time_dims[-1]}'}}",
                "For gridded data name both, e.g. dimensions={'time': 'time', 'x': 'lon', 'y': 'lat'}",
                "Omit the dimensions parameter entirely to use the defaults",
            ],
            data_info={"dimensions": dimensions, "unnamed_dimensions": non_time_dims},
        )

    # Handle coordinates parameter based on data structure.
    #
    # Gridded iff BOTH x and y are named. Testing "x present, y absent" instead would be
    # the same condition today but says the wrong thing: it reads as "malformed gridded
    # config", when the layout it actually describes -- a single horizontal dimension --
    # is the legitimate unstructured one, with or without extra dims such as depth.
    is_gridded = "x" in dimensions and "y" in dimensions
    if coordinates is None:
        if "x" in dimensions and not is_gridded:
            # Unstructured data (one horizontal dimension) - requires explicit
            # coordinate specification. Only reachable when 'x' is present, so the
            # message below can safely reference dimensions['x'].
            logger.error("Coordinates parameter required for unstructured data")
            raise create_data_validation_error(
                "Coordinates parameter must be explicitly specified for unstructured data",
                details="Unstructured data requires coordinate names for x and y spatial coordinates",
                suggestions=[
                    "Specify coordinates parameter with spatial coordinate names",
                    "Example: coordinates={'time': 'time', 'x': 'lon', 'y': 'lat'}",
                    f"Your x dimension '{dimensions['x']}' needs associated coordinate names",
                    "If data is gridded, ensure 'y' dimension is also specified",
                    "Extra dimensions such as depth or level need no entry here - they are "
                    "detected automatically and carried through as broadcast axes",
                ],
                data_info={
                    "data_structure": "unstructured (2D)",
                    "dimensions": dimensions,
                    "missing_coordinates": "x and y spatial coordinates",
                },
            )
        else:
            # Gridded (has both x and y) or 1D time series (no x and no y): copy the
            # dimension names to coordinates. This keeps the 1D-harmonic path reachable
            # with defaults instead of raising a bare KeyError on dimensions['x'].
            #
            # Only the coordinate-bearing keys are copied. An optional 'z' entry names a
            # dimension, not a coordinate, and copying it would demand a coordinate the
            # data need not carry.
            coordinates = {key: value for key, value in dimensions.items() if key in COORDINATE_KEYS}
            logger.debug("Copying dimensions to coordinates (gridded or 1D time series)")
    else:
        # Coordinates provided but ensure time coordinate is included if missing
        if "time" not in coordinates:
            coordinates = {"time": dimensions.get("time", "time"), **coordinates}
            logger.debug("Added default time coordinate to provided coordinates")

    # Validate dimensions and coordinates exist in dataset
    logger.debug("Validating dimensions and coordinates")
    _validate_dimensions_exist(da, dimensions)
    _validate_coordinates_exist(da, coordinates)

    return dimensions, coordinates


def _validate_data_values(da: xr.DataArray, dimensions: Dict[str, str]) -> None:
    """
    Validate that all unmasked data contains only finite values (no NaN or inf).

    Parameters
    ----------
    da : xarray.DataArray
        Input data array to validate
    dimensions : dict
        Mapping of conceptual dimensions to actual dimension names

    Raises
    ------
    DataValidationError
        If any unmasked data contains NaN or infinite values
    """
    # Create spatial mask from first time step (2D array)
    spatial_mask = np.isfinite(da.isel({dimensions["time"]: 0}))

    # Reduce first, then mask (avoids broadcasting across time)
    # Count invalid values at each spatial location across time dimension
    # This produces a 2D spatial array instead of a 3D array
    finite_mask = np.isfinite(da)
    invalid_per_location = (~finite_mask).sum(dim=dimensions["time"])

    # Now apply spatial mask to this 2D result (no broadcasting across time!)
    invalid_in_valid_locations = invalid_per_location.where(spatial_mask, 0)

    # One round-trip for both scans rather than two: they read the same input, so fusing
    # them lets dask share that read instead of walking the array twice (finding 2.12).
    has_valid_data, max_invalid = dask.compute(spatial_mask.any(), invalid_in_valid_locations.max())

    # Check if there's any valid data at all
    if not has_valid_data:
        raise create_data_validation_error(
            "Dataset contains no valid (finite) data",
            details="All values in the first time step are NaN or infinite",
            suggestions=[
                "Check your input data for data quality issues",
                "Verify the data was loaded correctly",
                "Check for issues in data preprocessing steps",
            ],
            data_info={
                "total_values": int(da.size),
                "total_spatial_locations": int(np.prod([da.sizes[d] for d in da.dims if d != dimensions["time"]])),
            },
        )

    if max_invalid > 0:
        # Error path: three more reductions over the same arrays, batched into one
        # round-trip rather than three sequential full re-scans.
        total_invalid_in_valid_region, total_valid_locations, locations_affected = (
            int(v)
            for v in dask.compute(
                invalid_in_valid_locations.sum(),
                spatial_mask.sum(),
                (invalid_in_valid_locations > 0).sum(),
            )
        )
        total_time_steps = int(da.sizes[dimensions["time"]])

        raise create_data_validation_error(
            f"Dataset contains {total_invalid_in_valid_region} invalid values at {locations_affected} locations",
            details=(
                f"Found invalid data across time series. Worst location has {int(max_invalid)} "
                f"invalid time steps out of {total_time_steps}."
            ),
            suggestions=[
                "Remove or interpolate NaN/infinite values before preprocessing",
                "Check data quality and loading procedures",
                "Consider using data.fillna() or data.interpolate_na() methods",
                "Verify coordinate/dimension alignment in your dataset",
                "If your field carries a land/sea or missing-data mask, ensure it is "
                "applied consistently across every time step",
            ],
            data_info={
                "total_invalid_values": total_invalid_in_valid_region,
                "locations_affected": locations_affected,
                "total_valid_locations": total_valid_locations,
                "max_invalid_at_one_location": int(max_invalid),
                "total_time_steps": total_time_steps,
                "percentage_affected": f"{100.0 * locations_affected / total_valid_locations:.2f}%",
            },
        )
