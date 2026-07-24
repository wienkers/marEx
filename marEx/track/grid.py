"""
MarEx Track: Coordinate unification and unstructured-grid setup.

Stateless helpers for coordinate-system handling and unstructured-grid
preparation, extracted from the tracker orchestrator. Functions that mutated
tracker state (auto-detecting ``coordinate_units``, converting coordinates to
degrees, building the sparse dilation matrix, dropping coordinate variables)
return the values they produced so the orchestrator can reassign them.
Behaviour and numerics are identical to the original ``tracker`` methods.
"""

from typing import Optional, Tuple

import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix, csr_matrix, eye

from .._dependencies import warn_missing_dependency
from ..exceptions import ConfigurationError, create_coordinate_error, create_data_validation_error
from ..logging_config import get_logger
from .validation import validate_unstructured_chunking

logger = get_logger(__name__)

try:
    import jax.numpy as jnp
except ImportError:
    jnp = np  # type: ignore[misc]  # Alias for jnp when JAX not available
    warn_missing_dependency("jax", "Some functionality")


def unify_coordinates(
    data_bin: xr.DataArray,
    regional_mode: bool,
    coordinate_units: Optional[str],
    xcoord: str,
    ycoord: str,
) -> Tuple[Optional[str], xr.DataArray]:
    """Unify the coordinate system to degrees.

    Returns ``(coordinate_units, data_bin)`` with units auto-detected (if not
    provided for global data) and radian coordinates converted to degrees.
    """
    if regional_mode:
        if coordinate_units is None:
            raise create_coordinate_error(
                "coordinate_units must be specified when regional_mode=True",
                suggestions=[
                    "Set coordinate_units='degrees' for degree-based coordinates",
                    "Set coordinate_units='radians' for radian-based coordinates",
                ],
            )
        if coordinate_units not in ["degrees", "radians"]:
            raise create_coordinate_error(
                f"Invalid coordinate_units '{coordinate_units}'",
                details="coordinate_units must be either 'degrees' or 'radians'",
                suggestions=["Use coordinate_units='degrees' or coordinate_units='radians'"],
            )
    else:
        # Check if coordinate_units is explicitly specified
        if coordinate_units is not None:
            if coordinate_units not in ["degrees", "radians"]:
                raise create_coordinate_error(
                    f"Invalid coordinate_units '{coordinate_units}'",
                    details="coordinate_units must be either 'degrees' or 'radians'",
                    suggestions=["Use coordinate_units='degrees' or coordinate_units='radians'"],
                )
            # Use explicitly specified coordinate units
        else:
            # Auto-detect coordinate units for global data
            lon = data_bin[xcoord]
            lon_range = float(lon.max()) - float(lon.min())
            # Global grids come in two conventions: endpoint-exclusive (e.g. 0 to 360-dlon,
            # so the range is one grid step short of a full turn) and endpoint-inclusive
            # (the range is exactly a full turn). Accept either, with a half-step tolerance.
            #
            # Matching only "close to a full turn" with a +-1.5*dlon window is unbounded: on a
            # short-range grid with few points dlon grows large enough to swallow unrelated
            # ranges, and a 10 degree grid with 5 points was mis-detected as radians because
            # |10 - 2*pi| = 3.72 fell inside a 3.75 tolerance. Matching only the
            # endpoint-exclusive span is the opposite error: it rejects genuine full-turn
            # grids whose range is exactly 360 (or 2*pi).
            dlon = lon_range / max(int(lon.size) - 1, 1)

            # Check for degrees (range close to 360, or to 360 - dlon)
            if min(abs(lon_range - 360.0), abs(lon_range - (360.0 - dlon))) <= max(1.0, 0.5 * dlon):
                coordinate_units = "degrees"

            # Check for radians (range close to 2π, or to 2π - dlon)
            elif min(abs(lon_range - 2 * np.pi), abs(lon_range - (2 * np.pi - dlon))) <= max(0.02, 0.5 * dlon):
                coordinate_units = "radians"

            # If neither, throw error
            else:
                raise create_coordinate_error(
                    f"Cannot auto-detect coordinate units from range {lon_range:.3f}",
                    details=(f"Expected ranges: ~360 degrees or ~{2*np.pi:.3f} radians. " f"Found range: {lon_range:.3f}"),
                    suggestions=[
                        "Use regional_mode=True with coordinate_units specified for regional data",
                        "Specify coordinate_units='degrees' or coordinate_units='radians' explicitly",
                        "Check that your coordinate values are correct",
                        "Verify x-dimension coordinate ranges",
                    ],
                    context={"detected_range": lon_range, "xdim": xcoord},
                )

    # Convert lat & lon to degrees. Use assign_coords (returns a new object) rather than
    # writing the coordinates in place, which mutated the caller's DataArray and would
    # double-convert on a second pass.
    if coordinate_units == "radians":
        data_bin = data_bin.assign_coords(
            {
                xcoord: data_bin[xcoord] * 180.0 / np.pi,
                ycoord: data_bin[ycoord] * 180.0 / np.pi,
            }
        )

    return coordinate_units, data_bin


def remap_coordinates(
    events_ds: xr.Dataset,
    lat_init: xr.DataArray,
    lon_init: xr.DataArray,
    coordinate_units: Optional[str],
    xcoord: str,
    ycoord: str,
) -> xr.Dataset:
    """Remap coordinates to original lat/lon values after processing.
    Map centroids from lat=[-180,180] back into original lat/lon units & range.
    """
    # Re-assign original coordinates from original marEx input
    events_ds = events_ds.assign_coords({ycoord: lat_init.compute(), xcoord: lon_init.compute()})

    if "centroid" in events_ds.data_vars:
        # Remap centroids to original coordinate system
        # (lat, lon) currently in degrees [-90,90], [-180,180]
        centroids = events_ds["centroid"].persist()

        # Split into components
        centroids_lat = centroids.isel(component=0)  # [-90, 90] degrees
        centroids_lon = centroids.isel(component=1)  # [-180, 180] degrees

        # Get original coordinate bounds
        lon_min = float(lon_init.min().compute().item())
        lon_max = float(lon_init.max().compute().item())

        # Convert units and adjust ranges
        if coordinate_units == "radians":
            # Convert from degrees to radians
            centroids_lat = centroids_lat * np.pi / 180.0  # Now in [-π/2, π/2]
            centroids_lon = centroids_lon * np.pi / 180.0  # Now in [-π, π]

            # Check if original longitude was in [0, 2π] range
            if lon_min >= 0 and lon_max > np.pi:
                # Shift from [-π, π] to [0, 2π]
                centroids_lon = xr.where(centroids_lon < 0, centroids_lon + 2 * np.pi, centroids_lon)
        else:
            # Coordinates remain in degrees
            # Check if original longitude was in [0, 360] range
            if lon_min >= 0 and lon_max > 180:
                # Shift from [-180, 180] to [0, 360]
                centroids_lon = xr.where(centroids_lon < 0, centroids_lon + 360, centroids_lon)

        # Reassemble centroids with remapped coordinates
        centroids_remapped = xr.concat([centroids_lat, centroids_lon], dim="component")

        # Update the dataset
        events_ds["centroid"] = centroids_remapped

    return events_ds


def setup_unstructured_grid(
    temp_dir: str,
    neighbours: xr.DataArray,
    cell_areas: xr.DataArray,
    max_iteration: int,
    data_bin: xr.DataArray,
    mask: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    xdim: str,
    xcoord: str,
    ycoord: str,
) -> Tuple[str, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, int, xr.DataArray, csr_matrix]:
    """Set up special handling for unstructured grids.

    Returns ``(scratch_dir, data_bin, mask, lat, lon, max_iteration,
    neighbours_int, dilate_sparse)``.
    """
    if not temp_dir:
        raise ConfigurationError(
            "Missing temporary directory for unstructured processing",
            details="Unstructured grids require temporary storage for memory efficiency",
            suggestions=[
                "Provide temp_dir parameter: tracker(..., temp_dir='/tmp/marex')",
                "Ensure directory has sufficient space and write permissions",
            ],
        )

    scratch_dir = temp_dir

    # Remove coordinate variables to avoid memory issues
    data_bin = data_bin.drop_vars({ycoord, xcoord})
    mask = mask.drop_vars({ycoord, xcoord})
    lat = lat.drop_vars(lat.coords)
    lon = lon.drop_vars(lon.coords)
    neighbours = neighbours.drop_vars({ycoord, xcoord, "nv"}.intersection(set(neighbours.coords)))

    # Validate spatial chunking for unstructured grid data
    validate_unstructured_chunking(neighbours, cell_areas, xdim)

    # Initialise dilation array for unstructured grid
    neighbours_int = neighbours.astype(np.int32) - 1  # Convert to 0-based indexing

    # Validate neighbour array structure
    if neighbours_int.shape[0] != 3:
        raise create_data_validation_error(
            "Invalid neighbour array for triangular grid",
            details=f"Expected shape (3, ncells), got {neighbours_int.shape}",
            suggestions=[
                "Ensure triangular grid connectivity",
                "Check neighbour array from grid file",
                "Verify unstructured grid format",
            ],
            data_info={
                "actual_shape": neighbours_int.shape,
                "expected_shape": "(3, ncells)",
            },
        )
    if neighbours_int.dims != ("nv", xdim):
        raise create_data_validation_error(
            "Invalid neighbour array dimensions",
            details=f"Expected dimensions ('nv', '{xdim}'), got {neighbours_int.dims}",
            suggestions=[
                "Check dimension names in grid file",
                "Verify coordinate mapping",
            ],
            data_info={
                "actual_dims": neighbours_int.dims,
                "expected_dims": ("nv", xdim),
            },
        )

    # Construct sparse dilation matrix
    dilate_sparse = build_sparse_dilation_matrix(neighbours_int)

    return scratch_dir, data_bin, mask, lat, lon, max_iteration, neighbours_int, dilate_sparse


def build_sparse_dilation_matrix(neighbours_int: xr.DataArray) -> csr_matrix:
    """Build sparse matrix for efficient dilation operations on unstructured grid."""
    # Create row and column indices for sparse matrix
    row_indices = jnp.repeat(jnp.arange(neighbours_int.shape[1]), 3)
    col_indices = neighbours_int.data.compute().T.flatten()

    # Filter out negative values (invalid connections)
    valid_mask = col_indices >= 0
    row_indices = row_indices[valid_mask]
    col_indices = col_indices[valid_mask]

    # Create the sparse matrix for dilation
    ncells = neighbours_int.shape[1]
    dilate_coo = coo_matrix(
        (jnp.ones_like(row_indices, dtype=bool), (row_indices, col_indices)),
        shape=(ncells, ncells),
    )
    dilate_sparse = csr_matrix(dilate_coo)

    # Add identity matrix to include self-connections
    identity = eye(neighbours_int.shape[1], dtype=bool, format="csr")
    dilate_sparse = dilate_sparse + identity

    logger.info("Finished constructing the sparse dilation matrix")

    return dilate_sparse
