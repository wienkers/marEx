"""
MarEx Track: Input validation and chunking checks.

Stateless validation helpers extracted from the tracker orchestrator. Each
function takes the data/config values it needs as explicit arguments and, where
the original method mutated tracker state (transposing ``data_bin``, rechunking
coordinates, resolving area-filter parameters), returns the possibly-mutated
values so the orchestrator can reassign them. Behaviour and numerics are
identical to the original ``tracker`` methods.
"""

import warnings
from typing import Optional, Tuple

import xarray as xr
from dask.base import is_dask_collection

from ..exceptions import ConfigurationError, TrackingError, create_data_validation_error
from ..logging_config import get_logger

logger = get_logger(__name__)


def validate_required_coordinates(data_bin: xr.DataArray, timecoord: str, xcoord: str, ycoord: str) -> None:
    """Raise a descriptive error if any required coordinate is absent from ``data_bin``.

    Called before the tracker touches ``data_bin[ycoord]`` so that a missing coordinate
    surfaces as this error rather than a bare ``KeyError`` (§4.4).
    """
    if timecoord not in data_bin.coords or xcoord not in data_bin.coords or ycoord not in data_bin.coords:
        raise create_data_validation_error(
            "Missing required coordinates in unstructured data",
            details=(f"Expected coordinates ({timecoord}, {xcoord}, {ycoord}), " f"but found {list(data_bin.coords)}"),
            suggestions=[
                "Ensure data_bin contains time, x, and y coordinates",
                "Check coordinate names in the dataset",
                "Specify coordinates in the tracker initialisation with `coordinates` parameter.",
            ],
            data_info={
                "actual_coords": list(data_bin.coords),
                "expected_coords": [timecoord, xcoord, ycoord],
            },
        )


def validate_inputs(
    data_bin: xr.DataArray,
    mask: xr.DataArray,
    regional_mode: bool,
    unstructured_grid: bool,
    timedim: str,
    xdim: str,
    ydim: Optional[str],
    timecoord: str,
    xcoord: str,
    ycoord: str,
    use_absolute_filtering: bool,
    area_filter_quartile: float,
    area_filter_absolute: int,
    T_fill: int,
    lat: xr.DataArray,
    lon: xr.DataArray,
    neighbours: Optional[xr.DataArray] = None,
    cell_areas: Optional[xr.DataArray] = None,
    grid_resolution: Optional[float] = None,
    temp_dir: Optional[str] = None,
) -> Tuple[xr.DataArray, Optional[str], xr.DataArray, xr.DataArray, xr.DataArray]:
    """Validate input parameters and data.

    Returns the (possibly mutated) ``data_bin``, ``ydim`` (set to ``None`` for
    unstructured grids), and the (possibly rechunked) ``mask``, ``lat`` and
    ``lon`` so the orchestrator can reassign them.
    """
    if regional_mode and unstructured_grid:
        raise NotImplementedError("regional_mode is not yet implemented for unstructured grids")

    # Rank guard, before the transpose below. Tracking is a single-horizontal-level
    # operation: connected-component labelling, the dilation matrix and the merge
    # ledger are all defined on one 2-D field per timestep. A field carrying an extra
    # dimension (depth, level, member) is a stack of independent tracking problems,
    # not one -- and the transpose below would otherwise reject it with a confusing
    # "expected 3D array" message that names the wrong problem.
    expected = (timedim, xdim) if unstructured_grid else (timedim, ydim, xdim)
    unexpected = [str(d) for d in data_bin.dims if d not in expected]
    if unexpected:
        raise TrackingError(
            f"Tracking does not support the extra dimension(s) {unexpected}",
            details=(
                f"Tracking operates on a single horizontal level. Expected dimensions "
                f"{[d for d in expected if d is not None]}, got {list(data_bin.dims)}."
            ),
            suggestions=[
                f"Select one level first, e.g. ds.isel({unexpected[0]}=0)",
                "Loop over the extra dimension and track each level separately",
                "Check the dimension mapping passed to the tracker",
            ],
            context={
                "unexpected_dimensions": unexpected,
                "actual_dims": [str(d) for d in data_bin.dims],
                "expected_dims": [d for d in expected if d is not None],
            },
        )

    # For unstructured grids, adjust dimensions
    if unstructured_grid:
        ydim = None
        if (timedim, xdim) != data_bin.dims:
            try:
                data_bin = data_bin.transpose(timedim, xdim)
            except Exception:
                raise create_data_validation_error(
                    "Invalid dimensions for unstructured data",
                    details=f"Expected 2D array with dimensions ({timedim}, {xdim}), got {list(data_bin.dims)}",
                    suggestions=[
                        "Ensure data has time and cell dimensions only",
                        "Check dimension mapping in function call",
                    ],
                    data_info={
                        "actual_dims": list(data_bin.dims),
                        "expected_dims": [timedim, xdim],
                    },
                )
    else:
        # For structured grids, ensure 3D data
        if (timedim, ydim, xdim) != data_bin.dims:
            try:
                data_bin = data_bin.transpose(timedim, ydim, xdim)
            except Exception:
                raise create_data_validation_error(
                    "Invalid dimensions for gridded data",
                    details=(f"Expected 3D array with dimensions ({timedim}, {ydim}, {xdim}), " f"got {list(data_bin.dims)}"),
                    suggestions=[
                        "Ensure data has time, latitude, and longitude dimensions",
                        "Check dimension mapping and coordinate names",
                    ],
                    data_info={
                        "actual_dims": list(data_bin.dims),
                        "expected_dims": [timedim, ydim, xdim],
                    },
                )

    # Check if timecoord, xcoord, and ycoord are in data_bin coords:
    validate_required_coordinates(data_bin, timecoord, xcoord, ycoord)

    # Check if timecoord is an index of timedim
    if timecoord != timedim and (timedim not in data_bin.indexes or data_bin.indexes[timedim].name != timecoord):
        logger.warning(
            f"timecoord '{timecoord}' is not an index of timedim '{timedim}'. "
            f"Setting '{timecoord}' as index for dimension '{timedim}'"
        )
        data_bin = data_bin.set_index({timedim: timecoord})

    # Check data type and structure
    if data_bin.data.dtype != bool:
        raise create_data_validation_error(
            "Input DataArray must be binary (boolean type)",
            details=f"Found dtype {data_bin.data.dtype}, expected bool",
            suggestions=[
                "Convert data using da > threshold for binary events",
                "Use xr.where(condition, True, False) for boolean conversion",
            ],
            data_info={
                "actual_dtype": str(data_bin.data.dtype),
                "expected_dtype": "bool",
            },
        )

    # Validate required parameters for unstructured grids
    if unstructured_grid:
        if temp_dir is None:
            raise create_data_validation_error(
                "temp_dir is required for unstructured grids",
                details="Unstructured grid processing requires a temporary directory",
                suggestions=["Provide a temp_dir parameter when using unstructured_grid=True"],
            )
        if neighbours is None:
            raise create_data_validation_error(
                "neighbours array is required for unstructured grids",
                details="Unstructured grid processing requires cell connectivity information",
                suggestions=["Provide a neighbours parameter when using unstructured_grid=True"],
            )
        if cell_areas is None:
            raise create_data_validation_error(
                "cell_areas array is required for unstructured grids",
                details="Unstructured grid processing requires cell area information",
                suggestions=["Provide a cell_areas parameter when using unstructured_grid=True"],
            )
    else:
        # For structured grids, cell_areas is optional
        if cell_areas is not None:
            # Validate dimensions if provided
            expected_spatial_dims = {ydim, xdim}
            if set(cell_areas.dims) != expected_spatial_dims:
                raise create_data_validation_error(
                    "Invalid cell_areas dimensions for structured grid",
                    details=f"Expected spatial dimensions {expected_spatial_dims}, got {set(cell_areas.dims)}",
                    suggestions=["Ensure cell_areas matches the spatial dimensions of your data"],
                )

    # Validate grid_resolution parameter
    if grid_resolution is not None:
        if unstructured_grid:
            raise create_data_validation_error(
                "grid_resolution parameter is not supported for unstructured grids",
                details="Grid resolution calculation requires structured (lat/lon) coordinates",
                suggestions=["Use cell_areas parameter directly for unstructured grids"],
            )
        if not isinstance(grid_resolution, (int, float)) or grid_resolution <= 0:
            raise create_data_validation_error(
                "grid_resolution must be a positive number",
                details=f"Received grid_resolution={grid_resolution}",
                suggestions=["Provide a positive float value representing grid resolution in degrees"],
            )

    if not is_dask_collection(data_bin.data):
        raise create_data_validation_error(
            "Input DataArray must be Dask-backed",
            details="Tracking requires chunked data for efficient processing",
            suggestions=[
                "Convert to Dask: data_bin = data_bin.chunk({'time': 10})",
                "Load with chunking: xr.open_dataset('file.nc', chunks={})",
            ],
            data_info={"data_type": type(data_bin.data).__name__},
        )

    if mask.data.dtype != bool:
        raise create_data_validation_error(
            "Mask must be binary (boolean type)",
            details=f"Found mask dtype {mask.data.dtype}, expected bool",
            suggestions=["Convert mask using mask > 0 or mask.astype(bool)"],
            data_info={"mask_dtype": str(mask.data.dtype)},
        )

    if not mask.any().compute().item():
        raise create_data_validation_error(
            "Mask contains only False values",
            details="Mask should indicate valid regions with True values",
            suggestions=[
                "Check mask orientation - it should mark valid (ocean) regions as True",
                "Invert mask if needed: mask = ~mask",
                "Create ocean mask from land mask",
            ],
        )

    # Check chunking for spatial dimensions
    data_bin, mask, lat, lon = validate_spatial_chunking(data_bin, mask, lat, lon, xdim, ydim)

    # Validate resolved area filtering parameters
    if not use_absolute_filtering:
        # Quartile-based filtering validation
        if (area_filter_quartile < 0) or (area_filter_quartile > 1):
            raise ConfigurationError(
                "Invalid area_filter_quartile value",
                details=f"Value {area_filter_quartile} is outside valid range [0, 1]",
                suggestions=[
                    "Use values between 0.0 and 1.0",
                    "Use 0.25 to filter smallest 25% of events",
                    "Use 0.5 to keep only larger events",
                ],
                context={
                    "provided_value": area_filter_quartile,
                    "valid_range": [0, 1],
                },
            )
    else:
        # Absolute filtering validation
        if area_filter_absolute <= 0:
            raise ConfigurationError(
                "Invalid area_filter_absolute value",
                details=f"area_filter_absolute={area_filter_absolute} must be positive",
                suggestions=[
                    "Set area_filter_absolute to a positive integer (e.g., 5, 10, 50)",
                ],
                context={
                    "area_filter_absolute": area_filter_absolute,
                },
            )

    if T_fill % 2 != 0:
        raise ConfigurationError(
            "T_fill must be even for temporal symmetry",
            details=f"Provided T_fill={T_fill} is odd",
            suggestions=["Use even values: 2, 4, 6, 8, etc."],
            context={"provided_value": T_fill, "requirement": "even number"},
        )

    return data_bin, ydim, mask, lat, lon


def resolve_area_filtering_parameters(
    area_filter_quartile: Optional[float], area_filter_absolute: Optional[int]
) -> Tuple[float, int, bool]:
    """Resolve area filtering parameters.

    Returns ``(area_filter_quartile, area_filter_absolute, use_absolute_filtering)``.
    """
    # Count non-None parameters
    provided_params = sum(x is not None for x in [area_filter_quartile, area_filter_absolute])

    if provided_params == 0:
        # Default case: use quartile-based filtering
        return 0.5, 0, False
    elif provided_params == 1:
        # Single parameter provided - use it
        if area_filter_quartile is not None:
            return area_filter_quartile, 0, False
        else:  # area_filter_absolute is not None
            return 0.0, area_filter_absolute, True  # Set quartile=0.0 for compatibility
    else:
        # Both provided - error
        raise ConfigurationError(
            "Cannot specify both area filtering parameters",
            details="area_filter_quartile and area_filter_absolute are mutually exclusive",
            suggestions=[
                "Use area_filter_quartile for percentile-based filtering (e.g., 0.25 for smallest 25%)",
                "Use area_filter_absolute for fixed minimum area (e.g., 10 for minimum 10 cells)",
                "Omit both parameters to use default quartile filtering (0.5)",
            ],
            context={
                "area_filter_quartile": area_filter_quartile,
                "area_filter_absolute": area_filter_absolute,
            },
        )


def validate_spatial_chunking(
    data_bin: xr.DataArray,
    mask: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    xdim: str,
    ydim: Optional[str],
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Validate that spatial dimensions are in single chunks for apply_ufunc operations.

    Returns the (possibly rechunked) ``data_bin``, ``mask``, ``lat`` and ``lon``.
    """
    rechunk_needed = False
    rechunk_dims = {}

    # Check xdim chunking in data_bin
    if xdim in data_bin.chunksizes:
        xdim_chunks = data_bin.chunksizes[xdim]
        if len(xdim_chunks) > 1:
            warnings.warn(
                f"Spatial dimension '{xdim}' has multiple chunks ({len(xdim_chunks)} chunks). "
                f"This will cause issues with apply_ufunc operations. Rechunking to single chunk."
                f"Consider directly loading dataset with proper chunking to optimise performance.",
                UserWarning,
                stacklevel=3,
            )
            rechunk_needed = True
            rechunk_dims[xdim] = -1

    # Check ydim chunking for structured grids
    if ydim is not None and ydim in data_bin.chunksizes:
        ydim_chunks = data_bin.chunksizes[ydim]
        if len(ydim_chunks) > 1:
            warnings.warn(
                f"Spatial dimension '{ydim}' has multiple chunks ({len(ydim_chunks)} chunks). "
                f"This will cause issues with apply_ufunc operations. Rechunking to single chunk."
                f"Consider directly loading dataset with proper chunking to optimise performance.",
                UserWarning,
                stacklevel=3,
            )
            rechunk_needed = True
            rechunk_dims[ydim] = -1

    # Rechunk data_bin if needed
    if rechunk_needed:
        logger.info(f"Rechunking spatial dimensions: {rechunk_dims}")
        data_bin = data_bin.chunk(rechunk_dims)

    # Check mask spatial dimensions for single chunks
    mask_rechunk_needed = False
    mask_rechunk_dims = {}

    # Check xdim chunking in mask
    if mask.chunks is not None and xdim in mask.chunksizes:
        xdim_chunks = mask.chunksizes[xdim]
        if len(xdim_chunks) > 1:
            warnings.warn(
                f"Mask spatial dimension '{xdim}' has multiple chunks ({len(xdim_chunks)} chunks). "
                f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                UserWarning,
                stacklevel=3,
            )
            mask_rechunk_needed = True
            mask_rechunk_dims[xdim] = -1

    # Check ydim chunking in mask for structured grids
    if ydim is not None and mask.chunks is not None and ydim in mask.chunksizes:
        ydim_chunks = mask.chunksizes[ydim]
        if len(ydim_chunks) > 1:
            warnings.warn(
                f"Mask spatial dimension '{ydim}' has multiple chunks ({len(ydim_chunks)} chunks). "
                f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                UserWarning,
                stacklevel=3,
            )
            mask_rechunk_needed = True
            mask_rechunk_dims[ydim] = -1

    # Rechunk mask if needed
    if mask_rechunk_needed:
        logger.info(f"Rechunking mask spatial dimensions: {mask_rechunk_dims}")
        mask = mask.chunk(mask_rechunk_dims)

    # Check coordinate spatial dimensions for single chunks
    coord_rechunk_needed = False
    coord_rechunk_dims = {}

    # Check xdim chunking in lon coordinate
    if lon.chunks is not None and xdim in lon.chunksizes:  # pragma: no cover
        xdim_chunks = lon.chunksizes[xdim]
        if len(xdim_chunks) > 1:
            warnings.warn(
                f"Longitude coordinate spatial dimension '{xdim}' has multiple chunks ({len(xdim_chunks)} chunks). "
                f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                UserWarning,
                stacklevel=3,
            )
            coord_rechunk_needed = True
            coord_rechunk_dims[xdim] = -1

    # Check ydim chunking in lat coordinate for structured grids
    if ydim is not None and lat.chunks is not None and ydim in lat.chunksizes:  # pragma: no cover
        ydim_chunks = lat.chunksizes[ydim]
        if len(ydim_chunks) > 1:
            warnings.warn(
                f"Latitude coordinate spatial dimension '{ydim}' has multiple chunks ({len(ydim_chunks)} chunks). "
                f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                UserWarning,
                stacklevel=3,
            )
            coord_rechunk_needed = True
            coord_rechunk_dims[ydim] = -1

    # Rechunk coordinates if needed
    if coord_rechunk_needed:  # pragma: no cover
        logger.info(f"Rechunking coordinate spatial dimensions: {coord_rechunk_dims}")
        lat = lat.chunk(coord_rechunk_dims).persist()
        lon = lon.chunk(coord_rechunk_dims).persist()

    return data_bin, mask, lat, lon


def validate_unstructured_chunking(
    neighbours: xr.DataArray, cell_areas: xr.DataArray, xdim: str
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Validate that neighbours and cell_areas are in single chunks for unstructured grids.

    Returns the (possibly rechunked) ``neighbours`` and ``cell_areas``.
    """
    # Check neighbours spatial dimensions for single chunks
    neighbours_rechunk_needed = False
    neighbours_rechunk_dims = {}

    # Check xdim chunking in neighbours
    if xdim in neighbours.chunksizes:
        xdim_chunks = neighbours.chunksizes[xdim]
        if len(xdim_chunks) > 1:
            warnings.warn(
                f"Neighbours spatial dimension '{xdim}' has multiple chunks ({len(xdim_chunks)} chunks). "
                f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                UserWarning,
                stacklevel=4,
            )
            neighbours_rechunk_needed = True
            neighbours_rechunk_dims[xdim] = -1

    # Check nv dimension chunking in neighbours
    if "nv" in neighbours.chunksizes:
        nv_chunks = neighbours.chunksizes["nv"]
        if len(nv_chunks) > 1:
            warnings.warn(
                f"Neighbours dimension 'nv' has multiple chunks ({len(nv_chunks)} chunks). "
                f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                UserWarning,
                stacklevel=4,
            )
            neighbours_rechunk_needed = True
            neighbours_rechunk_dims["nv"] = -1

    # Check cell_areas spatial dimensions for single chunks
    cell_areas_rechunk_needed = False
    cell_areas_rechunk_dims = {}

    # Check xdim chunking in cell_areas
    if xdim in cell_areas.chunksizes:
        xdim_chunks = cell_areas.chunksizes[xdim]
        if len(xdim_chunks) > 1:
            warnings.warn(
                f"Cell areas spatial dimension '{xdim}' has multiple chunks ({len(xdim_chunks)} chunks). "
                f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                UserWarning,
                stacklevel=4,
            )
            cell_areas_rechunk_needed = True
            cell_areas_rechunk_dims[xdim] = -1

    # Apply rechunking if needed
    if neighbours_rechunk_needed:
        logger.info(f"Rechunking neighbours spatial dimensions: {neighbours_rechunk_dims}")
        neighbours = neighbours.chunk(neighbours_rechunk_dims)

    if cell_areas_rechunk_needed:
        logger.info(f"Rechunking cell_areas spatial dimensions: {cell_areas_rechunk_dims}")
        cell_areas = cell_areas.chunk(cell_areas_rechunk_dims)

    return neighbours, cell_areas
