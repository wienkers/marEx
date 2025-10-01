"""
MarEx-Track: Marine Extreme Event Identification, Tracking, and Splitting/Merging Module

MarEx identifies and tracks extreme events in oceanographic data across time,
supporting both structured (regular grid) and unstructured datasets. It can identify
discrete objects at single time points and track them as evolving events through time,
seamlessly handling splitting and merging.

This package provides algorithms to:

* Identify binary objects in spatial data at each time step
* Track these objects across time to form coherent events
* Handle merging and splitting of objects over time
* Calculate and maintain object/event properties through time
* Filter by size criteria to focus on significant events

Key terminology:

* Object: A connected region in binary data at a single time point
* Event: One or more objects tracked through time and identified as the same entity
"""

import gc
import logging
import os
import shutil
import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import dask.array as dsa
import numpy as np
import xarray as xr
from dask import persist
from dask.base import is_dask_collection
from dask.distributed import wait
from dask_image.ndmeasure import label
from dask_image.ndmorph import binary_closing as binary_closing_dask
from dask_image.ndmorph import binary_opening as binary_opening_dask
from numba import jit, njit, prange
from numpy.typing import NDArray
from scipy.ndimage import binary_closing, binary_opening
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.csgraph import connected_components
from skimage.measure import regionprops_table

from ._dependencies import warn_missing_dependency
from .exceptions import ConfigurationError, TrackingError, create_coordinate_error, create_data_validation_error
from .logging_config import configure_logging, get_logger, log_dask_info, log_memory_usage, log_timing

# Get module logger
logger = get_logger(__name__)

try:
    import jax.numpy as jnp
except ImportError:
    jnp = np  # type: ignore[misc]  # Alias for jnp when JAX not available
    warn_missing_dependency("jax", "Some functionality")


# ============================
# Main Tracker Class
# ============================


class tracker:
    """
    Tracker identifies and tracks arbitrary binary objects in spatial data through time.

    The tracker supports both structured (regular grid) and unstructured data,
    and seamlessly handles splitting & merging of objects. It identifies
    connected regions in binary data at each time step, and tracks these as
    evolving events through time.

    Main workflow:

    1. Preprocessing: Fill spatiotemporal holes, filter small objects
    2. Object identification: Label connected components at each time
    3. Tracking: Determine object correspondences across time
    4. Optional splitting & merging: Handle complex event evolution

    Parameters
    ----------
    data_bin : xarray.DataArray
        Binary field of extreme points to group, label, and track (True = object, False = background)
        Must represent and underlying `dask` array.
    mask : xarray.DataArray
        Binary mask indicating valid regions (True = valid, False = invalid)
    R_fill : int
        The radius of the kernel used in morphological opening & closing, relating to the largest hole/gap that can be filled.
        In units of grid cells.
    area_filter_quartile : float, optional
        The fraction of the smallest objects to discard, i.e. the quantile defining the smallest area object retained.
        Quantile must be in (0-1) (e.g., 0.25 removes smallest 25%). Mutually exclusive with area_filter_absolute.
        Default is 0.5 if neither parameter is provided.
    area_filter_absolute : int, optional
        The minimum area (in grid cells) for an object to be retained. Mutually exclusive with area_filter_quartile.
        Use this for fixed minimum area thresholds (e.g., 10 cells minimum).
    temp_dir : str, optional
        Path to temporary directory for storing intermediate results
    T_fill : int, default=2
        The permissible temporal gap (in days) between objects for tracking continuity to be maintained (must be even)
    allow_merging : bool, default=True
        Allow objects to split and merge across time.
        Apply splitting & merging criteria, track merge events, and maintain original identities of merged objects across time.
        N.B.: `False` reverts to classical `ndmeasure.label` with simplar time connectivity, i.e. Scannell et al.
    nn_partitioning : bool, default=False
        Implement a better partitioning of merged child objects based on closest parent cell.
        `False` reverts to using parent centroids to determine partitioning between new child objects,
        i.e. Di Sun & Bohai Zhang 2023.
        N.B.: Centroid-based partitioning has major problems with small merging objects suddenly obtaining unrealistically-large
        (and often disjoint) fractions of the larger object.
    overlap_threshold : float, default=0.5
        The fraction of the smaller object's area that must overlap with the larger object's area to be considered the same event
        and continue tracking with the same ID.
    unstructured_grid : bool, default=False
        Whether data is on an unstructured grid
    dimensions : dict, default={"time": "time", "x": "lon", "y": "lat"}
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Coordinate names for unstructured grids.
        Should contain 'x' and 'y' keys for x and y coordinates.
        May also contain 'time' if the time coordinate name is different from
        the dimension name.
    neighbours : xarray.DataArray, optional
        For unstructured grid, indicates connectivity between cells
    cell_areas : xarray.DataArray, optional
        For unstructured grid, area of each cell (required).
        For structured grid, area of each cell (optional). If not provided,
        defaults to 1.0 for each cell (resulting in cell counts as areas).
        Note: Overridden by grid_resolution if provided for structured grids.
    grid_resolution : float, optional
        Grid resolution in degrees for structured grids only (ignored for unstructured grids).
        When provided, automatically calculates cell areas using spherical geometry.
        Overrides any provided cell_areas parameter.
    max_iteration : int, default=40
        Maximum number of iterations for merging/splitting algorithm
    checkpoint : str, default='None'
        Checkpoint strategy ('save', 'load', or None)
    debug : int, default=0
        Debug level (0-2)
    verbose : bool, optional
        Enable verbose logging with detailed progress information.
        If None, uses current global logging configuration.
    quiet : bool, optional
        Enable quiet logging with minimal output (warnings and errors only).
        If None, uses current global logging configuration.
        Note: quiet takes precedence over verbose if both are True.
    regional_mode : bool, default=False
        Enable regional mode for non-global coordinate ranges.
        When True, coordinate_units must be specified.
    coordinate_units : str, optional
        Coordinate units when regional_mode=True.
        Must be either 'degrees' or 'radians'.


    Examples
    --------
    Basic tracking of marine heatwave events from preprocessed data:

    >>> import xarray as xr
    >>> import marEx
    >>>
    >>> # Load preprocessed extreme events data
    >>> processed = xr.open_dataset('extreme_events.nc', chunks={})
    >>> extreme_events = processed.extreme_events  # Boolean array
    >>> mask = processed.mask  # Ocean/land mask
    >>>
    >>> # Initialise tracker with basic parameters
    >>> tracker = marEx.tracker(
    ...     extreme_events,
    ...     mask,
    ...     R_fill=8,                    # Fill holes up to 8 grid cells
    ...     area_filter_quartile=0.5     # Remove smallest 50% of objects
    ...     allow_merging=False          # Basic tracking without splitting/merging
    ... )
    >>>
    >>> # Run tracking algorithm
    >>> events = tracker.run()
    >>> print(f"Identified {events.ID.max().compute()} distinct events")
    Identified 1247 distinct events

    Using automatic grid area calculation from resolution:

    >>> # For regular lat/lon grids, automatically calculate physical areas
    >>> grid_tracker = marEx.tracker(
    ...     extreme_events,
    ...     mask,
    ...     R_fill=8,
    ...     area_filter_quartile=0.5,
    ...     grid_resolution=0.25  # Grid resolution in degrees
    ... )
    >>> # Cell areas are calculated automatically using spherical geometry
    >>> grid_events = grid_tracker.run()

    Advanced tracking with merging and splitting enabled:

    >>> # More sophisticated tracking with temporal gap filling
    >>> advanced_tracker = marEx.tracker(
    ...     extreme_events,
    ...     mask,
    ...     R_fill=12,               # Larger spatial gap filling
    ...     T_fill=4,                # Fill up to 4-day temporal gaps
    ...     area_filter_quartile=0.25,  # More aggressive size filtering
    ...     allow_merging=True,      # Enable split/merge detection
    ...     overlap_threshold=0.3    # Lower threshold for object linking
    ... )
    >>>
    >>> events_advanced, merges_log = advanced_tracker.run(return_merges=True)
    >>> print(events_advanced.data_vars)
    Data variables:
        event           (time, lat, lon)        int32           dask.array<chunksize=(25, 180, 360)>
        event_centroid  (time, lat, lon)        int32           dask.array<chunksize=(25, 180, 360)>
        ID_field        (time, lat, lon)        int32           dask.array<chunksize=(25, 180, 360)>
        global_ID       (time, ID)              int32           dask.array<chunksize=(25, 1247)>
        area            (time, ID)              float32         dask.array<chunksize=(25, 1247)>
        centroid        (component, time, ID)   float64         dask.array<chunksize=(2, 25, 1247)>
        presence        (time, ID)              bool            dask.array<chunksize=(25, 1247)>
        time_start      (ID)                    datetime64[ns]  dask.array<chunksize=(1247,)>
        time_end        (ID)                    datetime64[ns]  dask.array<chunksize=(1247,)>
        merge_ledger    (time, ID, sibling_ID)  int32           dask.array<chunksize=(25, 1247, 10)>

    Processing unstructured ocean model data (ICON):

    >>> # Load ICON ocean model data with connectivity
    >>> icon_data = xr.open_dataset('icon_extremes.nc', chunks={})
    >>> icon_extremes = icon_data.extreme_events  # (time, ncells)
    >>> icon_mask = icon_data.mask
    >>> neighbours = icon_data.neighbours  # Cell connectivity
    >>> cell_areas = icon_data.cell_areas  # Physical areas
    >>>
    >>> # Track events on unstructured grid
    >>> unstructured_tracker = marEx.tracker(
    ...     icon_extremes,
    ...     icon_mask,
    ...     R_fill=5,                                   # 5-neighbor radius for gap filling
    ...     area_filter_quartile=0.6,                   # Remove 60% of smallest events
    ...     unstructured_grid=True,                     # Enable unstructured mode
    ...     dimensions={"x": "ncells"},                 # Must specify the name of the spatial dimension
    ...     coordinates={"x": "lon", "y": "lat"},       # Spatial coordinate names
    ...     neighbours=neighbours,                      # Required for unstructured
    ...     cell_areas=cell_areas                       # Required for area calculations
    ... )
    >>> unstructured_events = unstructured_tracker.run()

    Memory management and checkpointing for large datasets:

    >>> # Use checkpointing for very large datasets
    >>> large_tracker = marEx.tracker(
    ...     extreme_events,
    ...     mask,
    ...     R_fill=8,
    ...     area_filter_quartile=0.5,
    ...     temp_dir='/scratch/user/tracking_temp',  # Temporary storage
    ...     checkpoint='save'             # Save intermediate results
    ... )
    >>> # Processing can be resumed if interrupted
    >>> large_events = large_tracker.run()

    Comparing different filtering strategies:

    >>> # Conservative filtering - keep more events
    >>> conservative = marEx.tracker(
    ...     extreme_events, mask, R_fill=5, area_filter_quartile=0.1
    ... )
    >>> conservative_events = conservative.run()
    >>>
    >>> # Aggressive filtering - focus on largest events
    >>> aggressive = marEx.tracker(
    ...     extreme_events, mask, R_fill=15, area_filter_quartile=0.8
    ... )
    >>> aggressive_events = aggressive.run()
    >>>
    >>> print(f"Conservative: {conservative_events.ID.max().compute()} events")
    >>> print(f"Aggressive: {aggressive_events.ID.max().compute()} events")

    Using absolute area filtering instead of percentile-based:

    >>> # Filter objects smaller than 25 grid cells
    >>> absolute_tracker = marEx.tracker(
    ...     extreme_events, mask, R_fill=8, area_filter_absolute=25
    ... )
    >>> absolute_events = absolute_tracker.run()
    >>>
    >>> # Default behavior (area_filter_quartile=0.5) when no parameters provided
    >>> default_tracker = marEx.tracker(extreme_events, mask, R_fill=8)
    >>> default_events = default_tracker.run()  # Uses quartile=0.5 filtering

    Using physical cell areas for structured grids:

    >>> # Load data with irregular grid cell areas
    >>> grid_areas = xr.open_dataset('grid_areas.nc').cell_area  # (lat, lon) in m²
    >>>
    >>> # Track events using physical areas instead of cell counts
    >>> physical_tracker = marEx.tracker(
    ...     extreme_events,
    ...     mask,
    ...     R_fill=8,
    ...     area_filter_quartile=0.5,
    ...     cell_areas=grid_areas  # Physical areas in m²
    ... )
    >>> events = physical_tracker.run()
    >>> # Now events.area contains physical areas in m² instead of cell counts

    Integration with full marEx workflow:

    >>> # Complete workflow from raw data to tracked events
    >>> raw_sst = xr.open_dataset('sst_data.nc', chunks={}).sst.chunk({'time': 30})
    >>>
    >>> # Step 1: Preprocess to identify extremes
    >>> processed = marEx.preprocess_data(raw_sst, threshold_percentile=95)
    >>>
    >>> # Step 2: Track extreme events
    >>> tracker = marEx.tracker(
    ...     processed.extreme_events,
    ...     processed.mask,
    ...     R_fill=8,
    ...     area_filter_quartile=0.5
    ... )
    >>> tracked_events = tracker.run()
    """

    def __init__(
        self,
        data_bin: xr.DataArray,
        mask: xr.DataArray,
        R_fill: Union[int, float],
        area_filter_quartile: Optional[float] = None,
        area_filter_absolute: Optional[int] = None,
        temp_dir: Optional[str] = None,
        T_fill: int = 2,
        allow_merging: bool = True,
        nn_partitioning: bool = False,
        overlap_threshold: float = 0.5,
        unstructured_grid: bool = False,
        dimensions: Optional[Dict[str, str]] = None,
        coordinates: Optional[Dict[str, str]] = None,
        neighbours: Optional[xr.DataArray] = None,
        cell_areas: Optional[xr.DataArray] = None,
        grid_resolution: Optional[float] = None,
        max_iteration: int = 40,
        checkpoint: Optional[Literal["save", "load", "None"]] = None,
        debug: int = 0,
        verbose: Optional[bool] = None,
        quiet: Optional[bool] = None,
        regional_mode: bool = False,
        coordinate_units: Optional[Literal["degrees", "radians"]] = None,
    ) -> None:
        """Initialise the tracker with parameters and data."""
        # Configure logging if verbose/quiet parameters are provided
        if verbose is not None or quiet is not None:
            configure_logging(verbose=verbose, quiet=quiet)

        # Store logging preferences
        self.verbose = verbose
        self.quiet = quiet

        # Log tracker initialisation
        logger.info("Initialising MarEx tracker")
        logger.info(f"Grid type: {'unstructured' if unstructured_grid else 'structured'}")
        logger.info(
            f"Parameters: R_fill={R_fill}, T_fill={T_fill}, "
            f"area_filter_quartile={area_filter_quartile}, area_filter_absolute={area_filter_absolute}"
        )
        logger.debug(
            f"Tracking options: allow_merging={allow_merging}, nn_partitioning={nn_partitioning}, "
            f"overlap_threshold={overlap_threshold}"
        )

        # Log input data info
        log_dask_info(logger, data_bin, "Binary input data")
        log_memory_usage(logger, "Tracker initialisation")

        self.data_bin = data_bin

        # Store coordinate parameters
        self.regional_mode = regional_mode
        self.coordinate_units = coordinate_units

        # Unify coordinate system: degrees
        dimensions = dimensions or {}
        self.timedim = dimensions.get("time", "time")
        self.xdim = dimensions.get("x", "lon")
        self.ydim: Optional[str] = dimensions.get("y", "lat")
        if unstructured_grid:
            self.timecoord = coordinates["time"] if coordinates and "time" in coordinates else self.timedim
            self.xcoord = coordinates["x"] if coordinates and "x" in coordinates else "lon"
            self.ycoord = coordinates["y"] if coordinates and "y" in coordinates else "lat"

        else:
            coordinates = coordinates or {}
            self.timecoord = coordinates.get("time", self.timedim)
            self.xcoord = coordinates.get("x", self.xdim)
            self.ycoord = coordinates.get("y", self.ydim)

        self.lat_init = data_bin[self.ycoord].persist()  # Save in original units
        self.lon_init = data_bin[self.xcoord].persist()
        self._unify_coordinates()

        self.mask = mask
        self.R_fill = int(R_fill)
        self.T_fill = T_fill

        # Resolve area filtering parameters
        self._resolve_area_filtering_parameters(area_filter_quartile, area_filter_absolute)
        self.allow_merging = allow_merging
        self.nn_partitioning = nn_partitioning
        self.overlap_threshold = overlap_threshold
        self.lat = data_bin[self.ycoord].persist()
        self.lon = data_bin[self.xcoord].persist()
        if data_bin.chunks is not None:
            self.timechunks = data_bin.chunks[data_bin.dims.index(self.timedim)][0]
        else:
            raise create_data_validation_error(
                "Data must be chunked",
                details="The input data_bin must have chunk information",
                suggestions=["Use data_bin.chunk({'time': 10}) to chunk the data"],
            )
        self.unstructured_grid = unstructured_grid
        self.checkpoint = checkpoint
        self.debug = debug

        logger.debug(f"Dimensions: time={self.timedim}, x={self.xdim}, y={self.ydim}")
        logger.debug(f"Coordinates: time={self.timecoord}, x={self.xcoord}, y={self.ycoord}")

        # Extract data_bin metadata to inherit
        if hasattr(self.data_bin, "attrs") and self.data_bin.attrs:
            self.data_attrs = self.data_bin.attrs.copy()
        else:
            self.data_attrs = {}

        # Input validation and preparation
        self._validate_inputs(neighbours, cell_areas, grid_resolution, temp_dir)

        # Handle cell_areas for both structured and unstructured grids
        if self.unstructured_grid:
            # Validation already done in _validate_inputs
            pass
        else:
            # Handle structured grids
            if grid_resolution is not None:
                # Calculate cell areas from grid resolution using spherical geometry
                logger.info(f"Calculating cell areas from grid resolution: {grid_resolution} degrees")

                # Earth radius in km
                R_earth = 6378.0

                # Get coordinate arrays (should be in degrees)
                lat_coords = data_bin[self.ycoord]

                # Convert to radians
                lat_r = np.radians(lat_coords)
                dlat = np.radians(grid_resolution)
                dlon = np.radians(grid_resolution)

                # Calculate grid areas using spherical geometry
                # Area = R² * |sin(lat + dlat/2) - sin(lat - dlat/2)| * dlon
                grid_area = (R_earth**2 * np.abs(np.sin(lat_r + dlat / 2) - np.sin(lat_r - dlat / 2)) * dlon).astype(np.float32)

                # Check if cell_areas was originally provided (and warn about override)
                if cell_areas is not None:
                    logger.warning("grid_resolution parameter overrides provided cell_areas for structured grid")

                cell_areas = grid_area

            elif cell_areas is None:
                # Create unit cell areas (resulting in cell counts)
                if self.ydim is None:
                    raise ValueError("ydim should not be None for structured grids")
                cell_areas = xr.ones_like(data_bin.isel({self.timedim: 0}), dtype=np.float32)
                logger.info("No cell_areas provided for structured grid - using unit areas (cell counts)")
            else:
                # Validation already done in _validate_inputs
                logger.info("Using provided cell_areas for structured grid")

        # Store cell_areas for both grid types
        self.cell_area = cell_areas.astype(np.float32).persist()
        if self.unstructured_grid:
            # Remove coordinate variables for unstructured
            self.cell_area = self.cell_area.drop_vars({self.ycoord, self.xcoord}.intersection(set(cell_areas.coords)))
            self.mean_cell_area = float(cell_areas.mean().compute().item())
        else:
            # For structured grids, calculate mean cell area
            self.mean_cell_area = float(cell_areas.mean().compute().item())

        # Special setup for unstructured grids
        if unstructured_grid:
            # Validation already done in _validate_inputs
            self._setup_unstructured_grid(temp_dir, neighbours, cell_areas, max_iteration)

        self._configure_warnings()

    def _validate_inputs(
        self,
        neighbours: Optional[xr.DataArray] = None,
        cell_areas: Optional[xr.DataArray] = None,
        grid_resolution: Optional[float] = None,
        temp_dir: Optional[str] = None,
    ) -> None:
        """Validate input parameters and data."""
        if self.regional_mode and self.unstructured_grid:
            raise NotImplementedError("regional_mode is not yet implemented for unstructured grids")

        # For unstructured grids, adjust dimensions
        if self.unstructured_grid:
            self.ydim = None
            if (self.timedim, self.xdim) != self.data_bin.dims:
                try:
                    self.data_bin = self.data_bin.transpose(self.timedim, self.xdim)
                except Exception:
                    raise create_data_validation_error(
                        "Invalid dimensions for unstructured data",
                        details=f"Expected 2D array with dimensions ({self.timedim}, {self.xdim}), got {list(self.data_bin.dims)}",
                        suggestions=[
                            "Ensure data has time and cell dimensions only",
                            "Check dimension mapping in function call",
                        ],
                        data_info={
                            "actual_dims": list(self.data_bin.dims),
                            "expected_dims": [self.timedim, self.xdim],
                        },
                    )
        else:
            # For structured grids, ensure 3D data
            if (self.timedim, self.ydim, self.xdim) != self.data_bin.dims:
                try:
                    self.data_bin = self.data_bin.transpose(self.timedim, self.ydim, self.xdim)
                except Exception:
                    raise create_data_validation_error(
                        "Invalid dimensions for gridded data",
                        details=(
                            f"Expected 3D array with dimensions ({self.timedim}, {self.ydim}, {self.xdim}), "
                            f"got {list(self.data_bin.dims)}"
                        ),
                        suggestions=[
                            "Ensure data has time, latitude, and longitude dimensions",
                            "Check dimension mapping and coordinate names",
                        ],
                        data_info={
                            "actual_dims": list(self.data_bin.dims),
                            "expected_dims": [self.timedim, self.ydim, self.xdim],
                        },
                    )

        # Check if self.timecoord, self.xcoord, and self.ycoord are in data_bin coords:
        if (
            self.timecoord not in self.data_bin.coords
            or self.xcoord not in self.data_bin.coords
            or self.ycoord not in self.data_bin.coords
        ):
            raise create_data_validation_error(
                "Missing required coordinates in unstructured data",
                details=(
                    f"Expected coordinates ({self.timecoord}, {self.xcoord}, {self.ycoord}), "
                    f"but found {list(self.data_bin.coords)}"
                ),
                suggestions=[
                    "Ensure data_bin contains time, x, and y coordinates",
                    "Check coordinate names in the dataset",
                    "Specify coordinates in the tracker initialisation with `coordinates` parameter.",
                ],
                data_info={
                    "actual_coords": list(self.data_bin.coords),
                    "expected_coords": [self.timecoord, self.xcoord, self.ycoord],
                },
            )

        # Check if timecoord is an index of timedim
        if self.timecoord != self.timedim and (
            self.timedim not in self.data_bin.indexes or self.data_bin.indexes[self.timedim].name != self.timecoord
        ):
            logger.warning(
                f"timecoord '{self.timecoord}' is not an index of timedim '{self.timedim}'. "
                f"Setting '{self.timecoord}' as index for dimension '{self.timedim}'"
            )
            self.data_bin = self.data_bin.set_index({self.timedim: self.timecoord})

        # Check data type and structure
        if self.data_bin.data.dtype != bool:
            raise create_data_validation_error(
                "Input DataArray must be binary (boolean type)",
                details=f"Found dtype {self.data_bin.data.dtype}, expected bool",
                suggestions=[
                    "Convert data using da > threshold for binary events",
                    "Use xr.where(condition, True, False) for boolean conversion",
                ],
                data_info={
                    "actual_dtype": str(self.data_bin.data.dtype),
                    "expected_dtype": "bool",
                },
            )

        # Validate required parameters for unstructured grids
        if self.unstructured_grid:
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
                expected_spatial_dims = {self.ydim, self.xdim}
                if set(cell_areas.dims) != expected_spatial_dims:
                    raise create_data_validation_error(
                        "Invalid cell_areas dimensions for structured grid",
                        details=f"Expected spatial dimensions {expected_spatial_dims}, got {set(cell_areas.dims)}",
                        suggestions=["Ensure cell_areas matches the spatial dimensions of your data"],
                    )

        # Validate grid_resolution parameter
        if grid_resolution is not None:
            if self.unstructured_grid:
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

        if not is_dask_collection(self.data_bin.data):
            raise create_data_validation_error(
                "Input DataArray must be Dask-backed",
                details="Tracking requires chunked data for efficient processing",
                suggestions=[
                    "Convert to Dask: data_bin = data_bin.chunk({'time': 10})",
                    "Load with chunking: xr.open_dataset('file.nc', chunks={})",
                ],
                data_info={"data_type": type(self.data_bin.data).__name__},
            )

        if self.mask.data.dtype != bool:
            raise create_data_validation_error(
                "Mask must be binary (boolean type)",
                details=f"Found mask dtype {self.mask.data.dtype}, expected bool",
                suggestions=["Convert mask using mask > 0 or mask.astype(bool)"],
                data_info={"mask_dtype": str(self.mask.data.dtype)},
            )

        if not self.mask.any().compute().item():
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
        self._validate_spatial_chunking()

        # Validate resolved area filtering parameters
        if not self._use_absolute_filtering:
            # Quartile-based filtering validation
            if (self.area_filter_quartile < 0) or (self.area_filter_quartile > 1):
                raise ConfigurationError(
                    "Invalid area_filter_quartile value",
                    details=f"Value {self.area_filter_quartile} is outside valid range [0, 1]",
                    suggestions=[
                        "Use values between 0.0 and 1.0",
                        "Use 0.25 to filter smallest 25% of events",
                        "Use 0.5 to keep only larger events",
                    ],
                    context={
                        "provided_value": self.area_filter_quartile,
                        "valid_range": [0, 1],
                    },
                )
        else:
            # Absolute filtering validation
            if self.area_filter_absolute <= 0:
                raise ConfigurationError(
                    "Invalid area_filter_absolute value",
                    details=f"area_filter_absolute={self.area_filter_absolute} must be positive",
                    suggestions=[
                        "Set area_filter_absolute to a positive integer (e.g., 5, 10, 50)",
                    ],
                    context={
                        "area_filter_absolute": self.area_filter_absolute,
                    },
                )

        if self.T_fill % 2 != 0:
            raise ConfigurationError(
                "T_fill must be even for temporal symmetry",
                details=f"Provided T_fill={self.T_fill} is odd",
                suggestions=["Use even values: 2, 4, 6, 8, etc."],
                context={"provided_value": self.T_fill, "requirement": "even number"},
            )

    def _resolve_area_filtering_parameters(
        self, area_filter_quartile: Optional[float], area_filter_absolute: Optional[int]
    ) -> None:
        """Resolve area filtering parameters and set internal state."""
        # Count non-None parameters
        provided_params = sum(x is not None for x in [area_filter_quartile, area_filter_absolute])

        if provided_params == 0:
            # Default case: use quartile-based filtering
            self.area_filter_quartile = 0.5
            self.area_filter_absolute = 0
            self._use_absolute_filtering = False
        elif provided_params == 1:
            # Single parameter provided - use it
            if area_filter_quartile is not None:
                self.area_filter_quartile = area_filter_quartile
                self.area_filter_absolute = 0
                self._use_absolute_filtering = False
            else:  # area_filter_absolute is not None
                self.area_filter_quartile = 0.0  # Set for compatibility
                self.area_filter_absolute = area_filter_absolute
                self._use_absolute_filtering = True
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

    def _validate_spatial_chunking(self) -> None:
        """Validate that spatial dimensions are in single chunks for apply_ufunc operations."""
        rechunk_needed = False
        rechunk_dims = {}

        # Check xdim chunking in data_bin
        if self.xdim in self.data_bin.chunksizes:
            xdim_chunks = self.data_bin.chunksizes[self.xdim]
            if len(xdim_chunks) > 1:
                warnings.warn(
                    f"Spatial dimension '{self.xdim}' has multiple chunks ({len(xdim_chunks)} chunks). "
                    f"This will cause issues with apply_ufunc operations. Rechunking to single chunk."
                    f"Consider directly loading dataset with proper chunking to optimise performance.",
                    UserWarning,
                    stacklevel=3,
                )
                rechunk_needed = True
                rechunk_dims[self.xdim] = -1

        # Check ydim chunking for structured grids
        if self.ydim is not None and self.ydim in self.data_bin.chunksizes:
            ydim_chunks = self.data_bin.chunksizes[self.ydim]
            if len(ydim_chunks) > 1:
                warnings.warn(
                    f"Spatial dimension '{self.ydim}' has multiple chunks ({len(ydim_chunks)} chunks). "
                    f"This will cause issues with apply_ufunc operations. Rechunking to single chunk."
                    f"Consider directly loading dataset with proper chunking to optimise performance.",
                    UserWarning,
                    stacklevel=3,
                )
                rechunk_needed = True
                rechunk_dims[self.ydim] = -1

        # Rechunk data_bin if needed
        if rechunk_needed:
            logger.info(f"Rechunking spatial dimensions: {rechunk_dims}")
            self.data_bin = self.data_bin.chunk(rechunk_dims)

        # Check mask spatial dimensions for single chunks
        mask_rechunk_needed = False
        mask_rechunk_dims = {}

        # Check xdim chunking in mask
        if self.mask.chunks is not None and self.xdim in self.mask.chunksizes:
            xdim_chunks = self.mask.chunksizes[self.xdim]
            if len(xdim_chunks) > 1:
                warnings.warn(
                    f"Mask spatial dimension '{self.xdim}' has multiple chunks ({len(xdim_chunks)} chunks). "
                    f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                    UserWarning,
                    stacklevel=3,
                )
                mask_rechunk_needed = True
                mask_rechunk_dims[self.xdim] = -1

        # Check ydim chunking in mask for structured grids
        if self.ydim is not None and self.mask.chunks is not None and self.ydim in self.mask.chunksizes:
            ydim_chunks = self.mask.chunksizes[self.ydim]
            if len(ydim_chunks) > 1:
                warnings.warn(
                    f"Mask spatial dimension '{self.ydim}' has multiple chunks ({len(ydim_chunks)} chunks). "
                    f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                    UserWarning,
                    stacklevel=3,
                )
                mask_rechunk_needed = True
                mask_rechunk_dims[self.ydim] = -1

        # Rechunk mask if needed
        if mask_rechunk_needed:
            logger.info(f"Rechunking mask spatial dimensions: {mask_rechunk_dims}")
            self.mask = self.mask.chunk(mask_rechunk_dims)

        # Check coordinate spatial dimensions for single chunks
        coord_rechunk_needed = False
        coord_rechunk_dims = {}

        # Check xdim chunking in lon coordinate
        if self.lon.chunks is not None and self.xdim in self.lon.chunksizes:  # pragma: no cover
            xdim_chunks = self.lon.chunksizes[self.xdim]
            if len(xdim_chunks) > 1:
                warnings.warn(
                    f"Longitude coordinate spatial dimension '{self.xdim}' has multiple chunks ({len(xdim_chunks)} chunks). "
                    f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                    UserWarning,
                    stacklevel=3,
                )
                coord_rechunk_needed = True
                coord_rechunk_dims[self.xdim] = -1

        # Check ydim chunking in lat coordinate for structured grids
        if self.ydim is not None and self.lat.chunks is not None and self.ydim in self.lat.chunksizes:  # pragma: no cover
            ydim_chunks = self.lat.chunksizes[self.ydim]
            if len(ydim_chunks) > 1:
                warnings.warn(
                    f"Latitude coordinate spatial dimension '{self.ydim}' has multiple chunks ({len(ydim_chunks)} chunks). "
                    f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                    UserWarning,
                    stacklevel=3,
                )
                coord_rechunk_needed = True
                coord_rechunk_dims[self.ydim] = -1

        # Rechunk coordinates if needed
        if coord_rechunk_needed:  # pragma: no cover
            logger.info(f"Rechunking coordinate spatial dimensions: {coord_rechunk_dims}")
            self.lat = self.lat.chunk(coord_rechunk_dims).persist()
            self.lon = self.lon.chunk(coord_rechunk_dims).persist()

    def _validate_unstructured_chunking(self, neighbours: xr.DataArray, cell_areas: xr.DataArray) -> None:
        """Validate that neighbours and cell_areas are in single chunks for unstructured grids."""
        # Check neighbours spatial dimensions for single chunks
        neighbours_rechunk_needed = False
        neighbours_rechunk_dims = {}

        # Check xdim chunking in neighbours
        if self.xdim in neighbours.chunksizes:
            xdim_chunks = neighbours.chunksizes[self.xdim]
            if len(xdim_chunks) > 1:
                warnings.warn(
                    f"Neighbours spatial dimension '{self.xdim}' has multiple chunks ({len(xdim_chunks)} chunks). "
                    f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                    UserWarning,
                    stacklevel=4,
                )
                neighbours_rechunk_needed = True
                neighbours_rechunk_dims[self.xdim] = -1

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
        if self.xdim in cell_areas.chunksizes:
            xdim_chunks = cell_areas.chunksizes[self.xdim]
            if len(xdim_chunks) > 1:
                warnings.warn(
                    f"Cell areas spatial dimension '{self.xdim}' has multiple chunks ({len(xdim_chunks)} chunks). "
                    f"This will cause issues with apply_ufunc operations. Rechunking to single chunk.",
                    UserWarning,
                    stacklevel=4,
                )
                cell_areas_rechunk_needed = True
                cell_areas_rechunk_dims[self.xdim] = -1

        # Apply rechunking if needed
        if neighbours_rechunk_needed:
            logger.info(f"Rechunking neighbours spatial dimensions: {neighbours_rechunk_dims}")
            # Note: We don't store the rechunked neighbours directly since it's a parameter
            # The caller should handle this if needed

        if cell_areas_rechunk_needed:
            logger.info(f"Rechunking cell_areas spatial dimensions: {cell_areas_rechunk_dims}")
            # Note: We don't store the rechunked cell_areas directly since it's a parameter
            # The caller should handle this if needed

    def _unify_coordinates(self) -> None:

        if self.regional_mode:
            if self.coordinate_units is None:
                raise create_coordinate_error(
                    "coordinate_units must be specified when regional_mode=True",
                    suggestions=[
                        "Set coordinate_units='degrees' for degree-based coordinates",
                        "Set coordinate_units='radians' for radian-based coordinates",
                    ],
                )
            if self.coordinate_units not in ["degrees", "radians"]:
                raise create_coordinate_error(
                    f"Invalid coordinate_units '{self.coordinate_units}'",
                    details="coordinate_units must be either 'degrees' or 'radians'",
                    suggestions=["Use coordinate_units='degrees' or coordinate_units='radians'"],
                )
        else:
            # Check if coordinate_units is explicitly specified
            if self.coordinate_units is not None:
                if self.coordinate_units not in ["degrees", "radians"]:
                    raise create_coordinate_error(
                        f"Invalid coordinate_units '{self.coordinate_units}'",
                        details="coordinate_units must be either 'degrees' or 'radians'",
                        suggestions=["Use coordinate_units='degrees' or coordinate_units='radians'"],
                    )
                # Use explicitly specified coordinate units
            else:
                # Auto-detect coordinate units for global data
                lon = self.data_bin[self.xcoord]
                lon_range = float(lon.max()) - float(lon.min())

                # Check for degrees (range close to 360)
                if abs(lon_range - 360.0) <= 1.0:
                    self.coordinate_units = "degrees"

                # Check for radians (range close to 2π)
                elif abs(lon_range - 2 * np.pi) <= 0.02:
                    self.coordinate_units = "radians"

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
                        context={"detected_range": lon_range, "xdim": self.xcoord},
                    )

        # Convert lat & lon to degrees
        if self.coordinate_units == "radians":
            self.data_bin[self.xcoord] = self.data_bin[self.xcoord] * 180.0 / np.pi
            self.data_bin[self.ycoord] = self.data_bin[self.ycoord] * 180.0 / np.pi

    def _remap_coordinates(self, events_ds: xr.Dataset) -> xr.Dataset:
        """Remap coordinates to original lat/lon values after processing.
        Map centroids from lat=[-180,180] back into original lat/lon units & range.
        """
        # Re-assign original coordinates from original marEx input
        events_ds = events_ds.assign_coords({self.ycoord: self.lat_init.compute(), self.xcoord: self.lon_init.compute()})

        if "centroid" in events_ds.data_vars:
            # Remap centroids to original coordinate system
            # (lat, lon) currently in degrees [-90,90], [-180,180]
            centroids = events_ds["centroid"].persist()

            # Split into components
            centroids_lat = centroids.isel(component=0)  # [-90, 90] degrees
            centroids_lon = centroids.isel(component=1)  # [-180, 180] degrees

            # Get original coordinate bounds
            lon_min = float(self.lon_init.min().compute().item())
            lon_max = float(self.lon_init.max().compute().item())

            # Convert units and adjust ranges
            if self.coordinate_units == "radians":
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

    def _setup_unstructured_grid(
        self,
        temp_dir: str,
        neighbours: xr.DataArray,
        cell_areas: xr.DataArray,
        max_iteration: int,
    ) -> None:
        """Set up special handling for unstructured grids."""
        if not temp_dir:
            raise ConfigurationError(
                "Missing temporary directory for unstructured processing",
                details="Unstructured grids require temporary storage for memory efficiency",
                suggestions=[
                    "Provide temp_dir parameter: tracker(..., temp_dir='/tmp/marex')",
                    "Ensure directory has sufficient space and write permissions",
                ],
            )

        self.scratch_dir = temp_dir

        # Clear any existing temporary storage
        if os.path.exists(f"{self.scratch_dir}/marEx_temp_field.zarr/"):
            shutil.rmtree(f"{self.scratch_dir}/marEx_temp_field.zarr/")

        # Remove coordinate variables to avoid memory issues
        self.data_bin = self.data_bin.drop_vars({self.ycoord, self.xcoord})
        self.mask = self.mask.drop_vars({self.ycoord, self.xcoord})
        self.lat = self.lat.drop_vars(self.lat.coords)
        self.lon = self.lon.drop_vars(self.lon.coords)
        neighbours = neighbours.drop_vars({self.ycoord, self.xcoord, "nv"}.intersection(set(neighbours.coords)))

        self.max_iteration = max_iteration

        # Validate spatial chunking for unstructured grid data
        self._validate_unstructured_chunking(neighbours, cell_areas)

        # Initialise dilation array for unstructured grid
        self.neighbours_int = neighbours.astype(np.int32) - 1  # Convert to 0-based indexing

        # Validate neighbour array structure
        if self.neighbours_int.shape[0] != 3:
            raise create_data_validation_error(
                "Invalid neighbour array for triangular grid",
                details=f"Expected shape (3, ncells), got {self.neighbours_int.shape}",
                suggestions=[
                    "Ensure triangular grid connectivity",
                    "Check neighbour array from grid file",
                    "Verify unstructured grid format",
                ],
                data_info={
                    "actual_shape": self.neighbours_int.shape,
                    "expected_shape": "(3, ncells)",
                },
            )
        if self.neighbours_int.dims != ("nv", self.xdim):
            raise create_data_validation_error(
                "Invalid neighbour array dimensions",
                details=f"Expected dimensions ('nv', '{self.xdim}'), got {self.neighbours_int.dims}",
                suggestions=[
                    "Check dimension names in grid file",
                    "Verify coordinate mapping",
                ],
                data_info={
                    "actual_dims": self.neighbours_int.dims,
                    "expected_dims": ("nv", self.xdim),
                },
            )

        # Construct sparse dilation matrix
        self._build_sparse_dilation_matrix()

    def _build_sparse_dilation_matrix(self) -> None:
        """Build sparse matrix for efficient dilation operations on unstructured grid."""
        # Create row and column indices for sparse matrix
        row_indices = jnp.repeat(jnp.arange(self.neighbours_int.shape[1]), 3)
        col_indices = self.neighbours_int.data.compute().T.flatten()

        # Filter out negative values (invalid connections)
        valid_mask = col_indices >= 0
        row_indices = row_indices[valid_mask]
        col_indices = col_indices[valid_mask]

        # Create the sparse matrix for dilation
        ncells = self.neighbours_int.shape[1]
        dilate_coo = coo_matrix(
            (jnp.ones_like(row_indices, dtype=bool), (row_indices, col_indices)),
            shape=(ncells, ncells),
        )
        self.dilate_sparse = csr_matrix(dilate_coo)

        # Add identity matrix to include self-connections
        identity = eye(self.neighbours_int.shape[1], dtype=bool, format="csr")
        self.dilate_sparse = self.dilate_sparse + identity

        logger.info("Finished constructing the sparse dilation matrix")

    def _configure_warnings(self) -> None:
        """Configure warning and logging suppression based on debug level."""
        logger.debug(f"Configuring warnings and logging for debug level: {self.debug}")
        if self.debug < 2:
            # Configure logging warning filters
            logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)

            def filter_dask_warnings(record):  # pragma: no cover
                msg = str(record.msg)

                if self.debug == 0:
                    # Suppress both run_spec and large graph warnings
                    if any(
                        pattern in msg
                        for pattern in [
                            "Detected different `run_spec`",
                            "Sending large graph",
                            "This may cause some slowdown",
                        ]
                    ):
                        return False
                    return True
                else:
                    # Suppress only run_spec warnings
                    if "Detected different `run_spec`" in msg:
                        return False
                    return True

            logging.getLogger("distributed.scheduler").addFilter(filter_dask_warnings)

            # Configure Python warnings
            if self.debug == 0:
                warnings.filterwarnings("ignore", category=UserWarning, module="distributed.client")
                warnings.filterwarnings(
                    "ignore",
                    message=".*Sending large graph.*\n.*This may cause some slowdown.*",
                    category=UserWarning,
                )

    # ============================
    # Main Public Methods
    # ============================

    def run(
        self, return_merges: bool = False, checkpoint: Optional[str] = None
    ) -> Union[xr.Dataset, Tuple[xr.Dataset, xr.Dataset]]:
        """
        Run the complete object identification and tracking pipeline.

        This method executes the full workflow:

        1. Preprocessing: morphological operations and size filtering
        2. Identification and tracking of objects through time
        3. Computing and attaching statistics to the results

        Parameters
        ----------
        return_merges : bool, default=False
            If True, return merge events dataset alongside the main events
        checkpoint : str, optional
            Override the instance checkpoint setting

        Returns
        -------
        events_ds : xarray.Dataset
            Dataset containing tracked events and their properties
        merges_ds : xarray.Dataset, optional
            Dataset with merge event information (only if return_merges=True)
        """
        logger.info("Starting complete tracking pipeline")
        log_memory_usage(logger, "Pipeline start")

        # Progress tracking
        total_steps = 3
        current_step = 0

        # Preprocess the binary data
        current_step += 1
        logger.info(f"Step {current_step}/{total_steps}: Data preprocessing")
        with log_timing(logger, "Data preprocessing", log_memory=True, show_progress=True):
            data_bin_preprocessed, object_stats = self.run_preprocess(checkpoint=checkpoint)

        # Run identification and tracking
        current_step += 1
        logger.info(f"Step {current_step}/{total_steps}: Object identification and tracking")
        with log_timing(
            logger,
            "Object identification and tracking",
            log_memory=True,
            show_progress=True,
        ):
            events_ds, merges_ds, N_events_final = self.run_tracking(data_bin_preprocessed)

        # Compute statistics and finalise output
        current_step += 1
        logger.info(f"Step {current_step}/{total_steps}: Computing event statistics and attributes")
        with log_timing(
            logger,
            "Computing event statistics and attributes",
            log_memory=True,
            show_progress=True,
        ):
            events_ds = self.run_stats_attributes(events_ds, merges_ds, object_stats, N_events_final)

        logger.info(f"Tracking pipeline completed successfully - {N_events_final} events identified")
        logger.debug(f"Final dataset dimensions: {events_ds.dims}")
        log_memory_usage(logger, "Pipeline completion")

        if self.allow_merging and return_merges:
            logger.debug("Returning both events and merge datasets")
            return events_ds, merges_ds
        else:
            logger.debug("Returning events dataset only")
            return events_ds

    def run_preprocess(self, checkpoint: Optional[str] = None) -> Tuple[xr.DataArray, Tuple[float, int, int, float, float, float]]:
        """
        Preprocess binary data to prepare for tracking.

        This performs morphological operations to fill holes/gaps in both space and time,
        then filters small objects according to the area_filter_quartile or area_filter_absolute.

        Parameters
        ----------
        checkpoint : str, optional
            Checkpoint strategy override

        Returns
        -------
        data_bin_filtered : xarray.DataArray
            Preprocessed binary data
        object_stats : tuple
            Statistics about the preprocessing
        """
        if not checkpoint:
            checkpoint = self.checkpoint

        def load_data_from_checkpoint() -> xr.DataArray:
            """Load preprocessed data from checkpoint files."""
            data_bin_preprocessed: xr.DataArray = xr.open_zarr(
                f"{self.scratch_dir}/marEx_checkpoint_proc_bin.zarr",
                chunks={self.timedim: self.timechunks},
            )["data_bin_preproc"]
            return data_bin_preprocessed

        def load_stats_from_checkpoint() -> Tuple[float, int, int, float, float, float]:
            object_stats_npz = np.load(f"{self.scratch_dir}/marEx_checkpoint_stats.npz")
            object_stats = [
                object_stats_npz[key]
                for key in [
                    "total_area_IDed",
                    "N_objects_prefiltered",
                    "N_objects_filtered",
                    "area_threshold",
                    "accepted_area_fraction",
                    "preprocessed_area_fraction",
                ]
            ]
            return tuple(object_stats)  # type: ignore[return-value]

        if checkpoint == "load":
            logger.info("Loading preprocessed data from checkpoint")
            return load_data_from_checkpoint(), load_stats_from_checkpoint()

        # Compute area of initial binary data
        logger.debug("Computing area of initial binary data")
        raw_area = self.compute_area(self.data_bin)
        logger.debug(f"Initial raw area: {raw_area}")

        # Fill small holes & gaps between objects
        logger.info(f"Filling spatial holes with radius R_fill={self.R_fill}")
        with log_timing(logger, "Spatial hole filling"):
            data_bin_filled = self.fill_holes(self.data_bin)
            del self.data_bin  # Free memory
            log_memory_usage(logger, "After spatial hole filling", logging.DEBUG)

        # Fill small time-gaps between objects
        logger.info(f"Filling temporal gaps with T_fill={self.T_fill}")
        with log_timing(logger, "Temporal gap filling"):
            data_bin_filled = self.fill_time_gaps(data_bin_filled).persist()
            log_memory_usage(logger, "After temporal gap filling", logging.DEBUG)

        # Remove small objects
        logger.info("Filtering small objects")
        with log_timing(logger, "Small object filtering"):
            (
                data_bin_filtered,
                area_threshold,
                object_areas,
                N_objects_prefiltered,
                N_objects_filtered,
            ) = self.filter_small_objects(data_bin_filled)
            del data_bin_filled  # Free memory
            logger.info(f"Filtered {N_objects_prefiltered} -> {N_objects_filtered} objects (threshold: {area_threshold})")
            log_memory_usage(logger, "After object filtering", logging.DEBUG)

        # Persist preprocessed data &/or Save checkpoint
        if checkpoint and "save" in checkpoint:
            logger.info("Saving preprocessed data to checkpoint")
            with log_timing(logger, "Checkpoint saving"):
                time.sleep(5)
                data_bin_filtered.name = "data_bin_preproc"
                data_bin_filtered.to_zarr(
                    f"{self.scratch_dir}/marEx_checkpoint_proc_bin.zarr", mode="w"
                )  # N.B.: This needs to be done without .persist() due to dask to_zarr tuple bug...
                data_bin_filtered = load_data_from_checkpoint()
        else:
            logger.debug("Persisting preprocessed data in memory")
            data_bin_filtered = data_bin_filtered.persist()
            wait(data_bin_filtered)

        # Compute area of processed data
        processed_area = self.compute_area(data_bin_filtered)

        # Compute statistics
        object_areas = object_areas.compute()
        total_area_IDed = float(object_areas.sum().item())

        accepted_area = float(object_areas.where(object_areas > area_threshold, drop=True).sum().item())
        accepted_area_fraction = accepted_area / total_area_IDed

        total_hobday_area = float(raw_area.sum().compute().item())
        total_processed_area = float(processed_area.sum().compute().item())
        preprocessed_area_fraction = total_hobday_area / total_processed_area

        object_stats = (
            total_area_IDed,
            N_objects_prefiltered,
            N_objects_filtered,
            area_threshold,
            accepted_area_fraction,
            preprocessed_area_fraction,
        )

        # Save checkpoint
        if checkpoint and "save" in checkpoint:
            np.savez(
                f"{self.scratch_dir}/marEx_checkpoint_stats.npz",
                total_area_IDed=total_area_IDed,
                N_objects_prefiltered=N_objects_prefiltered,
                N_objects_filtered=N_objects_filtered,
                area_threshold=area_threshold,
                accepted_area_fraction=accepted_area_fraction,
                preprocessed_area_fraction=preprocessed_area_fraction,
            )
            # Reload to refresh the dask graph
            data_bin_filtered = load_data_from_checkpoint()
            object_stats = load_stats_from_checkpoint()

        return data_bin_filtered, object_stats

    def run_tracking(self, data_bin_preprocessed: xr.DataArray) -> Tuple[xr.Dataset, xr.Dataset, int]:
        """
        Track objects through time to identify events.

        Parameters
        ----------
        data_bin_preprocessed : xarray.DataArray
            Preprocessed binary data

        Returns
        -------
        events_ds : xarray.Dataset
            Dataset containing tracked events
        merges_ds : xarray.Dataset
            Dataset with merge information
        N_events_final : int
            Final number of unique events
        """
        if self.allow_merging or self.unstructured_grid:
            # Track with merging & splitting
            events_ds, merges_ds, N_events_final = self.track_objects(data_bin_preprocessed)
        else:
            # Track without merging or splitting
            events_da, _, N_events_final = self.identify_objects(data_bin_preprocessed, time_connectivity=True)
            events_ds = xr.Dataset({"ID_field": events_da})
            merges_ds = xr.Dataset()

        # Set all filler IDs < 0 to 0
        events_ds["ID_field"] = events_ds.ID_field.where(events_ds.ID_field > 0, drop=False, other=0)

        # Restore original coordinate name if needed
        if self.timecoord != self.timedim and self.timedim in events_ds.coords and self.timecoord not in events_ds.coords:
            # Get the time coordinate data
            time_coord_data = events_ds.coords[self.timedim]
            # Create a new coordinate with the original name
            events_ds = events_ds.assign_coords({self.timecoord: time_coord_data})
            # Remove the dimension coordinate to avoid duplication
            if self.timedim in events_ds.coords and self.timecoord in events_ds.coords:
                events_ds = events_ds.drop_vars(self.timedim)

        logger.info("Finished tracking all extreme events!")

        return events_ds, merges_ds, N_events_final

    def run_stats_attributes(
        self,
        events_ds: xr.Dataset,
        merges_ds: xr.Dataset,
        object_stats: Tuple[float, int, int, float, float, float],
        N_events_final: int,
    ) -> xr.Dataset:
        """
        Add statistics and attributes to the events dataset.

        Parameters
        ----------
        events_ds : xarray.Dataset
            Dataset containing tracked events
        merges_ds : xarray.Dataset
            Dataset with merge information
        object_stats : tuple
            Preprocessed object statistics
        N_events_final : int
            Final number of events

        Returns
        -------
        events_ds : xarray.Dataset
            Dataset with added statistics and attributes
        """
        # Unpack object stats
        (
            total_area_IDed,
            N_objects_prefiltered,
            N_objects_filtered,
            area_threshold,
            accepted_area_fraction,
            preprocessed_area_fraction,
        ) = object_stats

        # Add general attributes to dataset
        events_ds.attrs["allow_merging"] = int(self.allow_merging)
        events_ds.attrs["N_objects_prefiltered"] = int(N_objects_prefiltered)
        events_ds.attrs["N_objects_filtered"] = int(N_objects_filtered)
        events_ds.attrs["N_events_final"] = int(N_events_final)
        events_ds.attrs["R_fill"] = self.R_fill
        events_ds.attrs["T_fill"] = self.T_fill
        events_ds.attrs["area_filter_quartile"] = self.area_filter_quartile
        events_ds.attrs["area_threshold (cells)"] = area_threshold
        events_ds.attrs["accepted_area_fraction"] = accepted_area_fraction
        events_ds.attrs["preprocessed_area_fraction"] = preprocessed_area_fraction

        # Print summary statistics
        print("Tracking Statistics:")
        print(f"   Binary Hobday to Processed Area Fraction: {preprocessed_area_fraction}")
        print(f"   Total Object Area IDed (cells): {total_area_IDed}")
        print(f"   Number of Initial Pre-Filtered Objects: {N_objects_prefiltered}")
        print(f"   Number of Final Filtered Objects: {N_objects_filtered}")
        print(f"   Area Cutoff Threshold (cells): {int(area_threshold)}")
        print(f"   Accepted Area Fraction: {accepted_area_fraction}")
        print(f"   Total Events Tracked: {N_events_final}")

        # Add merge-specific attributes if applicable
        if self.allow_merging:
            events_ds.attrs["overlap_threshold"] = self.overlap_threshold
            events_ds.attrs["nn_partitioning"] = int(self.nn_partitioning)

            # Add merge summary attributes
            events_ds.attrs["total_merges"] = len(merges_ds.merge_ID)
            events_ds.attrs["multi_parent_merges"] = int((merges_ds.n_parents > 2).sum().item())

            print(f"   Total Merging Events Recorded: {events_ds.attrs['total_merges']}")

        # Inherit metadata from input data_bin
        events_ds.attrs.update(self.data_attrs)

        # Restore coordinates & remap centroids
        # Add lat & lon back as coordinates
        events_ds = self._remap_coordinates(events_ds)

        # Rechunk to size 1 for better post-processing
        events_ds = events_ds.chunk({self.timedim: 1})

        return events_ds

    # ============================
    # Data Processing Methods
    # ============================

    def compute_area(self, data_bin: xr.DataArray) -> xr.DataArray:
        """
        Compute the total area of binary data at each time.

        Parameters
        ----------
        data_bin : xarray.DataArray
            Binary data

        Returns
        -------
        area : xarray.DataArray
            Total area at each time (units: pixels for structured grid, matching cell_area for unstructured)
        """
        if self.unstructured_grid:
            area = (data_bin * self.cell_area).sum(dim=[self.xdim])
        else:
            area = data_bin.sum(dim=[self.ydim, self.xdim])

        return area

    def fill_holes(self, data_bin: xr.DataArray, R_fill: Optional[int] = None) -> xr.DataArray:
        """
        Fill holes and gaps using morphological operations.

        This performs closing (dilation followed by erosion) to fill small gaps,
        then opening (erosion followed by dilation) to remove small isolated objects.

        Parameters
        ----------
        data_bin : xarray.DataArray
            Binary data to process
        R_fill : int, optional
            Fill radius override

        Returns
        -------
        data_bin_filled : xarray.DataArray
            Binary data with holes/gaps filled
        """
        if R_fill is None:
            R_fill = self.R_fill

        if self.unstructured_grid:
            # Process unstructured grid using sparse matrix operations
            # _Put the data into an xarray.DataArray to pass into the apply_ufunc_ -- Needed for correct memory management !
            sp_data = xr.DataArray(self.dilate_sparse.data, dims="sp_data")
            indices = xr.DataArray(self.dilate_sparse.indices, dims="indices")
            indptr = xr.DataArray(self.dilate_sparse.indptr, dims="indptr")

            def binary_open_close(
                bitmap_binary: NDArray[np.bool_],
                sp_data: NDArray[np.bool_],
                indices: NDArray[np.int32],
                indptr: NDArray[np.int32],
                mask: NDArray[np.bool_],
            ) -> NDArray[np.bool_]:
                """
                Binary opening and closing for unstructured grid.
                Uses sparse matrix power operations for efficiency.
                """
                # Closing: Dilation then Erosion (fills small gaps)

                # Dilation
                bitmap_binary = sparse_bool_power(bitmap_binary, sp_data, indices, indptr, R_fill)

                # Set land values to True (to avoid artificially eroding the shore)
                bitmap_binary[:, ~mask] = True

                # Erosion (negated dilation of negated image)
                bitmap_binary = ~sparse_bool_power(~bitmap_binary, sp_data, indices, indptr, R_fill)

                # Opening: Erosion then Dilation (removes small objects)

                # Set land values to True (to avoid artificially eroding the shore)
                bitmap_binary[:, ~mask] = True

                # Erosion
                bitmap_binary = ~sparse_bool_power(~bitmap_binary, sp_data, indices, indptr, R_fill)

                # Dilation
                bitmap_binary = sparse_bool_power(bitmap_binary, sp_data, indices, indptr, R_fill)

                return bitmap_binary

            # Apply the operations
            data_bin = xr.apply_ufunc(
                binary_open_close,
                data_bin,
                sp_data,
                indices,
                indptr,
                self.mask,
                input_core_dims=[
                    [self.xdim],
                    ["sp_data"],
                    ["indices"],
                    ["indptr"],
                    [self.xdim],
                ],
                output_core_dims=[[self.xdim]],
                output_dtypes=[np.bool_],
                vectorize=False,
                dask_gufunc_kwargs={
                    "output_sizes": {self.xdim: data_bin.sizes[self.xdim]},
                },
                dask="parallelized",
            )

        else:
            # Structured grid using dask-powered morphological operations
            use_dask_morph = True

            # Generate structuring element (disk-shaped)
            y, x = np.ogrid[-R_fill : R_fill + 1, -R_fill : R_fill + 1]
            r = x**2 + y**2
            diameter = 2 * R_fill
            se_kernel = r < (R_fill**2) + 1
            mode = "wrap" if not self.regional_mode else "edge"

            if use_dask_morph:
                # Skip all operations if R_fill is 0
                if R_fill == 0:
                    pass  # No morphological operations needed
                else:
                    # Pad data to avoid edge effects
                    data_bin = data_bin.pad({self.ydim: diameter, self.xdim: diameter}, mode=mode)
                    data_coords = data_bin.coords
                    data_dims = data_bin.dims

                    # Apply morphological operations
                    data_bin = binary_closing_dask(
                        data_bin.data, structure=se_kernel[np.newaxis, :, :]
                    )  # N.B.: There may be a rearing bug in constructing the dask task graph when we
                    # extract and then re-imbed the dask array into an xarray DataArray
                    data_bin = binary_opening_dask(data_bin, structure=se_kernel[np.newaxis, :, :])

                    # Convert back to xarray.DataArray and trim padding
                    data_bin = xr.DataArray(data_bin, coords=data_coords, dims=data_dims)
                    data_bin = data_bin.isel(
                        {
                            self.ydim: slice(diameter, -diameter),
                            self.xdim: slice(diameter, -diameter),
                        }
                    )
            else:  # pragma: no cover

                def binary_open_close(
                    bitmap_binary: NDArray[np.bool_],
                ) -> NDArray[np.bool_]:
                    """Apply binary opening and closing in one function."""
                    bitmap_binary_padded = np.pad(
                        bitmap_binary,
                        ((diameter, diameter), (diameter, diameter)),
                        mode=mode,
                    )
                    s1 = binary_closing(bitmap_binary_padded, se_kernel, iterations=1)
                    s2 = binary_opening(s1, se_kernel, iterations=1)
                    unpadded = s2[diameter:-diameter, diameter:-diameter]
                    return unpadded

                data_bin = xr.apply_ufunc(
                    binary_open_close,
                    data_bin,
                    input_core_dims=[[self.ydim, self.xdim]],
                    output_core_dims=[[self.ydim, self.xdim]],
                    output_dtypes=[data_bin.dtype],
                    vectorize=True,
                    dask="parallelized",
                )

            # Mask out edge features from morphological operations
            data_bin = data_bin.where(self.mask, drop=False, other=False)

        return data_bin

    def fill_time_gaps(self, data_bin: xr.DataArray) -> xr.DataArray:
        """
        Fill temporal gaps between objects.

        Performs binary closing (dilation then erosion) along the time dimension
        to fill small time gaps between objects.

        Parameters
        ----------
        data_bin : xarray.DataArray
            Binary data to process

        Returns
        -------
        data_bin_filled : xarray.DataArray
            Binary data with temporal gaps filled
        """
        if self.T_fill == 0:
            return data_bin

        # Create temporal structuring element
        kernel_size = self.T_fill + 1  # This will then fill a maximum hole size of self.T_fill
        time_kernel = np.ones(kernel_size, dtype=bool)

        if self.ydim is None:
            # Unstructured grid has only 1 additional dimension
            time_kernel = time_kernel[:, np.newaxis]
        else:
            time_kernel = time_kernel[:, np.newaxis, np.newaxis]

        # Pad in time to avoid edge effects
        data_bin = data_bin.pad({self.timedim: kernel_size}, mode="constant", constant_values=False)

        # Apply temporal closing
        data_bin_dask = data_bin.data
        closed_dask_array = binary_closing_dask(data_bin_dask, structure=time_kernel)

        # Convert back to xarray.DataArray
        data_bin_filled = xr.DataArray(
            closed_dask_array,
            coords=data_bin.coords,
            dims=data_bin.dims,
            attrs=data_bin.attrs,
        )

        # Remove padding
        data_bin_filled = data_bin_filled.isel({self.timedim: slice(kernel_size, -kernel_size)}).persist()

        # Fill newly-created spatial holes
        data_bin_filled = self.fill_holes(data_bin_filled, R_fill=self.R_fill // 2)

        return data_bin_filled

    def refresh_dask_graph(self, data_bin: xr.DataArray) -> xr.DataArray:
        """
        Clear and reset the Dask graph via save/load cycle.

        This is needed to work around a memory leak bug in Dask where
        "Unmanaged Memory" builds up within loops.

        Parameters
        ----------
        data_bin : xarray.DataArray
            Data to refresh

        Returns
        -------
        data_new : xarray.DataArray
            Data with fresh Dask graph
        """
        logger.debug("Refreshing Dask task graph...")

        data_bin.name = "temp"
        data_bin.to_zarr(f"{self.scratch_dir}/marEx_temp_field.zarr", mode="w")
        del data_bin
        gc.collect()

        data_new = xr.open_zarr(f"{self.scratch_dir}/marEx_temp_field.zarr", chunks={}).temp
        return data_new

    def filter_small_objects(self, data_bin: xr.DataArray) -> Tuple[xr.DataArray, float, xr.DataArray, int, int]:
        """
        Remove objects smaller than a threshold area.

        Parameters
        ----------
        data_bin : xarray.DataArray
            Binary data to filter

        Returns
        -------
        data_bin_filtered : xarray.DataArray
            Binary data with small objects removed
        area_threshold : float
            Area threshold used for filtering
        object_areas : xarray.DataArray
            Areas of all objects pre-filtering
        N_objects_prefiltered : int
            Number of objects before filtering
        N_objects_filtered : int
            Number of objects after filtering
        """
        # Cluster & Label Binary Data: Time-independent in 2D (i.e. no time connectivity!)
        object_id_field, _, N_objects_unfiltered = self.identify_objects(data_bin, time_connectivity=False)

        if self.unstructured_grid:
            # Get the maximum ID to dimension arrays
            #  Note: identify_objects() starts at ID=0 for every time slice
            max_ID = int(object_id_field.max().compute().item())

            def count_cluster_sizes(
                object_id_field: NDArray[np.int32],
            ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
                """Count the number of cells in each cluster."""
                unique, counts = np.unique(object_id_field[object_id_field > 0], return_counts=True)
                padded_sizes = np.zeros(max_ID, dtype=np.int32)
                padded_unique = np.zeros(max_ID, dtype=np.int32)
                padded_sizes[: len(counts)] = counts
                padded_unique[: len(counts)] = unique
                return padded_sizes, padded_unique

            # Calculate cluster sizes
            cluster_sizes, unique_cluster_IDs = xr.apply_ufunc(
                count_cluster_sizes,
                object_id_field,
                input_core_dims=[[self.xdim]],
                output_core_dims=[["ID"], ["ID"]],
                dask_gufunc_kwargs={"output_sizes": {"ID": max_ID}},
                output_dtypes=(np.int32, np.int32),
                vectorize=True,
                dask="parallelized",
            )

            results = persist(cluster_sizes, unique_cluster_IDs)
            cluster_sizes, unique_cluster_IDs = results

            # Pre-filter tiny objects for performance (greatly reduces the size for the percentile calculation)
            if self._use_absolute_filtering:
                cluster_sizes_filtered_dask = cluster_sizes.where(cluster_sizes > 5).data
            else:
                cluster_sizes_filtered_dask = cluster_sizes.where(cluster_sizes > 50).data
            cluster_areas_mask = dsa.isfinite(cluster_sizes_filtered_dask)
            object_areas = cluster_sizes_filtered_dask[cluster_areas_mask].compute()

            # Filter based on area threshold
            N_objects_unfiltered = len(object_areas)
            if N_objects_unfiltered == 0:  # pragma: no cover
                raise TrackingError(
                    "No objects found for area-based filtering",
                    details={
                        "objects_count": N_objects_unfiltered,
                        "area_filter_quartile": self.area_filter_quartile,
                        "grid_type": "unstructured",
                    },
                    suggestions=[
                        "Check if input data contains any extreme events",
                        "Verify that preprocessing parameters are appropriate",
                        "Consider lowering the extreme threshold percentile",
                    ],
                )
            if self._use_absolute_filtering:
                area_threshold = self.area_filter_absolute
            else:
                area_threshold = np.percentile(object_areas, self.area_filter_quartile * 100)
            N_objects_filtered = np.sum(object_areas > area_threshold)

            def filter_area_binary(cluster_IDs_0: NDArray[np.int32], keep_IDs_0: NDArray[np.int32]) -> NDArray[np.bool_]:
                """Keep only clusters above threshold area."""
                keep_IDs_0 = keep_IDs_0[keep_IDs_0 > 0]
                keep_where = np.isin(cluster_IDs_0, keep_IDs_0)
                return keep_where

            # Create filtered binary data
            keep_IDs = xr.where(cluster_sizes > area_threshold, unique_cluster_IDs, 0)

            data_bin_filtered = xr.apply_ufunc(
                filter_area_binary,
                object_id_field,
                keep_IDs,
                input_core_dims=[[self.xdim], ["ID"]],
                output_core_dims=[[self.xdim]],
                output_dtypes=[data_bin.dtype],
                vectorize=True,
                dask="parallelized",
            )

            object_areas = cluster_sizes  # Store pre-filtered areas

        else:
            # Structured grid approach

            # Calculate object properties including area
            object_props = self.calculate_object_properties(object_id_field)
            object_areas, object_ids = object_props.area, object_props.ID

            # Calculate area threshold
            if len(object_areas) == 0:  # pragma: no cover
                raise TrackingError(
                    "No objects found for area-based filtering",
                    details={
                        "objects_count": len(object_areas),
                        "area_filter_quartile": self.area_filter_quartile,
                        "grid_type": "structured",
                    },
                    suggestions=[
                        "Check if input data contains any extreme events",
                        "Verify that preprocessing parameters are appropriate",
                        "Consider lowering the extreme threshold percentile",
                    ],
                )
            if self._use_absolute_filtering:
                area_threshold = self.area_filter_absolute
            else:
                area_threshold = np.percentile(object_areas, self.area_filter_quartile * 100.0)

            # Keep only objects above threshold
            object_ids_keep = xr.where(object_areas >= area_threshold, object_ids, -1)
            object_ids_keep[0] = -1  # Don't keep ID=0

            # Create filtered binary data
            data_bin_filtered = object_id_field.isin(object_ids_keep)

            # Count objects after filtering
            N_objects_filtered = int(object_ids_keep.where(object_ids_keep > 0).count().item())

        return (
            data_bin_filtered,
            area_threshold,
            object_areas,
            N_objects_unfiltered,
            N_objects_filtered,
        )

    # ============================
    # Object Identification Methods
    # ============================

    def identify_objects(self, data_bin: xr.DataArray, time_connectivity: bool) -> Tuple[xr.DataArray, None, int]:
        """
        Identify connected regions in binary data.

        Parameters
        ----------
        data_bin : xarray.DataArray
            Binary data to identify objects in
        time_connectivity : bool
            Whether to connect objects across time

        Returns
        -------
        object_id_field : xarray.DataArray
            Field of integer IDs for each object
        None : NoneType
            Placeholder for compatibility with track_objects
        N_objects : int
            Number of objects identified
        """
        if self.unstructured_grid:
            # The resulting ID field for unstructured grid will start at 0 for each time-slice,
            # which differs from structured grid where IDs are unique across time.

            if time_connectivity:  # pragma: no cover
                raise ConfigurationError(
                    "Time connectivity not supported for unstructured grids",
                    details="Automatic time connectivity computation requires regular grids",
                    suggestions=[
                        "Set time_connectivity=False for unstructured data",
                        "Manually specify connectivity if needed",
                    ],
                )

            # Use Union-Find (Disjoint Set Union) clustering for unstructured grid
            def cluster_true_values(arr: NDArray[np.bool_], neighbours_int: NDArray[np.int32]) -> NDArray[np.int32]:
                """Cluster connected True values in binary data on unstructured grid."""
                t, n = arr.shape
                labels = np.full((t, n), -1, dtype=np.int32)

                for i in range(t):
                    # Get indices of True values
                    true_indices = np.where(arr[i])[0].astype(np.int32)
                    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(true_indices)}

                    # Find connected components
                    valid_mask = (neighbours_int != -1) & arr[i][neighbours_int]
                    row_ind, col_ind = np.where(valid_mask)
                    row_ind = row_ind.astype(np.int32)
                    col_ind = col_ind.astype(np.int32)

                    # Map to compact indices for graph algorithm
                    mapped_row_ind = []
                    mapped_col_ind = []
                    for r, c in zip(neighbours_int[row_ind, col_ind], col_ind):
                        if r in mapping and c in mapping:
                            mapped_row_ind.append(mapping[r])
                            mapped_col_ind.append(mapping[c])

                    # Create graph and find connected components
                    graph = csr_matrix(
                        (
                            np.ones(len(mapped_row_ind), dtype=np.int32),
                            (mapped_row_ind, mapped_col_ind),
                        ),
                        shape=(len(true_indices), len(true_indices)),
                    )
                    _, labels_true = connected_components(csgraph=graph, directed=False, return_labels=True)
                    labels[i, true_indices] = labels_true

                return labels + 1  # Add 1 so 0 represents no object

            # Apply mask and cluster
            data_bin = data_bin.where(self.mask, other=False)

            object_id_field = xr.apply_ufunc(
                cluster_true_values,
                data_bin,
                self.neighbours_int,
                input_core_dims=[[self.xdim], ["nv", self.xdim]],
                output_core_dims=[[self.xdim]],
                output_dtypes=[np.int32],
                dask_gufunc_kwargs={
                    "output_sizes": {self.xdim: data_bin.sizes[self.xdim]},
                },
                vectorize=False,
                dask="parallelized",
            )

            # Ensure ID = 0 on invalid regions
            object_id_field = object_id_field.where(self.mask, other=0)
            object_id_field = object_id_field.persist()
            object_id_field = object_id_field.rename("ID_field")
            N_objects = 1  # Placeholder (IDs aren't unique across time)

        else:  # Structured Grid
            # Create connectivity kernel for labeling
            neighbours = np.zeros((3, 3, 3))

            if time_connectivity:
                # ID objects in 3D (i.e. space & time) -- N.B. IDs are unique across time
                neighbours[:, :, :] = 1  # +-1 in time, _and also diagonal in time_ -- i.e. edges can touch
            else:
                # ID objects only in 2D (i.e. space) -- N.B. IDs are _not_ unique across time (i.e. each time starts at 0 again)
                neighbours[1, :, :] = 1  # All 8 neighbours, but ignore time

            # Cluster & label binary data
            # Apply dask-powered ndimage & persist in memory
            if self.regional_mode:
                object_id_field, N_objects = label(
                    data_bin,
                    structure=neighbours,
                )
            else:
                object_id_field, N_objects = label(
                    data_bin,
                    structure=neighbours,
                    wrap_axes=(2,),  # Wrap in x-direction !
                )
            results = persist(object_id_field, N_objects)
            object_id_field, N_objects = results

            N_objects = N_objects.compute()

            # Convert to DataArray with same coordinates as input
            object_id_field = (
                xr.DataArray(
                    object_id_field,
                    coords=data_bin.coords,
                    dims=data_bin.dims,
                    attrs=data_bin.attrs,
                )
                .rename("ID_field")
                .astype(np.int32)
            )

        return object_id_field, None, N_objects

    def calculate_centroid(
        self,
        binary_mask: NDArray[np.bool_],
        original_centroid: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """
        Calculate object centroid, handling edge cases for periodic boundaries.

        Parameters
        ----------
        binary_mask : numpy.ndarray
            2D binary array where True indicates the object (dimensions are (y,x))
        original_centroid : tuple, optional
            (y_centroid, x_centroid) from regionprops_table

        Returns
        -------
        tuple
            (y_centroid, x_centroid)
        """
        if self.regional_mode:  # pragma: no cover
            # We don't need to adjust centroids for periodic boundaries
            return original_centroid

        # Check if object is near either edge of x dimension
        near_left_BC = np.any(binary_mask[:, :100])
        near_right_BC = np.any(binary_mask[:, -100:])

        if original_centroid is None:  # pragma: no cover
            # Calculate y centroid from scratch
            y_indices = np.nonzero(binary_mask)[0]
            y_centroid = np.mean(y_indices)
        else:
            y_centroid = original_centroid[0]

        # If object is near both edges, recalculate x-centroid to handle wrapping
        # N.B.: We calculate _near_ rather than touching, to catch the edge case where the
        # object may be split and straddling the boundary !
        if near_left_BC and near_right_BC:
            # Adjust x coordinates that are near right edge
            x_indices = np.nonzero(binary_mask)[1]
            x_indices_adj = x_indices.copy()
            right_side = x_indices > binary_mask.shape[1] // 2
            x_indices_adj[right_side] -= binary_mask.shape[1]

            x_centroid = np.mean(x_indices_adj)
            if x_centroid < 0:  # Ensure centroid is positive
                x_centroid += binary_mask.shape[1]

        elif original_centroid is None:  # pragma: no cover
            # Calculate x-centroid from scratch
            x_indices = np.nonzero(binary_mask)[1]
            x_centroid = np.mean(x_indices)

        else:
            x_centroid = original_centroid[1]

        return (y_centroid, x_centroid)

    def calculate_object_properties(self, object_id_field: xr.DataArray, properties: Optional[List[str]] = None) -> xr.Dataset:
        """
        Calculate properties of objects from ID field.

        Parameters
        ----------
        object_id_field : xarray.DataArray
            Field containing object IDs
        properties : list, optional
            List of properties to calculate (defaults to ['label', 'area'])

        Returns
        -------
        object_props : xarray.Dataset
            Dataset containing calculated properties with 'ID' dimension
        """
        # Set default properties
        if properties is None:
            properties = ["label", "area"]

        # Ensure 'label' is included
        if "label" not in properties:
            properties = ["label"] + properties  # 'label' is actually 'ID' within regionprops

        check_centroids = "centroid" in properties

        if self.unstructured_grid:
            # Compute properties on unstructured grid

            # Convert lat/lon to radians
            lat_rad = np.radians(self.lat)
            lon_rad = np.radians(self.lon)

            # Broadcast coordinate arrays to match object_id_field shape for vectorisation
            lat_rad_broadcast, _ = xr.broadcast(lat_rad, object_id_field)
            lon_rad_broadcast, _ = xr.broadcast(lon_rad, object_id_field)
            cell_area_broadcast, _ = xr.broadcast(self.cell_area, object_id_field)

            # Calculate buffer size for IDs in chunks
            max_ID = int(object_id_field.max().compute().item()) + 1

            # Handle case where object_id_field may not have time dimension (e.g., single time slice)
            if self.timedim in object_id_field.dims:
                time_steps = object_id_field.sizes[self.timedim]
            else:
                # For single time slice, use 1 as time steps
                time_steps = 1

            ID_buffer_size = max(int(max_ID / time_steps) * 4 + 2, max_ID)

            def object_properties_chunk(
                ids: NDArray[np.int32],
                lat: NDArray[np.float32],
                lon: NDArray[np.float32],
                area: NDArray[np.float32],
                buffer_IDs: bool = True,
            ) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
                """
                Calculate object properties for a chunk of data.
                Uses vectorised operations for efficiency.
                """
                # Find valid IDs
                valid_mask = ids > 0
                ids_chunk = np.unique(ids[valid_mask])
                n_ids = len(ids_chunk)

                if n_ids == 0:
                    # No objects in this chunk
                    if buffer_IDs:
                        result = np.zeros((3, ID_buffer_size), dtype=np.float32)
                        padded_ids = np.zeros(ID_buffer_size, dtype=np.int32)
                        return result, padded_ids
                    else:  # pragma: no cover
                        result = np.zeros((3, 0), dtype=np.float32)
                        padded_ids = np.array([], dtype=np.int32)
                        return result, padded_ids

                # Map IDs to consecutive indices
                mapped_indices = np.searchsorted(ids_chunk, ids[valid_mask]).astype(np.int32)

                # Pre-allocate arrays
                areas = np.zeros(n_ids, dtype=np.float32)
                weighted_x = np.zeros(n_ids, dtype=np.float32)
                weighted_y = np.zeros(n_ids, dtype=np.float32)
                weighted_z = np.zeros(n_ids, dtype=np.float32)

                # Convert to Cartesian for centroid calculation
                cos_lat = np.cos(lat[valid_mask])
                x = cos_lat * np.cos(lon[valid_mask])
                y = cos_lat * np.sin(lon[valid_mask])
                z = np.sin(lat[valid_mask])

                # Compute areas
                valid_areas = area[valid_mask]
                np.add.at(areas, mapped_indices, valid_areas)

                # Compute weighted coordinates
                np.add.at(weighted_x, mapped_indices, valid_areas * x)
                np.add.at(weighted_y, mapped_indices, valid_areas * y)
                np.add.at(weighted_z, mapped_indices, valid_areas * z)

                # Clean intermediate arrays
                del x, y, z, cos_lat, valid_areas

                # Normalise vectors
                norm = np.sqrt(weighted_x**2 + weighted_y**2 + weighted_z**2)
                norm = np.where(norm > 0, norm, 1)  # Avoid division by zero

                weighted_x /= norm
                weighted_y /= norm
                weighted_z /= norm

                # Convert back to lat/lon
                centroid_lat = np.degrees(np.arcsin(np.clip(weighted_z, -1, 1)))
                centroid_lon = np.degrees(np.arctan2(weighted_y, weighted_x))

                # Fix longitude range to [-180, 180]
                centroid_lon = np.where(
                    centroid_lon > 180.0,
                    centroid_lon - 360.0,
                    np.where(centroid_lon < -180.0, centroid_lon + 360.0, centroid_lon),
                )

                assert areas.shape == (n_ids,)
                assert centroid_lat.shape == (n_ids,)
                assert centroid_lon.shape == (n_ids,)

                if buffer_IDs:
                    # Create padded output arrays
                    result = np.zeros((3, ID_buffer_size), dtype=np.float32)
                    padded_ids = np.zeros(ID_buffer_size, dtype=np.int32)

                    # Fill arrays up to n_ids
                    result[0, :n_ids] = areas
                    result[1, :n_ids] = centroid_lat
                    result[2, :n_ids] = centroid_lon
                    padded_ids[:n_ids] = ids_chunk
                else:  # pragma: no cover
                    result = np.vstack((areas, centroid_lat, centroid_lon))
                    padded_ids = ids_chunk

                return result, padded_ids

            # Process single time or multiple times
            # If time dimension doesn't exist, treat as single time slice
            if self.timedim not in object_id_field.dims or object_id_field.sizes[self.timedim] == 1:  # pragma: no cover
                props_np, ids = object_properties_chunk(
                    object_id_field.values,
                    lat_rad_broadcast.values,
                    lon_rad_broadcast.values,
                    cell_area_broadcast.values,
                    buffer_IDs=False,
                )
                props = xr.DataArray(props_np, dims=["prop", "out_id"])

            else:
                # Process in parallel
                props_buffer, ids_buffer = xr.apply_ufunc(
                    object_properties_chunk,
                    object_id_field,
                    lat_rad_broadcast,
                    lon_rad_broadcast,
                    cell_area_broadcast,
                    input_core_dims=[
                        [self.xdim],
                        [self.xdim],
                        [self.xdim],
                        [self.xdim],
                    ],
                    output_core_dims=[["prop", "out_id"], ["out_id"]],
                    output_dtypes=[np.float32, np.int32],
                    dask_gufunc_kwargs={"output_sizes": {"prop": 3, "out_id": ID_buffer_size}},
                    vectorize=True,
                    dask="parallelized",
                )
                results = persist(props_buffer, ids_buffer)
                props_buffer, ids_buffer = results
                ids_buffer = ids_buffer.compute().values.reshape(-1)

                # Get valid IDs (non-zero)
                valid_ids_mask = ids_buffer > 0

                # Check if we have any valid IDs before stacking
                if np.any(valid_ids_mask):
                    ids = ids_buffer[valid_ids_mask]
                    props = props_buffer.stack(combined=(self.timedim, "out_id")).isel(combined=valid_ids_mask)
                else:  # pragma: no cover
                    # No valid IDs found
                    ids = np.array([], dtype=np.int32)
                    props = xr.DataArray(np.zeros((3, 0), dtype=np.float32), dims=["prop", "out_id"])

            # Create object properties dataset
            if len(ids) > 0:
                object_props = (
                    xr.Dataset(
                        {
                            "area": ("out_id", props.isel(prop=0).data),
                            "centroid-0": ("out_id", props.isel(prop=1).data),
                            "centroid-1": ("out_id", props.isel(prop=2).data),
                        },
                        coords={"ID": ("out_id", ids)},
                    )
                    .set_index(out_id="ID")
                    .rename({"out_id": "ID"})
                )
            else:  # pragma: no cover
                # Create empty dataset with correct structure
                object_props = xr.Dataset(
                    {
                        "area": ("ID", []),
                        "centroid-0": ("ID", []),
                        "centroid-1": ("ID", []),
                    },
                    coords={"ID": []},
                )

        else:
            # Structured grid approach
            # N.B.: These operations are simply done on a pixel grid
            #       i.e. with no cartesian conversion
            #       (therefore, polar regions are doubly biased)

            # Define function to calculate properties for each chunk
            def object_properties_chunk(
                ids: NDArray[np.int32],
            ) -> Dict[str, List[Union[int, float]]]:
                """Calculate object properties for a chunk of data."""
                # Use regionprops_table for standard properties
                props_slice = regionprops_table(ids, properties=properties)

                # Handle centroid calculation for objects that wrap around edges
                if check_centroids and not self.regional_mode and len(props_slice["label"]) > 0:
                    # Get original centroids
                    centroids = list(zip(props_slice["centroid-0"], props_slice["centroid-1"]))
                    centroids_wrapped = []

                    # Process each object
                    for ID_idx, ID in enumerate(props_slice["label"]):
                        binary_mask = ids == ID
                        centroids_wrapped.append(self.calculate_centroid(binary_mask, centroids[ID_idx]))

                    # Update centroid values
                    props_slice["centroid-0"] = [c[0] for c in centroids_wrapped]
                    props_slice["centroid-1"] = [c[1] for c in centroids_wrapped]

                return props_slice

            # Process single time or multiple times
            # If time dimension doesn't exist, treat as single time slice
            if self.timedim not in object_id_field.dims or object_id_field.sizes[self.timedim] == 1:
                object_props = object_properties_chunk(object_id_field.values)
                object_props = xr.Dataset({key: (["ID"], value) for key, value in object_props.items()})
            else:
                # Run in parallel
                object_props = xr.apply_ufunc(
                    object_properties_chunk,
                    object_id_field,
                    input_core_dims=[[self.ydim, self.xdim]],
                    output_core_dims=[[]],
                    output_dtypes=[object],
                    vectorize=True,
                    dask="parallelized",
                )

                # Concatenate and convert to dataset
                object_props = xr.concat(
                    [xr.Dataset({key: (["ID"], value) for key, value in item.items()}) for item in object_props.values],
                    dim="ID",
                )

            # Set ID as coordinate
            object_props = object_props.set_index(ID="label")

        # Combine centroid components into a single variable
        if "centroid" in properties and "centroid-0" in object_props and "centroid-1" in object_props:
            object_props["centroid"] = xr.concat(
                [object_props["centroid-0"], object_props["centroid-1"]],
                dim="component",
            )
            object_props = object_props.drop_vars(["centroid-0", "centroid-1"])

        return object_props

    # ============================
    # Overlap and Tracking Methods
    # ============================

    def check_overlap_slice(self, ids_t0: NDArray[np.int32], ids_next: NDArray[np.int32]) -> NDArray[Union[np.float32, np.int32]]:
        """
        Find overlapping objects between two consecutive time slices.

        Parameters
        ----------
        ids_t0 : numpy.ndarray
            Object IDs at current time
        ids_next : numpy.ndarray
            Object IDs at next time

        Returns
        -------
        numpy.ndarray
            Array of shape (n_overlaps, 3) with [id_t0, id_next, overlap_area]
        """
        # Create masks for valid IDs
        mask_t0 = ids_t0 > 0
        mask_next = ids_next > 0

        # Only process cells where both times have valid IDs
        combined_mask = mask_t0 & mask_next

        if not np.any(combined_mask):
            return np.empty((0, 3), dtype=np.float32 if self.unstructured_grid else np.int32)

        # Extract the overlapping points
        ids_t0_valid = ids_t0[combined_mask]
        ids_next_valid = ids_next[combined_mask]

        # Create a unique identifier for each pair
        # This is faster than using np.unique with axis=1
        max_id = max(ids_t0.max(), ids_next.max() + 1).astype(np.int64)
        pair_ids = ids_t0_valid.astype(np.int64) * max_id + ids_next_valid.astype(np.int64)

        if self.unstructured_grid:
            # Get unique pairs and their inverse indices
            unique_pairs, inverse_indices = np.unique(pair_ids, return_inverse=True)
            inverse_indices = inverse_indices.astype(np.int32)  # Ensure int32 for serialisation

            # Sum areas for overlapping cells
            areas_valid = self.cell_area.values[combined_mask]
            areas = np.zeros(len(unique_pairs), dtype=np.float32)
            np.add.at(areas, inverse_indices, areas_valid)
        else:
            # Get unique pairs and their counts (pixel counts)
            unique_pairs, areas = np.unique(pair_ids, return_counts=True)
            areas = areas.astype(np.int32)

        # Convert back to original ID pairs
        id_t0 = (unique_pairs // max_id).astype(np.int32)
        id_next = (unique_pairs % max_id).astype(np.int32)

        # Stack results
        result = np.column_stack((id_t0, id_next, areas))

        return result

    def find_overlapping_objects(self, object_id_field: xr.DataArray) -> NDArray[Union[np.float32, np.int32]]:
        """
        Find all overlapping objects across time.

        Parameters
        ----------
        object_id_field : xarray.DataArray
            Field containing object IDs

        Returns
        -------
        overlap_objects_list_unique_filtered : (N x 3) numpy.ndarray
            Array of object ID pairs that overlap across time, with overlap area
            The object in the first column precedes the second column in time.
            The third column contains:
                * For structured grid: number of overlapping pixels (int32)
                * For unstructured grid: total overlapping area in m^2 (float32)
        """
        # Check just for overlap with next time slice.
        #  Keep a running list of all object IDs that overlap
        object_id_field_next = object_id_field.shift({self.timedim: -1}, fill_value=0)

        # Calculate overlaps in parallel
        input_dims = [self.xdim] if self.unstructured_grid else [self.ydim, self.xdim]
        overlap_object_pairs_list = xr.apply_ufunc(
            self.check_overlap_slice,
            object_id_field,
            object_id_field_next,
            input_core_dims=[input_dims, input_dims],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[object],
        ).persist()

        # Concatenate all pairs from different chunks
        all_pairs_with_areas = np.concatenate(overlap_object_pairs_list.values)

        # Get unique pairs and their indices
        unique_pairs, inverse_indices = np.unique(all_pairs_with_areas[:, :2], axis=0, return_inverse=True)
        inverse_indices = inverse_indices.astype(np.int32)  # Ensure int32 for serialisation

        # Sum the overlap areas using the inverse indices
        output_dtype = np.float32 if self.unstructured_grid else np.int32
        total_summed_areas = np.zeros(len(unique_pairs), dtype=output_dtype)
        np.add.at(total_summed_areas, inverse_indices, all_pairs_with_areas[:, 2])

        # Stack the pairs with their summed areas
        overlap_objects_list_unique = np.column_stack((unique_pairs, total_summed_areas))

        return overlap_objects_list_unique

    def enforce_overlap_threshold(
        self,
        overlap_objects_list: NDArray[Union[np.float32, np.int32]],
        object_props: xr.Dataset,
    ) -> NDArray[Union[np.float32, np.int32]]:
        """
        Filter object pairs based on overlap threshold.

        Parameters
        ----------
        overlap_objects_list : (N x 3) numpy.ndarray
            Array of object ID pairs with overlap area
        object_props : xarray.Dataset
            Object properties including area

        Returns
        -------
        overlap_objects_list_filtered : (M x 3) numpy.ndarray
            Filtered array of object ID pairs that meet the overlap threshold
        """
        if len(overlap_objects_list) == 0:
            return np.empty((0, 3), dtype=np.float32 if self.unstructured_grid else np.int32)

        # Filter out overlaps where either ID doesn't exist in object_props
        existing_ids = set(object_props.ID.values)
        valid_mask = np.array([(overlap[0] in existing_ids) and (overlap[1] in existing_ids) for overlap in overlap_objects_list])

        if not np.any(valid_mask):
            return np.empty((0, 3), dtype=np.float32 if self.unstructured_grid else np.int32)

        valid_overlaps = overlap_objects_list[valid_mask]

        # Calculate overlap fractions
        areas_0 = object_props["area"].sel(ID=valid_overlaps[:, 0]).values
        areas_1 = object_props["area"].sel(ID=valid_overlaps[:, 1]).values
        min_areas = np.minimum(areas_0, areas_1)
        overlap_fractions = valid_overlaps[:, 2].astype(float) / min_areas

        if np.any(overlap_fractions > 1.0):
            logger.warning(f"Found {np.sum(overlap_fractions > 1.0)} overlap fractions > 1.0")
            logger.warning(f"Max overlap fraction: {overlap_fractions.max()}")

        # Filter by threshold
        threshold_mask = overlap_fractions >= self.overlap_threshold
        overlap_objects_list_filtered = valid_overlaps[threshold_mask]

        return overlap_objects_list_filtered

    def consolidate_object_ids(
        self, data_t_minus_2: xr.DataArray, data_t_minus_1: xr.DataArray, object_props: xr.Dataset, timestep: int
    ) -> Tuple[xr.DataArray, xr.Dataset]:
        """
        Consolidate object IDs between t-2 and t-1 to ensure consistent tracking.

        This identifies objects at t-1 that are actually continuations of objects
        from t-2 (but got different IDs due to partitioning) and renames them
        to maintain consistent IDs across timesteps.

        Parameters
        ----------
        data_t_minus_2 : xr.DataArray
            Object field at timestep t-2
        data_t_minus_1 : xr.DataArray
            Object field at timestep t-1 (will be modified)
        object_props : xr.Dataset
            Object properties dataset (will be modified)
        timestep : int
            Current timestep number for logging purposes

        Returns
        -------
        data_t_minus_1_consolidated : xr.DataArray
            Updated t-1 field with consolidated IDs
        object_props_updated : xr.Dataset
            Updated object properties with merged/deleted objects

        Notes
        -----
        - Uses self.overlap_threshold for determining consolidation eligibility
        - Updates object properties by recalculating for consolidated objects
        - Removes redundant child objects from object_props
        """
        # Find overlaps between t-2 and t-1
        backward_overlaps = self.check_overlap_slice(data_t_minus_2.values, data_t_minus_1.values)
        if len(backward_overlaps) == 0:
            return data_t_minus_1, object_props

        backward_overlaps = self.enforce_overlap_threshold(backward_overlaps, object_props)
        if len(backward_overlaps) == 0:  # pragma: no cover
            return data_t_minus_1, object_props

        # Find parent IDs that connect to multiple children (partition boundary jumps)
        parent_ids, parent_counts = np.unique(backward_overlaps[:, 0], return_counts=True)
        splitting_parents = parent_ids[parent_counts > 1]

        if len(splitting_parents) == 0:
            return data_t_minus_1, object_props

        # Track ID mappings for logging
        id_mappings = {}  # child_id -> parent_id

        for parent_id in splitting_parents:
            # Skip if parent doesn't exist in properties
            if parent_id not in object_props.ID.values:
                continue

            # Get all children for this parent
            child_mask = backward_overlaps[:, 0] == parent_id
            children_for_parent = backward_overlaps[child_mask, 1].astype(int)

            # Consolidate all children to use first child_id
            if len(children_for_parent) > 1:
                first_child_id = int(children_for_parent[0])

                # Skip if first child doesn't exist in properties
                if first_child_id not in object_props.ID.values:
                    continue

                # Rename all other children to first_child_id
                for child_id in children_for_parent[1:]:
                    child_id = int(child_id)
                    # Skip if child doesn't exist in properties
                    if child_id not in object_props.ID.values:
                        continue

                    # Rename child_id to first_child_id in data_t_minus_1
                    data_t_minus_1 = data_t_minus_1.where(data_t_minus_1 != child_id, first_child_id)

                    # Remove redundant child_id from object_props
                    if child_id in object_props.ID:
                        object_props = object_props.drop_sel(ID=child_id)

                    # Track the mapping
                    id_mappings[child_id] = first_child_id

                # Recalculate properties for the consolidated object
                consolidated_mask = data_t_minus_1 == first_child_id
                if consolidated_mask.any():
                    # Create temporary field with only this object for property calculation
                    temp_field = xr.where(consolidated_mask, first_child_id, 0)
                    consolidated_props = self.calculate_object_properties(temp_field, properties=["area", "centroid"])

                    if first_child_id in consolidated_props.ID:
                        # Update first child properties with consolidated values
                        for var_name in ["area", "centroid"]:
                            if var_name in consolidated_props:
                                object_props[var_name].loc[{"ID": first_child_id}] = consolidated_props[var_name].sel(
                                    ID=first_child_id
                                )

        return data_t_minus_1, object_props

    def compute_id_time_dict(
        self,
        da: xr.DataArray,
        child_objects: Union[List[int], NDArray[np.int32]],
        max_objects: int,
        all_objects: bool = True,
    ) -> Dict[int, int]:
        """
        Generate lookup table mapping object IDs to their time index.

        Parameters
        ----------
        da : xarray.DataArray
            Field of object IDs
        child_objects : list or array
            Object IDs to include in the dictionary
        max_objects : int
            Maximum number of objects
        all_objects : bool, default=True
            Whether to process all objects or just child_objects

        Returns
        -------
        time_index_map : dict
            Dictionary mapping object IDs to time indices
        """
        # Estimate max objects per time
        est_objects_per_time_max = int(max_objects / da[self.timedim].shape[0] * 100)

        def unique_pad(x: NDArray[np.int32]) -> NDArray[np.int32]:
            """Extract unique values and pad to fixed size."""
            uniq = np.unique(x)
            result = np.zeros(est_objects_per_time_max, dtype=x.dtype)  # Pad output to maximum size
            result[: len(uniq)] = uniq
            return result

        # Get unique IDs for each time slice
        input_dims = [self.xdim] if self.unstructured_grid else [self.ydim, self.xdim]
        unique_ids_by_time = xr.apply_ufunc(
            unique_pad,
            da,
            input_core_dims=[input_dims],
            output_core_dims=[["unique_values"]],
            dask="parallelized",
            vectorize=True,
            dask_gufunc_kwargs={"output_sizes": {"unique_values": est_objects_per_time_max}},
        )

        # Set up IDs to search for
        if not all_objects:
            # Just search for the specified child objects
            search_ids = xr.DataArray(child_objects, dims=["child_id"], coords={"child_id": child_objects})
        else:
            # Search for all possible IDs
            search_ids = xr.DataArray(
                np.arange(max_objects, dtype=np.int32),
                dims=["child_id"],
                coords={"child_id": np.arange(max_objects, dtype=np.int32)},
            ).chunk(
                {"child_id": 10000}
            )  # Chunk for better parallelism

        # Find the first time index where each ID appears
        time_indices = (
            (unique_ids_by_time == search_ids).any(dim=["unique_values"]).argmax(dim=self.timedim).compute().astype(np.int32)
        )

        # Convert to dictionary for fast lookup
        time_index_map = {int(id_val): int(idx.values) for id_val, idx in zip(time_indices.child_id, time_indices)}

        return time_index_map

    # ============================
    # Event Tracking Methods
    # ============================

    def track_objects(self, data_bin: xr.DataArray) -> Tuple[xr.Dataset, xr.Dataset, int]:
        """
        Track objects through time to form events.

        This is the main tracking method that handles splitting and merging of objects.

        Parameters
        ----------
        data_bin : xarray.DataArray
            Preprocessed binary data:  Field of globally unique integer IDs of each element in connected regions.
            ID = 0 indicates no object.

        Returns
        -------
        split_merged_events_ds : xarray.Dataset
            Dataset containing tracked events
        merge_events : xarray.Dataset
            Dataset with merge information
        N_events : int
            Final number of events
        """
        # Identify objects at each time step
        object_id_field, _, _ = self.identify_objects(data_bin, time_connectivity=False)
        object_id_field = object_id_field.persist()
        del data_bin
        logger.info("Finished object identification")

        # For unstructured grid, make objects unique across time
        if self.unstructured_grid:
            cumsum_ids = (object_id_field.max(dim=self.xdim)).cumsum(self.timedim).shift({self.timedim: 1}, fill_value=0)
            object_id_field = xr.where(object_id_field > 0, object_id_field + cumsum_ids, 0)
            object_id_field = self.refresh_dask_graph(object_id_field)
            logger.info(f"Finished assigning c. {cumsum_ids.max().compute().values} globally unique object IDs")

        # Calculate object properties
        object_props = self.calculate_object_properties(object_id_field, properties=["area", "centroid"])
        object_props = object_props.persist()
        wait(object_props)
        logger.info("Finished calculating object properties")

        # Apply splitting & merging logic
        #  This is the most intricate step due to non-trivial loop-wise dependencies
        #  In v2.0_unstruct, this loop has been painstakingly parallelised
        split_and_merge = self.split_and_merge_objects_parallel if self.unstructured_grid else self.split_and_merge_objects
        object_id_field, object_props, overlap_objects_list, merge_events = split_and_merge(object_id_field, object_props)
        logger.info("Finished splitting and merging objects")

        # Persist results (This helps avoid block-wise task fusion run_spec issues with dask)
        results = persist(object_id_field, object_props, overlap_objects_list, merge_events)
        object_id_field, object_props, overlap_objects_list, merge_events = results

        # Cluster & rename objects to get globally unique event IDs
        split_merged_events_ds = self.cluster_rename_objects_and_props(
            object_id_field, object_props, overlap_objects_list, merge_events
        )

        # Rechunk final output
        chunk_dict = {
            self.timedim: self.timechunks,
            "ID": -1,
            "component": -1,
            "sibling_ID": -1,
            self.xdim: -1,
        }
        if not self.unstructured_grid:
            chunk_dict[self.ydim] = -1

        split_merged_events_ds = split_merged_events_ds.chunk(chunk_dict)  # .persist()
        logger.info("Finished clustering and renaming objects into coherent consistent events")

        # Count final number of events
        N_events = split_merged_events_ds.ID_field.max().compute().data

        return split_merged_events_ds, merge_events, N_events

    def cluster_rename_objects_and_props(
        self,
        object_id_field_unique: xr.DataArray,
        object_props: xr.Dataset,
        overlap_objects_list: NDArray[np.int32],
        merge_events: xr.Dataset,
    ) -> xr.Dataset:
        """
        Cluster the object pairs and relabel to determine final event IDs.

        Parameters
        ----------
        object_id_field_unique : xarray.DataArray
            Field of unique object IDs. IDs must not be repeated across time.
        object_props : xarray.Dataset
            Properties of each object that also need to be relabeled.
        overlap_objects_list : (N x 2) numpy.ndarray
            Array of object ID pairs that indicate which objects are in the same event.
            The object in the first column precedes the second column in time.
        merge_events : xarray.Dataset
            Information about merge events

        Returns
        -------
        split_merged_events_ds : xarray.Dataset
            Dataset with relabeled events and their properties. ID = 0 indicates no object.
        """
        # Cluster the overlap_pairs into groups of IDs that are actually the same object
        # Get IDs from overlap pairs
        # Step 1: Find all IDs that actually exist in the data
        max_ID = int(object_id_field_unique.max().compute().values.item())

        # Get unique IDs from overlap list
        if len(overlap_objects_list) > 0:
            overlap_ids = np.unique(overlap_objects_list[:, :2].flatten())
            overlap_ids = overlap_ids[overlap_ids > 0]  # Remove 0 (background)
        else:
            overlap_ids = np.array([], dtype=np.int32)  # pragma: no cover

        # Get unique IDs from object_id_field
        field_ids = np.unique(object_id_field_unique.compute().values)
        field_ids = field_ids[field_ids > 0]  # Remove 0 (background)

        # Combine and get all valid IDs
        all_valid_ids = np.unique(np.concatenate([overlap_ids, field_ids]))

        logger.info(f"Found {len(all_valid_ids)} valid object IDs (out of max ID {max_ID})")

        # Step 2: Create dense mapping: original_ID -> dense_index
        # This ensures continuous indices for connected_components
        original_to_dense = {int(original_id): dense_idx for dense_idx, original_id in enumerate(all_valid_ids)}
        dense_to_original = {dense_idx: int(original_id) for original_id, dense_idx in original_to_dense.items()}

        n_valid = len(all_valid_ids)

        # Step 3: Convert overlap pairs to dense indices
        if len(overlap_objects_list) > 0:
            # Map to dense indices
            overlap_pairs_dense = np.array(
                [
                    [original_to_dense[int(pair[0])], original_to_dense[int(pair[1])]]
                    for pair in overlap_objects_list
                    if int(pair[0]) in original_to_dense and int(pair[1]) in original_to_dense
                ]
            )

            # Create sparse graph with dense indices
            row_indices, col_indices = overlap_pairs_dense.T
            data = np.ones(len(overlap_pairs_dense), dtype=np.bool_)
            graph = csr_matrix((data, (row_indices, col_indices)), shape=(n_valid, n_valid), dtype=np.bool_)
        else:
            graph = csr_matrix((n_valid, n_valid), dtype=np.bool_)  # pragma: no cover

        # Step 4: Solve for connected components (on dense graph)
        num_components, component_IDs_dense = connected_components(csgraph=graph, directed=False, return_labels=True)

        logger.info(f"Identified {num_components} connected components (events)")

        # Step 5: Create lookup from original IDs to event IDs
        # Event IDs will be continuous: 1, 2, 3, ... num_components
        original_to_event = {}
        for dense_idx, event_id in enumerate(component_IDs_dense):
            original_id = dense_to_original[dense_idx]
            original_to_event[original_id] = event_id + 1  # +1 so events start at 1, not 0

        # Step 6: Create full lookup array for fast remapping
        ID_to_cluster_index_array = np.full(max_ID + 1, 0, dtype=np.int32)  # 0 = background
        for original_id, event_id in original_to_event.items():
            ID_to_cluster_index_array[original_id] = np.int32(event_id)

        # Convert to DataArray for apply_ufunc
        #  N.B.: **Need to pass da into apply_ufunc, otherwise it doesn't manage the memory correctly
        #          with large shared-mem numpy arrays**
        ID_to_cluster_index_da = xr.DataArray(
            ID_to_cluster_index_array,
            dims="ID",
            coords={"ID": np.arange(max_ID + 1, dtype=np.int32)},
        )

        def map_IDs_to_indices(block: NDArray[np.int32], ID_to_cluster_index_array: NDArray[np.int32]) -> NDArray[np.int32]:
            """Map original IDs to cluster indices."""
            mask = block > 0
            new_block = np.zeros_like(block, dtype=np.int32)
            new_block[mask] = ID_to_cluster_index_array[block[mask]]
            return new_block

        # Apply the mapping
        input_dims = [self.xdim] if self.unstructured_grid else [self.ydim, self.xdim]
        split_merged_relabeled_object_id_field = xr.apply_ufunc(
            map_IDs_to_indices,
            object_id_field_unique,
            ID_to_cluster_index_da,
            input_core_dims=[input_dims, ["ID"]],
            output_core_dims=[input_dims],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int32],
        ).persist()

        # Relabel the object_props to match the new IDs (and add time dimension)

        max_new_ID = num_components + 1  # New IDs range from 0 to max_new_ID
        new_ids = np.arange(1, max_new_ID + 1, dtype=np.int32)

        # Create new object_props dataset - use dimension coordinate for time data
        time_coord_data = object_id_field_unique.coords[self.timedim].data
        object_props_extended = xr.Dataset(coords={"ID": new_ids, self.timecoord: (self.timedim, time_coord_data)})

        # Create mapping from new IDs to the original IDs _at the corresponding time_
        valid_new_ids = split_merged_relabeled_object_id_field > 0
        original_ids_field = object_id_field_unique.where(valid_new_ids)
        new_ids_field = split_merged_relabeled_object_id_field.where(valid_new_ids)

        if not self.unstructured_grid:
            original_ids_field = original_ids_field.stack(z=(self.ydim, self.xdim), create_index=False)
            new_ids_field = new_ids_field.stack(z=(self.ydim, self.xdim), create_index=False)

        new_id_to_idx = {id_val: idx for idx, id_val in enumerate(new_ids)}

        def process_timestep(orig_ids: NDArray[np.int32], new_ids_t: NDArray[np.int32]) -> NDArray[np.int32]:
            """Process a single timestep to create ID mapping."""
            result = np.zeros(len(new_id_to_idx), dtype=np.int32)

            valid_mask = new_ids_t > 0

            # Get valid points for this timestep
            if not valid_mask.any():
                return result

            orig_valid = orig_ids[valid_mask]
            new_valid = new_ids_t[valid_mask]

            if len(orig_valid) == 0:
                return result

            unique_pairs = np.unique(np.column_stack((orig_valid, new_valid)), axis=0)

            # Create mapping
            for orig_id, new_id in unique_pairs:
                if new_id in new_id_to_idx:
                    result[new_id_to_idx[new_id]] = orig_id

            return result

        # Process in parallel
        input_dim = [self.xdim] if self.unstructured_grid else ["z"]
        global_id_mapping = (
            xr.apply_ufunc(
                process_timestep,
                original_ids_field,
                new_ids_field,
                input_core_dims=[input_dim, input_dim],
                output_core_dims=[["ID"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.int32],
                dask_gufunc_kwargs={"output_sizes": {"ID": len(new_ids)}},
            )
            .assign_coords(ID=new_ids)
            .compute()
        )

        # Store original ID mapping
        object_props_extended["global_ID"] = global_id_mapping
        # Post-condition: Now, e.g. global_id_mapping.sel(ID=10)
        #    --> Given the new ID (10), returns corresponding original_id at every time

        # Transfer all properties from original object_props
        dummy = object_props.isel(ID=0) * np.nan  # Add vale of ID = 0 to this coordinate ID
        object_props = xr.concat([dummy.assign_coords(ID=0), object_props], dim="ID")

        for var_name in object_props.data_vars:
            # Filter global_id_mapping to only include IDs that exist in object_props
            existing_ids = set(object_props.ID.values)
            valid_mapping_mask = global_id_mapping.isin(existing_ids)

            # Only select existing IDs
            valid_global_mapping = global_id_mapping.where(valid_mapping_mask, drop=True)

            if len(valid_global_mapping.ID) == 0:
                # No valid IDs - create empty result
                temp = object_props[var_name].isel(ID=slice(0, 0))
            else:
                temp = (
                    object_props[var_name]
                    .sel(ID=valid_global_mapping.rename({"ID": "new_id"}))
                    .drop_vars("ID")
                    .rename({"new_id": "ID"})
                )

            if var_name == "ID":
                temp = temp.astype(np.int32)
            else:
                temp = temp.astype(np.float32)

            object_props_extended[var_name] = temp

        # Map the merge_events using the old IDs to be from dimensions (merge_ID, parent_idx)
        #     --> new merge_ledger with dimensions (time, ID, sibling_ID)
        # i.e. for each merge_ID --> merge_parent_IDs   gives the old IDs  --> map to new ID using ID_to_cluster_index_da
        #                   --> merge_time

        old_parent_IDs = xr.where(merge_events.parent_IDs > 0, merge_events.parent_IDs, 0)
        new_IDs_parents = ID_to_cluster_index_da.sel(ID=old_parent_IDs)

        # Replace the coordinate merge_ID in new_IDs_parents with merge_time.
        #    merge_events.merge_time gives merge_time for each merge_ID
        new_IDs_parents_t = (
            new_IDs_parents.assign_coords({"merge_time": merge_events.merge_time})
            .drop_vars("ID")
            .swap_dims({"merge_ID": "merge_time"})
            .persist()
        )

        # Map new_IDs_parents_t into a new data array with dimensions time, ID, and sibling_ID
        merge_ledger = (
            xr.full_like(global_id_mapping, fill_value=-1)
            .chunk({self.timedim: self.timechunks})
            .expand_dims({"sibling_ID": new_IDs_parents_t.parent_idx.shape[0]})
            .copy()
        )

        # Wrapper for processing/mapping mergers in parallel
        def process_time_group(
            time_block: xr.DataArray,
            IDs_data: NDArray[np.int32],
            IDs_coords: Dict[str, Any],
        ) -> xr.DataArray:
            """Process all mergers for a single block of timesteps."""
            result = xr.full_like(time_block, -1)

            # Get unique times in this block
            # time_block might not have the coordinate, so get it from the dimension index
            if self.timecoord in time_block.coords:
                unique_times = np.unique(time_block.coords[self.timecoord])
            else:
                # Fall back to using the dimension index
                unique_times = np.unique(time_block[self.timedim])

            for time_val in unique_times:
                # Get IDs for this time
                time_mask = IDs_coords["merge_time"] == time_val
                if not np.any(time_mask):
                    continue

                IDs_at_time = IDs_data[time_mask]

                # Handle single merger case
                if IDs_at_time.ndim == 1:
                    valid_mask = IDs_at_time > 0
                    if np.any(valid_mask):
                        # Create expanded array for sibling_ID dimension
                        expanded_IDs = np.broadcast_to(IDs_at_time, (len(time_block.sibling_ID), len(IDs_at_time)))
                        result.loc[{self.timedim: time_val, "ID": IDs_at_time[valid_mask]}] = expanded_IDs[:, valid_mask]

                # Handle multiple mergers case
                else:
                    for merger_IDs in IDs_at_time:
                        valid_mask = merger_IDs > 0
                        if np.any(valid_mask):
                            expanded_IDs = np.broadcast_to(
                                merger_IDs,
                                (len(time_block.sibling_ID), len(merger_IDs)),
                            )
                            result.loc[{self.timedim: time_val, "ID": merger_IDs[valid_mask]}] = expanded_IDs[:, valid_mask]

            return result

        # Map blocks in parallel
        merge_ledger = xr.map_blocks(
            process_time_group,
            merge_ledger,
            args=(new_IDs_parents_t.values, new_IDs_parents_t.coords),
            template=merge_ledger,
        )

        # Format merge ledger
        merge_ledger = merge_ledger.rename("merge_ledger").transpose(self.timedim, "ID", "sibling_ID").persist()

        # Add start and end time indices for each ID
        valid_presence = object_props_extended["global_ID"] > 0  # i.e. where there is valid data

        object_props_extended["presence"] = valid_presence
        object_props_extended["time_start"] = valid_presence[self.timecoord][
            valid_presence.argmax(dim=self.timedim).astype(np.int32)
        ]
        object_props_extended["time_end"] = valid_presence[self.timecoord][
            ((valid_presence.sizes[self.timedim] - 1) - (valid_presence[::-1]).argmax(dim=self.timedim)).astype(np.int32)
        ]

        # Recompute area & centroid (now that the IDs have been consolidated & merged & made continuous)
        if "area" in object_props_extended.data_vars or "centroid" in object_props_extended.data_vars:
            logger.info("Recalculating area and centroid properties for potentially disjoint events...")

            def calculate_area_centroid_for_slice(
                slice_data: NDArray[np.int32],
                cell_areas_slice: NDArray[np.float32],
                present_mask: NDArray[np.bool_],
                all_event_ids: NDArray[np.int32],
                lat_vals: NDArray[np.float32],
                lon_vals: NDArray[np.float32],
                is_unstructured: bool,
                regional_mode: bool,
            ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
                """
                Calculate area and area-weighted centroid for IDs present at this timestep.
                Returns three arrays with full ID dimension (NaN for absent IDs).

                Parameters
                ----------
                slice_data : array
                    Spatial field of event IDs for this timestep
                cell_areas_slice : array
                    Spatial field of cell areas
                present_mask : array
                    1D boolean array indicating which IDs are present (length = n_IDs)
                all_event_ids : array
                    All event IDs (length = n_IDs)
                """
                n_ids = len(all_event_ids)

                # Initialise output arrays with NaN
                areas = np.full(n_ids, np.nan, dtype=np.float32)
                centroid_lats = np.full(n_ids, np.nan, dtype=np.float32)
                centroid_lons = np.full(n_ids, np.nan, dtype=np.float32)

                # Get indices of IDs that are present at this timestep
                present_indices = np.where(present_mask)[0]

                if len(present_indices) == 0:
                    return areas, centroid_lats, centroid_lons

                if is_unstructured:
                    # Unstructured grid: area-weighted centroid using spherical geometry

                    # Convert to radians for Cartesian calculation
                    lat_rad = np.radians(lat_vals)
                    lon_rad = np.radians(lon_vals)

                    # Process each present ID
                    for id_idx in present_indices:
                        event_id = all_event_ids[id_idx]
                        mask = slice_data == event_id

                        if not np.any(mask):
                            continue  # pragma: no cover

                        # Calculate physical area
                        areas_masked = cell_areas_slice[mask]
                        total_area = np.sum(areas_masked)
                        areas[id_idx] = total_area

                        # Calculate area-weighted centroid using spherical geometry
                        cos_lat = np.cos(lat_rad[mask])
                        x = cos_lat * np.cos(lon_rad[mask])
                        y = cos_lat * np.sin(lon_rad[mask])
                        z = np.sin(lat_rad[mask])

                        # Weighted average in Cartesian coordinates
                        weighted_x = np.sum(areas_masked * x)
                        weighted_y = np.sum(areas_masked * y)
                        weighted_z = np.sum(areas_masked * z)

                        # Normalise
                        norm = np.sqrt(weighted_x**2 + weighted_y**2 + weighted_z**2)
                        if norm > 0:
                            weighted_x /= norm
                            weighted_y /= norm
                            weighted_z /= norm

                        # Convert back to lat/lon
                        centroid_lat = np.degrees(np.arcsin(np.clip(weighted_z, -1, 1)))
                        centroid_lon = np.degrees(np.arctan2(weighted_y, weighted_x))

                        # Fix longitude range to [-180, 180]
                        if centroid_lon > 180:
                            centroid_lon -= 360  # pragma: no cover
                        elif centroid_lon < -180:
                            centroid_lon += 360  # pragma: no cover

                        centroid_lats[id_idx] = centroid_lat
                        centroid_lons[id_idx] = centroid_lon
                else:
                    # Structured grid: area-weighted centroid with periodic boundary handling
                    ny, nx = slice_data.shape

                    # Process each present ID
                    for id_idx in present_indices:
                        event_id = all_event_ids[id_idx]

                        # Get binary mask for this event
                        binary_mask = slice_data == event_id

                        if not np.any(binary_mask):
                            continue  # pragma: no cover

                        # Get indices where object exists
                        y_indices, x_indices = np.nonzero(binary_mask)

                        # Get cell areas for these indices
                        pixel_areas = cell_areas_slice[binary_mask]
                        total_area = np.sum(pixel_areas)
                        areas[id_idx] = total_area

                        # Calculate area-weighted y centroid (latitude)
                        centroid_y_pix = np.sum(y_indices * pixel_areas) / total_area

                        # Calculate area-weighted x centroid (longitude) - handle wrapping if needed
                        if not regional_mode:
                            # Check if object is near both edges (wrapping around periodic boundary)
                            near_left = np.any(x_indices < 100)
                            near_right = np.any(x_indices >= nx - 100)

                            if near_left and near_right:
                                # Object wraps around - adjust coordinates
                                x_adjusted = x_indices.copy().astype(np.float64)
                                right_side = x_indices > nx / 2
                                x_adjusted[right_side] -= nx

                                # Area-weighted mean with adjusted coordinates
                                centroid_x_pix = np.sum(x_adjusted * pixel_areas) / total_area

                                # Ensure centroid is positive
                                if centroid_x_pix < 0:
                                    centroid_x_pix += nx
                            else:
                                # No wrapping - standard area-weighted calculation
                                centroid_x_pix = np.sum(x_indices * pixel_areas) / total_area
                        else:
                            # Regional mode - no wrapping, area-weighted
                            centroid_x_pix = np.sum(x_indices * pixel_areas) / total_area

                        # Convert pixel indices to coordinate values
                        centroid_lat = np.interp(centroid_y_pix, np.arange(len(lat_vals)), lat_vals)
                        centroid_lon = np.interp(centroid_x_pix, np.arange(len(lon_vals)), lon_vals)

                        centroid_lats[id_idx] = centroid_lat
                        centroid_lons[id_idx] = centroid_lon

                return areas, centroid_lats, centroid_lons

            # Prepare spatial dimensions
            spatial_dims = [self.xdim] if self.unstructured_grid else [self.ydim, self.xdim]

            # Ensure cell_area has correct dimensions for apply_ufunc
            if not self.unstructured_grid and self.cell_area.ndim == 1:
                # Broadcast 1D latitude-dependent cell areas to 2D (lat, lon)
                template = split_merged_relabeled_object_id_field.isel({self.timedim: 0}, drop=True)
                cell_area_broadcast, _ = xr.broadcast(self.cell_area, template)
            else:
                cell_area_broadcast = self.cell_area

            # Apply calculation in parallel across time slices
            logger.info("Computing area and centroid properties in parallel...")
            areas_computed, centroid_lats_computed, centroid_lons_computed = xr.apply_ufunc(
                calculate_area_centroid_for_slice,
                split_merged_relabeled_object_id_field,
                cell_area_broadcast,  # Broadcasted to match spatial dimensions
                object_props_extended.presence,  # Boolean mask of which IDs are present at each time
                object_props_extended.ID,
                self.lat,  # Latitude coordinate values
                self.lon,  # Longitude coordinate values
                kwargs={"is_unstructured": self.unstructured_grid, "regional_mode": self.regional_mode},
                input_core_dims=[
                    spatial_dims,
                    spatial_dims,
                    ["ID"],
                    ["ID"],
                    [self.ydim] if not self.unstructured_grid else [self.xdim],
                    [self.xdim],
                ],
                output_core_dims=[["ID"], ["ID"], ["ID"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.float32, np.float32, np.float32],
            )

            results = persist(areas_computed, centroid_lats_computed, centroid_lons_computed)
            areas_computed, centroid_lats_computed, centroid_lons_computed = results

            # Update area with proper dimension ordering (time, ID)
            object_props_extended["area"] = areas_computed.transpose(self.timedim, "ID")

            # Combine lat/lon centroids along component dimension
            new_centroid = xr.concat([centroid_lats_computed, centroid_lons_computed], dim="component")
            new_centroid = new_centroid.assign_coords(component=[0, 1])

            # Update centroid with proper dimension ordering (component, time, ID)
            object_props_extended["centroid"] = new_centroid.transpose("component", self.timedim, "ID")

            logger.info("Property recalculation complete.")

        # Combine all components into final dataset
        split_merged_relabeled_events_ds = xr.merge(
            [
                split_merged_relabeled_object_id_field.rename("ID_field"),
                object_props_extended,
                merge_ledger,
            ]
        )

        # Remove the last ID -- it is all 0s (because we added an extra padding one above)
        return split_merged_relabeled_events_ds.isel(ID=slice(0, -1))

    # ============================
    # Splitting and Merging Methods
    # ============================

    def split_and_merge_objects(
        self, object_id_field_unique: xr.DataArray, object_props: xr.Dataset
    ) -> Tuple[xr.DataArray, xr.Dataset, NDArray[np.int32], xr.Dataset]:
        """
        Implement object splitting and merging logic.

        This identifies and processes cases where objects split or merge over time,
        creating new object IDs as needed.

        Parameters
        ----------
        object_id_field_unique : xarray.DataArray
            Field of unique object IDs. IDs are required to be monotonically increasing with time.
        object_props : xarray.Dataset
            Properties of each object

        Returns
        -------
        tuple
            (object_id_field, object_props, overlap_objects_list, merge_events)
        """
        # Find overlapping objects
        overlap_objects_list = self.find_overlapping_objects(
            object_id_field_unique
        )  # List object pairs that overlap by at least overlap_threshold percent
        overlap_objects_list = self.enforce_overlap_threshold(overlap_objects_list, object_props)
        logger.info("Finished finding overlapping objects")

        # Initialise merge tracking lists
        merge_times = []  # When the merge occurred
        merge_child_ids = []  # Resulting child ID
        merge_parent_ids = []  # List of parent IDs that merged
        merge_areas = []  # Areas of overlap
        next_new_id = int(object_props.ID.max().item()) + 1  # Start new IDs after highest existing ID

        Nx = object_id_field_unique[self.xdim].size
        object_id_field_unique = object_id_field_unique.persist()
        updated_chunks = []

        # Process each time chunk with timestep-first approach
        chunk_boundaries = np.cumsum([0] + list(object_id_field_unique.chunks[0]))

        for chunk_idx in range(len(object_id_field_unique.chunks[0])):
            # Extract and load an entire chunk into memory
            chunk_start = chunk_boundaries[chunk_idx]
            chunk_end = chunk_boundaries[chunk_idx + 1]
            # Ensure we don't exceed array bounds
            chunk_end = min(chunk_end, object_id_field_unique.sizes[self.timedim])

            chunk_data = object_id_field_unique.isel({self.timedim: slice(chunk_start, chunk_end)}).compute()

            # Process each timestep within chunk sequentially
            for relative_t in range(chunk_data.sizes[self.timedim]):
                absolute_t = chunk_start + relative_t

                # Get data slices for current timestep
                data_t = chunk_data.isel({self.timedim: relative_t})

                # Get previous timesteps for consolidation and partitioning
                if relative_t > 1:  # Need both t-1 and t-2 for consolidation
                    data_t_minus_2 = chunk_data.isel({self.timedim: relative_t - 2})
                    data_t_minus_1 = chunk_data.isel({self.timedim: relative_t - 1})
                elif relative_t == 1:  # t-1 is in current chunk, t-2 might be in previous chunk
                    data_t_minus_1 = chunk_data.isel({self.timedim: 0})  # relative_t - 1 = 0
                    if updated_chunks:
                        _, _, last_chunk_data = updated_chunks[-1]
                        data_t_minus_2 = last_chunk_data[-1]  # Last timestep from previous chunk
                    else:
                        data_t_minus_2 = xr.full_like(data_t, 0)
                else:  # relative_t == 0, get both from previous chunk if available
                    if updated_chunks:
                        _, _, last_chunk_data = updated_chunks[-1]
                        if len(last_chunk_data) >= 2:
                            data_t_minus_2 = last_chunk_data[-2]
                            data_t_minus_1 = last_chunk_data[-1]
                        elif len(last_chunk_data) == 1:
                            data_t_minus_2 = xr.full_like(data_t, 0)
                            data_t_minus_1 = last_chunk_data[-1]
                        else:
                            data_t_minus_2 = xr.full_like(data_t, 0)
                            data_t_minus_1 = xr.full_like(data_t, 0)
                    else:
                        data_t_minus_2 = xr.full_like(data_t, 0)
                        data_t_minus_1 = xr.full_like(data_t, 0)

                # ID Consolidation of objects at t-1
                if relative_t > 0:  # Only consolidate if we have meaningful t-1 and t-2
                    data_t_minus_1, object_props = self.consolidate_object_ids(
                        data_t_minus_2, data_t_minus_1, object_props, absolute_t - 1
                    )

                    # Update the chunk with consolidated data whenever t-1 is in current chunk
                    chunk_data[{self.timedim: relative_t - 1}] = data_t_minus_1

                # Normal overlap detection and partitioning (now with consolidated IDs)

                # Calculate overlaps for this timestep
                #   Here, parents are at previous time=t-1 (LHS), children are at current time=t (RHS)
                timestep_overlaps = self.check_overlap_slice(data_t_minus_1.values, data_t.values)
                timestep_overlaps = self.enforce_overlap_threshold(timestep_overlaps, object_props)

                # Iterative processing within timestep=t until convergence
                #  Only modifies data_t, which contains the children to be partitioned/relabelled
                timestep_converged = False
                iteration = 0

                while not timestep_converged and iteration < 10:  # Prevent infinite loops
                    # Find merging objects for current timestep
                    unique_children, children_counts = np.unique(timestep_overlaps[:, 1], return_counts=True)
                    merging_children = unique_children[children_counts > 1]

                    if len(merging_children) == 0:
                        timestep_converged = True
                        continue

                    # Process all merging objects in this timestep
                    #   Parents exist in this timestep, but
                    for child_id in merging_children:

                        # Get mask of child object
                        child_mask_2d = (data_t == child_id).values

                        # Find all pairs involving this child
                        child_mask = timestep_overlaps[:, 1] == child_id
                        child_where = np.where(timestep_overlaps[:, 1] == child_id)[0].astype(np.int32)
                        merge_group = timestep_overlaps[child_mask]

                        # Get parent objects (LHS) that overlap with this child object
                        parent_ids = merge_group[:, 0]
                        num_parents = len(parent_ids)

                        # Create new IDs for the other half of the child object & record in the merge ledger
                        new_object_id = np.arange(next_new_id, next_new_id + (num_parents - 1), dtype=np.int32)
                        next_new_id += num_parents - 1

                        # Replace the 2nd+ child in the overlap objects list with the new child ID
                        timestep_overlaps[child_where[1:], 1] = new_object_id
                        child_ids = np.concatenate((np.array([child_id]), new_object_id))

                        # Record merge event - extract time value using dimension name
                        merge_times.append(data_t.coords[self.timedim].values)
                        merge_child_ids.append(child_ids)
                        merge_parent_ids.append(parent_ids)
                        merge_areas.append(timestep_overlaps[child_mask, 2])

                        # Relabel the Original Child Object ID Field to account for the New ID:
                        # Get parent centroids for partitioning
                        parent_centroids = object_props.sel(ID=parent_ids).centroid.values.T

                        # Partition the child object based on parent associations
                        if self.nn_partitioning:
                            # Nearest-neighbor partitioning
                            # --> For every (Original) Child Cell in the ID Field, Find the closest (t-1) Parent _Cell_
                            if self.unstructured_grid:
                                # Prepare parent masks
                                parent_masks = np.zeros((len(parent_ids), data_t_minus_1.shape[0]), dtype=bool)
                                for idx, parent_id in enumerate(parent_ids):
                                    parent_masks[idx] = (data_t_minus_1 == parent_id).values

                                # Calculate maximum search distance
                                max_area = np.max(object_props.sel(ID=parent_ids).area.values) / self.mean_cell_area
                                max_distance = int(np.sqrt(max_area) * 2.0)

                                # Use optimised unstructured partitioning
                                new_labels = partition_nn_unstructured(
                                    child_mask_2d,
                                    parent_masks,
                                    child_ids,
                                    parent_centroids,
                                    self.neighbours_int.values,
                                    self.lat.values,  # Need to pass these as NumPy arrays for JIT compatibility
                                    self.lon.values,
                                    max_distance=max(max_distance, 20) * 2,  # Set minimum threshold, in cells
                                )
                            else:
                                # Prepare parent masks for structured grid
                                parent_masks = np.zeros(
                                    (
                                        len(parent_ids),
                                        data_t_minus_1.shape[0],
                                        data_t_minus_1.shape[1],
                                    ),
                                    dtype=bool,
                                )
                                for idx, parent_id in enumerate(parent_ids):
                                    parent_masks[idx] = (data_t_minus_1 == parent_id).values

                                # Calculate maximum search distance
                                max_area = np.max(object_props.sel(ID=parent_ids).area.values)
                                max_distance = int(np.sqrt(max_area) * 3.0)  # Use 3x the max blob radius

                                # Use optimised structured grid partitioning
                                new_labels = partition_nn_grid(
                                    child_mask_2d,
                                    parent_masks,
                                    child_ids,
                                    parent_centroids,
                                    Nx,
                                    max_distance=max(max_distance, 40),  # Set minimum threshold, in cells
                                    wrap=not self.regional_mode,  # Turn longitude periodic wrapping off when in regional mode
                                )

                        else:
                            # Centroid-based partitioning
                            # --> For every (Original) Child Cell in the ID Field, Find the closest (t-1) Parent _Centroid_
                            if self.unstructured_grid:
                                new_labels = partition_centroid_unstructured(
                                    child_mask_2d,
                                    parent_centroids,
                                    child_ids,
                                    self.lat.values,
                                    self.lon.values,
                                )
                            else:
                                # Calculate distances to each parent centroid
                                distances = wrapped_euclidian_distance_mask_parallel(
                                    child_mask_2d, parent_centroids, Nx, not self.regional_mode
                                )

                                # Assign based on closest parent
                                new_labels = child_ids[np.argmin(distances, axis=1).astype(np.int32)]

                        # Update values in data_t and assign the updated slice back to the chunk
                        temp = np.zeros_like(data_t)
                        temp[child_mask_2d] = new_labels
                        data_t = data_t.where(~child_mask_2d, temp)
                        chunk_data[{self.timedim: relative_t}] = data_t

                        # Update the Properties of the N Children Objects
                        new_child_props = self.calculate_object_properties(data_t, properties=["area", "centroid"])

                        # Update the object_props DataArray:  (but first, check if the original children still exists)
                        if child_id in new_child_props.ID:
                            # Update existing entry
                            object_props.loc[{"ID": child_id}] = new_child_props.sel(ID=child_id)
                        else:
                            # Delete child_id: The object has split/morphed such that it doesn't get a partition of this child...
                            object_props = object_props.drop_sel(
                                ID=child_id
                            )  # N.B.: This means that the IDs are no longer continuous...
                            logger.info(f"Deleted child_id {child_id} because parents have split/morphed")

                        # Add the properties for the N-1 other new child ID
                        new_object_ids_still = new_child_props.ID.where(new_child_props.ID.isin(new_object_id), drop=True).ID
                        object_props = xr.concat(
                            [object_props, new_child_props.sel(ID=new_object_ids_still)],
                            dim="ID",
                        )

                        missing_ids = set(new_object_id) - set(new_object_ids_still.values)
                        if len(missing_ids) > 0:
                            logger.warning(
                                f"Missing newly created child_ids {missing_ids} "
                                f"because parents have split/morphed in the meantime..."
                            )

                    # After processing all merging objects in this iteration
                    # Recalculate overlaps to check for newly viable merges
                    timestep_overlaps = self.check_overlap_slice(data_t_minus_1.values, data_t.values)
                    timestep_overlaps = self.enforce_overlap_threshold(timestep_overlaps, object_props)
                    iteration += 1

                if iteration == 10:
                    logger.warning(f"Resolving mergers at timestep {absolute_t} did not converge after 10 iterations")

            # End-of-chunk consolidation: consolidate the last timestep if chunk has multiple timesteps
            if chunk_data.sizes[self.timedim] >= 2:

                # Get last and second-to-last timesteps
                last_t_data = chunk_data.isel({self.timedim: -1})
                second_last_t_data = chunk_data.isel({self.timedim: -2})

                # Consolidate last timestep using second-to-last as reference
                consolidated_last, object_props = self.consolidate_object_ids(
                    second_last_t_data, last_t_data, object_props, chunk_end - 1
                )

                # Update the last timestep in chunk
                chunk_data[{self.timedim: -1}] = consolidated_last

            # Store the processed chunk
            updated_chunks.append(
                (
                    chunk_start,
                    chunk_end,
                    chunk_data[: (chunk_end - chunk_start)],
                )
            )

            if chunk_idx % 10 == 0:
                logger.info(f"Processing splitting and merging in chunk {chunk_idx} of {len(object_id_field_unique.chunks[0])}")

                # Periodically update main array to manage memory
                if len(updated_chunks) > 1:  # Keep the last chunk for potential reference
                    for start, end, processed_chunk_data in updated_chunks[:-1]:
                        object_id_field_unique[{self.timedim: slice(start, end)}] = processed_chunk_data
                    updated_chunks = updated_chunks[-1:]  # Keep only the last chunk
                    object_id_field_unique = object_id_field_unique.persist()

        # Apply final chunk updates
        for start, end, processed_chunk_data in updated_chunks:
            object_id_field_unique[{self.timedim: slice(start, end)}] = processed_chunk_data
        object_id_field_unique = object_id_field_unique.persist()

        # Recompute final overlapping objects
        overlap_objects_list = self.find_overlapping_objects(object_id_field_unique)
        overlap_objects_list = self.enforce_overlap_threshold(overlap_objects_list, object_props)
        logger.info("Finished final overlapping objects search")

        # Check for duplicate children (multiple parents per child)
        if len(overlap_objects_list) > 0:
            child_ids = overlap_objects_list[:, 1]  # RHS column (children)
            unique_children, child_counts = np.unique(child_ids, return_counts=True)

            # Find children with multiple parents
            duplicate_children = unique_children[child_counts > 1]

            # Enhanced validation with comprehensive spatial and temporal information
            if len(duplicate_children) > 0:
                logger.warning(f"There is {len(duplicate_children)} potentially problematic children:")

                # Log problematic child IDs (time info not available at this stage)
                logger.warning(f"Children IDs: {duplicate_children[:10].tolist()}")

                # Detailed analysis of each problematic child
                for child_id in duplicate_children[:5]:  # Limit to first 5 for readability
                    # Find all parent-child relationships for this child
                    child_relationships = overlap_objects_list[overlap_objects_list[:, 1] == child_id]
                    parent_ids = child_relationships[:, 0]
                    overlap_areas = child_relationships[:, 2]

                    logger.warning(f"\n--- Details for child ID {child_id} ---")
                    logger.warning(f"Number of parents: {len(parent_ids)}")
                    logger.warning(f"Parent IDs: {parent_ids.tolist()}")
                    logger.warning(f"Raw overlap areas: {overlap_areas.tolist()}")

                    # Get child object properties if available
                    try:
                        if child_id in object_props.ID.values:
                            child_area = object_props.sel(ID=child_id).area.values.item()
                            child_centroid = object_props.sel(ID=child_id).centroid.values

                            logger.warning(f"Child total area: {child_area}")
                            logger.warning(f"Child centroid: {child_centroid}")

                            # Calculate overlap fractions for each parent
                            overlap_fractions = []
                            parent_areas = []
                            for i, parent_id in enumerate(parent_ids):
                                if parent_id in object_props.ID.values:
                                    parent_area = object_props.sel(ID=parent_id).area.values.item()
                                    parent_areas.append(parent_area)

                                    # Calculate overlap fraction based on smaller object
                                    min_area = min(child_area, parent_area)
                                    overlap_fraction = float(overlap_areas[i]) / min_area
                                    overlap_fractions.append(overlap_fraction)
                                else:
                                    parent_areas.append("N/A")
                                    overlap_fractions.append("N/A")

                            logger.warning(f"Parent areas: {parent_areas}")
                            logger.warning(f"Overlap fractions: {overlap_fractions}")

                            # Check for suspicious patterns
                            total_overlap_area = sum(overlap_areas)
                            logger.warning(f"Sum of overlap areas: {total_overlap_area}")
                            logger.warning(f"Sum/Child area ratio: {total_overlap_area/child_area:.3f}")

                            # Flag potential issues
                            valid_fractions = [f for f in overlap_fractions if isinstance(f, (int, float))]
                            if valid_fractions and max(valid_fractions) > 1.0:
                                logger.warning(f"WARNING: Overlap fraction > 1.0 detected (max: {max(valid_fractions):.3f})")
                            if total_overlap_area > child_area * 1.1:  # Allow 10% tolerance
                                logger.warning(
                                    f"WARNING: Total overlap exceeds child area by {(total_overlap_area/child_area - 1)*100:.1f}%"
                                )

                        else:
                            logger.warning(f"Child ID {child_id} not found in object_props")

                    except Exception as e:
                        logger.warning(f"Error analysing child ID {child_id}: {str(e)}")

                    # Try to find timestep information by checking where this child appears
                    try:
                        child_timesteps = []
                        for t_idx in range(object_id_field_unique.sizes[self.timedim]):
                            time_slice = object_id_field_unique.isel({self.timedim: t_idx})
                            if (time_slice == child_id).any():
                                time_coord = time_slice.coords[self.timedim].values
                                child_timesteps.append((t_idx, time_coord))

                        if child_timesteps:
                            logger.warning(f"Child appears at timesteps: {child_timesteps}")
                        else:
                            logger.warning("Child timestep information not found")

                    except Exception as e:
                        logger.warning(f"Error finding timestep for child ID {child_id}: {str(e)}")

                    logger.warning("--- End detailed analysis ---\n")

                # Log summary information as warnings instead of raising error
                logger.warning("=" * 80)
                logger.warning("Tracker Warning: Multiple parents for single child detected after splitting/merging")
                logger.warning(f"Details: {len(duplicate_children)} children have multiple parents")
                logger.warning("Note: This is likely due to consolidation of IDs after splitting/merging")
                logger.warning("      and still is the correct behaviour (as per the tracking overlap logic")
                logger.warning("      applied to disjoint objects that will be grouped together.)")
                logger.warning("=" * 80)
            else:
                logger.info(f"Validation passed: All {len(unique_children)} children have unique parents")
        else:
            logger.info("No overlaps found - validation skipped")

        # Process merge events into a dataset
        # Handle case where there are no merge events
        if merge_parent_ids and merge_child_ids:
            max_parents = max(len(ids) for ids in merge_parent_ids)
            max_children = max(len(ids) for ids in merge_child_ids)
        else:
            max_parents = 1  # Default minimum size
            max_children = 1

        # Convert lists to padded numpy arrays
        parent_ids_array = np.full((len(merge_parent_ids), max_parents), -1, dtype=np.int32)
        child_ids_array = np.full((len(merge_child_ids), max_children), -1, dtype=np.int32)
        overlap_areas_array = np.full((len(merge_areas), max_parents), -1, dtype=np.int32)

        for i, parents in enumerate(merge_parent_ids):
            parent_ids_array[i, : len(parents)] = parents

        for i, children in enumerate(merge_child_ids):
            child_ids_array[i, : len(children)] = children

        for i, areas in enumerate(merge_areas):
            overlap_areas_array[i, : len(areas)] = areas

        # Create merge events dataset
        merge_events = xr.Dataset(
            {
                "parent_IDs": (("merge_ID", "parent_idx"), parent_ids_array),
                "child_IDs": (("merge_ID", "child_idx"), child_ids_array),
                "overlap_areas": (("merge_ID", "parent_idx"), overlap_areas_array),
                "merge_time": ("merge_ID", merge_times),
                "n_parents": (
                    "merge_ID",
                    np.array([len(p) for p in merge_parent_ids], dtype=np.int8),
                ),
                "n_children": (
                    "merge_ID",
                    np.array([len(c) for c in merge_child_ids], dtype=np.int8),
                ),
            },
            attrs={"fill_value": -1},
        )

        object_props = object_props.persist()

        return (
            object_id_field_unique,
            object_props,
            overlap_objects_list[:, :2],  # Only return first 2 columns (ID pairs)
            merge_events,
        )

    def split_and_merge_objects_parallel(
        self, object_id_field_unique: xr.DataArray, object_props: xr.Dataset
    ) -> Tuple[xr.DataArray, xr.Dataset, NDArray[np.int32], xr.Dataset]:
        """
        Optimised parallel implementation of object splitting and merging.

        This version is specifically designed for unstructured grids with more efficient
        memory handling and better parallelism than the standard split_and_merge_objects
        method. It processes data in chunks, handles merging events, and efficiently
        updates object IDs.

        Parameters
        ----------
        object_id_field_unique : xarray.DataArray
            Field of unique object IDs
        object_props : xarray.Dataset
            Properties of each object

        Returns
        -------
        tuple
            (object_id_field, object_props, overlap_objects_list, merge_events)
        """
        # Constants for memory allocation
        MAX_MERGES = 20  # Maximum number of merges per timestep
        MAX_PARENTS = 10  # Maximum number of parents per merge
        MAX_CHILDREN = MAX_PARENTS

        def process_chunk(
            chunk_data_m1_full: NDArray[np.int32],
            chunk_data_p1_full: NDArray[np.int32],
            merging_objects: NDArray[np.int64],
            next_id_start: NDArray[np.int64],
            lat: NDArray[np.float32],
            lon: NDArray[np.float32],
            area: NDArray[np.float32],
            neighbours_int: NDArray[np.int32],
        ) -> Tuple[
            NDArray[np.int32],  # merge_child_ids
            NDArray[np.int32],  # merge_parent_ids
            NDArray[np.float32],  # merge_areas
            NDArray[np.int16],  # merge_counts
            NDArray[np.bool_],  # has_merge
            NDArray[np.uint8],  # updates_array
            NDArray[np.int32],  # updates_ids
            NDArray[np.int32],  # final_merging_objects
        ]:
            """
            Process a single chunk of merging objects.

            This function handles the complex batch processing of splitting and merging objects
            across timesteps within a single chunk. It finds overlapping objects, determines
            parent-child relationships, and creates new IDs as needed.

            Parameters
            ----------
            chunk_data_m1_full : numpy.ndarray
                Data from previous timestep (t-1) and current timestep (t)
            chunk_data_p1_full : numpy.ndarray
                Data from next timestep (t+1)
            merging_objects : (n_time, max_merges) numpy.ndarray
                IDs of objects to process
            next_id_start : (n_time, max_merges) numpy.ndarray
                Starting ID values for new objects
            lat, lon : numpy.ndarray
                Latitude/longitude arrays
            area : numpy.ndarray
                Cell area array
            neighbours_int : numpy.ndarray
                Neighbor connectivity array

            Returns
            -------
            tuple
                Contains merge events, object updates, and newly created objects
            """
            # Fix Broadcasted dimensions of inputs:
            #    Remove extra dimension if present while preserving time chunks
            #    N.B.: This is a weird artefact/choice of xarray apply_ufunc broadcasting...
            #           (i.e. 'nv' dimension gets injected into all the other arrays!)

            chunk_data_m1 = chunk_data_m1_full.squeeze()[0].astype(np.int32).copy()
            chunk_data = chunk_data_m1_full.squeeze()[1].astype(np.int32).copy()
            del chunk_data_m1_full  # Free memory immediately
            chunk_data_p1 = chunk_data_p1_full.astype(np.int32).copy()
            # Remove any singleton dimensions except time and space
            while chunk_data_p1.ndim > 2:
                chunk_data_p1 = chunk_data_p1.squeeze(axis=-1)
            del chunk_data_p1_full

            # Extract and prepare input arrays
            lat = lat.squeeze().astype(np.float32)
            lon = lon.squeeze().astype(np.float32)
            area = area.squeeze().astype(np.float32)
            next_id_start = next_id_start.squeeze()

            # Handle neighbours_int with correct dimensions (nv, ncells)
            neighbours_int = neighbours_int.squeeze()
            if neighbours_int.shape[1] != lat.shape[0]:
                neighbours_int = neighbours_int.T

            # Handle multiple merging objects - ensure proper dimensionality
            merging_objects = merging_objects.squeeze()
            if merging_objects.ndim == 1:
                merging_objects = merging_objects[:, None]  # Add dimension for max_merges

            # Pre-convert lat/lon to Cartesian coordinates for efficiency
            x = (np.cos(np.radians(lat)) * np.cos(np.radians(lon))).astype(np.float32)
            y = (np.cos(np.radians(lat)) * np.sin(np.radians(lon))).astype(np.float32)
            z = np.sin(np.radians(lat)).astype(np.float32)

            # Pre-allocate output arrays
            n_time = chunk_data_p1.shape[0]
            n_points = chunk_data_p1.shape[1]

            merge_child_ids = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.int32)
            merge_parent_ids = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.int32)
            merge_areas = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.float32)
            merge_counts = np.zeros(n_time, dtype=np.int16)  # Number of merges per timestep

            updates_array = np.full((n_time, n_points), 255, dtype=np.uint8)
            updates_ids = np.full((n_time, 255), -1, dtype=np.int32)
            has_merge = np.zeros(n_time, dtype=np.bool_)

            # Prepare merging objects list for each timestep
            merging_objects_list = [list(merging_objects[i][merging_objects[i] > 0]) for i in range(merging_objects.shape[0])]
            final_merging_objects = np.full((n_time, MAX_MERGES), -1, dtype=np.int32)
            final_merge_count = 0

            # Process each timestep
            data_p1 = []
            for t in range(n_time):
                next_new_id = next_id_start[t]  # Use the offset for this timestep

                # Get current time slice data
                if t == 0:
                    data_m1 = chunk_data_m1
                    data_t = chunk_data
                    del chunk_data_m1, chunk_data  # Free memory
                else:
                    data_m1 = data_t  # Previous data_t becomes data_m1
                    data_t = data_p1  # Previous data_p1 becomes data_t
                data_p1 = chunk_data_p1[t]

                # Process each merging object at this timestep
                while merging_objects_list[t]:
                    child_id = merging_objects_list[t].pop(0)

                    # Get child mask and identify overlapping parents
                    child_mask = data_t == child_id

                    # Find parent objects that overlap with this child
                    potential_parents = np.unique(data_m1[child_mask])
                    parent_iterator = 0
                    parent_masks_uint = np.full(n_points, 255, dtype=np.uint8)
                    parent_centroids = np.full((MAX_PARENTS, 2), -1.0e10, dtype=np.float32)
                    parent_ids = np.full(MAX_PARENTS, -1, dtype=np.int32)
                    parent_areas = np.zeros(MAX_PARENTS, dtype=np.float32)
                    overlap_areas = np.zeros(MAX_PARENTS, dtype=np.float32)
                    n_parents = 0

                    # Find all unique parent IDs with significant overlap
                    for parent_id in potential_parents[potential_parents > 0]:
                        if n_parents >= MAX_PARENTS:  # pragma: no cover
                            raise TrackingError(
                                "Too many parent objects for tracking",
                                details=f"Child {child_id} at timestep {t} has {n_parents} parents (limit: {MAX_PARENTS})",
                                suggestions=[
                                    "Increase overlap_threshold to reduce fragmentation",
                                    "Apply stronger area filtering",
                                ],
                                context={
                                    "child_id": child_id,
                                    "timestep": t,
                                    "n_parents": n_parents,
                                    "limit": MAX_PARENTS,
                                },
                            )

                        parent_mask = data_m1 == parent_id
                        if np.any(parent_mask & child_mask):
                            # Calculate overlap area and check if it's large enough
                            area_0 = area[parent_mask].sum()  # Parent area
                            area_1 = area[child_mask].sum()  # Child area
                            min_area = np.minimum(area_0, area_1)
                            overlap_area = area[parent_mask & child_mask].sum()

                            # Skip if overlap is below threshold
                            if overlap_area / min_area < self.overlap_threshold:
                                continue

                            # Record parent information
                            parent_masks_uint[parent_mask] = parent_iterator
                            parent_ids[n_parents] = parent_id
                            overlap_areas[n_parents] = overlap_area

                            # Calculate area-weighted centroid for this parent
                            mask_area = area[parent_mask]
                            weighted_coords = np.array(
                                [
                                    np.sum(mask_area * x[parent_mask]),
                                    np.sum(mask_area * y[parent_mask]),
                                    np.sum(mask_area * z[parent_mask]),
                                ],
                                dtype=np.float32,
                            )

                            norm = np.sqrt(np.sum(weighted_coords * weighted_coords))

                            # Convert back to lat/lon
                            parent_centroids[n_parents, 0] = np.degrees(np.arcsin(weighted_coords[2] / norm))
                            parent_centroids[n_parents, 1] = np.degrees(np.arctan2(weighted_coords[1], weighted_coords[0]))

                            # Fix longitude range to [-180, 180]
                            if parent_centroids[n_parents, 1] > 180:
                                parent_centroids[n_parents, 1] -= 360
                            elif parent_centroids[n_parents, 1] < -180:
                                parent_centroids[n_parents, 1] += 360

                            parent_areas[n_parents] = area_0
                            parent_iterator += 1
                            n_parents += 1

                    # Need at least 2 parents for merging
                    if n_parents < 2:
                        continue

                    # Create new IDs for each partition
                    new_child_ids = np.arange(next_new_id, next_new_id + (n_parents - 1), dtype=np.int32)
                    child_ids = np.concatenate((np.array([child_id]), new_child_ids))

                    # Record merge event
                    curr_merge_idx = merge_counts[t]
                    if curr_merge_idx > MAX_MERGES:  # pragma: no cover
                        raise TrackingError(
                            "Too many merge operations",
                            details=f"Timestep {t} requires {curr_merge_idx} merges (limit: {MAX_MERGES})",
                            suggestions=[
                                "Increase area_filter_quartile to reduce small objects",
                                "Consider adjusting tracking parameters",
                            ],
                            context={
                                "timestep": t,
                                "merge_count": curr_merge_idx,
                                "limit": MAX_MERGES,
                            },
                        )

                    merge_child_ids[t, curr_merge_idx, :n_parents] = child_ids[:n_parents]
                    merge_parent_ids[t, curr_merge_idx, :n_parents] = parent_ids[:n_parents]
                    merge_areas[t, curr_merge_idx, :n_parents] = overlap_areas[:n_parents]
                    merge_counts[t] += 1
                    has_merge[t] = True

                    # Partition the child object based on parent associations
                    if self.nn_partitioning:
                        # Estimate maximum search distance based on object size
                        max_area = parent_areas.max() / self.mean_cell_area
                        max_distance = int(np.sqrt(max_area) * 2.0)

                        # Use optimised nearest-neighbor partitioning
                        new_labels_uint = partition_nn_unstructured_optimised(
                            child_mask.copy(),
                            parent_masks_uint.copy(),
                            parent_centroids,
                            neighbours_int.copy(),
                            lat,
                            lon,
                            max_distance=max(max_distance, 20) * 2,
                        )
                        # Returned 'new_labels_uint' is just the index of the child_ids
                        new_labels = child_ids[new_labels_uint]

                        # Help garbage collection
                        new_labels_uint = None

                    else:
                        # Use centroid-based partitioning
                        new_labels = partition_centroid_unstructured(child_mask, parent_centroids, child_ids, lat, lon)

                    # Update slice data for subsequent merging in process_chunk
                    data_t[child_mask] = new_labels

                    # Record which cells get which new IDs for later updates
                    spatial_indices_all = np.where(child_mask)[0].astype(np.int32)
                    child_mask = None  # Free memory
                    gc.collect()

                    # Record update information for each new ID
                    for new_id in child_ids[1:]:
                        update_idx = np.where(updates_ids[t] == -1)[0].astype(np.int32)[
                            0
                        ]  # Find next non-negative index in updates_ids
                        updates_ids[t, update_idx] = new_id
                        updates_array[t, spatial_indices_all[new_labels == new_id]] = update_idx

                    next_new_id += n_parents - 1

                    # Find all child objects in the next timestep that overlap with our newly labeled regions
                    new_merging_list = []
                    for new_id in child_ids:
                        parent_mask = data_t == new_id
                        if np.any(parent_mask):
                            area_0 = area[parent_mask].sum()
                            potential_children = np.unique(data_p1[parent_mask])

                            for potential_child in potential_children[potential_children > 0]:
                                potential_child_mask = data_p1 == potential_child
                                area_1 = area[potential_child_mask].sum()
                                min_area = min(area_0, area_1)
                                overlap_area = area[parent_mask & potential_child_mask].sum()

                                if overlap_area / min_area > self.overlap_threshold:
                                    new_merging_list.append(potential_child)

                    # Add newly found merging objects to processing queue
                    if t < n_time - 1:
                        # Add to next timestep in this chunk
                        for new_object_id in new_merging_list:
                            if new_object_id not in merging_objects_list[t + 1]:
                                merging_objects_list[t + 1].append(new_object_id)
                    else:
                        # Record for next chunk
                        for new_object_id in new_merging_list:
                            if final_merge_count > MAX_MERGES:  # pragma: no cover
                                raise TrackingError(
                                    "Excessive merge operations detected",
                                    details=f"Final merge count {final_merge_count} exceeds limit {MAX_MERGES} at timestep {t}",
                                    suggestions=[
                                        "Increase area_filter_quartile to reduce small objects",
                                        "Consider adjusting tracking parameters",
                                    ],
                                    context={
                                        "timestep": t,
                                        "final_merge_count": final_merge_count,
                                        "limit": MAX_MERGES,
                                    },
                                )

                            if not np.any(final_merging_objects[t][:final_merge_count] == new_object_id):
                                final_merging_objects[t][final_merge_count] = new_object_id
                                final_merge_count += 1

            return (
                merge_child_ids,
                merge_parent_ids,
                merge_areas,
                merge_counts,
                has_merge,
                updates_array,
                updates_ids,
                final_merging_objects,
            )

        def update_object_id_field_inplace(
            object_id_field: xr.DataArray,
            id_lookup: Dict[int, int],
            updates_array: xr.DataArray,
            updates_ids: xr.DataArray,
            has_merge: xr.DataArray,
        ) -> xr.DataArray:  # pragma: no cover
            """
            Update the object field with chunk results using xarray operations.

            This is memory efficient as it avoids creating full copies of the object_id_field.

            Parameters
            ----------
            object_id_field : xarray.DataArray
                The full object field to update
            id_lookup : dict
                Dictionary mapping temporary IDs to new IDs
            updates_array : xarray.DataArray
                Array indicating which spatial indices to update
            updates_ids : xarray.DataArray
                The new IDs to assign to updated indices
            has_merge : xarray.DataArray
                Boolean indicating whether each timestep has merges

            Returns
            -------
            xarray.DataArray
                Updated object field
            """
            # Quick return if no merges to update
            if not has_merge.any():
                return object_id_field

            def update_timeslice(
                data: NDArray[np.int32],
                updates: NDArray[np.uint8],
                update_ids: NDArray[np.int32],
                lookup_values: NDArray[np.int32],
            ) -> NDArray[np.int32]:
                """Process a single timeslice."""
                # Extract valid update IDs
                valid_ids = update_ids[update_ids > -1]
                if len(valid_ids) == 0:
                    return data

                # Create result array starting with original values
                result = data.copy()

                # Apply each update
                for idx, update_id in enumerate(valid_ids):
                    mask = updates == idx
                    if mask.any():
                        result = np.where(mask, lookup_values[update_id], result)

                return result

            # Convert lookup dict to array for vectorized access
            max_id = max(id_lookup.keys()) + 1
            lookup_array = np.full(max_id, -1, dtype=np.int32)
            for temp_id, new_id in id_lookup.items():
                lookup_array[temp_id] = new_id

            # Apply updates in parallel
            result = xr.apply_ufunc(
                update_timeslice,
                object_id_field,
                updates_array,
                updates_ids,
                kwargs={"lookup_values": lookup_array},
                input_core_dims=[[self.xdim], [self.xdim], ["update_idx"]],
                output_core_dims=[[self.xdim]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.int32],
            )

            return result

        def update_object_id_field_zarr(
            self,
            object_id_field: xr.DataArray,
            id_lookup: Dict[int, int],
            updates_array: xr.DataArray,
            updates_ids: xr.DataArray,
            has_merge: xr.DataArray,
        ) -> xr.DataArray:
            """
            Update object field using a temporary zarr store for better memory efficiency.

            This approach minimises memory usage by writing changes directly to disk,
            allowing for more efficient parallel processing of large datasets.

            Parameters
            ----------
            object_id_field : xarray.DataArray
                The object field to update
            id_lookup : dict
                Dictionary mapping temporary IDs to new IDs
            updates_array : xarray.DataArray
                Array indicating which spatial indices to update
            updates_ids : xarray.DataArray
                The new IDs to assign to updated indices
            has_merge : xarray.DataArray
                Boolean indicating whether each timestep has merges

            Returns
            -------
            xarray.DataArray
                Updated object field from zarr store
            """
            # Early return if no merges to save memory
            if not bool(has_merge.any().compute().item()):
                return object_id_field

            zarr_path = f"{self.scratch_dir}/marEx_temp_field.zarr/"

            # Initialise zarr store if needed
            if not os.path.exists(zarr_path):
                object_id_field.name = "temp"
                object_id_field.to_zarr(zarr_path, mode="w")

            def update_time_chunk(ds_chunk: xr.Dataset, lookup_dict: Dict[int, int]) -> xr.DataArray:
                """Process a single chunk with optimised memory usage."""
                # Skip processing if no merges in this chunk
                needs_update = bool(ds_chunk["has_merge"].any().compute().item())
                if not needs_update:
                    return ds_chunk["object_field"]

                # Extract data from the chunk
                chunk_data = ds_chunk["object_field"]
                chunk_updates = ds_chunk["updates"]
                chunk_update_ids = ds_chunk["update_ids"]

                # Get zarr region indices
                time_idx_start = int(ds_chunk["time_indices"].values[0])
                time_idx_end = int(ds_chunk["time_indices"].values[-1]) + 1

                updated_chunk = chunk_data.copy()

                # Process each time slice in the chunk
                for t in range(chunk_data.sizes[self.timedim]):
                    # Get update information for this time
                    updates_slice = chunk_updates.isel({self.timedim: t}).values
                    update_ids_slice = chunk_update_ids.isel({self.timedim: t}).values

                    # Get valid update IDs
                    valid_mask = update_ids_slice > -1
                    if not np.any(valid_mask):
                        continue

                    valid_ids = update_ids_slice[valid_mask]

                    # Get the time slice data and apply updates
                    result_slice = updated_chunk.isel({self.timedim: t})

                    for idx, update_id in enumerate(valid_ids):
                        mask = updates_slice == idx
                        if np.any(mask):
                            new_id = lookup_dict.get(int(update_id), update_id)
                            result_slice = xr.where(mask, new_id, result_slice)

                    # Store updated slice
                    updated_chunk[t] = result_slice

                # Write the updated chunk directly to zarr
                updated_chunk.name = "temp"
                updated_chunk.to_zarr(
                    zarr_path,
                    region={self.timedim: slice(time_idx_start, time_idx_end)},
                )

                return chunk_data  # Return original data for dask graph consistency

            # Create time indices for slicing
            time_coords = object_id_field[self.timecoord].values
            time_indices = np.arange(len(time_coords), dtype=np.int32)
            time_index_da = xr.DataArray(time_indices, dims=[self.timedim], coords={self.timecoord: time_coords})

            # Create dataset with all necessary components
            ds = xr.Dataset(
                {
                    "object_field": object_id_field,
                    "updates": updates_array,
                    "update_ids": updates_ids,
                    "time_indices": time_index_da,
                    "has_merge": has_merge,
                }
            ).chunk({self.timedim: self.timechunks})

            # Process chunks in parallel
            result = xr.map_blocks(
                update_time_chunk,
                ds,
                kwargs={"lookup_dict": id_lookup},
                template=object_id_field,
            )

            # Force computation to ensure all writes complete
            result = result.persist()
            wait(result)

            # Release resources
            del result, ds, object_id_field
            gc.collect()

            # Load the updated data from zarr store
            object_id_field_new = xr.open_zarr(zarr_path, chunks={self.timedim: self.timechunks}).temp

            return object_id_field_new

        def merge_objects_parallel_iteration(
            object_id_field_unique: xr.DataArray,
            merging_objects: Set[int],
            global_id_counter: int,
        ) -> Tuple[
            xr.DataArray,  # updated_field
            Tuple[
                NDArray[np.int32],
                NDArray[np.int32],
                NDArray[np.float32],
                NDArray[np.int32],
            ],  # merge_data
            Set[int],  # new_merging_objects
            int,  # updated_counter
        ]:
            """
            Perform a single iteration of the parallel merging process.

            This function handles one complete batch of merging objects across all
            timesteps, updating object IDs and tracking merge events.

            Parameters
            ----------
            object_id_field_unique : xarray.DataArray
                Field of unique object IDs
            merging_objects : set
                Set of object IDs to process in this iteration
            global_id_counter : int
                Current counter for assigning new global IDs

            Returns
            -------
            tuple
                (updated_field, merge_data, new_merging_objects, updated_counter)
            """
            n_time = len(object_id_field_unique[self.timecoord])

            # Pre-allocate arrays for this iteration
            child_ids_iter = np.full(
                (n_time, MAX_MERGES, MAX_CHILDREN), -1, dtype=np.int32
            )  # List of child ID arrays for this time
            parent_ids_iter = np.full(
                (n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.int32
            )  # List of parent ID arrays for this time
            merge_areas_iter = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.float32)  # List of areas for this time
            merge_counts_iter = np.zeros(n_time, dtype=np.int32)

            # Prepare neighbour information
            neighbours_int = self.neighbours_int.chunk({self.xdim: -1, "nv": -1})

            logger.info(f"Processing Parallel Iteration {iteration + 1} with {len(merging_objects)} Merging Objects...")

            # Pre-compute the child_time_idx for merging_objects
            time_index_map = self.compute_id_time_dict(object_id_field_unique, list(merging_objects), global_id_counter)
            logger.debug("Finished Mapping Children to Time Indices")

            # Create uniform array of merging objects for each timestep
            max_merges = max(len([b for b in merging_objects if time_index_map.get(b, -1) == t]) for t in range(n_time))
            uniform_merging_objects_array = np.zeros((n_time, max_merges), dtype=np.int32)
            for t in range(n_time):
                objects_at_t = [b for b in merging_objects if time_index_map.get(b, -1) == t]
                if objects_at_t:  # Only fill if there are objects at this time
                    uniform_merging_objects_array[t, : len(objects_at_t)] = np.array(objects_at_t, dtype=np.int32)

            # Create DataArrays for parallel processing
            merging_objects_da = xr.DataArray(
                uniform_merging_objects_array,
                dims=[self.timedim, "merges"],
                coords={self.timecoord: object_id_field_unique[self.timecoord]},
            )

            # Calculate ID offsets for each timestep to ensure unique IDs
            next_id_offsets = np.arange(n_time, dtype=np.int64) * max_merges * self.timechunks + global_id_counter
            # N.B.: We also need to account for possibility of newly-split objects subsequently creating
            #          more than max_merges by the end of the iteration through the chunk
            # !!! This is likely the root cause of any errors such as "ID needs to be contiguous/continuous/full/unrepeated"
            next_id_offsets_da = xr.DataArray(
                next_id_offsets,
                dims=[self.timedim],
                coords={self.timecoord: object_id_field_unique[self.timecoord]},
            )

            # Create shifted arrays for time connectivity
            object_id_field_unique_p1 = object_id_field_unique.shift({self.timedim: -1}, fill_value=0)
            object_id_field_unique_m1 = object_id_field_unique.shift({self.timedim: 1}, fill_value=0)

            # Align chunks for better parallel processing
            object_id_field_unique_m1 = object_id_field_unique_m1.chunk({self.timedim: self.timechunks})
            object_id_field_unique_p1 = object_id_field_unique_p1.chunk({self.timedim: self.timechunks})
            merging_objects_da = merging_objects_da.chunk({self.timedim: self.timechunks})
            next_id_offsets_da = next_id_offsets_da.chunk({self.timedim: self.timechunks})

            # Process chunks in parallel
            results = xr.apply_ufunc(
                process_chunk,
                object_id_field_unique_m1,
                object_id_field_unique_p1,
                merging_objects_da,
                next_id_offsets_da,
                self.lat,
                self.lon,
                self.cell_area,
                neighbours_int,
                input_core_dims=[
                    [self.xdim],
                    [self.xdim],
                    ["merges"],
                    [],
                    [self.xdim],
                    [self.xdim],
                    [self.xdim],
                    ["nv", self.xdim],
                ],
                output_core_dims=[
                    ["merge", "parent"],
                    ["merge", "parent"],
                    ["merge", "parent"],
                    [],
                    [],
                    [self.xdim],
                    ["update_idx"],
                    ["merge"],
                ],
                output_dtypes=[
                    np.int32,
                    np.int32,
                    np.float32,
                    np.int16,
                    np.bool_,
                    np.uint8,
                    np.int32,
                    np.int32,
                ],
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "merge": MAX_MERGES,
                        "parent": MAX_PARENTS,
                        "update_idx": 255,
                    }
                },
                vectorize=False,
                dask="parallelized",
            )

            # Unpack and persist results
            (
                merge_child_ids,
                merge_parent_ids,
                merge_areas,
                merge_counts,
                has_merge,
                updates_array,
                updates_ids,
                final_merging_objects,
            ) = results

            results = persist(
                merge_child_ids,
                merge_parent_ids,
                merge_areas,
                merge_counts,
                has_merge,
                updates_array,
                updates_ids,
                final_merging_objects,
            )
            (
                merge_child_ids,
                merge_parent_ids,
                merge_areas,
                merge_counts,
                has_merge,
                updates_array,
                updates_ids,
                final_merging_objects,
            ) = results

            # Get time indices where merges occurred
            has_merge = has_merge.compute()
            time_indices = np.where(has_merge)[0].astype(np.int32)

            # Clean up temporary arrays to save memory
            del (
                object_id_field_unique_p1,
                object_id_field_unique_m1,
                merging_objects_da,
                next_id_offsets_da,
            )
            gc.collect()

            logger.debug("Finished Batch Processing Step")

            # ====== Global Consolidation of Data ======

            # 1. Collect all temporary IDs and create global mapping
            all_temp_ids = np.unique(merge_child_ids.where(merge_child_ids >= global_id_counter, other=0).compute().values)
            all_temp_ids = all_temp_ids[all_temp_ids > 0]  # Remove the 0

            if not len(all_temp_ids):  # If no temporary IDs exist
                id_lookup = {}
            else:
                # Create mapping from temporary to permanent IDs
                id_lookup = {
                    temp_id: np.int32(new_id)
                    for temp_id, new_id in zip(
                        all_temp_ids,
                        range(global_id_counter, global_id_counter + len(all_temp_ids)),
                    )
                }
                global_id_counter += len(all_temp_ids)

            logger.debug("Finished Consolidation Step 1: Temporary ID Mapping")

            # 2. Update object ID field with new IDs
            update_on_disk = True  # This is more memory efficient because it refreshes the dask graph every iteration

            if update_on_disk:
                object_id_field_unique = update_object_id_field_zarr(
                    self,
                    object_id_field_unique,
                    id_lookup,
                    updates_array,
                    updates_ids,
                    has_merge,
                )
            else:  # pragma: no cover
                object_id_field_unique = update_object_id_field_inplace(
                    object_id_field_unique,
                    id_lookup,
                    updates_array,
                    updates_ids,
                    has_merge,
                )
                object_id_field_unique = object_id_field_unique.chunk(
                    {self.timedim: self.timechunks}
                )  # Rechunk to avoid accumulating chunks...

            # Clean up arrays no longer needed
            del updates_array, updates_ids
            gc.collect()

            logger.debug("Finished Consolidation Step 2: Data Field Update")

            # 3. Update merge events
            new_merging_objects = set()
            merge_counts = merge_counts.compute()

            for t in time_indices:
                count = int(merge_counts.isel({self.timedim: t}).item())
                if count > 0:
                    merge_counts_iter[t] = count

                    # Extract valid IDs and areas for each merge event
                    for merge_idx in range(count):
                        # Get child IDs
                        child_ids = merge_child_ids.isel({self.timedim: t, "merge": merge_idx}).compute().values
                        child_ids = child_ids[child_ids >= 0]

                        # Get parent IDs and areas
                        parent_ids = merge_parent_ids.isel({self.timedim: t, "merge": merge_idx}).compute().values
                        areas = merge_areas.isel({self.timedim: t, "merge": merge_idx}).compute().values
                        valid_mask = parent_ids >= 0
                        parent_ids = parent_ids[valid_mask]
                        areas = areas[valid_mask]

                        # Map temporary IDs to permanent IDs
                        mapped_child_ids = [id_lookup.get(int(id_.item()), int(id_.item())) for id_ in child_ids]
                        mapped_parent_ids = [id_lookup.get(int(id_.item()), int(id_.item())) for id_ in parent_ids]

                        # Store in pre-allocated arrays
                        child_ids_iter[t, merge_idx, : len(mapped_child_ids)] = mapped_child_ids
                        parent_ids_iter[t, merge_idx, : len(mapped_parent_ids)] = mapped_parent_ids
                        merge_areas_iter[t, merge_idx, : len(areas)] = areas

            # Process final merging objects for next iteration
            final_merging_objects = final_merging_objects.compute().values
            final_merging_objects = final_merging_objects[final_merging_objects > 0]
            mapped_final_objects = [id_lookup.get(id_, id_) for id_ in final_merging_objects]
            new_merging_objects.update(mapped_final_objects)

            logger.debug("Finished Consolidation Step 3: Merge List Dictionary Consolidation")

            # Clean up memory
            del merge_child_ids, merge_parent_ids, merge_areas, merge_counts, has_merge
            gc.collect()

            return (
                object_id_field_unique,
                (child_ids_iter, parent_ids_iter, merge_areas_iter, merge_counts_iter),
                new_merging_objects,
                global_id_counter,
            )

        # ============================
        # Main Loop for Parallel Merging
        # ============================

        # Find overlapping objects
        overlap_objects_list = self.find_overlapping_objects(
            object_id_field_unique
        )  # List object pairs that overlap by at least overlap_threshold percent
        overlap_objects_list = self.enforce_overlap_threshold(overlap_objects_list, object_props)
        logger.info("Finished finding overlapping objects")

        # Find initial merging objects
        unique_children, children_counts = np.unique(overlap_objects_list[:, 1], return_counts=True)
        merging_objects = set(unique_children[children_counts > 1].astype(np.int32))
        del overlap_objects_list

        # Process chunks iteratively until no new merging objects remain

        iteration = 0
        processed_chunks = set()
        global_id_counter = int(object_props.ID.max().item()) + 1

        # Initialise global merge event tracking
        global_child_ids = []
        global_parent_ids = []
        global_merge_areas = []
        global_merge_tidx = []

        while merging_objects and iteration < self.max_iteration:
            (
                object_id_field_new,
                merge_data_iter,
                new_merging_objects,
                global_id_counter,
            ) = merge_objects_parallel_iteration(object_id_field_unique, merging_objects, global_id_counter)
            child_ids_iter, parent_ids_iter, merge_areas_iter, merge_counts_iter = merge_data_iter

            # Consolidate merge events from this iteration
            for t in range(len(merge_counts_iter)):
                count = merge_counts_iter[t]
                if count > 0:
                    for merge_idx in range(count):
                        # Extract valid children
                        children = child_ids_iter[t, merge_idx]
                        children = children[children >= 0]

                        # Extract valid parents and areas
                        parents = parent_ids_iter[t, merge_idx]
                        areas = merge_areas_iter[t, merge_idx]
                        valid_mask = parents >= 0
                        parents = parents[valid_mask]
                        areas = areas[valid_mask]

                        # Record valid merge events
                        if len(children) > 0 and len(parents) > 0:
                            global_child_ids.append(children)
                            global_parent_ids.append(parents)
                            global_merge_areas.append(areas)
                            global_merge_tidx.append(t)

            # Prepare for next iteration - only process objects not already handled
            merging_objects = new_merging_objects - processed_chunks
            processed_chunks.update(new_merging_objects)
            iteration += 1

            # Update the object field
            object_id_field_unique = object_id_field_new
            del object_id_field_new

        # Check if we reached maximum iterations
        if iteration == self.max_iteration:  # pragma: no cover
            raise TrackingError(
                "Maximum iterations reached in tracking algorithm",
                details=f"Algorithm failed to converge after {self.max_iteration} iterations",
                suggestions=[
                    "Increase max_iteration parameter",
                    "Increase area_filter_quartile to reduce small objects",
                    "Consider adjusting tracking parameters",
                ],
                context={
                    "max_iteration": self.max_iteration,
                    "reached_iteration": iteration,
                },
            )

        # Process the collected merge events

        times = object_id_field_unique[self.timecoord].values

        # Find maximum dimensions for arrays
        # Handle case where there are no merge events
        if global_parent_ids and global_child_ids:
            max_parents = max(len(ids) for ids in global_parent_ids)
            max_children = max(len(ids) for ids in global_child_ids)
        else:
            max_parents = 1  # Default minimum size
            max_children = 1

        # Create padded arrays for merge events
        parent_ids_array = np.full((len(global_parent_ids), max_parents), -1, dtype=np.int32)
        child_ids_array = np.full((len(global_child_ids), max_children), -1, dtype=np.int32)
        overlap_areas_array = np.full(
            (len(global_merge_areas), max_parents),
            -1,
            dtype=np.float32 if self.unstructured_grid else np.int32,
        )

        # Fill arrays with merge data
        for i, parents in enumerate(global_parent_ids):
            parent_ids_array[i, : len(parents)] = parents

        for i, children in enumerate(global_child_ids):
            child_ids_array[i, : len(children)] = children

        for i, areas in enumerate(global_merge_areas):
            overlap_areas_array[i, : len(areas)] = areas

        # Create merge events dataset
        merge_events = xr.Dataset(
            {
                "parent_IDs": (("merge_ID", "parent_idx"), parent_ids_array),
                "child_IDs": (("merge_ID", "child_idx"), child_ids_array),
                "overlap_areas": (("merge_ID", "parent_idx"), overlap_areas_array),
                "merge_time": ("merge_ID", times[global_merge_tidx]),
                "n_parents": (
                    "merge_ID",
                    np.array([len(p) for p in global_parent_ids], dtype=np.int8),
                ),
                "n_children": (
                    "merge_ID",
                    np.array([len(c) for c in global_child_ids], dtype=np.int8),
                ),
            },
            attrs={"fill_value": -1},
        )

        # Recompute object properties and overlaps after all merging
        object_id_field_unique = object_id_field_unique.persist(optimize_graph=True)
        object_props = self.calculate_object_properties(object_id_field_unique, properties=["area", "centroid"]).persist(
            optimize_graph=True
        )

        # Recompute overlaps based on final object configuration
        overlap_objects_list = self.find_overlapping_objects(object_id_field_unique)
        overlap_objects_list = self.enforce_overlap_threshold(overlap_objects_list, object_props)
        overlap_objects_list = overlap_objects_list[:, :2].astype(np.int32)

        return (
            object_id_field_unique,
            object_props,
            overlap_objects_list,
            merge_events,
        )


"""
MarEx Helper Functions

These are the remaining implementations of helper functions for the MarEx package,
providing optimised algorithms for partitioning, distance calculations, and spatial
operations on both structured and unstructured grids.
"""


@jit(nopython=True, parallel=True, fastmath=True)
def wrapped_euclidian_distance_mask_parallel(
    mask_values: NDArray[np.bool_],
    parent_centroids_values: NDArray[np.float64],
    Nx: int,
    wrap: bool,
) -> NDArray[np.float64]:  # pragma: no cover
    """
    Optimised function for computing wrapped Euclidean distances.

    Efficiently calculates distances between points in a binary mask and a set of
    centroids, accounting for periodic boundaries in the x dimension.

    Parameters
    ----------
    mask_values : np.ndarray
        2D boolean array where True indicates points to calculate distances for
    parent_centroids_values : np.ndarray
        Array of shape (n_parents, 2) containing (y, x) coordinates of parent centroids
    Nx : int
        Size of the x-dimension for periodic boundary wrapping
    wrap : bool
        Whether to treat x-dimension as periodic and wrap

    Returns
    -------
    distances : np.ndarray
        Array of shape (n_true_points, n_parents) with minimum distances
    """
    n_parents = len(parent_centroids_values)
    half_Nx = Nx / 2

    y_indices, x_indices = np.nonzero(mask_values)
    n_true = len(y_indices)

    distances = np.empty((n_true, n_parents), dtype=np.float64)

    # Precompute for faster access
    parent_y = parent_centroids_values[:, 0]
    parent_x = parent_centroids_values[:, 1]

    # Parallel loop over true positions
    for idx in prange(n_true):
        y, x = y_indices[idx], x_indices[idx]

        # Pre-compute y differences for all parents
        dy = y - parent_y

        # Pre-compute x differences for all parents
        dx = x - parent_x

        # Wrapping correction
        if wrap:
            dx = np.where(dx > half_Nx, dx - Nx, dx)
            dx = np.where(dx < -half_Nx, dx + Nx, dx)

        distances[idx] = np.sqrt(dy * dy + dx * dx)

    return distances


@jit(nopython=True, fastmath=True)
def create_grid_index_arrays(
    points_y: NDArray[np.int32],
    points_x: NDArray[np.int32],
    grid_size: int,
    ny: int,
    nx: int,
) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:  # pragma: no cover
    """
    Create a grid-based spatial index for efficient point lookup.

    This function divides space into a grid and assigns points to grid cells
    for more efficient spatial queries compared to brute force comparisons.

    Parameters
    ----------
    points_y, points_x : np.ndarray
        Coordinates of points to index
    grid_size : int
        Size of each grid cell
    ny, nx : int
        Dimensions of the overall grid

    Returns
    -------
    grid_points : np.ndarray
        3D array mapping grid cells to point indices
    grid_counts : np.ndarray
        2D array with count of points in each grid cell
    """
    n_grids_y = (ny + grid_size - 1) // grid_size
    n_grids_x = (nx + grid_size - 1) // grid_size
    max_points_per_cell = len(points_y)

    grid_points = np.full((n_grids_y, n_grids_x, max_points_per_cell), -1, dtype=np.int32)
    grid_counts = np.zeros((n_grids_y, n_grids_x), dtype=np.int32)

    for idx in range(len(points_y)):
        grid_y = min(points_y[idx] // grid_size, n_grids_y - 1)
        grid_x = min(points_x[idx] // grid_size, n_grids_x - 1)
        count = grid_counts[grid_y, grid_x]
        if count < max_points_per_cell:
            grid_points[grid_y, grid_x, count] = idx
            grid_counts[grid_y, grid_x] += 1

    return grid_points, grid_counts


@jit(nopython=True, fastmath=True)
def wrapped_euclidian_distance_points(
    y1: float, x1: float, y2: float, x2: float, nx: int, half_nx: float, wrap: bool
) -> float:  # pragma: no cover
    """
    Calculate distance with periodic boundary conditions in x dimension.

    Parameters
    ----------
    y1, x1 : float
        Coordinates of first point
    y2, x2 : float
        Coordinates of second point
    nx : int
        Size of x dimension
    half_nx : float
        Half the size of x dimension
    wrap : bool
        Whether to apply periodic boundary conditions in x

    Returns
    -------
    float
        Euclidean distance accounting for periodic boundary in x (or not)
    """
    dy = y1 - y2
    dx = x1 - x2

    if wrap:
        if dx > half_nx:
            dx -= nx
        elif dx < -half_nx:
            dx += nx

    return np.sqrt(dy * dy + dx * dx)


@jit(nopython=True, parallel=True, fastmath=True)
def partition_nn_grid(
    child_mask: NDArray[np.bool_],
    parent_masks: NDArray[np.bool_],
    child_ids: NDArray[np.int32],
    parent_centroids: NDArray[np.float64],
    Nx: int,
    max_distance: int = 20,
    wrap: bool = True,
) -> NDArray[np.int32]:  # pragma: no cover
    """
    Partition a child object based on nearest parent object points.

    This implementation uses spatial indexing and highly-threaded processing
    for efficient distance calculations. The algorithm assigns each point
    in the child object to the closest parent object.

    Parameters
    ----------
    child_mask : np.ndarray
        Binary mask of the child object
    parent_masks : np.ndarray
        List of binary masks for each parent object
    child_ids : np.ndarray
        List of IDs to assign to partitions
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) with parent centroids
    Nx : int
        Size of x dimension for periodic boundaries
    max_distance : int, default=20
        Maximum search distance
    wrap : bool, default=True
        Whether to apply periodic boundary conditions in the x dimension

    Returns
    -------
    new_labels : np.ndarray
        Array containing assigned child_ids for each point
    """
    ny, nx = child_mask.shape
    half_Nx = Nx / 2
    n_parents = len(parent_masks)
    grid_size = max(2, max_distance // 4)

    y_indices, x_indices = np.nonzero(child_mask)
    n_child_points = len(y_indices)

    min_distances = np.full(n_child_points, np.inf)
    parent_assignments = np.zeros(n_child_points, dtype=np.int32)
    found_close = np.zeros(n_child_points, dtype=np.bool_)

    for parent_idx in range(n_parents):
        py, px = np.nonzero(parent_masks[parent_idx])

        if len(py) == 0:  # Skip empty parents
            continue

        # Create grid index for this parent
        n_grids_y = (ny + grid_size - 1) // grid_size
        n_grids_x = (nx + grid_size - 1) // grid_size
        grid_points, grid_counts = create_grid_index_arrays(py, px, grid_size, ny, nx)

        # Process child points in parallel
        for child_idx in prange(n_child_points):
            if found_close[child_idx]:  # Skip if we already found an exact match
                continue

            child_y, child_x = y_indices[child_idx], x_indices[child_idx]
            grid_y = min(child_y // grid_size, n_grids_y - 1)
            grid_x = min(child_x // grid_size, n_grids_x - 1)

            min_dist_to_parent = np.inf

            # Check nearby grid cells
            for dy in range(-1, 2):
                grid_y_check = (grid_y + dy) % n_grids_y

                for dx in range(-1, 2):
                    grid_x_check = (grid_x + dx) % n_grids_x

                    # Process points in this grid cell
                    n_points = grid_counts[grid_y_check, grid_x_check]

                    for p_idx in range(n_points):
                        point_idx = grid_points[grid_y_check, grid_x_check, p_idx]
                        if point_idx == -1:
                            break

                        dist = wrapped_euclidian_distance_points(child_y, child_x, py[point_idx], px[point_idx], Nx, half_Nx, wrap)

                        if dist > max_distance:
                            continue

                        if dist < min_dist_to_parent:
                            min_dist_to_parent = dist

                        if dist < 1e-6:  # Found exact same point (within numerical precision)
                            min_dist_to_parent = dist
                            found_close[child_idx] = True
                            break

                    if found_close[child_idx]:
                        break

                if found_close[child_idx]:
                    break

            # Update assignment if this parent is closer
            if min_dist_to_parent < min_distances[child_idx]:
                min_distances[child_idx] = min_dist_to_parent
                parent_assignments[child_idx] = parent_idx

    # Handle any unassigned points using centroids
    unassigned = min_distances == np.inf
    if np.any(unassigned):
        for child_idx in np.nonzero(unassigned)[0]:
            child_y, child_x = y_indices[child_idx], x_indices[child_idx]
            min_dist = np.inf
            best_parent = 0

            for parent_idx in range(n_parents):
                # Calculate distance to centroid with periodic boundary conditions
                dist = wrapped_euclidian_distance_points(
                    child_y,
                    child_x,
                    parent_centroids[parent_idx, 0],
                    parent_centroids[parent_idx, 1],
                    Nx,
                    half_Nx,
                    wrap,
                )

                if dist < min_dist:
                    min_dist = dist
                    best_parent = parent_idx

            parent_assignments[child_idx] = best_parent

    # Convert from parent indices to child_ids
    new_labels = child_ids[parent_assignments]

    return new_labels


@jit(nopython=True, fastmath=True)
def partition_nn_unstructured(
    child_mask: NDArray[np.bool_],
    parent_masks: NDArray[np.bool_],
    child_ids: NDArray[np.int32],
    parent_centroids: NDArray[np.float64],
    neighbours_int: NDArray[np.int32],
    lat: NDArray[np.float32],
    lon: NDArray[np.float32],
    max_distance: int = 20,
) -> NDArray[np.int32]:  # pragma: no cover
    """
    Partition a child object on an unstructured grid based on nearest parent points.

    This function implements an efficient algorithm for assigning each cell in a child
    object to the nearest parent object, using graph traversal and spatial distances.
    It is optimised for unstructured grids.

    Parameters
    ----------
    child_mask : np.ndarray
        1D boolean array where True indicates points in the child object
    parent_masks : np.ndarray
        2D boolean array of shape (n_parents, n_points) where True indicates points in each parent object
    child_ids : np.ndarray
        1D array containing the IDs to assign to each partition of the child object
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) containing (lat, lon) coordinates of parent centroids in degrees
    neighbours_int : np.ndarray
        2D array of shape (3, n_points) containing indices of neighboring cells for each point
    lat, lon : np.ndarray
        Latitude/longitude arrays in degrees
    max_distance : int, default=20
        Maximum number of edge hops to search for parent points

    Returns
    -------
    new_labels : np.ndarray
        1D array containing the assigned child_ids for each True point in child_mask
    """
    # Force contiguous arrays in memory for optimal vectorised performance
    child_mask = np.ascontiguousarray(child_mask)
    parent_masks = np.ascontiguousarray(parent_masks)

    n_points = len(child_mask)
    n_parents = len(parent_masks)

    # Pre-allocate arrays
    distances = np.full(n_points, np.inf, dtype=np.float32)
    parent_assignments = np.full(n_points, -1, dtype=np.int32)
    visited = np.zeros((n_parents, n_points), dtype=np.bool_)

    # Initialise with direct overlaps
    for parent_idx in range(n_parents):
        overlap_mask = parent_masks[parent_idx] & child_mask
        if np.any(overlap_mask):
            visited[parent_idx, overlap_mask] = True
            unclaimed_overlap = distances[overlap_mask] == np.inf
            if np.any(unclaimed_overlap):
                overlap_points = np.where(overlap_mask)[0].astype(np.int32)
                valid_points = overlap_points[unclaimed_overlap]
                distances[valid_points] = 0
                parent_assignments[valid_points] = parent_idx

    # Pre-compute trig values for efficiency
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    cos_lat = np.cos(lat_rad)

    # Graph traversal for remaining points - expanding from parent frontiers
    current_distance = 0
    any_unassigned = np.any(child_mask & (parent_assignments == -1))

    while current_distance < max_distance and any_unassigned:
        current_distance += 1
        updates_made = False

        for parent_idx in range(n_parents):
            # Get current frontier points
            frontier_mask = visited[parent_idx]
            if not np.any(frontier_mask):
                continue

            # Process neighbors
            for i in range(3):  # For each neighbor direction
                neighbors = neighbours_int[i, frontier_mask]
                valid_neighbors = neighbors >= 0
                if not np.any(valid_neighbors):
                    continue

                valid_points = neighbors[valid_neighbors]
                unvisited = ~visited[parent_idx, valid_points]
                new_points = valid_points[unvisited]

                if len(new_points) > 0:
                    visited[parent_idx, new_points] = True
                    update_mask = distances[new_points] > current_distance
                    if np.any(update_mask):
                        points_to_update = new_points[update_mask]
                        distances[points_to_update] = current_distance
                        parent_assignments[points_to_update] = parent_idx
                        updates_made = True

        if not updates_made:
            break

        any_unassigned = np.any(child_mask & (parent_assignments == -1))

    # Handle remaining unassigned points using great circle distances
    unassigned_mask = child_mask & (parent_assignments == -1)
    if np.any(unassigned_mask):
        parent_lat_rad = np.deg2rad(parent_centroids[:, 0])
        parent_lon_rad = np.deg2rad(parent_centroids[:, 1])
        cos_parent_lat = np.cos(parent_lat_rad)

        unassigned_points = np.where(unassigned_mask)[0].astype(np.int32)
        for point in unassigned_points:
            # Vectoised haversine calculation
            dlat = parent_lat_rad - lat_rad[point]
            dlon = parent_lon_rad - lon_rad[point]
            a = np.sin(dlat / 2) ** 2 + cos_lat[point] * cos_parent_lat * np.sin(dlon / 2) ** 2
            dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            parent_assignments[point] = np.argmin(dist).astype(np.int32)

    # Return only the assignments for points in child_mask
    child_points = np.where(child_mask)[0].astype(np.int32)
    return child_ids[parent_assignments[child_points]]


@jit(nopython=True, fastmath=True)
def partition_nn_unstructured_optimised(
    child_mask: NDArray[np.bool_],
    parent_frontiers: NDArray[np.uint8],
    parent_centroids: NDArray[np.float64],
    neighbours_int: NDArray[np.int32],
    lat: NDArray[np.float32],
    lon: NDArray[np.float32],
    max_distance: int = 20,
) -> NDArray[np.uint8]:  # pragma: no cover
    """
    Memory-optimised nearest neighbor partitioning for unstructured grids.

    This version uses more efficient memory management compared to partition_nn_unstructured,
    making it suitable for very large grids. It uses a compact representation of parent
    frontiers to reduce memory usage during graph traversal.

    Parameters
    ----------
    child_mask : np.ndarray
        1D boolean array indicating which cells belong to the child object
    parent_frontiers : np.ndarray
        1D uint8 array with parent indices (255 for unvisited points)
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) containing (lat, lon) coordinates
    neighbours_int : np.ndarray
        2D array of shape (3, n_points) containing indices of neighboring cells
    lat, lon : np.ndarray
        1D arrays of latitude/longitude in degrees
    max_distance : int, default=20
        Maximum number of edge hops to search for parent points

    Returns
    -------
    result : np.ndarray
        1D array containing the assigned parent indices for points in child_mask
    """
    # Create working copies to ensure memory cleanup
    parent_frontiers_working = parent_frontiers.copy()
    child_mask_working = child_mask.copy()

    n_parents = np.max(parent_frontiers_working[parent_frontiers_working < 255]) + 1

    # Graph traversal - expanding frontiers
    current_distance = 0
    any_unassigned = np.any(child_mask_working & (parent_frontiers_working == 255))

    while current_distance < max_distance and any_unassigned:
        current_distance += 1
        updates_made = False

        for parent_idx in range(n_parents):
            # Skip if no frontier points for this parent
            if not np.any(parent_frontiers_working == parent_idx):
                continue

            # Process neighbours for current parent's frontier
            for i in range(3):
                neighbors = neighbours_int[i, parent_frontiers_working == parent_idx]
                valid_neighbors = neighbors >= 0

                if not np.any(valid_neighbors):
                    continue

                valid_points = neighbors[valid_neighbors]
                unvisited = parent_frontiers_working[valid_points] == 255

                if not np.any(unvisited):
                    continue

                # Update new frontier points
                new_points = valid_points[unvisited]
                parent_frontiers_working[new_points] = parent_idx

                if np.any(child_mask_working[new_points]):
                    updates_made = True

        if not updates_made:
            break

        any_unassigned = np.any(child_mask_working & (parent_frontiers_working == 255))

    # Handle remaining unassigned points using great circle distances
    unassigned_mask = child_mask_working & (parent_frontiers_working == 255)
    if np.any(unassigned_mask):
        # Pre-compute parent coordinates in radians
        parent_lat_rad = np.deg2rad(parent_centroids[:, 0])
        parent_lon_rad = np.deg2rad(parent_centroids[:, 1])
        cos_parent_lat = np.cos(parent_lat_rad)

        # Process each unassigned point
        unassigned_points = np.where(unassigned_mask)[0].astype(np.int32)
        for point in unassigned_points:
            dlat = parent_lat_rad - np.deg2rad(lat[point])
            dlon = parent_lon_rad - np.deg2rad(lon[point])

            a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat[point])) * cos_parent_lat * np.sin(dlon / 2) ** 2
            dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            parent_frontiers_working[point] = np.int32(np.argmin(dist))

    # Extract result for child points only
    result = parent_frontiers_working[child_mask_working].copy()

    # Explicitly clear working arrays to help with memory management
    parent_frontiers_working = None
    child_mask_working = None

    return result


@jit(nopython=True, parallel=True, fastmath=True)
def partition_centroid_unstructured(
    child_mask: NDArray[np.bool_],
    parent_centroids: NDArray[np.float64],
    child_ids: NDArray[np.int32],
    lat: NDArray[np.float32],
    lon: NDArray[np.float32],
) -> NDArray[np.int32]:  # pragma: no cover
    """
    Partition a child object based on closest parent centroids on an unstructured grid.

    This function assigns each cell in the child object to the parent with the closest
    centroid, using great circle distances on a spherical grid.

    Parameters
    ----------
    child_mask : np.ndarray
        1D boolean array indicating which cells belong to the child object
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) containing (lat, lon) coordinates of parent centroids in degrees
    child_ids : np.ndarray
        Array of IDs to assign to each partition of the child object
    lat, lon : np.ndarray
        Latitude/longitude arrays in degrees

    Returns
    -------
    new_labels : np.ndarray
        1D array containing assigned child_ids for cells in child_mask
    """
    n_cells = len(child_mask)
    n_parents = len(parent_centroids)

    # Convert to radians for spherical calculations
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    parent_coords_rad = np.deg2rad(parent_centroids)

    new_labels = np.zeros(n_cells, dtype=child_ids.dtype)

    # Process each child cell in parallel
    for i in prange(n_cells):
        if not child_mask[i]:
            continue

        min_dist = np.inf
        closest_parent = 0

        # Calculate great circle distance to each parent centroid
        for j in range(n_parents):
            dlat = parent_coords_rad[j, 0] - lat_rad[i]
            dlon = parent_coords_rad[j, 1] - lon_rad[i]

            # Use haversine formula for great circle distance
            a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad[i]) * np.cos(parent_coords_rad[j, 0]) * np.sin(dlon / 2) ** 2
            dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            if dist < min_dist:
                min_dist = dist
                closest_parent = j

        new_labels[i] = child_ids[closest_parent]

    return new_labels


@njit(fastmath=True, parallel=True)
def sparse_bool_power(
    vec: NDArray[np.bool_],
    sp_data: NDArray[np.bool_],
    indices: NDArray[np.int32],
    indptr: NDArray[np.int32],
    exponent: int,
) -> NDArray[np.bool_]:  # pragma: no cover
    """
    Efficient sparse boolean matrix power operation.

    This function implements a fast sparse matrix power operation for boolean matrices,
    avoiding memory leaks present in scipy+Dask implementations. It's used for efficient
    morphological operations on unstructured grids.

    Parameters
    ----------
    vec : np.ndarray
        Boolean vector to multiply
    sp_data, indices, indptr : np.ndarray
        Sparse matrix in CSR format
    exponent : int
        Number of times to apply the matrix

    Returns
    -------
    np.ndarray
        Result of (sparse_matrix ^ exponent) * vec
    """
    vec = vec.T
    num_rows = indptr.size - 1
    num_cols = vec.shape[1]
    result = vec.copy()

    for _ in range(exponent):
        temp_result = np.zeros((num_rows, num_cols), dtype=np.bool_)

        for i in prange(num_rows):
            for j in range(indptr[i], indptr[i + 1]):
                if sp_data[j]:
                    for k in range(num_cols):
                        if result[indices[j], k]:
                            temp_result[i, k] = True

        result = temp_result

    return result.T


def regional_tracker(
    data_bin: xr.DataArray,
    mask: xr.DataArray,
    coordinate_units: Literal["degrees", "radians"],
    R_fill: Union[int, float],
    area_filter_quartile: Optional[float] = None,
    area_filter_absolute: Optional[int] = None,
    **kwargs,
) -> "tracker":
    """
    Create a tracker instance configured for regional (non-global) data.

    This is a convenience function that automatically sets regional_mode=True
    and requires explicit specification of coordinate units, since auto-detection
    may fail for regional coordinate ranges.

    Parameters
    ----------
    data_bin : xr.DataArray
        Binary data to identify and track objects in (True = object, False = background)
    mask : xr.DataArray
        Binary mask indicating valid regions (True = valid, False = invalid)
    coordinate_units : {'degrees', 'radians'}
        Units of the coordinate system. Must be specified for regional data.
    R_fill : int or float
        Radius for filling holes/gaps in spatial domain (in grid cells)
    area_filter_quartile : float, optional
        Quantile (0-1) for filtering smallest objects (e.g., 0.25 removes smallest 25%).
        Mutually exclusive with area_filter_absolute. Default is 0.5 if neither parameter is provided.
    area_filter_absolute : int, optional
        The minimum area (in grid cells) for an object to be retained. Mutually exclusive with area_filter_quartile.
    **kwargs
        Additional parameters passed to the tracker class

    Returns
    -------
    tracker
        Configured tracker instance with regional_mode=True

    Examples
    --------
    Track events in regional Mediterranean Sea data:

    >>> import marEx
    >>> # For regional data with degree coordinates
    >>> regional_tracker = marEx.regional_tracker(
    ...     extreme_events,
    ...     mask,
    ...     coordinate_units='degrees',
    ...     R_fill=5,
    ...     area_filter_quartile=0.3
    ... )
    >>> events = regional_tracker.run()

    Track events in regional data with radian coordinates:

    >>> # For model output with radian coordinates
    >>> regional_tracker = marEx.regional_tracker(
    ...     extreme_events,
    ...     mask,
    ...     coordinate_units='radians',
    ...     R_fill=8,
    ...     area_filter_quartile=0.5
    ... )
    >>> events = regional_tracker.run()

    Using absolute area filtering in regional mode:

    >>> # Keep only features larger than 15 grid cells
    >>> absolute_regional = marEx.regional_tracker(
    ...     extreme_events,
    ...     mask,
    ...     coordinate_units='degrees',
    ...     R_fill=5,
    ...     area_filter_absolute=15
    ... )
    >>> events = absolute_regional.run()
    """
    return tracker(
        data_bin=data_bin,
        mask=mask,
        R_fill=R_fill,
        area_filter_quartile=area_filter_quartile,
        area_filter_absolute=area_filter_absolute,
        regional_mode=True,
        coordinate_units=coordinate_units,
        **kwargs,
    )
