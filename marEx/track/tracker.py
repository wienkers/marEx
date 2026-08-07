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

import logging
import os
import shutil
import time
import uuid
import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr
from dask import is_dask_collection
from dask.distributed import wait
from numpy.typing import NDArray

from .._dependencies import warn_missing_dependency
from ..detect.compute_mode import Materialiser, create_staging_dir
from ..exceptions import ConfigurationError, TrackingError, create_data_validation_error
from ..logging_config import configure_logging, get_logger, log_dask_info, log_memory_usage, log_timing
from . import grid as _grid
from . import merge_split as _merge_split
from . import morphology as _morphology
from . import objects as _objects
from . import overlap as _overlap
from . import validation as _validation

# Get module logger
logger = get_logger(__name__)

# Prefixes of the per-run temporary zarr stores written during tracking. The bare
# "marEx_temp_field.zarr" written by older versions also matches, so stale legacy stores
# get cleaned up too. Checkpoint artefacts ("marEx_checkpoint_*") deliberately do not match.
_TEMP_STORE_PREFIXES = ("marEx_temp_field", "marEx_temp_refresh")
_TEMP_STORE_MAX_AGE_S = 24 * 3600


def _prune_stale_temp_stores(temp_dir: str, max_age_s: float = _TEMP_STORE_MAX_AGE_S) -> None:
    """Delete temp zarr stores left behind by earlier tracking runs.

    These stores cannot be removed when ``run()`` returns: both ``refresh_dask_graph`` and
    the merge loop hand back ``xr.open_zarr`` views, so the returned dataset still reads
    from them lazily until the caller computes. Pruning by age instead bounds disk usage
    without touching a store a concurrent run may still own.
    """
    try:
        entries = os.listdir(temp_dir)
    except OSError:
        return
    now = time.time()
    for name in entries:
        if not name.startswith(_TEMP_STORE_PREFIXES):
            continue
        path = os.path.join(temp_dir, name)
        try:
            if now - os.path.getmtime(path) > max_age_s:
                shutil.rmtree(path, ignore_errors=True)
                logger.debug(f"Pruned stale temp store {path}")
        except OSError:
            continue


def _is_uniform_time_chunking(chunks: Tuple[int, ...]) -> bool:
    """Check that a chunk tuple is uniform except for a possibly-smaller final chunk.

    This is exactly the shape ``.chunk({"time": k})`` always produces: ``(k, k, ..., k, r)``
    with ``r <= k``. :class:`~marEx.track.region_writer.ObjectIDRegionWriter` requires this
    for the zarr region writes streaming mode relies on. A genuinely ragged chunking --
    e.g. from ``open_mfdataset`` over uneven per-year files, giving ``(365, 365, 366, 365)``
    -- reaches the zarr write intact and fails there with a low-level ``ValueError``,
    possibly after hours of earlier processing.
    """
    if len(chunks) <= 1:
        return True
    first = chunks[0]
    if any(c != first for c in chunks[:-1]):
        return False
    return chunks[-1] <= first


def _format_chunk_pattern(chunks: Tuple[int, ...], max_show: int = 10) -> str:
    """Render a chunk tuple for an error message, truncated so it never dumps thousands
    of numbers into an exception.
    """
    if len(chunks) <= max_show:
        return str(tuple(chunks))
    half = max_show // 2
    head = ", ".join(str(c) for c in chunks[:half])
    tail = ", ".join(str(c) for c in chunks[-half:])
    return f"({head}, ..., {tail}) [{len(chunks)} chunks total]"


# Module-level state for the distributed.scheduler warning filter. The filter is
# installed exactly once on the global logger (guarded by ``_DASK_WARNING_FILTER_INSTALLED``)
# so that repeated ``tracker(...)`` construction does not accumulate duplicate filters (and
# pin dead instances). The active debug level is read from module state rather than closed
# over ``self``; the most recently configured tracker wins.
_DASK_WARNING_FILTER_INSTALLED = False
_dask_warning_debug_level = 0


def _filter_dask_warnings(record: logging.LogRecord) -> bool:  # pragma: no cover
    """Filter noisy distributed.scheduler warnings based on the active debug level."""
    msg = str(record.msg)

    if _dask_warning_debug_level == 0:
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
    compute_mode : {'persist', 'streaming'}, default='persist'
        Materialisation policy for whole-field intermediates during tracking.

        * 'persist' (default): pins intermediates in cluster RAM. Fastest, and correct
          whenever the run fits (measured peak ~57 GB at n_time=3804, 0.25 deg global).
        * 'streaming': stages the ID field, the filled/filtered fields, and the merge
          loop's output accumulator to zarr under ``temp_dir`` instead of pinning them,
          so memory scales with cluster size rather than series length (measured peak
          20.4 GB on the same run, bytes pinned 337 -> 1.47 GB). Requires ``temp_dir``.
          Disk cost is roughly 5 stores of 2 bool + 3 int32 whole fields, ~14 bytes per
          cell-timestep uncompressed (~55 GB uncompressed at that same n_time=3804 x
          720 x 1440 run; less on disk, since the ID fields are mostly zeros).

          **Restrictions**: gridded grids only -- rejected for ``unstructured_grid=True``,
          since the unstructured merge/split loop uses its own separate zarr writer that
          is not wired through this mode. Also requires the input's time chunking to be
          uniform (every chunk equal except a possibly-smaller last one, exactly what
          ``.chunk({'time': k})`` always produces); a genuinely ragged chunking is
          rejected at construction time rather than failing inside the zarr write.

          **Staging-lifetime contract**: the returned dataset reads *lazily* from the
          staged zarr store, so the staging directory deliberately **outlives**
          ``run()``. Write your output first, then call
          ``marEx.clear_staging(events_ds)`` -- the path is on
          ``events_ds.attrs["marex_staging_dir"]``. An ``atexit`` hook cleans up on
          normal interpreter exit, but it does **not** survive SIGKILL (e.g. a
          wall-clock kill), so sweep ``temp_dir`` periodically. See CHUNKING_NOTES.md
          §3.1/§5.2 for the full contract and measurements.
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
        *,
        compute_mode: Literal["persist", "streaming"] = "persist",
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

        # Validate coordinate presence before touching them, so a missing coordinate raises
        # the descriptive error rather than a bare KeyError from the indexing below (§4.4).
        _validation.validate_required_coordinates(data_bin, self.timecoord, self.xcoord, self.ycoord)

        self.lat_init = data_bin[self.ycoord].persist()  # Save in original units
        self.lon_init = data_bin[self.xcoord].persist()
        self.coordinate_units, self.data_bin = _grid.unify_coordinates(
            self.data_bin,
            self.regional_mode,
            self.coordinate_units,
            self.xcoord,
            self.ycoord,
        )

        self.mask = mask
        self.R_fill = int(R_fill)
        self.T_fill = T_fill

        # Resolve area filtering parameters
        (
            self.area_filter_quartile,
            self.area_filter_absolute,
            self._use_absolute_filtering,
        ) = _validation.resolve_area_filtering_parameters(area_filter_quartile, area_filter_absolute)
        self.allow_merging = allow_merging
        self.nn_partitioning = nn_partitioning
        self.overlap_threshold = overlap_threshold
        # Read the degree-unit coordinates from self.data_bin (the converted output of
        # unify_coordinates); the input data_bin is no longer mutated in place (§1.3).
        self.lat = self.data_bin[self.ycoord].persist()
        self.lon = self.data_bin[self.xcoord].persist()
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

        # Resolve the scratch directory used for checkpointing and temporary zarr stores.
        # This must be available on BOTH grid branches: previously it was only set inside the
        # unstructured setup, so checkpoint='save'/'load' raised AttributeError on structured
        # grids (§4.1). The path is kept stable and user-managed so that a later
        # checkpoint='load' can find the files written by an earlier checkpoint='save' run.
        if self.checkpoint and not temp_dir:
            raise ConfigurationError(
                "Checkpointing requires a temporary directory",
                details=f"checkpoint={self.checkpoint!r} was requested but temp_dir is None",
                suggestions=[
                    "Provide temp_dir: tracker(..., temp_dir='/scratch/user/marex')",
                    "Disable checkpointing by leaving checkpoint=None",
                ],
            )
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
        self.scratch_dir = temp_dir

        # compute_mode: two modes only. `lazy` is rejected rather than silently
        # accepted -- the merge loop is inherently sequential, so accepting recompute
        # buys nothing there, and detect's lazy performance figures were never
        # measured at scale.
        if compute_mode == "lazy":
            raise ConfigurationError(
                "compute_mode='lazy' is not supported by the tracker",
                details="The merge/split loop is sequential in time; recompute buys nothing",
                suggestions=[
                    "Use compute_mode='persist' (default) when the run fits in cluster RAM",
                    "Use compute_mode='streaming' with temp_dir for long time series",
                ],
                context={"provided_mode": compute_mode},
            )
        if compute_mode not in ("persist", "streaming"):
            raise ConfigurationError(
                f"Unknown compute_mode '{compute_mode}'",
                details="Invalid compute_mode parameter",
                suggestions=[
                    "Use 'persist' (default) or 'streaming'",
                ],
                context={"provided_mode": str(compute_mode), "valid_modes": ["persist", "streaming"]},
            )
        if compute_mode == "streaming" and not temp_dir:
            raise ConfigurationError(
                "compute_mode='streaming' requires temp_dir",
                details="Streaming mode stages the whole-field intermediates to zarr on disk",
                suggestions=[
                    "Pass temp_dir pointing at large transient storage, e.g. tracker(..., temp_dir='/scratch/user/marex')",
                    "Use compute_mode='persist' if no scratch space is available",
                ],
                context={"provided_mode": compute_mode},
            )

        # streaming currently covers the gridded code path only. The unstructured merge/split
        # loop (split_and_merge_objects_parallel) writes its own zarr store via
        # update_object_id_field_zarr / temp_merge_path and is not wired through the
        # Materialiser, so accepting this combination would silently stream only PART of the
        # pipeline (the shared preprocessing stages in run_preprocess() and objects.py's
        # _anchor do stage; the unstructured core does not). That partial combination has zero
        # test and zero benchmark coverage. See CHUNKING_NOTES.md §3.1/§5.2.
        if compute_mode == "streaming" and self.unstructured_grid:
            raise ConfigurationError(
                "compute_mode='streaming' does not support unstructured grids",
                details=(
                    "Streaming currently supports gridded grids only. The unstructured "
                    "merge/split loop (split_and_merge_objects_parallel) uses its own separate "
                    "zarr writer that is not wired through the Materialiser, so this combination "
                    "would silently stream only part of the pipeline."
                ),
                suggestions=[
                    "Use compute_mode='persist' for unstructured grids",
                ],
                context={"provided_mode": compute_mode, "unstructured_grid": self.unstructured_grid},
            )

        # streaming's zarr region writer (ObjectIDRegionWriter) requires uniform chunking
        # along the time dimension: every chunk equal except possibly a smaller final one.
        # `.chunk({"time": k})` always produces exactly that shape, so this never rejects the
        # common case. A genuinely ragged chunking (e.g. open_mfdataset over uneven per-year
        # files) would otherwise reach the zarr write intact and fail there with a confusing
        # low-level zarr ValueError, after potentially hours of earlier processing.
        if compute_mode == "streaming":
            time_chunks = data_bin.chunks[data_bin.dims.index(self.timedim)]
            if not _is_uniform_time_chunking(time_chunks):
                raise ConfigurationError(
                    "compute_mode='streaming' requires uniform time chunking",
                    details=(
                        f"Every chunk along {self.timedim!r} must be equal except the last, which "
                        f"may be smaller. Got: {_format_chunk_pattern(time_chunks)}"
                    ),
                    suggestions=[
                        f"Rechunk uniformly, e.g. data_bin.chunk({{'{self.timedim}': k}})",
                        "Use compute_mode='persist' if the input cannot be rechunked uniformly",
                    ],
                    context={"provided_mode": compute_mode, "time_chunks": _format_chunk_pattern(time_chunks)},
                )

        self.compute_mode = compute_mode
        self.staging_dir = create_staging_dir(temp_dir) if compute_mode == "streaming" else None
        self.materialiser = Materialiser(compute_mode, self.staging_dir)

        # Per-run temp stores. These used to share one fixed path
        # ({scratch_dir}/marEx_temp_field.zarr) between refresh_dask_graph and the merge
        # loop: on unstructured grids refresh created the store first, so the merge loop
        # skipped its own initialisation and region-wrote into it, leaving time chunks
        # without merges holding stale pre-filter IDs. Concurrent runs sharing a
        # scratch_dir corrupted each other the same way.
        if temp_dir:
            _prune_stale_temp_stores(temp_dir)
            run_token = uuid.uuid4().hex[:12]
            self.temp_refresh_path = os.path.join(temp_dir, f"marEx_temp_refresh_{run_token}.zarr")
            self.temp_merge_path = os.path.join(temp_dir, f"marEx_temp_field_{run_token}.zarr")
        else:
            self.temp_refresh_path = None
            self.temp_merge_path = None

        logger.debug(f"Dimensions: time={self.timedim}, x={self.xdim}, y={self.ydim}")
        logger.debug(f"Coordinates: time={self.timecoord}, x={self.xcoord}, y={self.ycoord}")

        # Extract data_bin metadata to inherit
        if hasattr(self.data_bin, "attrs") and self.data_bin.attrs:
            self.data_attrs = self.data_bin.attrs.copy()
        else:
            self.data_attrs = {}

        # Input validation and preparation
        (
            self.data_bin,
            self.ydim,
            self.mask,
            self.lat,
            self.lon,
        ) = _validation.validate_inputs(
            self.data_bin,
            self.mask,
            self.regional_mode,
            self.unstructured_grid,
            self.timedim,
            self.xdim,
            self.ydim,
            self.timecoord,
            self.xcoord,
            self.ycoord,
            self._use_absolute_filtering,
            self.area_filter_quartile,
            self.area_filter_absolute,
            self.T_fill,
            self.lat,
            self.lon,
            neighbours=neighbours,
            cell_areas=cell_areas,
            grid_resolution=grid_resolution,
            temp_dir=temp_dir,
        )

        # Handle cell_areas for both structured and unstructured grids
        if self.unstructured_grid:
            # Validation already done in _validate_inputs, but the spatial chunking of
            # neighbours/cell_areas still has to be enforced here: self.cell_area is
            # persisted below, before setup_unstructured_grid runs, so a rechunk applied
            # any later would never reach it (§4.2).
            neighbours, cell_areas = _validation.validate_unstructured_chunking(neighbours, cell_areas, self.xdim)
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
            (
                self.scratch_dir,
                self.data_bin,
                self.mask,
                self.lat,
                self.lon,
                self.max_iteration,
                self.neighbours_int,
                self.dilate_sparse,
            ) = _grid.setup_unstructured_grid(
                temp_dir,
                neighbours,
                cell_areas,
                max_iteration,
                self.data_bin,
                self.mask,
                self.lat,
                self.lon,
                self.xdim,
                self.xcoord,
                self.ycoord,
            )

        # Materialise the small per-cell coordinate and area arrays rather than leaving them
        # as persisted dask collections. persist() binds a collection to whichever client was
        # active at construction AND replaces its graph with futures, so once that client is
        # closed the data is orphaned with nothing left to recompute from. That is exactly
        # what the documented two-cluster pattern does -- run_preprocess(checkpoint="save"),
        # close the cluster, then run(checkpoint="load") on a differently-sized one -- and it
        # died in calculate_object_properties with
        # "FutureCancelledError: ... cancelled for reason: lost dependencies".
        #
        # It only ever bit unstructured grids, which is why nothing caught it: there these
        # are genuinely dask-backed (14 886 338 elements each on ICON R02B09), whereas on a
        # gridded store lat/lon are small numpy coords and persist() was a silent no-op.
        #
        # This removes work rather than adding it. Every consumer reads them as numpy:
        # overlap.py already materialises cell_area for precisely this reason, and grid.py
        # calls .compute() on lat_init/lon_init. The arrays also stop being pinned in worker
        # memory for the whole run. Done here, at the end of __init__, because
        # validate_inputs() and setup_unstructured_grid() both reassign self.lat/self.lon.
        for _attr in ("lat", "lon", "lat_init", "lon_init", "cell_area"):
            _value = getattr(self, _attr, None)
            if _value is not None and is_dask_collection(getattr(_value, "data", None)):
                setattr(self, _attr, _value.compute())

        self._configure_warnings()

    def _remap_coordinates(self, events_ds: xr.Dataset) -> xr.Dataset:
        """Remap coordinates to original lat/lon values after processing.
        Map centroids from lat=[-180,180] back into original lat/lon units & range.
        """
        return _grid.remap_coordinates(
            events_ds,
            self.lat_init,
            self.lon_init,
            self.coordinate_units,
            self.xcoord,
            self.ycoord,
        )

    def _configure_warnings(self) -> None:
        """Configure warning and logging suppression based on debug level."""
        logger.debug(f"Configuring warnings and logging for debug level: {self.debug}")
        if self.debug < 2:
            # Configure logging warning filters
            logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)

            # Record the active debug level in module state and install the module-level
            # filter exactly once, so repeated tracker construction cannot stack duplicate
            # filters on the global distributed.scheduler logger.
            global _DASK_WARNING_FILTER_INSTALLED, _dask_warning_debug_level
            _dask_warning_debug_level = self.debug
            if not _DASK_WARNING_FILTER_INSTALLED:
                logging.getLogger("distributed.scheduler").addFilter(_filter_dask_warnings)
                _DASK_WARNING_FILTER_INSTALLED = True

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
        # The single-use guard must not fire when the preprocessed data is being reloaded
        # from a checkpoint: run_preprocess(checkpoint="load") returns straight from the
        # zarr store and never reads self.data_bin. That is exactly the documented
        # two-cluster pattern -- run_preprocess(checkpoint="save") on a large cluster,
        # close it, then run(checkpoint="load") on a differently-sized one -- and the
        # unconditional guard made it impossible (the unstructured 02 notebook, which had
        # never been executed, died here).
        effective_checkpoint = checkpoint if checkpoint is not None else self.checkpoint
        if self.data_bin is None and effective_checkpoint != "load":
            raise TrackingError(
                "This tracker instance has already been run",
                details="run() frees the binary input to save memory, so a tracker is single-use",
                suggestions=[
                    "Construct a fresh instance for another run: tracker(data_bin, mask, ...)",
                    "Or reuse the preprocessed result: run(checkpoint='load') after run_preprocess(checkpoint='save')",
                ],
            )

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

        # Streaming mode returns a dataset that reads lazily from the staged zarr stores,
        # so the staging directory deliberately OUTLIVES this call. The caller writes its
        # output first, then calls marEx.clear_staging(events_ds). Note the atexit backstop
        # does NOT survive SIGKILL -- a wall-clock kill leaves the directory behind -- so
        # sweep temp_dir periodically.
        if self.staging_dir is not None:
            events_ds.attrs["marex_staging_dir"] = str(self.staging_dir)

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
            # Force compute here (persist + wait) so this step's time is attributed to it,
            # rather than being deferred into the later "Small object filtering" step.
            # raw_area rides along in the same persist: it is only consumed much later, for
            # one scalar diagnostic, and by then self.data_bin has been released -- so on
            # its own it forced a second full read of the entire raw input
            # (review finding 4.3).
            data_bin_filled, raw_area = self.materialiser.pin(self.fill_holes(self.data_bin), raw_area)
            if self.materialiser.is_streaming:
                # pin is a no-op in streaming, which would leave raw_area lazy and force the
                # second full read of the raw input that the comment above says was removed.
                # raw_area is tiny (a per-timestep reduction, shape (n_time,)), so pinning it
                # costs nothing.
                raw_area = raw_area.persist()
            wait(data_bin_filled)
            self.data_bin = None  # Free memory (tracker instance is now single-run)
            log_memory_usage(logger, "After spatial hole filling", logging.DEBUG)

        # Fill small time-gaps between objects
        logger.info(f"Filling temporal gaps with T_fill={self.T_fill}")
        with log_timing(logger, "Temporal gap filling"):
            data_bin_filled = self.materialiser.stage(self.fill_time_gaps(data_bin_filled), "data_bin_filled", preserve_chunks=True)
            wait(data_bin_filled)  # Force compute so this step's time is attributed correctly
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
                data_bin_filtered.name = "data_bin_preproc"
                # Write lazily (no .persist()): the store computes the preprocessing graph
                # straight to disk, then we reload to break the graph for downstream steps.
                data_bin_filtered.to_zarr(f"{self.scratch_dir}/marEx_checkpoint_proc_bin.zarr", mode="w")
                data_bin_filtered = load_data_from_checkpoint()
        else:
            logger.debug("Persisting preprocessed data in memory")
            data_bin_filtered = self.materialiser.stage(data_bin_filtered, "data_bin_filtered", preserve_chunks=True)
            wait(data_bin_filtered)

        # Compute area of processed data
        processed_area = self.compute_area(data_bin_filtered)

        # Compute statistics
        object_areas = object_areas.compute()
        total_area_IDed = float(object_areas.sum().item())

        # Use >= so the accepted-area statistic matches the area filter, which keeps ties
        # (area == threshold) on both grid types.
        accepted_area = float(object_areas.where(object_areas >= area_threshold, drop=True).sum().item())
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
            # The zarr store is already reloaded above (its contents have not changed since),
            # and object_stats holds the values just written -- re-reading the npz only
            # replaced Python scalars with 0-d numpy arrays (§4.6).

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
        return _morphology.compute_area(data_bin, self.unstructured_grid, self.cell_area, self.xdim, self.ydim)

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
        return _morphology.fill_holes(
            data_bin,
            self.R_fill,
            self.unstructured_grid,
            getattr(self, "dilate_sparse", None),
            self.xdim,
            self.mask,
            self.regional_mode,
            self.ydim,
            R_fill=R_fill,
        )

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
        return _morphology.fill_time_gaps(
            data_bin,
            self.T_fill,
            self.R_fill,
            self.timedim,
            self.ydim,
            self.unstructured_grid,
            getattr(self, "dilate_sparse", None),
            self.xdim,
            self.mask,
            self.regional_mode,
            materialiser=self.materialiser,
        )

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
        return _morphology.refresh_dask_graph(data_bin, self.temp_refresh_path)

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
        return _morphology.filter_small_objects(
            data_bin,
            self.unstructured_grid,
            self.xdim,
            self._use_absolute_filtering,
            self.area_filter_absolute,
            self.area_filter_quartile,
            self.mask,
            getattr(self, "neighbours_int", None),
            self.regional_mode,
            self.lat,
            self.lon,
            self.cell_area,
            self.timedim,
            self.ydim,
            materialiser=self.materialiser,
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
        return _objects.identify_objects(
            data_bin,
            time_connectivity,
            self.unstructured_grid,
            self.mask,
            getattr(self, "neighbours_int", None),
            self.xdim,
            self.regional_mode,
            materialiser=self.materialiser,
        )

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
        return _objects.calculate_centroid(binary_mask, self.regional_mode, original_centroid)

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
        return _objects.calculate_object_properties(
            object_id_field,
            self.unstructured_grid,
            self.lat,
            self.lon,
            self.cell_area,
            self.timedim,
            self.regional_mode,
            self.ydim,
            self.xdim,
            properties=properties,
        )

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
        return _overlap.check_overlap_slice(ids_t0, ids_next, self.unstructured_grid, self.cell_area)

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
        return _overlap.find_overlapping_objects(
            object_id_field,
            self.timedim,
            self.unstructured_grid,
            self.ydim,
            self.xdim,
            self.cell_area,
        )

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
        return _overlap.enforce_overlap_threshold(
            overlap_objects_list,
            object_props,
            self.unstructured_grid,
            self.overlap_threshold,
        )

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
        return _overlap.consolidate_object_ids(
            data_t_minus_2,
            data_t_minus_1,
            object_props,
            timestep,
            self.unstructured_grid,
            self.cell_area,
            self.overlap_threshold,
            self.lat,
            self.lon,
            self.timedim,
            self.regional_mode,
            self.ydim,
            self.xdim,
        )

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
        return _overlap.compute_id_time_dict(
            da,
            child_objects,
            max_objects,
            self.timedim,
            self.unstructured_grid,
            self.ydim,
            self.xdim,
            all_objects=all_objects,
        )

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
        # pin, NOT stage: this is the same array identify_objects already anchored
        # (objects.py:250, Task 5). Staging it again would write a second copy of the
        # whole field to zarr. In persist mode dask.persist on an already-persisted
        # collection returns the same futures, so the default is unchanged.
        object_id_field = self.materialiser.pin_one(object_id_field)
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
        # object_id_field is deliberately excluded: it is already anchored by
        # split_and_merge (Tasks 6/7), so pinning it again here would be redundant. The
        # other three are small. NOTE: if the golden or graph-structure test moves,
        # restore object_id_field to this call.
        results = self.materialiser.pin(object_props, overlap_objects_list, merge_events)
        object_props, overlap_objects_list, merge_events = results

        # Cluster & rename objects to get globally unique event IDs
        split_merged_events_ds = self.cluster_rename_objects_and_props(
            object_id_field, object_props, overlap_objects_list, merge_events
        )

        # Rechunk final output. The time dimension is deliberately left alone here: run()
        # rechunks the returned dataset to {time: 1} immediately afterwards, so setting it
        # to timechunks first only layered a superseded rechunk over the graph
        # (review finding 4.7).
        chunk_dict = {
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
        return _merge_split.cluster_rename_objects_and_props(
            object_id_field_unique,
            object_props,
            overlap_objects_list,
            merge_events,
            self.unstructured_grid,
            self.timedim,
            self.timecoord,
            self.timechunks,
            self.ydim,
            self.xdim,
            self.cell_area,
            self.lat,
            self.lon,
            self.regional_mode,
            materialiser=self.materialiser,
        )

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
        return _merge_split.split_and_merge_objects(
            object_id_field_unique,
            object_props,
            self.unstructured_grid,
            self.timedim,
            self.ydim,
            self.xdim,
            self.cell_area,
            self.lat,
            self.lon,
            self.mean_cell_area,
            # neighbours_int only exists for unstructured grids; this method runs on
            # structured grids where it is never read (used only inside the
            # ``unstructured_grid`` branch), matching the original lazy ``self.`` access.
            getattr(self, "neighbours_int", None),
            self.nn_partitioning,
            self.overlap_threshold,
            self.regional_mode,
            materialiser=self.materialiser,
            id_field_path=(os.path.join(str(self.staging_dir), "merge_id_field.zarr") if self.staging_dir else None),
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
        return _merge_split.split_and_merge_objects_parallel(
            object_id_field_unique,
            object_props,
            self.unstructured_grid,
            self.timedim,
            self.timecoord,
            self.timechunks,
            self.ydim,
            self.xdim,
            self.cell_area,
            self.lat,
            self.lon,
            self.mean_cell_area,
            self.neighbours_int,
            self.nn_partitioning,
            self.overlap_threshold,
            self.regional_mode,
            self.max_iteration,
            self.temp_merge_path,
        )
