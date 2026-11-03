"""
MarEx Track: the 4-D tracker.

:class:`VolumetricTracker` is what ``marEx.tracker(...)`` returns when ``dimensions``
declares a ``"z"`` entry. It validates and prepares a ``(time, depth, lat, lon)``
extremes field, builds the vertical geometry and sets up execution. ``run()`` (inherited
from :class:`marEx.tracker`) then preprocesses the field, identifies volumes per timestep
(:func:`identify_volumes`), resolves splits and merges, clusters objects into events and
computes the 4-D diagnostics.
"""

import os
import uuid
import warnings
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr

from ...core.compute_mode import Materialiser, create_staging_dir
from ...core.encoding import write_zarr
from ...exceptions import ConfigurationError, TrackingError
from ...logging_config import configure_logging, get_logger, log_dask_info, log_memory_usage, log_timing
from .. import grid as _grid
from .. import morphology as _morphology
from .. import validation as _validation
from ..objects import _anchor
from ..tracker import _format_chunk_pattern, _is_uniform_time_chunking, _prune_stale_temp_stores, tracker
from .config import VolumetricDims, check_volumetric_config, prepare_volumetric_data
from .diagnostics import event_diagnostics
from .events import cluster_events
from .geometry import EARTH_RADIUS_M, VerticalGeometry, _z_order, build_vertical_geometry
from .ids import IDAllocator
from .labelling import label_volumes_lazy
from .merge_loop import (
    _RETAINED_CHUNKS,
    _STATIC_FLOAT64_PER_VOXEL,
    final_links,
    merge_loop_client_bytes,
    split_and_merge_volumetric,
)
from .partition import ScaledCoords
from .preprocess import filter_small_volumes
from .store import VolumeStore

logger = get_logger(__name__)

# Client memory (bytes) above which construction warns that the merge loop's chunk buffers and
# static geometry may not fit on the client. See merge_loop_client_bytes for the model.
CLIENT_WARN_BYTES: int = 8 * 2**30

# Statistics written to and read from the preprocessing checkpoint, in this order.
_CHECKPOINT_STATS = (
    "total_volume_IDed",
    "N_objects_prefiltered",
    "N_objects_filtered",
    "volume_threshold",
    "accepted_volume_fraction",
    "preprocessed_volume_fraction",
)


def identify_volumes(
    data: xr.DataArray,
    geom: VerticalGeometry,
    max_gap_m: float,
    wrap_x: bool,
    dims: VolumetricDims,
    allocator: IDAllocator,
    *,
    materialiser,
) -> xr.DataArray:
    """
    Label the volumes of every timestep with IDs unique across time and depth.

    Parameters
    ----------
    data : xarray.DataArray
        Boolean extremes field in internal order ``(time, z, y, x)`` with ascending
        positive-down depth, as returned by :func:`prepare_volumetric_data`.
    geom : VerticalGeometry
        Vertical geometry on the same grid.
    max_gap_m : float
        Largest vertical gap (m) of wet cells bridged between two occupied cells.
    wrap_x : bool
        Join objects across the periodic-longitude seam.
    dims : VolumetricDims
        Dimension names.
    allocator : IDAllocator
        Source of IDs. Each timestep's labels are shifted by the IDs already issued, so an
        object spanning several levels has one ID and later timesteps start above earlier ones.
    materialiser : Materialiser or None
        Anchors the result: staged to disk in streaming mode, persisted otherwise.

    Returns
    -------
    xarray.DataArray
        int32 ``ID_field`` with ``data``'s dimensions, 0 for background.

    Notes
    -----
    The per-timestep object counts are computed first (one labelling pass), then the
    shifted field is anchored (a second pass). Both are deterministic and identical.
    """
    labels = label_volumes_lazy(data, geom, max_gap_m, wrap_x, dims.z, dims.y, dims.x)
    # Local labels are compact (1..n per timestep), so the per-timestep maximum is the object
    # count. The counts are a small nt-length vector and are summed in Python ints by the allocator.
    counts = labels.max(dim=[dims.z, dims.y, dims.x]).values
    offsets = allocator.reserve_offsets([int(c) for c in counts])
    offsets_da = xr.DataArray(offsets, dims=[dims.time]).chunk({dims.time: labels.chunksizes[dims.time]})
    field = xr.where(labels > 0, labels + offsets_da, 0).astype(np.int32).transpose(*data.dims).rename("ID_field")
    return _anchor(field, "volume_id_field", materialiser)


class VolumetricTracker(tracker):
    """
    Tracker for 4-D ``(time, depth, lat, lon)`` events.

    Created by ``marEx.tracker(...)`` when ``dimensions`` contains a ``"z"`` key. The
    parameters are those of :class:`marEx.tracker`, plus the keyword-only 4-D
    parameters below. 4-D tracking needs a wet ``mask`` with dimensions
    ``(depth, lat, lon)``, ``allow_merging=True``, and a structured lat/lon grid.
    ``nn_partitioning`` defaults to ``True`` here and ``False`` raises.

    Parameters
    ----------
    layer_bounds : xarray.DataArray or ``"from_centres"``
        Layer interfaces in metres, ``(z, 2)`` or ``(z, y, x, 2)``, passed explicitly.
        ``"from_centres"`` reconstructs them at midpoints between level centres (approximate).
    layer_thickness : xarray.DataArray, optional
        ``(z, y, x)`` thickness in metres for partial cells.
    cell_volumes : xarray.DataArray, optional
        ``(z, y, x)`` volumes in m^3, replacing area times thickness.
    max_vertical_gap_m : float, default=0.0
        Largest total thickness (m) of unoccupied wet cells bridged within a column.
        0 joins consecutive stored levels only.
    volume_filter_quartile : float, optional
        Quantile of the pooled per-(time, object) volumes below which events are dropped.
        Default 0.5 when neither volume filter is given.
    volume_filter_absolute : float, optional
        Minimum event volume in m^3. Mutually exclusive with ``volume_filter_quartile``.
    partition_scales : tuple of float, optional
        ``(L_H, L_Z)`` in metres scaling the nearest-cell partition metric. ``None``
        derives both from the grid.
    """

    def __init__(
        self,
        data_bin: xr.DataArray,
        mask: Optional[xr.DataArray] = None,
        R_fill: Optional[Union[int, float]] = None,
        area_filter_quartile: Optional[float] = None,
        area_filter_absolute: Optional[int] = None,
        temp_dir: Optional[str] = None,
        T_fill: int = 2,
        allow_merging: bool = True,
        nn_partitioning: bool = True,
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
        prefilter_min_cells: Optional[int] = None,
        layer_bounds: Union[xr.DataArray, Literal["from_centres"], None] = None,
        layer_thickness: Optional[xr.DataArray] = None,
        cell_volumes: Optional[xr.DataArray] = None,
        max_vertical_gap_m: float = 0.0,
        volume_filter_quartile: Optional[float] = None,
        volume_filter_absolute: Optional[float] = None,
        partition_scales: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Validate the configuration, prepare the data and build the vertical geometry."""
        # The parent __init__ is deliberately not called: it rejects the depth dimension.
        if mask is not None and not isinstance(mask, xr.DataArray):
            raise ConfigurationError(
                f"mask must be an xarray.DataArray, got {type(mask).__name__}",
                details="`mask` is the second positional parameter and `R_fill` the third",
                suggestions=["Pass R_fill by keyword: tracker(data_bin, mask, R_fill=8, dimensions={..., 'z': ...})"],
            )
        if R_fill is None:
            raise ConfigurationError(
                "R_fill is required",
                details="R_fill sets the morphological kernel radius, in grid cells; there is no sensible default",
                suggestions=["Pass R_fill explicitly, e.g. tracker(data_bin, mask, R_fill=8, ...)"],
            )
        (self.volume_filter_quartile, self.volume_filter_absolute) = check_volumetric_config(
            area_filter_quartile=area_filter_quartile,
            area_filter_absolute=area_filter_absolute,
            volume_filter_quartile=volume_filter_quartile,
            volume_filter_absolute=volume_filter_absolute,
            allow_merging=allow_merging,
            nn_partitioning=nn_partitioning,
            unstructured_grid=unstructured_grid,
            prefilter_min_cells=prefilter_min_cells,
            max_vertical_gap_m=max_vertical_gap_m,
            partition_scales=partition_scales,
        )
        if neighbours is not None:
            logger.warning("neighbours is ignored: 4-D tracking uses structured lat/lon grids only")

        if verbose is not None or quiet is not None:
            configure_logging(verbose=verbose, quiet=quiet)
        self.verbose = verbose
        self.quiet = quiet
        logger.info("Initialising MarEx 4-D tracker")
        logger.info(
            f"Parameters: R_fill={R_fill}, T_fill={T_fill}, volume_filter_quartile={self.volume_filter_quartile}, "
            f"volume_filter_absolute={self.volume_filter_absolute}, max_vertical_gap_m={max_vertical_gap_m}"
        )
        log_dask_info(logger, data_bin, "Binary input data")
        log_memory_usage(logger, "Tracker initialisation")

        # Dimension and coordinate names.
        dimensions = dimensions or {}
        self.dims = VolumetricDims(
            time=dimensions.get("time", "time"),
            z=dimensions["z"],
            y=dimensions.get("y", "lat"),
            x=dimensions.get("x", "lon"),
        )
        self.timedim, self.zdim, self.ydim, self.xdim = self.dims.internal_order()
        coordinates = coordinates or {}
        self.timecoord = coordinates.get("time", self.timedim)
        self.xcoord = coordinates.get("x", self.xdim)
        self.ycoord = coordinates.get("y", self.ydim)
        self.regional_mode = regional_mode
        self.coordinate_units = coordinate_units
        self.unstructured_grid = False

        # Order, rank, depth coordinate, mask and chunking checks.
        data_bin, wet_mask, self.z_values, self.z_attrs, self.original_dims = prepare_volumetric_data(data_bin, mask, self.dims)
        self.data_attrs = dict(data_bin.attrs)

        _validation.validate_required_coordinates(data_bin, self.timecoord, self.xcoord, self.ycoord)
        if self.timecoord != self.timedim and (
            self.timedim not in data_bin.indexes or data_bin.indexes[self.timedim].name != self.timecoord
        ):
            logger.warning(
                f"timecoord '{self.timecoord}' is not an index of timedim '{self.timedim}'. "
                f"Setting '{self.timecoord}' as index for dimension '{self.timedim}'"
            )
            data_bin = data_bin.set_index({self.timedim: self.timecoord})

        self.lat_init = data_bin[self.ycoord].compute()  # original units
        self.lon_init = data_bin[self.xcoord].compute()
        self.coordinate_units, data_bin = _grid.unify_coordinates(
            data_bin, self.regional_mode, self.coordinate_units, self.xcoord, self.ycoord
        )
        self.lat = data_bin[self.ycoord].compute()
        self.lon = data_bin[self.xcoord].compute()

        self.R_fill = int(R_fill)
        self.T_fill = T_fill
        self.allow_merging = True
        self.nn_partitioning = True
        self.overlap_threshold = overlap_threshold
        self.max_vertical_gap_m = float(max_vertical_gap_m)
        self.max_iteration = max_iteration
        self.checkpoint = checkpoint
        self.debug = debug

        # Vertical geometry, in internal depth order.
        self.geom = build_vertical_geometry(
            self.z_values,
            self.z_attrs,
            wet_mask,
            zdim=self.zdim,
            ydim=self.ydim,
            xdim=self.xdim,
            lat_deg=np.asarray(self.lat.values, dtype=np.float64),
            lon_deg=np.asarray(self.lon.values, dtype=np.float64),
            layer_bounds=layer_bounds,
            layer_thickness=layer_thickness,
            cell_volumes=cell_volumes,
            cell_areas=cell_areas,
            grid_resolution=grid_resolution,
        )
        # The data was reordered in prepare_volumetric_data by the same rule the geometry used.
        assert _z_order(self.z_values, self.z_attrs)[1] == self.geom.flip_z
        self.partition_scales = (
            (float(partition_scales[0]), float(partition_scales[1])) if partition_scales is not None else self.geom.default_scales
        )
        self.allocator = IDAllocator(1)

        # Time chunking (identical rules to the 3-D tracker).
        time_chunks = data_bin.chunks[data_bin.dims.index(self.timedim)]
        if not _is_uniform_time_chunking(time_chunks):
            width = int(max(time_chunks))
            logger.warning(
                f"Input time chunks are not uniform ({_format_chunk_pattern(time_chunks)}); "
                f"rechunking {self.timedim!r} to {width} for the zarr region writes"
            )
            data_bin = data_bin.chunk({self.timedim: width})
            time_chunks = data_bin.chunks[data_bin.dims.index(self.timedim)]
        self.timechunks = int(time_chunks[0])
        self.data_bin = data_bin
        # The wet mask on the data's own coordinate labels, so the morphology's mask (an
        # xarray `where`, which aligns by label) cannot drop or misalign levels whose depth
        # labels differ only in dtype or rounding from the mask's.
        self.mask = wet_mask.drop_vars(list(wet_mask.coords)).assign_coords(
            {d: data_bin[d] for d in (self.zdim, self.ydim, self.xdim) if d in data_bin.coords}
        )

        # Execution: compute mode, scratch and staging.
        # Same rule as the 3-D tracker: checkpoint files live in temp_dir.
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
                suggestions=["Use 'persist' (default) or 'streaming'"],
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
        self.compute_mode = compute_mode
        self.staging_dir = create_staging_dir(temp_dir) if compute_mode == "streaming" else None
        self.materialiser = Materialiser(compute_mode, self.staging_dir)
        if temp_dir:
            _prune_stale_temp_stores(temp_dir)
            run_token = uuid.uuid4().hex[:12]
            self.temp_refresh_path = os.path.join(temp_dir, f"marEx_temp_refresh_{run_token}.zarr")
            self.temp_merge_path = os.path.join(temp_dir, f"marEx_temp_field_{run_token}.zarr")
        else:
            self.temp_refresh_path = None
            self.temp_merge_path = None

        self._warn_client_memory()
        self._configure_warnings()

    def _warn_client_memory(self) -> None:
        """Warn when the merge loop's modelled client memory exceeds ``CLIENT_WARN_BYTES``."""
        n_voxels = int(np.prod(self.geom.cell_volume.shape))
        needed = merge_loop_client_bytes(n_voxels, self.timechunks, self.compute_mode)
        budget = CLIENT_WARN_BYTES
        if needed <= budget:
            return
        retained = _RETAINED_CHUNKS[self.compute_mode]
        message = (
            f"The 4-D merge loop is modelled to need {needed:,} bytes on the client, above the "
            f"{budget:,}-byte budget: {retained} retained chunks x {self.timechunks} time steps x "
            f"{n_voxels} voxels x 4 B (int32 IDs) + {_STATIC_FLOAT64_PER_VOXEL} x {n_voxels} voxels x 8 B "
            "(float64 cell volumes and layer top, bottom, thickness and midpoint depths). Per-merge partition buffers come on top. "
            f"Reduce the time chunk of the input (now {self.timechunks})"
            + (", or use compute_mode='streaming', which retains 2 chunks" if self.compute_mode == "persist" else "")
            + "."
        )
        logger.warning(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    def identify_volumes(self, data_bin: xr.DataArray) -> xr.DataArray:
        """Label the volumes of ``data_bin`` (internal order) with IDs unique across time and depth."""
        return identify_volumes(
            data_bin,
            self.geom,
            self.max_vertical_gap_m,
            not self.regional_mode,
            self.dims,
            self.allocator,
            materialiser=self.materialiser,
        )

    # ============================
    # Pipeline steps (driven by the inherited run())
    # ============================

    def _cell_volume_da(self) -> xr.DataArray:
        """Cell volumes (m^3) as a single-chunk dask ``(z, y, x)`` array, never broadcast over time."""
        core = (self.zdim, self.ydim, self.xdim)
        values = np.ascontiguousarray(self.geom.cell_volume, dtype=np.float64)
        return xr.DataArray(values, dims=core).chunk({d: -1 for d in core})

    def _total_volume(self, data_bin: xr.DataArray) -> xr.DataArray:
        """Occupied volume (m^3) per timestep. Dry cells have zero volume, so they never count."""
        core = [self.zdim, self.ydim, self.xdim]
        occupied = data_bin.drop_vars([c for c in data_bin.coords if c in core])
        return (occupied * self._cell_volume_da()).sum(dim=core)

    def _checkpoint_paths(self) -> Tuple[str, str]:
        return (
            f"{self.scratch_dir}/marEx_checkpoint_proc_bin.zarr",
            f"{self.scratch_dir}/marEx_checkpoint_stats.npz",
        )

    def run_preprocess(self, checkpoint: Optional[str] = None) -> Tuple[xr.DataArray, Tuple[float, int, int, float, float, float]]:
        """
        Preprocess the 4-D extremes field.

        In order: the horizontal closing/opening with radius ``R_fill`` on every level, the
        temporal closing with ``T_fill`` followed by a horizontal pass with ``R_fill // 2``
        (both inside :func:`marEx.track.morphology.fill_time_gaps`), the wet mask, and the
        volume filter (:func:`filter_small_volumes`).

        Parameters
        ----------
        checkpoint : str, optional
            ``"save"`` writes the preprocessed field and its statistics to ``temp_dir``;
            ``"load"`` reads them back instead of preprocessing. Defaults to the instance setting.

        Returns
        -------
        data_bin_filtered : xarray.DataArray
            Preprocessed boolean field in internal order ``(time, z, y, x)``.
        object_stats : tuple
            ``(total_volume_IDed, N_objects_prefiltered, N_objects_filtered, volume_threshold,
            accepted_volume_fraction, preprocessed_volume_fraction)``; volumes in m^3.
        """
        if not checkpoint:
            checkpoint = self.checkpoint
        field_path, stats_path = self._checkpoint_paths()

        def load_data_from_checkpoint() -> xr.DataArray:
            return xr.open_zarr(field_path, chunks={self.timedim: self.timechunks})["data_bin_preproc"]

        if checkpoint == "load":
            logger.info("Loading preprocessed data from checkpoint")
            stats = np.load(stats_path)
            values = [stats[key].item() for key in _CHECKPOINT_STATS]
            object_stats = (float(values[0]), int(values[1]), int(values[2]), float(values[3]), float(values[4]), float(values[5]))
            return load_data_from_checkpoint(), object_stats  # type: ignore[return-value]

        # Mask before the morphology too: a True on a dry cell (a boolean that was never masked)
        # would otherwise take part in the closing and fill the wet cells between it and an event.
        data_bin = self.data_bin.where(self.mask, other=False).transpose(*self.dims.internal_order())
        raw_volume = self._total_volume(data_bin)

        logger.info(f"Filling spatial holes on every level with radius R_fill={self.R_fill}")
        with log_timing(logger, "Spatial hole filling"):
            filled = _morphology.fill_holes(data_bin, self.R_fill, False, None, self.xdim, self.mask, self.regional_mode, self.ydim)
            del data_bin
            filled, raw_volume = self.materialiser.pin(filled, raw_volume)
            if self.materialiser.is_streaming:
                raw_volume = raw_volume.persist()
            self.data_bin = None  # Free memory (tracker instance is now single-run)

        logger.info(f"Filling temporal gaps with T_fill={self.T_fill}")
        with log_timing(logger, "Temporal gap filling"):
            filled = _morphology.fill_time_gaps(
                filled,
                self.T_fill,
                self.R_fill,
                self.timedim,
                self.ydim,
                False,
                None,
                self.xdim,
                self.mask,
                self.regional_mode,
                materialiser=self.materialiser,
            )
            filled = filled.where(self.mask, other=False).transpose(*self.dims.internal_order())
            filled = self.materialiser.stage(filled, "data_bin_filled", preserve_chunks=True)

        logger.info("Filtering small volumes")
        with log_timing(logger, "Small volume filtering"):
            result = filter_small_volumes(
                filled,
                self.geom,
                self.max_vertical_gap_m,
                not self.regional_mode,
                self.dims,
                quantile=self.volume_filter_quartile,
                absolute=self.volume_filter_absolute,
                materialiser=self.materialiser,
            )
            del filled
        data_bin_filtered = result.filtered

        if checkpoint and "save" in checkpoint:
            logger.info("Saving preprocessed data to checkpoint")
            with log_timing(logger, "Checkpoint saving"):
                write_zarr(data_bin_filtered.rename("data_bin_preproc"), field_path, mode="w")
                data_bin_filtered = load_data_from_checkpoint()
        else:
            data_bin_filtered = self.materialiser.stage(data_bin_filtered, "data_bin_filtered", preserve_chunks=True)

        volumes = result.volumes
        total_volume_IDed = float(volumes.sum())
        accepted_volume_fraction = float(volumes[volumes >= result.threshold].sum()) / total_volume_IDed
        total_raw_volume = float(raw_volume.sum().compute().item())
        total_processed_volume = float(self._total_volume(data_bin_filtered).sum().compute().item())
        if total_processed_volume <= 0:
            raise TrackingError(
                "The objects kept after filtering have zero total volume",
                details="Their cell volumes sum to zero, so the preprocessed volume fraction is undefined.",
                suggestions=["Check that the wet mask covers the extremes and that cell volumes are positive"],
            )
        object_stats = (
            total_volume_IDed,
            int(result.n_before),
            int(result.n_after),
            float(result.threshold),
            accepted_volume_fraction,
            total_raw_volume / total_processed_volume,
        )
        if checkpoint and "save" in checkpoint:
            np.savez(stats_path, **dict(zip(_CHECKPOINT_STATS, object_stats)))
        return data_bin_filtered, object_stats

    def _volume_store(self, id_field: xr.DataArray) -> VolumeStore:
        """Volume of every object in ``id_field``, read one time chunk at a time."""
        store = VolumeStore()
        cell_volume = self.geom.cell_volume
        start = 0
        for size in id_field.chunks[0]:
            values = np.asarray(id_field.isel({self.timedim: slice(start, start + size)}).values)
            for t in range(size):
                store.update_from_slice(values[t], cell_volume)
            start += size
        return store

    def _restore_layout(self, events_ds: xr.Dataset) -> xr.Dataset:
        """Put the depth axis back in the input's order and labels, and the input's dimension order."""
        if self.geom.flip_z:
            events_ds = events_ds.isel({self.zdim: slice(None, None, -1)})
        z = xr.DataArray(np.asarray(self.z_values), dims=self.zdim, attrs=dict(self.z_attrs))
        events_ds = events_ds.assign_coords({self.zdim: z})
        events_ds["ID_field"] = events_ds["ID_field"].transpose(*self.original_dims)
        return events_ds

    def run_tracking(self, data_bin_preprocessed: xr.DataArray) -> Tuple[xr.Dataset, xr.Dataset, int]:
        """
        Identify volumes, resolve splits and merges, and cluster objects into events.

        Parameters
        ----------
        data_bin_preprocessed : xarray.DataArray
            Preprocessed boolean field in internal order ``(time, z, y, x)``.

        Returns
        -------
        events_ds : xarray.Dataset
            ``ID_field`` (input dimension order and depth order), ``global_ID``, ``presence``,
            ``time_start``, ``time_end``, ``merge_ledger`` and the 4-D diagnostics.
        merges_ds : xarray.Dataset
            The merge ledger in object IDs (as the 3-D tracker returns its merge events):
            ``parent_IDs``, ``child_IDs``, ``overlap_volumes`` (m^3), ``merge_time``,
            ``n_parents``, ``n_children``, the same rows in event IDs (``parent_event_IDs``,
            ``child_event_IDs``, ``n_parent_events``), and the splits (``split_time``,
            ``split_parent_IDs``, ``split_child_IDs``, ``split_event_IDs``, ``n_split_children``).
        N_events_final : int
            Number of events.
        """
        geom = self.geom
        lat_deg = np.asarray(self.lat.values, dtype=np.float64)
        lon_deg = np.asarray(self.lon.values, dtype=np.float64)

        with log_timing(logger, "Volume identification"):
            id_field = self.identify_volumes(data_bin_preprocessed)
            del data_bin_preprocessed
            store = self._volume_store(id_field)
        logger.info(f"Identified {len(store)} objects")

        coords = ScaledCoords(lat_deg, lon_deg, geom.midpoint, *self.partition_scales)
        with log_timing(logger, "4-D split and merge"):
            merged, store, ledger = split_and_merge_volumetric(
                id_field,
                store,
                self.allocator,
                geom,
                coords,
                self.overlap_threshold,
                self.dims,
                materialiser=self.materialiser,
                id_field_path=None,
                max_gap_m=self.max_vertical_gap_m,
                wrap_x=not self.regional_mode,
            )
            del id_field
        links = final_links(merged, store, geom.cell_volume, self.overlap_threshold, self.timedim)
        merges_ds = ledger.to_dataset()

        with log_timing(logger, "Clustering objects into events"):
            events_ds = cluster_events(
                merged, links, merges_ds, self.dims, self.timecoord, self.timechunks, materialiser=self.materialiser
            )
        # Filler IDs < 0 become background.
        events_ds["ID_field"] = events_ds["ID_field"].where(events_ds["ID_field"] > 0, other=0)

        with log_timing(logger, "Event diagnostics"):
            diagnostics = event_diagnostics(
                events_ds["ID_field"],
                events_ds["presence"],
                geom,
                lat_deg,
                lon_deg,
                self.dims,
                materialiser=self.materialiser,
            )
        for name in diagnostics.data_vars:
            events_ds[name] = diagnostics[name]

        events_ds = events_ds.chunk({"ID": -1, "component": -1, "sibling_ID": -1, self.zdim: -1, self.ydim: -1, self.xdim: -1})
        N_events_final = int(events_ds["ID_field"].max().compute().item())
        events_ds = self._restore_layout(events_ds)

        # Restore the original time coordinate name, as the 3-D tracker does.
        if self.timecoord != self.timedim and self.timedim in events_ds.coords and self.timecoord not in events_ds.coords:
            events_ds = events_ds.assign_coords({self.timecoord: events_ds.coords[self.timedim]})
            events_ds = events_ds.drop_vars(self.timedim)

        logger.info("Finished tracking all 4-D events")
        return events_ds, merges_ds, N_events_final

    def run_stats_attributes(
        self,
        events_ds: xr.Dataset,
        merges_ds: xr.Dataset,
        object_stats: Tuple[float, int, int, float, float, float],
        N_events_final: int,
    ) -> xr.Dataset:
        """
        Record the run's settings and statistics as attributes and restore the lat/lon coordinates.

        Parameters
        ----------
        events_ds : xarray.Dataset
            Events from :meth:`run_tracking`.
        merges_ds : xarray.Dataset
            The merge ledger.
        object_stats : tuple
            Statistics from :meth:`run_preprocess`.
        N_events_final : int
            Number of events.

        Returns
        -------
        xarray.Dataset
            The events with attributes, chunked one timestep per chunk.
        """
        (
            total_volume_IDed,
            N_objects_prefiltered,
            N_objects_filtered,
            volume_threshold,
            accepted_volume_fraction,
            preprocessed_volume_fraction,
        ) = object_stats
        gap = self.max_vertical_gap_m
        seam = "" if self.regional_mode else ", joined across the periodic longitude seam"
        attrs = {
            "allow_merging": 1,
            "nn_partitioning": 1,
            "spatial_connectivity": (
                f"8-neighbour horizontal on each level{seam}; vertical within a water column between occupied "
                f"cells separated by wet unoccupied cells of total thickness <= max_vertical_gap_m = {gap} m "
                "(dry cells are barriers)"
            ),
            "max_vertical_gap_m": gap,
            "vertical_geometry": self.geom.source,
            "volume_units": "m3",
            "partition_metric": (
                "nearest parent voxel; horizontal distance is the chord on a sphere of radius "
                f"{EARTH_RADIUS_M} m divided by L_H, vertical distance is the layer-midpoint depth difference divided by L_Z"
            ),
            "partition_scales_m": np.asarray(self.partition_scales, dtype=np.float64),
            "overlap_threshold": self.overlap_threshold,
        }
        if self.volume_filter_quartile is not None:
            attrs["volume_filter_quartile"] = self.volume_filter_quartile
        else:
            attrs["volume_filter_absolute"] = self.volume_filter_absolute
        attrs.update(
            {
                "volume_threshold (m^3)": float(volume_threshold),
                "accepted_volume_fraction": float(accepted_volume_fraction),
                "preprocessed_volume_fraction": float(preprocessed_volume_fraction),
                "R_fill": self.R_fill,
                "T_fill": self.T_fill,
                "N_objects_prefiltered": int(N_objects_prefiltered),
                "N_objects_filtered": int(N_objects_filtered),
                "N_events_final": int(N_events_final),
                "total_merges": len(merges_ds.merge_ID),
                "total_splits": int(merges_ds.sizes.get("split_ID", 0)),
                "multi_parent_merges": int((merges_ds.n_parents > 2).sum().item()),
            }
        )
        events_ds.attrs.update(attrs)

        logger.info(
            f"4-D tracking statistics: {N_objects_prefiltered} objects before and {N_objects_filtered} after "
            f"the volume filter (threshold {float(volume_threshold):.6g} m^3, total {float(total_volume_IDed):.6g} m^3, "
            f"accepted fraction {float(accepted_volume_fraction):.6g}); {N_events_final} events; "
            f"{attrs['total_merges']} merges"
        )

        events_ds.attrs.update(self.data_attrs)
        events_ds = self._remap_coordinates(events_ds)
        return events_ds.chunk({self.timedim: 1})


def _not_yet_volumetric(name: str):
    def method(self, *args, **kwargs):
        raise NotImplementedError(
            f"VolumetricTracker.{name} is not implemented yet: the inherited 3-D implementation does not apply to "
            "(time, depth, lat, lon) data"
        )

    method.__name__ = name
    method.__qualname__ = f"VolumetricTracker.{name}"
    return method


# The 3-D pipeline steps that have no 4-D meaning raise, so that nothing runs 3-D code
# on 4-D data. run() itself is inherited and drives the 4-D run_* methods above.
for _name in (
    "compute_area",
    "fill_holes",
    "fill_time_gaps",
    "refresh_dask_graph",
    "filter_small_objects",
    "identify_objects",
    "calculate_centroid",
    "calculate_object_properties",
    "check_overlap_slice",
    "find_overlapping_objects",
    "enforce_overlap_threshold",
    "consolidate_object_ids",
    "compute_id_time_dict",
    "track_objects",
    "cluster_rename_objects_and_props",
    "split_and_merge_objects",
    "split_and_merge_objects_parallel",
):
    setattr(VolumetricTracker, _name, _not_yet_volumetric(_name))
del _name
