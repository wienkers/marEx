"""
MarEx Track: preprocessing for 4-D (time, depth, lat, lon) events.

The horizontal closing/opening and hole fill (``R_fill``) and the temporal closing
(``T_fill``) reuse :func:`marEx.track.morphology.fill_holes` and
:func:`marEx.track.morphology.fill_time_gaps`, which broadcast over the depth axis, so
this module only adds the size filter. Small events are removed by **volume**: objects
are the connected components of each timestep, keyed by ``(time, local label)``
because labels restart at 1 every step, and an object is kept when the sum of its cell
volumes (m^3) reaches the threshold.
"""

from typing import NamedTuple, Optional

import dask
import dask.array as dsa
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ...exceptions import ConfigurationError, TrackingError
from ...logging_config import get_logger
from .config import VolumetricDims
from .geometry import VerticalGeometry
from .labelling import label_volumes_lazy

logger = get_logger(__name__)

# Largest number of objects one timestep may hold in the per-step volume census. The census
# is padded to this width so every step returns an array of one shape; a step that exceeds
# it is reported rather than silently truncated.
_OBJECT_BUFFER = 8192


class VolumeFilterResult(NamedTuple):
    """
    Outcome of :func:`filter_small_volumes`.

    Attributes
    ----------
    filtered : xarray.DataArray
        Boolean field with small objects removed; same dimensions and chunks as the input.
    threshold : float
        Volume threshold in m^3; objects with ``volume >= threshold`` are kept.
    volumes : numpy.ndarray
        Volume (m^3, float64) of every object before filtering, pooled over time in
        ``(time, local label)`` order.
    n_before : int
        Number of objects before filtering.
    n_after : int
        Number of objects kept.
    """

    filtered: xr.DataArray
    threshold: float
    volumes: NDArray[np.float64]
    n_before: int
    n_after: int


def _object_volumes(labels: NDArray[np.int32], cell_volume: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Volume of every object of one timestep, indexed by label (entry 0 is background).

    ``np.bincount`` accumulates the weights sequentially in C order, so the sums are
    deterministic.
    """
    n_labels = int(labels.max()) if labels.size else 0
    return np.bincount(labels.ravel(), weights=cell_volume.ravel(), minlength=n_labels + 1)


def _padded(volumes: NDArray[np.float64], buffer: int) -> NDArray[np.float64]:
    """Object volumes (background dropped) padded with zeros to ``buffer`` entries."""
    out = np.zeros(buffer, dtype=np.float64)
    objects = volumes[1:]
    out[: min(len(objects), buffer)] = objects[:buffer]
    return out


def filter_small_volumes(
    data: xr.DataArray,
    geom: VerticalGeometry,
    max_gap_m: float,
    wrap_x: bool,
    dims: VolumetricDims,
    *,
    quantile: Optional[float],
    absolute: Optional[float],
    materialiser=None,
) -> VolumeFilterResult:
    """
    Remove objects smaller than a volume threshold.

    Each timestep is labelled into connected components (see
    :func:`marEx.track.volumetric.labelling.label_volume`) and the volume of an object is
    the sum of its cell volumes. Labels restart at 1 every step, so objects are identified
    by ``(time, local label)`` and the quantile pools the volumes of all steps.

    Parameters
    ----------
    data : xarray.DataArray
        Boolean extremes field in internal order ``(time, z, y, x)`` with ``z`` ascending
        positive-down depth and ``z``, ``y``, ``x`` each a single chunk.
    geom : VerticalGeometry
        Supplies the wet mask, layer thickness (for the vertical gap rule) and cell volumes.
    max_gap_m : float
        Vertical gap threshold in metres.
    wrap_x : bool
        Join across the periodic-longitude seam (False in regional mode).
    dims : VolumetricDims
        Dimension names of ``data``.
    quantile : float, optional
        Threshold is ``np.percentile(volumes, quantile * 100)`` over the pooled census.
    absolute : float, optional
        Threshold in m^3. Exactly one of ``quantile`` and ``absolute`` must be given.
    materialiser : marEx.core.compute_mode.Materialiser, optional
        Decides how the labelled intermediates are held. With an absolute threshold the keep
        mask and the volume census come from one labelling pass and are staged together;
        with a quantile the threshold needs the census first, so the labelling runs twice
        (nothing is cached). ``None`` persists.

    Returns
    -------
    VolumeFilterResult

    Raises
    ------
    ConfigurationError
        If the threshold arguments or the data layout are invalid.
    TrackingError
        If the data holds no object, if one timestep holds more than the census buffer, or if
        the threshold removes every object.
    """
    if (quantile is None) == (absolute is None):
        raise ConfigurationError(
            "Exactly one of quantile and absolute must be given to the volume filter",
            context={"quantile": quantile, "absolute": absolute},
        )
    if quantile is not None and not 0.0 <= float(quantile) <= 1.0:
        raise ConfigurationError(
            f"quantile must be in [0, 1], got {quantile!r}",
            suggestions=["Use e.g. 0.5 to drop the smaller half of the events"],
        )
    expected = dims.internal_order()
    if tuple(data.dims) != expected:
        raise ConfigurationError(
            f"The volume filter needs dimensions in internal order {expected}, got {tuple(data.dims)}",
            suggestions=["Pass the output of prepare_volumetric_data"],
        )
    if data.chunks is not None:
        split = [d for d in expected[1:] if len(data.chunksizes[d]) > 1]
        if split:
            raise ConfigurationError(
                f"4-D dimensions {split} must be a single chunk",
                suggestions=[f"Rechunk with data.chunk({{{', '.join(repr(d) + ': -1' for d in split)}}})"],
            )

    buffer = _OBJECT_BUFFER
    core = [dims.z, dims.y, dims.x]
    whole = {d: -1 for d in core}
    cell_volume = xr.DataArray(np.ascontiguousarray(geom.cell_volume, dtype=np.float64), dims=core).chunk(whole)
    labels = label_volumes_lazy(data, geom, max_gap_m, wrap_x, dims.z, dims.y, dims.x)

    def census(label_field, volume_field):
        """Return the padded per-object volumes and the object count of one timestep."""
        volumes = _object_volumes(label_field, volume_field)
        return _padded(volumes, buffer), np.int64(len(volumes) - 1)

    def census_and_keep(label_field, volume_field):
        """Return the keep mask for the absolute threshold, with the census, from one labelled step."""
        volumes = _object_volumes(label_field, volume_field)
        keep = volumes >= absolute
        keep[0] = False  # background
        return keep[label_field], _padded(volumes, buffer), np.int64(len(volumes) - 1)

    census_kwargs = dict(
        input_core_dims=[core, core],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"object_buffer": buffer}},
    )
    filtered = None
    if absolute is not None:
        # The threshold is known up front, so one labelling pass gives both outputs. They
        # are materialised together: computing them separately would label every step twice.
        filtered, padded, counts = xr.apply_ufunc(
            census_and_keep,
            labels,
            cell_volume,
            output_core_dims=[core, ["object_buffer"], []],
            output_dtypes=[np.bool_, np.float64, np.int64],
            **census_kwargs,
        )
        filtered = filtered.transpose(*data.dims)
        if materialiser is None:
            filtered, padded, counts = dask.persist(filtered, padded, counts)
        else:
            filtered, padded, counts = materialiser.stage_many(
                {"data_bin_filtered": filtered, "padded_volumes": padded, "object_counts": counts},
                "small_volume_filter",
                preserve_chunks=True,
            )
    else:
        padded, counts = xr.apply_ufunc(
            census,
            labels,
            cell_volume,
            output_core_dims=[["object_buffer"], []],
            output_dtypes=[np.float64, np.int64],
            **census_kwargs,
        )

    # Drop the padding inside each chunk before gathering, so the driver receives one float
    # per object rather than the whole padded census.
    counts_dask = dsa.asarray(counts.data)
    padded_dask = dsa.asarray(padded.data)
    occupied = dsa.arange(buffer)[None, :] < counts_dask[:, None]
    volumes, overflow = dask.compute(padded_dask[occupied], (counts_dask > buffer).any())
    if overflow:
        raise TrackingError(
            "Per-timestep object-volume buffer overflow during small-volume filtering",
            details={"object_buffer": buffer},
            suggestions=["Increase _OBJECT_BUFFER in marEx.track.volumetric.preprocess"],
        )
    volumes = np.asarray(volumes, dtype=np.float64)
    n_before = int(volumes.size)
    if n_before == 0:
        raise TrackingError(
            "No objects found for volume-based filtering",
            suggestions=[
                "Check that the input contains extreme events inside the wet mask",
                "Consider lowering the extreme threshold percentile",
            ],
        )

    threshold = float(absolute) if absolute is not None else float(np.percentile(volumes, float(quantile) * 100.0))
    n_after = int(np.sum(volumes >= threshold))
    setting = f"volume_filter_absolute={absolute}" if absolute is not None else f"volume_filter_quartile={quantile}"
    if n_after == 0:
        raise TrackingError(
            f"The volume filter removed every object ({setting}, threshold {threshold:.6g} m^3)",
            details=f"All {n_before} objects are smaller than the threshold; the largest is {volumes.max():.6g} m^3.",
            suggestions=[
                "Lower volume_filter_absolute, or use volume_filter_quartile",
                "Check the units: the threshold is in cubic metres",
            ],
            context={"n_objects": n_before, "largest_volume_m3": float(volumes.max())},
        )

    if filtered is None:  # quantile mode: the threshold needed the census first

        def keep_large(label_field, volume_field):
            """Keep the objects of one timestep whose volume is >= the threshold."""
            keep = _object_volumes(label_field, volume_field) >= threshold
            keep[0] = False  # background
            return keep[label_field]

        filtered = xr.apply_ufunc(
            keep_large,
            labels,
            cell_volume,
            input_core_dims=[core, core],
            output_core_dims=[core],
            output_dtypes=[np.bool_],
            vectorize=True,
            dask="parallelized",
        ).transpose(*data.dims)

    logger.info(f"Volume filter: kept {n_after} of {n_before} objects (threshold {threshold:.6g} m^3)")
    return VolumeFilterResult(filtered, threshold, volumes, n_before, n_after)
