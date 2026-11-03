"""
MarEx Track: configuration checks and data preparation for 4-D tracking.

4-D (``time``, ``depth``, ``lat``, ``lon``) tracking is selected by declaring a
``"z"`` entry in ``dimensions``. This module rejects the 3-D parameters that have no
meaning for volumes, resolves the volume-filter defaults, and brings the extremes field
and wet mask into the internal layout the labelling kernels expect.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ...exceptions import ConfigurationError, TrackingError, create_data_validation_error
from ...logging_config import get_logger
from .geometry import _z_order

logger = get_logger(__name__)


@dataclass(frozen=True)
class VolumetricDims:
    """Names of the four dimensions of a 4-D field."""

    time: str
    z: str
    y: str
    x: str

    def internal_order(self) -> Tuple[str, str, str, str]:
        """Return the internal dimension order ``(time, z, y, x)``."""
        return (self.time, self.z, self.y, self.x)


# Keyword arguments that only exist for 4-D tracking.
FOUR_D_ONLY_KWARGS: Tuple[str, ...] = (
    "layer_bounds",
    "layer_thickness",
    "cell_volumes",
    "max_vertical_gap_m",
    "volume_filter_quartile",
    "volume_filter_absolute",
    "partition_scales",
)

_DECLARE_Z = 'Declare depth as dimensions["z"] to track 4-D events'


def _is_real_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def check_volumetric_config(
    *,
    area_filter_quartile,
    area_filter_absolute,
    volume_filter_quartile,
    volume_filter_absolute,
    allow_merging,
    nn_partitioning,
    unstructured_grid,
    prefilter_min_cells,
    max_vertical_gap_m,
    partition_scales,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Reject parameters that are not valid for 4-D tracking.

    Returns
    -------
    (volume_filter_quartile, volume_filter_absolute)
        The resolved volume filter. When neither is given the default is
        ``(0.5, None)``, the median, mirroring the 3-D default.

    Raises
    ------
    ConfigurationError
        For every 3-D-only or invalid setting.
    """
    if area_filter_quartile is not None or area_filter_absolute is not None:
        raise ConfigurationError(
            "area_filter_quartile and area_filter_absolute are not used for 4-D tracking",
            details="Events are filtered by volume (m^3), not by horizontal area",
            suggestions=[
                "Use volume_filter_quartile (e.g. 0.5 for the median) or volume_filter_absolute (m^3)",
            ],
        )
    if unstructured_grid:
        raise ConfigurationError(
            "4-D tracking supports structured lat/lon grids only",
            details="unstructured_grid=True cannot be combined with dimensions['z']",
            suggestions=["Use unstructured_grid=False with a (time, depth, lat, lon) field"],
        )
    if not allow_merging:
        raise ConfigurationError(
            "4-D tracking requires allow_merging=True",
            details="4-D events are tracked through splits and merges",
            suggestions=["Pass allow_merging=True"],
        )
    if not nn_partitioning:
        raise ConfigurationError(
            "4-D tracking requires nn_partitioning=True",
            details="Merged volumes are partitioned among their parents by nearest cell",
            suggestions=["Pass nn_partitioning=True"],
        )
    if prefilter_min_cells is not None:
        raise ConfigurationError(
            "prefilter_min_cells is not supported for 4-D tracking",
            details="The pre-filter counts cells in a horizontal slice and would act per level, not per volume",
            suggestions=["Remove prefilter_min_cells; use volume_filter_quartile or volume_filter_absolute"],
        )
    if not _is_real_number(max_vertical_gap_m) or not math.isfinite(float(max_vertical_gap_m)) or max_vertical_gap_m < 0:
        raise ConfigurationError(
            f"max_vertical_gap_m must be a finite number of metres >= 0, got {max_vertical_gap_m!r}",
            details="0 connects only consecutive stored levels",
            suggestions=["Pass max_vertical_gap_m=0.0 or a positive gap in metres"],
        )
    if partition_scales is not None:
        ok = isinstance(partition_scales, (Sequence, np.ndarray)) and not isinstance(partition_scales, str)
        ok = ok and np.ndim(partition_scales) == 1 and len(partition_scales) == 2
        ok = ok and all(_is_real_number(s) and math.isfinite(float(s)) and s > 0 for s in partition_scales)
        if not ok:
            raise ConfigurationError(
                f"partition_scales must be two positive lengths (L_H, L_Z) in metres, got {partition_scales!r}",
                suggestions=["Pass partition_scales=(L_H, L_Z), e.g. (20_000.0, 50.0), or None for grid-derived scales"],
            )
    if volume_filter_quartile is not None and volume_filter_absolute is not None:
        raise ConfigurationError(
            "Cannot specify both volume filtering parameters",
            details="volume_filter_quartile and volume_filter_absolute are mutually exclusive",
            suggestions=["Pass only one of volume_filter_quartile and volume_filter_absolute"],
            context={
                "volume_filter_quartile": volume_filter_quartile,
                "volume_filter_absolute": volume_filter_absolute,
            },
        )
    if volume_filter_quartile is not None and (
        not _is_real_number(volume_filter_quartile) or not 0.0 <= float(volume_filter_quartile) <= 1.0
    ):
        raise ConfigurationError(
            f"volume_filter_quartile must be a number in [0, 1], got {volume_filter_quartile!r}",
            suggestions=["Use e.g. volume_filter_quartile=0.5 to drop the smaller half of the events"],
        )
    if volume_filter_absolute is not None and (
        not _is_real_number(volume_filter_absolute)
        or not math.isfinite(float(volume_filter_absolute))
        or volume_filter_absolute <= 0
    ):
        raise ConfigurationError(
            f"volume_filter_absolute must be a positive volume in m^3, got {volume_filter_absolute!r}",
            suggestions=["Pass the minimum event volume in cubic metres, e.g. volume_filter_absolute=1e12"],
        )
    if volume_filter_quartile is None and volume_filter_absolute is None:
        return 0.5, None
    if volume_filter_quartile is not None:
        return float(volume_filter_quartile), None
    return None, float(volume_filter_absolute)


def validate_volumetric_rank(data_bin: xr.DataArray, dims: VolumetricDims) -> None:
    """
    Check that ``data_bin`` carries exactly the four declared dimensions.

    Raises
    ------
    ConfigurationError
        If a declared dimension is absent from the data.
    TrackingError
        If the data has a dimension beyond time, depth, lat and lon.
    """
    expected = list(dims.internal_order())
    missing = [d for d in expected if d not in data_bin.dims]
    if missing:
        raise ConfigurationError(
            f"The dimension mapping names {missing} but the data has dimensions {[str(d) for d in data_bin.dims]}",
            suggestions=["Check dimensions={'time': ..., 'z': ..., 'y': ..., 'x': ...} against the data"],
            context={"missing": missing, "actual_dims": [str(d) for d in data_bin.dims]},
        )
    unexpected = [str(d) for d in data_bin.dims if d not in expected]
    if unexpected:
        raise TrackingError(
            f"4-D tracking does not support the extra dimension(s) {unexpected}",
            details=f"Expected dimensions {expected}, got {[str(d) for d in data_bin.dims]}.",
            suggestions=[
                f"Select one member first, e.g. ds.isel({unexpected[0]}=0)",
                "Loop over the extra dimension and track each member separately",
            ],
            context={
                "unexpected_dimensions": unexpected,
                "actual_dims": [str(d) for d in data_bin.dims],
                "expected_dims": expected,
            },
        )


def _check_z_coordinate(data_bin: xr.DataArray, zdim: str) -> Tuple[NDArray, dict]:
    """Validate the depth coordinate and return ``(values, attrs)``."""
    if zdim not in data_bin.coords:
        raise ConfigurationError(
            f"The depth dimension '{zdim}' has no coordinate",
            details="4-D tracking needs a 1-D numeric depth coordinate in metres",
            suggestions=[f"Assign the level depths, e.g. data.assign_coords({zdim}=depths_in_metres)"],
        )
    z = data_bin[zdim]
    if z.ndim != 1:
        raise ConfigurationError(f"The depth coordinate '{zdim}' must be one-dimensional, got dims {z.dims}")
    if not (np.issubdtype(z.dtype, np.floating) or np.issubdtype(z.dtype, np.integer)):
        raise ConfigurationError(
            f"The depth coordinate '{zdim}' must be numeric, got dtype {z.dtype}",
            suggestions=["Convert the depth coordinate to floating-point metres"],
        )
    attrs = dict(z.attrs)
    values = np.asarray(z.values)
    # Units, 'positive' and strict monotonicity are checked by the same routine the vertical
    # geometry uses, so the data order and the geometry order cannot disagree.
    _z_order(values, attrs)
    return values, attrs


def _coordinate_rtol(a: NDArray, b: NDArray) -> float:
    """
    Relative tolerance for comparing two coordinate arrays, set by the lower-precision one.

    A float32 coordinate (e.g. GLORYS depth) and its float64 original differ by float32 rounding,
    so a few float32 epsilon is the bar whenever either side is float32; two float64 coordinates
    keep the tight 1e-9 bar.
    """
    eps = max(np.finfo(d).eps if np.issubdtype(d, np.floating) else np.finfo(np.float64).eps for d in (a.dtype, b.dtype))
    return 1.0e-9 if eps <= np.finfo(np.float64).eps else 4.0 * eps


def _align_mask(mask: xr.DataArray, data_bin: xr.DataArray, dims: VolumetricDims) -> xr.DataArray:
    """
    Put the mask on the data's coordinate labels along ``z``, ``y`` and ``x``.

    Where both carry a coordinate on a dimension, the mask is reordered to the data's labels
    (it may list them in a different order, e.g. an ascending depth mask for descending data).
    Coordinates equal up to rounding in the same order are taken as aligned. Coordinates that
    do not hold the same set of labels raise.
    """
    for d in (dims.z, dims.y, dims.x):
        if d not in mask.coords or d not in data_bin.coords:
            continue
        want = np.asarray(data_bin[d].values)
        have = np.asarray(mask[d].values)
        if (
            want.shape == have.shape
            and np.issubdtype(want.dtype, np.number)
            and np.allclose(want, have, rtol=_coordinate_rtol(want, have), atol=0.0)
        ):
            continue
        if want.shape != have.shape or not np.all(np.isin(want, have)):
            raise ConfigurationError(
                f"The mask's '{d}' coordinate does not match the data's",
                details="The mask and the extremes field must be defined on the same grid",
                suggestions=[f"Build the mask on the same '{d}' labels as the extremes field, or drop the mask's '{d}' coordinate"],
                context={"dim": d},
            )
        mask = mask.reindex({d: want})
    return mask


def prepare_volumetric_data(
    data_bin: xr.DataArray,
    mask: Optional[xr.DataArray],
    dims: VolumetricDims,
) -> Tuple[xr.DataArray, xr.DataArray, NDArray, dict, Tuple[str, ...]]:
    """
    Bring a 4-D field and its wet mask into internal order.

    Parameters
    ----------
    data_bin : xarray.DataArray
        Boolean extremes field, dask-backed, containing exactly the declared dimensions.
    mask : xarray.DataArray
        Boolean wet mask with dimensions ``(z, y, x)``. Required.
    dims : VolumetricDims
        The dimension names.

    Returns
    -------
    data_bin : xarray.DataArray
        Field in order ``(time, z, y, x)`` with ascending positive-down depth, and
        ``z``, ``y`` and ``x`` each in a single chunk.
    mask : xarray.DataArray
        Boolean wet mask ``(z, y, x)`` in the same depth order.
    z_values : numpy.ndarray
        The original depth coordinate values.
    z_attrs : dict
        The original depth coordinate attributes.
    original_dims : tuple of str
        The original dimension order of ``data_bin``.

    Notes
    -----
    A UserWarning is emitted once when ``z``, ``y`` or ``x`` had to be rechunked.
    """
    validate_volumetric_rank(data_bin, dims)
    if mask is None:
        raise ConfigurationError(
            "4-D tracking requires a wet mask",
            details="A surface mask cannot describe bathymetry; dry cells are barriers for vertical links",
            suggestions=[f"Pass mask as a boolean ({dims.z}, {dims.y}, {dims.x}) DataArray, True where the cell is ocean"],
        )
    if set(mask.dims) != {dims.z, dims.y, dims.x} or mask.ndim != 3:
        raise ConfigurationError(
            f"The wet mask must have dimensions ({dims.z}, {dims.y}, {dims.x}), got {tuple(mask.dims)}",
            details="4-D tracking needs a (depth, lat, lon) wet mask",
            suggestions=[f"Pass a boolean ({dims.z}, {dims.y}, {dims.x}) mask, e.g. from the (depth, lat, lon) land-sea mask"],
            context={"mask_dims": tuple(str(d) for d in mask.dims)},
        )
    if mask.dtype != bool:
        raise create_data_validation_error(
            "The wet mask must be boolean",
            details=f"Found dtype {mask.dtype}, expected bool",
            suggestions=["Convert the mask with mask.astype(bool)"],
        )
    if data_bin.dtype != bool:
        raise create_data_validation_error(
            "Input DataArray must be binary (boolean type)",
            details=f"Found dtype {data_bin.dtype}, expected bool",
            suggestions=["Convert data using da > threshold for binary events"],
        )
    if data_bin.chunks is None:
        raise create_data_validation_error(
            "Data must be chunked",
            details="The input data_bin must have chunk information",
            suggestions=["Use data_bin.chunk({'time': 10}) to chunk the data"],
        )
    for d in (dims.z, dims.y, dims.x):
        if data_bin.sizes[d] != mask.sizes[d]:
            raise ConfigurationError(
                f"The mask has {mask.sizes[d]} cells along '{d}' but the data has {data_bin.sizes[d]}",
                suggestions=["Build the mask on the same grid as the extremes field"],
                context={"dim": d, "mask": mask.sizes[d], "data": data_bin.sizes[d]},
            )

    z_values, z_attrs = _check_z_coordinate(data_bin, dims.z)
    _, flip, _ = _z_order(z_values, z_attrs)
    original_dims = tuple(str(d) for d in data_bin.dims)

    mask = _align_mask(mask, data_bin, dims)
    data = data_bin.transpose(*dims.internal_order())
    wet = mask.transpose(dims.z, dims.y, dims.x)
    if flip:
        data = data.isel({dims.z: slice(None, None, -1)})
        wet = wet.isel({dims.z: slice(None, None, -1)})

    # z, y and x must each be one chunk: labelling is global in space, including depth.
    spatial = (dims.z, dims.y, dims.x)
    split = [d for d in spatial if len(data.chunksizes[d]) > 1]
    if split:
        warnings.warn(
            f"Rechunking {split} of the extremes field to a single chunk each: 4-D labelling is "
            "global in depth and the horizontal. Chunk only the time dimension to avoid this copy.",
            UserWarning,
            stacklevel=3,
        )
        data = data.chunk({d: -1 for d in spatial})
    if wet.chunks is not None and any(len(c) > 1 for c in wet.chunks):
        wet = wet.chunk({d: -1 for d in spatial})
    return data, wet, z_values, z_attrs, original_dims
