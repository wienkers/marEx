"""
MarEx Track: per-event diagnostics of a finished 4-D event field.

Every diagnostic is computed from the final event labels, one timestep at a time, with the
static geometry passed as single-chunk ``(z, y, x)`` arrays (never broadcast over time).
Each per-event sum is a float64 ``np.bincount`` over the occupied voxels in C ``(z, y, x)``
order, which accumulates strictly sequentially, so the result does not depend on the time
chunking or on the compute mode.
"""

from typing import Tuple

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ...exceptions import ConfigurationError, TrackingError
from ...logging_config import get_logger
from .config import VolumetricDims
from .geometry import VerticalGeometry

logger = get_logger(__name__)

# Rounding allowance (degrees, ~0.1 mm at the equator) for the longitude wrap: a centroid this
# close west of the westmost longitude is that longitude, not one a full turn east of it. The
# unit-vector round trip errs by ~1e-13 degrees, so this is far above rounding and far below a cell.
LON_WRAP_TOLERANCE_DEG: float = 1e-9

_PER_EVENT = ("volume", "footprint_area", "centroid_lat", "centroid_lon", "depth_centroid", "depth_top", "depth_bottom")


def _horizontal_2d(lat_deg: NDArray, lon_deg: NDArray, ny: int, nx: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Latitude and longitude as contiguous float64 ``(y, x)`` arrays."""
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    if lat.ndim == 1 and lon.ndim == 1:
        if lat.size != ny or lon.size != nx:
            raise ConfigurationError(f"lat_deg/lon_deg have sizes ({lat.size}, {lon.size}), expected ({ny}, {nx})")
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
    elif lat.shape != (ny, nx) or lon.shape != (ny, nx):
        raise ConfigurationError(f"lat_deg/lon_deg have shapes {lat.shape}, {lon.shape}, expected ({ny},)/({nx},) or ({ny}, {nx})")
    return np.ascontiguousarray(lat), np.ascontiguousarray(lon)


def _event_slice(
    ids: NDArray[np.int32],
    present: NDArray[np.bool_],
    event_ids: NDArray[np.int32],
    cell_volume: NDArray[np.float64],
    midpoint: NDArray[np.float64],
    top: NDArray[np.float64],
    bottom: NDArray[np.float64],
    cell_area: NDArray[np.float64],
    ex: NDArray[np.float64],
    ey: NDArray[np.float64],
    ez: NDArray[np.float64],
    lon_origin: float,
) -> Tuple[NDArray[np.float64], ...]:
    """
    Diagnostics of every event in one ``(z, y, x)`` slice.

    Returns the seven ``(ID,)`` float64 arrays of ``_PER_EVENT``; NaN where an event is absent.
    """
    n_ids = event_ids.size
    outputs = [np.full(n_ids, np.nan, dtype=np.float64) for _ in _PER_EVENT]
    flat_ids = ids.ravel()
    occupied = np.flatnonzero(flat_ids > 0)  # ascending: C (z, y, x) order
    if occupied.size == 0:
        return tuple(outputs)

    labels = flat_ids[occupied]
    codes = np.searchsorted(event_ids, labels)
    if np.any(codes >= n_ids) or np.any(event_ids[np.minimum(codes, n_ids - 1)] != labels):
        raise TrackingError(
            "The event field holds an ID that is not on the ID axis",
            details=f"{np.setdiff1d(labels, event_ids).size} unknown IDs, e.g. {np.setdiff1d(labels, event_ids)[:5].tolist()}",
        )

    n_columns = cell_area.size
    columns = occupied % n_columns
    v = cell_volume.ravel()[occupied]

    # Sequential float64 sums in C order (np.bincount accumulates in input order).
    volume = np.bincount(codes, weights=v, minlength=n_ids)
    weighted_mid = np.bincount(codes, weights=v * midpoint.ravel()[occupied], minlength=n_ids)
    sx = np.bincount(codes, weights=v * ex.ravel()[columns], minlength=n_ids)
    sy = np.bincount(codes, weights=v * ey.ravel()[columns], minlength=n_ids)
    sz = np.bincount(codes, weights=v * ez.ravel()[columns], minlength=n_ids)

    # Footprint: each occupied column once per event, columns in C (y, x) order.
    keys = np.unique(codes.astype(np.int64) * n_columns + columns)
    footprint = np.bincount(keys // n_columns, weights=cell_area.ravel()[keys % n_columns], minlength=n_ids)

    shallowest = np.full(n_ids, np.inf)
    deepest = np.full(n_ids, -np.inf)
    np.minimum.at(shallowest, codes, top.ravel()[occupied])
    np.maximum.at(deepest, codes, bottom.ravel()[occupied])

    # Volume-weighted mean of unit vectors on the sphere: continuous across the lon seam.
    lat_c = np.degrees(np.arctan2(sz, np.hypot(sx, sy)))
    lon_c = np.degrees(np.arctan2(sy, sx))
    # Into the coordinate's convention, [lon_origin, lon_origin + 360). An event in the origin
    # column comes back from the unit-vector round trip a few ulps west of lon_origin, which
    # the plain mod would send to lon_origin + 360; offsets within LON_WRAP_TOLERANCE_DEG below
    # 360 are taken as that rounding and folded back to just below lon_origin.
    offset = np.mod(lon_c - lon_origin, 360.0)
    offset = np.where(offset > 360.0 - LON_WRAP_TOLERANCE_DEG, offset - 360.0, offset)
    lon_c = lon_origin + offset

    with np.errstate(invalid="ignore", divide="ignore"):
        depth_centroid = weighted_mid / volume

    has = present & (volume > 0)
    for out, values in zip(outputs, (volume, footprint, lat_c, lon_c, depth_centroid, shallowest, deepest)):
        out[has] = values[has]
    return tuple(outputs)


def event_diagnostics(
    id_field: xr.DataArray,
    presence: xr.DataArray,
    geom: VerticalGeometry,
    lat_deg: NDArray,
    lon_deg: NDArray,
    dims: VolumetricDims,
    *,
    materialiser,
) -> xr.Dataset:
    """
    Volume, footprint, centroid and depth diagnostics of every event at every timestep.

    Parameters
    ----------
    id_field : xarray.DataArray
        int32 ``(time, z, y, x)`` event IDs (0 = background), dask-backed, chunked in time only.
    presence : xarray.DataArray
        bool ``(time, ID)``; its ``ID`` coordinate (sorted) is the output ID axis.
    geom : VerticalGeometry
        Vertical geometry on the same grid (ascending positive-down depth).
    lat_deg, lon_deg : numpy.ndarray
        Latitude and longitude in degrees: ``(y,)`` and ``(x,)``, or both ``(y, x)``.
    dims : VolumetricDims
        Dimension names.
    materialiser : Materialiser or None
        Anchors the result: staged to disk in streaming mode, persisted otherwise.

    Returns
    -------
    xarray.Dataset
        ``volume`` (m^3), ``footprint_area`` (m^2, area of the ``any(z)`` projection),
        ``depth_centroid``, ``depth_top``, ``depth_bottom`` and ``mean_thickness``
        (``volume / footprint_area``), all float64 ``(time, ID)`` in positive-down metres,
        and ``centroid`` ``(component=[0, 1], time, ID)``: the volume-weighted horizontal
        centroid (latitude, longitude) in degrees, with longitude in
        ``[min(lon), min(lon) + 360)``, up to ``LON_WRAP_TOLERANCE_DEG`` of rounding west of
        ``min(lon)``. NaN where an event is absent.
    """
    timedim, zdim, ydim, xdim = dims.internal_order()
    if tuple(id_field.dims) != dims.internal_order():
        raise ConfigurationError(f"The event field must have dimensions {dims.internal_order()}, got {tuple(id_field.dims)}")
    if tuple(presence.dims) != (timedim, "ID"):
        raise ConfigurationError(f"presence must have dimensions ({timedim!r}, 'ID'), got {tuple(presence.dims)}")
    nz, ny, nx = (id_field.sizes[d] for d in (zdim, ydim, xdim))
    if geom.cell_volume.shape != (nz, ny, nx):
        raise ConfigurationError(f"The geometry has shape {geom.cell_volume.shape}, the event field {(nz, ny, nx)}")
    event_ids = np.asarray(presence["ID"].values)
    if event_ids.size > 1 and np.any(np.diff(event_ids) <= 0):
        raise ConfigurationError("presence's ID coordinate must be strictly increasing")

    lat2, lon2 = _horizontal_2d(lat_deg, lon_deg, ny, nx)
    lat_r, lon_r = np.radians(lat2), np.radians(lon2)
    ex = np.cos(lat_r) * np.cos(lon_r)
    ey = np.cos(lat_r) * np.sin(lon_r)
    ez = np.sin(lat_r)

    # Static geometry as single-chunk dask arrays: apply_ufunc loops over time lazily, so
    # nothing is copied or broadcast to (time, z, y, x).
    def static(values: NDArray, core: Tuple[str, ...]) -> xr.DataArray:
        return xr.DataArray(np.ascontiguousarray(values, dtype=np.float64), dims=core).chunk({d: -1 for d in core})

    zyx = (zdim, ydim, xdim)
    yx = (ydim, xdim)
    ids_axis = xr.DataArray(event_ids.astype(np.int32), dims="ID").chunk({"ID": -1})
    presence = presence.chunk({timedim: id_field.chunks[0], "ID": -1})
    field = id_field.drop_vars([c for c in id_field.coords if c not in id_field.dims or c in zyx])

    results = xr.apply_ufunc(
        _event_slice,
        field,
        presence.drop_vars("ID"),
        ids_axis,
        static(geom.cell_volume, zyx),
        static(geom.midpoint, zyx),
        static(geom.top, zyx),
        static(geom.bottom, zyx),
        static(geom.cell_area_m2, yx),
        static(ex, yx),
        static(ey, yx),
        static(ez, yx),
        kwargs={"lon_origin": float(np.min(lon2))},
        input_core_dims=[
            list(zyx),
            ["ID"],
            ["ID"],
            list(zyx),
            list(zyx),
            list(zyx),
            list(zyx),
            list(yx),
            list(yx),
            list(yx),
            list(yx),
        ],
        output_core_dims=[["ID"]] * len(_PER_EVENT),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64] * len(_PER_EVENT),
    )
    named = {name: r.assign_coords(ID=event_ids).transpose(timedim, "ID") for name, r in zip(_PER_EVENT, results)}
    if materialiser is None:
        staged = xr.Dataset(named).persist()
        named = {name: staged[name] for name in _PER_EVENT}
    else:
        named = dict(zip(_PER_EVENT, materialiser.stage_many(named, "volumetric_diagnostics", preserve_chunks=True)))

    centroid = xr.concat([named["centroid_lat"], named["centroid_lon"]], dim="component").assign_coords(component=[0, 1])
    centroid = centroid.chunk({"component": -1})
    out = xr.Dataset(
        {
            "volume": named["volume"],
            "footprint_area": named["footprint_area"],
            "centroid": centroid.transpose("component", timedim, "ID"),
            "depth_centroid": named["depth_centroid"],
            "depth_top": named["depth_top"],
            "depth_bottom": named["depth_bottom"],
            "mean_thickness": named["volume"] / named["footprint_area"],
        }
    )
    out["volume"].attrs.update(units="m^3", long_name="Event volume")
    out["footprint_area"].attrs.update(units="m^2", long_name="Area of the event's horizontal projection")
    out["centroid"].attrs.update(units="degrees", long_name="Volume-weighted horizontal centroid (lat, lon)")
    for name, long_name in (
        ("depth_centroid", "Volume-weighted mean cell-midpoint depth"),
        ("depth_top", "Shallowest occupied top interface"),
        ("depth_bottom", "Deepest occupied bottom interface"),
        ("mean_thickness", "Volume divided by footprint area"),
    ):
        out[name].attrs.update(units="m", positive="down", long_name=long_name)
    return out
