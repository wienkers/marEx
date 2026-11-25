"""
MarEx Track: vertical geometry for 4-D tracking.

Builds, once, the per-cell vertical geometry that every 4-D tracking step
consumes: layer top and bottom interfaces, partial-cell-aware thickness,
midpoints, horizontal cell areas, cell volumes and grid-derived length scales.
All depths are metres, positive down, in ascending-depth (internal) order; areas
are float64 m^2 and volumes float64 m^3.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ...exceptions import ConfigurationError, DataValidationError
from ...logging_config import get_logger

logger = get_logger(__name__)

# Earth radius (m) for horizontal areas and chord coordinates alike.
EARTH_RADIUS_M: float = 6_378_000.0

_METRE_UNITS = {"m", "metre", "metres", "meter", "meters"}
_AREA_FACTORS = {"m2": 1.0, "m^2": 1.0, "km2": 1.0e6, "km^2": 1.0e6}

# A partial-cell thickness may exceed its layer's bounds thickness by float rounding only.
_THICKNESS_RTOL = 1.0e-6


@dataclass(frozen=True)
class VerticalGeometry:
    """
    Vertical geometry of a structured ``(z, y, x)`` grid.

    Depth arrays are positive-down metres in ascending-depth order. Layer quantities
    are ``(z,)`` for depth-only interfaces or ``(z, y, x)`` for per-cell geometry,
    and are only meaningful where ``wet`` is True. Dense layer quantities and cell
    volumes are zero on dry cells.

    Attributes
    ----------
    top, bottom : numpy.ndarray
        Layer interfaces. A partial cell keeps its standard
        top interface and is shortened from below.
    thickness : numpy.ndarray
        Partial-cell-aware layer thickness.
    midpoint : numpy.ndarray
        ``(top + bottom) / 2``.
    cell_area_m2 : numpy.ndarray
        Horizontal cell area (m^2), shape ``(y, x)``.
    cell_volume : numpy.ndarray
        Cell volume (m^3), shape ``(z, y, x)``.
    wet : numpy.ndarray
        Boolean wet mask, shape ``(z, y, x)``.
    source : str
        ``"bounds"``, ``"bounds+thickness"`` or ``"reconstructed_from_centres"``.
    flip_z : bool
        True when the input z order is descending in positive-down depth.
    default_scales : tuple of float
        ``(L_H, L_Z)`` in metres: the median of ``sqrt(cell_area_m2)`` and of the
        thickness, each taken over wet cells (every wet ``(z, y, x)`` cell counts once).
    """

    top: NDArray[np.float64]
    bottom: NDArray[np.float64]
    thickness: NDArray[np.float64]
    midpoint: NDArray[np.float64]
    cell_area_m2: NDArray[np.float64]
    cell_volume: NDArray[np.float64]
    wet: NDArray[np.bool_]
    source: str
    flip_z: bool
    default_scales: Tuple[float, float]


def _fail(message: str, details: Optional[str] = None, suggestions=None, context: Optional[Dict] = None):
    """Build a ConfigurationError for a geometry problem."""
    return ConfigurationError(message, details=details, suggestions=suggestions, context=context)


def _z_order(z_values: NDArray, z_attrs: dict) -> Tuple[NDArray[np.float64], bool, bool]:
    """Return positive-down centres in ascending order, ``flip_z`` and the up-sign flag."""
    units = str(z_attrs.get("units", "")).strip().lower()
    if units not in _METRE_UNITS:
        raise _fail(
            f"The depth coordinate must have units in metres, got units={z_attrs.get('units')!r}",
            suggestions=[
                "Set the depth coordinate's 'units' attribute to 'm'",
                "Convert the depth coordinate to metres before tracking",
            ],
            context={"units": z_attrs.get("units")},
        )
    positive = str(z_attrs.get("positive", "down")).strip().lower()
    if positive not in ("up", "down"):
        raise _fail(
            f"Invalid 'positive' attribute on the depth coordinate: {z_attrs.get('positive')!r}",
            suggestions=["Use positive='down' or positive='up'"],
        )
    z = np.asarray(z_values, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise _fail("The depth coordinate must be one-dimensional and non-empty", context={"shape": z.shape})
    if not np.isfinite(z).all():
        raise DataValidationError("The depth coordinate must be finite")
    if positive == "up":
        z = -z
    step = np.diff(z)
    if step.size and np.all(step > 0):
        flip = False
    elif step.size and np.all(step < 0):
        flip = True
    elif z.size == 1:
        flip = False
    else:
        raise _fail(
            "The depth coordinate must be strictly monotonic",
            suggestions=["Sort the depth coordinate and drop duplicate levels"],
        )
    return (z[::-1].copy() if flip else z), flip, positive == "up"


def _to_zyx(da: xr.DataArray, name: str, zdim: str, ydim: str, xdim: str, nz: int, shape: Tuple[int, int, int]) -> NDArray:
    """Return a ``(z, y, x)`` float64 array from a named-dim DataArray."""
    if set(da.dims) != {zdim, ydim, xdim}:
        raise _fail(
            f"{name} must have dimensions ({zdim}, {ydim}, {xdim}), got {tuple(da.dims)}",
            context={"dims": tuple(da.dims)},
        )
    arr = np.asarray(da.transpose(zdim, ydim, xdim).values, dtype=np.float64)
    if arr.shape != shape:
        raise _fail(f"{name} has shape {arr.shape}, expected {shape}", context={"shape": arr.shape})
    return arr


def _interfaces_from_bounds(
    layer_bounds: xr.DataArray,
    zdim: str,
    ydim: str,
    xdim: str,
    negate: bool,
    flip: bool,
    shape: Tuple[int, int, int],
) -> Tuple[NDArray, NDArray]:
    """Return positive-down float64 interfaces, retaining the bounds' spatial rank."""
    dims = set(layer_bounds.dims)
    extra = dims - {zdim, ydim, xdim}
    if zdim not in dims or len(extra) != 1:
        raise _fail(
            f"layer_bounds must have the depth dimension '{zdim}' plus exactly one bounds dimension "
            f"of size 2, got dims {tuple(layer_bounds.dims)}",
            suggestions=[f"Pass layer_bounds with dims ({zdim}, <bounds>) or ({zdim}, {ydim}, {xdim}, <bounds>)"],
            context={"dims": tuple(layer_bounds.dims)},
        )
    (bdim,) = extra
    if layer_bounds.sizes[bdim] != 2:
        raise _fail(
            f"layer_bounds dimension '{bdim}' must have size 2 (top and bottom), got {layer_bounds.sizes[bdim]}",
            context={"dims": dict(layer_bounds.sizes)},
        )
    spatial = dims - {zdim, bdim}
    if spatial not in (set(), {ydim, xdim}):
        raise _fail(
            f"layer_bounds must be ({zdim}, {bdim}) or ({zdim}, {ydim}, {xdim}, {bdim}), got dims {tuple(layer_bounds.dims)}",
            context={"dims": tuple(layer_bounds.dims)},
        )
    nz, ny, nx = shape
    if layer_bounds.sizes[zdim] != nz:
        raise _fail(
            f"layer_bounds has {layer_bounds.sizes[zdim]} levels along '{zdim}', the depth coordinate has {nz}",
            context={"dims": dict(layer_bounds.sizes)},
        )
    if spatial:
        arr = np.asarray(layer_bounds.transpose(zdim, ydim, xdim, bdim).values, dtype=np.float64)
        if arr.shape != (nz, ny, nx, 2):
            raise _fail(f"layer_bounds has shape {arr.shape}, expected {(nz, ny, nx, 2)}")
    else:
        arr = np.asarray(layer_bounds.transpose(zdim, bdim).values, dtype=np.float64)
    if negate:
        arr = -arr
    if flip:
        arr = arr[::-1]
    return np.ascontiguousarray(arr[..., 0]), np.ascontiguousarray(arr[..., 1])


def _interfaces_from_centres(centres: NDArray[np.float64]) -> Tuple[NDArray, NDArray]:
    """Reconstruct ``(top, bottom)`` from level centres by the midpoint rule."""
    if centres.size < 2:
        raise _fail(
            'layer_bounds="from_centres" needs at least two depth levels',
            suggestions=["Pass explicit layer_bounds"],
        )
    mid = 0.5 * (centres[:-1] + centres[1:])
    first = max(2.0 * centres[0] - mid[0], 0.0)
    last = 2.0 * centres[-1] - mid[-1]
    interfaces = np.concatenate([[first], mid, [last]])
    return interfaces[:-1].copy(), interfaces[1:].copy()


def _area_units_factor(cell_areas: xr.DataArray) -> float:
    units = str(cell_areas.attrs.get("units", "")).strip().lower().replace(" ", "")
    if units not in _AREA_FACTORS:
        raise _fail(
            f"cell_areas must declare units 'm2', 'm^2' or 'km2', got units={cell_areas.attrs.get('units')!r}",
            suggestions=["Set the 'units' attribute on cell_areas to 'm2' or 'km2'"],
            context={"units": cell_areas.attrs.get("units")},
        )
    return _AREA_FACTORS[units]


def _has_duplicate_longitude_endpoint(lon_deg: NDArray) -> bool:
    """Whether a rectilinear longitude axis spans a full turn and repeats its first meridian at the end."""
    lon = np.asarray(lon_deg, dtype=np.float64)
    if lon.ndim != 1 or lon.size < 2 or not np.all(np.isfinite(lon)):
        return False
    gaps = np.abs(np.diff(lon))
    step = float(np.median(np.minimum(gaps, 360.0 - np.mod(gaps, 360.0))))
    if step <= 0 or not np.isclose((lon.size - 1) * step, 360.0):
        return False
    turns = abs(float(lon[-1]) - float(lon[0])) / 360.0
    return bool(np.isclose(turns, np.rint(turns), rtol=0.0, atol=np.finfo(np.float32).eps))


def _cell_area_m2(
    cell_areas: Optional[xr.DataArray],
    grid_resolution: Optional[float],
    lat_deg: NDArray,
    ydim: str,
    xdim: str,
    ny: int,
    nx: int,
) -> NDArray[np.float64]:
    """Return the horizontal cell area as ``(y, x)`` float64 m^2."""
    if grid_resolution is not None:
        if cell_areas is not None:
            logger.warning("grid_resolution parameter overrides provided cell_areas for structured grid")
        lat = np.asarray(lat_deg, dtype=np.float64)
        if lat.ndim == 1:
            lat = lat[:, None]
        logger.info(f"Calculating cell areas from grid resolution: {grid_resolution} degrees")
        lat_r = np.radians(lat)
        dlat = np.radians(float(grid_resolution))
        dlon = np.radians(float(grid_resolution))
        north = np.clip(lat_r + dlat / 2, -np.pi / 2, np.pi / 2)
        south = np.clip(lat_r - dlat / 2, -np.pi / 2, np.pi / 2)
        area = EARTH_RADIUS_M**2 * np.abs(np.sin(north) - np.sin(south)) * dlon
        return np.ascontiguousarray(np.broadcast_to(area, (ny, nx)), dtype=np.float64)
    if cell_areas is None:
        raise _fail(
            "4-D tracking needs a horizontal cell area: pass cell_areas or grid_resolution",
            details="Without a horizontal area, volumes cannot be computed in physical units",
            suggestions=[
                "Pass cell_areas as a (lat, lon) DataArray with units 'm2' or 'km2'",
                "Or pass grid_resolution (degrees) for a regular lat/lon grid",
            ],
        )
    factor = _area_units_factor(cell_areas)
    if set(cell_areas.dims) != {ydim, xdim}:
        raise _fail(f"cell_areas must have dimensions ({ydim}, {xdim}), got {tuple(cell_areas.dims)}")
    area = np.asarray(cell_areas.transpose(ydim, xdim).values, dtype=np.float64) * factor
    if area.shape != (ny, nx):
        raise _fail(f"cell_areas has shape {area.shape}, expected {(ny, nx)}")
    return np.ascontiguousarray(area)


def _median_over_wet_cells(values: NDArray[np.float64], counts: NDArray[np.int64]) -> float:
    """Median of ``values`` where each entry is repeated ``counts`` times (matches ``numpy.median``)."""
    keep = counts > 0
    v = values[keep]
    c = counts[keep]
    order = np.argsort(v, kind="stable")
    v = v[order]
    cum = np.cumsum(c[order])
    n = int(cum[-1])
    lo = v[np.searchsorted(cum, (n - 1) // 2, side="right")]
    if n % 2:
        return float(lo)
    hi = v[np.searchsorted(cum, n // 2, side="right")]
    return float(0.5 * (lo + hi))


def build_vertical_geometry(
    z_values: NDArray,
    z_attrs: dict,
    wet_mask: xr.DataArray,
    *,
    zdim: str,
    ydim: str,
    xdim: str,
    lat_deg: NDArray[np.float64],
    lon_deg: NDArray[np.float64],
    layer_bounds: Union[xr.DataArray, Literal["from_centres"], None],
    layer_thickness: Optional[xr.DataArray],
    cell_volumes: Optional[xr.DataArray],
    cell_areas: Optional[xr.DataArray],
    grid_resolution: Optional[float],
    regional_mode: bool = False,
) -> VerticalGeometry:
    """
    Build the vertical geometry of a structured ``(z, y, x)`` grid.

    Parameters
    ----------
    z_values : numpy.ndarray
        Original depth coordinate (metres, possibly descending, possibly positive up).
    z_attrs : dict
        Attributes of the depth coordinate. ``units`` must be metres; ``positive``
        (``"up"`` or ``"down"``) defaults to down.
    wet_mask : xarray.DataArray
        Boolean wet mask already in internal (ascending positive-down depth) order.
    zdim, ydim, xdim : str
        Dimension names.
    lat_deg, lon_deg : numpy.ndarray
        Latitude and longitude in degrees. If the last longitude of a one-dimensional
        axis repeats its first meridian after a full turn, the geometry omits the last column.
    layer_bounds : xarray.DataArray, ``"from_centres"`` or None
        Top and bottom interfaces in metres, ``(z, 2)`` or ``(z, y, x, 2)``, in the
        original depth order and sign. The bounds dimension may have any name. ``"from_centres"``
        opts in to reconstructing interfaces at midpoints between level centres.
    layer_thickness : xarray.DataArray, optional
        ``(z, y, x)`` thickness (m) for partial cells, in the original depth order.
        A partial cell keeps its top interface and is shortened from below.
    cell_volumes : xarray.DataArray, optional
        ``(z, y, x)`` volumes (m^3), in the original depth order, overriding the
        area-times-thickness product.
    cell_areas : xarray.DataArray, optional
        ``(y, x)`` horizontal areas with units ``m2``/``m^2`` or ``km2``.
    grid_resolution : float, optional
        Regular grid spacing in degrees; areas follow from spherical geometry. When
        given it overrides ``cell_areas``, as in 3-D tracking.

    Returns
    -------
    VerticalGeometry
        Geometry in internal (ascending positive-down depth) order.

    Raises
    ------
    ConfigurationError
        If the depth units are not metres, no vertical geometry or horizontal area is
        supplied, or any wet cell has non-finite or non-positive thickness or volume.
    DataValidationError
        If wet bounds are inverted or overlap, derived wet volumes are invalid, wet
        coordinates are non-finite, or the wet mask differs at a duplicate seam.
    """
    centres, flip, negate = _z_order(z_values, z_attrs)
    wet_da = wet_mask
    if set(wet_da.dims) != {zdim, ydim, xdim}:
        raise _fail(f"wet_mask must have dimensions ({zdim}, {ydim}, {xdim}), got {tuple(wet_da.dims)}")
    wet = np.array(wet_da.transpose(zdim, ydim, xdim).values, dtype=bool)  # copy: frozen below
    lat_deg, lon_deg = np.asarray(lat_deg), np.asarray(lon_deg)
    if not regional_mode and _has_duplicate_longitude_endpoint(lon_deg):
        if lon_deg.size != wet.shape[2]:
            raise _fail(f"lon_deg has {lon_deg.size} values, expected {wet.shape[2]}")
        if not np.array_equal(wet[..., 0], wet[..., -1]):
            raise DataValidationError("The wet mask differs on the duplicate longitude endpoint")
        wet_seam_2d = wet.any(axis=0)[:, 0]
        for label, da_seam in (
            ("layer_bounds", layer_bounds),
            ("layer_thickness", layer_thickness),
            ("cell_volumes", cell_volumes),
            ("cell_areas", None if grid_resolution is not None else cell_areas),
        ):
            if isinstance(da_seam, xr.DataArray) and xdim in da_seam.dims and ydim in da_seam.dims:
                extra = [d for d in da_seam.dims if d not in (zdim, ydim, xdim)]
                order = ([zdim] if zdim in da_seam.dims else []) + [ydim, xdim] + extra
                ordered = da_seam.transpose(*order)
                expected_head = ((wet.shape[0],) if zdim in da_seam.dims else ()) + wet.shape[1:]
                if ordered.shape[: len(expected_head)] == expected_head:
                    arr_seam = np.asarray(ordered.values, dtype=np.float64)
                    if flip and zdim in da_seam.dims:
                        arr_seam = arr_seam[::-1]
                    c0 = arr_seam[:, :, 0] if zdim in da_seam.dims else arr_seam[:, 0]
                    c1 = arr_seam[:, :, -1] if zdim in da_seam.dims else arr_seam[:, -1]
                    active = wet[:, :, 0] if zdim in da_seam.dims else wet_seam_2d
                    while active.ndim < c0.ndim:
                        active = active[..., None]
                    if not np.allclose(np.where(active, c0, 0.0), np.where(active, c1, 0.0), rtol=1e-6, atol=0.0, equal_nan=True):
                        raise DataValidationError(f"{label} differs on wet cells at the duplicate longitude endpoint")
        wet, lon_deg = np.ascontiguousarray(wet[..., :-1]), lon_deg[:-1]
        layer_bounds, layer_thickness, cell_volumes, cell_areas = (
            da.isel({xdim: slice(None, -1)}) if isinstance(da, xr.DataArray) and xdim in da.dims else da
            for da in (layer_bounds, layer_thickness, cell_volumes, cell_areas)
        )
    nz, ny, nx = wet.shape
    shape = (nz, ny, nx)
    if centres.size != nz:
        raise _fail(f"The depth coordinate has {centres.size} levels but wet_mask has {nz} along '{zdim}'")
    if not wet.any():
        raise _fail("wet_mask has no wet cells")
    if lat_deg.shape == (ny,) and lon_deg.shape == (nx,):
        finite_coords = np.isfinite(lat_deg[:, None]) & np.isfinite(lon_deg)
    elif lat_deg.shape == lon_deg.shape == (ny, nx):
        finite_coords = np.isfinite(lat_deg) & np.isfinite(lon_deg)
    else:
        raise _fail(f"lat_deg/lon_deg have shapes {lat_deg.shape}, {lon_deg.shape}, expected ({ny},)/({nx},) or ({ny}, {nx})")
    wet_per_column = wet.sum(axis=0)
    if np.any((wet_per_column > 0) & ~finite_coords):
        raise DataValidationError("Latitude and longitude must be finite in wet columns")

    # Which geometry source applies.
    if layer_bounds is None:
        if layer_thickness is not None:
            raise _fail(
                "layer_thickness needs layer_bounds: thickness alone does not locate the layer interfaces",
                suggestions=['Also pass layer_bounds (a (z, 2) or (z, y, x, 2) DataArray) or layer_bounds="from_centres"'],
            )
        if z_attrs.get("bounds"):
            raise _fail(
                f"The depth coordinate declares bounds {z_attrs['bounds']!r} but layer_bounds was not passed",
                details="A bounds variable is not carried along with the coordinate DataArray",
                suggestions=[f"Pass layer_bounds=ds[{z_attrs['bounds']!r}]"],
            )
        raise _fail(
            "4-D tracking needs vertical geometry and none was given",
            suggestions=[
                "Pass layer_bounds, a (z, 2) or (z, y, x, 2) DataArray of layer interfaces in metres",
                "Pass layer_thickness (z, y, x) in metres together with layer_bounds for partial cells",
                'Pass layer_bounds="from_centres" to reconstruct interfaces from the level centres (approximate)',
            ],
        )

    if isinstance(layer_bounds, str):
        if layer_bounds != "from_centres":
            raise _fail(
                f"Invalid layer_bounds string {layer_bounds!r}",
                suggestions=['The only string accepted is "from_centres"'],
            )
        logger.warning(
            'layer_bounds="from_centres": layer interfaces are reconstructed at midpoints between level '
            "centres, so layer thicknesses and volumes are approximate. Pass explicit layer_bounds for exact volumes."
        )
        top, bottom = _interfaces_from_centres(centres)
        source = "reconstructed_from_centres"
    else:
        top, bottom = _interfaces_from_bounds(layer_bounds, zdim, ydim, xdim, negate, flip, shape)
        source = "bounds"

    depth_only = top.ndim == 1 and layer_thickness is None
    bounds_wet = wet
    if top.ndim == 1:
        top, bottom = top[:, None, None], bottom[:, None, None]
        wet_per_level = wet.sum(axis=(1, 2))
        bounds_wet = (wet_per_level > 0)[:, None, None]
    if np.any(bounds_wet & (bottom < top)):
        raise DataValidationError("layer_bounds are inverted on wet cells. The first interface must be the top")
    # Depths are positive down, so a wet cell above the surface means the coordinate is really
    # positive up and is missing its 'positive' attribute.
    with np.errstate(invalid="ignore"):
        above_surface = wet & (top < 0)
    if above_surface.any():
        raise _fail(
            f"Negative depths on {int(above_surface.sum())} wet cell(s): depths must be positive down",
            details="The depth coordinate has no 'positive' attribute, or it contradicts the values"
            if "positive" not in z_attrs
            else "The 'positive' attribute contradicts the sign of the depth coordinate or layer_bounds",
            suggestions=[
                "If depth increases upward (values <= 0), set positive='up' on the depth coordinate",
                "Otherwise check that layer_bounds are positive-down metres",
            ],
            context={"n_cells": int(above_surface.sum()), "positive": z_attrs.get("positive")},
        )

    thickness = bottom - top
    bad = wet & ~(np.isfinite(top) & np.isfinite(bottom) & (thickness > 0))
    if bad.any():
        raise _fail(
            f"layer_bounds give non-finite or non-positive thickness on {int(bad.sum())} wet cell(s)",
            suggestions=["Check layer_bounds against the wet mask", "Mask cells without valid bounds as dry"],
            context={"n_cells": int(bad.sum())},
        )

    # Remember the deepest preceding wet interface, including across dry levels.
    previous_bottom = np.full(top.shape[1:], -np.inf)
    for layer_top, layer_bottom, layer_wet in zip(top, bottom, bounds_wet):
        if np.any(layer_wet & (layer_top < previous_bottom)):
            raise DataValidationError("layer_bounds overlap or are out of depth order on wet cells")
        np.copyto(previous_bottom, layer_bottom, where=layer_wet)

    if layer_thickness is not None:
        thk = _to_zyx(layer_thickness, "layer_thickness", zdim, ydim, xdim, nz, shape)
        if flip:
            thk = thk[::-1]
        with np.errstate(invalid="ignore"):
            bad = wet & ~(np.isfinite(thk) & (thk > 0) & (thk <= thickness * (1.0 + _THICKNESS_RTOL)))
        if bad.any():
            raise _fail(
                f"layer_thickness is outside (0, bounds thickness] on {int(bad.sum())} wet cell(s)",
                details="A partial cell keeps its top interface and is shortened from below",
                suggestions=[
                    "Check that layer_thickness and layer_bounds describe the same levels",
                    "Mask cells with zero thickness as dry",
                ],
                context={"n_cells": int(bad.sum())},
            )
        # Rounding-level excess over the layer is clipped so interfaces stay consistent.
        thickness = np.minimum(thk, thickness)
        bottom = top + thickness
        if source == "bounds":
            source = "bounds+thickness"

    area = _cell_area_m2(cell_areas, grid_resolution, lat_deg, ydim, xdim, ny, nx)
    bad_area = (wet_per_column > 0) & ~(np.isfinite(area) & (area > 0))
    if bad_area.any():
        raise _fail(
            f"Horizontal cell area is non-finite or non-positive in {int(bad_area.sum())} wet column(s)",
            context={"n_columns": int(bad_area.sum())},
        )

    if cell_volumes is not None:
        vol = _to_zyx(cell_volumes, "cell_volumes", zdim, ydim, xdim, nz, shape)
        if flip:
            vol = vol[::-1]
        bad = wet & ~(np.isfinite(vol) & (vol > 0))
        if bad.any():
            raise _fail(
                f"cell_volumes are non-finite or non-positive on {int(bad.sum())} wet cell(s)",
                context={"n_cells": int(bad.sum())},
            )
        volume = np.where(wet, vol, 0.0)
    else:
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            volume = np.zeros(shape, dtype=np.float64)
            np.multiply(area[None, :, :], thickness, out=volume, where=wet)
        bad = wet & ~(np.isfinite(volume) & (volume > 0))
        if bad.any():
            raise DataValidationError(
                f"Area times thickness is non-finite or non-positive on {int(bad.sum())} wet cell(s)",
                context={"n_cells": int(bad.sum())},
            )

    if depth_only:
        top, bottom, thickness = (a[:, 0, 0] for a in (top, bottom, thickness))
    keep = wet_per_level > 0 if depth_only else wet
    top, bottom, thickness = (np.where(keep, a, 0.0) for a in (top, bottom, thickness))
    midpoint = 0.5 * (top + bottom)
    l_z = _median_over_wet_cells(thickness, wet_per_level) if depth_only else float(np.median(thickness[wet]))

    l_h = _median_over_wet_cells(np.sqrt(area).ravel(), wet_per_column.ravel())

    for arr in (top, bottom, thickness, midpoint, area, volume, wet):
        arr.setflags(write=False)
    return VerticalGeometry(
        top=top,
        bottom=bottom,
        thickness=thickness,
        midpoint=midpoint,
        cell_area_m2=area,
        cell_volume=volume,
        wet=wet,
        source=source,
        flip_z=flip,
        default_scales=(l_h, l_z),
    )
