"""
Synthetic volumetric extremes for tracking tests.

Events are painted as moving ellipsoids ("balls") on a small ``(time, depth, lat, lon)``
index grid. Every scenario is small enough for the interactive node and is built so its
per-timestep object counts are known by construction; ``scenarios()`` records them.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

EARTH_RADIUS_M = 6_378_000.0

DIM_NAMES = ("time", "depth", "lat", "lon")


@dataclass(frozen=True)
class Ball:
    """
    An ellipsoid present for timesteps ``t0..t1`` (inclusive), in index units.

    The centre at timestep ``t`` is ``(z, y, x) + (vz, vy, vx) * (t - t0)``. A voxel
    ``(k, j, i)`` is inside when ``((k-zc)/rz)^2 + ((j-yc)/ry)^2 + ((i-xc)/rx)^2 <= 1``.
    """

    t0: int
    t1: int
    z: float
    y: float
    x: float
    rz: float
    ry: float
    rx: float
    vz: float = 0.0
    vy: float = 0.0
    vx: float = 0.0


def paint(nt: int, nz: int, ny: int, nx: int, balls: Sequence[Ball]) -> NDArray[np.bool_]:
    """Return the ``(nt, nz, ny, nx)`` union of ``balls``."""
    out = np.zeros((nt, nz, ny, nx), dtype=bool)
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    for b in balls:
        for t in range(max(b.t0, 0), min(b.t1, nt - 1) + 1):
            dt = t - b.t0
            zc, yc, xc = b.z + b.vz * dt, b.y + b.vy * dt, b.x + b.vx * dt
            inside = ((k - zc) / b.rz) ** 2 + ((j - yc) / b.ry) ** 2 + ((i - xc) / b.rx) ** 2 <= 1.0
            out[t] |= inside
    return out


def make_inputs(
    binary: NDArray[np.bool_],
    *,
    thickness: Sequence[float],
    lat0: float = 30.0,
    lon0: float = -170.0,
    dlat: float = 0.5,
    dlon: float = 0.5,
    time_chunk: int = 4,
    dry: Optional[NDArray[np.bool_]] = None,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Wrap a boolean ``(time, depth, lat, lon)`` array as tracker inputs.

    Returns
    -------
    extremes : xarray.DataArray
        Dask-backed bool ``(time, depth, lat, lon)``, chunked ``time_chunk`` in time and whole in space.
    wet : xarray.DataArray
        ``(depth, lat, lon)`` wet mask (``~dry``, all wet by default).
    layer_bounds : xarray.DataArray
        ``(depth, nv)`` layer interfaces in metres, positive down, from ``thickness``.
    cell_areas : xarray.DataArray
        ``(lat, lon)`` cell areas in m^2 on a sphere of radius ``EARTH_RADIUS_M``.
    """
    binary = np.asarray(binary, dtype=bool)
    nt, nz, ny, nx = binary.shape
    thickness = np.asarray(thickness, dtype=np.float64)
    if thickness.shape != (nz,):
        raise ValueError(f"thickness has shape {thickness.shape}, expected ({nz},)")
    edges = np.concatenate([[0.0], np.cumsum(thickness)])
    depth = 0.5 * (edges[:-1] + edges[1:])
    lat = lat0 + dlat * np.arange(ny, dtype=np.float64)
    lon = lon0 + dlon * np.arange(nx, dtype=np.float64)
    coords = {
        "time": pd.date_range("2000-01-01", periods=nt, freq="D"),
        "depth": xr.DataArray(depth, dims="depth", attrs={"units": "m", "positive": "down"}),
        "lat": lat,
        "lon": lon,
    }
    extremes = xr.DataArray(binary, dims=DIM_NAMES, coords=coords, name="extreme_events").chunk(
        {"time": time_chunk, "depth": -1, "lat": -1, "lon": -1}
    )
    wet_values = np.ones((nz, ny, nx), dtype=bool) if dry is None else ~np.asarray(dry, dtype=bool)
    spatial = {k: coords[k] for k in ("depth", "lat", "lon")}
    wet = xr.DataArray(wet_values, dims=("depth", "lat", "lon"), coords=spatial, name="mask")
    layer_bounds = xr.DataArray(
        np.stack([edges[:-1], edges[1:]], axis=1),
        dims=("depth", "nv"),
        coords={"depth": coords["depth"]},
        attrs={"units": "m"},
    )
    band = np.sin(np.deg2rad(lat + dlat / 2)) - np.sin(np.deg2rad(lat - dlat / 2))
    areas = EARTH_RADIUS_M**2 * np.deg2rad(dlon) * np.broadcast_to(band[:, None], (ny, nx))
    cell_areas = xr.DataArray(
        np.ascontiguousarray(areas), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, attrs={"units": "m2"}
    )
    return extremes, wet, layer_bounds, cell_areas


_SHAPE = (4, 10, 12)  # (nz, ny, nx)
_THICKNESS = [10.0, 10.0, 20.0, 20.0]


def _scenario(
    nt: int,
    balls: Sequence[Ball],
    n_objects: Sequence[int],
    shape: Tuple[int, int, int] = _SHAPE,
    thickness: Sequence[float] = tuple(_THICKNESS),
    **extra,
) -> Tuple[NDArray[np.bool_], dict]:
    binary = paint(nt, *shape, balls)
    meta = {
        "make_inputs": {"thickness": list(thickness), "time_chunk": extra.pop("time_chunk", 4)},
        "max_gap_m": extra.pop("max_gap_m", 0.0),
        "n_objects": list(n_objects),
        "has_merge": extra.pop("has_merge", False),
    }
    meta.update(extra)
    return binary, meta


def scenarios() -> Dict[str, Tuple[NDArray[np.bool_], dict]]:
    """
    Named synthetic scenarios, each ``(binary, meta)``.

    ``meta["make_inputs"]`` holds the keyword arguments for :func:`make_inputs`,
    ``meta["max_gap_m"]`` the vertical gap to bridge, ``meta["n_objects"]`` the number of
    volumetric objects at each timestep, and ``meta["has_merge"]`` whether some child
    overlaps two parents.
    """
    out: Dict[str, Tuple[NDArray[np.bool_], dict]] = {}

    # One event: absent at t=0 and t=7, grows at t=3..4 (a larger ball unioned on), then decays.
    out["persist_grow_decay"] = _scenario(
        8,
        [
            Ball(1, 6, z=1.5, y=5.0, x=5.0, rz=1.2, ry=2.0, rx=2.0, vx=0.25),
            Ball(3, 4, z=1.5, y=5.0, x=5.5, rz=2.0, ry=3.5, rx=3.5),
        ],
        [0, 1, 1, 1, 1, 1, 1, 0],
    )

    # Two concurrent but spatially separate events with different lifetimes.
    out["birth_death"] = _scenario(
        8,
        [
            Ball(0, 4, z=1.0, y=2.0, x=2.0, rz=1.0, ry=1.5, rx=1.5),
            Ball(2, 7, z=2.0, y=7.0, x=9.0, rz=1.0, ry=1.5, rx=1.5),
        ],
        [1, 1, 2, 2, 2, 1, 1, 1],
    )

    # Two stacked blobs in the same columns separated by the 10 m level 1: one object when
    # gaps up to 10 m are bridged.
    out["vertical_gap_join"] = _scenario(
        5,
        [
            Ball(0, 4, z=0.0, y=5.0, x=5.0, rz=0.4, ry=2.0, rx=2.0),
            Ball(0, 4, z=2.0, y=5.0, x=5.0, rz=0.4, ry=2.0, rx=2.0),
        ],
        [1, 1, 1, 1, 1],
        max_gap_m=10.0,
    )

    # Two objects sit side by side (t=0..2) and are covered by one stationary object from t=3.
    out["one_merge"] = _scenario(
        6,
        [
            Ball(0, 2, z=1.5, y=5.0, x=3.0, rz=1.5, ry=2.0, rx=1.2),
            Ball(0, 2, z=1.5, y=5.0, x=8.0, rz=1.5, ry=2.0, rx=1.2),
            Ball(3, 5, z=1.5, y=5.0, x=5.5, rz=1.5, ry=2.0, rx=3.6),
        ],
        [2, 2, 2, 1, 1, 1],
        has_merge=True,
    )

    # The time reverse: one object at t=0..2 separates into two at t=3.
    out["one_split"] = _scenario(
        6,
        [
            Ball(0, 2, z=1.5, y=5.0, x=5.5, rz=1.5, ry=2.0, rx=3.6),
            Ball(3, 5, z=1.5, y=5.0, x=3.0, rz=1.5, ry=2.0, rx=1.2),
            Ball(3, 5, z=1.5, y=5.0, x=8.0, rz=1.5, ry=2.0, rx=1.2),
        ],
        [1, 1, 1, 2, 2, 2],
    )

    # Merge at t=3, separate again at t=6.
    out["merge_then_split"] = _scenario(
        9,
        [
            Ball(0, 2, z=1.5, y=5.0, x=3.0, rz=1.5, ry=2.0, rx=1.2),
            Ball(0, 2, z=1.5, y=5.0, x=8.0, rz=1.5, ry=2.0, rx=1.2),
            Ball(3, 5, z=1.5, y=5.0, x=5.5, rz=1.5, ry=2.0, rx=3.6),
            Ball(6, 8, z=1.5, y=5.0, x=3.0, rz=1.5, ry=2.0, rx=1.2),
            Ball(6, 8, z=1.5, y=5.0, x=8.0, rz=1.5, ry=2.0, rx=1.2),
        ],
        [2, 2, 2, 1, 1, 1, 2, 2, 2],
        has_merge=True,
    )

    # A merge exactly at the first timestep of the second time chunk (time_chunk=4).
    out["cascade_at_chunk_boundary"] = _scenario(
        8,
        [
            Ball(0, 3, z=1.5, y=5.0, x=3.0, rz=1.5, ry=2.0, rx=1.2),
            Ball(0, 3, z=1.5, y=5.0, x=8.0, rz=1.5, ry=2.0, rx=1.2),
            Ball(4, 7, z=1.5, y=5.0, x=5.5, rz=1.5, ry=2.0, rx=3.6),
        ],
        [2, 2, 2, 2, 1, 1, 1, 1],
        has_merge=True,
        time_chunk=4,
    )

    # 130 single-voxel parents (two cells apart, so none touch) all covered by one child at t=1.
    parents = [Ball(0, 0, z=0.0, y=2.0 * j, x=2.0 * i, rz=0.4, ry=0.4, rx=0.4) for j in range(10) for i in range(13)]
    out["many_parents_130"] = _scenario(
        2,
        parents + [Ball(1, 1, z=0.0, y=9.0, x=12.0, rz=0.4, ry=1.0e3, rx=1.0e3)],
        [130, 1],
        shape=(1, 19, 25),
        thickness=[10.0],
        has_merge=True,
    )

    # A merge cascade in one water column. At t=0..1: a one-voxel parent B in the top level and
    # eleven flat parents Q1 (deepest) .. Q11 (shallowest), each one level thick and separated
    # by a thin empty level. At t=2 one column covers every level. Layer thicknesses double with
    # depth, so each Q level holds at least half of the column above it but less than half of
    # the column down to the next Q. The column first links to B and Q1 only; each partition
    # leaves B a piece small enough for the next Q up to reach the overlap threshold, so
    # resolving the merge takes 11 partition passes. The Q parents are wider than the column, so
    # their own volume never sets the overlap fraction.
    n_q = 11
    cascade_thickness = [0.25]
    for k in range(n_q, 0, -1):
        cascade_thickness += [0.25, 4.0 * 2.0 ** (n_q - k)]
    nz = len(cascade_thickness)
    cascade = [Ball(0, 1, z=0.0, y=2.0, x=2.0, rz=0.4, ry=0.4, rx=0.4)]
    cascade += [Ball(0, 1, z=2.0 + 2.0 * j, y=2.0, x=2.0, rz=0.4, ry=1.0, rx=1.0) for j in range(n_q)]
    cascade.append(Ball(2, 2, z=nz / 2.0, y=2.0, x=2.0, rz=2.0 * nz, ry=0.4, rx=0.4))
    out["cascade_11_iterations"] = _scenario(
        3,
        cascade,
        [n_q + 1, n_q + 1, 1],
        shape=(nz, 5, 5),
        thickness=cascade_thickness,
        has_merge=True,
    )
    return out
