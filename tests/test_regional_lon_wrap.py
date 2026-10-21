"""The seasonal spatial window wraps longitude only on a grid that spans the globe.

``window_spatial`` pools each cell's samples with its neighbours before the per-slot histogram
quantile. Longitude is periodic only when the grid closes on itself; on a regional box the
western edge has no neighbour beyond it, so its thresholds must not see the eastern edge's data.
Both engines of ``_compute_histogram_quantile_2d`` are pinned: changing ONLY the eastern-most
columns must leave the western-most thresholds untouched on a regional grid and must move them
on a global one.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from marEx.core.time_axis import resolve_cycle
from marEx.extremes import histogram as H

GRID_DIMS = {"time": "time", "y": "lat", "x": "lon"}
N_LAT = 6

REGIONAL_LON = 10.0 + 0.25 * np.arange(8)  # a 2 degree box
GLOBAL_LON = 45.0 * np.arange(8)  # 8 x 45 = 360 degrees


def _field(lon, shift_east=0, edge=2, seed=0):
    """Daily anomalies on (time, lat, lon); ``shift_east`` is added to the last ``edge`` columns only."""
    t = pd.date_range("2000-01-01", periods=365 * 6 + 1, freq="D")
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1, (t.size, N_LAT, lon.size)).astype(np.float32)
    values[:, :, -edge:] += np.float32(shift_east)
    da = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        name="anom",
        coords={"time": t, "lat": np.linspace(-5, 5, N_LAT), "lon": lon},
    )
    cycle = resolve_cycle(da, "time")
    return da.assign_coords({cycle.index_name: cycle.index_of(da["time"]).compute()}).chunk({"time": 365}), cycle


def _thresholds(lon, engine, window_spatial, shift_east):
    da, cycle = _field(lon, shift_east=shift_east)
    out = H._compute_histogram_quantile_2d(
        da,
        q=0.95,
        dimensions=GRID_DIMS,
        cycle=cycle,
        window_steps=cycle.window_steps(11),
        window_spatial=window_spatial,
        precision=0.05,
        max_anomaly=5.0,
        engine=engine,
    )
    return out.transpose("lat", "lon", ...).values


# The slab engine is the production path; the dense engine is the reference and the path even
# windows take.
ENGINES = [("slab", 5), ("dense", 5), ("dense", 4)]


@pytest.mark.parametrize("engine, window_spatial", ENGINES)
def test_regional_west_edge_ignores_east_edge(engine, window_spatial):
    base = _thresholds(REGIONAL_LON, engine, window_spatial, shift_east=0.0)
    moved = _thresholds(REGIONAL_LON, engine, window_spatial, shift_east=2.0)
    half = window_spatial // 2

    # The perturbation is large enough to move the thresholds it reaches...
    assert not np.array_equal(base[:, -half:], moved[:, -half:], equal_nan=True)
    # ...and it does not reach across a regional box.
    np.testing.assert_array_equal(base[:, :half], moved[:, :half])


@pytest.mark.parametrize("engine, window_spatial", ENGINES)
def test_global_west_edge_pools_east_edge(engine, window_spatial):
    base = _thresholds(GLOBAL_LON, engine, window_spatial, shift_east=0.0)
    moved = _thresholds(GLOBAL_LON, engine, window_spatial, shift_east=2.0)
    half = window_spatial // 2

    # On a global grid the western columns' window wraps onto the eastern ones.
    assert not np.array_equal(base[:, :half], moved[:, :half], equal_nan=True)


def _da_with_lon(lon):
    return xr.DataArray(np.zeros((2, lon.size)), dims=("lat", "lon"), coords={"lon": lon})


@pytest.mark.parametrize(
    "lon, expected",
    [
        (REGIONAL_LON, False),
        (GLOBAL_LON, True),
        (np.arange(-180.0, 180.0, 0.25), True),
        (np.arange(360.0, 0.0, -1.0), True),  # descending
        (np.roll(np.arange(0.0, 360.0, 2.5), 50), True),  # rolled across the seam
        (np.arange(0.0, 360.5, 0.5), True),  # repeated seam column, 0 to 360
        (np.arange(0.0, 358.0, 1.0), False),  # two steps short of closing
        (np.arange(0.0, 180.0, 0.5), False),  # a hemisphere
        (10.0 + 0.25 * np.arange(25), False),  # spans 6.25, near 2 pi, but in degrees
    ],
)
def test_x_is_periodic(lon, expected):
    assert H._x_is_periodic(_da_with_lon(lon), "lon") is expected


def test_x_is_periodic_reads_radians_from_units():
    lon = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    da = _da_with_lon(lon)
    assert H._x_is_periodic(da, "lon") is False  # unlabelled: read as 0..6.2 degrees
    da["lon"].attrs["units"] = "radians"
    assert H._x_is_periodic(da, "lon") is True


def test_x_is_periodic_keeps_wrap_without_a_usable_coordinate():
    # No coordinate, or a 2-D one, gives no span to measure: keep the periodic behaviour.
    bare = xr.DataArray(np.zeros((2, 8)), dims=("lat", "lon"))
    assert H._x_is_periodic(bare, "lon") is True
    curvilinear = bare.assign_coords(lon=(("lat", "lon"), np.tile(REGIONAL_LON, (2, 1))))
    assert H._x_is_periodic(curvilinear, "lon") is True


@pytest.mark.parametrize("lon", [REGIONAL_LON, GLOBAL_LON], ids=["regional", "global"])
def test_slab_matches_dense_across_x_tiles(monkeypatch, lon):
    # A small cell cap cuts x into tiles narrower than the halo, so the wrap (global) and the
    # dropped-index pad (regional) both sit on tile boundaries the overlap has to stitch.
    monkeypatch.setattr(H, "_SLAB_TILE_CELLS", 4)
    da, cycle = _field(lon)
    assert H._slab_tile_chunks(da, GRID_DIMS, 5, cycle.length)["lon"] < lon.size
    common = {
        "q": 0.95,
        "dimensions": GRID_DIMS,
        "cycle": cycle,
        "window_steps": cycle.window_steps(11),
        "window_spatial": 5,
        "precision": 0.05,
    }
    xr.testing.assert_identical(
        H._compute_histogram_quantile_2d(da, engine="dense", max_anomaly=5.0, **common),
        H._compute_histogram_quantile_2d(da, engine="slab", max_anomaly=5.0, **common),
    )
