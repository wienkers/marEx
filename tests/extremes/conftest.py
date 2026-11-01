"""Shared inputs and helpers for the extremes-stage tests.

Every builder returns a field whose values live in memory (numpy, optionally wrapped in a dask
graph by ``.chunk``), never a ``.persist()``ed one: a persisted fixture holds futures on
whichever client was current when it was built.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from marEx.core.time_axis import resolve_cycle
from marEx.extremes import histogram as H

DATA_DIR = Path(__file__).parent.parent / "data"

GRID_DIMS = {"time": "time", "x": "lon", "y": "lat"}
MESH_DIMS = {"time": "time", "x": "ncells"}
MESH_COORDS = {"time": "time", "x": "lon", "y": "lat"}


def check(failures, case, func):
    """Run one table row's assertions; record the first that fails, or the error the row raised, under its case id."""
    try:
        func()
    except (AssertionError, pytest.fail.Exception) as e:
        # pytest.fail.Exception is a BaseException: a row whose pytest.raises/warns did not
        # fire ("DID NOT RAISE") would otherwise abort the table instead of being recorded.
        failures.append(f"{case}: {str(e).strip().splitlines()[0] if str(e).strip() else 'assertion failed'}")
    except Exception as e:  # noqa: BLE001 - a row that raises is a failing row; the others still run
        failures.append(f"{case}: raised {type(e).__name__}: {e}")


def legacy_bin_edges(precision=0.01, max_anomaly=5.0, dtype=np.float64):
    """The legacy asymmetric edges, one bin for every negative value, verbatim from each driver."""
    if dtype == np.float32:
        return np.concatenate(
            [[-np.inf], np.arange(-precision, max_anomaly + precision, precision, dtype=np.float32)], dtype=np.float32
        )
    return np.concatenate([[-np.inf], np.arange(-precision, max_anomaly + precision, precision)])


def anomaly_fixture(n_time=400, n_y=6, n_x=7, seed=0, chunks=None, name="da", dates=True):
    """A deterministic dask-backed N(0, 1) anomaly field on (time, lat, lon), time unchunked by default."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(n_time, n_y, n_x)).astype(np.float32)
    time = pd.date_range("2000-01-01", periods=n_time, freq="D") if dates else np.arange(n_time)
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": np.arange(n_y, dtype=np.float32), "lon": np.arange(n_x, dtype=np.float32)},
        name=name,
    )
    return da.chunk(chunks or {"time": -1, "lat": 3, "lon": 7})


def with_dayofyear(da):
    """``da`` with the ``dayofyear`` coordinate the 2-D driver expects."""
    return da.assign_coords(dayofyear=da.time.dt.dayofyear)


class warnings_as_errors:
    """Context manager turning UserWarning into an error, for the no-warning controls."""

    def __enter__(self):
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error", UserWarning)
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


# ── slab engine inputs ────────────────────────────────────────────────


def slab_field(freq="D", periods=365 * 6 + 1, mesh=False, seed=0, shape=(6, 7)):
    """Daily (by default) N(0, 1) anomalies with an all-NaN strip and a gappy cell, and its cycle index."""
    t = pd.date_range("2000-01-01", periods=periods, freq=freq)
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1, (t.size,) + shape).astype(np.float32)
    values[:, 0, :2] = np.nan  # an all-NaN strip
    values[::5, 3, 3] = np.nan  # a cell with gaps
    if mesh:
        da = xr.DataArray(values.reshape(t.size, 42), dims=("time", "ncells"), coords={"time": t}, name="anom")
    else:
        da = xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            name="anom",
            coords={"time": t, "lat": np.linspace(-5, 5, shape[0]), "lon": np.linspace(0, 12, shape[1])},
        )
    cycle = resolve_cycle(da, "time")
    return da.assign_coords({cycle.index_name: cycle.index_of(da["time"]).compute()}), cycle


def both_engines(da, cycle, dims, tchunk, **kw):
    """The dense and the slab thresholds of ``da`` (chunked ``tchunk`` in time), unevaluated."""
    da = da.chunk({"time": tchunk})
    common = dict(dimensions=dims, cycle=cycle, window_steps=cycle.window_steps(11), precision=0.05, max_anomaly=5.0, **kw)
    dense = H._compute_histogram_quantile_2d(da, engine="dense", **common)
    slab = H._compute_histogram_quantile_2d(da, engine="slab", **common)
    return dense, slab


def edges_and_centres(precision=0.05, max_anomaly=5.0):
    edges = H._symmetric_bin_edges(precision, max_anomaly, np.float32)
    return edges, ((edges[1:] + edges[:-1]) / 2).astype(np.float32)


# ── longitude wrap inputs ─────────────────────────────────────────────

REGIONAL_LON = 10.0 + 0.25 * np.arange(8)  # a 2 degree box
GLOBAL_LON = 45.0 * np.arange(8)  # 8 x 45 = 360 degrees


def lon_field(lon, shift_east=0.0, edge=2, seed=0):
    """Daily anomalies on (time, 6 lat, lon) with the cycle index; ``shift_east`` is added to the last ``edge`` columns only."""
    t = pd.date_range("2000-01-01", periods=365 * 6 + 1, freq="D")
    values = np.random.default_rng(seed).normal(0, 1, (t.size, 6, lon.size)).astype(np.float32)
    values[:, :, -edge:] += np.float32(shift_east)
    da = xr.DataArray(
        values, dims=("time", "lat", "lon"), name="anom", coords={"time": t, "lat": np.linspace(-5, 5, 6), "lon": lon}
    )
    cycle = resolve_cycle(da, "time")
    return da.assign_coords({cycle.index_name: cycle.index_of(da["time"]).compute()}).chunk({"time": 365}), cycle
