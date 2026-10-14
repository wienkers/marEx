"""The climatology's rolling mean keeps the canonical spatial tile.

xarray's rolling ``mean`` counts the valid samples in each window through a ``construct`` that
lets dask rechunk automatically, which splits every spatial tile about ``window`` ways. The
split reaches the flox reduction, multiplying its task count, and is then undone by the
restore to the caller's layout, so a test of the returned chunks or values cannot see it.
These tests inspect the smoothed field before restoration and the size of the whole graph.
"""

import dask.array as dsa
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.anomaly import climatology
from marEx.core.dimensions import canonical_time_chunks
from marEx.core.numerics import rolling_numerics

DIMENSIONS = {"time": "time", "x": "ncells"}
COORDINATES = {"time": "time", "x": "lon", "y": "lat"}
GRID_DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}


class _Stop(Exception):
    pass


def _cells(nt, nc, chunks):
    t = pd.date_range("2000-01-01", periods=nt, freq="D")
    return xr.DataArray(
        dsa.zeros((nt, nc), dtype=np.float32, chunks=chunks),
        dims=("time", "ncells"),
        coords={"time": t, "lat": ("ncells", np.zeros(nc, np.float32)), "lon": ("ncells", np.zeros(nc, np.float32))},
        name="sst",
    )


def _grid(nt, nlat, nlon, chunks):
    t = pd.date_range("2000-01-01", periods=nt, freq="D")
    return xr.DataArray(
        dsa.zeros((nt, nlat, nlon), dtype=np.float32, chunks=chunks),
        dims=("time", "lat", "lon"),
        coords={"time": t, "lat": np.linspace(-89.5, 89.5, nlat), "lon": np.linspace(0.5, 359.5, nlon)},
        name="sst",
    )


def _smoothed_chunks(monkeypatch, da, dimensions, coordinates, smooth_days):
    """Chunks of the smoothed field as ``rolling_climatology`` receives it."""
    seen = []

    def spy(da_smoothed, *args, **kwargs):
        seen.append(dict(zip(da_smoothed.dims, da_smoothed.chunks)))
        raise _Stop

    monkeypatch.setattr(climatology, "rolling_climatology", spy)
    with pytest.raises(_Stop):
        climatology.smoothed_rolling_climatology(
            da, window_years=2, smooth_days=smooth_days, dimensions=dimensions, coordinates=coordinates
        )
    assert len(seen) == 1
    return seen[0]


@pytest.mark.parametrize("window", [1, 11, 21])
def test_unstructured_smoothing_keeps_the_spatial_tile(monkeypatch, window):
    da = _cells(1096, 50_000, (21, 25_000))
    tile = canonical_time_chunks(da, DIMENSIONS)
    assert tile == {"time": -1, "ncells": 25_000}
    chunks = _smoothed_chunks(monkeypatch, da, DIMENSIONS, COORDINATES, window)
    assert chunks["ncells"] == (25_000, 25_000), chunks


@pytest.mark.parametrize("window", [1, 11, 21])
def test_gridded_smoothing_keeps_the_spatial_tile(monkeypatch, window):
    da = _grid(1096, 60, 360, (30, 30, 360))
    tile = canonical_time_chunks(da, GRID_DIMENSIONS)
    expected = da.chunk(tile)
    chunks = _smoothed_chunks(monkeypatch, da, GRID_DIMENSIONS, None, window)
    assert chunks["lat"] == expected.chunks[1] == (30, 30), chunks
    assert chunks["lon"] == expected.chunks[2] == (360,), chunks


def _anomaly_tasks(nc, smooth_days):
    da = _cells(5844, nc, (21, 25_000))
    ds = marEx.anomaly.compute(
        da,
        "shifting_baseline",
        window_years=5,
        smooth_days=smooth_days,
        compute_mode="lazy",
        validate=False,
        dimensions=DIMENSIONS,
        coordinates=COORDINATES,
        quiet=True,
    )
    return len(dict(ds["dat_anomaly"].data.__dask_graph__()))


@pytest.mark.parametrize("nc", [50_000, 100_000])
def test_smoothing_task_overhead_does_not_scale_with_window(nc):
    """The graph the smoothing adds is the same at every window and small next to the rest, at two sizes."""
    base = _anomaly_tasks(nc, 1)
    smoothed = {window: _anomaly_tasks(nc, window) for window in (11, 21, 61)}
    assert len(set(smoothed.values())) == 1, smoothed
    assert smoothed[21] - base < 0.15 * base, (base, smoothed)


@pytest.mark.parametrize("keep_attrs", ["default", False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("steps", [1, 2, 5, 11, 21])
@pytest.mark.parametrize("cell_chunk", [7, 40])
def test_tile_keeping_mean_matches_xarray_bit_for_bit(dtype, steps, cell_chunk, keep_attrs):
    """Same numbers as xarray's numpy-path rolling mean, NaN gaps and edges included."""
    from marEx.anomaly.climatology import _rolling_mean_keep_tiles

    rng = np.random.default_rng(steps * 100 + cell_chunk)
    values = rng.normal(size=(400, 40)) * 3 + 290
    if np.issubdtype(dtype, np.floating):
        values[rng.random(values.shape) < 0.05] = np.nan
        values[:30, 3] = np.nan
        values[-25:, 5] = np.nan
        values[100:160, 9] = np.nan
    da = xr.DataArray(values.astype(dtype), dims=("time", "ncells"), attrs={"units": "K"})
    da = da.chunk({"time": -1, "ncells": cell_chunk})
    with xr.set_options(keep_attrs=keep_attrs), rolling_numerics():
        got = _rolling_mean_keep_tiles(da, "time", steps)
        want = da.rolling(time=steps, center=True).mean()
    assert got.chunks[1] == da.chunks[1]
    assert got.dtype == want.dtype
    assert got.attrs == want.attrs
    np.testing.assert_array_equal(got.values, want.values)
    xr.testing.assert_identical(got, want)


@pytest.mark.parametrize("steps", [5, 11])
def test_wide_count_branch_matches_xarray(monkeypatch, steps):
    """Windows beyond float32's exact integer range count in float64; same numbers as xarray."""
    from marEx.anomaly import climatology as clim

    monkeypatch.setattr(clim, "_FLOAT32_EXACT_COUNT", 4)
    rng = np.random.default_rng(steps)
    values = (rng.normal(size=(300, 20)) + 290).astype(np.float32)
    values[rng.random(values.shape) < 0.05] = np.nan
    da = xr.DataArray(values, dims=("time", "ncells")).chunk({"time": -1, "ncells": 7})
    with rolling_numerics():
        got = clim._rolling_mean_keep_tiles(da, "time", steps)
        want = da.rolling(time=steps, center=True).mean()
    xr.testing.assert_identical(got, want)
