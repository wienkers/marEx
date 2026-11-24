"""Exact horizontal and seasonal pooling against native NumPy percentiles."""

import os
from pathlib import Path
from time import perf_counter

import dask
import numpy as np
import pytest
import xarray as xr
from distributed import Client, LocalCluster, futures_of

import marEx
from marEx.core.time_axis import infer_cycle
from marEx.exceptions import ConfigurationError
from marEx.extremes import _exact_spatial_pool as pool
from marEx.extremes import seasonal_percentile as seasonal

from .conftest import GRID_DIMS


def _oracle(values, slots, n_slots, window_steps, width, percentile, periodic):
    """Enumerate every real neighbour in time, y and x-offset order."""
    _, ny, nx = values.shape
    half = width // 2
    result = np.full((n_slots, ny, nx), np.nan, dtype=np.float32)
    full = window_steps >= n_slots
    for index in range(1 if full else n_slots):
        times = [
            t
            for t, slot in enumerate(slots)
            if 1 <= slot <= n_slots
            and (full or min((slot - index - 1) % n_slots, (index + 1 - slot) % n_slots) <= window_steps // 2)
        ]
        if not times:
            continue
        for y, x in np.ndindex(ny, nx):
            ys = range(max(0, y - half), min(ny, y + half + 1))
            xs = [xx % nx for xx in range(x - half, x + half + 1)] if periodic else range(max(0, x - half), min(nx, x + half + 1))
            observations = values[np.ix_(times, ys, xs)].ravel()
            if not np.isnan(observations).all():
                with np.errstate(over="ignore", invalid="ignore"):
                    result[index, y, x] = np.nanpercentile(observations, percentile, method="linear")
    if full:
        result[1:] = result[0]
    return result


def _assert_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    wet = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[wet].view(np.uint32), expected[wet].view(np.uint32))


def _grid(values, *, periodic=False, calendar="standard", start="2000-12-26"):
    _, ny, nx = values.shape
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={
            "time": xr.date_range(start, periods=len(values), freq="D", calendar=calendar),
            "lat": np.arange(ny),
            "lon": np.arange(nx) * (360 / nx if periodic else 0.5),
        },
        name="temperature_anomaly",
    )


def _identify(da, **kwargs):
    return marEx.extremes.identify(da, dimensions=GRID_DIMS, method_percentile="exact", verbose=False, **kwargs)


def _assert_public(result, da, days, width, percentile, tail, periodic):
    cycle = infer_cycle(da.time)
    slots = cycle.index_of(da.time).values
    steps = cycle.window_slots(days, calendar=da.time.dt.calendar)
    reference = _oracle(da.values, slots, cycle.length, steps, width, percentile, periodic)
    reference[:, np.isnan(da.values).all(axis=0)] = np.nan
    tiny = np.finfo(np.float32).tiny
    if tail == "upper":
        reference[reference <= 0] = tiny
        flags = da.values >= reference[slots - 1]
    else:
        reference[reference >= 0] = -tiny
        flags = da.values <= reference[slots - 1]
    np.testing.assert_array_equal(result[cycle.index_name], cycle.labels)
    assert result.thresholds.dims == (cycle.index_name, "lat", "lon")
    _assert_bits(result.thresholds, reference)
    np.testing.assert_array_equal(result.extreme_events, flags)


@pytest.mark.benchmark
@pytest.mark.skipif(not os.environ.get("MAREX_RUN_TIMING_TESTS"), reason="Run in a fresh process on an idle core")
def test_cold_and_warm():
    """Measure actual first compilation, then compare warm execution with the oracle."""
    assert not pool._crossing_values.signatures
    values = np.random.default_rng(12).normal(size=(366 * 3, 8, 8)).astype(np.float32)
    slots = np.arange(len(values)) % 366 + 1
    args = values, slots, 366, 11, 2, (0, 0), (8, 8), 95.0
    start = perf_counter()
    result = pool._pooled_percentiles(*args)
    cold = perf_counter() - start
    assert len(pool._crossing_values.signatures) == 1
    warm = []
    for _ in range(3):
        start = perf_counter()
        pool._pooled_percentiles(*args)
        warm.append(perf_counter() - start)
    start = perf_counter()
    ref = _oracle(values, slots, 366, 11, 5, 95.0, False)
    direct = perf_counter() - start
    _assert_bits(result, ref)
    print(f"Cold: {cold:.3f} s, warm median: {np.median(warm):.3f} s, NumPy: {direct:.3f} s")
    assert cold <= 1.2
    assert np.median(warm) * 2 < direct


@pytest.mark.parametrize("width", [1, 3, 5, 7])
@pytest.mark.parametrize("percentile", [5.2, 10, 95, 99])
@pytest.mark.parametrize("dtype,tail", [(np.float32, "upper"), (np.float64, "lower"), (np.float64, "upper"), (np.float32, "lower")])
def test_public_exact(width, percentile, dtype, tail, monkeypatch):
    """Compare flags and float32 bits, including seasonal gaps and wet halos around land."""
    monkeypatch.setattr(pool, "_MIN_COMPILED_WORK", 0)
    periodic = width in (3, 7)
    nx = 2 if width == 7 else 5
    values = np.random.default_rng(41).normal(loc=0.1, size=(16, 3, nx)).astype(dtype)
    values[:, 1, 1] = np.nan
    values[4:10, 2, 0] = np.nan
    da = _grid(values, periodic=periodic)
    result = _identify(
        da.chunk({"time": 5, "lat": 2, "lon": 2}),
        window_days=5,
        window_spatial=width,
        threshold_percentile=percentile,
        tail=tail,
        compute_mode="lazy",
    ).compute()
    assert result.attrs["window_spatial"] == width
    _assert_public(result, da, 5, width, percentile, tail, periodic)
    if width > 1:
        assert np.isfinite(result.thresholds.sel(dayofyear=1).values[2, 0])
        assert result.thresholds.isel(lat=1, lon=1).isnull().all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "kind", ["zeros", "negative_zero", "infinities", "overflow", "subnormals", "native_rounding", "cell_constants"]
)
def test_native_endpoints(dtype, kind, monkeypatch):
    """Force compiled refinement on floating-point cases visible before clamping."""
    monkeypatch.setattr(pool, "_MIN_COMPILED_WORK", 0)
    tiny = np.finfo(dtype).smallest_subnormal
    largest = np.finfo(dtype).max
    native = {
        "zeros": [-0.0, 0.0, np.nan, -0.0, np.nan, 0.0, 1.0, -1.0],
        "negative_zero": [-0.0, np.nan, -0.0],
        "infinities": [-np.inf, np.inf, -np.inf, np.inf, np.nan, -1.0, 1.0],
        "overflow": [-largest] * 20 + [largest],
        "subnormals": [-tiny, tiny, -tiny, tiny, np.nan, tiny, -tiny],
        "native_rounding": [1.000000055, 1.000000064, 1.000000057],
        "cell_constants": [1.0] * 18 + [1.001, 1.002, 1.003],
    }
    data = np.array(native[kind], dtype=dtype)
    values = np.broadcast_to(data[:, None, None], (data.size, 2, 3)).copy()
    if kind == "overflow":
        # Only one wet cell gives N=21 and an integral p95 rank.
        values[:, 0, 1:] = np.nan
        values[:, 1] = np.nan
    if kind == "cell_constants":
        for x in range(3):
            values[:, 0, x] = dtype(1 + x / 1000)
        # Put the differing cell constants and variable postings in one bucket.
        monkeypatch.setattr(pool, "_sample_edges", lambda _: np.zeros(pool._BUCKETS - 1))
    slots = np.arange(len(values)) % 5 + 1
    slots[-3:-1] = [-1, 6] if kind != "overflow" else [1, 1]
    if kind == "overflow":
        slots[:] = 1
    values.flags.writeable = False
    values = values[:, :, ::-1]
    for steps in (3, 5):
        for percentile in (0.0, 5.2, 50.0, 95.0, 99.0, 100.0):
            actual = pool._pooled_percentiles(values, slots, 5, steps, 1, (0, 0), (2, 3), percentile)
            expected = _oracle(values, slots, 5, steps, 3, percentile, False)
            _assert_bits(actual, expected)
    if kind == "overflow":
        actual = pool._pooled_percentiles(values, slots, 5, 5, 1, (0, 0), (2, 3), 95.0)
        assert np.isnan(actual).all()


@pytest.mark.parametrize("dtype,percentile", [(np.float32, np.float64(5.2)), (np.float16, 5.2), (np.int64, 95)])
def test_numpy_dispatch(dtype, percentile, monkeypatch):
    """Retain oracle promotion and support non-native dtypes."""
    monkeypatch.setattr(pool, "_MIN_COMPILED_WORK", 0)
    values = np.arange(30).reshape(5, 2, 3).astype(dtype)
    slots = np.arange(5) + 1

    def reject(*args):
        pytest.fail("Compiled refinement cannot reproduce this dtype promotion")

    monkeypatch.setattr(pool, "_crossing_values", reject)
    actual = pool._pooled_percentiles(values, slots, 5, 3, 1, (0, 0), (2, 3), percentile, periodic=True)
    _assert_bits(actual, _oracle(values, slots, 5, 3, 3, percentile, True))


def test_counts_exceed_uint16(monkeypatch):
    """A full-cycle pooled bucket can contain more than 65,535 observations."""
    monkeypatch.setattr(pool, "_MIN_COMPILED_WORK", 0)
    values = np.ones((7283, 3, 3), dtype=np.float32)
    values.ravel()[65_536:] = 2
    slots = np.ones(len(values), dtype=np.int32)
    actual = pool._pooled_percentiles(values, slots, 1, 1, 1, (0, 0), (3, 3), 99.0)
    _assert_bits(actual, _oracle(values, slots, 1, 1, 3, 99.0, False))


@pytest.mark.parametrize("calendar", ["noleap", "360_day"])
@pytest.mark.parametrize("window_days", [5, 367])
def test_non_gregorian_calendar(calendar, window_days, monkeypatch):
    """Use expected cycle labels across the year seam and count full-cycle slots once."""
    monkeypatch.setattr(pool, "_MIN_COMPILED_WORK", 0)
    da = _grid(np.random.default_rng(10).normal(size=(18, 2, 3)).astype(np.float32), calendar=calendar)
    result = _identify(da.chunk({"time": 4}), window_days=window_days, window_spatial=3, compute_mode="lazy").compute()
    _assert_public(result, da, window_days, 3, 95, "upper", False)


@pytest.mark.parametrize("periodic", [False, True])
def test_mosaic_extra_slices_and_seam(periodic, monkeypatch):
    """Force multiple halo tiles, preserving extra slices and the periodic seam."""
    monkeypatch.setattr(pool, "_MIN_COMPILED_WORK", 0)
    values = np.random.default_rng(44).normal(size=(25, 7, 9)).astype(np.float32)
    values[::3, 1, 1] = -0.0
    values[::4, 2, 1] = +0.0
    da = _grid(values, periodic=periodic)
    volume = xr.concat((da, da * -2.0), dim=xr.IndexVariable("depth", [1, 10])).transpose("time", "depth", "lat", "lon")
    options = dict(window_spatial=3, window_days=5, threshold_percentile=10, tail="lower", compute_mode="lazy")
    single = _identify(volume.chunk({"time": 5, "depth": 2, "lat": 4, "lon": 3}), **options)
    output_chunks = single.extreme_events.chunksizes
    single = single.compute()
    monkeypatch.setattr(seasonal, "_EXACT_POOL_TASK_BYTES", 1800 * 1024)
    cycle = infer_cycle(da.time)
    haloed = da.assign_coords(dayofyear=cycle.index_of(da.time))
    tiled = seasonal._exact_spatial_threshold(haloed, GRID_DIMS, cycle, 5, 3, 10)
    assert len(tiled.chunksizes["lat"]) * len(tiled.chunksizes["lon"]) > 1
    mosaic = _identify(volume.chunk({"time": 5, "depth": 2, "lat": 4, "lon": 3}), **options)
    assert mosaic.extreme_events.chunksizes == output_chunks
    mosaic = mosaic.compute()
    _assert_bits(mosaic.thresholds, single.thresholds)
    np.testing.assert_array_equal(mosaic.extreme_events, single.extreme_events)
    for depth in volume.depth.values:
        panel = volume.sel(depth=depth, drop=True)
        _assert_public(mosaic.sel(depth=depth, drop=True), panel, 5, 3, 10, "lower", periodic)
    if periodic:
        moved = (
            _identify(volume.roll(lon=4, roll_coords=True).chunk({"time": 5}), **options).compute().roll(lon=-4, roll_coords=True)
        )
        _assert_bits(moved.thresholds, mosaic.thresholds)
        np.testing.assert_array_equal(moved.extreme_events, mosaic.extreme_events)


def test_compute_modes(tmp_path):
    """Check resident bytes with a live scheduler and read the streamed result back."""
    da = _grid(np.random.default_rng(21).normal(size=(15, 3, 4)).astype(np.float32)).chunk({"time": 5, "lat": 2})
    options = dict(window_spatial=3, window_days=3)
    with LocalCluster(n_workers=1, threads_per_worker=1, processes=False, memory_limit="1.1GiB", dashboard_address=None) as cluster:
        with Client(cluster) as client, dask.config.set(scheduler=None):
            assert next(iter(client.scheduler_info()["workers"].values()))["memory_limit"] > 0
            lazy = _identify(da, compute_mode="lazy", **options)
            assert not futures_of(lazy)
            assert not client.futures
            assert not any(client.nbytes().values())
            expected = lazy.compute()
            persisted = _identify(da, compute_mode="persist", **options).compute()
            streamed = _identify(da, compute_mode="streaming", scratch_dir=tmp_path, **options)
            staging = Path(streamed.encoding["marex_staging_dir"])
            try:
                assert staging.is_dir()
                xr.testing.assert_identical(persisted, expected)
                xr.testing.assert_identical(streamed.compute(), expected)
            finally:
                marEx.clear_staging(streamed)
            assert not staging.exists()


def test_pipeline_passes_spatial_window():
    """The combined pipeline uses the same estimator and records its spatial width."""
    values = np.random.default_rng(14).normal(size=(367, 2, 3)).astype(np.float32)
    da = _grid(values).chunk({"time": 60})
    result = marEx.preprocess_data(
        da,
        dimensions=GRID_DIMS,
        method_anomaly="fixed_baseline",
        smooth_days=1,
        method_percentile="exact",
        window_days=367,
        window_spatial=np.int64(3),
        compute_mode="lazy",
        verbose=False,
    ).compute()
    assert result.attrs["window_spatial"] == 3
    _assert_public(result, result.dat_anomaly.transpose("time", "lat", "lon"), 367, 3, 95, "upper", False)


@pytest.mark.parametrize("width", [True, np.bool_(True), False, 0, -3, 3.0, 2.5, "3"])
def test_invalid_width(width):
    """Spatial widths must be positive odd integers; booleans are invalid."""
    da = _grid(np.ones((5, 2, 2), dtype=np.float32))
    with pytest.raises(ConfigurationError, match="window_spatial must be a positive odd integer"):
        _identify(da, window_spatial=width, compute_mode="lazy")
