"""The rolling-mean and strided-window kernels under the climatology's smoothing.

``_rolling_mean_keep_tiles`` is xarray's numpy-path rolling mean with both windows built in
place, so it keeps the spatial tile; ``_strided_window_sum`` takes every ``stride``-th step,
i.e. the same time of day on consecutive days, a few rows at a time. Both are pinned against
an independent reference: xarray itself, the daily kernel per time of day, or one numpy sum
over the whole stepped window view.

The tests over parameter grids are tables of one behaviour each: every row runs, a failing row
is reported with its case id and does not hide the others, and the chunk-structure assertion is
made on every row.
"""

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from marEx.anomaly import climatology
from marEx.anomaly.climatology import _min_count, _rolling_mean_keep_tiles, _strided_window_sum, _subdaily_grid_positions
from marEx.core.numerics import rolling_numerics

from ..conftest import assert_no_failures


def _check(failures, case, func):
    """Run one row's assertions; record the first that fails, or the error the row raised, under its case id."""
    try:
        func()
    except AssertionError as e:
        failures.append(f"{case}: {str(e).strip().splitlines()[0] if str(e).strip() else 'assertion failed'}")
    except Exception as e:  # noqa: BLE001 - a row that raises is a failing row; the others still run
        failures.append(f"{case}: raised {type(e).__name__}: {e}")


def _gappy(steps_seed, dtype, extra_gaps=False):
    rng = np.random.default_rng(steps_seed)
    values = rng.normal(size=(400, 40)) * 3 + 290
    if np.issubdtype(dtype, np.floating):
        values[rng.random(values.shape) < 0.05] = np.nan
        values[:30, 3] = np.nan
        values[-25:, 5] = np.nan
        values[100:160, 9] = np.nan
        if extra_gaps:
            values[200:400:7, 11] = np.nan
    return values


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32], ids=["float32", "float64", "int32"])
def test_tile_keeping_mean_matches_xarray_bit_for_bit(dtype):
    """Same numbers, dtype and attrs as xarray's numpy-path rolling mean, NaN gaps and edges included.

    Rows: window steps {1, 2, 5, 11, 21} x ``keep_attrs`` {default, False}, on 7-wide uneven
    spatial tiles; every row asserts the spatial chunks are kept.
    """
    failures = []
    for steps in (1, 2, 5, 11, 21):
        values = _gappy(steps * 100 + 7, dtype)
        da = xr.DataArray(values.astype(dtype), dims=("time", "ncells"), attrs={"units": "K"}).chunk({"time": -1, "ncells": 7})
        for keep_attrs in ("default", False):

            def row(da=da, steps=steps, keep_attrs=keep_attrs):
                with xr.set_options(keep_attrs=keep_attrs), rolling_numerics():
                    got = _rolling_mean_keep_tiles(da, "time", steps)
                    want = da.rolling(time=steps, center=True).mean()
                assert got.chunks[1] == da.chunks[1], f"spatial chunks {got.chunks[1]} != {da.chunks[1]}"
                assert got.dtype == want.dtype, f"dtype {got.dtype} != {want.dtype}"
                assert got.attrs == want.attrs, f"attrs {got.attrs} != {want.attrs}"
                np.testing.assert_array_equal(got.values, want.values)
                xr.testing.assert_identical(got, want)

            _check(failures, f"steps={steps} keep_attrs={keep_attrs}", row)
    assert_no_failures(failures)


def test_wide_count_branch_matches_xarray(monkeypatch):
    """Windows beyond float32's exact integer range count in float64; same numbers as xarray."""
    monkeypatch.setattr(climatology, "_FLOAT32_EXACT_COUNT", 4)
    rng = np.random.default_rng(5)
    values = (rng.normal(size=(300, 20)) + 290).astype(np.float32)
    values[rng.random(values.shape) < 0.05] = np.nan
    da = xr.DataArray(values, dims=("time", "ncells")).chunk({"time": -1, "ncells": 7})
    with rolling_numerics():
        got = climatology._rolling_mean_keep_tiles(da, "time", 5)
        want = da.rolling(time=5, center=True).mean()
    xr.testing.assert_identical(got, want)


def test_missing_fraction_resolves_to_a_valid_count():
    """A window of ``steps`` needs ``steps - floor(max_missing * steps)`` valid samples, never fewer than one."""
    rows = [(10, 0.0, 10), (5, 0.1, 5), (10, 0.1, 9), (11, 0.1, 10), (21, 0.1, 19), (100, 0.29, 71), (1000, 0.9999999999, 1)]
    failures = [
        f"steps={steps} max_missing={mm}: {_min_count(steps, mm)}, expected {want}"
        for steps, mm, want in rows
        if _min_count(steps, mm) != want
    ]
    assert_no_failures(failures)


def test_missing_fraction_matches_xarray_min_periods_inside_the_record():
    """Up to ``max_missing`` of a window may be NaN; a window reaching past the record stays NaN.

    Rows: float32 at window steps {2, 5, 10, 11, 21} x ``max_missing`` {0.05, 0.1, 0.25, 0.5},
    plus float64 at (21, 0.25). Each row also checks the rule is live: some window with a gap
    inside the record has a value that the strict (complete-window) mean does not.
    """
    rows = [(np.float32, steps, mm) for steps in (2, 5, 10, 11, 21) for mm in (0.05, 0.1, 0.25, 0.5)]
    rows.append((np.float64, 21, 0.25))
    failures = []
    for dtype, steps, mm in rows:
        values = _gappy(steps * 100 + int(mm * 100), np.float64, extra_gaps=True)
        da = xr.DataArray(values.astype(dtype), dims=("time", "ncells"), attrs={"units": "K"}).chunk({"time": -1, "ncells": 7})

        def row(da=da, dtype=dtype, steps=steps, mm=mm):
            with rolling_numerics():
                got = _rolling_mean_keep_tiles(da, "time", steps, max_missing=mm)
                want = da.rolling(time=steps, center=True, min_periods=_min_count(steps, mm)).mean().values
            back, fwd = steps // 2, (steps - 1) // 2
            want[:back] = np.nan
            want[want.shape[0] - fwd :] = np.nan
            assert got.chunks[1] == da.chunks[1], f"spatial chunks {got.chunks[1]} != {da.chunks[1]}"
            assert got.dtype == dtype, f"dtype {got.dtype}"
            assert got.attrs == da.attrs, f"attrs {got.attrs}"
            np.testing.assert_array_equal(got.values, want)
            if _min_count(steps, mm) < steps:
                with rolling_numerics():
                    strict = _rolling_mean_keep_tiles(da, "time", steps).values
                assert (np.isnan(strict) & ~np.isnan(got.values)).any(), "no gap-spanning window gained a value"

        _check(failures, f"{np.dtype(dtype).name} steps={steps} max_missing={mm}", row)
    assert_no_failures(failures)


@pytest.mark.parametrize("stride", [4, 24])
def test_strided_window_is_the_daily_window_per_time_of_day(stride):
    """Each time of day is smoothed as a daily series would be, even windows and ragged ends included.

    Rows: days {1, 2, 4, 5, 21} x a record ending on a whole day or 3 steps into one x
    ``max_missing`` {0, 0.25}, plus 0.1 at 21 days, so every distinct minimum count is kept. The
    kernel never reads the time coordinate, so the start hour is not a dimension of the table.
    """
    rows = [(days, partial, mm) for days in (1, 2, 4, 5, 21) for partial in (0, 3) for mm in (0.0, 0.25)]
    rows += [(21, partial, 0.1) for partial in (0, 3)]
    failures = []
    for days, partial_steps, max_missing in rows:
        nt = 60 * stride + partial_steps
        rng = np.random.default_rng(stride * 1000 + days * 10 + partial_steps)
        values = (rng.normal(size=(nt, 6)) * 3 + 290).astype(np.float32)
        values[rng.random(values.shape) < 0.04] = np.nan
        time = pd.date_range("2001-01-01", periods=nt, freq=f"{24 // stride}h")
        da = xr.DataArray(values, dims=("time", "ncells"), coords={"time": time}).chunk({"time": -1, "ncells": 4})

        def row(da=da, days=days, max_missing=max_missing, values=values):
            got = _rolling_mean_keep_tiles(da, "time", days, stride, max_missing).compute()
            want = np.full(values.shape, np.nan, dtype=np.float32)
            for phase in range(stride):
                daily = da.isel(time=slice(phase, None, stride))
                want[phase::stride] = _rolling_mean_keep_tiles(daily, "time", days, max_missing=max_missing).values
            # Same samples and masks; numpy's pairwise sum groups a strided window of 8 or more samples
            # differently, which moves the mean by at most a couple of float32 ULP.
            np.testing.assert_array_equal(np.isnan(got.values), np.isnan(want))
            np.testing.assert_allclose(got.values, want, rtol=1e-6)
            assert got.dims == da.dims, f"dims {got.dims}"

        _check(failures, f"stride={stride} days={days} partial_steps={partial_steps} max_missing={max_missing}", row)
    assert_no_failures(failures)


def _old_strided_window_sum(obj, timedim, steps, stride, fill_value, **kwargs):
    """The strided window sum as first written: one numpy sum over the whole stepped window view."""
    span = (steps - 1) * stride + 1
    windows = obj.rolling({timedim: span}).construct(
        "_w", fill_value=fill_value, sliding_window_view_kwargs={"automatic_rechunk": False}
    )
    return windows.isel(_w=slice(None, None, stride)).sum("_w", skipna=False, **kwargs)


@pytest.mark.parametrize("time_chunk", [-1, 37, 5], ids=["whole", "overlap", "halo-merge"])
def test_strided_window_sum_is_unchanged_by_blocking(time_chunk):
    """The blocked strided sum is bit-identical to one numpy sum over the whole stepped window view.

    The time chunking is the case: whole, chunks longer than the halo, and chunks shorter than it
    (merged locally, then restored). Rows: stride {4, 24} x days {1, 2, 5, 21} x a whole-day or
    ragged record end; every row asserts the returned chunks are the input's.
    """
    failures = []
    for stride in (4, 24):
        for days in (1, 2, 5, 21):
            for partial_steps in (0, 3):
                nt = 60 * stride + partial_steps
                rng = np.random.default_rng(stride * 100 + days)
                values = (rng.normal(size=(nt, 7)) * 3 + 290).astype(np.float32)
                values[rng.random(values.shape) < 0.04] = np.nan
                da = xr.DataArray(values, dims=("time", "ncells")).chunk({"time": time_chunk, "ncells": 4})

                def row(da=da, stride=stride, days=days):
                    for obj, fill, kwargs in ((da.fillna(0), 0, {}), (da.notnull(), False, {"dtype": np.float32})):
                        want = _old_strided_window_sum(obj, "time", days, stride, fill, **kwargs).values
                        got = _strided_window_sum(obj, "time", days, stride, fill, block_elements=50, **kwargs).values
                        assert got.dtype == want.dtype, f"dtype {got.dtype} != {want.dtype}"
                        np.testing.assert_array_equal(got, want)
                    chunks = _strided_window_sum(da.fillna(0), "time", days, stride, 0).chunks
                    assert chunks == da.chunks, f"chunks {chunks} != {da.chunks}"

                _check(failures, f"stride={stride} days={days} partial_steps={partial_steps}", row)
    assert_no_failures(failures)


def test_window_longer_than_the_record_is_nan():
    """A strided window reaching past the record end everywhere leaves every value NaN.

    nt 40 and 41 straddle the window's forward reach (40 steps for 21 days at stride 4).
    """
    failures = []
    for nt in (3, 39, 40, 41):
        da = xr.DataArray(np.ones((nt, 3), np.float32), dims=("time", "ncells")).chunk({"time": -1, "ncells": 2})
        got = _rolling_mean_keep_tiles(da, "time", 21, 4, 0.1).compute()
        if got.shape != da.shape:
            failures.append(f"nt={nt}: shape {got.shape}")
        if not bool(got.isnull().all()):
            failures.append(f"nt={nt}: {int(got.notnull().sum())} values are not NaN")
    assert_no_failures(failures)


def test_strided_smoothing_memory_does_not_scale_with_stride():
    """A 6-hourly (stride 4) smoothing of one tile peaks no higher than a daily (stride 1) one of the same tile.

    One numpy sum over the stepped window view copies the whole (time, cells, window) view first, so the
    peak grew with the window: about 23 tiles at 21 days x 4 steps a day against 4 at stride 1.
    """
    import tracemalloc

    values = np.random.default_rng(0).standard_normal((1460, 4000)).astype(np.float32)
    da = xr.DataArray(values, dims=("time", "ncells")).chunk({"time": -1, "ncells": -1})

    def peak(stride):
        with dask.config.set(scheduler="synchronous"):
            tracemalloc.start()
            _rolling_mean_keep_tiles(da, "time", 21, stride, 0.1).compute()
            _, top = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        return top

    daily, subdaily = peak(1), peak(4)
    assert subdaily <= 1.5 * daily, (subdaily / values.nbytes, daily / values.nbytes)


def test_long_nanosecond_axis_is_on_its_grid():
    """Offsets are taken in microseconds, so a regular axis longer than 292 years is not rejected."""
    time = pd.date_range("1800-01-01", "2100-01-01", freq="6h").values.astype("datetime64[ns]")
    assert _subdaily_grid_positions(xr.DataArray(time, dims="time"), 4) is None
    gappy = np.delete(time, [5, 400_000])
    positions = _subdaily_grid_positions(xr.DataArray(gappy, dims="time"), 4)
    assert positions[-1] == time.size - 1
    np.testing.assert_array_equal(positions, np.delete(np.arange(time.size), [5, 400_000]))


def _sst_like(n_time=24 * 365, n_cells=40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    field = 290.0 + 5.0 * np.sin(2 * np.pi * t / 365.25)[:, None] + rng.normal(0.0, 1.0, (n_time, n_cells))
    return xr.DataArray(field.astype(np.float32), dims=("time", "cell"))


def test_rolling_numerics_pins_the_numpy_path():
    """``rolling_numerics`` pins xarray's rolling mean to the numpy path on every xarray release.

    bottleneck's ``move_mean`` keeps a float32 running sum over the whole chunk, so on a
    time-whole series (the canonical layout detect smooths on) its error grows with the series
    length: ~6e-3 K after 24 years of SST, against ~1e-4 K for the numpy path, and it restarts
    at every block boundary, so the result also depended on the time chunking.
    """
    da = _sst_like()
    window = 11
    reference = sliding_window_view(da.values.astype(np.float64), window, axis=0).mean(axis=-1)
    with rolling_numerics():
        smoothed = da.chunk({"time": -1}).rolling(time=window, center=True).mean().values
    got = smoothed[window // 2 : window // 2 + reference.shape[0]]
    # numpy path: ~1e-4 at 290 K in float32. bottleneck's running sum reaches 1.5e-3 on this series.
    assert np.abs(got - reference).max() < 5e-4

    short = _sst_like(n_time=4 * 365)
    with rolling_numerics():
        whole = short.chunk({"time": -1}).rolling(time=11, center=True).mean().values
        split = short.chunk({"time": 25}).rolling(time=11, center=True).mean().values
    np.testing.assert_array_equal(whole, split)
