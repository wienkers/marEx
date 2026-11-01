"""The fixed baselines: circular smoothing of the climatology, day-366 pooling, reference periods and detrending.

Hobday et al. (2016) smooth the climatology with a moving average that wraps the year. The wrap
is pinned by SUPPORT (where a spike lands), not by a float tolerance, and ``smooth_days=1`` is
pinned as an exact off switch against an independent unsmoothed reference.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.anomaly.fixed_baseline import (
    _compute_anomaly_detrend_fixed_baseline,
    _compute_anomaly_fixed_baseline,
    _smooth_climatology_circular,
)
from marEx.core.time_axis import DAILY_CYCLE, SeasonalCycle
from marEx.exceptions import ConfigurationError

from ..conftest import assert_no_failures, check
from .conftest import GRID_DIMS, dec31_values, gridded_sst, trend


def _clim_with_spike(slot, length=366, n_x=2):
    data = np.zeros((length, n_x), dtype=np.float32)
    data[slot - 1, :] = 1.0
    return xr.DataArray(data, dims=("dayofyear", "x"), coords={"dayofyear": np.arange(1, length + 1)}).chunk({"dayofyear": -1})


def _daily_field(start="2001-01-01", end="2008-12-31", seed=0):
    time = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    seasonal = np.sin(2 * np.pi * time.dayofyear.values / 365.25)[:, None, None]
    data = (seasonal + rng.normal(0.0, 0.5, (len(time), 2, 3))).astype(np.float32)
    return xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]}).chunk(
        {"time": 200}
    )


def _support(smoothed):
    return np.flatnonzero(smoothed.values[:, 0] != 0)


def test_the_smoothing_support_wraps_the_cycle():
    """A spike spreads over exactly the window, wrapping the year end, and on sub-daily cycles only across days.

    Hourly and 6-hourly: the window spans 21 DAYS at the same hour, never neighbouring hours, or
    the diurnal cycle would be averaged out of the climatology.
    """
    failures = []
    for slot in (1, 366):
        support = set((_support(_smooth_climatology_circular(_clim_with_spike(slot), DAILY_CYCLE, 21).compute()) + 1).tolist())
        expected = {(slot - 1 + k) % 366 + 1 for k in range(-10, 11)}
        if support != expected or len(support) != 21:
            failures.append(f"daily spike at {slot}: support {sorted(support)}, expected {sorted(expected)}")

    hourly = SeasonalCycle("hourofyear", 366 * 24, 1 / 24)
    support = _support(
        _smooth_climatology_circular(_clim_with_spike(5000, length=366 * 24).rename(dayofyear="hourofyear"), hourly, 21).compute()
    )
    if len(support) != 21 or set(((support - 4999) % 24).tolist()) != {0}:
        failures.append(f"hourly spike at 5000: {len(support)} slots, hour offsets {sorted(set(((support - 4999) % 24).tolist()))}")

    sixhourly = SeasonalCycle("hourofyear", 366 * 4, 0.25)
    support = _support(
        _smooth_climatology_circular(_clim_with_spike(3, length=366 * 4).rename(dayofyear="hourofyear"), sixhourly, 21).compute()
    )
    days = set((support // 4 + 1).tolist())
    if days != set(range(1, 12)) | set(range(357, 367)) or set((support % 4).tolist()) != {2}:
        failures.append(f"6-hourly spike at day 1 step 3: days {sorted(days)}, steps {sorted(set((support % 4).tolist()))}")
    assert_no_failures(failures)


def test_the_smoothing_is_the_padded_moving_average():
    """Without NaN the skip-NaN average equals the full-window average over a year padded at both ends.

    It conserves a spike's mass, keeps the single-chunk layout, the labels and float32.
    """
    rng = np.random.default_rng(3)
    clim = xr.DataArray(
        rng.normal(size=(366, 4)).astype(np.float32), dims=("dayofyear", "x"), coords={"dayofyear": np.arange(1, 367)}
    ).chunk({"dayofyear": -1})
    smoothed = _smooth_climatology_circular(clim, DAILY_CYCLE, 21)
    assert smoothed.chunks[0] == (366,)
    assert smoothed.dtype == np.float32
    np.testing.assert_array_equal(smoothed["dayofyear"].values, np.arange(1, 367))
    padded = np.concatenate([clim.values[-10:], clim.values, clim.values[:10]])
    reference = np.stack([padded[i : i + 21].mean(axis=0) for i in range(366)])
    np.testing.assert_allclose(smoothed.compute().values, reference, rtol=0, atol=1e-6)

    spike = _smooth_climatology_circular(_clim_with_spike(200), DAILY_CYCLE, 21).compute()
    assert float(spike.sum("dayofyear")[0]) == pytest.approx(1.0, rel=1e-6)


def test_the_smoothing_off_switches_and_the_rejected_window():
    """``smooth_days=1`` and a window under one monthly slot return the climatology itself; a whole-cycle window is rejected."""
    clim = _clim_with_spike(50)
    assert _smooth_climatology_circular(clim, DAILY_CYCLE, 1) is clim
    monthly = _clim_with_spike(3, length=12).rename(dayofyear="month")
    # 21 days is under one ~30-day slot, so the default does nothing on a month axis.
    assert _smooth_climatology_circular(monthly, SeasonalCycle("month", 12, 30.0), 21) is monthly
    with pytest.raises(ConfigurationError, match="spans 366 of the 366 rows"):
        _smooth_climatology_circular(clim, DAILY_CYCLE, 366)


def test_smooth_days_one_matches_an_independent_unsmoothed_reference():
    da = _daily_field()
    result = _compute_anomaly_fixed_baseline(da, smooth_days=1).dat_anomaly.compute()
    clim = da.groupby("time.dayofyear").mean("time").compute()
    # Day 366 takes every year's 31 December, not only the leap years'.
    dec31 = (da.time.dt.month == 12) & (da.time.dt.day == 31)
    clim.loc[{"dayofyear": 366}] = da.isel(time=dec31.values).mean("time").compute()
    reference = (da.groupby("time.dayofyear") - clim).astype(np.float32).drop_vars("dayofyear").compute()
    np.testing.assert_allclose(result.values, reference.values, rtol=0, atol=1e-6)


def test_default_smooths():
    """Smoothing removes the noise the per-day mean absorbs, so the anomaly variance rises towards the true 0.5**2."""
    da = _daily_field()
    raw = _compute_anomaly_fixed_baseline(da, smooth_days=1).dat_anomaly.compute()
    smoothed = _compute_anomaly_fixed_baseline(da).dat_anomaly.compute()
    assert not np.array_equal(raw.values, smoothed.values)
    assert float(smoothed.std()) > float(raw.std())
    assert raw.chunks == smoothed.chunks


def test_smooth_days_reaches_and_is_recorded_by_both_fixed_methods():
    """``smooth_days`` passed to ``anomaly.compute`` changes the values and is written to the attributes, for both fixed methods."""
    da = _daily_field()
    failures = []
    for method in ("fixed_baseline", "detrend_fixed_baseline"):
        ds = marEx.anomaly.compute(da, method=method, smooth_days=15)
        raw = marEx.anomaly.compute(da, method=method, smooth_days=1)
        if ds.attrs["smooth_days"] != 15:
            failures.append(f"{method}: attrs smooth_days {ds.attrs['smooth_days']}")
        if not any("15-day circular" in step for step in ds.attrs["preprocessing_steps"]):
            failures.append(f"{method}: steps {ds.attrs['preprocessing_steps']}")
        if np.array_equal(ds.dat_anomaly.values, raw.dat_anomaly.values):
            failures.append(f"{method}: smooth_days=15 gives the smooth_days=1 anomaly")
    # The private detrended entry point takes the argument too.
    raw = _compute_anomaly_detrend_fixed_baseline(da, smooth_days=1).dat_anomaly.compute()
    smoothed = _compute_anomaly_detrend_fixed_baseline(da, smooth_days=21).dat_anomaly.compute()
    if np.array_equal(raw.values, smoothed.values):
        failures.append("_compute_anomaly_detrend_fixed_baseline: smooth_days=21 gives the smooth_days=1 anomaly")
    assert_no_failures(failures)


def test_a_seasonal_nan_cell_keeps_exactly_its_valid_days():
    """Smoothing never changes WHICH slots are valid, only their values."""
    da = _daily_field()
    sea_ice = da.time.dt.dayofyear <= 90
    da = da.where(~(sea_ice & (da.lat == 0.0) & (da.lon == 0.0)))
    raw = _compute_anomaly_fixed_baseline(da, smooth_days=1).dat_anomaly.compute()
    smoothed = _compute_anomaly_fixed_baseline(da, smooth_days=21).dat_anomaly.compute()
    np.testing.assert_array_equal(np.isnan(raw.values), np.isnan(smoothed.values))
    assert int(np.isnan(raw.values).sum()) > 0


def test_time_only_fixed_baseline_smooths():
    """A 1-D series (``dimensions={"time"}``, a scalar validity mask) is smoothed as a gridded one is.

    A spike on every 1 January: unsmoothed, the climatology is the spike and the anomaly is zero
    everywhere; smoothed over 21 days, the spike spreads over 21 days of the year, wrapping into
    late December, and the anomaly is non-zero on exactly those days.
    """
    time = pd.date_range("2001-01-01", "2008-12-31", freq="D")
    values = np.zeros(time.size, np.float32)
    values[(time.month == 1) & (time.day == 1)] = 1.0
    series = xr.DataArray(values, dims="time", coords={"time": time}).chunk({"time": 200})
    smoothed = marEx.anomaly.compute(series, "fixed_baseline", dimensions={"time": "time"}, smooth_days=21)
    raw = marEx.anomaly.compute(series, "fixed_baseline", dimensions={"time": "time"}, smooth_days=1)
    assert smoothed.mask.dims == () and bool(smoothed.mask)
    assert not np.any(raw.dat_anomaly.values), "the unsmoothed climatology should be the spike itself"
    doy = time.dayofyear.values
    support = set(doy[smoothed.dat_anomaly.values != 0].tolist())
    assert support == {(k % 366) + 1 for k in range(-10, 11)}, sorted(support)


def test_fixed_baseline_slot_366_pools_every_dec31():
    """Day 366 of the fixed baseline takes every 31 December of its baseline, with or without a leap year in it.

    The input is a pure linear trend, so the expected anomaly on 2008-12-31 is exact. With a
    reference period of 2001-2003 no baseline step falls on day 366 at all.
    """
    da = trend()
    t = da.time.to_index()
    i = t.get_loc(pd.Timestamp("2008-12-31"))
    failures = []
    for case, kwargs, years in (
        ("whole record", {}, range(2000, 2012)),
        ("reference 2001-2003", {"reference_period": (2001, 2003)}, range(2001, 2004)),
    ):
        anom = marEx.anomaly.compute(da, method="fixed_baseline", smooth_days=1, **kwargs)
        got = float(anom.dat_anomaly.transpose("time", ...).values[i, 0, 0])
        want = float(da.values[i, 0, 0]) - dec31_values(da, years).mean()
        if not np.isclose(got, want, rtol=1e-6, atol=0):
            failures.append(f"{case}: 2008-12-31 anomaly {got}, expected {want}")
    assert_no_failures(failures)


def test_subdaily_reference_period_without_a_leap_year_pools_dec31_per_time_of_day():
    """A sub-daily day 366 has one slot per time of day, each pooled from the same time on 31 December.

    With no leap year in the reference period no step of it falls on day 366, so the pool must still be
    built for the full series that reads day 366. Forward-filling the empty slots instead would give
    every time of day the last slot's (18:00) climatology.
    """
    time = pd.date_range("2000-01-01", "2008-12-31T18:00", freq="6h")
    values = np.arange(time.size, dtype=np.float64)[:, None, None] * np.ones((1, 1, 3))
    da = xr.DataArray(
        values.astype(np.float32),
        coords={"time": time, "lat": [0.0], "lon": np.arange(3, dtype=float)},
        dims=("time", "lat", "lon"),
    ).chunk({"time": 1460})
    anom = marEx.anomaly.compute(da, method="fixed_baseline", smooth_days=1, reference_period=(2001, 2003))
    out = anom.dat_anomaly.transpose("time", ...).values
    t = da.time.to_index()
    for hour in (0, 6, 12, 18):
        i = t.get_loc(pd.Timestamp(f"2008-12-31T{hour:02d}:00"))
        pooled = np.mean([da.values[t.get_loc(pd.Timestamp(f"{y}-12-31T{hour:02d}:00")), 0, 0] for y in range(2001, 2004)])
        np.testing.assert_allclose(out[i, 0, 0], float(da.values[i, 0, 0]) - pooled, rtol=1e-6, err_msg=f"{hour:02d}:00")


def test_fixed_baseline_reference_period_sets_the_climatology():
    """A reference period restricts the climatology to its years: a warm first half reads warm against a cool baseline.

    Two warm years (+2) then two cool ones: against the whole record the warm years' anomaly is
    about +1; against 2002-2003 it is about +2. Every timestep is kept.
    """
    times = pd.date_range("2000-01-01", "2003-12-31", freq="D")
    base = np.random.default_rng(42).standard_normal((len(times), 3, 3)).astype(np.float32)
    base[times.year <= 2001] += 2.0
    da = xr.DataArray(
        base, dims=["time", "lat", "lon"], coords={"time": times, "lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0]}
    ).chunk({"time": -1})
    full = marEx.anomaly.compute_normalised_anomaly(da, method_anomaly="fixed_baseline", dimensions=GRID_DIMS)
    ref = marEx.anomaly.compute_normalised_anomaly(
        da, method_anomaly="fixed_baseline", reference_period=(2002, 2003), dimensions=GRID_DIMS
    )
    assert ref.sizes["time"] == da.sizes["time"]
    warm = {"time": slice("2000-01-01", "2001-12-31")}
    mean_full = float(full.dat_anomaly.sel(**warm).mean())
    mean_ref = float(ref.dat_anomaly.sel(**warm).mean())
    assert mean_ref > mean_full + 0.5, (
        f"warm-year anomaly against 2002-2003 {mean_ref:.2f}, against the whole record {mean_full:.2f}"
    )


def _detrend_input():
    """Three by four cells of the gridded fixture over ~4 years, one cell masked out."""
    sst = gridded_sst().isel(lon=slice(0, 4), lat=slice(0, 3), time=slice(0, 1500)).load()
    return sst.where(~((sst.lat == sst.lat[1]) & (sst.lon == sst.lon[1]))).chunk({"time": 25})


def test_detrend_orders(marex_warnings, marex_records, capsys):
    """``detrend_orders`` defaults to linear and is recorded; orders without the linear term warn through logging.

    The warning is a WARNING record on the marEx logger, never a bare print: ``quiet=True`` still
    records it, and with console output off nothing reaches stdout. Time need not be the first
    dimension of the input: the output is time-first with the same NaN pattern.
    """
    sst = _detrend_input()

    def run(da, orders):
        return marEx.anomaly.compute(
            da, "detrend_fixed_baseline", detrend_orders=orders, dimensions=GRID_DIMS, dask_chunks={"time": 25}
        )

    failures = []
    for orders, recorded in ((None, [1]), ([1, 2], [1, 2])):
        attrs = run(sst, orders).attrs
        if attrs["detrend_orders"] != recorded:
            failures.append(f"detrend_orders={orders}: attrs {attrs['detrend_orders']}, expected {recorded}")

    message = "without linear term"
    for orders, n_expected in (([2, 3], 1), ([1, 2, 3], 0)):
        with marex_warnings as cap:
            run(sst, orders)
        n = sum(message in m for m in cap.messages)
        if n != n_expected:
            failures.append(f"detrend_orders={orders}: {n} warnings naming the missing linear term, expected {n_expected}")

    try:
        for case, logging_kwargs, check_stdout in (
            ("quiet=True", {"quiet": True}, False),
            ("console_output=False", {"console_output": False}, True),
        ):
            marEx.configure_logging(**logging_kwargs)
            records = marex_records(logging.DEBUG)
            capsys.readouterr()
            run(sst, [2, 3])
            out = capsys.readouterr().out
            n = sum(r.levelno == logging.WARNING and message in r.getMessage() for r in records)
            if n != 1:
                failures.append(f"{case}: {n} WARNING records naming the missing linear term, expected 1")
            if check_stdout and message in out:
                failures.append(f"{case}: the warning reached stdout")
    finally:
        marEx.configure_logging()

    reference = run(sst, [1]).dat_anomaly
    transposed = run(sst.transpose("lat", "lon", "time"), [1]).dat_anomaly
    if transposed.dims[0] != "time":
        failures.append(f"time-last input: output dims {transposed.dims}")
    elif not np.array_equal(np.isnan(transposed.transpose(*reference.dims).values), np.isnan(reference.values)):
        failures.append("time-last input: NaN pattern differs from the time-first run")
    if not np.isnan(reference.values).any():
        failures.append("the input has no NaN cell; the NaN-pattern check would prove nothing")
    assert_no_failures(failures)


def test_detrend_orders_and_zero_mean_through_the_pipeline():
    """Every detrend order set gives a 95th-percentile event frequency near 5 %, through the whole pipeline.

    Rows: linear, quadratic only, linear to cubic, on the full gridded fixture with one masked
    cell. ``force_zero_mean`` (default True) leaves a near-zero mean, and turning it off changes
    the anomaly.
    """
    sst = gridded_sst()
    sst = sst.where(~((sst.lat == sst.lat[1]) & (sst.lon == sst.lon[1])), np.nan)
    failures = []
    for name, orders in (("linear", [1]), ("quadratic", [2]), ("cubic", [1, 2, 3])):
        result = marEx.preprocess_data(
            sst, method_anomaly="detrend_fixed_baseline", detrend_orders=orders, dimensions=GRID_DIMS, dask_chunks={"time": 25}
        )
        frequency = float(result.extreme_events.mean())
        if not 0.025 < frequency < 0.075:
            failures.append(f"{name} {orders}: event frequency {frequency}")
        if result.attrs["detrend_orders"] != orders:
            failures.append(f"{name} {orders}: attrs {result.attrs['detrend_orders']}")

    anomalies = {
        flag: marEx.anomaly.compute(
            sst, "detrend_fixed_baseline", force_zero_mean=flag, dimensions=GRID_DIMS, dask_chunks={"time": 25}
        ).dat_anomaly.values
        for flag in (True, False)
    }
    mean_true = float(np.nanmean(anomalies[True]))
    if abs(mean_true) >= 0.01:
        failures.append(f"force_zero_mean=True: mean {mean_true}")
    if np.array_equal(anomalies[True], anomalies[False], equal_nan=True):
        failures.append("force_zero_mean=False gives the force_zero_mean=True anomaly")
    assert_no_failures(failures)


def test_harmonic_1d_leap_free_standardise():
    """A 1-D series with ``dimensions={"time"}`` reaches the harmonic path, and standardises over a span with no 29 February.

    The leap-free span used to raise an align error in the standardisation's groupby-divide; a
    partial dimensions dict used to raise a bare KeyError.
    """
    time = pd.date_range("2021-01-01", "2024-01-01", freq="D")
    rng = np.random.default_rng(0)
    values = 15.0 + 3.0 * np.sin(2 * np.pi * time.dayofyear / 365.25) + rng.normal(0, 0.5, time.size)
    da = xr.DataArray(values.astype(np.float32), dims=["time"], coords={"time": time}, name="sst").chunk({"time": 90})
    result = marEx.preprocess_data(
        da,
        method_anomaly="detrend_harmonic",
        method_extreme="global_percentile",
        standardise=True,
        dimensions={"time": "time"},
        dask_chunks={"time": 90},
    )
    assert "dat_anomaly" in result
    assert bool(np.isfinite(result.dat_anomaly).any().compute())


MASK_INPUT_ATTRS = {"standard_name": "sea_surface_temperature", "units": "K", "valid_min": 250.0}
MASK_METHODS = {"detrend_harmonic": {}, "fixed_baseline": {}, "shifting_baseline": {"window_years": 2}}


def test_mask_attrs_are_its_own():
    """The output ``mask`` describes itself, not the input field, under every anomaly method.

    It is built from the input's first timestep, and xarray carries the input's attrs through that
    comparison, so without care a boolean mask reports the SST's ``standard_name``, ``units`` and
    ``valid_min``. The caller's input attrs are left untouched.
    """
    time = pd.date_range("2000-01-01", periods=6 * 365, freq="D")
    data = np.random.default_rng(0).standard_normal((time.size, 3, 4)).astype(np.float32) + 290.0
    data[:, 0, 0] = np.nan
    failures = []
    for method, kwargs in MASK_METHODS.items():

        def row(method=method, kwargs=kwargs):
            sst = xr.DataArray(
                data,
                coords={"time": time, "lat": np.arange(3.0), "lon": np.arange(4.0)},
                dims=("time", "lat", "lon"),
                name="sst",
                attrs=dict(MASK_INPUT_ATTRS),
            ).chunk({"time": 365})
            mask = marEx.anomaly.compute(sst, method=method, **kwargs).mask
            assert not set(MASK_INPUT_ATTRS) & set(mask.attrs), mask.attrs
            assert mask.attrs["long_name"] == "valid-data mask", mask.attrs
            assert mask.dtype == bool, mask.dtype
            assert not bool(mask[0, 0]) and bool(mask[1, 1]), mask.values
            assert sst.attrs == MASK_INPUT_ATTRS, f"the caller's input attrs changed: {sst.attrs}"

        check(failures, method, row)
    assert_no_failures(failures)
