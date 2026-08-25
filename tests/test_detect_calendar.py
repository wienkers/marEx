"""
Calendar and leap-year regression tests for the marEx detection pipeline.

Covers the §2 findings of the code review:
- §2.1  detrend_harmonic + standardise must not crash on a leap-free span.
- §2.2  fixed_baseline must not NaN day-of-year 366 when the reference period has
        no leap year.
- §2.8  the 1D-harmonic path must be reachable with a partial dimensions dict.
- §2.14 add_decimal_year must work for cftime / non-standard calendars.

Phase C adds the non-daily cadences per calendar. The cftime branch of
``add_decimal_year`` carries its own sub-day conditional, threaded to match the
datetime64 branch exactly, and this is the only place it is exercised end to end --
a calendar-dependent behaviour split would hide precisely here.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.core.time_axis import add_decimal_year


# ── §2.14 cftime decimal year ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "calendar,days_in_year",
    [("noleap", 365), ("360_day", 360), ("all_leap", 366)],
)
def test_add_decimal_year_fixed_length_calendars(calendar, days_in_year):
    """Decimal year is finite, monotone and correctly scaled for fixed-length calendars."""
    time = xr.date_range("2001-01-01", periods=3 * days_in_year, freq="D", calendar=calendar, use_cftime=True)
    da = xr.DataArray(np.arange(time.size, dtype=float), dims=["time"], coords={"time": time})

    result = add_decimal_year(da, dim="time", coord="time")
    dy = np.asarray(result.decimal_year.values)

    assert np.all(np.isfinite(dy))
    assert np.all(np.diff(dy) > 0)  # strictly increasing
    assert abs(dy[0] - 2001.0) < 1e-6  # Jan 1 of first year
    # One day advances the decimal year by 1/days_in_year for these calendars.
    assert abs((dy[1] - dy[0]) - 1.0 / days_in_year) < 1e-6


def test_add_decimal_year_standard_cftime_leap_year():
    """Standard cftime calendar handles a leap year (366 days) without error."""
    time = xr.date_range("2020-01-01", periods=366, freq="D", calendar="standard", use_cftime=True)
    da = xr.DataArray(np.arange(time.size, dtype=float), dims=["time"], coords={"time": time})

    dy = np.asarray(add_decimal_year(da, dim="time", coord="time").decimal_year.values)
    assert np.all(np.isfinite(dy))
    # 2020 is a leap year → 366 days → last day just under 2021.
    assert 2020.0 <= dy[0] < dy[-1] < 2021.0
    assert abs((dy[1] - dy[0]) - 1.0 / 366) < 1e-6


def test_add_decimal_year_numeric_coordinate_still_supported():
    """Regression: a plain numeric time coordinate must use the legacy pd.to_datetime
    path, not the cftime branch (which raised AttributeError: no attribute 'dt')."""
    t = np.arange(0, 20, dtype="int64")
    da = xr.DataArray(np.arange(20.0), dims=["time"], coords={"t": ("time", t)})
    dy = np.asarray(add_decimal_year(da, dim="time", coord="t").decimal_year.values)
    assert np.all(np.isfinite(dy))


def test_add_decimal_year_datetime64_unchanged():
    """The numpy-datetime path still works (regression guard for the branch split)."""
    time = pd.date_range("2000-01-01", periods=365, freq="D")
    da = xr.DataArray(np.arange(time.size, dtype=float), dims=["time"], coords={"time": time})

    dy = np.asarray(add_decimal_year(da, dim="time", coord="time").decimal_year.values)
    assert abs(dy[0] - 2000.0) < 1e-9
    # 2000 is a leap year in the proleptic Gregorian calendar → 366-day divisor.
    assert abs((dy[1] - dy[0]) - 1.0 / 366) < 1e-9


# ── §2.1 leap-free standardise ─────────────────────────────────────────────
def _daily_series(start, end, seed=0):
    """A small dask-backed 1D daily time series over [start, end)."""
    time = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    values = 15.0 + 3.0 * np.sin(2 * np.pi * time.dayofyear / 365.25) + rng.normal(0, 0.5, time.size)
    da = xr.DataArray(values.astype(np.float32), dims=["time"], coords={"time": time}, name="sst")
    return da.chunk({"time": 90})


def test_harmonic_standardise_leap_free_span():
    """§2.1: standardise on a span with no 29 Feb must not raise an align error."""
    # 2021-2023 contains no leap year.
    da = _daily_series("2021-01-01", "2024-01-01")
    result = marEx.preprocess_data(
        da,
        method_anomaly="detrend_harmonic",
        method_extreme="global_percentile",
        standardise=True,
        dimensions={"time": "time"},
        dask_chunks={"time": 90},
    )
    assert "dat_anomaly" in result
    # Force compute to exercise the previously-crashing groupby-divide.
    assert bool(np.isfinite(result.dat_anomaly).any().compute())


# ── §2.8 1D harmonic reachable with partial dimensions ───────────────────────
def test_harmonic_1d_default_dimensions_reachable():
    """§2.8: 1D input with dimensions={'time': 'time'} must not raise a bare KeyError."""
    da = _daily_series("2000-01-01", "2003-01-01")
    result = marEx.preprocess_data(
        da,
        method_anomaly="detrend_harmonic",
        method_extreme="global_percentile",
        dimensions={"time": "time"},
        dask_chunks={"time": 90},
    )
    assert "dat_anomaly" in result


# ── §2.2 fixed_baseline leap reference ───────────────────────────────────────
def test_fixed_baseline_day366_not_nan_with_nonleap_reference():
    """§2.2: day-of-year 366 must not be all-NaN when the reference period is leap-free."""
    # Full series spans 2019-2024 (includes leap years 2020 and 2024 → day 366 exists).
    da = _daily_series("2019-01-01", "2025-01-01")
    result = marEx.preprocess_data(
        da,
        method_anomaly="fixed_baseline",
        method_extreme="global_percentile",
        reference_period=(2021, 2023),  # no leap year
        dimensions={"time": "time"},
        dask_chunks={"time": 90},
    )
    anom = result.dat_anomaly.compute()
    doy = anom["time"].dt.dayofyear.values
    day366 = anom.values[doy == 366]
    assert day366.size >= 1  # there are day-366 timesteps
    assert np.isfinite(day366).any()  # and they are not all NaN


# ── Phase C: non-daily cadences, per calendar ────────────────────────────────
@pytest.mark.parametrize(
    "calendar,days_in_year",
    [("noleap", 365), ("360_day", 360), ("all_leap", 366), ("standard", None)],
)
def test_infer_cycle_monthly_per_calendar(calendar, days_in_year):
    """A monthly axis resolves to the 12-slot cycle on every calendar.

    The 360_day calendar is the discriminating one: its months are all exactly 30 days,
    so the spacings have zero spread, while a standard calendar's run 28-31.
    """
    time = xr.date_range("2001-01-01", periods=48, freq="MS", calendar=calendar, use_cftime=True)
    da = xr.DataArray(np.zeros(len(time)), dims=["time"], coords={"time": time})
    cycle = marEx.infer_cycle(da.time)
    assert cycle.index_name == "month"
    assert cycle.length == 12


@pytest.mark.parametrize("calendar", ["noleap", "360_day", "all_leap", "standard"])
def test_infer_cycle_six_hourly_per_calendar(calendar):
    time = xr.date_range("2001-01-01", periods=400, freq="6h", calendar=calendar, use_cftime=True)
    da = xr.DataArray(np.zeros(len(time)), dims=["time"], coords={"time": time})
    cycle = marEx.infer_cycle(da.time)
    assert cycle.index_name == "hourofyear"
    assert cycle.length == 366 * 4


@pytest.mark.parametrize(
    "calendar,days_in_year",
    [("noleap", 365), ("360_day", 360), ("all_leap", 366)],
)
def test_add_decimal_year_subdaily_cftime_per_calendar(calendar, days_in_year):
    """The cftime branch takes the sub-day term on a sub-daily axis, and scales it by
    that calendar's own year length -- the same arithmetic the datetime64 branch uses."""
    time = xr.date_range("2001-01-01", periods=8, freq="6h", calendar=calendar, use_cftime=True)
    da = xr.DataArray(np.zeros(len(time)), dims=["time"], coords={"time": time})
    dy = np.asarray(add_decimal_year(da).decimal_year.values)
    assert np.all(np.isfinite(dy))
    # Four distinct values inside the first day, each a quarter-day apart.
    assert len(np.unique(dy[:4])) == 4
    np.testing.assert_allclose(np.diff(dy), 0.25 / days_in_year, rtol=1e-9)


@pytest.mark.parametrize(
    "calendar,days_in_year",
    [("noleap", 365), ("360_day", 360), ("all_leap", 366)],
)
def test_add_decimal_year_daily_cftime_takes_no_subday_term(calendar, days_in_year):
    """...and a daily cftime axis does NOT, even when stamped away from midnight.

    This is the conditional that keeps both goldens still. It must depend on the
    cadence, never on the calendar attribute.
    """
    time = xr.date_range("2001-01-01T12:00:00", periods=30, freq="D", calendar=calendar, use_cftime=True)
    da = xr.DataArray(np.zeros(len(time)), dims=["time"], coords={"time": time})
    dy = np.asarray(add_decimal_year(da).decimal_year.values)
    expected = 2001 + (da.time.dt.dayofyear.values - 1) / days_in_year
    np.testing.assert_array_equal(dy, expected)


@pytest.mark.parametrize("calendar", ["noleap", "360_day"])
def test_monthly_pipeline_runs_on_a_cftime_calendar(calendar):
    """End-to-end monthly run on a non-standard calendar: thresholds land on the
    12-slot month axis, not a 366-slot day-of-year one."""
    time = xr.date_range("1990-01-01", periods=12 * 20, freq="MS", calendar=calendar, use_cftime=True)
    rng = np.random.default_rng(3)
    signal = 4.0 * np.sin(2 * np.pi * (np.arange(len(time)) % 12) / 12.0)
    data = signal[:, None, None] + rng.normal(0.0, 0.5, size=(len(time), 3, 3))
    da = xr.DataArray(
        data.astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": np.arange(3.0), "lon": np.arange(3.0)},
    ).chunk({"time": 24})

    ds = marEx.preprocess_data(
        da,
        method_anomaly="fixed_baseline",
        method_extreme="seasonal_percentile",
        dimensions={"time": "time", "x": "lon", "y": "lat"},
        dask_chunks={"time": 24},
    )
    assert ds.sizes["month"] == 12
    assert "dayofyear" not in ds.thresholds.dims
