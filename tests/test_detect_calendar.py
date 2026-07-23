"""
Calendar and leap-year regression tests for the marEx detection pipeline.

Covers the §2 findings of the code review:
- §2.1  detrend_harmonic + std_normalise must not crash on a leap-free span.
- §2.2  fixed_baseline must not NaN day-of-year 366 when the reference period has
        no leap year.
- §2.8  the 1D-harmonic path must be reachable with a partial dimensions dict.
- §2.14 add_decimal_year must work for cftime / non-standard calendars.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.detect.utils import add_decimal_year


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


# ── §2.1 leap-free std_normalise ─────────────────────────────────────────────
def _daily_series(start, end, seed=0):
    """A small dask-backed 1D daily time series over [start, end)."""
    time = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    values = 15.0 + 3.0 * np.sin(2 * np.pi * time.dayofyear / 365.25) + rng.normal(0, 0.5, time.size)
    da = xr.DataArray(values.astype(np.float32), dims=["time"], coords={"time": time}, name="sst")
    return da.chunk({"time": 90})


def test_harmonic_std_normalise_leap_free_span():
    """§2.1: std_normalise on a span with no 29 Feb must not raise an align error."""
    # 2021-2023 contains no leap year.
    da = _daily_series("2021-01-01", "2024-01-01")
    result = marEx.preprocess_data(
        da,
        method_anomaly="detrend_harmonic",
        method_extreme="global_extreme",
        std_normalise=True,
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
        method_extreme="global_extreme",
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
        method_extreme="global_extreme",
        reference_period=(2021, 2023),  # no leap year
        dimensions={"time": "time"},
        dask_chunks={"time": 90},
    )
    anom = result.dat_anomaly.compute()
    doy = anom["time"].dt.dayofyear.values
    day366 = anom.values[doy == 366]
    assert day366.size >= 1  # there are day-366 timesteps
    assert np.isfinite(day366).any()  # and they are not all NaN
