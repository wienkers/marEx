"""
Day-of-year 366 climatology pools every 31 December, not only the leap years'.

Grouping by calendar day of year gives slot 366 only the leap years' 31 December: one sample in a
five-year window where every other slot has five. On a trending series that lone sample sits
years away from the target year, and the anomaly on a leap year's 31 December carries the trend
(a whole-grid one-day "event"). Slot 366 instead takes the 31 December of every year, which in a
non-leap year is day 365, so it is as well sampled as its neighbours.

The input is a pure linear trend, so every expected climatology is an exact mean of known values.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.anomaly.climatology import rolling_climatology


def _trend(start="2000-01-01", end="2011-12-31", n_cells=3):
    time = pd.date_range(start, end, freq="D")
    values = np.arange(time.size, dtype=np.float64)[:, None, None] * np.ones((1, n_cells))
    return xr.DataArray(
        values.astype(np.float32),
        coords={"time": time, "lat": [0.0], "lon": np.arange(n_cells, dtype=float)},
        dims=("time", "lat", "lon"),
        name="sst",
    ).chunk({"time": 365})


def _dec31_values(da, years):
    t = da.time.to_index()
    return np.array([float(da.values[t.get_loc(pd.Timestamp(f"{y}-12-31")), 0, 0]) for y in years])


@pytest.mark.parametrize("target", [2008, 2009])
def test_rolling_climatology_slot_366_pools_every_dec31(target):
    da = _trend()
    clim = rolling_climatology(da, window_years=5).transpose("time", ...).compute()
    t = da.time.to_index()

    expected = _dec31_values(da, range(target - 5, target)).mean()
    if pd.Timestamp(f"{target}-12-31").dayofyear == 366:
        got = float(clim.values[t.get_loc(pd.Timestamp(f"{target}-12-31")), 0, 0])
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    # Day 365 keeps its own calendar grouping: a leap year's 30 December and the others' 31 December.
    day365 = [f"{y}-12-30" if pd.Timestamp(f"{y}-12-31").dayofyear == 366 else f"{y}-12-31" for y in range(target - 5, target)]
    expected365 = np.mean([float(da.values[t.get_loc(pd.Timestamp(d)), 0, 0]) for d in day365])
    on_365 = f"{target}-12-30" if pd.Timestamp(f"{target}-12-31").dayofyear == 366 else f"{target}-12-31"
    np.testing.assert_allclose(float(clim.values[t.get_loc(pd.Timestamp(on_365)), 0, 0]), expected365, rtol=1e-6)


def test_window_without_a_leap_year_still_has_a_day_366_climatology():
    # 2005-2007 holds no leap year; target 2008 used to get a NaN climatology on 31 December.
    da = _trend("2005-01-01", "2008-12-31")
    clim = rolling_climatology(da, window_years=3).transpose("time", ...).compute()
    t = da.time.to_index()
    got = float(clim.values[t.get_loc(pd.Timestamp("2008-12-31")), 0, 0])
    np.testing.assert_allclose(got, _dec31_values(da, range(2005, 2008)).mean(), rtol=1e-6)


def test_fixed_baseline_slot_366_pools_every_dec31():
    da = _trend()
    anom = marEx.anomaly.compute(da, method="fixed_baseline", smooth_days=1)
    t = da.time.to_index()
    i = t.get_loc(pd.Timestamp("2008-12-31"))
    expected = float(da.values[i, 0, 0]) - _dec31_values(da, range(2000, 2012)).mean()
    np.testing.assert_allclose(float(anom.dat_anomaly.transpose("time", ...).values[i, 0, 0]), expected, rtol=1e-6)


@pytest.mark.parametrize("calendar", ["noleap", "standard"])
def test_series_without_a_day_366_is_unchanged(calendar):
    # A no-leap calendar, or a span with no leap year, has no timestep on day 366: its slot is
    # only read through the fixed baseline's circular smoothing, which must not see it change.
    start, end = ("2000-01-01", "2005-12-31") if calendar == "noleap" else ("2001-01-01", "2003-12-31")
    time = xr.date_range(start, end, freq="D", calendar=calendar, use_cftime=calendar == "noleap")
    from marEx.anomaly.climatology import last_day_pool
    from marEx.core.time_axis import resolve_cycle

    da = xr.DataArray(np.zeros(len(time)), coords={"time": time}, dims="time")
    assert last_day_pool(da.time, resolve_cycle(da, "time", None)) is None


def test_fixed_baseline_reference_period_without_a_leap_year_still_pools_dec31():
    da = _trend()
    anom = marEx.anomaly.compute(da, method="fixed_baseline", smooth_days=1, reference_period=(2001, 2003))
    t = da.time.to_index()
    i = t.get_loc(pd.Timestamp("2008-12-31"))
    expected = float(da.values[i, 0, 0]) - _dec31_values(da, range(2001, 2004)).mean()
    np.testing.assert_allclose(float(anom.dat_anomaly.transpose("time", ...).values[i, 0, 0]), expected, rtol=1e-6)


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
