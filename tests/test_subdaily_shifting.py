"""``shifting_baseline`` on a sub-daily axis removes the diurnal cycle.

The climatology is resolved by ``hourofyear``, so a repeating diurnal signal belongs to it and
must not survive into the anomaly. Smoothing consecutive time steps over ``smooth_days`` first
averages that signal away, so the climatology cannot subtract it. The smoothing therefore runs
across days at a fixed time of day, which needs the timesteps on a regular grid: a missing
timestep must not shift later samples into another time of day.
"""

import functools

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx

DIMENSIONS = {"time": "time", "x": "ncells"}
COORDINATES = {"time": "time", "x": "lon", "y": "lat"}


def _diurnal_field(calendar="standard", drop=()):
    if calendar == "standard":
        time = pd.date_range("2000-01-01", "2005-12-31T18:00:00", freq="6h")
    else:
        time = xr.cftime_range("2000-01-01", "2005-12-31T18:00:00", freq="6h", calendar=calendar)
    time = time[~np.isin(np.arange(time.size), [time.get_loc(t) for t in drop])]
    hour = np.asarray(time.hour, dtype=np.float64)
    values = np.repeat((3.0 * np.sin(2 * np.pi * hour / 24.0))[:, None], 2, axis=1)
    return xr.DataArray(
        values.astype(np.float32),
        dims=("time", "ncells"),
        coords={"time": time, "lat": ("ncells", np.zeros(2, np.float32)), "lon": ("ncells", np.zeros(2, np.float32))},
        name="sst",
    ).chunk({"time": 400})


@pytest.mark.parametrize("smooth_days", [1, 5, 21])
def test_diurnal_cycle_is_removed(smooth_days):
    ds = marEx.anomaly.compute(
        _diurnal_field(),
        "shifting_baseline",
        window_years=3,
        smooth_days=smooth_days,
        dimensions=DIMENSIONS,
        coordinates=COORDINATES,
        quiet=True,
    )
    anomaly = ds["dat_anomaly"].compute()
    anomaly = anomaly.where(anomaly.notnull().all("ncells"), drop=True)
    assert anomaly.sizes["time"] > 0
    by_hour = anomaly.groupby(anomaly.time.dt.hour).mean()
    peak_to_peak = float(by_hour.max() - by_hour.min())
    assert peak_to_peak < 0.1, f"diurnal cycle survived: {peak_to_peak:.3f} of a 6.0 peak-to-peak signal"


# A single timestep, a whole day, and the steps either side of a year boundary.
GAPS = {
    "single": ("2002-07-01 12:00",),
    "day": ("2003-03-15 00:00", "2003-03-15 06:00", "2003-03-15 12:00", "2003-03-15 18:00"),
    "new_year": ("2001-12-31 18:00", "2002-01-01 00:00"),
}


def _gap_times(calendar, gap):
    if calendar == "standard":
        return tuple(pd.Timestamp(t) for t in GAPS[gap])
    import cftime

    return tuple(cftime.datetime(*pd.Timestamp(t).timetuple()[:5], calendar=calendar) for t in GAPS[gap])


@functools.lru_cache(maxsize=None)
def _complete_anomaly(calendar):
    ds = marEx.anomaly.compute(
        _diurnal_field(calendar), "shifting_baseline", window_years=3, dimensions=DIMENSIONS, coordinates=COORDINATES, quiet=True
    )
    return ds["dat_anomaly"].compute()


@pytest.mark.parametrize("time_chunk", [400, -1])
@pytest.mark.parametrize("smooth_days", [5, 21])
@pytest.mark.parametrize("gap", sorted(GAPS))
@pytest.mark.parametrize("calendar", ["standard", "noleap"])
def test_missing_timesteps_keep_the_time_of_day(calendar, gap, smooth_days, time_chunk):
    """Every retained sample of a pure diurnal signal has a near-zero anomaly when timesteps are missing."""
    field = _diurnal_field(calendar, _gap_times(calendar, gap)).chunk({"time": time_chunk})
    ds = marEx.anomaly.compute(
        field,
        "shifting_baseline",
        window_years=3,
        smooth_days=smooth_days,
        dimensions=DIMENSIONS,
        coordinates=COORDINATES,
        quiet=True,
    )
    anomaly = ds["dat_anomaly"].compute()
    kept = field.time.values[np.asarray(field.time.dt.year) >= 2003]
    assert np.array_equal(anomaly.time.values, kept)
    # NaN only where the complete axis is NaN too (a leap day with no leap year in its window).
    complete = _complete_anomaly(calendar).sel(time=kept)
    np.testing.assert_array_equal(anomaly.isnull().values, complete.isnull().values)
    assert int(anomaly.notnull().sum()) > 0.99 * anomaly.size
    worst = float(np.abs(anomaly).max())
    assert worst < 1e-3, f"largest anomaly {worst:.3g} K on a pure diurnal signal"


def test_off_grid_timestep_is_rejected():
    """A timestep between grid points would put a window across two times of day."""
    field = _diurnal_field()
    time = field.time.values.copy()
    time[2000] = time[2000] + np.timedelta64(1, "h")
    field = field.assign_coords(time=time)
    with pytest.raises(marEx.exceptions.ConfigurationError, match="time of day"):
        marEx.anomaly.compute(
            field, "shifting_baseline", window_years=3, smooth_days=5, dimensions=DIMENSIONS, coordinates=COORDINATES, quiet=True
        )


def test_slightly_off_grid_axis_names_the_rounding_that_fixes_it():
    """Labels decoded from float offsets sit nanoseconds off the grid; the error says how to snap them."""
    field = _diurnal_field()
    exact = marEx.anomaly.compute(
        field, "shifting_baseline", window_years=3, smooth_days=5, dimensions=DIMENSIONS, coordinates=COORDINATES, quiet=True
    )
    time = field.time.values.copy()
    time[1::3] = time[1::3] - np.timedelta64(3, "ns")
    noisy = field.assign_coords(time=time)
    with pytest.raises(marEx.exceptions.ConfigurationError, match="time of day") as caught:
        marEx.anomaly.compute(
            noisy, "shifting_baseline", window_years=3, smooth_days=5, dimensions=DIMENSIONS, coordinates=COORDINATES, quiet=True
        )
    suggestion = "da['time'] = da['time'].dt.round('21600s')"
    assert any(suggestion in s for s in caught.value.suggestions), caught.value.suggestions
    snapped = noisy.assign_coords(time=noisy["time"].dt.round("21600s"))
    rerun = marEx.anomaly.compute(
        snapped, "shifting_baseline", window_years=3, smooth_days=5, dimensions=DIMENSIONS, coordinates=COORDINATES, quiet=True
    )
    xr.testing.assert_identical(rerun.compute(), exact.compute())


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, True, "0.1", None])
def test_smooth_max_missing_is_validated(bad):
    with pytest.raises(marEx.exceptions.ConfigurationError, match="smooth_max_missing"):
        marEx.anomaly.compute(
            _diurnal_field(),
            "shifting_baseline",
            window_years=3,
            smooth_max_missing=bad,
            dimensions=DIMENSIONS,
            coordinates=COORDINATES,
            quiet=True,
        )


def test_smooth_max_missing_is_recorded():
    ds = marEx.anomaly.compute(
        _diurnal_field(),
        "shifting_baseline",
        window_years=3,
        smooth_days=5,
        smooth_max_missing=0.25,
        compute_mode="lazy",
        dimensions=DIMENSIONS,
        coordinates=COORDINATES,
        quiet=True,
    )
    assert ds.attrs["smooth_max_missing"] == 0.25


@pytest.mark.parametrize("max_missing", [0.0, 0.1, 0.25])
@pytest.mark.parametrize("start_hour", [0, 18])
@pytest.mark.parametrize("partial_steps", [0, 3])
@pytest.mark.parametrize("days", [1, 2, 4, 5, 21])
@pytest.mark.parametrize("stride", [4, 8, 24])
def test_strided_window_is_the_daily_window_per_time_of_day(stride, days, partial_steps, start_hour, max_missing):
    """Each time of day is smoothed as a daily series would be, even windows and ragged ends included."""
    from marEx.anomaly.climatology import _rolling_mean_keep_tiles

    nt = 60 * stride + partial_steps
    rng = np.random.default_rng(stride * 1000 + days * 10 + partial_steps)
    values = (rng.normal(size=(nt, 6)) * 3 + 290).astype(np.float32)
    values[rng.random(values.shape) < 0.04] = np.nan
    time = pd.date_range("2001-01-01", periods=nt, freq=f"{24 // stride}h") + pd.Timedelta(hours=start_hour)
    da = xr.DataArray(values, dims=("time", "ncells"), coords={"time": time}).chunk({"time": -1, "ncells": 4})
    got = _rolling_mean_keep_tiles(da, "time", days, stride, max_missing).compute()
    want = np.full(values.shape, np.nan, dtype=np.float32)
    for phase in range(stride):
        daily = da.isel(time=slice(phase, None, stride))
        want[phase::stride] = _rolling_mean_keep_tiles(daily, "time", days, max_missing=max_missing).values
    # Same samples and masks; numpy's pairwise sum groups a strided window of 8 or more samples
    # differently, which moves the mean by at most a couple of float32 ULP.
    np.testing.assert_array_equal(np.isnan(got.values), np.isnan(want))
    np.testing.assert_allclose(got.values, want, rtol=1e-6)
    assert got.dims == da.dims


def test_new_parameter_does_not_move_positional_arguments():
    """``smooth_max_missing`` is appended, so an existing positional call binds as before."""
    import inspect

    params = list(inspect.signature(marEx.preprocess_data).parameters)
    assert params[-1] == "smooth_max_missing"
    assert params[params.index("smooth_days") + 1] == "window_days"


def test_long_nanosecond_axis_is_on_its_grid():
    """Offsets are taken in microseconds, so a regular axis longer than 292 years is not rejected."""
    from marEx.anomaly.climatology import _subdaily_grid_positions

    time = pd.date_range("1800-01-01", "2100-01-01", freq="6h").values.astype("datetime64[ns]")
    assert _subdaily_grid_positions(xr.DataArray(time, dims="time"), 4) is None
    gappy = np.delete(time, [5, 400_000])
    positions = _subdaily_grid_positions(xr.DataArray(gappy, dims="time"), 4)
    assert positions[-1] == time.size - 1
    np.testing.assert_array_equal(positions, np.delete(np.arange(time.size), [5, 400_000]))


@pytest.mark.parametrize("nt", [3, 39, 40, 41])
def test_window_longer_than_the_record_is_nan(nt):
    """A strided window reaching past the record end everywhere leaves every value NaN."""
    from marEx.anomaly.climatology import _rolling_mean_keep_tiles

    da = xr.DataArray(np.ones((nt, 3), np.float32), dims=("time", "ncells")).chunk({"time": -1, "ncells": 2})
    got = _rolling_mean_keep_tiles(da, "time", 21, 4, 0.1).compute()
    assert got.shape == da.shape
    assert bool(got.isnull().all())


def test_record_shorter_than_the_window_gives_nan_anomalies():
    time = pd.date_range("2000-12-31 18:00", "2001-01-01 06:00", freq="6h")
    field = xr.DataArray(
        np.ones((time.size, 2), np.float32),
        dims=("time", "ncells"),
        coords={"time": time, "lat": ("ncells", np.zeros(2, np.float32)), "lon": ("ncells", np.zeros(2, np.float32))},
    ).chunk({"time": -1})
    ds = marEx.anomaly.compute(
        field, "shifting_baseline", window_years=1, smooth_days=21, dimensions=DIMENSIONS, coordinates=COORDINATES, quiet=True
    )
    assert bool(ds["dat_anomaly"].isnull().all())


def _noisy_field(drop):
    time = pd.date_range("2000-01-01", "2004-12-31T18:00:00", freq="6h")
    rng = np.random.default_rng(7)
    hour = np.asarray(time.hour, dtype=np.float64)
    values = 3.0 * np.sin(2 * np.pi * hour / 24.0)[:, None] + rng.normal(size=(time.size, 3))
    field = xr.DataArray(
        values.astype(np.float32),
        dims=("time", "ncells"),
        coords={"time": time, "lat": ("ncells", np.zeros(3, np.float32)), "lon": ("ncells", np.zeros(3, np.float32))},
    )
    return field.where(~np.isin(np.arange(time.size), drop)[:, None]), field.isel(time=~np.isin(np.arange(time.size), drop))


@pytest.mark.parametrize("smooth_days, max_missing", [(21, 0.1), (5, 0.25), (21, 0.0)])
def test_an_absent_step_is_a_missing_step(smooth_days, max_missing):
    """A timestep left out of the axis and one present as NaN give the same anomaly everywhere else."""
    drop = np.r_[100, 2001:2005, 3333, 4100:4103, 5000]
    with_nan, without = _noisy_field(drop)
    kwargs = {
        "window_years": 2,
        "smooth_days": smooth_days,
        "smooth_max_missing": max_missing,
        "validate": False,
        "dimensions": DIMENSIONS,
        "coordinates": COORDINATES,
        "quiet": True,
    }
    a = marEx.anomaly.compute(with_nan.chunk({"time": 300}), "shifting_baseline", **kwargs)["dat_anomaly"].compute()
    b = marEx.anomaly.compute(without.chunk({"time": 300}), "shifting_baseline", **kwargs)["dat_anomaly"].compute()
    kept = a.sel(time=b.time)
    np.testing.assert_array_equal(kept.values, b.values)
    assert int(b.notnull().sum()) > 0.9 * b.size


@pytest.mark.parametrize("compute_mode", ["lazy", "streaming"])
def test_gappy_axis_is_the_same_in_every_compute_mode(compute_mode, tmp_path):
    _, field = _noisy_field(np.r_[100, 2001:2005, 3333])
    kwargs = {"window_years": 2, "smooth_days": 21, "dimensions": DIMENSIONS, "coordinates": COORDINATES, "quiet": True}
    ref = marEx.anomaly.compute(field.chunk({"time": 300}), "shifting_baseline", **kwargs)
    got = marEx.anomaly.compute(
        field.chunk({"time": 300}), "shifting_baseline", compute_mode=compute_mode, scratch_dir=str(tmp_path), **kwargs
    )
    np.testing.assert_array_equal(got["dat_anomaly"].values, ref["dat_anomaly"].values)
    marEx.clear_staging(got)
