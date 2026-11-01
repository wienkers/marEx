"""Pipeline-level behaviour on sub-daily, monthly and irregular time axes, and the explicit-cycle escape hatch.

marEx was written for daily data, and every within-year reduction was resolved on a hardcoded
366-slot day-of-year axis. :class:`~marEx.SeasonalCycle` states that axis explicitly, so the
same code paths serve monthly and sub-daily series. What these tests are for, in order of worth:

1. **The cycle dimension of the output is the right one and the right length.** A monthly run
   produces ``thresholds`` on a 12-long ``month`` axis, not a 366-long ``dayofyear`` axis with
   354 empty slots; a 6-hourly run on a 1464-long ``hourofyear`` axis.
2. **A repeating diurnal signal belongs to the climatology** and does not survive into the
   anomaly. ``shifting_baseline`` therefore smooths across days at a fixed time of day, which
   needs the timesteps on a regular grid: a missing timestep must not shift later samples into
   another time of day. ``detrend_harmonic`` refuses sub-daily input rather than silently leaving
   the diurnal cycle in.
3. **The durations stay physical.** ``window_days`` and ``smooth_days`` are days, so a monthly
   axis clamps them to one step and says so; a 6-hourly axis expands them.

The fixtures are derived at test time, never committed, and hold their values in memory.
"""

import functools

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError

from ..conftest import assert_no_failures
from .conftest import GRID_DIMS, MESH_COORDS, MESH_DIMS, gridded_sst

# --------------------------------------------------------------------------------------
# Fixtures, all derived
# --------------------------------------------------------------------------------------


def _build_monthly_sst():
    """The shipped gridded fixture resampled to month-start means (40 years, 481 steps), in memory."""
    return gridded_sst().resample(time="MS").mean().load().chunk({"time": 24})


@pytest.fixture(scope="module")
def monthly_sst():
    return _build_monthly_sst()


def _build_sixhourly_sst():
    """A synthetic 6-hourly field carrying a 1.5 K diurnal cycle on top of a seasonal one.

    Six years is enough for ``fixed_baseline`` and for a 5-year ``shifting_baseline`` window, and
    small enough (8768 steps x 4 x 5 cells) to stay quick.
    """
    time = pd.date_range("2000-01-01", "2005-12-31T18:00:00", freq="6h")
    lat = np.linspace(35.0, 36.0, 4).astype(np.float32)
    lon = np.linspace(-40.0, -39.0, 5).astype(np.float32)
    seasonal = 3.0 * np.sin(2 * np.pi * time.dayofyear.values.astype(np.float64) / 365.25)
    diurnal = 1.5 * np.sin(2 * np.pi * time.hour.values.astype(np.float64) / 24.0)
    noise = np.random.default_rng(42).normal(0.0, 0.5, size=(len(time), len(lat), len(lon)))
    values = (15.0 + seasonal + diurnal)[:, None, None] + noise
    da = xr.DataArray(
        values.astype(np.float32), dims=["time", "lat", "lon"], coords={"time": time, "lat": lat, "lon": lon}, name="to"
    )
    return da.chunk({"time": 400})


@pytest.fixture(scope="module")
def sixhourly_sst():
    return _build_sixhourly_sst()


def test_fixtures_carry_no_futures(client):
    """Neither builder leaves futures under a live distributed client.

    A ``.persist()`` in a builder pins the fixture to whichever client was current when it was
    built, and the tests fail with ``lost dependencies`` once that client closes.
    """
    from distributed.client import futures_of

    failures = []
    for name, build in (("monthly_sst", _build_monthly_sst), ("sixhourly_sst", _build_sixhourly_sst)):
        held = futures_of(build())
        if held:
            failures.append(f"{name} holds {len(held)} future(s); keep its values in memory or in the store")
    assert_no_failures(failures)


# --------------------------------------------------------------------------------------
# Sub-daily
# --------------------------------------------------------------------------------------


def _diurnal_field(calendar="standard", drop=()):
    """A pure 3 K diurnal sinusoid, 6-hourly over 2000-2005, on two cells."""
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


def _shifting(field, **kwargs):
    kwargs = {"window_years": 3, "dimensions": MESH_DIMS, "coordinates": MESH_COORDS, "quiet": True, **kwargs}
    return marEx.anomaly.compute(field, "shifting_baseline", **kwargs)


def test_subdaily_shifting_unsmoothed_removes_the_diurnal_cycle():
    """With no smoothing the hour-of-year climatology removes the diurnal cycle outright."""
    anomaly = _shifting(_diurnal_field(), smooth_days=1)["dat_anomaly"].compute()
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
    return tuple(cftime.datetime(*pd.Timestamp(t).timetuple()[:5], calendar=calendar) for t in GAPS[gap])


@functools.lru_cache(maxsize=None)
def _complete_anomaly(calendar):
    return _shifting(_diurnal_field(calendar))["dat_anomaly"].compute()


# Calendar x gap in full; each row also takes one (smooth_days, time_chunk) pair, so every pair
# of values across the four factors occurs. The canonical rechunk makes the time chunk irrelevant
# to the kernel; the climatology's chunk invariance is pinned on its own.
MISSING_STEP_CASES = [
    ("standard", "single", 5, 400),
    ("standard", "day", 21, -1),
    ("standard", "new_year", 5, -1),
    ("noleap", "single", 21, -1),
    ("noleap", "day", 5, 400),
    ("noleap", "new_year", 21, 400),
]


@pytest.mark.parametrize("calendar, gap, smooth_days, time_chunk", MISSING_STEP_CASES)
def test_missing_timesteps_keep_the_time_of_day(calendar, gap, smooth_days, time_chunk):
    """Every retained sample of a pure diurnal signal has a near-zero anomaly when timesteps are missing."""
    complete = _complete_anomaly(calendar)
    worst_complete = float(np.nanmax(np.abs(complete.values)))
    assert worst_complete < 1e-3, f"largest anomaly {worst_complete:.3g} K on the complete axis"

    field = _diurnal_field(calendar, _gap_times(calendar, gap)).chunk({"time": time_chunk})
    anomaly = _shifting(field, smooth_days=smooth_days)["dat_anomaly"].compute()
    kept = field.time.values[np.asarray(field.time.dt.year) >= 2003]
    assert np.array_equal(anomaly.time.values, kept)
    # NaN only where the complete axis is NaN too (a leap day with no leap year in its window).
    np.testing.assert_array_equal(anomaly.isnull().values, complete.sel(time=kept).isnull().values)
    assert int(anomaly.notnull().sum()) > 0.99 * anomaly.size
    worst = float(np.abs(anomaly).max())
    assert worst < 1e-3, f"largest anomaly {worst:.3g} K on a pure diurnal signal"


def test_off_grid_timesteps_are_rejected_with_the_rounding_that_fixes_them():
    """A timestep between grid points would put a window across two times of day, so it is rejected.

    Labels decoded from float offsets sit nanoseconds off the grid: the error names the rounding,
    and the snapped axis gives exactly the on-grid result.
    """
    field = _diurnal_field()
    time = field.time.values.copy()
    time[2000] = time[2000] + np.timedelta64(1, "h")
    with pytest.raises(ConfigurationError, match="time of day"):
        _shifting(field.assign_coords(time=time), smooth_days=5)

    exact = _shifting(field, smooth_days=5)
    time = field.time.values.copy()
    time[1::3] = time[1::3] - np.timedelta64(3, "ns")
    noisy = field.assign_coords(time=time)
    with pytest.raises(ConfigurationError, match="time of day") as caught:
        _shifting(noisy, smooth_days=5)
    suggestion = "da['time'] = da['time'].dt.round('21600s')"
    assert any(suggestion in s for s in caught.value.suggestions), caught.value.suggestions
    rerun = _shifting(noisy.assign_coords(time=noisy["time"].dt.round("21600s")), smooth_days=5)
    xr.testing.assert_identical(rerun.compute(), exact.compute())


def test_smooth_max_missing_is_validated_and_recorded():
    """Only a fraction in [0, 1) is accepted, and the accepted value is written to the attributes."""
    failures = []
    for bad in (-0.1, 1.0, 1.5, True, "0.1", None):
        try:
            _shifting(_diurnal_field(), smooth_max_missing=bad)
        except ConfigurationError as e:
            if "smooth_max_missing" not in str(e):
                failures.append(f"{bad!r}: error does not name smooth_max_missing: {e}")
        except Exception as e:  # noqa: BLE001 - a wrong exception type is a row failure
            failures.append(f"{bad!r}: raised {type(e).__name__} instead of ConfigurationError: {e}")
        else:
            failures.append(f"{bad!r}: accepted")
    ds = _shifting(_diurnal_field(), smooth_days=5, smooth_max_missing=0.25, compute_mode="lazy")
    if ds.attrs["smooth_max_missing"] != 0.25:
        failures.append(f"0.25: recorded as {ds.attrs['smooth_max_missing']!r}")
    assert_no_failures(failures)


def test_record_shorter_than_the_window_gives_nan_anomalies():
    time = pd.date_range("2000-12-31 18:00", "2001-01-01 06:00", freq="6h")
    field = xr.DataArray(
        np.ones((time.size, 2), np.float32),
        dims=("time", "ncells"),
        coords={"time": time, "lat": ("ncells", np.zeros(2, np.float32)), "lon": ("ncells", np.zeros(2, np.float32))},
    ).chunk({"time": -1})
    ds = _shifting(field, window_years=1, smooth_days=21)
    assert bool(ds["dat_anomaly"].isnull().all())


def _noisy_field(drop):
    """A diurnal signal plus noise, 6-hourly over 2000-2004: with ``drop`` set to NaN, and with ``drop`` removed."""
    time = pd.date_range("2000-01-01", "2004-12-31T18:00:00", freq="6h")
    hour = np.asarray(time.hour, dtype=np.float64)
    values = 3.0 * np.sin(2 * np.pi * hour / 24.0)[:, None] + np.random.default_rng(7).normal(size=(time.size, 3))
    field = xr.DataArray(
        values.astype(np.float32),
        dims=("time", "ncells"),
        coords={"time": time, "lat": ("ncells", np.zeros(3, np.float32)), "lon": ("ncells", np.zeros(3, np.float32))},
    )
    present = ~np.isin(np.arange(time.size), drop)
    return field.where(present[:, None]), field.isel(time=present)


def test_an_absent_step_is_a_missing_step():
    """A timestep left out of the axis and one present as NaN give the same anomaly everywhere else.

    Rows cover complete windows only (``smooth_max_missing=0``) and the interior gap mask.
    """
    with_nan, without = _noisy_field(np.r_[100, 2001:2005, 3333, 4100:4103, 5000])
    failures = []
    for smooth_days, max_missing in ((21, 0.1), (5, 0.25), (21, 0.0)):
        kwargs = {"window_years": 2, "smooth_days": smooth_days, "smooth_max_missing": max_missing, "validate": False}
        a = _shifting(with_nan.chunk({"time": 300}), **kwargs)["dat_anomaly"].compute()
        b = _shifting(without.chunk({"time": 300}), **kwargs)["dat_anomaly"].compute()
        case = f"smooth_days={smooth_days} max_missing={max_missing}"
        if not np.array_equal(a.sel(time=b.time).values, b.values, equal_nan=True):
            failures.append(f"{case}: the NaN step and the absent step give different anomalies")
        if not int(b.notnull().sum()) > 0.9 * b.size:
            failures.append(f"{case}: only {int(b.notnull().sum())} of {b.size} values are finite")
    assert_no_failures(failures)


def test_gappy_axis_is_the_same_in_every_compute_mode(tmp_path):
    _, field = _noisy_field(np.r_[100, 2001:2005, 3333])
    kwargs = {"window_years": 2, "smooth_days": 21}
    ref = _shifting(field.chunk({"time": 300}), **kwargs)["dat_anomaly"].values
    failures = []
    for compute_mode in ("lazy", "streaming"):
        got = _shifting(field.chunk({"time": 300}), compute_mode=compute_mode, scratch_dir=str(tmp_path / compute_mode), **kwargs)
        try:
            if not np.array_equal(got["dat_anomaly"].values, ref, equal_nan=True):
                failures.append(f"{compute_mode}: anomaly differs from persist")
        finally:
            marEx.clear_staging(got)
    assert_no_failures(failures)


def test_subdaily_fixed_baseline_resolves_and_names_hourofyear(sixhourly_sst):
    """A 6-hourly fixed baseline with seasonal thresholds resolves on a 1464-long ``hourofyear`` axis and says so."""
    ds = marEx.preprocess_data(
        sixhourly_sst,
        method_anomaly="fixed_baseline",
        method_extreme="seasonal_percentile",
        dimensions=GRID_DIMS,
        dask_chunks={"time": 400},
    )
    assert "hourofyear" in ds.thresholds.dims
    assert ds.sizes["hourofyear"] == 366 * 4
    assert ds.attrs["preprocessing_steps"][-1].startswith("Hour-of-year thresholds"), ds.attrs["preprocessing_steps"]


def test_cycle_resolved_climatologies_remove_the_diurnal_cycle(sixhourly_sst):
    """Grouped by ``hourofyear`` the 1.5 K diurnal signal is part of the climatology, so it is removed.

    This is also what makes rejecting ``detrend_harmonic`` the right call rather than a
    limitation; ``detrend_fixed_baseline`` runs the harmonic module with harmonics off and then
    subtracts a cycle-resolved climatology, so it is allowed.
    """
    failures = []
    for method in ("fixed_baseline", "detrend_fixed_baseline"):
        anomaly = marEx.anomaly.compute(
            sixhourly_sst, method=method, dimensions=GRID_DIMS, dask_chunks={"time": 400}
        ).dat_anomaly.compute()
        by_hour = anomaly.groupby(anomaly.time.dt.hour).mean()
        residual = float(by_hour.max() - by_hour.min())
        if not residual < 0.2:
            failures.append(f"{method}: diurnal cycle survived into the anomaly: {residual:.3f} K of 1.5 K")
    assert_no_failures(failures)


def test_detrend_harmonic_refuses_subdaily_input_and_names_a_way_forward(sixhourly_sst):
    """Rejected, not computed: the annual/semi-annual basis cannot represent a diurnal cycle."""
    with pytest.raises(ConfigurationError, match="does not support sub-daily data") as exc:
        marEx.anomaly.compute(sixhourly_sst, method="detrend_harmonic", dimensions=GRID_DIMS, dask_chunks={"time": 400})
    text = " ".join(exc.value.suggestions)
    assert "shifting_baseline" in text or "fixed_baseline" in text, exc.value.suggestions


def test_subdaily_shifting_tile_is_budgeted_on_the_cycle_and_binds(sixhourly_sst, monkeypatch):
    """The climatology's tile budget is sized for the 1464-slot cycle, and a small budget really tiles the field.

    ``rolling_climatology`` budgets ``output_elements_per_cell = n_target_years * cycle.length``;
    on daily-only code that was ``n_years * 366`` from a hardcoded length, a quarter of the
    output a 6-hourly axis produces. The call is intercepted, so this is a tripwire rather than a
    restatement. Exact equality, not divisibility: ``n_years * 366`` is a multiple of 1464
    whenever ``n_years`` is a multiple of 4.
    """
    import marEx.anomaly.climatology as C
    import marEx.core.dimensions as D

    calls = []
    real = C.tile_spatial_chunks

    def spy(*args, **kwargs):
        tile = real(*args, **kwargs)
        calls.append((kwargs, tile))
        return tile

    monkeypatch.setattr(C, "tile_spatial_chunks", spy)
    ds = marEx.preprocess_data(
        sixhourly_sst,
        method_anomaly="shifting_baseline",
        method_extreme="seasonal_percentile",
        window_years=3,
        smooth_days=5,
        window_days=11,
        dimensions=GRID_DIMS,
        dask_chunks={"time": 400},
    )
    assert ds.sizes["hourofyear"] == 366 * 4
    # The first window_years years are trimmed, as on any other cadence.
    assert ds.sizes["time"] < sixhourly_sst.sizes["time"]
    assert bool(np.isfinite(ds.dat_anomaly).any().compute())
    assert calls, "rolling_climatology did not budget its spatial tiling at all"
    per_cell = calls[0][0]["output_elements_per_cell"]
    n_years = len(np.unique(sixhourly_sst.time.dt.year.values))
    assert per_cell == n_years * 366 * 4, (
        f"the climatology budgeted {per_cell} output elements per cell; the 6-hourly cycle needs {n_years * 366 * 4}. "
        f"{n_years * 366} means it is still sized on a hardcoded day-of-year axis."
    )

    calls.clear()
    monkeypatch.setattr(D, "TASK_ELEMENTS", 200_000)
    small = marEx.anomaly.compute(
        sixhourly_sst, method="shifting_baseline", window_years=3, smooth_days=5, dimensions=GRID_DIMS, dask_chunks={"time": 400}
    )
    assert calls, "rolling_climatology did not budget its spatial tiling at all"
    tile = calls[0][1]
    assert any(tile[dim] < sixhourly_sst.sizes[dim] for dim in tile), f"a 200k-element budget did not tile the 4 x 5 field: {tile}"
    assert bool(np.isfinite(small.dat_anomaly).any().compute()), "tiled run produced nothing finite"


def test_global_percentile_needs_no_cycle_at_all(sixhourly_sst):
    ds = marEx.preprocess_data(
        sixhourly_sst,
        method_anomaly="fixed_baseline",
        method_extreme="global_percentile",
        dimensions=GRID_DIMS,
        dask_chunks={"time": 400},
    )
    assert "hourofyear" not in ds.thresholds.dims


# --------------------------------------------------------------------------------------
# Monthly, irregular and the escape hatch
# --------------------------------------------------------------------------------------


def _fixed_seasonal(da, **kwargs):
    kwargs = {"dimensions": GRID_DIMS, "dask_chunks": {"time": 24}, **kwargs}
    return marEx.preprocess_data(da, method_anomaly="fixed_baseline", method_extreme="seasonal_percentile", **kwargs)


def test_monthly_run_resolves_on_twelve_populated_month_slots(monthly_sst):
    """The headline assertion: a monthly run resolves on ``month``, not ``dayofyear``, and says so.

    On daily-only code this produces a 366-long ``dayofyear`` axis whose 354 unpopulated slots
    are NaN, so the tripwire is the dimension NAME. 40 years reach all twelve slots, the anomaly
    keeps the full series, and forcing the monthly cycle explicitly gives the same thresholds.
    """
    ds = _fixed_seasonal(monthly_sst)
    assert "month" in ds.thresholds.dims and "dayofyear" not in ds.thresholds.dims, ds.thresholds.dims
    assert ds.sizes["month"] == 12
    np.testing.assert_array_equal(ds.month.values, np.arange(1, 13))
    thresholds = ds.thresholds.compute()
    assert bool(np.isfinite(thresholds).all()), "a month slot has no data behind it"
    assert ds.sizes["time"] == monthly_sst.sizes["time"]
    step = ds.attrs["preprocessing_steps"][-1]
    assert step.startswith("Monthly thresholds") and "Day-of-year" not in step, step

    explicit = _fixed_seasonal(monthly_sst, cycle=marEx.SeasonalCycle("month", 12, 30.44))
    assert explicit.sizes["month"] == 12
    np.testing.assert_array_equal(explicit.thresholds.values, thresholds.values)


def test_a_pure_annual_sinusoid_anomalises_to_zero():
    """With a perfectly repeating seasonal cycle and no trend, a fixed monthly climatology removes all of it."""
    time = pd.date_range("1990-01-01", periods=12 * 30, freq="MS")
    signal = 5.0 * np.sin(2 * np.pi * (time.month.values - 1) / 12.0)
    da = xr.DataArray(
        np.broadcast_to(signal[:, None, None], (len(time), 3, 3)).astype(np.float32).copy(),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": np.arange(3.0), "lon": np.arange(3.0)},
    ).chunk({"time": 24})
    ds = marEx.preprocess_data(
        da, method_anomaly="fixed_baseline", method_extreme="global_percentile", dimensions=GRID_DIMS, dask_chunks={"time": 24}
    )
    assert float(np.abs(ds.dat_anomaly).max().compute()) < 1e-4


def test_monthly_window_days_is_counted_in_steps(monthly_sst, marex_warnings):
    """An 11-day window on a monthly axis is one month and says so; an even ``window_days`` is accepted there.

    Oddness is a property of the window in STEPS: demanding an odd number of days on a monthly
    axis would reject a well-posed request.
    """
    with marex_warnings as cap:
        _fixed_seasonal(monthly_sst, window_days=11)
    assert any("cannot be represented" in m for m in cap.messages), cap.messages
    assert _fixed_seasonal(monthly_sst, window_days=10).sizes["month"] == 12


def test_an_even_window_days_is_still_rejected_on_a_daily_axis():
    """The daily rejection, whose wording several tests rely on, stays."""
    sst = gridded_sst().isel(time=slice(0, 400))
    with pytest.raises(ConfigurationError, match="window_days must be an odd number"):
        marEx.preprocess_data(
            sst.chunk({"time": 25}),
            method_anomaly="fixed_baseline",
            method_extreme="seasonal_percentile",
            window_days=10,
            dimensions=GRID_DIMS,
        )


def test_smooth_days_degenerating_to_one_step_is_logged(monthly_sst, marex_warnings):
    """``shifting_baseline`` with ``smooth_days=21`` on a monthly axis reduces to ``rolling_climatology``, and says so."""
    with marex_warnings as cap:
        marEx.anomaly.compute(
            monthly_sst, method="shifting_baseline", window_years=5, smooth_days=21, dimensions=GRID_DIMS, dask_chunks={"time": 24}
        )
    assert any("reduces to rolling_climatology" in m for m in cap.messages), cap.messages


def _mixed_cadence():
    time = pd.DatetimeIndex(
        list(pd.date_range("2000-01-01", periods=60, freq="D")) + list(pd.date_range("2001-01-01", periods=40, freq="MS"))
    )
    return xr.DataArray(
        np.random.default_rng(11).normal(15.0, 1.0, (len(time), 3, 3)).astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": np.arange(3.0), "lon": np.arange(3.0)},
    ).chunk({"time": 20})


def test_mixed_cadence_axis_fails_only_where_a_cycle_is_needed():
    """``infer_cycle`` rejects a mixed-cadence axis; that reaches only the paths that need a within-year cycle.

    ``global_percentile`` has no cycle and must still run; the seasonal path raises; an explicit
    cycle unblocks it, which is what the escape hatch is for.
    """
    kwargs = {"dimensions": GRID_DIMS, "dask_chunks": {"time": 20}}
    ds = marEx.preprocess_data(_mixed_cadence(), method_anomaly="detrend_harmonic", method_extreme="global_percentile", **kwargs)
    assert "extreme_events" in ds
    with pytest.raises(ConfigurationError, match="too irregular"):
        marEx.preprocess_data(_mixed_cadence(), method_anomaly="fixed_baseline", method_extreme="seasonal_percentile", **kwargs)
    ds = marEx.preprocess_data(
        _mixed_cadence(),
        method_anomaly="fixed_baseline",
        method_extreme="seasonal_percentile",
        cycle=marEx.SeasonalCycle("month", 12, 30.0),
        window_days=40,
        **kwargs,
    )
    assert ds.sizes["month"] == 12


def test_an_explicit_daily_cycle_matches_the_inferred_one():
    """Passing the cycle marEx would have inferred changes nothing, so the hatch is an override, not a separate path.

    The daily run also keeps the day-of-year wording the goldens were captured with.
    """
    sst = gridded_sst().isel(time=slice(0, 1200)).chunk({"time": 25})
    kwargs = {
        "method_anomaly": "fixed_baseline",
        "method_extreme": "seasonal_percentile",
        "window_days": 3,
        "dimensions": GRID_DIMS,
    }
    inferred = marEx.preprocess_data(sst, **kwargs)
    assert inferred.attrs["preprocessing_steps"][-1].startswith("Day-of-year thresholds with 3 day window"), inferred.attrs[
        "preprocessing_steps"
    ]
    explicit = marEx.preprocess_data(sst, cycle=marEx.SeasonalCycle("dayofyear", 366, 1.0), **kwargs)
    np.testing.assert_array_equal(inferred.thresholds.values, explicit.thresholds.values)


def test_the_hatch_is_honoured_by_the_standalone_entry_points_too(monthly_sst):
    """``cycle=`` is declared on ``anomaly.compute`` and ``extremes.identify`` as well as on the chainer."""
    cycle = marEx.SeasonalCycle("month", 12, 30.44)
    anom = marEx.anomaly.compute(monthly_sst, method="fixed_baseline", cycle=cycle, dimensions=GRID_DIMS, dask_chunks={"time": 24})
    ext = marEx.extremes.identify(
        anom.dat_anomaly, method="seasonal_percentile", window_days=40, cycle=cycle, dimensions=GRID_DIMS, dask_chunks={"time": 24}
    )
    assert ext.sizes["month"] == 12
    assert "dayofyear" not in ext.thresholds.dims


def test_monthly_pipeline_runs_on_cftime_calendars():
    """End-to-end monthly runs on non-standard calendars land on the 12-slot month axis."""
    failures = []
    for calendar in ("noleap", "360_day"):
        time = xr.date_range("1990-01-01", periods=12 * 20, freq="MS", calendar=calendar, use_cftime=True)
        signal = 4.0 * np.sin(2 * np.pi * (np.arange(len(time)) % 12) / 12.0)
        data = signal[:, None, None] + np.random.default_rng(3).normal(0.0, 0.5, size=(len(time), 3, 3))
        da = xr.DataArray(
            data.astype(np.float32),
            dims=["time", "lat", "lon"],
            coords={"time": time, "lat": np.arange(3.0), "lon": np.arange(3.0)},
        ).chunk({"time": 24})
        ds = _fixed_seasonal(da)
        if ds.sizes.get("month") != 12 or "dayofyear" in ds.thresholds.dims:
            failures.append(f"{calendar}: thresholds on {dict(ds.thresholds.sizes)}")
    assert_no_failures(failures)
