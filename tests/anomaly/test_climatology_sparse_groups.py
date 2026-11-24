"""Exact parity with the Cartesian flox reduction on incomplete calendars."""

import dask
import flox.xarray
import numpy as np
import pytest
import xarray as xr

from marEx.anomaly import climatology
from marEx.core.time_axis import infer_cycle


@pytest.mark.parametrize(
    "calendar,freq,years,window,blocks",
    [
        ("noleap", "D", 6, 1, 1),
        ("360_day", "D", 10, 2, 2),
        ("standard", "5D", 20, 5, 4),
        ("standard", "6h", 20, 5, 4),
        ("noleap", "D", 55, 15, 11),
    ],
)
@pytest.mark.parametrize("smoothed", [False, True])
def test_occupied_pairs_match_cartesian_flox(monkeypatch, calendar, freq, years, window, blocks, smoothed):
    end_day = 30 if calendar == "360_day" else 31
    time = xr.date_range(
        "1980-01-01", f"{1979 + years}-12-{end_day} 23:59", freq=freq, calendar=calendar, use_cftime=calendar != "standard"
    )
    # Leave a gap in the five-day and subdaily records. Keep one occupied group
    # entirely NaN and another NaN in one spatial cell only.
    if freq in ("5D", "6h"):
        keep = (np.arange(len(time)) % 97 != 15) & (time.year != 1984)
        time = time[keep]
    rng = np.random.default_rng(191)
    values = rng.standard_normal((len(time), 1, 4)).astype(np.float32)
    values *= np.float32(2.0) ** rng.integers(-12, 13, size=values.shape).astype(np.float32)
    values[time.dayofyear == 40] = np.nan
    values[time.dayofyear == 51, :, 1] = np.nan
    data = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [29.5], "lon": np.arange(4)},
        attrs={"units": "K"},
        name="temperature",
    ).chunk({"time": 128, "lon": 2})
    cycle = infer_cycle(data.time)
    fn = climatology.smoothed_rolling_climatology if smoothed else climatology.rolling_climatology
    options = {"smooth_days": 21} if smoothed else {}
    sparse_calls = []
    real_reduce = flox.xarray.xarray_reduce

    def spy(obj, *by, **kwargs):
        result = real_reduce(obj, *by, **kwargs)
        codes = by[0].values
        expected = kwargs["expected_groups"][0]
        np.testing.assert_array_equal(np.unique(codes), expected)
        assert not any("setitem" in layer for layer in result.data.dask.layers)
        sparse_calls.append((obj.chunksizes[kwargs["dim"]], kwargs["method"]))
        return result

    with dask.config.set(scheduler="synchronous"), monkeypatch.context() as patch:
        patch.setattr(flox.xarray, "xarray_reduce", spy)
        actual = fn(data, window_years=window, cycle=cycle, **options)
        assert dict(actual.chunksizes) == dict(data.chunksizes)
        actual = actual.compute()

    dense_calls = []

    def cartesian(obj, dim, keys, occupied, method):
        n_cycle = len(cycle.labels)
        years, slots = keys // n_cycle, cycle.labels[keys % n_cycle]
        labels = cycle.labels if method else cycle.labels[-cycle.steps_per_day :]
        reduced = real_reduce(
            obj,
            xr.DataArray(years, dims=dim, name="target_year"),
            xr.DataArray(slots, dims=dim, name=cycle.index_name),
            dim=dim,
            func="nanmean",
            expected_groups=(np.unique(time.year), labels),
            isbin=(False, False),
            method=method,
            dtype=np.float32,
            fill_value=np.nan,
        )
        dense_calls.append((obj.chunksizes[dim], method))
        # Compare arithmetic on the same occupied keys; absence and the actual
        # day-366 lookup are checked separately below.
        axis = "group"
        return (
            reduced.sel(
                target_year=xr.DataArray(occupied // n_cycle, dims=axis),
                **{cycle.index_name: xr.DataArray(cycle.labels[occupied % n_cycle], dims=axis)},
            )
            .drop_vars(["target_year", cycle.index_name])
            .rename({axis: "target_year"})
            .assign_coords(target_year=np.arange(occupied.size))
        )

    with dask.config.set(scheduler="synchronous"), monkeypatch.context() as patch:
        patch.setattr(climatology, "_reduce_occupied", cartesian)
        expected = fn(data, window_years=window, cycle=cycle, **options).compute()
    assert sparse_calls == dense_calls
    assert len(sparse_calls[0][0]) == blocks
    assert sparse_calls[0][1] == "map-reduce"
    if freq == "6h":
        assert len(sparse_calls) == 2
        assert sparse_calls[1][1] is None
    xr.testing.assert_identical(actual, expected)
    finite = np.isfinite(expected.values)
    assert finite.any() and not finite.all()
    np.testing.assert_array_equal(actual.values[finite].view(np.uint32), expected.values[finite].view(np.uint32))
    warmup = time.year < time.year.min() + window
    assert actual.isel(time=warmup).isnull().all()
    if not smoothed and freq != "5D":
        assert actual.isel(time=time.dayofyear == 40).isnull().all()


def test_no_eligible_history_needs_no_reduction(monkeypatch):
    time = xr.date_range("2001", periods=10, freq="D")
    data = xr.DataArray(
        np.ones((10, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0], "lon": [0]},
    ).chunk({"time": 3})

    def unexpected(*args, **kwargs):
        pytest.fail("An empty climatology reached flox")

    monkeypatch.setattr(flox.xarray, "xarray_reduce", unexpected)
    result = climatology.rolling_climatology(data, window_years=3)
    assert result.dtype == np.float32
    assert dict(result.chunksizes) == dict(data.chunksizes)
    assert result.isnull().all().compute()


def test_explicit_365_slot_cycle_does_not_alias_leap_day_and_partial_december_splice():
    from marEx.core.time_axis import SeasonalCycle

    leap = xr.DataArray(
        np.array([1000.0, 2.0, 4.0], dtype="f4")[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": np.array(["2000-12-31", "2001-01-01", "2002-01-01"], dtype="datetime64[D]"), "lat": [0], "lon": [0]},
    ).chunk({"time": 2})
    clim_365 = climatology.rolling_climatology(leap, window_years=1, cycle=SeasonalCycle("dayofyear", 365, 1)).compute()
    np.testing.assert_array_equal(clim_365.values.ravel(), np.array([np.nan, np.nan, 2.0], dtype="f4"))

    dates = np.array(
        [
            "1999-12-31T00",
            "1999-12-31T12",
            "2000-12-31T00",
            "2000-12-31T06",
            "2000-12-31T12",
            "2001-01-01T00",
            "2004-12-31T00",
            "2004-12-31T06",
        ],
        dtype="datetime64[h]",
    )
    subdaily = xr.DataArray(
        np.array([2, 10, 1, 3, 5, 11, 20, 21], dtype="f4")[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": dates, "lat": [0], "lon": [0]},
    ).chunk({"time": 2})
    clim_sub = climatology.rolling_climatology(subdaily, window_years=1, cycle=SeasonalCycle("hourofyear", 1464, 0.25)).compute()
    np.testing.assert_array_equal(
        clim_sub.values.ravel(),
        np.array([np.nan, np.nan, 2.0, np.nan, 10.0, np.nan, np.nan, np.nan], dtype="f4"),
    )


def test_custom_cycle_labels_borrowed_december_and_intermediate_task_budget():
    from dask.callbacks import Callback

    from marEx.core import dimensions as dims_mod
    from marEx.core.time_axis import SeasonalCycle

    class SubsetCycle(SeasonalCycle):
        def __init__(self, index_name, length, step_days, custom_labels):
            super().__init__(index_name, length, step_days)
            self._custom = np.asarray(custom_labels, dtype=np.int32)

        @property
        def labels(self):
            return self._custom

    # Reordered monthly labels [3, 1] and borrowed-only December [1, 366]
    m_dates = np.array(["1999-01-01", "1999-03-01", "2000-01-01", "2000-03-01"], dtype="datetime64[D]")
    m_da = xr.DataArray(
        np.array([1, 3, 1, 3], dtype="f4")[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": m_dates, "lat": [0], "lon": [0]},
    ).chunk({"time": 2})
    got_m = climatology.rolling_climatology(m_da, window_years=1, cycle=SubsetCycle("month", 12, 30, [3, 1])).compute()
    np.testing.assert_array_equal(got_m.values.ravel(), np.array([np.nan, np.nan, 1.0, 3.0], dtype="f4"))

    b_da = xr.DataArray(
        np.array([2.0, 7.0], dtype="f4")[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": np.array(["1999-12-31", "2000-12-31"], dtype="datetime64[D]"), "lat": [0], "lon": [0]},
    ).chunk({"time": 1})
    got_b = climatology.rolling_climatology(b_da, window_years=1, cycle=SubsetCycle("dayofyear", 366, 1, [1, 366])).compute()
    np.testing.assert_array_equal(got_b.values.ravel(), np.array([np.nan, 2.0], dtype="f4"))

    # Six-hourly subset [1, 1463] across a leap-year December splice
    h_dates = np.array(["2000-01-01T00", "2000-12-31T12", "2004-01-01T00", "2004-12-31T12"], dtype="datetime64[h]")
    h_da = xr.DataArray(
        np.array([5, 8, 55, 88], dtype="f4")[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": h_dates, "lat": [0], "lon": [0]},
    ).chunk({"time": 2})
    got_h = climatology.rolling_climatology(h_da, window_years=4, cycle=SubsetCycle("hourofyear", 1464, 0.25, [1, 1463])).compute()
    np.testing.assert_array_equal(got_h.values.ravel(), np.array([np.nan, np.nan, 5.0, 8.0], dtype="f4"))

    # Intermediate rechunk task budget on sparse history
    import dask.array as dsa

    dates_sp = np.r_[np.datetime64("2000-01-01"), np.arange("2001-01-01", "2001-05-01", dtype="datetime64[D]")]
    src = dsa.ones((len(dates_sp), 10, 10), chunks=(5, 10, 10), dtype="f4")
    field = xr.DataArray(src, dims=("time", "lat", "lon"), coords={"time": dates_sp, "lat": np.arange(10), "lon": np.arange(10)})
    task_sizes = []
    orig_budget = dims_mod.TASK_ELEMENTS
    try:
        dims_mod.TASK_ELEMENTS = 1200
        res = climatology.rolling_climatology(field, cycle=SeasonalCycle("dayofyear", 366, 1), window_years=1)
        with (
            dask.config.set(scheduler="synchronous"),
            Callback(posttask=lambda k, v, *_: task_sizes.append(v.size) if isinstance(v, np.ndarray) else None),
        ):
            res.data.blocks[0, 0, 0].compute()
    finally:
        dims_mod.TASK_ELEMENTS = orig_budget
    assert max(task_sizes) <= 1200
