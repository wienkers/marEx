"""``window_days`` is measured in cycle slots, not in data steps.

The seasonal window is applied along the cycle axis: the exact path pools the slots within
``window // 2`` of each slot, the histogram path sums that many cycle rows. Data coarser than
its cycle (5-day data still sits on the 366-slot ``dayofyear`` cycle, one slot per day) used to
have ``window_days`` converted with the data cadence, so an 11-day window pooled 3 slots, i.e. 3
calendar days. The window is now converted with the slot width, and the warning on too few
samples counts the samples a window actually reaches. Daily and evenly sub-daily axes, where a
slot is one step, are unchanged.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.core.time_axis import DAILY_CYCLE, SeasonalCycle, resolve_cycle
from marEx.extremes import histogram as H

DIMS = {"time": "time", "x": "ncells"}
COORDS = {"time": "time", "x": "lon", "y": "lat"}


def _mesh(values, time):
    n_cells = values.shape[1]
    cells = np.arange(n_cells, dtype=np.float32)
    return xr.DataArray(
        values,
        dims=("time", "ncells"),
        coords={"time": time, "lon": ("ncells", cells), "lat": ("ncells", cells)},
        name="dat_anomaly",
    ).chunk({"time": -1})


def _five_day(n_years=40, n_cells=2):
    """A 5-day series whose value is its day of year / 100, so a threshold names a calendar day."""
    time = pd.date_range("1980-01-01", periods=int(n_years * 365.25 / 5), freq="5D")
    doy = (time.dayofyear.values / 100.0).astype(np.float32)
    return _mesh(np.repeat(doy[:, None], n_cells, axis=1), time)


def _calendar_reference(da, slot, window_days, percentile):
    """The percentile over every sample within ``window_days // 2`` calendar days of ``slot``, wrapping the year."""
    doy = da["time"].dt.dayofyear.values
    half = window_days // 2
    targets = {((slot - 1 + k) % 366) + 1 for k in range(-half, half + 1)}
    members = np.isin(doy, list(targets))
    return float(np.percentile(da.isel(ncells=0).values[members], percentile))


@pytest.mark.parametrize("window_days", [11, 31])
@pytest.mark.parametrize("method_percentile", ["exact", "approximate"])
def test_five_day_window_pools_its_calendar_days(window_days, method_percentile):
    da = _five_day()
    thresholds = marEx.extremes.identify(
        da,
        threshold_percentile=99,
        window_days=window_days,
        method_percentile=method_percentile,
        dimensions=DIMS,
        coordinates=COORDS,
    ).thresholds.isel(ncells=0)
    for slot in (100, 180):
        reference = _calendar_reference(da, slot, window_days, 99)
        got = float(thresholds.sel(dayofyear=slot))
        # The exact path pools the same samples as the reference. One slot is 0.01 here, and
        # HEAD's 3-slot window at 11 days landed 4 slots low at slot 180.
        tolerance = 1e-6 if method_percentile == "exact" else 0.01
        assert abs(got - reference) <= tolerance, f"slot {slot}: {got * 100:.4f} vs calendar window {reference * 100:.4f}"


def test_the_window_is_counted_in_slots():
    five_day = resolve_cycle(_five_day(n_years=2), "time")
    assert five_day.is_daily and five_day.step_days == 5.0
    assert five_day.window_slots(11) == 11
    assert five_day.window_slots(31) == 31


@pytest.mark.parametrize(
    "cycle",
    [DAILY_CYCLE, SeasonalCycle("hourofyear", 366 * 24, 1.0 / 24), SeasonalCycle("hourofyear", 366 * 4, 0.25)],
    ids=["daily", "hourly", "6-hourly"],
)
def test_a_slot_per_step_axis_keeps_its_window(cycle):
    # Where one slot is one step the slot count is the old step count, so those thresholds cannot move.
    for days in range(1, 62, 2):
        assert cycle.window_slots(days) == cycle.window_steps(days)


def test_a_monthly_window_uses_the_calendars_month():
    # 360-day calendar: every month is 30 days, so 106 days is 3.5 months -> 4 -> odd 5 slots.
    time = xr.date_range("2001-01-01", periods=12 * 20, freq="MS", calendar="360_day", use_cftime=True)
    da = _mesh(np.repeat(np.arange(time.size, dtype=np.float32)[:, None] % 12 + 1, 2, axis=1), time)
    cycle = resolve_cycle(da, "time")
    assert cycle.index_name == "month" and cycle.step_days == 30.0
    assert cycle.window_slots(106, calendar=da["time"].dt.calendar) == 5


def test_a_quarterly_window_uses_the_calendars_month():
    # Quarterly data on a 360-day calendar still has 30-day months: 165 days is 5.5 months -> 6 ->
    # odd 7, so April pools January to July. The data's 90-day cadence must not set the month.
    time = xr.date_range("2001-01-01", periods=80, freq="3MS", calendar="360_day", use_cftime=True)
    month = np.array([t.month for t in time.values])
    values = (month + np.repeat(np.linspace(-0.2, 0.2, 20), 4)).astype(np.float32)
    da = _mesh(np.repeat(values[:, None], 2, axis=1), time)
    thresholds = marEx.extremes.identify(
        da, threshold_percentile=90, window_days=165, method_percentile="exact", dimensions=DIMS, coordinates=COORDS
    ).thresholds
    april = float(thresholds.sel(month=4).isel(ncells=0))
    assert april == pytest.approx(float(np.percentile(values[month <= 7], 90)), abs=1e-6)
    assert resolve_cycle(da, "time").window_slots(165, calendar="360_day") == 7


def test_a_short_record_uses_the_calendars_month():
    # Two monthly samples, February to March: the median spacing is 28 days, a Gregorian month is not.
    time = pd.date_range("2001-02-01", periods=2, freq="MS")
    cycle = resolve_cycle(_mesh(np.zeros((2, 2), dtype=np.float32), time), "time")
    assert cycle.index_name == "month" and cycle.step_days == 28.0
    assert cycle.window_slots(106) == 3


@pytest.mark.parametrize(
    "freq, periods, days, slots",
    [("MS", 120, 46, 3), ("5h", 2000, 11, 55), ("7h", 2000, 11, 33)],
    ids=["monthly-gregorian", "5-hourly", "7-hourly"],
)
def test_inferred_axes_count_their_slots(freq, periods, days, slots):
    # A Gregorian month is 30.4 days, not the 31-day median spacing; a 5-hourly axis has 5 slots per
    # day (0.2 days each), not 4.8 steps. Counting the cadence gave 1, 53 and 39.
    time = pd.date_range("2001-01-01", periods=periods, freq=freq)
    cycle = resolve_cycle(_mesh(np.zeros((periods, 2), dtype=np.float32), time), "time")
    assert cycle.window_slots(days) == slots


def _uneven_counts(n_slots, n_bins, seed):
    """Per-slot counts whose slots differ in size and in where they sit, so pooling a slot twice moves the quantile."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_slots, n_bins), dtype=np.int64)
    for slot in range(n_slots):
        centre = rng.integers(n_bins // 4, 3 * n_bins // 4)
        samples = np.clip(rng.normal(centre, n_bins / 10, size=rng.integers(20, 200)).round(), 0, n_bins - 1)
        counts[slot] = np.bincount(samples.astype(np.int64), minlength=n_bins)
    return counts


@pytest.mark.parametrize("n_slots, window", [(12, 13), (12, 23), (12, 25), (366, 401)])
@pytest.mark.parametrize("q", [0.9, 0.5])
def test_the_estimator_pools_each_slot_once_past_the_cycle(n_slots, window, q):
    counts = _uneven_counts(n_slots, 200, seed=window)
    centres = np.linspace(-5, 5, 200, dtype=np.float32)
    got = H._rolling_histogram_quantile(counts, window, q, centres)
    whole_cycle = H._rolling_histogram_quantile(counts.sum(axis=0, keepdims=True), 1, q, centres)
    assert got.shape == (n_slots,)
    np.testing.assert_array_equal(got, np.full(n_slots, whole_cycle[0]))


@pytest.mark.parametrize("window", [13, 23, 25])
@pytest.mark.parametrize("mirror", [False, True])
def test_the_slab_kernel_pools_each_slot_once_past_the_cycle(window, mirror):
    # The compiled engine against the estimator on the same counts: bit for bit, as at every in-range window.
    rng = np.random.default_rng(window)
    n_slots, n_bins, ntime, n_cells = 12, 200, 12 * 30, 3
    slots = np.tile(np.arange(1, n_slots + 1, dtype=np.int64), ntime // n_slots)
    block = np.clip(rng.normal(80 + 5 * slots[:, None], 15, size=(ntime, n_cells)).round(), 0, n_bins - 1).astype(np.uint16)
    centres = np.linspace(-5, 5, n_bins, dtype=np.float32)
    got, _ = H._slab_quantile_cells(block, slots, n_slots, n_bins, [0], window, 0.9, 0.1, mirror, centres)
    for cell in range(n_cells):
        counts = np.zeros((n_slots, n_bins), dtype=np.int64)
        np.add.at(counts, (slots - 1, block[:, cell].astype(np.int64)), 1)
        if mirror:
            want = -H._rolling_histogram_quantile(counts[:, ::-1], window, 0.1, centres)
        else:
            want = H._rolling_histogram_quantile(counts, window, 0.9, centres)
        assert np.unique(want).size == 1
        np.testing.assert_array_equal(got[cell], want)


def _monthly(n_years=40, n_cells=2, seed=0):
    """Monthly data whose level depends on the month, so pooling one month twice moves the quantile."""
    time = pd.date_range("1981-01-01", periods=12 * n_years, freq="MS")
    rng = np.random.default_rng(seed)
    level = 0.2 * time.month.values[:, None]
    return _mesh((level + 0.3 * rng.normal(size=(time.size, n_cells))).astype(np.float32), time)


def test_a_year_long_monthly_window_pools_the_whole_year():
    # 365 days is 12 Gregorian months -> odd 13 slots on a 12-month cycle. Both paths pool every
    # month once, so every month gets the same threshold: the percentile of the whole record.
    da = _monthly()
    cycle = resolve_cycle(da, "time")
    assert cycle.index_name == "month" and cycle.window_slots(365) == 13

    def thresholds(method):
        return (
            marEx.extremes.identify(
                da, threshold_percentile=90, window_days=365, method_percentile=method, dimensions=DIMS, coordinates=COORDS
            )
            .thresholds.transpose("month", "ncells")
            .values
        )

    exact = thresholds("exact")
    approximate = thresholds("approximate")
    for cell in range(da.sizes["ncells"]):
        whole_record = np.float32(np.percentile(da.isel(ncells=cell).values, 90))
        np.testing.assert_allclose(exact[:, cell], whole_record, rtol=0, atol=1e-6)
        assert np.unique(approximate[:, cell]).size == 1
        assert abs(approximate[0, cell] - whole_record) <= 0.02


def test_both_histogram_engines_pool_each_slot_once_past_the_cycle():
    da = _monthly()
    cycle = resolve_cycle(da, "time")
    da = da.assign_coords({cycle.index_name: cycle.index_of(da["time"]).compute()})
    common = {"q": 0.9, "window_steps": 13, "dimensions": DIMS, "cycle": cycle, "precision": 0.01, "max_anomaly": 5.0}
    dense = H._compute_histogram_quantile_2d(da, engine="dense", **common)
    slab = H._compute_histogram_quantile_2d(da, engine="slab", **common)
    xr.testing.assert_identical(dense, slab)
    for cell in range(da.sizes["ncells"]):
        assert np.unique(dense.isel(ncells=cell).values).size == 1
