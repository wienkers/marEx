"""The time axis: :class:`~marEx.SeasonalCycle`, cadence inference and naming, and ``add_decimal_year``.

Two properties the non-daily cadence support rests on are pinned here:

1. **The daily path is the identity.** ``steps_for_days`` on a daily axis returns the number of
   days it was given, ``index_of`` is ``dt.dayofyear``, and the cycle length is 366, so every
   hardcoded 366 that now reads from a ``SeasonalCycle`` produces exactly what it did before.
2. **``add_decimal_year``'s sub-day term is resolution-conditional.** Both shipped fixtures are
   stamped away from midnight (12:00:00 gridded, 23:59:59 unstructured), so an unconditional
   sub-day fraction would shift ``decimal_year``, change the harmonic design matrix, and move
   both golden configurations.

The tests over many small axes are tables: one behaviour each, every row runs, a failing row
is reported with its case id and does not hide the others.
"""

import cftime
import numpy as np
import pandas as pd
import xarray as xr

import marEx
from marEx.core.time_axis import (
    DAILY_CYCLE,
    SeasonalCycle,
    _median_step_days,
    add_decimal_year,
    cadence_index_name,
    infer_cycle,
    is_subdaily_axis,
    resolve_cycle,
)
from marEx.exceptions import ConfigurationError

from ..conftest import assert_no_failures


def _series(times, dim="time"):
    return xr.DataArray(np.zeros(len(times)), dims=[dim], coords={dim: times})


def _cftime_daily(n, calendar="noleap"):
    return xr.CFTimeIndex(
        [cftime.datetime(2000, 1, 1, calendar=calendar) + i * pd.Timedelta(days=1).to_pytimedelta() for i in range(n)]
    )


def _cf_range(start, periods, freq, calendar):
    return xr.date_range(start, periods=periods, freq=freq, calendar=calendar, use_cftime=True)


def _raises(func, exc, match=None):
    """None if ``func()`` raises ``exc`` (with ``match`` in its message), else what happened instead."""
    try:
        func()
    except exc as e:
        return None if match is None or match in str(e) else f"raised {type(e).__name__} without {match!r}: {e}"
    except Exception as e:  # noqa: BLE001 - a wrong exception type is a row failure, reported with the rest
        return f"raised {type(e).__name__} instead of {exc.__name__}: {e}"
    return f"did not raise {exc.__name__}"


def test_infer_cycle_reads_every_regular_cadence_on_every_calendar():
    """Every regular axis infers its own cycle: index name, length and steps per day.

    Monthly is the ragged case (28-31 day spacings); on ``360_day`` its spacings have zero
    spread. One and two timesteps fall back to daily rather than raising on an ordinary slice.
    """
    rows = [
        ("daily", pd.date_range("1982-01-01", periods=800, freq="D"), ("dayofyear", 366, 1), True),
        ("noon-stamped daily", pd.date_range("1982-01-01T12:00:00", periods=400, freq="D"), ("dayofyear", 366, 1), True),
        ("monthly", pd.date_range("2000-01-01", periods=60, freq="MS"), ("month", 12, 0), False),
        ("6-hourly", pd.date_range("2000-01-01", periods=200, freq="6h"), ("hourofyear", 366 * 4, 4), False),
        ("hourly", pd.date_range("2000-01-01", periods=200, freq="h"), ("hourofyear", 366 * 24, 24), False),
        ("12-hourly", pd.date_range("2000-01-01", periods=200, freq="12h"), ("hourofyear", 366 * 2, 2), False),
        ("one timestep", pd.date_range("2000-01-01", periods=1), ("dayofyear", 366, 1), True),
        ("two timesteps", pd.date_range("2000-01-01", periods=2, freq="D"), ("dayofyear", 366, 1), True),
        ("regular daily 400", pd.date_range("2000-01-01", periods=400, freq="D"), ("dayofyear", 366, 1), True),
        ("regular hourly 500", pd.date_range("2000-01-01", periods=500, freq="h"), ("hourofyear", 366 * 24, 24), False),
        ("cftime noleap daily", _cftime_daily(100), ("dayofyear", 366, 1), True),
    ]
    for calendar in ("noleap", "360_day", "all_leap", "standard"):
        rows.append((f"monthly {calendar}", _cf_range("2001-01-01", 48, "MS", calendar), ("month", 12, 0), False))
        rows.append((f"6-hourly {calendar}", _cf_range("2001-01-01", 400, "6h", calendar), ("hourofyear", 366 * 4, 4), False))

    failures = []
    for case, times, expected, is_daily_cycle in rows:
        try:
            cycle = infer_cycle(_series(times).time)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{case}: raised {type(e).__name__}: {e}")
            continue
        got = (cycle.index_name, cycle.length, cycle.steps_per_day)
        if got != expected:
            failures.append(f"{case}: (index_name, length, steps_per_day) {got}, expected {expected}")
        if is_daily_cycle and cycle != DAILY_CYCLE:
            failures.append(f"{case}: {cycle!r} is not DAILY_CYCLE")
    assert_no_failures(failures)


def test_infer_cycle_survives_gaps_and_nat_but_rejects_mixed_cadence():
    """The cadence is a median: missing files and NaT do not redefine it, two interleaved cadences do."""
    one_gap = list(pd.date_range("2000-01-01", periods=400, freq="D"))
    del one_gap[100:130]
    gap_runs = list(pd.date_range("2000-01-01", periods=400, freq="D"))
    del gap_runs[100:160]
    del gap_runs[250:260]
    # The pre-fix bug was NaT-fraction dependent: one NaT among 399 spacings never moved the median,
    # while two NaT spacings of three raised. Both axes now infer the daily cycle.
    one_nat = pd.DatetimeIndex(pd.date_range("2000-01-01", periods=400, freq="D")).delete(200).insert(200, pd.NaT)
    short_nat = pd.DatetimeIndex(["2000-01-01", "NaT", "2000-01-03", "2000-01-04"])

    failures = []
    for case, times in (("single gap", pd.DatetimeIndex(one_gap)), ("runs of missing steps", pd.DatetimeIndex(gap_runs))):
        cycle = infer_cycle(_series(times).time)
        if cycle != DAILY_CYCLE:
            failures.append(f"{case}: {cycle!r}")
    for case, times in (("one NaT in 400", one_nat), ("two NaT spacings of three", short_nat)):
        try:
            cycle = infer_cycle(_series(times).time, "time")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{case}: raised {type(e).__name__}: {e}")
            continue
        if (cycle.index_name, cycle.length, cycle.step_days) != ("dayofyear", 366, 1.0):
            failures.append(f"{case}: {cycle!r}")
    mixed = pd.DatetimeIndex(
        list(pd.date_range("2000-01-01", periods=40, freq="D")) + list(pd.date_range("2001-01-01", periods=40, freq="MS"))
    )
    problem = _raises(lambda: infer_cycle(_series(mixed).time), ConfigurationError, "too irregular")
    if problem:
        failures.append(f"daily + monthly: {problem}")
    assert_no_failures(failures)


def test_cadence_helpers_read_real_spacings_and_never_raise():
    """``_median_step_days``, ``is_subdaily_axis`` and ``cadence_index_name`` on awkward axes.

    ``cadence_index_name`` words an output attribute, so it never raises on an axis that exists:
    ``infer_cycle`` rejects a mixed-cadence axis and ``global_percentile`` needs no cycle at all.
    NaT spacings are dropped before the median (NaT's int64 sentinel survives a float cast as
    -2**63 and once read as a negative, sub-daily spacing), but dropping them must not push a
    genuinely sub-daily axis to daily. A decreasing axis keeps the non-positive guard live.
    """

    def da(times):
        times = pd.DatetimeIndex(times)
        return xr.DataArray(np.zeros(len(times), np.float32), dims=["time"], coords={"time": times})

    # (case, axis, cycle, expected median step, expected is_subdaily, expected name); None skips a check.
    rows = [
        ("monthly", da(pd.date_range("2000-01-01", periods=200, freq="MS")), None, None, None, "month"),
        ("daily", da(pd.date_range("2000-01-01", periods=200, freq="D")), None, None, None, "dayofyear"),
        ("6-hourly", da(pd.date_range("2000-01-01", periods=200, freq="6h")), None, None, None, "hourofyear"),
        ("hourly", da(pd.date_range("2000-01-01", periods=200, freq="h")), None, None, None, "hourofyear"),
        (
            "supplied cycle wins",
            da(pd.date_range("2000-01-01", periods=50, freq="D")),
            SeasonalCycle("month", 12, 30.0),
            None,
            None,
            "month",
        ),
        ("one step", da(pd.date_range("2000-01-01", periods=1, freq="D")), None, None, None, "dayofyear"),
        ("NaT in a daily axis", da(["2000-01-01", "NaT", "2000-01-03", "2000-01-04"]), None, 1.0, False, "dayofyear"),
        ("NaT in a sub-daily axis", da(["2000-01-01T00", "NaT", "2000-01-01T12", "2000-01-01T18"]), None, 0.25, True, "hourofyear"),
        ("every spacing NaT", da(["2000-01-01", "NaT"]), None, 1.0, False, "dayofyear"),
        ("no surviving spacing", da(["2000-01-01T00", "NaT", "2000-01-01T02"]), None, 1.0, False, None),
        ("decreasing", da(["2000-01-04", "2000-01-03", "2000-01-02", "2000-01-01"]), None, -1.0, None, "dayofyear"),
    ]
    failures = []
    for case, axis, cycle, median, subdaily, name in rows:
        if median is not None and _median_step_days(axis.time) != median:
            failures.append(f"{case}: median step {_median_step_days(axis.time)}, expected {median}")
        if subdaily is not None and is_subdaily_axis(axis, "time") is not subdaily:
            failures.append(f"{case}: is_subdaily_axis {is_subdaily_axis(axis, 'time')}, expected {subdaily}")
        if name is not None:
            got = cadence_index_name(axis, "time", cycle) if cycle is not None else cadence_index_name(axis, "time")
            if got != name:
                failures.append(f"{case}: cadence_index_name {got!r}, expected {name!r}")

    # infer_cycle rejects this axis; wording it must not raise.
    mixed = da(["2000-01-01", "2000-01-02", "2000-03-01", "2000-03-02", "2000-09-14", "2000-09-15"])
    problem = _raises(lambda: marEx.infer_cycle(mixed.time, "time"), ConfigurationError)
    if problem:
        failures.append(f"mixed cadence, infer_cycle: {problem}")
    try:
        if cadence_index_name(mixed, "time") not in {"month", "dayofyear", "hourofyear"}:
            failures.append(f"mixed cadence: named {cadence_index_name(mixed, 'time')!r}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"mixed cadence: cadence_index_name raised {type(e).__name__}: {e}")
    # The no-raise guarantee is bounded to a coordinate that exists: a missing name is a caller bug.
    problem = _raises(lambda: cadence_index_name(da(pd.date_range("2000-01-01", periods=10)), "not_a_coordinate"), KeyError)
    if problem:
        failures.append(f"missing coordinate: {problem}")
    assert_no_failures(failures)


def test_index_of_and_labels_follow_the_cycle():
    """``index_of`` is ``dt.dayofyear`` / ``dt.month`` on those cycles, and a one-based slot inside the cycle sub-daily."""
    failures = []
    daily_time = _series(pd.date_range("2004-01-01", periods=400, freq="D")).time
    monthly_time = _series(pd.date_range("2000-01-01", periods=36, freq="MS")).time
    sixhourly = SeasonalCycle("hourofyear", 366 * 4, 0.25)
    rows = [
        ("daily", DAILY_CYCLE, daily_time, "dayofyear", daily_time.dt.dayofyear.values),
        ("monthly", SeasonalCycle("month", 12, 30.0), monthly_time, "month", monthly_time.dt.month.values),
        ("6-hourly", sixhourly, _series(pd.date_range("2000-01-01", periods=8, freq="6h")).time, "hourofyear", np.arange(1, 9)),
    ]
    for case, cycle, time, name, expected in rows:
        got = cycle.index_of(time)
        if got.name != name:
            failures.append(f"{case}: index named {got.name!r}, expected {name!r}")
        if not np.array_equal(got.values, expected):
            failures.append(f"{case}: index {got.values[:8]}..., expected {np.asarray(expected)[:8]}...")
    year_end = sixhourly.index_of(_series(pd.date_range("2004-12-31T18:00:00", periods=4, freq="6h")).time)
    if int(year_end.max()) > sixhourly.length:
        failures.append(f"6-hourly across a leap year end: slot {int(year_end.max())} > {sixhourly.length}")
    if not (DAILY_CYCLE.labels[0] == 1 and DAILY_CYCLE.labels[-1] == 366):
        failures.append(f"daily labels span {DAILY_CYCLE.labels[0]}..{DAILY_CYCLE.labels[-1]}")
    if len(SeasonalCycle("month", 12, 30.0).labels) != 12:
        failures.append("monthly labels are not 12")
    assert_no_failures(failures)


def test_durations_convert_to_steps():
    """``steps_for_days`` converts a duration; ``window_steps`` also forces it odd. Never zero steps.

    The daily rows are the identity that keeps every daily result bit-identical, and an even
    step count stays even when not forced (the climatology smoothing and the standardisation
    window rely on it: forcing 30 days to 31 steps would move every standardised daily result).
    """
    hourly = SeasonalCycle("hourofyear", 366 * 24, 1 / 24)
    rows = [(f"daily steps_for_days({d})", DAILY_CYCLE.steps_for_days(d), d) for d in (3, 11, 21, 30, 31)]
    rows += [
        ("daily window_steps(11)", DAILY_CYCLE.window_steps(11), 11),
        ("daily window_steps(3)", DAILY_CYCLE.window_steps(3), 3),
        ("daily window_steps(10) rounds up", DAILY_CYCLE.window_steps(10), 11),
        ("hourly steps_for_days(11)", hourly.steps_for_days(11), 264),
        ("hourly window_steps(11)", hourly.window_steps(11), 265),
        ("monthly steps_for_days(1) is not zero", SeasonalCycle("month", 12, 30.0).steps_for_days(1), 1),
    ]
    failures = [f"{case}: {got}, expected {want}" for case, got, want in rows if got != want]
    assert_no_failures(failures)


def test_steps_for_days_warns_only_when_unrepresentable(marex_warnings):
    """An 11-day request on a monthly axis clamps to one step and says so; a representable one is silent."""
    with marex_warnings as cap:
        steps = SeasonalCycle("month", 12, 30.0).steps_for_days(11, name="window_days")
    assert steps == 1
    assert any("window_days" in m and "cannot be represented" in m for m in cap.messages), cap.messages
    with marex_warnings as cap:
        DAILY_CYCLE.steps_for_days(11)
    assert not cap.messages, cap.messages


def _mesh(values, time):
    cells = np.arange(values.shape[1], dtype=np.float32)
    return xr.DataArray(
        values,
        dims=("time", "ncells"),
        coords={"time": time, "lon": ("ncells", cells), "lat": ("ncells", cells)},
        name="dat_anomaly",
    ).chunk({"time": -1})


def test_window_slots_count_calendar_slots():
    """``window_days`` is measured in cycle slots (calendar days or months), not data steps.

    5-day data still sits on the 366-slot cycle, one slot per day, so an 11-day window is 11
    slots, not 3 data steps. A month is the calendar's month: 30 days on ``360_day``, 30.4 on
    the Gregorian, whatever the median spacing of a short record or a quarterly axis says.
    Where one slot is one step the slot count is the old step count, so those thresholds cannot move.
    """
    failures = []
    five_day_time = pd.date_range("1980-01-01", periods=int(2 * 365.25 / 5), freq="5D")
    five_day = resolve_cycle(_mesh(np.zeros((five_day_time.size, 2), np.float32), five_day_time), "time")
    if not (five_day.is_daily and five_day.step_days == 5.0):
        failures.append(f"5-day axis: {five_day!r}")
    for days in (11, 31):
        if five_day.window_slots(days) != days:
            failures.append(f"5-day axis, {days} days: {five_day.window_slots(days)} slots")

    per_step = {
        "daily": DAILY_CYCLE,
        "hourly": SeasonalCycle("hourofyear", 366 * 24, 1.0 / 24),
        "6-hourly": SeasonalCycle("hourofyear", 366 * 4, 0.25),
    }
    for case, cycle in per_step.items():
        moved = [days for days in range(1, 62, 2) if cycle.window_slots(days) != cycle.window_steps(days)]
        if moved:
            failures.append(f"{case}: window_slots != window_steps at {moved}")

    # 360-day calendar: 106 days is 3.5 months -> 4 -> odd 5 slots.
    time360 = xr.date_range("2001-01-01", periods=12 * 20, freq="MS", calendar="360_day", use_cftime=True)
    da360 = _mesh(np.repeat(np.arange(time360.size, dtype=np.float32)[:, None] % 12 + 1, 2, axis=1), time360)
    cycle360 = resolve_cycle(da360, "time")
    if not (cycle360.index_name == "month" and cycle360.step_days == 30.0):
        failures.append(f"360_day monthly: {cycle360!r}")
    if cycle360.window_slots(106, calendar=da360["time"].dt.calendar) != 5:
        failures.append(f"360_day monthly, 106 days: {cycle360.window_slots(106, calendar='360_day')} slots, expected 5")

    # Two monthly samples, February to March: the median spacing is 28 days, a Gregorian month is not.
    short = resolve_cycle(_mesh(np.zeros((2, 2), np.float32), pd.date_range("2001-02-01", periods=2, freq="MS")), "time")
    if not (short.index_name == "month" and short.step_days == 28.0):
        failures.append(f"short monthly record: {short!r}")
    if short.window_slots(106) != 3:
        failures.append(f"short monthly record, 106 days: {short.window_slots(106)} slots, expected 3")

    # A Gregorian month is 30.4 days, not the 31-day median spacing; a 5-hourly axis has 5 slots per
    # day (0.2 days each), not 4.8 steps. Counting the cadence gave 1, 53 and 39.
    for case, freq, periods, days, slots in (
        ("monthly-gregorian", "MS", 120, 46, 3),
        ("5-hourly", "5h", 2000, 11, 55),
        ("7-hourly", "7h", 2000, 11, 33),
    ):
        time = pd.date_range("2001-01-01", periods=periods, freq=freq)
        got = resolve_cycle(_mesh(np.zeros((periods, 2), np.float32), time), "time").window_slots(days)
        if got != slots:
            failures.append(f"{case}, {days} days: {got} slots, expected {slots}")
    assert_no_failures(failures)


def _legacy_decimal_year(times):
    """The daily-only arithmetic, transcribed."""
    t = pd.to_datetime(times)
    start = pd.to_datetime(t.year.astype(str) + "-01-01")
    nxt = pd.to_datetime((t.year + 1).astype(str) + "-01-01")
    return np.asarray(t.year + (t - start).days / (nxt - start).days)


def test_add_decimal_year_daily_axes_take_no_subday_term():
    """On a daily axis ``decimal_year`` is the legacy whole-day value, exactly, on every calendar.

    Zero tolerance: any movement here moves the harmonic design matrix and both goldens. The
    two non-midnight stamps are the shipped fixtures'. The cftime conditional must depend on the
    cadence, never on the calendar attribute.
    """
    failures = []

    def check(case, got, want):
        if not np.array_equal(np.asarray(got), np.asarray(want)):
            failures.append(f"{case}: {np.asarray(got)[:3]}..., expected {np.asarray(want)[:3]}...")

    for stamp in ("1982-01-01T12:00:00", "1991-01-01T23:59:59", "2000-01-01"):
        da = _series(pd.date_range(stamp, periods=1200, freq="D"))
        check(f"legacy values from {stamp}", add_decimal_year(da).decimal_year.values, _legacy_decimal_year(da.time.values))

    noon = _series(pd.date_range("1982-01-01T12:00:00", periods=500, freq="D"))
    check(
        "inferred == subdaily=False",
        add_decimal_year(noon).decimal_year.values,
        add_decimal_year(noon, subdaily=False).decimal_year.values,
    )
    # Whole-day steps only: no half-day offset from a noon stamp.
    noon2000 = add_decimal_year(_series(pd.date_range("2000-01-01T12:00:00", periods=40, freq="D"))).decimal_year.values
    check("noon stamp", noon2000, 2000 + np.arange(40) / 366)
    # The datetime64 path divides by the proleptic Gregorian year: 2000 is a leap year.
    dt64 = add_decimal_year(
        _series(pd.date_range("2000-01-01", periods=365, freq="D")), dim="time", coord="time"
    ).decimal_year.values
    check("datetime64 leap divisor", dt64, 2000 + np.arange(365) / 366)

    # Known dates on the datetime64 path.
    leap = add_decimal_year(_series(pd.date_range("2020-01-01", "2020-12-31", freq="D")), "time").decimal_year.values
    check("2020-01-01, 2020-06-01, 2020-12-31", leap[[0, 152, 365]], [2020.0, 2020 + 152 / 366, 2020 + 365 / 366])
    non_leap = add_decimal_year(_series(pd.date_range("2021-01-01", "2021-12-31", freq="D")), "time").decimal_year.values
    check("2021-12-31", non_leap[-1], 2021 + 364 / 365)
    single = xr.DataArray([1.0], coords={"time": [pd.Timestamp("2020-07-01")]}, dims=["time"])
    check("single date 2020-07-01", add_decimal_year(single, "time").decimal_year.values, [2020 + 182 / 366])
    custom_dim = add_decimal_year(_series(pd.date_range("2020-01-01", "2020-01-10", freq="D"), "custom_time"), "custom_time")
    check("custom dimension name", custom_dim.decimal_year.values, 2020 + np.arange(10) / 366)
    if custom_dim.decimal_year.dims != ("custom_time",):
        failures.append(f"custom dimension name: decimal_year on {custom_dim.decimal_year.dims}")

    # A coordinate named differently from its dimension is read through ``coord``.
    dates = pd.date_range("1982-01-01", "1982-01-10", freq="D")
    renamed = xr.DataArray(np.zeros((10, 3, 4)), dims=["t", "y", "x"], coords={"T": ("t", dates)})
    by_coord = add_decimal_year(renamed, dim="t", coord="T")
    check("dim 't', coord 'T'", by_coord.decimal_year.values, 1982 + np.arange(10) / 365)
    if by_coord.decimal_year.dims != ("t",):
        failures.append(f"dim 't', coord 'T': decimal_year on {by_coord.decimal_year.dims}")
    # Without ``coord`` the dimension's own index (0..9) is read as nanoseconds since 1970.
    by_dim = add_decimal_year(renamed, dim="t").decimal_year.values
    if np.all(by_dim >= 1982.0):
        failures.append(f"dim 't' without coord: read the 'T' dates ({by_dim[:3]}), so coord is not what selects them")

    for calendar, days_in_year in (("noleap", 365), ("360_day", 360), ("all_leap", 366)):
        time = _cf_range("2001-01-01", 3 * days_in_year, "D", calendar)
        cf = _series(time)
        want = cf.time.dt.year.values + (cf.time.dt.dayofyear.values - 1) / days_in_year
        check(f"cftime {calendar}", add_decimal_year(cf, dim="time", coord="time").decimal_year.values, want)
        noon_cf = _series(_cf_range("2001-01-01T12:00:00", 30, "D", calendar))
        check(
            f"cftime {calendar} noon stamp",
            add_decimal_year(noon_cf).decimal_year.values,
            2001 + (noon_cf.time.dt.dayofyear.values - 1) / days_in_year,
        )
    standard = _series(_cf_range("2020-01-01", 366, "D", "standard"))
    want = 2020 + (standard.time.dt.dayofyear.values - 1) / 366
    check("cftime standard leap year", add_decimal_year(standard, dim="time", coord="time").decimal_year.values, want)
    noleap = _series(_cftime_daily(50))
    check(
        "cftime noleap from 2000",
        add_decimal_year(noleap).decimal_year.values,
        2000 + (noleap.time.dt.dayofyear.values - 1) / 365.0,
    )

    # A plain numeric coordinate takes the legacy pd.to_datetime path, not the cftime branch.
    numeric = xr.DataArray(np.arange(20.0), dims=["time"], coords={"t": ("time", np.arange(20, dtype="int64"))})
    try:
        values = add_decimal_year(numeric, dim="time", coord="t").decimal_year.values
        if not np.all(np.isfinite(values)):
            failures.append(f"numeric coordinate: non-finite {values}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"numeric coordinate: raised {type(e).__name__}: {e}")
    assert_no_failures(failures)


def test_add_decimal_year_subdaily_axes_take_the_subday_term():
    """On a sub-daily axis each step advances by its fraction of that calendar's year, on both branches."""
    failures = []
    rows = [("datetime64 6-hourly", _series(pd.date_range("2000-01-01", periods=40, freq="6h")), {}, 366)]
    for calendar, days_in_year in (("noleap", 365), ("360_day", 360), ("all_leap", 366)):
        rows.append((f"cftime {calendar} 6-hourly", _series(_cf_range("2001-01-01", 8, "6h", calendar)), {}, days_in_year))
    forced = _series(xr.CFTimeIndex([cftime.DatetimeNoLeap(2000, 1, 1, 6 * i) for i in range(4)]))
    rows.append(("cftime noleap, subdaily=True", forced, {"subdaily": True}, 365))
    for case, da, kwargs, days_in_year in rows:
        got = add_decimal_year(da, **kwargs).decimal_year.values
        if len(np.unique(got[:4])) != 4:
            failures.append(f"{case}: first day has {len(np.unique(got[:4]))} distinct values")
        steps = np.diff(got[:8])
        if not np.allclose(steps, 0.25 / days_in_year, rtol=1e-9, atol=0):
            failures.append(f"{case}: steps {steps[:3]}, expected {0.25 / days_in_year}")
    assert_no_failures(failures)


def test_add_decimal_year_rejects_nat():
    """A NaT axis gets marEx's own error, on datetime64 and on an object (cftime) axis alike.

    The legacy branch would build "2000.0-01-01" from a NaT's float year and raise pandas'
    ``DateParseError`` with no suggestion. The object-dtype row needs ``pd.isna``, not ``np.isnat``.
    """
    dt64 = xr.DataArray(
        np.zeros(4, np.float32), dims=["time"], coords={"time": pd.DatetimeIndex(["2000-01-01", "NaT", "2000-01-03", "2000-01-04"])}
    )
    objects = np.array([cftime.DatetimeNoLeap(2000, 1, 1), pd.NaT, cftime.DatetimeNoLeap(2000, 1, 3)], dtype=object)
    obj = xr.DataArray(np.zeros(3, np.float32), dims=["time"], coords={"time": objects})
    failures = []
    for case, da in (("datetime64", dt64), ("object dtype", obj)):
        problem = _raises(lambda da=da: add_decimal_year(da, "time"), ConfigurationError, "NaT")
        if problem:
            failures.append(f"{case}: {problem}")
    assert_no_failures(failures)
