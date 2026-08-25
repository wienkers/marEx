"""Unit tests for the :class:`~marEx.SeasonalCycle` primitive and cadence inference.

These pin the two properties Phase C's whole gate rests on:

1. **The daily path is the identity.** ``steps_for_days`` on a daily axis returns the
   number of days it was given, ``index_of`` is ``dt.dayofyear``, and the cycle length
   is 366 -- so every hardcoded 366 that now reads from a ``SeasonalCycle`` produces
   exactly what it produced before.
2. **``add_decimal_year``'s sub-day term is resolution-conditional.** Both shipped
   fixtures are stamped away from midnight (12:00:00 gridded, 23:59:59 unstructured),
   so an unconditional sub-day fraction would shift ``decimal_year``, change the
   harmonic design matrix, and move both golden configurations.
"""

import logging

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from marEx.core.time_axis import DAILY_CYCLE, SeasonalCycle, add_decimal_year, infer_cycle
from marEx.exceptions import ConfigurationError


def _series(times):
    return xr.DataArray(np.zeros(len(times)), dims=["time"], coords={"time": times})


class _Capture(logging.Handler):
    """Collect records straight off the marEx logger.

    ``caplog`` cannot see them: :func:`marEx.logging_config.configure_logging` sets
    ``propagate = False`` on the package root logger, so nothing reaches pytest's
    handler on the true root.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def __enter__(self):
        logging.getLogger("marEx").addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger("marEx").removeHandler(self)
        return False


class TestInferCycle:
    def test_daily_axis_gives_the_daily_cycle(self):
        cycle = infer_cycle(_series(pd.date_range("1982-01-01", periods=800, freq="D")).time)
        assert cycle == DAILY_CYCLE
        assert cycle.index_name == "dayofyear"
        assert cycle.length == 366

    def test_noon_stamped_daily_axis_is_still_daily(self):
        # The gridded fixture's stamp. The cadence, not the offset, sets the cycle.
        cycle = infer_cycle(_series(pd.date_range("1982-01-01T12:00:00", periods=400, freq="D")).time)
        assert cycle == DAILY_CYCLE

    def test_monthly_axis_gives_twelve_slots(self):
        cycle = infer_cycle(_series(pd.date_range("2000-01-01", periods=60, freq="MS")).time)
        assert cycle.index_name == "month"
        assert cycle.length == 12

    @pytest.mark.parametrize("freq,steps_per_day", [("6h", 4), ("h", 24), ("12h", 2)])
    def test_subdaily_axis_scales_the_cycle(self, freq, steps_per_day):
        cycle = infer_cycle(_series(pd.date_range("2000-01-01", periods=200, freq=freq)).time)
        assert cycle.index_name == "hourofyear"
        assert cycle.length == 366 * steps_per_day
        assert cycle.steps_per_day == steps_per_day

    def test_single_timestep_falls_back_to_daily(self):
        # Nothing to measure. Falling back beats guessing, and beats raising on a
        # perfectly ordinary one-step slice.
        assert infer_cycle(_series(pd.date_range("2000-01-01", periods=1)).time) == DAILY_CYCLE

    def test_two_timesteps_are_enough(self):
        # One diff: a median exists, and the IQR guard is deliberately not applied.
        cycle = infer_cycle(_series(pd.date_range("2000-01-01", periods=2, freq="D")).time)
        assert cycle == DAILY_CYCLE

    def test_a_single_gap_does_not_redefine_the_cycle(self):
        # The median, never the first diff or the range: a missing file must not turn a
        # daily series into a monthly one.
        times = list(pd.date_range("2000-01-01", periods=400, freq="D"))
        del times[100:130]
        assert infer_cycle(_series(pd.DatetimeIndex(times)).time) == DAILY_CYCLE

    def test_genuinely_irregular_axis_raises_rather_than_guessing(self):
        # A daily series concatenated with a monthly one: two interleaved cadences, so
        # the quartiles are pushed apart and no single cadence describes the axis.
        times = pd.DatetimeIndex(
            list(pd.date_range("2000-01-01", periods=40, freq="D")) + list(pd.date_range("2001-01-01", periods=40, freq="MS"))
        )
        with pytest.raises(ConfigurationError, match="too irregular"):
            infer_cycle(_series(times).time)

    @pytest.mark.parametrize("freq,periods", [("D", 400), ("MS", 60), ("6h", 200), ("h", 500)])
    def test_every_regular_cadence_is_well_clear_of_the_guard(self, freq, periods):
        # The guard must never fire on a regular axis. Monthly is the ragged case
        # (28-31 day spacings) and is the one worth pinning explicitly.
        infer_cycle(_series(pd.date_range("2000-01-01", periods=periods, freq=freq)).time)

    def test_a_run_of_missing_timesteps_does_not_trip_the_guard(self):
        # Gaps live in the tails; the middle half is untouched.
        times = list(pd.date_range("2000-01-01", periods=400, freq="D"))
        del times[100:160]
        del times[250:260]
        assert infer_cycle(_series(pd.DatetimeIndex(times)).time) == DAILY_CYCLE

    def test_cftime_axis_is_inferred_too(self):
        times = [cftime.DatetimeNoLeap(2000, 1, 1) + i * pd.Timedelta(days=1).to_pytimedelta() for i in range(100)]
        assert infer_cycle(_series(xr.CFTimeIndex(times)).time) == DAILY_CYCLE


class TestIndexOf:
    def test_daily_index_is_dt_dayofyear(self):
        time = _series(pd.date_range("2004-01-01", periods=400, freq="D")).time
        got = DAILY_CYCLE.index_of(time)
        assert got.name == "dayofyear"
        np.testing.assert_array_equal(got.values, time.dt.dayofyear.values)

    def test_monthly_index_is_dt_month(self):
        cycle = SeasonalCycle("month", 12, 30.0)
        time = _series(pd.date_range("2000-01-01", periods=36, freq="MS")).time
        got = cycle.index_of(time)
        assert got.name == "month"
        np.testing.assert_array_equal(got.values, time.dt.month.values)

    def test_subdaily_index_is_one_based_and_distinct_within_a_day(self):
        cycle = SeasonalCycle("hourofyear", 366 * 4, 0.25)
        time = _series(pd.date_range("2000-01-01", periods=8, freq="6h")).time
        got = cycle.index_of(time)
        assert got.name == "hourofyear"
        np.testing.assert_array_equal(got.values, [1, 2, 3, 4, 5, 6, 7, 8])

    def test_subdaily_index_stays_inside_the_cycle(self):
        cycle = SeasonalCycle("hourofyear", 366 * 4, 0.25)
        time = _series(pd.date_range("2004-12-31T18:00:00", periods=4, freq="6h")).time
        assert int(cycle.index_of(time).max()) <= cycle.length

    def test_labels_span_the_cycle(self):
        assert DAILY_CYCLE.labels[0] == 1 and DAILY_CYCLE.labels[-1] == 366
        assert len(SeasonalCycle("month", 12, 30.0).labels) == 12


class TestStepsForDays:
    @pytest.mark.parametrize("days", [3, 11, 21, 30, 31])
    def test_daily_conversion_is_the_identity(self, days):
        # This is what keeps every existing daily result bit-identical.
        assert DAILY_CYCLE.steps_for_days(days) == days

    def test_daily_odd_forcing_leaves_odd_windows_alone(self):
        assert DAILY_CYCLE.window_steps(11) == 11
        assert DAILY_CYCLE.window_steps(3) == 3

    def test_odd_forcing_rounds_an_even_request_up(self):
        assert DAILY_CYCLE.window_steps(10) == 11

    def test_even_step_counts_are_allowed_when_not_forced(self):
        # The climatology smoothing and the standardisation window rely on this: forcing
        # 30 days to 31 steps would move every standardised daily result.
        assert DAILY_CYCLE.steps_for_days(30) == 30

    def test_hourly_converts_a_duration_to_many_steps(self):
        cycle = SeasonalCycle("hourofyear", 366 * 24, 1 / 24)
        assert cycle.steps_for_days(11) == 264
        assert cycle.window_steps(11) == 265

    def test_monthly_clamps_and_warns(self):
        cycle = SeasonalCycle("month", 12, 30.0)
        with _Capture() as cap:
            assert cycle.steps_for_days(11, name="window_days") == 1
        assert any("window_days" in m and "cannot be represented" in m for m in cap.messages)

    def test_no_warning_when_the_request_is_representable(self):
        with _Capture() as cap:
            DAILY_CYCLE.steps_for_days(11)
        assert not cap.messages

    def test_never_returns_zero(self):
        assert SeasonalCycle("month", 12, 30.0).steps_for_days(1) == 1


class TestAddDecimalYear:
    def _legacy(self, times):
        """The exact pre-Phase-C arithmetic, transcribed."""
        t = pd.to_datetime(times)
        start = pd.to_datetime(t.year.astype(str) + "-01-01")
        nxt = pd.to_datetime((t.year + 1).astype(str) + "-01-01")
        return t.year + (t - start).days / (nxt - start).days

    @pytest.mark.parametrize("stamp", ["1982-01-01T12:00:00", "1991-01-01T23:59:59", "2000-01-01"])
    def test_daily_input_reproduces_the_legacy_values_exactly(self, stamp):
        # The two non-midnight stamps are the shipped fixtures'. Zero tolerance: any
        # movement here moves the harmonic design matrix and both goldens.
        da = _series(pd.date_range(stamp, periods=1200, freq="D"))
        got = add_decimal_year(da).decimal_year.values
        np.testing.assert_array_equal(got, np.asarray(self._legacy(da.time.values)))

    def test_inference_matches_an_explicit_false_on_daily_data(self):
        da = _series(pd.date_range("1982-01-01T12:00:00", periods=500, freq="D"))
        np.testing.assert_array_equal(
            add_decimal_year(da).decimal_year.values,
            add_decimal_year(da, subdaily=False).decimal_year.values,
        )

    def test_subdaily_input_resolves_within_the_day(self):
        da = _series(pd.date_range("2000-01-01", periods=40, freq="6h"))
        got = add_decimal_year(da).decimal_year.values
        assert len(np.unique(got[:4])) == 4
        np.testing.assert_allclose(np.diff(got[:4]), 0.25 / 366, rtol=1e-9)

    def test_subdaily_is_not_applied_to_a_daily_axis(self):
        da = _series(pd.date_range("2000-01-01T12:00:00", periods=40, freq="D"))
        got = add_decimal_year(da).decimal_year.values
        # Whole-day steps only: no half-day offset from the noon stamp.
        np.testing.assert_allclose(np.diff(got[:3]), 1.0 / 366, rtol=1e-9)

    def test_cftime_daily_branch_is_unchanged_by_the_conditional(self):
        times = [cftime.DatetimeNoLeap(2000, 1, 1) + i * pd.Timedelta(days=1).to_pytimedelta() for i in range(50)]
        da = _series(xr.CFTimeIndex(times))
        got = add_decimal_year(da).decimal_year.values
        expected = 2000 + (da.time.dt.dayofyear.values - 1) / 365.0
        np.testing.assert_array_equal(got, expected)

    def test_cftime_branch_takes_the_subday_term_when_asked(self):
        # Threaded identically to the datetime64 branch: the behaviour must depend on
        # the cadence, never on the calendar attribute.
        times = [cftime.DatetimeNoLeap(2000, 1, 1, 6 * i) for i in range(4)]
        da = _series(xr.CFTimeIndex(times))
        got = add_decimal_year(da, subdaily=True).decimal_year.values
        np.testing.assert_allclose(np.diff(got), 0.25 / 365.0, rtol=1e-9)
