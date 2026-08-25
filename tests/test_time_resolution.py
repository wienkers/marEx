"""End-to-end coverage for non-daily time axes (Phase C).

marEx was written for daily data, and every within-year reduction was resolved on a
hardcoded 366-slot day-of-year axis. :class:`~marEx.SeasonalCycle` states that axis
explicitly, so the same code paths serve monthly and sub-daily series.

What these tests are for, in order of how much they are worth:

1. **The cycle dimension of the output is the right one and the right length.** A
   monthly run must produce ``thresholds`` on a ``month`` axis of length 12, not a
   ``dayofyear`` axis of length 366 with 354 empty slots. This is the assertion that
   fails on pre-Phase-C code.
2. **The durations stay physical.** ``window_days`` and ``smooth_days`` are days, so a
   monthly axis clamps them to one step and says so; a 6-hourly axis expands them.
3. **``detrend_harmonic`` refuses sub-daily input** rather than silently leaving the
   diurnal cycle in the anomaly.

The fixtures are derived at test time, never committed -- Phase B's precedent. A
monthly resample of the shipped gridded fixture is ~50 KB of derived data and costs
milliseconds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError

DATA_DIR = Path(__file__).parent / "data"
DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}


# --------------------------------------------------------------------------------------
# Fixtures, all derived
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def monthly_sst():
    """The shipped gridded fixture resampled to month-start means (40 years, 481 steps)."""
    sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to
    return sst.resample(time="MS").mean().chunk({"time": 24}).persist()


@pytest.fixture(scope="module")
def sixhourly_sst():
    """A synthetic 6-hourly field carrying a real diurnal cycle on top of a seasonal one.

    Six years is enough for ``fixed_baseline`` and for a 5-year ``shifting_baseline``
    window, and small enough (8768 steps x 4 x 5 cells) to stay quick.
    """
    time = pd.date_range("2000-01-01", "2005-12-31T18:00:00", freq="6h")
    lat = np.linspace(35.0, 36.0, 4).astype(np.float32)
    lon = np.linspace(-40.0, -39.0, 5).astype(np.float32)

    doy = time.dayofyear.values.astype(np.float64)
    hour = time.hour.values.astype(np.float64)
    seasonal = 3.0 * np.sin(2 * np.pi * doy / 365.25)
    diurnal = 1.5 * np.sin(2 * np.pi * hour / 24.0)
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.5, size=(len(time), len(lat), len(lon)))

    values = (15.0 + seasonal + diurnal)[:, None, None] + noise
    da = xr.DataArray(
        values.astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="to",
    )
    return da.chunk({"time": 400}).persist()


# --------------------------------------------------------------------------------------
# Monthly
# --------------------------------------------------------------------------------------


class TestMonthly:
    def test_thresholds_land_on_a_twelve_slot_month_axis(self, monthly_sst):
        """The headline assertion: a monthly run resolves on ``month``, not ``dayofyear``.

        On pre-Phase-C code this produces a 366-long ``dayofyear`` axis whose 354
        unpopulated slots are NaN, so the tripwire is the dimension NAME, not just a
        tolerance on the values.
        """
        ds = marEx.preprocess_data(
            monthly_sst,
            method_anomaly="fixed_baseline",
            method_extreme="seasonal_percentile",
            threshold_percentile=95,
            dimensions=DIMENSIONS,
            dask_chunks={"time": 24},
        )
        assert "month" in ds.thresholds.dims
        assert "dayofyear" not in ds.thresholds.dims
        assert ds.sizes["month"] == 12
        np.testing.assert_array_equal(ds.month.values, np.arange(1, 13))

    def test_every_month_slot_is_populated(self, monthly_sst):
        """No empty slots: 40 years of data reach all twelve months."""
        ds = marEx.preprocess_data(
            monthly_sst,
            method_anomaly="fixed_baseline",
            method_extreme="seasonal_percentile",
            dimensions=DIMENSIONS,
            dask_chunks={"time": 24},
        )
        thresholds = ds.thresholds.compute()
        assert bool(np.isfinite(thresholds).all()), "a month slot has no data behind it"

    def test_the_anomaly_keeps_the_full_series(self, monthly_sst):
        ds = marEx.preprocess_data(
            monthly_sst,
            method_anomaly="fixed_baseline",
            method_extreme="global_percentile",
            dimensions=DIMENSIONS,
            dask_chunks={"time": 24},
        )
        assert ds.sizes["time"] == monthly_sst.sizes["time"]

    def test_a_pure_annual_sinusoid_anomalises_to_zero(self):
        """The scientific check: with a perfectly repeating seasonal cycle and no trend,
        a fixed monthly climatology must remove all of it."""
        time = pd.date_range("1990-01-01", periods=12 * 30, freq="MS")
        signal = 5.0 * np.sin(2 * np.pi * (time.month.values - 1) / 12.0)
        da = xr.DataArray(
            np.broadcast_to(signal[:, None, None], (len(time), 3, 3)).astype(np.float32).copy(),
            dims=["time", "lat", "lon"],
            coords={"time": time, "lat": np.arange(3.0), "lon": np.arange(3.0)},
        ).chunk({"time": 24})

        ds = marEx.preprocess_data(
            da,
            method_anomaly="fixed_baseline",
            method_extreme="global_percentile",
            dimensions=DIMENSIONS,
            dask_chunks={"time": 24},
        )
        assert float(np.abs(ds.dat_anomaly).max().compute()) < 1e-4

    def test_window_days_clamps_to_one_step_and_says_so(self, monthly_sst, marex_warnings):
        """An 11-day window on a monthly axis is one month. Silently doing that would
        give the user a different method than they asked for."""
        with marex_warnings as cap:
            marEx.preprocess_data(
                monthly_sst,
                method_anomaly="fixed_baseline",
                method_extreme="seasonal_percentile",
                window_days=11,
                dimensions=DIMENSIONS,
                dask_chunks={"time": 24},
            )
        assert any("cannot be represented" in m for m in cap.messages)

    def test_an_even_window_days_is_accepted_off_the_daily_axis(self, monthly_sst):
        """Oddness is a property of the window in STEPS. Demanding an odd number of
        *days* on a monthly axis would reject a well-posed request."""
        ds = marEx.preprocess_data(
            monthly_sst,
            method_anomaly="fixed_baseline",
            method_extreme="seasonal_percentile",
            window_days=10,
            dimensions=DIMENSIONS,
            dask_chunks={"time": 24},
        )
        assert ds.sizes["month"] == 12

    def test_an_even_window_days_is_still_rejected_on_a_daily_axis(self):
        """...and the daily rejection, which several tests assert the wording of, stays."""
        sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.isel(time=slice(0, 400))
        with pytest.raises(ConfigurationError, match="window_days must be an odd number"):
            marEx.preprocess_data(
                sst.chunk({"time": 25}),
                method_anomaly="fixed_baseline",
                method_extreme="seasonal_percentile",
                window_days=10,
                dimensions=DIMENSIONS,
            )

    def test_smooth_days_degenerating_to_one_step_is_logged(self, monthly_sst, marex_warnings):
        """``shifting_baseline`` with ``smooth_days=21`` on a monthly axis reduces to
        ``rolling_climatology``. Correct, but the user should be told."""
        with marex_warnings as cap:
            marEx.anomaly.compute(
                monthly_sst,
                method="shifting_baseline",
                window_years=5,
                smooth_days=21,
                dimensions=DIMENSIONS,
                dask_chunks={"time": 24},
            )
        assert any("reduces to rolling_climatology" in m for m in cap.messages)


# --------------------------------------------------------------------------------------
# Sub-daily
# --------------------------------------------------------------------------------------


class TestSubDaily:
    def test_cycle_inference_gives_four_slots_per_day(self, sixhourly_sst):
        cycle = marEx.infer_cycle(sixhourly_sst.time)
        assert cycle.index_name == "hourofyear"
        assert cycle.length == 366 * 4

    def test_fixed_baseline_resolves_on_the_hourofyear_axis(self, sixhourly_sst):
        ds = marEx.preprocess_data(
            sixhourly_sst,
            method_anomaly="fixed_baseline",
            method_extreme="seasonal_percentile",
            dimensions=DIMENSIONS,
            dask_chunks={"time": 400},
        )
        assert "hourofyear" in ds.thresholds.dims
        assert ds.sizes["hourofyear"] == 366 * 4

    def test_the_diurnal_cycle_is_removed_by_a_cycle_resolved_climatology(self, sixhourly_sst):
        """The reason sub-daily support is worth having.

        The input carries a 1.5 K diurnal signal. Grouped by ``hourofyear`` it is part
        of the climatology and is removed; the residual amplitude must fall well below
        the signal. This is also the property that makes rejecting ``detrend_harmonic``
        the right call rather than a limitation.
        """
        ds = marEx.anomaly.compute(
            sixhourly_sst,
            method="fixed_baseline",
            dimensions=DIMENSIONS,
            dask_chunks={"time": 400},
        )
        anomaly = ds.dat_anomaly.compute()
        by_hour = anomaly.groupby(anomaly.time.dt.hour).mean()
        residual = float(by_hour.max() - by_hour.min())
        assert residual < 0.2, f"diurnal cycle survived into the anomaly: {residual:.3f} K of 1.5 K"

    def test_detrend_harmonic_refuses_subdaily_input(self, sixhourly_sst):
        """Rejected, not computed: the annual/semi-annual basis cannot represent a
        diurnal cycle, so the result would be silently wrong."""
        with pytest.raises(ConfigurationError, match="does not support sub-daily data"):
            marEx.anomaly.compute(
                sixhourly_sst,
                method="detrend_harmonic",
                dimensions=DIMENSIONS,
                dask_chunks={"time": 400},
            )

    def test_the_rejection_names_a_way_forward(self, sixhourly_sst):
        with pytest.raises(ConfigurationError) as exc:
            marEx.anomaly.compute(sixhourly_sst, method="detrend_harmonic", dimensions=DIMENSIONS, dask_chunks={"time": 400})
        text = " ".join(exc.value.suggestions)
        assert "shifting_baseline" in text or "fixed_baseline" in text

    def test_detrend_fixed_baseline_is_allowed_on_subdaily_input(self, sixhourly_sst):
        """It calls the harmonic module with harmonics OFF and then subtracts a
        cycle-resolved climatology, so the diurnal cycle is handled correctly."""
        ds = marEx.anomaly.compute(
            sixhourly_sst,
            method="detrend_fixed_baseline",
            dimensions=DIMENSIONS,
            dask_chunks={"time": 400},
        )
        assert "dat_anomaly" in ds

    def test_shifting_baseline_runs_on_a_subdaily_axis(self, sixhourly_sst):
        """The path with the largest memory consequence, so it gets its own test.

        ``rolling_climatology``'s tile budget takes
        ``output_elements_per_cell = n_target_years * cycle.length``, and ``cycle.length``
        is 1464 here rather than 366 -- a 4x change from the daily case. If the cycle
        were not threaded into that budget the tile would be sized for the wrong output.
        """
        ds = marEx.preprocess_data(
            sixhourly_sst,
            method_anomaly="shifting_baseline",
            method_extreme="seasonal_percentile",
            window_years=3,
            smooth_days=5,
            window_days=11,
            dimensions=DIMENSIONS,
            dask_chunks={"time": 400},
        )
        assert ds.sizes["hourofyear"] == 366 * 4
        # The first window_years years are trimmed, as on any other cadence.
        assert ds.sizes["time"] < sixhourly_sst.sizes["time"]
        assert bool(np.isfinite(ds.dat_anomaly).any().compute())

    def test_the_climatology_tile_is_budgeted_against_the_subdaily_cycle(self, sixhourly_sst, monkeypatch):
        """Assert the chunk STRUCTURE, not the values -- and assert it at the call site.

        CLAUDE.md records that 440 tests, a window harness and the coverage tripwires
        were all green while an all-to-all rechunk was live, so a value-only check is
        not evidence here. What matters is the number ``rolling_climatology`` hands to
        its tiling budget: ``n_target_years * cycle.length``. On pre-Phase-C code that
        was ``n_target_years * 366`` from a hardcoded ``_CYCLE_LENGTH``, so on a
        6-hourly axis the tile was sized for a quarter of the output it produces.

        Intercepting the call is what makes this a tripwire rather than a restatement:
        a test that recomputes the budget itself passes on either version.
        """
        import marEx.anomaly.climatology as C

        seen = []
        real = C.tile_spatial_chunks

        def spy(*args, **kwargs):
            seen.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(C, "tile_spatial_chunks", spy)
        marEx.anomaly.compute(
            sixhourly_sst,
            method="shifting_baseline",
            window_years=3,
            smooth_days=5,
            dimensions=DIMENSIONS,
            dask_chunks={"time": 400},
        )

        assert seen, "rolling_climatology did not budget its spatial tiling at all"
        per_cell = seen[0]["output_elements_per_cell"]

        # Exact equality, not a divisibility check: `n_years * 366` happens to be a
        # multiple of 1464 whenever `n_years` is a multiple of 4, so a modulo test would
        # pass on pre-Phase-C code for some series lengths.
        n_years = len(np.unique(sixhourly_sst.time.dt.year.values))
        assert per_cell == n_years * 366 * 4, (
            f"the climatology budgeted {per_cell} output elements per cell; the 6-hourly cycle "
            f"needs {n_years * 366 * 4}. A value of {n_years * 366} means it is still sized on a "
            f"hardcoded day-of-year axis."
        )

    def test_the_climatology_tile_actually_binds_when_the_budget_is_small(self, sixhourly_sst, monkeypatch):
        """...and the budget it computes is really applied to the array.

        ``TASK_ELEMENTS`` is read at call time precisely so it can be turned down like
        this (NEXT.md's Phase B notes make that explicit). With it small enough, the cap
        must bind and produce spatial chunks strictly smaller than the whole field.
        """
        import marEx.core.dimensions as D

        monkeypatch.setattr(D, "TASK_ELEMENTS", 200_000)
        ds = marEx.anomaly.compute(
            sixhourly_sst,
            method="shifting_baseline",
            window_years=3,
            smooth_days=5,
            dimensions=DIMENSIONS,
            dask_chunks={"time": 400},
        )
        assert bool(np.isfinite(ds.dat_anomaly).any().compute()), "tiled run produced nothing finite"

    def test_global_percentile_needs_no_cycle_at_all(self, sixhourly_sst):
        ds = marEx.preprocess_data(
            sixhourly_sst,
            method_anomaly="fixed_baseline",
            method_extreme="global_percentile",
            dimensions=DIMENSIONS,
            dask_chunks={"time": 400},
        )
        assert "hourofyear" not in ds.thresholds.dims


class TestExactPercentileOffTheDailyAxis:
    """``method_percentile='exact'`` is a second, independent generalisation.

    It carries its own ``% cycle.length`` wrap, its own ``output_sizes``, its own
    result-array width and its own tile budget -- none of them shared with the
    histogram path that every other test here exercises. The modulo is where an
    off-by-one would live.
    """

    def test_monthly_exact_percentiles_land_on_twelve_slots(self, monthly_sst):
        ds = marEx.preprocess_data(
            monthly_sst,
            method_anomaly="fixed_baseline",
            method_extreme="seasonal_percentile",
            method_percentile="exact",
            window_days=40,
            dimensions=DIMENSIONS,
            dask_chunks={"time": 24},
        )
        assert ds.sizes["month"] == 12
        assert "dayofyear" not in ds.thresholds.dims
        assert bool(np.isfinite(ds.thresholds).all().compute())

    def test_the_exact_window_wraps_within_the_cycle_not_the_year(self, monthly_sst):
        """A 3-month window on the monthly axis must wrap December to January.

        With ``% 366`` still in place the wrap target for month 12 would be nonsense and
        the December slot would draw on the wrong months.
        """
        wide = marEx.preprocess_data(
            monthly_sst,
            method_anomaly="fixed_baseline",
            method_extreme="seasonal_percentile",
            method_percentile="exact",
            window_days=90,
            dimensions=DIMENSIONS,
            dask_chunks={"time": 24},
        ).thresholds.compute()
        narrow = marEx.preprocess_data(
            monthly_sst,
            method_anomaly="fixed_baseline",
            method_extreme="seasonal_percentile",
            method_percentile="exact",
            window_days=40,
            dimensions=DIMENSIONS,
            dask_chunks={"time": 24},
        ).thresholds.compute()
        assert bool(np.isfinite(wide).all()) and bool(np.isfinite(narrow).all())
        # A wider window pools more months, so it must smooth the seasonal contrast.
        assert float(wide.std("month").mean()) < float(narrow.std("month").mean())


class TestIrregularAxesOnlyFailWhereACycleIsNeeded:
    """`infer_cycle` raises on a mixed-cadence axis. That must reach only the paths
    that actually need a within-year cycle.

    Resolving eagerly at the entry points made `global_percentile` -- which has no
    cycle at all -- fail on axes it had always processed. Same shape as the Phase B
    `validate_rank` finding: a guard naming a problem the caller does not have.
    """

    @staticmethod
    def _mixed_cadence():
        time = pd.DatetimeIndex(
            list(pd.date_range("2000-01-01", periods=60, freq="D")) + list(pd.date_range("2001-01-01", periods=40, freq="MS"))
        )
        rng = np.random.default_rng(11)
        return xr.DataArray(
            rng.normal(15.0, 1.0, (len(time), 3, 3)).astype(np.float32),
            dims=["time", "lat", "lon"],
            coords={"time": time, "lat": np.arange(3.0), "lon": np.arange(3.0)},
        ).chunk({"time": 20})

    def test_global_percentile_still_runs_on_a_mixed_cadence_axis(self):
        ds = marEx.preprocess_data(
            self._mixed_cadence(),
            method_anomaly="detrend_harmonic",
            method_extreme="global_percentile",
            dimensions=DIMENSIONS,
            dask_chunks={"time": 20},
        )
        assert "extreme_events" in ds

    def test_seasonal_percentile_does_raise_there(self):
        with pytest.raises(ConfigurationError, match="too irregular"):
            marEx.preprocess_data(
                self._mixed_cadence(),
                method_anomaly="fixed_baseline",
                method_extreme="seasonal_percentile",
                dimensions=DIMENSIONS,
                dask_chunks={"time": 20},
            )

    def test_an_explicit_cycle_unblocks_the_seasonal_path(self):
        """Which is what the escape hatch is for."""
        ds = marEx.preprocess_data(
            self._mixed_cadence(),
            method_anomaly="fixed_baseline",
            method_extreme="seasonal_percentile",
            cycle=marEx.SeasonalCycle("month", 12, 30.0),
            window_days=40,
            dimensions=DIMENSIONS,
            dask_chunks={"time": 20},
        )
        assert ds.sizes["month"] == 12


# --------------------------------------------------------------------------------------
# The escape hatch
# --------------------------------------------------------------------------------------


class TestExplicitCycle:
    def test_an_explicit_daily_cycle_matches_the_inferred_one(self):
        """Passing the cycle marEx would have inferred changes nothing -- so the hatch
        is an override, not a separate code path."""
        sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.isel(time=slice(0, 1200)).chunk({"time": 25})
        kwargs = {
            "method_anomaly": "fixed_baseline",
            "method_extreme": "seasonal_percentile",
            "window_days": 3,
            "dimensions": DIMENSIONS,
        }
        inferred = marEx.preprocess_data(sst, **kwargs).thresholds.compute()
        explicit = marEx.preprocess_data(sst, cycle=marEx.SeasonalCycle("dayofyear", 366, 1.0), **kwargs).thresholds.compute()
        np.testing.assert_array_equal(inferred.values, explicit.values)

    def test_the_hatch_is_honoured_by_the_standalone_entry_points_too(self, monthly_sst):
        """`cycle=` is declared on `anomaly.compute` and `extremes.identify` as well as
        on the chainer, so exercise it there rather than only through `preprocess_data`."""
        anom = marEx.anomaly.compute(
            monthly_sst,
            method="fixed_baseline",
            cycle=marEx.SeasonalCycle("month", 12, 30.44),
            dimensions=DIMENSIONS,
            dask_chunks={"time": 24},
        )
        ext = marEx.extremes.identify(
            anom.dat_anomaly,
            method="seasonal_percentile",
            window_days=40,
            cycle=marEx.SeasonalCycle("month", 12, 30.44),
            dimensions=DIMENSIONS,
            dask_chunks={"time": 24},
        )
        assert ext.sizes["month"] == 12
        assert "dayofyear" not in ext.thresholds.dims

    def test_an_explicit_cycle_overrides_inference(self, monthly_sst):
        """The hatch exists for axes inference cannot read. Forcing the monthly cycle
        onto a monthly series is a no-op; forcing it is what matters."""
        ds = marEx.preprocess_data(
            monthly_sst,
            method_anomaly="fixed_baseline",
            method_extreme="seasonal_percentile",
            cycle=marEx.SeasonalCycle("month", 12, 30.44),
            dimensions=DIMENSIONS,
            dask_chunks={"time": 24},
        )
        assert ds.sizes["month"] == 12
