"""
The fixed baselines smooth their day-of-year climatology circularly (D-138).

Hobday et al. (2016) smooth the climatology with a moving average that wraps the year.
These tests pin the wrap by SUPPORT (where a spike lands), not by a float tolerance,
and pin ``smooth_days=1`` as an exact off switch against an independent unsmoothed
reference.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.anomaly.fixed_baseline import (
    _compute_anomaly_detrend_fixed_baseline,
    _compute_anomaly_fixed_baseline,
    _smooth_climatology_circular,
)
from marEx.core.time_axis import DAILY_CYCLE, SeasonalCycle
from marEx.exceptions import ConfigurationError


def _clim_with_spike(slot, length=366, n_x=2):
    data = np.zeros((length, n_x), dtype=np.float32)
    data[slot - 1, :] = 1.0
    return xr.DataArray(data, dims=("dayofyear", "x"), coords={"dayofyear": np.arange(1, length + 1)}).chunk({"dayofyear": -1})


def _daily_field(start="2001-01-01", end="2008-12-31", seed=0):
    time = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    seasonal = np.sin(2 * np.pi * time.dayofyear.values / 365.25)[:, None, None]
    data = (seasonal + rng.normal(0.0, 0.5, (len(time), 2, 3))).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
    ).chunk({"time": 200})


class TestCircularSmoothing:
    @pytest.mark.parametrize("slot", [1, 366])
    def test_a_spike_at_the_year_edge_wraps_onto_the_other_end(self, slot):
        smoothed = _smooth_climatology_circular(_clim_with_spike(slot), DAILY_CYCLE, 21).compute()
        support = set((np.flatnonzero(smoothed.values[:, 0] != 0) + 1).tolist())
        offsets = range(-10, 11)
        expected = {(slot - 1 + k) % 366 + 1 for k in offsets}
        assert support == expected
        assert len(support) == 21

    def test_mass_is_conserved(self):
        smoothed = _smooth_climatology_circular(_clim_with_spike(200), DAILY_CYCLE, 21).compute()
        assert float(smoothed.sum("dayofyear")[0]) == pytest.approx(1.0, rel=1e-6)

    def test_layout_labels_and_dtype_are_kept(self):
        smoothed = _smooth_climatology_circular(_clim_with_spike(50), DAILY_CYCLE, 21)
        assert smoothed.chunks[0] == (366,)
        assert smoothed.dtype == np.float32
        np.testing.assert_array_equal(smoothed["dayofyear"].values, np.arange(1, 367))

    def test_one_day_is_the_identity(self):
        clim = _clim_with_spike(50)
        assert _smooth_climatology_circular(clim, DAILY_CYCLE, 1) is clim

    def test_a_window_spanning_the_whole_cycle_is_rejected(self):
        with pytest.raises(ConfigurationError, match="spans 366 of the 366 rows"):
            _smooth_climatology_circular(_clim_with_spike(50), DAILY_CYCLE, 366)

    def test_monthly_cycle_is_left_unsmoothed_by_the_default(self):
        """21 days is under one ~30.4-day slot, so the default does nothing on a month axis."""
        monthly = SeasonalCycle("month", 12, 30.0)
        clim = _clim_with_spike(3, length=12).rename(dayofyear="month")
        assert _smooth_climatology_circular(clim, monthly, 21) is clim

    def test_subdaily_smooths_across_days_at_a_fixed_hour(self):
        """Hourly: the window spans 21 DAYS of the same hour, never neighbouring hours,
        or the diurnal cycle would be averaged out of the climatology."""
        hourly = SeasonalCycle("hourofyear", 366 * 24, 1 / 24)
        slot = 5000
        clim = _clim_with_spike(slot, length=366 * 24).rename(dayofyear="hourofyear")
        smoothed = _smooth_climatology_circular(clim, hourly, 21).compute()
        support = np.flatnonzero(smoothed.values[:, 0] != 0)
        assert len(support) == 21
        assert set(((support - (slot - 1)) % 24).tolist()) == {0}

    def test_subdaily_wraps_at_the_same_hour(self):
        hourly = SeasonalCycle("hourofyear", 366 * 4, 0.25)
        slot = 3  # day 1, third six-hourly step
        clim = _clim_with_spike(slot, length=366 * 4).rename(dayofyear="hourofyear")
        smoothed = _smooth_climatology_circular(clim, hourly, 21).compute()
        days = set((np.flatnonzero(smoothed.values[:, 0] != 0) // 4 + 1).tolist())
        assert days == set(range(1, 12)) | set(range(357, 367))


class TestFixedBaselineAnomaly:
    def test_smooth_days_one_matches_an_independent_unsmoothed_reference(self):
        da = _daily_field()
        result = _compute_anomaly_fixed_baseline(da, smooth_days=1).dat_anomaly.compute()
        clim = da.groupby("time.dayofyear").mean("time").compute()
        reference = (da.groupby("time.dayofyear") - clim).astype(np.float32).drop_vars("dayofyear").compute()
        np.testing.assert_allclose(result.values, reference.values, rtol=0, atol=1e-6)

    def test_default_smooths(self):
        da = _daily_field()
        raw = _compute_anomaly_fixed_baseline(da, smooth_days=1).dat_anomaly.compute()
        smoothed = _compute_anomaly_fixed_baseline(da).dat_anomaly.compute()
        assert not np.array_equal(raw.values, smoothed.values)
        # Smoothing removes the noise the per-day mean absorbs, so the anomaly variance rises
        # towards the true 0.5**2.
        assert float(smoothed.std()) > float(raw.std())
        assert raw.chunks == smoothed.chunks

    def test_detrend_fixed_baseline_passes_smooth_days_through(self):
        da = _daily_field()
        raw = _compute_anomaly_detrend_fixed_baseline(da, smooth_days=1).dat_anomaly.compute()
        smoothed = _compute_anomaly_detrend_fixed_baseline(da, smooth_days=21).dat_anomaly.compute()
        assert not np.array_equal(raw.values, smoothed.values)

    @pytest.mark.parametrize("method", ["fixed_baseline", "detrend_fixed_baseline"])
    def test_smooth_days_is_recorded(self, method):
        ds = marEx.anomaly.compute(_daily_field(), method=method, smooth_days=15)
        assert ds.attrs["smooth_days"] == 15
        assert any("15-day circular" in step for step in ds.attrs["preprocessing_steps"])


class TestNaNPattern:
    """D-138 add. 2: smoothing never changes WHICH slots are valid, only their values."""

    def test_a_seasonal_nan_cell_keeps_exactly_its_valid_days(self):
        da = _daily_field()
        sea_ice = da.time.dt.dayofyear <= 90
        da = da.where(~(sea_ice & (da.lat == 0.0) & (da.lon == 0.0)))
        raw = _compute_anomaly_fixed_baseline(da, smooth_days=1).dat_anomaly.compute()
        smoothed = _compute_anomaly_fixed_baseline(da, smooth_days=21).dat_anomaly.compute()
        np.testing.assert_array_equal(np.isnan(raw.values), np.isnan(smoothed.values))
        assert int(np.isnan(raw.values).sum()) > 0

    def test_without_nan_the_skipna_average_equals_the_full_window_average(self):
        rng = np.random.default_rng(3)
        clim = xr.DataArray(
            rng.normal(size=(366, 4)).astype(np.float32), dims=("dayofyear", "x"), coords={"dayofyear": np.arange(1, 367)}
        ).chunk({"dayofyear": -1})
        smoothed = _smooth_climatology_circular(clim, DAILY_CYCLE, 21).compute().values
        padded = np.concatenate([clim.values[-10:], clim.values, clim.values[:10]])
        reference = np.stack([padded[i : i + 21].mean(axis=0) for i in range(366)])
        np.testing.assert_allclose(smoothed, reference, rtol=0, atol=1e-6)
