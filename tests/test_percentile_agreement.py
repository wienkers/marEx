"""How far the histogram (``approximate``) percentile may sit from the ``exact`` one.

The two seasonal paths do NOT target the same order statistic, and that is documented
rather than fixed. ``exact`` is ``np.nanpercentile``'s default linear rule
(rank ``q * (n - 1)``). The 2-D histogram kernel returns, to within its bin-centre
interpolation, the sample of rank ``floor(q * n) + 1`` (the lower tail is the mirrored
upper estimator). That equals numpy's ``higher`` (upper) / ``lower`` (lower) only
while ``frac(q * n) <= q``, which always holds for the percentiles tested here. At
p=90.9 with n=110 it does not, and the gap to ``higher`` is ~50 bins.
In a day-of-year window of a few hundred samples the tail samples sit several bins
apart, so against the shipped ``exact`` the two can differ by tens of bins. Against
the matching rank rule they must agree to 1.5 bins: half a bin from the centre
offset plus one bin of interpolation span.

The 1-D (global) path pools the whole series, where neighbouring samples are far
closer than a bin, so every rank rule coincides and it must track the shipped
``exact`` to about one bin.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.core.time_axis import resolve_cycle
from marEx.extremes.base import resolve_bin_spec

DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}
PRECISION = 0.01
MAX_ANOMALY = 6.0
# Centre interpolation spans adjacent bin centres, and the sample can sit anywhere in
# the upper one: half a bin plus one bin. A hair of float32 slack on top.
SEASONAL_BOUND_BINS = 1.5 + 1e-3


def _anomaly(n_years=10, seed=7):
    t = pd.date_range("1990-01-01", periods=int(365.25 * n_years), freq="D")
    data = np.random.default_rng(seed).normal(0.0, 1.0, size=(t.size, 4, 5)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": t, "lat": np.arange(4.0), "lon": np.arange(5.0)},
        name="dat_anomaly",
    ).chunk({"time": -1})


def _identify(da, method, method_percentile, percentile, tail):
    kw = {}
    if method_percentile == "approximate":
        kw = {"precision": PRECISION, "max_anomaly": MAX_ANOMALY}
        if method == "seasonal_percentile":
            kw["window_spatial"] = 1  # the samples exact uses; the 5x5 default is another statistic
    return marEx.extremes.identify(
        da,
        method=method,
        method_percentile=method_percentile,
        threshold_percentile=percentile,
        tail=tail,
        dimensions=DIMENSIONS,
        quiet=True,
        **kw,
    ).compute()


def _rank_rule_reference(da, percentile, rank_rule, window_days=11):
    """Per-slot numpy percentile under `rank_rule`, over the same day-of-year window samples the
    exact path pools. Plain numpy in the test process: monkeypatching numpy instead is invisible
    to tasks that a session-wide distributed client runs in worker processes."""
    cycle = resolve_cycle(da, "time")
    slots = cycle.index_of(da["time"]).values
    half = cycle.window_steps(window_days) // 2
    values = da.values  # (time, lat, lon)
    out = np.full((cycle.length,) + values.shape[1:], np.nan)
    for slot in range(1, cycle.length + 1):
        targets = {((slot - 1 + k) % cycle.length) + 1 for k in range(-half, half + 1)}
        members = np.isin(slots, list(targets))
        if members.any():
            out[slot - 1] = np.percentile(values[members], percentile, axis=0, method=rank_rule)
    return out


@pytest.mark.parametrize("percentile,tail", [(90, "upper"), (95, "upper"), (5, "lower"), (10, "lower")])
def test_seasonal_approximate_matches_the_rank_rule_for_whole_percentiles(percentile, tail):
    da = _anomaly()
    approx = _identify(da, "seasonal_percentile", "approximate", percentile, tail)
    rank_rule = "higher" if tail == "upper" else "lower"
    reference = _rank_rule_reference(da.compute(), percentile, rank_rule)

    got = approx.thresholds.transpose("dayofyear", "lat", "lon").values
    diff_bins = np.abs(got - reference) / PRECISION
    assert np.isfinite(diff_bins).all()
    assert diff_bins.max() <= SEASONAL_BOUND_BINS, f"max {diff_bins.max():.3f} bins from numpy '{rank_rule}'"


@pytest.mark.parametrize("percentile,tail", [(90, "upper"), (95, "upper"), (5, "lower"), (10, "lower")])
def test_seasonal_gap_to_shipped_exact_is_a_rank_gap_not_precision(percentile, tail):
    """The documented gap is real: against numpy's linear rule the seasonal paths sit
    several bins apart on average, with the approximate threshold outward. If this
    ever collapses to within a bin the convention changed and the documented gap is stale."""
    da = _anomaly()
    approx = _identify(da, "seasonal_percentile", "approximate", percentile, tail)
    exact = _identify(da, "seasonal_percentile", "exact", percentile, tail)
    signed = (approx.thresholds - exact.thresholds).values / PRECISION
    outward = signed if tail == "upper" else -signed
    assert np.abs(signed).max() > 5
    assert outward.mean() > 0.5


@pytest.mark.parametrize("percentile,tail", [(90, "upper"), (95, "upper"), (5, "lower"), (10, "lower")])
def test_global_approximate_tracks_shipped_exact_to_about_a_bin(percentile, tail):
    """Holds only while the tail is dense. p99 is left out on purpose: at 10 years only
    ~37 samples lie above it per cell, the rank rules part again, and the gap was
    measured at up to 7.8 bins depending on the seed (1.09 at 20 years) -- the seasonal story in miniature."""
    da = _anomaly()
    approx = _identify(da, "global_percentile", "approximate", percentile, tail)
    exact = _identify(da, "global_percentile", "exact", percentile, tail)
    diff_bins = np.abs((approx.thresholds - exact.thresholds).values) / PRECISION
    assert diff_bins.max() <= 1.25, f"max {diff_bins.max():.3f} bins"


class _CaptureMarExLogs(logging.Handler):
    """marEx WARNINGs; ``caplog`` cannot see them (the ``marEx`` logger does not propagate)."""

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


class TestCoarseDerivedBins:
    """One outlier sets the derived range and so every bin: warn, never change the value."""

    def _field(self, heavy):
        rng = np.random.default_rng(3)
        data = rng.standard_t(3, size=(3000, 4, 5)) * 0.6 if heavy else rng.normal(0.0, 1.0, size=(3000, 4, 5))
        return xr.DataArray(data.astype(np.float32), dims=("time", "lat", "lon"), name="dat_anomaly").chunk({"time": -1})

    def test_a_heavy_tailed_field_warns_and_keeps_the_derived_bins(self):
        da = self._field(heavy=True)
        with _CaptureMarExLogs() as captured:
            precision, max_anomaly = resolve_bin_spec(da, None, None, 1000)
        assert any("Derived histogram bins are coarse" in m for m in captured.messages), captured.messages
        # The range is the upper tail's own extreme, not max|anomaly|.
        observed = float(da.max())
        assert max_anomaly == pytest.approx(observed)
        assert precision == pytest.approx(2 * observed / 1000)

    def test_a_gaussian_field_is_silent(self):
        with _CaptureMarExLogs() as captured:
            resolve_bin_spec(self._field(heavy=False), None, None, 1000)
        assert not any("coarse" in m for m in captured.messages), captured.messages

    def test_an_explicit_precision_is_never_second_guessed(self):
        with _CaptureMarExLogs() as captured:
            resolve_bin_spec(self._field(heavy=True), 0.5, None, 1000)
        assert not any("coarse" in m for m in captured.messages), captured.messages
