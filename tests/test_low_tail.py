"""Low-tail extremes: ``tail='lower'``.

``tail`` selects which side of the distribution counts as extreme. The threshold
is the ``threshold_percentile``-th percentile either way -- only the comparison
flips -- so the coldest 5 % is ``threshold_percentile=5, tail='lower'``.

The strongest gate here is the **symmetry oracle**: thresholding ``-x`` at the
upper ``100-p`` th percentile must give the negation of thresholding ``x`` at the
lower ``p`` th. Negate-and-reuse is used as the ORACLE only. The implementation
computes ``q = p/100`` directly and compares with ``<=``: negating the data would
add a full-size lazy op and would make the returned ``thresholds`` mean something
other than what the caller asked for.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError

DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}


def _anomaly(n_time=2000, n_y=4, n_x=5, seed=42, scale=1.0):
    """Daily, mean-zero, dask-backed anomalies -- the shape both stages expect."""
    rng = np.random.default_rng(seed)
    data = (rng.normal(0.0, scale, size=(n_time, n_y, n_x))).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=n_time, freq="D"),
            "lat": np.arange(n_y, dtype=np.float32),
            "lon": np.arange(n_x, dtype=np.float32),
        },
        name="dat_anomaly",
    )
    return da.chunk({"time": -1, "lat": 2, "lon": 5})


class TestSymmetryOracle:
    """thresholds(-x, upper, 100-p) == -thresholds(x, lower, p)."""

    @pytest.mark.parametrize("method", ["global_percentile", "seasonal_percentile"])
    @pytest.mark.parametrize("percentile", [5, 10])
    def test_thresholds_and_masks_mirror(self, method, percentile):
        da = _anomaly()
        kw = {"method": method, "dimensions": DIMENSIONS, "dask_chunks": {"time": 100}}
        if method == "seasonal_percentile":
            kw["window_days"] = 11

        low = marEx.extremes.identify(da, threshold_percentile=percentile, tail="lower", **kw).compute()
        high = marEx.extremes.identify(-da, threshold_percentile=100 - percentile, tail="upper", **kw).compute()

        # Bins are exactly symmetric about zero, so negating the data maps every
        # sample onto the mirrored bin exactly and the two estimators see the same
        # counts in mirrored order. What does NOT mirror exactly is the cumulative
        # search's tie handling: `idx_upper = (cumsum <= q*N).sum()` is one-sided, so
        # the mirrored problem resolves ties on the other side of a bin boundary.
        #
        # The two paths therefore get different tolerances, and the difference is the
        # sample count per estimate, not the tail. The 1-D path pools the whole series
        # into its bins, so the bin holding the quantile is well populated and the
        # interpolation fraction is meaningful: measured, it mirrors to within one bin.
        # The 2-D path pools only a per-day-of-year window (~60 samples here), so its
        # bins hold 0 or 1 sample and the fraction degenerates -- measured, up to three
        # bin widths, with a mean offset of about one. Tightening this would mean
        # changing the 2-D path's centre-based interpolation, which is empirically
        # justified for the upper tail and must not move.
        atol = 0.01 if method == "global_percentile" else 0.04
        np.testing.assert_allclose(low.thresholds.values, -high.thresholds.values, atol=atol)
        offset = np.nanmean(low.thresholds.values + high.thresholds.values)
        assert abs(offset) <= 0.01, f"systematic mirror offset {offset:.4f} exceeds one bin"
        # The masks are what a user acts on: they must agree except where a sample
        # sits between two thresholds that differ by less than a bin.
        disagree = (low.extreme_events.values != high.extreme_events.values).mean()
        assert disagree < 0.005, f"masks disagree on {disagree:.4%} of points"

    def test_exact_percentile_mirrors_too(self):
        """The exact path shares none of the histogram machinery -- gate it separately."""
        da = _anomaly(n_time=400)
        kw = {
            "method": "global_percentile",
            "method_percentile": "exact",
            "dimensions": DIMENSIONS,
            "dask_chunks": {"time": 100},
        }
        low = marEx.extremes.identify(da, threshold_percentile=5, tail="lower", **kw).compute()
        high = marEx.extremes.identify(-da, threshold_percentile=95, tail="upper", **kw).compute()
        np.testing.assert_allclose(low.thresholds.values, -high.thresholds.values, rtol=1e-6)
        assert np.array_equal(low.extreme_events.values, high.extreme_events.values)


class TestAnalyticRecovery:
    def test_lower_tail_recovers_the_gaussian_5th_percentile(self):
        """A synthetic N(0,1) has a known 5th percentile; recover it to a bin width."""
        da = _anomaly(n_time=20000, n_y=2, n_x=2)
        ds = marEx.extremes.identify(
            da, method="global_percentile", threshold_percentile=5, tail="lower", dimensions=DIMENSIONS
        ).compute()
        analytic = -1.6448536269514722  # scipy.stats.norm.ppf(0.05)
        np.testing.assert_allclose(ds.thresholds.values, analytic, atol=0.05)

    def test_lower_tail_flags_about_the_requested_fraction(self):
        da = _anomaly(n_time=20000, n_y=2, n_x=2)
        ds = marEx.extremes.identify(
            da, method="global_percentile", threshold_percentile=5, tail="lower", dimensions=DIMENSIONS
        ).compute()
        assert ds.extreme_events.values.mean() == pytest.approx(0.05, abs=0.01)

    def test_upper_and_lower_flag_opposite_ends(self):
        da = _anomaly(n_time=5000, n_y=2, n_x=2)
        kw = {"method": "global_percentile", "dimensions": DIMENSIONS}
        low = marEx.extremes.identify(da, threshold_percentile=5, tail="lower", **kw).compute()
        high = marEx.extremes.identify(da, threshold_percentile=95, tail="upper", **kw).compute()
        # Disjoint, and the flagged values sit on opposite sides of zero.
        assert not (low.extreme_events.values & high.extreme_events.values).any()
        assert da.values[low.extreme_events.values].max() < 0 < da.values[high.extreme_events.values].min()


class TestGuardRail:
    def test_a_constant_zero_cell_is_not_extreme_in_either_tail(self):
        """The guard rail exists for sea ice: a flat-zero anomaly must never flag.

        Under the legacy bins this was enforced only upward. The symmetric bins make
        the low tail reachable, so the guard has to be mirrored or every masked cell
        becomes a permanent cold extreme.
        """
        da = _anomaly(n_time=2000).load()
        da[:, 0, 0] = 0.0
        da = da.chunk({"time": -1, "lat": 2, "lon": 5})
        for tail, percentile in [("lower", 5), ("upper", 95)]:
            with pytest.warns(UserWarning, match="constant anomaly"):
                ds = marEx.extremes.identify(
                    da, method="global_percentile", threshold_percentile=percentile, tail=tail, dimensions=DIMENSIONS
                ).compute()
            assert not ds.extreme_events.isel(lat=0, lon=0).values.any(), f"{tail} tail flagged a constant-zero cell"


class TestParameterHandling:
    def test_low_percentiles_are_now_accepted_by_the_approximate_method(self):
        """The ``<60`` rejection was correct under asymmetric bins and is now obsolete.

        Every negative value shared one bin, so any percentile landing in it was
        undefined by construction. Symmetric bins resolve a low percentile at exactly
        the same precision as a high one.
        """
        da = _anomaly(n_time=4000, n_y=2, n_x=2)
        ds = marEx.extremes.identify(
            da, method="global_percentile", threshold_percentile=25, tail="lower", dimensions=DIMENSIONS
        ).compute()
        analytic = -0.6744897501960817  # norm.ppf(0.25)
        np.testing.assert_allclose(ds.thresholds.values, analytic, atol=0.05)

    def test_an_unknown_tail_is_rejected(self):
        da = _anomaly(n_time=400, n_y=2, n_x=2)
        with pytest.raises(ConfigurationError, match="Unknown tail"):
            marEx.extremes.identify(da, method="global_percentile", tail="both", dimensions=DIMENSIONS)

    def test_tail_is_recorded_in_the_attributes(self):
        da = _anomaly(n_time=400, n_y=2, n_x=2)
        ds = marEx.extremes.identify(da, method="global_percentile", threshold_percentile=5, tail="lower", dimensions=DIMENSIONS)
        assert ds.attrs["tail"] == "lower"

    def test_default_tail_is_upper(self):
        da = _anomaly(n_time=400, n_y=2, n_x=2)
        ds = marEx.extremes.identify(da, method="global_percentile", dimensions=DIMENSIONS)
        assert ds.attrs["tail"] == "upper"


class _CaptureMarExLogs(logging.Handler):
    """Collect marEx WARNINGs.

    ``caplog`` cannot see them: ``configure_logging`` sets ``propagate = False`` on
    the ``marEx`` root logger, so records never reach pytest's handler. Attaching
    one directly is the only way to observe them.
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


class TestSampleCountWarning:
    def test_the_sparse_sample_warning_uses_the_tail_it_was_asked_about(self):
        """``N_samples * (1 - p/100)`` is the wrong count for a low tail.

        With ``threshold_percentile=5`` only 5 % of samples fall below the threshold,
        so the complement would report 95 % and keep the warning silent in exactly
        the sparse case it exists for.
        """
        da = _anomaly(n_time=730, n_y=2, n_x=2)  # 2 years x 11-day window = 22 slots
        with _CaptureMarExLogs() as captured:
            marEx.extremes.identify(da, method="seasonal_percentile", threshold_percentile=5, tail="lower", dimensions=DIMENSIONS)
        messages = captured.messages
        assert any("Not enough samples" in m for m in messages), messages
        # 5 % of 550 samples, not 95 % of them.
        assert any("27.5 < 50" in m for m in messages), messages
