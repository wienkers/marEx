"""Percentile accuracy against an independent reference: the rank rule, numpy, the analytic Gaussian, the calendar window.

The approximate (histogram) and exact paths do NOT target the same order statistic, and that is
documented rather than fixed. ``exact`` is ``np.nanpercentile``'s default linear rule (rank
``q * (n - 1)``). The 2-D histogram kernel returns, to within its bin-centre interpolation, the
sample of rank ``floor(q * n) + 1`` (the lower tail is the mirrored upper estimator). That equals
numpy's ``higher`` (upper) / ``lower`` (lower) only while ``frac(q * n) <= q``, which holds for the
percentiles tested here. In a day-of-year window of a few hundred samples the tail samples sit
several bins apart, so against ``exact`` the two can differ by tens of bins; against the matching
rank rule they agree to 1.5 bins (half a bin from the centre offset plus one bin of interpolation
span). The 1-D (global) path pools the whole series, where every rank rule coincides, and tracks
``exact`` to about one bin.

``tail='lower'`` flips only the comparison: the coldest 5 % is ``threshold_percentile=5,
tail='lower'``. The symmetry oracle is ``thresholds(-x, upper, 100-p) == -thresholds(x, lower, p)``;
negate-and-reuse is the ORACLE only, the implementation computes ``q = p/100`` directly.

A (near-)constant anomaly is never extreme in either tail: the approximate path clamps its
threshold onto a guard rail one bin from zero, the exact path keeps it strictly on its own side of
zero, and every cell already past zero keeps its threshold and its events bit for bit.

``window_days`` is measured in cycle slots, not data steps: 5-day data still sits on the 366-slot
``dayofyear`` cycle, so an 11-day window pools 11 calendar days, and an exact window wraps within
the cycle, not the year.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.core.time_axis import resolve_cycle
from marEx.exceptions import ConfigurationError
from marEx.extremes import histogram as H
from marEx.extremes.base import identify_extremes

from ..conftest import assert_no_failures
from .conftest import DATA_DIR, GRID_DIMS, MESH_COORDS, MESH_DIMS, anomaly_fixture, check

PRECISION = 0.01
MAX_ANOMALY = 6.0
# Centre interpolation spans adjacent bin centres, and the sample can sit anywhere in the upper
# one: half a bin plus one bin. A hair of float32 slack on top.
SEASONAL_BOUND_BINS = 1.5 + 1e-3


def _gaussian(n_time=2000, n_y=4, n_x=5):
    """Daily, mean-zero, dask-backed anomalies -- the shape both stages expect."""
    return anomaly_fixture(n_time=n_time, n_y=n_y, n_x=n_x, seed=42, chunks={"time": -1, "lat": 2, "lon": 5}, name="dat_anomaly")


# ── approximate against exact and the rank rule ───────────────────────


def _decade(n_years=10, seed=7):
    t = pd.date_range("1990-01-01", periods=int(365.25 * n_years), freq="D")
    data = np.random.default_rng(seed).normal(0.0, 1.0, size=(t.size, 4, 5)).astype(np.float32)
    return xr.DataArray(
        data, dims=("time", "lat", "lon"), coords={"time": t, "lat": np.arange(4.0), "lon": np.arange(5.0)}, name="dat_anomaly"
    ).chunk({"time": -1})


def _identify(da, method, method_percentile, percentile, tail, window_days=11):
    kw = {}
    if method_percentile == "approximate":
        kw = {"precision": PRECISION, "max_anomaly": MAX_ANOMALY}
        if method == "seasonal_percentile":
            kw["window_spatial"] = 1  # the samples exact uses; the 5x5 default is another statistic
    if method == "seasonal_percentile":
        kw["window_days"] = window_days
    return marEx.extremes.identify(
        da,
        method=method,
        method_percentile=method_percentile,
        threshold_percentile=percentile,
        tail=tail,
        dimensions=GRID_DIMS,
        quiet=True,
        **kw,
    ).compute()


def _rank_rule_reference(da, percentile, rank_rule, window_days):
    """Per-slot numpy percentile under ``rank_rule``, over the day-of-year window samples the exact path pools.

    Plain numpy in the test process: monkeypatching numpy instead is invisible to tasks that a
    distributed client runs in worker processes.
    """
    cycle = resolve_cycle(da, "time")
    slots = cycle.index_of(da["time"]).values
    half = cycle.window_steps(window_days) // 2
    values = da.values
    out = np.full((cycle.length,) + values.shape[1:], np.nan)
    for slot in range(1, cycle.length + 1):
        targets = {((slot - 1 + k) % cycle.length) + 1 for k in range(-half, half + 1)}
        members = np.isin(slots, list(targets))
        if members.any():
            out[slot - 1] = np.percentile(values[members], percentile, axis=0, method=rank_rule)
    return out


def test_global_approximate_tracks_exact():
    """Holds only while the tail is dense. p99 is left out on purpose: at 10 years only ~37 samples
    lie above it per cell, the rank rules part again, and the gap was measured at up to 7.8 bins
    depending on the seed (1.09 at 20 years)."""
    da = _decade()
    failures = []
    for percentile, tail in ((90, "upper"), (95, "upper"), (5, "lower"), (10, "lower")):

        def row(percentile=percentile, tail=tail):
            approx = _identify(da, "global_percentile", "approximate", percentile, tail)
            exact = _identify(da, "global_percentile", "exact", percentile, tail)
            diff_bins = np.abs((approx.thresholds - exact.thresholds).values) / PRECISION
            assert diff_bins.max() <= 1.25, f"max {diff_bins.max():.3f} bins"

        check(failures, f"p{percentile} {tail}", row)
    assert_no_failures(failures)


@pytest.mark.parametrize("tail", ["upper", "lower"])
def test_seasonal_approximate_vs_rank_rule_and_exact(tail):
    """Within 1.5 bins of the matching rank rule at every slot and cell, and a real rank gap to ``exact``.

    The rank-rule bound is checked at windows of 11 and 41 days. Against numpy's linear rule (the
    shipped ``exact``) the seasonal paths sit several bins apart, with the approximate threshold
    outward: if that ever collapses to within a bin the convention changed and the documented gap
    is stale. The 11-day approximate run is shared by both checks.
    """
    da = _decade()
    values = da.compute()
    rank_rule = "higher" if tail == "upper" else "lower"
    failures = []
    for percentile in (90, 95) if tail == "upper" else (5, 10):
        for window_days in (11, 41):

            def row(percentile=percentile, window_days=window_days):
                approx = _identify(da, "seasonal_percentile", "approximate", percentile, tail, window_days)
                reference = _rank_rule_reference(values, percentile, rank_rule, window_days)
                diff_bins = np.abs(approx.thresholds.transpose("dayofyear", "lat", "lon").values - reference) / PRECISION
                assert np.isfinite(diff_bins).all()
                assert diff_bins.max() <= SEASONAL_BOUND_BINS, f"max {diff_bins.max():.3f} bins from numpy '{rank_rule}'"
                if window_days == 11:
                    exact = _identify(da, "seasonal_percentile", "exact", percentile, tail, window_days)
                    signed = (approx.thresholds - exact.thresholds).values / PRECISION
                    outward = signed if tail == "upper" else -signed
                    assert np.abs(signed).max() > 5, f"gap to exact collapsed to {np.abs(signed).max():.2f} bins"
                    assert outward.mean() > 0.5, f"mean outward gap {outward.mean():.2f} bins"

            check(failures, f"p{percentile} window={window_days}", row)
    assert_no_failures(failures)


# ── the low tail ──────────────────────────────────────────────────────


def test_lower_tail_recovers_gaussian_percentiles():
    """A synthetic N(0,1) has known percentiles: p5 recovered to a bin width and ~5 % flagged; p25 too.

    The ``<60`` rejection was correct under asymmetric bins (every negative value shared one bin)
    and is obsolete: symmetric bins resolve a low percentile at the precision of a high one.
    """
    da = _gaussian(n_time=20000, n_y=2, n_x=2)
    ds = marEx.extremes.identify(
        da, method="global_percentile", threshold_percentile=5, tail="lower", dimensions=GRID_DIMS
    ).compute()
    np.testing.assert_allclose(ds.thresholds.values, -1.6448536269514722, atol=0.05)  # norm.ppf(0.05)
    assert ds.extreme_events.values.mean() == pytest.approx(0.05, abs=0.01)

    da = _gaussian(n_time=4000, n_y=2, n_x=2)
    ds = marEx.extremes.identify(
        da, method="global_percentile", threshold_percentile=25, tail="lower", dimensions=GRID_DIMS
    ).compute()
    np.testing.assert_allclose(ds.thresholds.values, -0.6744897501960817, atol=0.05)  # norm.ppf(0.25)


def test_upper_and_lower_flag_opposite_ends():
    da = _gaussian(n_time=5000, n_y=2, n_x=2)
    kw = {"method": "global_percentile", "dimensions": GRID_DIMS}
    low = marEx.extremes.identify(da, threshold_percentile=5, tail="lower", **kw).compute()
    high = marEx.extremes.identify(da, threshold_percentile=95, tail="upper", **kw).compute()
    # Disjoint, and the flagged values sit on opposite sides of zero.
    assert not (low.extreme_events.values & high.extreme_events.values).any()
    assert da.values[low.extreme_events.values].max() < 0 < da.values[high.extreme_events.values].min()


def test_a_constant_zero_cell_is_not_extreme_in_either_tail(marex_records):
    """The guard rail exists for sea ice: a (near-)constant anomaly must never flag, on either driver.

    Under the legacy bins this was enforced only upward. The symmetric bins make the low tail
    reachable, so the guard has to be mirrored or every masked cell becomes a permanent cold
    extreme. Rows: the 1-D (global) and the 2-D (seasonal, ``window_spatial=1`` so the cells do not
    pool their neighbours) drivers, both tails; each must report the clamp at INFO. Besides the
    flat-zero cell, each tail has a constant cell a hair on its wrong side of zero (-3.05e-05 for
    the upper tail, +3.05e-05 for the lower): on the 2-D path a flat zero lands on the bin centre
    past zero and would not flag even unclamped, so only that cell shows the clamp is applied.
    """
    da = _gaussian().load()
    da[:, 0, 0] = 0.0
    da[:, 0, 1] = np.float32(-3.05e-05)
    da[:, 1, 0] = np.float32(3.05e-05)
    da = da.chunk({"time": -1, "lat": 2, "lon": 5})
    failures = []
    for method, extra in (("global_percentile", {}), ("seasonal_percentile", {"window_spatial": 1})):
        for tail, percentile, tiny in (("lower", 5, (1, 0)), ("upper", 95, (0, 1))):

            def row(method=method, extra=extra, tail=tail, percentile=percentile, tiny=tiny):
                records = marex_records(logging.INFO)
                ds = marEx.extremes.identify(
                    da, method=method, threshold_percentile=percentile, tail=tail, dimensions=GRID_DIMS, **extra
                ).compute()
                messages = [r.getMessage() for r in records]
                assert any("constant" in m and "never extreme" in m for m in messages), "the clamp was not reported"
                for cell in ((0, 0), tiny):
                    assert not ds.extreme_events.isel(lat=cell[0], lon=cell[1]).values.any(), f"flagged the constant cell {cell}"

            check(failures, f"{method} {tail}", row)
    assert_no_failures(failures)


def test_tail_parameter_and_metadata():
    """What the tail argument accepts and what the output says about it (lazy attrs, no compute).

    ``ds.attrs["tail"]`` recorded the tail but ``preprocessing_steps`` did not, so a lower-tail run
    described itself in prose as an upper-tail one. The exact path builds no histogram, so it never
    reached the empty-series check in ``resolve_bin_spec`` and died on a bare ZeroDivisionError.
    """
    small = _gaussian(n_time=400, n_y=2, n_x=2)
    empty = xr.DataArray(
        np.zeros((0, 2, 2), np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2000-01-01", periods=0, freq="D"), "lat": np.arange(2.0), "lon": np.arange(2.0)},
    ).chunk({"time": 1})

    def identify(da=small, **kw):
        return marEx.extremes.identify(da, dimensions=GRID_DIMS, **{"method": "global_percentile", **kw})

    def unknown_tail():
        with pytest.raises(ConfigurationError, match="Unknown tail"):
            identify(tail="both")

    def empty_series(method_percentile):
        with pytest.raises(ConfigurationError, match="anomaly series is empty"):
            identify(empty, method_percentile=method_percentile)

    def default_tail_is_upper():
        assert identify().attrs["tail"] == "upper"

    def tail_is_recorded():
        assert identify(threshold_percentile=5, tail="lower").attrs["tail"] == "lower"

    def lower_seasonal_wording():
        ds = identify(
            _gaussian(n_time=1200, n_y=2, n_x=2), method="seasonal_percentile", threshold_percentile=5, tail="lower", window_days=11
        )
        assert ds.attrs["preprocessing_steps"][-1].startswith("Day-of-year lower-tail thresholds")

    def lower_global_wording():
        ds = identify(threshold_percentile=5, tail="lower")
        assert ds.attrs["preprocessing_steps"][-1] == "Global lower-tail percentile threshold applied to all days"

    def upper_wording_unchanged():
        assert identify().attrs["preprocessing_steps"][-1] == "Global percentile threshold applied to all days"

    rows = {
        "unknown tail": unknown_tail,
        "empty series approximate": lambda: empty_series("approximate"),
        "empty series exact": lambda: empty_series("exact"),
        "default tail": default_tail_is_upper,
        "tail recorded": tail_is_recorded,
        "lower seasonal steps": lower_seasonal_wording,
        "lower global steps": lower_global_wording,
        "upper steps": upper_wording_unchanged,
    }
    failures = []
    for case, row in rows.items():
        check(failures, case, row)
    assert_no_failures(failures)


def test_the_sparse_sample_warning_uses_the_requested_tail(marex_warnings):
    """``N_samples * (1 - p/100)`` is the wrong count for a low tail.

    With ``threshold_percentile=1`` only 1 % of samples fall below the threshold, so the complement
    would report 99 % and keep the warning silent in exactly the sparse case it exists for.
    """
    da = _gaussian(n_time=730, n_y=2, n_x=2)  # 2 years x 11-day window = 22 slots
    with marex_warnings as cap:
        marEx.extremes.identify(da, method="seasonal_percentile", threshold_percentile=1, tail="lower", dimensions=GRID_DIMS)
    assert any("Not enough samples" in m for m in cap.messages), cap.messages
    assert any("5.5 < 10" in m for m in cap.messages), cap.messages  # 1 % of 550 samples, not 99 %


# ── the symmetry oracle ───────────────────────────────────────────────


def test_exact_percentile_mirrors_too():
    """The exact path shares none of the histogram machinery: tested separately."""
    da = _gaussian(n_time=400)
    kw = {"method": "global_percentile", "method_percentile": "exact", "dimensions": GRID_DIMS, "dask_chunks": {"time": 100}}
    low = marEx.extremes.identify(da, threshold_percentile=5, tail="lower", **kw).compute()
    high = marEx.extremes.identify(-da, threshold_percentile=95, tail="upper", **kw).compute()
    np.testing.assert_allclose(low.thresholds.values, -high.thresholds.values, rtol=1e-6)
    assert np.array_equal(low.extreme_events.values, high.extreme_events.values)


def test_seasonal_mirror_with_samples_on_bin_edges():
    """Samples exactly ON a bin edge pin the right-closed digitize of the lower tail, at p 5, 34 and 50.

    Normal data almost never lands on an edge (~0.2 %), so the Gaussian oracle passes with
    ``right=False`` too. Here every sample sits on a float32 edge of explicit bins, so a left-closed
    lower tail puts each one in the wrong mirror bin.
    """
    edges = H._symmetric_bin_edges(0.01, 5.0, np.float32)
    da = _gaussian()
    on_edges = edges[np.clip(np.searchsorted(edges, da.values), 1, len(edges) - 2)]
    da = da.copy(data=on_edges.astype(np.float32)).chunk(da.chunks)
    kw = {
        "method": "seasonal_percentile",
        "dimensions": GRID_DIMS,
        "dask_chunks": {"time": 100},
        "window_days": 11,
        "precision": 0.01,
        "max_anomaly": 5.0,
    }
    failures = []
    for percentile in (5, 34, 50):

        def row(percentile=percentile):
            low = marEx.extremes.identify(da, threshold_percentile=percentile, tail="lower", **kw).compute()
            high = marEx.extremes.identify(-da, threshold_percentile=100 - percentile, tail="upper", **kw).compute()
            np.testing.assert_array_equal(low.thresholds.values, -high.thresholds.values)
            np.testing.assert_array_equal(low.extreme_events.values, high.extreme_events.values)

        check(failures, f"p{percentile}", row)
    assert_no_failures(failures)


def test_thresholds_and_masks_mirror():
    """thresholds(-x, upper, 100-p) == -thresholds(x, lower, p), per method at p 5, 10 and 34.

    The 2-D path evaluates the lower tail as the upper-tail estimator on the reversed histogram at
    1 - q and closes its bins on the right, so a sample on an edge lands in the mirror bin: the
    exact mirror, no tolerance (run directly, the estimator sat one bin colder and under-flagged by
    27 % against the exact path at 15 yr). The 1-D path's cumulative search is one-sided
    (``cdf >= q``), so the mirrored problem resolves ties on the other side of a bin boundary; it
    pools the whole series, so it mirrors to within one bin, and the masks a user acts on agree
    except where a sample sits between two thresholds less than a bin apart.
    """
    da = _gaussian()
    failures = []
    for method in ("global_percentile", "seasonal_percentile"):
        for percentile in (5, 10, 34):

            def row(method=method, percentile=percentile):
                kw = {"method": method, "dimensions": GRID_DIMS, "dask_chunks": {"time": 100}}
                if method == "seasonal_percentile":
                    kw["window_days"] = 11
                low = marEx.extremes.identify(da, threshold_percentile=percentile, tail="lower", **kw).compute()
                high = marEx.extremes.identify(-da, threshold_percentile=100 - percentile, tail="upper", **kw).compute()
                if method == "seasonal_percentile":
                    np.testing.assert_array_equal(low.thresholds.values, -high.thresholds.values)
                    np.testing.assert_array_equal(low.extreme_events.values, high.extreme_events.values)
                    return
                np.testing.assert_allclose(low.thresholds.values, -high.thresholds.values, atol=0.01)
                offset = np.nanmean(low.thresholds.values + high.thresholds.values)
                assert abs(offset) <= 0.01, f"systematic mirror offset {offset:.4f} exceeds one bin"
                disagree = (low.extreme_events.values != high.extreme_events.values).mean()
                assert disagree < 0.005, f"masks disagree on {disagree:.4%} of points"

            check(failures, f"{method} p{percentile}", row)
    assert_no_failures(failures)


def test_the_exact_path_records_no_spatial_window():
    """``resolve_window_spatial`` defaults a gridded seasonal run to 5x5, but only the approximate path uses it.

    The attribute site did not thread ``method_percentile``, so it recorded a 5x5 window an exact
    run never applied.
    """
    da = _gaussian(n_time=1200)
    exact = marEx.extremes.identify(
        da, method="seasonal_percentile", method_percentile="exact", window_days=11, dimensions=GRID_DIMS
    )
    approximate = marEx.extremes.identify(
        da, method="seasonal_percentile", method_percentile="approximate", window_days=11, dimensions=GRID_DIMS
    )
    assert "window_spatial" not in exact.attrs
    assert "spatial neighbours" not in exact.attrs["preprocessing_steps"][-1]
    assert approximate.attrs["window_spatial"] == 5
    assert "5 spatial neighbours" in approximate.attrs["preprocessing_steps"][-1]


# ── the exact path's guard and accuracy ───────────────────────────────

# (lat, lon) of the degenerate cells; every other cell is N(0, 1) noise.
ZERO, NEG_TINY, POS_TINY = (0, 0), (0, 1), (1, 0)


def _guard_anomaly(sign):
    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 1.0, size=(1460, 3, 4)).astype(np.float32)
    data[:, ZERO[0], ZERO[1]] = 0.0
    # The L1 probe cell: almost always one tiny negative value (-3.05e-05), the threshold equal to it.
    data[:, NEG_TINY[0], NEG_TINY[1]] = np.float32(-3.05e-05)
    data[::50, NEG_TINY[0], NEG_TINY[1]] = np.float32(-1.0)
    data[:, POS_TINY[0], POS_TINY[1]] = np.float32(3.05e-05)
    da = xr.DataArray(
        sign * data,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=1460, freq="D"),
            "lat": np.arange(3, dtype=np.float32),
            "lon": np.arange(4, dtype=np.float32),
        },
        name="dat_anomaly",
    )
    return da.chunk({"time": -1, "lat": 2, "lon": 2})


@pytest.mark.parametrize("method", ["global_percentile", "seasonal_percentile"])
def test_exact_guard(method):
    """A (near-)constant anomaly never flags on the exact path; every cell past zero is untouched, bit for bit.

    A sea-ice cell's exact percentile EQUALS its constant, and the inclusive comparison flagged every
    tie: at L1 the exact path flagged 9.8 % of all cell-days, mostly poleward of 60 degrees. One run
    per tail (the lower tail mirrors the data, so the same cells sit on the wrong side of zero)
    carries the degenerate-cell, past-zero and (global) numpy checks.
    """
    failures = []
    for tail in ("upper", "lower"):

        def row(tail=tail):
            sign = 1.0 if tail == "upper" else -1.0
            da = _guard_anomaly(sign)
            q = 95 if tail == "upper" else 5
            kw = {"method": method, "method_percentile": "exact", "dimensions": GRID_DIMS, "tail": tail, "threshold_percentile": q}
            if method == "seasonal_percentile":
                kw["window_days"] = 11
            ds = marEx.extremes.identify(da, **kw).compute()

            for cell in (ZERO, NEG_TINY):
                assert not ds.extreme_events.isel(lat=cell[0], lon=cell[1]).values.any(), f"{cell} flagged"
                thr = ds.thresholds.isel(lat=cell[0], lon=cell[1]).values
                # A NORMAL float32 guard: a subnormal one reads as 0 under flush-to-zero, and a float64
                # subnormal (the global path's dtype) becomes 0 when the output is cast to float32.
                assert (thr.astype(np.float32) == np.float32(sign * np.finfo(np.float32).tiny)).all(), f"{cell} guard value"

            thr = ds.thresholds
            past = (thr > 0) if tail == "upper" else (thr < 0)
            normal = np.ones((3, 4), bool)
            normal[ZERO], normal[NEG_TINY] = False, False
            assert bool(past.values[..., normal].all())
            anom = da.compute()
            thr_t = thr.sel(dayofyear=anom.time.dt.dayofyear).drop_vars("dayofyear") if method == "seasonal_percentile" else thr
            expected = (anom >= thr_t) if tail == "upper" else (anom <= thr_t)
            np.testing.assert_array_equal(ds.extreme_events.values[:, normal], expected.values[:, normal])
            # The constant cell just past zero is a legitimate every-day "extreme" under `>=`: the guard
            # is about the zero side only.
            assert ds.extreme_events.isel(lat=POS_TINY[0], lon=POS_TINY[1]).values.all()

            if method == "global_percentile":  # the seasonal window reference is test_seasonal_approximate_vs_rank_rule_and_exact's
                ref = np.percentile(da.values, q, axis=0)
                past_ref = (ref > 0) if tail == "upper" else (ref < 0)
                np.testing.assert_allclose(ds.thresholds.values[past_ref], ref[past_ref], rtol=1e-6)
                assert not np.isnan(ds.thresholds.values).any()

        check(failures, tail, row)
    assert_no_failures(failures)


def test_global_exact_on_a_small_mesh():
    """A small mesh (< ~4445 cells, where the old rechunk size rounded to 0) runs, and its exact thresholds ARE xarray's.

    The exact path is xarray's linear ``quantile`` over each cell's series; re-tiling the cells
    (200 -> 2 x 100 here) and the zero-side clamp must leave it bit for bit. The reference is that
    same call on the untiled input, which is dask-backed (``anom`` is chunked), not ``np.percentile``:
    on numpy 2.5 / xarray 2026.9 xarray's DASK quantile differs from numpy by 1 ULP in some cells,
    while xarray's in-memory quantile agrees with numpy. The rows above check against numpy.
    """
    rng = np.random.default_rng(5)
    values = rng.normal(0.0, 1.0, size=(400, 200)).astype(np.float32)
    anom = xr.DataArray(
        values, dims=["time", "cell"], coords={"time": np.arange(400), "cell": np.arange(200)}, name="dat_anomaly"
    ).chunk({"time": -1, "cell": -1})
    _, thresholds = identify_extremes(
        anom,
        method_extreme="global_percentile",
        method_percentile="exact",
        dimensions={"time": "time", "x": "cell"},
        coordinates={"time": "time", "x": "cell"},
    )
    np.testing.assert_array_equal(thresholds.compute().values, anom.quantile(0.95, dim="time").values)


# ── calendar windows off the daily axis ───────────────────────────────


def _mesh(values, time):
    cells = np.arange(values.shape[1], dtype=np.float32)
    return xr.DataArray(
        values,
        dims=("time", "ncells"),
        coords={"time": time, "lon": ("ncells", cells), "lat": ("ncells", cells)},
        name="dat_anomaly",
    ).chunk({"time": -1})


def _calendar_reference(da, slot, window_days, percentile):
    """The percentile over every sample within ``window_days // 2`` calendar days of ``slot``, wrapping the year."""
    doy = da["time"].dt.dayofyear.values
    half = window_days // 2
    targets = {((slot - 1 + k) % 366) + 1 for k in range(-half, half + 1)}
    return float(np.percentile(da.isel(ncells=0).values[np.isin(doy, list(targets))], percentile))


def test_the_seasonal_window_pools_its_calendar_slots():
    """The window pools the calendar slots it names, against independent calendar references.

    Rows: 5-day data (value = day of year / 100, so a threshold names a calendar day) at windows 11
    and 31 on both paths (one slot is 0.01; HEAD's old 3-slot window at 11 days landed 4 slots low
    at slot 180); quarterly data on a 360-day calendar, whose month is still 30 days (165 days is
    5.5 months -> 6 -> odd 7, so April pools January to July, whatever the 90-day cadence); and a
    365-day window on monthly data (12 months -> odd 13 slots on a 12-month cycle), where both paths
    pool every month once, so every month gets the whole record's percentile.
    """
    failures = []
    time5 = pd.date_range("1980-01-01", periods=int(40 * 365.25 / 5), freq="5D")
    five_day = _mesh(np.repeat((time5.dayofyear.values / 100.0).astype(np.float32)[:, None], 2, axis=1), time5)
    for window_days, method_percentile in [(w, m) for w in (11, 31) for m in ("exact", "approximate")]:

        def five(window_days=window_days, method_percentile=method_percentile):
            thresholds = marEx.extremes.identify(
                five_day,
                threshold_percentile=99,
                window_days=window_days,
                method_percentile=method_percentile,
                dimensions=MESH_DIMS,
                coordinates=MESH_COORDS,
            ).thresholds.isel(ncells=0)
            tolerance = 1e-6 if method_percentile == "exact" else 0.01
            for slot in (100, 180):
                reference = _calendar_reference(five_day, slot, window_days, 99)
                got = float(thresholds.sel(dayofyear=slot))
                assert abs(got - reference) <= tolerance, f"slot {slot}: {got * 100:.4f} vs calendar window {reference * 100:.4f}"

        check(failures, f"5-day window={window_days} {method_percentile}", five)

    def quarterly_360_day():
        time = xr.date_range("2001-01-01", periods=80, freq="3MS", calendar="360_day", use_cftime=True)
        month = np.array([t.month for t in time.values])
        values = (month + np.repeat(np.linspace(-0.2, 0.2, 20), 4)).astype(np.float32)
        da = _mesh(np.repeat(values[:, None], 2, axis=1), time)
        thresholds = marEx.extremes.identify(
            da, threshold_percentile=90, window_days=165, method_percentile="exact", dimensions=MESH_DIMS, coordinates=MESH_COORDS
        ).thresholds
        assert float(thresholds.sel(month=4).isel(ncells=0)) == pytest.approx(
            float(np.percentile(values[month <= 7], 90)), abs=1e-6
        )
        assert resolve_cycle(da, "time").window_slots(165, calendar="360_day") == 7

    def year_long_monthly():
        time = pd.date_range("1981-01-01", periods=12 * 40, freq="MS")
        level = 0.2 * time.month.values[:, None]
        da = _mesh((level + 0.3 * np.random.default_rng(0).normal(size=(time.size, 2))).astype(np.float32), time)
        cycle = resolve_cycle(da, "time")
        assert cycle.index_name == "month" and cycle.window_slots(365) == 13

        def thresholds(method):
            return (
                marEx.extremes.identify(
                    da,
                    threshold_percentile=90,
                    window_days=365,
                    method_percentile=method,
                    dimensions=MESH_DIMS,
                    coordinates=MESH_COORDS,
                )
                .thresholds.transpose("month", "ncells")
                .values
            )

        exact, approximate = thresholds("exact"), thresholds("approximate")
        for cell in range(da.sizes["ncells"]):
            whole_record = np.float32(np.percentile(da.isel(ncells=cell).values, 90))
            np.testing.assert_allclose(exact[:, cell], whole_record, rtol=0, atol=1e-6)
            assert np.unique(approximate[:, cell]).size == 1
            assert abs(approximate[0, cell] - whole_record) <= 0.02

    check(failures, "quarterly 360_day", quarterly_360_day)
    check(failures, "year-long monthly", year_long_monthly)
    assert_no_failures(failures)


@pytest.fixture(scope="module")
def monthly_sst():
    """The shipped gridded fixture resampled to month-start means (40 years, 481 steps), values in memory."""
    sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to
    return sst.resample(time="MS").mean().compute().chunk({"time": 24})


def test_monthly_exact_window_wraps_within_the_cycle(monthly_sst):
    """``method_percentile='exact'`` carries its own ``% cycle.length`` wrap, separate from the histogram path.

    A 40-day window is one month; a 90-day window is three, so December pools November, December
    and January: its threshold must equal numpy's over exactly those months' anomalies. With
    ``% 366`` (or no wrap at all) the December slot would draw on the wrong months. The wider window
    must also smooth the seasonal contrast.
    """
    assert resolve_cycle(monthly_sst, "time").window_slots(90) == 3

    def run(window_days):
        return marEx.preprocess_data(
            monthly_sst,
            method_anomaly="fixed_baseline",
            method_extreme="seasonal_percentile",
            method_percentile="exact",
            window_days=window_days,
            dimensions=GRID_DIMS,
            dask_chunks={"time": 24},
        )

    narrow = run(40)
    assert narrow.sizes["month"] == 12
    assert "dayofyear" not in narrow.thresholds.dims
    narrow_thr = narrow.thresholds.compute()
    wide = run(90).compute()
    assert bool(np.isfinite(wide.thresholds).all()) and bool(np.isfinite(narrow_thr).all())
    assert float(wide.thresholds.std("month").mean()) < float(narrow_thr.std("month").mean())

    anomaly = wide.dat_anomaly.transpose("time", "lat", "lon").values
    months = wide.time.dt.month.values
    for month, pooled in ((12, (11, 12, 1)), (1, (12, 1, 2))):
        reference = np.nanpercentile(anomaly[np.isin(months, pooled)], 95, axis=0)
        got = wide.thresholds.sel(month=month).transpose("lat", "lon").values
        np.testing.assert_array_equal(got, reference, err_msg=f"month {month} does not pool {pooled}")
