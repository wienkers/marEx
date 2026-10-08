"""The exact path's guard: a (near-)constant anomaly is never extreme.

A sea-ice cell's anomaly is constant (or nearly), so its exact percentile EQUALS that constant
and the inclusive comparison flags every tie: at L1 the exact path flagged 9.8 % of all cell-days,
mostly poleward of 60 degrees, where the approximate path (which clamps onto its guard rail)
flagged none of them. The exact threshold is now kept strictly on its own side of zero.

Two halves, both needed: the degenerate cells stop flagging, and every cell whose threshold was
already strictly past zero keeps its threshold and its events bit for bit.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx

DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}
# (lat, lon) of the degenerate cells; every other cell is N(0, 1) noise.
ZERO, NEG_TINY, POS_TINY = (0, 0), (0, 1), (1, 0)


def _anomaly(sign=1.0):
    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 1.0, size=(1460, 3, 4)).astype(np.float32)
    data[:, ZERO[0], ZERO[1]] = 0.0
    # The L1 probe cell: almost always one tiny negative value (-3.05e-05), the threshold equal to it.
    data[:, NEG_TINY[0], NEG_TINY[1]] = np.float32(-3.05e-05)
    data[::50, NEG_TINY[0], NEG_TINY[1]] = np.float32(-1.0)
    data[:, POS_TINY[0], POS_TINY[1]] = np.float32(3.05e-05)
    data = sign * data
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=1460, freq="D"),
            "lat": np.arange(3, dtype=np.float32),
            "lon": np.arange(4, dtype=np.float32),
        },
        name="dat_anomaly",
    )
    return da.chunk({"time": -1, "lat": 2, "lon": 2})


def _identify(da, method, tail):
    kw = {"method": method, "method_percentile": "exact", "dimensions": DIMENSIONS, "tail": tail}
    kw["threshold_percentile"] = 95 if tail == "upper" else 5
    if method == "seasonal_percentile":
        kw["window_days"] = 11
    return marEx.extremes.identify(da, **kw).compute()


@pytest.mark.parametrize("method", ["global_percentile", "seasonal_percentile"])
@pytest.mark.parametrize("tail", ["upper", "lower"])
class TestExactGuard:
    def test_degenerate_cells_never_flag(self, method, tail):
        # The lower tail mirrors the data, so the same cells sit on the wrong side of zero.
        ds = _identify(_anomaly(1.0 if tail == "upper" else -1.0), method, tail)
        sign = 1.0 if tail == "upper" else -1.0
        for cell in (ZERO, NEG_TINY):
            assert not ds.extreme_events.isel(lat=cell[0], lon=cell[1]).values.any(), f"{cell} flagged"
            thr = ds.thresholds.isel(lat=cell[0], lon=cell[1]).values
            # A NORMAL float32 guard: a subnormal one reads as 0 under flush-to-zero, and a float64
            # subnormal (the global path's dtype) becomes 0 when the output is cast to float32.
            assert (thr.astype(np.float32) == np.float32(sign * np.finfo(np.float32).tiny)).all()

    def test_cells_past_zero_are_untouched(self, method, tail):
        """Thresholds strictly past zero, and their events, equal the unclamped comparison bit for bit."""
        da = _anomaly(1.0 if tail == "upper" else -1.0)
        ds = _identify(da, method, tail)
        thr = ds.thresholds
        past = (thr > 0) if tail == "upper" else (thr < 0)
        # Every N(0,1) cell's p95 / p5 is far from zero, and so is the constant +/-3.05e-05 cell's.
        normal = np.ones((3, 4), bool)
        normal[ZERO], normal[NEG_TINY] = False, False
        assert bool(past.values[..., normal].all())
        anom = da.compute()
        if method == "seasonal_percentile":
            thr_t = thr.sel(dayofyear=anom.time.dt.dayofyear).drop_vars("dayofyear")
        else:
            thr_t = thr
        expected = (anom >= thr_t) if tail == "upper" else (anom <= thr_t)
        np.testing.assert_array_equal(ds.extreme_events.values[:, normal], expected.values[:, normal])
        # The constant cell just past zero is a legitimate, every-day "extreme" under `>=`: the guard
        # is about the zero side only, it does not second-guess a threshold that is already past it.
        assert ds.extreme_events.isel(lat=POS_TINY[0], lon=POS_TINY[1]).values.all()

    def test_thresholds_match_numpy_where_past_zero(self, method, tail):
        if method != "global_percentile":
            pytest.skip("the seasonal window reference is pinned in test_percentile_agreement")
        da = _anomaly(1.0 if tail == "upper" else -1.0)
        ds = _identify(da, method, tail)
        q = 95 if tail == "upper" else 5
        ref = np.percentile(da.values, q, axis=0)
        past = (ref > 0) if tail == "upper" else (ref < 0)
        np.testing.assert_allclose(ds.thresholds.values[past], ref[past], rtol=1e-6)
        assert not np.isnan(ds.thresholds.values).any()
