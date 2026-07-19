"""The detect stage must give the same answer however its input is chunked.

Every reduction along time -- the smoothing's running sum, flox's grouped means, the harmonic
fit -- accumulates block by block, so the floating-point result used to move with the
input's time chunk boundaries: ~1e-4 K, enough to move a threshold across a 0.01 bin and flip
extreme events. The anomaly stage now reduces with the time axis whole inside bounded spatial
tiles, so every layout must reproduce the time-whole reference bit for bit, for every anomaly
and threshold method.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
import marEx.core.dimensions as dims

ANOMALY_METHODS = ["shifting_baseline", "fixed_baseline", "detrend_fixed_baseline", "detrend_harmonic"]
EXTREME_METHODS = ["global_percentile", "seasonal_percentile"]
PERCENTILE_METHODS = ["exact", "approximate"]
COMPARED_VARS = ("dat_anomaly", "extreme_events", "thresholds", "mask")
DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}
# (input layout, per-task element budget). The fixture fits one spatial tile at the default budget, so
# the last case shrinks the budget until the canonical layout splits space into several tiles.
LAYOUTS = [({"time": 17}, None), ({"time": 60, "lat": 4, "lon": 7}, None), ({"lat": 3, "lon": 5}, None), ({"time": 30}, 30_000)]


@pytest.fixture(scope="module")
def sst():
    path = Path(__file__).parent / "data" / "sst_gridded.zarr"
    return xr.open_zarr(str(path), chunks={}).to.isel(time=slice(-6 * 365, None), lat=slice(0, 8), lon=slice(0, 14)).load()


def _detect(sst, layout, method_anomaly, method_extreme, method_percentile, tail="upper"):
    kw = {
        "method_anomaly": method_anomaly,
        "method_extreme": method_extreme,
        "method_percentile": method_percentile,
        "threshold_percentile": 95 if tail == "upper" else 5,
        "tail": tail,
        "dimensions": DIMENSIONS,
        "dask_chunks": {"time": 25},
    }
    if method_anomaly == "shifting_baseline":
        kw.update(window_years=3, smooth_days=11)
    if method_extreme == "seasonal_percentile":
        kw.update(window_days=3)
    if method_percentile == "approximate":
        kw.update(precision=0.01, max_anomaly=5.0)
    return marEx.preprocess_data(sst.chunk(layout), **kw).compute()


@pytest.mark.parametrize("method_percentile", PERCENTILE_METHODS)
@pytest.mark.parametrize("method_extreme", EXTREME_METHODS)
@pytest.mark.parametrize("method_anomaly", ANOMALY_METHODS)
def test_every_layout_matches_the_time_whole_reference(sst, method_anomaly, method_extreme, method_percentile, monkeypatch):
    ref = _detect(sst, {"time": -1}, method_anomaly, method_extreme, method_percentile)
    for layout, budget in LAYOUTS:
        if budget is not None:
            monkeypatch.setattr(dims, "TASK_ELEMENTS", budget)
            tile = dims.canonical_time_chunks(sst.chunk(layout), DIMENSIONS)
            assert tile["lat"] < sst.sizes["lat"] and tile["lon"] < sst.sizes["lon"], f"budget {budget} did not tile: {tile}"
        got = _detect(sst, layout, method_anomaly, method_extreme, method_percentile)
        monkeypatch.undo()
        for name in COMPARED_VARS:
            if name not in ref.data_vars:
                continue
            a, b = ref[name].values, got[name].values
            assert a.dtype == b.dtype, f"{layout}: {name} dtype moved"
            assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), f"{layout}: {name} is not bit-identical"


# A representative subset, not the full grid: the histogram path (`_apply_threshold_bounds`) under both
# anomaly families and both threshold methods, and the exact path's lower-tail clamp under both as well.
LOWER_TAIL_CASES = [
    ("shifting_baseline", "global_percentile", "approximate"),
    ("detrend_harmonic", "seasonal_percentile", "approximate"),
    ("fixed_baseline", "global_percentile", "exact"),
    ("detrend_fixed_baseline", "seasonal_percentile", "exact"),
]
FLAT_CELL = {"lat": 2, "lon": 3}


@pytest.fixture(scope="module")
def sst_with_flat_cell(sst):
    """The same field with one cell held constant, so its anomaly is zero and the guard rail must bind."""
    flat = sst.copy(deep=True)
    flat[:, FLAT_CELL["lat"], FLAT_CELL["lon"]] = 15.0
    return flat


@pytest.mark.parametrize(
    ("method_anomaly", "method_extreme", "method_percentile"), LOWER_TAIL_CASES, ids=["-".join(c) for c in LOWER_TAIL_CASES]
)
def test_lower_tail_is_chunk_invariant_and_stays_below_zero(
    sst_with_flat_cell, method_anomaly, method_extreme, method_percentile, monkeypatch
):
    """tail='lower' has its own guard and range branches: bit-identical across layouts AND anchored in absolute terms.

    Invariance alone cannot see a lower-tail branch that is wrong the same way for every layout, so the reference
    is also held to what a lower-tail threshold must be: negative, nowhere near all clamped onto the guard rail,
    and on the guard rail (one bin from zero, or the smallest normal float) in the constant cell.
    """
    sst = sst_with_flat_cell
    ref = _detect(sst, {"time": -1}, method_anomaly, method_extreme, method_percentile, tail="lower")

    thr = ref["thresholds"]
    assert float(thr.max()) < 0, "a lower-tail threshold must lie below zero"
    assert float(thr.median()) < -0.1, "nearly every lower-tail threshold was clamped onto the guard rail"
    # The seasonal approximate path smooths thresholds across neighbouring cells, so the flat cell borrows theirs.
    if not (method_extreme == "seasonal_percentile" and method_percentile == "approximate"):
        flat_thr = thr.isel(**FLAT_CELL).values
        guard = -0.01 if method_percentile == "approximate" else -np.finfo(np.float32).tiny
        assert np.allclose(
            flat_thr, guard, rtol=1e-6, atol=0
        ), f"flat cell threshold {flat_thr.ravel()[:3]} is not on the guard rail {guard}"
        assert int(ref["extreme_events"].isel(**FLAT_CELL).sum()) == 0, "a constant anomaly was flagged extreme"

    for layout, budget in LAYOUTS:
        if budget is not None:
            monkeypatch.setattr(dims, "TASK_ELEMENTS", budget)
        got = _detect(sst, layout, method_anomaly, method_extreme, method_percentile, tail="lower")
        monkeypatch.undo()
        for name in COMPARED_VARS:
            if name not in ref.data_vars:
                continue
            a, b = ref[name].values, got[name].values
            assert a.dtype == b.dtype, f"{layout}: {name} dtype moved"
            assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), f"{layout}: {name} is not bit-identical"
