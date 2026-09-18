"""The detect stage must give the same answer however its input is chunked (D-091).

Every reduction along time -- the smoothing's running sum, flox's grouped means, the harmonic
fit -- accumulates block by block, so before D-091 the floating-point result moved with the
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


def _detect(sst, layout, method_anomaly, method_extreme, method_percentile):
    kw = {
        "method_anomaly": method_anomaly,
        "method_extreme": method_extreme,
        "method_percentile": method_percentile,
        "threshold_percentile": 95,
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
