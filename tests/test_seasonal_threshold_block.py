"""The seasonal threshold handed to the comparison must not be one cycle x whole-field block.

On a lat/lon grid the input is spatially whole (D-028), so aligning the thresholds to the input's
spatial chunks alone left the cycle axis whole too: one block of 366 x 720 x 1440 x 4 B = 1.5 GB at
0.25 deg. The groupby comparison indexes that block once per run of consecutive slots, every such
task holding a copy, and the L1 full streaming run died on it (D-132). Fixture fields are far too
small to reach the budget, so the budget is shrunk here instead; the block bound is then pinned at
two field sizes, because a bound that only holds at one size says nothing about how it scales.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.core.compute_mode import Materialiser
from marEx.extremes import seasonal_percentile

DIMS = {"time": "time", "x": "lon", "y": "lat"}


def _field(n_y, n_x, n_years=4, seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2000-01-01", periods=365 * n_years, freq="D")
    doy = time.dayofyear.values[:, None, None]
    data = (np.sin(2 * np.pi * doy / 365.25) + rng.normal(0.0, 0.5, size=(time.size, n_y, n_x))).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": np.linspace(-10, 10, n_y), "lon": np.linspace(0, 20, n_x)},
        name="sst",
    )
    return da.chunk({"time": 30, "lat": -1, "lon": -1})


def _kwargs(method_percentile):
    return {
        "method_anomaly": "shifting_baseline",
        "method_extreme": "seasonal_percentile",
        "method_percentile": method_percentile,
        "window_years": 2,
        "smooth_days": 21,
        "threshold_percentile": 95,
        "dimensions": DIMS,
        "dask_chunks": {"time": 30},
    }


@pytest.fixture
def staged_thresholds(monkeypatch):
    """Record the chunks of every threshold anchored before the comparison."""
    seen = []
    original = Materialiser.stage

    def spy(self, obj, label, *args, **kwargs):
        if label == "thresholds":
            seen.append(dict(zip(obj.dims, obj.chunks)))
        return original(self, obj, label, *args, **kwargs)

    monkeypatch.setattr(Materialiser, "stage", spy)
    return seen


@pytest.mark.parametrize("method_percentile", ["approximate", "exact"])
@pytest.mark.parametrize("n_y, n_x", [(6, 8), (12, 16)])
def test_threshold_block_stays_inside_the_budget(staged_thresholds, monkeypatch, method_percentile, n_y, n_x):
    budget = 20 * 6 * 8  # one 6 x 8 field holds 20 slots; the 12 x 16 field holds 5
    monkeypatch.setattr(seasonal_percentile, "TASK_ELEMENTS", budget, raising=False)
    marEx.preprocess_data(_field(n_y, n_x), **_kwargs(method_percentile))
    assert staged_thresholds, "the comparison's threshold was never staged"
    for chunks in staged_thresholds:
        assert chunks["lat"] == (n_y,) and chunks["lon"] == (n_x,), chunks
        block = max(chunks["dayofyear"]) * n_y * n_x
        assert block <= budget, f"threshold block {block} elements > budget {budget}: {chunks}"


@pytest.mark.parametrize("method_percentile", ["approximate", "exact"])
def test_bounding_the_cycle_axis_leaves_the_values_alone(monkeypatch, method_percentile):
    da = _field(6, 8)
    reference = marEx.preprocess_data(da, **_kwargs(method_percentile)).compute()
    monkeypatch.setattr(seasonal_percentile, "TASK_ELEMENTS", 7 * 6 * 8, raising=False)
    bounded = marEx.preprocess_data(da, **_kwargs(method_percentile)).compute()
    for var in ("extreme_events", "thresholds", "dat_anomaly", "mask"):
        xr.testing.assert_identical(bounded[var], reference[var])
