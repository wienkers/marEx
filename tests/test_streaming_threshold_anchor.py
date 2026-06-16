"""Streaming mode must PIN the histogram threshold before the bounds check computes on it.

``_apply_threshold_bounds`` makes one eager ``dask.compute`` over the threshold. ``pin_one`` is a
no-op in streaming mode, so that compute ran the whole histogram graph on its own and the later
``stage`` of the thresholds ran it a second time. Staging it to zarr instead removed the second
pass but still crashed at the L1 smoke scale (731 x 120 x 1440, 16 x 14 GB; 27619732, 27619736),
where persisting the same graph completed with 0 restarts (27619742, 27619762); D-125.

What is pinned here is graph STRUCTURE, which bit-identity tests are blind to: the array the
bounds check receives must hold materialised blocks, carrying none of the histogram graph.
The threshold is bounded by cycle x space, so pinning it keeps streaming time-invariant; the
last test pins that invariance in n_time, not an absolute bound.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.core import istask

import marEx.extremes.histogram as H
from marEx.core.compute_mode import Materialiser, create_staging_dir
from marEx.extremes.histogram import _compute_histogram_quantile_1d, _compute_histogram_quantile_2d


def _anomaly_fixture(n_time=400, n_y=6, n_x=7, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(n_time, n_y, n_x)).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=n_time, freq="D"),
            "lat": np.arange(n_y, dtype=np.float32),
            "lon": np.arange(n_x, dtype=np.float32),
        },
        name="da",
    )
    return da.chunk({"time": -1, "lat": 3, "lon": 7})


DIMS = {"time": "time", "x": "lon", "y": "lat"}


def _q2d(da, mat):
    da = da.assign_coords(dayofyear=da.time.dt.dayofyear)
    return _compute_histogram_quantile_2d(da, q=0.95, window_spatial=3, dimensions=DIMS, materialiser=mat)


@pytest.fixture
def bounds_input(monkeypatch):
    """Capture the graph layer names of every array handed to the bounds check."""
    seen = []
    original = H._apply_threshold_bounds

    def spy(threshold, *args, **kwargs):
        seen.append(_unmaterialised(threshold))
        return original(threshold, *args, **kwargs)

    monkeypatch.setattr(H, "_apply_threshold_bounds", spy)
    return seen


def _unmaterialised(threshold):
    """Graph entries that are still tasks (pre-fix: the whole histogram graph)."""
    return [k for k, v in dict(threshold.data.__dask_graph__()).items() if istask(v)]


class TestStreamingPinsBeforeBounds:
    def test_2d_bounds_check_reads_a_pinned_threshold(self, bounds_input, tmp_path):
        mat = Materialiser("streaming", create_staging_dir(str(tmp_path)))
        _q2d(_anomaly_fixture(), mat)
        assert len(bounds_input) == 1
        assert bounds_input[0] == [], bounds_input[0][:5]

    def test_1d_bounds_check_reads_a_pinned_threshold(self, bounds_input, tmp_path):
        mat = Materialiser("streaming", create_staging_dir(str(tmp_path)))
        _compute_histogram_quantile_1d(_anomaly_fixture(), 0.95, dim="time", materialiser=mat)
        assert len(bounds_input) == 1
        assert bounds_input[0] == [], bounds_input[0][:5]

    @pytest.mark.parametrize("mode", ["persist", "lazy"])
    def test_other_modes_values_equal_streaming(self, mode, tmp_path):
        """persist/lazy never touch disk: values equal streaming's, bit for bit."""
        da = _anomaly_fixture()
        ref = _q2d(da, Materialiser(mode)).compute()
        mat = Materialiser("streaming", create_staging_dir(str(tmp_path)))
        got = _q2d(da, mat).compute()
        np.testing.assert_array_equal(ref.values, got.values)

    def test_streaming_pinned_bytes_do_not_grow_with_n_time(self, monkeypatch, tmp_path):
        """The one thing streaming now pins is O(cycle x space): doubling n_time must not move it."""
        pinned = []
        real = xr.DataArray.persist
        monkeypatch.setattr(xr.DataArray, "persist", lambda self, **kw: (pinned.append(self.nbytes), real(self, **kw))[1])
        totals = []
        for n_time in (730, 1460):
            pinned.clear()
            mat = Materialiser("streaming", create_staging_dir(str(tmp_path / str(n_time))))
            _q2d(_anomaly_fixture(n_time=n_time), mat)
            totals.append(sum(pinned))
        assert totals[0] > 0, "streaming pinned nothing: the anchor is not reached"
        assert totals[0] == totals[1], totals


def _synthetic(n_time):
    rng = np.random.default_rng(0)
    t = pd.date_range("2000-01-01", periods=n_time, freq="D")
    seasonal = 10 + 3 * np.sin(2 * np.pi * t.dayofyear.values / 365.25)[:, None, None]
    data = (seasonal + rng.normal(0, 1, (n_time, 12, 16))).astype(np.float32)
    coords = {"time": t, "lat": np.linspace(-10, 10, 12), "lon": np.linspace(0, 30, 16)}
    return xr.DataArray(data, dims=("time", "lat", "lon"), coords=coords, name="sst").chunk({"time": 30})


@pytest.mark.parametrize("method_extreme", ["seasonal_percentile", "global_percentile"])
def test_pipeline_streaming_pins_the_same_bytes_at_every_n_time(method_extreme, monkeypatch, tmp_path):
    """Every persist route through a streaming ``preprocess_data`` (standardised, so both
    threshold runs), recorded at three record lengths: the total must not move with n_time.
    The driver-level test above only shows that the anchor pins something."""
    import dask

    import marEx

    sizes = []
    real_dask, real_da, real_ds = dask.persist, xr.DataArray.persist, xr.Dataset.persist
    monkeypatch.setattr(dask, "persist", lambda *a, **k: (sizes.extend(getattr(x, "nbytes", 0) for x in a), real_dask(*a, **k))[1])
    monkeypatch.setattr(xr.DataArray, "persist", lambda s, *a, **k: (sizes.append(s.nbytes), real_da(s, *a, **k))[1])
    monkeypatch.setattr(xr.Dataset, "persist", lambda s, *a, **k: (sizes.append(s.nbytes), real_ds(s, *a, **k))[1])
    totals = []
    for n_time in (730, 1460, 2920):
        sizes.clear()
        marEx.preprocess_data(
            _synthetic(n_time),
            method_anomaly="detrend_harmonic",
            method_extreme=method_extreme,
            window_years=2,
            standardise=True,
            compute_mode="streaming",
            scratch_dir=str(tmp_path / str(n_time)),
        )
        totals.append(sum(sizes))
    assert totals[0] > 0, "nothing pinned: the recorder is not wired"
    assert totals[0] == totals[1] == totals[2], totals
