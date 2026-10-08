"""``rolling_numerics`` pins xarray's rolling means to the numpy path on every xarray release.

bottleneck's ``move_mean`` keeps a float32 running sum over the whole chunk, so on a time-whole
series (the canonical layout detect smooths on) its error grows with the series length: ~6e-3 K
after 24 years of SST, against ~1e-4 K for the numpy path. xarray < 2026.9 used bottleneck by
default whenever it was installed, so without the pin the result depended on the xarray release.
"""

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from marEx.core.numerics import rolling_numerics


def _sst_like(n_time=24 * 365, n_cells=40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    field = 290.0 + 5.0 * np.sin(2 * np.pi * t / 365.25)[:, None] + rng.normal(0.0, 1.0, (n_time, n_cells))
    return xr.DataArray(field.astype(np.float32), dims=("time", "cell"))


def test_rolling_mean_stays_at_float32_rounding_on_a_long_time_whole_series():
    da = _sst_like()
    window = 11
    reference = sliding_window_view(da.values.astype(np.float64), window, axis=0).mean(axis=-1)
    with rolling_numerics():
        smoothed = da.chunk({"time": -1}).rolling(time=window, center=True).mean().values
    got = smoothed[window // 2 : window // 2 + reference.shape[0]]
    # numpy path: ~1e-4 at 290 K in float32. bottleneck's running sum reaches 1.5e-3 on this series.
    assert np.abs(got - reference).max() < 5e-4


def test_rolling_mean_does_not_depend_on_time_chunking():
    da = _sst_like(n_time=4 * 365)
    with rolling_numerics():
        whole = da.chunk({"time": -1}).rolling(time=11, center=True).mean().values
        split = da.chunk({"time": 25}).rolling(time=11, center=True).mean().values
    np.testing.assert_array_equal(whole, split)
