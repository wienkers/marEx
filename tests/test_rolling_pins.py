"""
Each float rolling mean in detect runs under ``rolling_numerics`` (D-141), one test per site.

A spy on xarray's rolling ``mean`` records the accelerator options in force when it is
called (the graph is built there, and that is when xarray reads them). ``preprocess_data``
runs under a global ``use_bottleneck=True``, so a site that loses its pin records True, and
so does a site whose pin is flipped. Comparing output values cannot see the second case:
a flipped pin ignores the global option exactly as a correct one does.
"""

from pathlib import Path

import pytest
import xarray as xr
from xarray.core.options import OPTIONS

import marEx

DATA_DIR = Path(__file__).parent / "data"
DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}
ROLLING_CLASSES = {type(xr.DataArray([0.0]).rolling(dim_0=1)), type(xr.Dataset({"v": ("t", [0.0])}).rolling(t=1))}


@pytest.fixture(scope="module")
def sst():
    ds = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={})
    return ds.to.isel(time=slice(-6 * 365, None)).chunk({"time": 25})


@pytest.fixture
def rolling_mean_options(monkeypatch):
    seen = []
    for cls in ROLLING_CLASSES:
        original = cls.mean

        def spy(self, *args, _original=original, **kwargs):
            seen.append({key: OPTIONS.get(key) for key in ("use_bottleneck", "use_numbagg")})
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "mean", spy)
    return seen


def _assert_every_rolling_mean_pinned(sst, seen, **kwargs):
    with xr.set_options(use_bottleneck=True):
        marEx.preprocess_data(
            sst, dimensions=DIMENSIONS, precision=0.01, max_anomaly=5.0, method_extreme="global_percentile", **kwargs
        )
    assert seen, "no rolling mean was reached; the guard is vacuous"
    unpinned = [options for options in seen if options["use_bottleneck"] or options["use_numbagg"]]
    assert not unpinned, f"{len(unpinned)} of {len(seen)} rolling means ran unpinned: {unpinned}"


def test_climatology_smoothing_is_pinned(sst, rolling_mean_options, dask_client):
    _assert_every_rolling_mean_pinned(sst, rolling_mean_options, method_anomaly="shifting_baseline", window_years=5, smooth_days=11)


def test_fixed_baseline_smoothing_is_pinned(sst, rolling_mean_options, dask_client):
    _assert_every_rolling_mean_pinned(sst, rolling_mean_options, method_anomaly="fixed_baseline")


def test_harmonic_rolling_std_is_pinned(sst, rolling_mean_options, dask_client):
    _assert_every_rolling_mean_pinned(sst, rolling_mean_options, method_anomaly="detrend_harmonic", standardise=True)
