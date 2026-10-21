"""
The output ``mask`` describes itself, not the input field.

It is built from the input's first timestep, and xarray carries the input's attrs through that
comparison, so without care a boolean mask reports the SST's ``standard_name``, ``units`` and
``valid_min``.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx

INPUT_ATTRS = {"standard_name": "sea_surface_temperature", "units": "K", "valid_min": 250.0}


def _sst():
    time = pd.date_range("2000-01-01", periods=6 * 365, freq="D")
    rng = np.random.default_rng(0)
    data = rng.standard_normal((time.size, 3, 4)).astype(np.float32) + 290.0
    data[:, 0, 0] = np.nan
    return xr.DataArray(
        data,
        coords={"time": time, "lat": np.arange(3.0), "lon": np.arange(4.0)},
        dims=("time", "lat", "lon"),
        name="sst",
        attrs=dict(INPUT_ATTRS),
    ).chunk({"time": 365})


METHODS = {
    "detrend_harmonic": {},
    "fixed_baseline": {},
    "shifting_baseline": {"window_years": 2},
}


@pytest.mark.parametrize("method", list(METHODS))
def test_mask_attrs_are_its_own(method):
    sst = _sst()
    ds = marEx.anomaly.compute(sst, method=method, **METHODS[method])

    assert not set(INPUT_ATTRS) & set(ds.mask.attrs), ds.mask.attrs
    assert ds.mask.attrs["long_name"] == "valid-data mask"
    assert ds.mask.dtype == bool
    assert not bool(ds.mask[0, 0]) and bool(ds.mask[1, 1])
    assert sst.attrs == INPUT_ATTRS  # the caller's input is untouched


def test_preprocess_data_mask_attrs():
    ds = marEx.preprocess_data(_sst(), method_anomaly="detrend_harmonic", method_extreme="global_percentile")

    assert not set(INPUT_ATTRS) & set(ds.mask.attrs), ds.mask.attrs
    assert ds.mask.attrs["long_name"] == "valid-data mask"
