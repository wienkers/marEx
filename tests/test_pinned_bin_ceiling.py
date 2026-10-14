"""Every histogram geometry respects the uint16 bin-index ceiling.

Both seasonal engines store the bin index as uint16, and NaN digitizes to the index one past
the last bin, so at most 65535 bins can be realised without an index wrapping. A wrapped index
lands in a low bin and is counted there, which gives a plausible-looking wrong threshold rather
than an error. The realised count is what ``_symmetric_bin_edges`` builds, and float edge
rounding can make it one more than ``2 * max_anomaly / precision`` (``n_bins=65535`` realises
65536 bins), so the ceiling is checked on the realised count for every geometry: a pinned range
(deprecated ``max_anomaly``), an explicit ``n_bins`` and a range derived from the data.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError
from marEx.extremes.base import resolve_bin_spec
from marEx.extremes.histogram import _symmetric_bin_edges

DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}
CEILING = np.iinfo(np.uint16).max


def _alternating(n_time=20, nan_steps=()):
    """One cell of daily samples alternating 2 and 3, optionally with NaN steps."""
    values = np.where(np.arange(n_time) % 2 == 0, 2.0, 3.0).astype(np.float32)
    values[list(nan_steps)] = np.nan
    return xr.DataArray(
        values[:, None, None],
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=n_time, freq="D"),
            "lat": np.array([0.0], dtype=np.float32),
            "lon": np.array([0.0], dtype=np.float32),
        },
        name="dat_anomaly",
    ).chunk({"time": -1})


def _identify(da, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        return marEx.extremes.identify(
            da,
            method="seasonal_percentile",
            threshold_percentile=95,
            window_days=11,
            window_spatial=1,
            dimensions=DIMENSIONS,
            **kwargs,
        ).compute()


def _exact_reference(da):
    """The exact-path thresholds: slots no window reaches are NaN on both paths."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        return marEx.extremes.identify(
            da,
            method="seasonal_percentile",
            threshold_percentile=95,
            window_days=11,
            method_percentile="exact",
            dimensions=DIMENSIONS,
        ).thresholds.compute()


def _assert_tracks_exact(da, ds, precision):
    exact = _exact_reference(da)
    approx, exact = ds.thresholds.transpose(*exact.dims).values, exact.values
    np.testing.assert_array_equal(np.isfinite(approx), np.isfinite(exact))
    assert np.isfinite(approx).sum() > 0
    assert np.nanmax(np.abs(approx - exact)) <= precision


def _realised(precision, max_anomaly):
    return len(_symmetric_bin_edges(precision, max_anomaly, np.float32)) - 1


class TestBoundaryGeometry:
    """The boundary cases, derived from the edge function itself."""

    def test_the_pinned_boundary_is_where_the_edges_say(self):
        assert _realised(1.0, 32767.0) == CEILING - 1
        assert _realised(1.0, 32768.0) == CEILING + 1

    def test_n_bins_65535_realises_one_bin_too_many(self):
        assert _realised(2 * 3.0 / 65534, 3.0) == 65534
        assert _realised(2 * 3.0 / 65535, 3.0) == CEILING + 1


class TestPinnedRange:
    def test_a_fine_precision_over_a_pinned_range_is_rejected(self):
        """precision=1e-4 over +/-5 is 100000 bins: the uint16 index wrapped and the threshold came back as 1e-4."""
        with pytest.raises(ConfigurationError, match="n_bins must not exceed 65535"):
            _identify(_alternating(), precision=1e-4, max_anomaly=5.0)

    def test_the_largest_pinned_geometry_that_fits_runs(self):
        precision, max_anomaly = 1.0, 32767.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            assert resolve_bin_spec(_alternating(), precision, max_anomaly) == (precision, max_anomaly)
        ds = _identify(_alternating(), precision=precision, max_anomaly=max_anomaly)
        _assert_tracks_exact(_alternating(), ds, precision)

    def test_one_bin_more_is_rejected(self):
        with pytest.raises(ConfigurationError, match="n_bins must not exceed 65535"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                warnings.simplefilter("ignore", FutureWarning)
                resolve_bin_spec(_alternating(), 1.0, 32768.0)


class TestUnpinnedRange:
    def test_n_bins_65535_is_rejected_when_it_realises_65536(self):
        """NaN's index 65536 would wrap to 0 and be counted in the coldest bin."""
        with pytest.raises(ConfigurationError, match="n_bins must not exceed 65535"):
            _identify(_alternating(nan_steps=(4, 9)), n_bins=65535)

    def test_n_bins_65534_runs_and_drops_nan(self):
        """NaN steps are dropped, so the thresholds equal those of the series without them, bit for bit."""
        da = _alternating(nan_steps=(4, 9))
        ds = _identify(da, n_bins=65534)
        assert ds.attrs["max_anomaly"] == pytest.approx(3.0)
        valid = np.isfinite(da.isel(lat=0, lon=0).values)
        without = _identify(da.isel(time=np.flatnonzero(valid)), n_bins=65534)
        assert without.attrs["precision"] == ds.attrs["precision"]
        assert np.isfinite(ds.thresholds.values).sum() > 0
        np.testing.assert_array_equal(ds.thresholds.values, without.thresholds.values)
