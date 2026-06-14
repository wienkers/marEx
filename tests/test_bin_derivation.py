"""Auto-derived histogram bin geometry.

``precision=0.01, max_anomaly=5.0`` are calibrated for SST anomalies in kelvin. On
precipitation (mm/day, anomalies of tens) that range clips almost everything into the
end bins; on pressure in Pa it is off by three orders of magnitude. Phase D therefore
derives whichever of the two the caller does not state, with ``n_bins`` as the
invariant.

The fixed point matters as much as the derivation: ``precision=0.01`` alone still
spans +/-5.0, because ``0.01 * 1000 / 2 == 5.0``. Anyone who pinned ``precision``
gets exactly the bins they had.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError
from marEx.extremes.base import resolve_bin_spec

DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}


def _field(scale=1.0, offset=0.0, n_time=1500, n_y=3, n_x=4, seed=1):
    rng = np.random.default_rng(seed)
    data = (offset + rng.normal(0.0, scale, size=(n_time, n_y, n_x))).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=n_time, freq="D"),
            "lat": np.arange(n_y, dtype=np.float32),
            "lon": np.arange(n_x, dtype=np.float32),
        },
        name="dat_anomaly",
    )
    return da.chunk({"time": -1, "lat": 3, "lon": 4})


class TestResolution:
    def test_both_supplied_are_honoured_untouched(self):
        assert resolve_bin_spec(_field(), 0.02, 3.0, 1000) == (0.02, 3.0)

    def test_precision_alone_reproduces_the_historical_range(self):
        """The fixed point: the old defaults are what the old default arguments give."""
        assert resolve_bin_spec(_field(), 0.01, None, 1000) == (0.01, 5.0)

    def test_max_anomaly_alone_derives_the_precision(self):
        precision, max_anomaly = resolve_bin_spec(_field(), None, 40.0, 1000)
        assert max_anomaly == 40.0
        assert precision == pytest.approx(0.08)

    def test_neither_derives_the_range_from_the_data(self):
        da = _field(scale=10.0)
        precision, max_anomaly = resolve_bin_spec(da, None, None, 1000)
        observed = float(max(abs(da.min().compute()), abs(da.max().compute())))
        assert max_anomaly == pytest.approx(observed)
        assert precision == pytest.approx(2 * observed / 1000)

    def test_a_constant_field_falls_back_to_the_sst_defaults(self):
        """A zero range would give zero-width bins; fall back and say so."""
        da = _field(scale=0.0)
        assert resolve_bin_spec(da, None, None, 1000) == (0.01, 5.0)

    def test_an_all_nan_field_falls_back_to_the_sst_defaults(self):
        da = _field() * np.nan
        assert resolve_bin_spec(da, None, None, 1000) == (0.01, 5.0)

    @pytest.mark.parametrize("n_bins", [1, 0, -5])
    def test_a_degenerate_n_bins_is_rejected(self, n_bins):
        with pytest.raises(ConfigurationError, match="n_bins must be at least 2"):
            resolve_bin_spec(_field(), None, None, n_bins)

    def test_n_bins_above_the_uint16_ceiling_is_rejected(self):
        """Bin indices are uint16; above 65535 they wrap silently rather than fail."""
        with pytest.raises(ConfigurationError, match="n_bins must not exceed 65535"):
            resolve_bin_spec(_field(), None, None, 70000)


class TestScaling:
    """The reason the derivation exists: a variable that is not SST in kelvin."""

    def test_a_precipitation_like_field_gets_a_sane_threshold_by_default(self):
        da = _field(scale=15.0, seed=3)  # anomalies of tens, as mm/day
        ds = marEx.extremes.identify(da, method="global_percentile", threshold_percentile=95, dimensions=DIMENSIONS).compute()
        # Against the EMPIRICAL per-cell percentile, not the analytic one: with 1500
        # samples per cell the sampling error of the 95th percentile is itself ~5 %,
        # which would swamp what this test is about. The histogram estimate must track
        # the true percentile to a few of its own (derived, ~0.12-wide) bins.
        reference = np.percentile(da.compute().values, 95, axis=0)
        np.testing.assert_allclose(ds.thresholds.values, reference, atol=5 * ds.attrs["precision"])
        assert ds.attrs["max_anomaly"] > 40

    def test_the_same_field_pinned_to_max_anomaly_5_warns_and_clips(self):
        """The failure the derivation removes, still reachable when pinned explicitly."""
        da = _field(scale=15.0, seed=3)
        with pytest.warns(UserWarning, match="exceed expected range"):
            marEx.extremes.identify(
                da,
                method="global_percentile",
                threshold_percentile=95,
                precision=0.01,
                max_anomaly=5.0,
                dimensions=DIMENSIONS,
            ).compute()

    def test_the_resolved_geometry_is_what_the_attributes_report(self):
        da = _field(scale=15.0, seed=3)
        ds = marEx.extremes.identify(da, method="global_percentile", dimensions=DIMENSIONS)
        assert ds.attrs["precision"] == pytest.approx(2 * ds.attrs["max_anomaly"] / 1000)
        assert ds.attrs["max_anomaly"] == pytest.approx(float(max(abs(da.min()), abs(da.max()))), rel=1e-6)

    def test_exact_percentile_reports_no_bin_geometry(self):
        """Nothing is binned on that path, so nothing is claimed about bins."""
        ds = marEx.extremes.identify(
            _field(n_time=400), method="global_percentile", method_percentile="exact", dimensions=DIMENSIONS
        )
        # Serialised as the string "None": `core/attrs.make_netcdf_safe_attrs` coerces
        # None so the dataset stays writable to NetCDF.
        assert ds.attrs["precision"] == "None"
        assert ds.attrs["max_anomaly"] == "None"


class TestExactCompatibility:
    """The sentinel check: `precision != 0.01` would fire on every derived run."""

    def test_explicit_precision_is_still_rejected_with_exact(self):
        with pytest.raises(ConfigurationError, match="Parameter 'precision' cannot be used"):
            marEx.extremes.identify_extremes(_field(n_time=400), method_percentile="exact", precision=0.02)

    def test_explicit_max_anomaly_is_still_rejected_with_exact(self):
        with pytest.raises(ConfigurationError, match="Parameter 'max_anomaly' cannot be used"):
            marEx.extremes.identify_extremes(_field(n_time=400), method_percentile="exact", max_anomaly=10.0)

    def test_the_historical_default_values_are_now_rejected_too(self):
        """0.01 and 5.0 stopped being defaults, so passing them IS an explicit request."""
        with pytest.raises(ConfigurationError, match="Parameter 'precision' cannot be used"):
            marEx.extremes.identify_extremes(_field(n_time=400), method_percentile="exact", precision=0.01)

    def test_exact_runs_clean_when_neither_is_given(self):
        extremes, thresholds = marEx.extremes.identify_extremes(
            _field(n_time=400), method_extreme="global_percentile", method_percentile="exact"
        )
        assert extremes.dtype == bool


class TestDerivationCost:
    """How many passes over the anomaly the derivation costs.

    Counted from the INFO line, not from ``dask.compute`` calls: ``dask`` is one
    module object, so patching ``marEx.extremes.base.dask`` also intercepts the
    histogram path's own bounds check and counts it. That mistake makes the test look
    like it caught a double derivation when it has caught a legitimate compute.
    """

    @staticmethod
    def _derivations(fn):
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture(level=logging.INFO)
        marEx_logger = logging.getLogger("marEx")
        previous = marEx_logger.level
        marEx_logger.setLevel(logging.INFO)
        marEx_logger.addHandler(handler)
        try:
            fn()
        finally:
            marEx_logger.removeHandler(handler)
            marEx_logger.setLevel(previous)
        return [m for m in records if "Histogram bins derived from the data" in m]

    def test_the_exact_path_never_derives(self):
        """Deriving there would cost a full pass over the anomaly for nothing."""
        n = self._derivations(
            lambda: marEx.extremes.identify(
                _field(n_time=400), method="global_percentile", method_percentile="exact", dimensions=DIMENSIONS
            )
        )
        assert n == []

    def test_the_approximate_path_derives_exactly_once(self):
        """`_extremes_core` resolves, then `identify_extremes` must find it already done."""
        n = self._derivations(
            lambda: marEx.extremes.identify(_field(n_time=400), method="global_percentile", dimensions=DIMENSIONS)
        )
        assert len(n) == 1, n

    def test_a_pinned_geometry_derives_nothing(self):
        n = self._derivations(
            lambda: marEx.extremes.identify(
                _field(n_time=400), method="global_percentile", precision=0.01, max_anomaly=5.0, dimensions=DIMENSIONS
            )
        )
        assert n == []

    def test_the_derivation_fuses_its_min_and_max(self, monkeypatch):
        """One traversal for both, not one each.

        Safe to count `dask.compute` here because `resolve_bin_spec` is called in
        isolation -- nothing else runs inside this block.
        """
        calls = []
        real = marEx.extremes.base.dask.compute

        def counting_compute(*args, **kwargs):
            calls.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(marEx.extremes.base.dask, "compute", counting_compute)
        resolve_bin_spec(_field(n_time=400), None, None, 1000)
        assert len(calls) == 1
        assert len(calls[0]) == 2


class TestLogging:
    def test_the_derived_geometry_is_logged(self):
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture(level=logging.INFO)
        marEx_logger = logging.getLogger("marEx")
        marEx_logger.addHandler(handler)
        try:
            resolve_bin_spec(_field(scale=10.0), None, None, 1000)
        finally:
            marEx_logger.removeHandler(handler)
        assert any("Histogram bins derived from the data" in m for m in records), records
