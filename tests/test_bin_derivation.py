"""Auto-derived histogram bin geometry (D-152).

``precision=0.01, max_anomaly=5.0`` were calibrated for SST anomalies in kelvin. On
precipitation (mm/day, anomalies of tens) that range clips almost everything into the
end bins; on pressure in Pa it is off by three orders of magnitude. So only ``precision``
is public, and the range is derived from the data in its own units:

* the requested tail's own extreme (``max`` upper, ``-min`` lower) is a hard cap, since a
  threshold is a quantile of those samples;
* a per-cell normal estimate, ``3 * max_cells(mean + z_p * std)``, lowers it when that
  extreme is an outlier;
* a threshold that reaches the edge of an estimated range regrows it to the cap at the same
  ``precision`` and is recomputed, bit-identical to a run over the cap.

``precision`` defaults to 3000 bins over the range; given, the bin count follows (a warning
above 10000, an error above 65000). ``max_anomaly`` and ``n_bins`` are deprecated: the
first still pins the range, the second replaces the 3000-bin target.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError
from marEx.extremes import base
from marEx.extremes.base import _derive_bin_spec, resolve_bin_spec

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

    def test_precision_alone_keeps_its_width_and_derives_the_range(self):
        """D-152 retires the old fixed point (``0.01`` alone spanned +/-5.0): the range is the data's."""
        da = _field()
        assert resolve_bin_spec(da, 0.01, None) == (0.01, pytest.approx(float(da.max())))

    def test_max_anomaly_alone_derives_the_precision(self):
        precision, max_anomaly = resolve_bin_spec(_field(), None, 40.0, 1000)
        assert max_anomaly == 40.0
        assert precision == pytest.approx(0.08)

    def test_neither_derives_the_range_from_the_data(self):
        da = _field(scale=10.0)
        precision, max_anomaly = resolve_bin_spec(da, None, None, 1000)
        observed = float(da.max())  # the upper tail's own extreme
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

    def test_the_same_field_pinned_to_max_anomaly_5_raises(self):
        """The failure the derivation removes, still reachable when pinned explicitly (D-138: an error)."""
        da = _field(scale=15.0, seed=3)
        with pytest.raises(ConfigurationError, match="exceed expected range"):
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
        # A Gaussian field: the per-cell estimate (3 x 1.645 sigma) lies past the data's own maximum,
        # so the range is that maximum and the width gives 3000 bins over it (D-152).
        assert ds.attrs["max_anomaly"] == pytest.approx(float(da.max()), rel=1e-6)
        assert ds.attrs["precision"] == pytest.approx(2 * float(da.max()) / 3000, rel=1e-6)


def _capture_warnings(fn):
    """Run ``fn`` and return (its result, the marEx WARNING messages it logged)."""
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture(level=logging.WARNING)
    marEx_logger = logging.getLogger("marEx")
    marEx_logger.addHandler(handler)
    try:
        return fn(), records
    finally:
        marEx_logger.removeHandler(handler)


def _outlier_field(n_time=1095, storm=400.0, seed=4):
    """A field whose upper extreme is one far outlier, so the per-cell estimate binds."""
    da = _field(n_time=n_time, n_y=6, n_x=6, seed=seed)
    data = da.values.copy()
    data[100, 2, 2] = storm
    return da.copy(data=data).chunk({"time": 200, "lat": 3, "lon": 3})


class TestDefaultGeometry:
    """D-152: the range comes from the requested tail, the default width gives 3000 bins."""

    def test_the_upper_tail_range_is_the_data_maximum(self):
        da = _field(scale=3.0, offset=1.0)  # skewed in sign: max and -min differ
        precision, max_anomaly = resolve_bin_spec(da, None, None)
        assert max_anomaly == pytest.approx(float(da.max()))
        assert precision == pytest.approx(2 * max_anomaly / 3000)

    def test_the_lower_tail_range_is_minus_the_data_minimum(self):
        da = _field(scale=3.0, offset=1.0)
        precision, max_anomaly = resolve_bin_spec(da, None, None, tail="lower")
        assert max_anomaly == pytest.approx(-float(da.min()))
        assert precision == pytest.approx(2 * max_anomaly / 3000)

    def test_the_per_cell_estimate_lowers_an_outlier_range(self):
        da = _outlier_field()
        spec = _derive_bin_spec(da, None, None, None, 95, "upper", "time")
        assert spec.mode == "estimated"
        assert spec.cap == pytest.approx(400.0)
        assert spec.max_anomaly < 100  # far below the single 400 outlier
        assert spec.n_bins == pytest.approx(3000, abs=1)

    def test_a_gaussian_field_keeps_the_data_maximum(self):
        """3 x (mean + 1.645 std) lies past a normal sample's own maximum: the estimate never binds."""
        da = _field(scale=2.0)
        spec = _derive_bin_spec(da, None, None, None, 95, "upper", "time")
        assert spec.mode == "data"
        assert spec.max_anomaly == pytest.approx(float(da.max()))

    def test_a_large_unit_field_gets_3000_bins_whatever_its_units(self):
        da = _field(scale=1000.0)  # pressure-like, in Pa
        precision, max_anomaly = resolve_bin_spec(da, None, None)
        assert 2 * max_anomaly / precision == pytest.approx(3000)
        assert max_anomaly > 2000


class TestPrecisionOnly:
    """With only ``precision``, the range is still derived and the bin count follows."""

    def test_the_bin_count_follows_the_derived_range(self):
        da = _field(scale=3.0)
        precision, max_anomaly = resolve_bin_spec(da, 0.01, None)
        assert precision == 0.01
        assert max_anomaly == pytest.approx(float(da.max()))

    def test_more_than_10000_bins_warns(self):
        da = _field(scale=3.0)  # max ~12: 0.001 needs ~24000 bins
        (precision, _), records = _capture_warnings(lambda: resolve_bin_spec(da, 0.001, None))
        assert precision == 0.001
        assert any("10000 bins" in m for m in records), records

    def test_fewer_than_10000_bins_is_silent(self):
        _, records = _capture_warnings(lambda: resolve_bin_spec(_field(scale=3.0), 0.01, None))
        assert not any("10000 bins" in m for m in records), records

    def test_more_than_65000_bins_is_rejected_before_any_histogram(self):
        with pytest.raises(ConfigurationError, match="above the 65000"):
            resolve_bin_spec(_field(scale=1000.0), 0.01, None)


class TestRegrow:
    """An estimated range that a threshold reaches is regrown to the data's extreme, exactly."""

    @staticmethod
    def _run(da, monkeypatch, safety, **kwargs):
        monkeypatch.setattr(base, "_RANGE_SAFETY", safety)
        used = []
        extremes, thresholds = base.identify_extremes(
            da, method_extreme="global_percentile", threshold_percentile=99, bin_spec_out=used, **kwargs
        )
        return extremes.compute(), thresholds.compute(), used[0]

    def test_a_saturated_estimate_is_regrown_and_matches_a_run_over_the_cap(self, monkeypatch):
        da = _outlier_field()
        data = da.values.copy()
        data[::40, 4, 4] = 150.0  # 2.5 % of one cell's days: its p99 IS 150, past a 0.5x estimate
        da = da.copy(data=data).chunk({"time": 200, "lat": 3, "lon": 3})
        (ext, thr, spec), records = _capture_warnings(lambda: self._run(da, monkeypatch, 0.5))
        assert spec.mode == "data" and spec.max_anomaly == pytest.approx(400.0)
        assert any("recomputing the thresholds" in m for m in records), records
        # The reference: the same bin width over the cap from the start (estimate disabled).
        ext_ref, thr_ref, spec_ref = self._run(da, monkeypatch, 1e9, precision=spec.precision)
        assert spec_ref == spec
        xr.testing.assert_identical(thr, thr_ref)
        xr.testing.assert_identical(ext, ext_ref)

    def test_an_unsaturated_estimate_is_bit_identical_to_the_cap(self, monkeypatch):
        """At a fixed width the interior edges are the same floats whatever the range, so the range
        only matters where a threshold reaches its edge (the premise of the regrow)."""
        da = _outlier_field()
        ext, thr, spec = self._run(da, monkeypatch, 3.0)
        assert spec.mode == "estimated" and spec.max_anomaly < 100
        ext_ref, thr_ref, spec_ref = self._run(da, monkeypatch, 1e9, precision=spec.precision)
        assert spec_ref.mode == "data" and spec_ref.max_anomaly == pytest.approx(400.0)
        xr.testing.assert_identical(thr, thr_ref)
        xr.testing.assert_identical(ext, ext_ref)

    def test_a_seasonal_end_bin_crossing_below_the_inner_edge_is_regrown(self, monkeypatch):
        """The 2-D path interpolates between bin CENTRES, so a quantile crossing in the clipped end
        bin can land below that bin's inner edge (falsifier finding 3). Here every window's end-bin
        mass is 2/30 of its samples, between (1-q) and 2(1-q) at q=0.95, so no threshold passes the
        inner edge: only a centre-based bound sees the saturation."""
        n_years = 30
        time = pd.date_range("2001-01-01", "2030-12-31", freq="D")
        rng = np.random.default_rng(7)
        data = rng.normal(0.0, 1.0, size=(len(time), 4, 4)).astype(np.float32)
        data[time.year < 2003] = 50.0  # two of thirty years: the same storm mass in every window
        da = xr.DataArray(
            data,
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": np.arange(4.0), "lon": np.arange(4.0)},
            name="dat_anomaly",
        ).chunk({"time": 730})
        assert len(set(time.year)) == n_years

        def run(safety, **kwargs):
            monkeypatch.setattr(base, "_RANGE_SAFETY", safety)
            used = []
            extremes, thresholds = base.identify_extremes(
                da, method_extreme="seasonal_percentile", threshold_percentile=95, bin_spec_out=used, **kwargs
            )
            return extremes.compute(), thresholds.compute(), used[0]

        ext, thr, spec = run(0.3)
        assert spec.mode == "data", spec  # regrown from the estimate to the cap
        assert float(thr.min()) > 40  # the storms, not the clipped estimate (~7)
        ext_ref, thr_ref, spec_ref = run(1e9, precision=spec.precision)
        assert spec_ref == spec
        xr.testing.assert_identical(thr, thr_ref)
        xr.testing.assert_identical(ext, ext_ref)


class TestDeprecatedArguments:
    def test_max_anomaly_warns_and_still_pins(self):
        da = _field(scale=15.0, seed=3)
        with pytest.warns(FutureWarning, match="`max_anomaly` is deprecated"):
            with pytest.raises(ConfigurationError, match="exceed expected range"):
                marEx.extremes.identify(da, method="global_percentile", precision=0.01, max_anomaly=5.0, dimensions=DIMENSIONS)

    def test_n_bins_warns_and_replaces_the_3000_bin_target(self):
        da = _field(scale=3.0)
        with pytest.warns(FutureWarning, match="`n_bins` is deprecated"):
            ds = marEx.extremes.identify(da, method="global_percentile", n_bins=500, dimensions=DIMENSIONS)
        assert 2 * ds.attrs["max_anomaly"] / ds.attrs["precision"] == pytest.approx(500)

    def test_one_sided_max_anomaly_keeps_the_1000_bin_invariant(self):
        assert resolve_bin_spec(_field(), None, 40.0) == (pytest.approx(0.08), 40.0)

    def test_the_default_range_is_not_pinned(self):
        """Only a caller-supplied range turns an out-of-range threshold into an error (D-138 r3)."""
        da = _field(scale=15.0, seed=3)
        ds = marEx.extremes.identify(da, method="global_percentile", threshold_percentile=95, dimensions=DIMENSIONS).compute()
        assert ds.attrs["max_anomaly"] > 40

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

    def test_precision_is_rejected_with_exact_through_identify(self):
        """aec73e9's `_extremes_core` replaced the caller's values with None on the exact path,
        so `identify` and `preprocess_data` ignored `precision` silently instead of raising."""
        with pytest.raises(ConfigurationError, match="Parameter 'precision' cannot be used"):
            marEx.extremes.identify(
                _field(n_time=400), method="global_percentile", method_percentile="exact", precision=0.02, dimensions=DIMENSIONS
            )

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
        # min, max and the std behind the coarse-bin warning (D-142), all in that one call.
        assert len(calls[0]) == 3


class TestLogging:
    def test_the_derived_geometry_is_logged(self):
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture(level=logging.INFO)
        marEx_logger = logging.getLogger("marEx")
        # A `quiet=True` run elsewhere in the session leaves the logger at WARNING: pin INFO here.
        previous_level = marEx_logger.level
        marEx_logger.setLevel(logging.INFO)
        marEx_logger.addHandler(handler)
        try:
            resolve_bin_spec(_field(scale=10.0), None, None, 1000)
        finally:
            marEx_logger.removeHandler(handler)
            marEx_logger.setLevel(previous_level)
        assert any("Histogram bins derived from the data" in m for m in records), records
