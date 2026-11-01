"""The histogram's bin geometry: derived range, symmetric edges, the uint16 ceiling, warnings, regrow.

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

The bins themselves are finite and symmetric about zero, which is what makes a low-tail
percentile resolvable at all. Their positive half is bit for bit the legacy positive half, and
the guard rail a threshold is clamped onto has the same value under both. Both seasonal
engines store the bin index as uint16, and NaN digitizes one past the last bin, so at most
65535 bins can be realised; float edge rounding can make the realised count one more than
``2 * max_anomaly / precision``, so the ceiling is checked on the realised count.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError
from marEx.extremes import base
from marEx.extremes.base import _derive_bin_spec, resolve_bin_spec
from marEx.extremes.histogram import _end_clips, _symmetric_bin_edges, _zero_bin_edges

from ..conftest import assert_no_failures
from .conftest import GRID_DIMS, check, legacy_bin_edges

PRECISION = 0.01
MAX_ANOMALY = 5.0
CEILING = np.iinfo(np.uint16).max
DERIVED = "Histogram bins derived from the data"


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


def _outlier_field(n_time=1095, storm=400.0, seed=4):
    """A field whose upper extreme is one far outlier, so the per-cell estimate binds."""
    da = _field(n_time=n_time, n_y=6, n_x=6, seed=seed)
    data = da.values.copy()
    data[100, 2, 2] = storm
    return da.copy(data=data).chunk({"time": 200, "lat": 3, "lon": 3})


def _heavy_or_gaussian(heavy):
    rng = np.random.default_rng(3)
    data = rng.standard_t(3, size=(3000, 4, 5)) * 0.6 if heavy else rng.normal(0.0, 1.0, size=(3000, 4, 5))
    return xr.DataArray(data.astype(np.float32), dims=("time", "lat", "lon"), name="dat_anomaly").chunk({"time": -1})


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


def _seasonal(da, **kwargs):
    """The seasonal approximate run the uint16 ceiling is reached through (deprecations silenced)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        return marEx.extremes.identify(
            da,
            method="seasonal_percentile",
            threshold_percentile=95,
            window_days=11,
            window_spatial=1,
            dimensions=GRID_DIMS,
            **kwargs,
        ).compute()


def _realised(precision, max_anomaly):
    return len(_symmetric_bin_edges(precision, max_anomaly, np.float32)) - 1


# ── resolve_bin_spec: the geometry each argument combination gets ──────────────


def test_resolve_bin_spec_geometry():
    """Every row is a pure call on a tiny field; the range is the requested tail's own extreme."""
    failures = []

    def both_supplied():
        assert resolve_bin_spec(_field(), 0.02, 3.0, 1000) == (0.02, 3.0)

    def precision_alone():
        # The derived range retires the old fixed point (0.01 alone spanned +/-5.0): the range is the data's.
        da = _field()
        assert resolve_bin_spec(da, 0.01, None) == (0.01, pytest.approx(float(da.max())))

    def max_anomaly_alone():
        precision, max_anomaly = resolve_bin_spec(_field(), None, 40.0, 1000)
        assert max_anomaly == 40.0
        assert precision == pytest.approx(0.08)

    def neither():
        da = _field(scale=10.0)
        precision, max_anomaly = resolve_bin_spec(da, None, None, 1000)
        observed = float(da.max())  # the upper tail's own extreme
        assert max_anomaly == pytest.approx(observed)
        assert precision == pytest.approx(2 * observed / 1000)

    def constant_field():
        # A zero range would give zero-width bins; fall back to the SST defaults.
        assert resolve_bin_spec(_field(scale=0.0), None, None, 1000) == (0.01, 5.0)

    def all_nan_field():
        assert resolve_bin_spec(_field() * np.nan, None, None, 1000) == (0.01, 5.0)

    def upper_tail_is_the_data_maximum():
        da = _field(scale=3.0, offset=1.0)  # skewed in sign: max and -min differ
        precision, max_anomaly = resolve_bin_spec(da, None, None)
        assert max_anomaly == pytest.approx(float(da.max()))
        assert precision == pytest.approx(2 * max_anomaly / 3000)

    def lower_tail_is_minus_the_data_minimum():
        da = _field(scale=3.0, offset=1.0)
        precision, max_anomaly = resolve_bin_spec(da, None, None, tail="lower")
        assert max_anomaly == pytest.approx(-float(da.min()))
        assert precision == pytest.approx(2 * max_anomaly / 3000)

    def large_units_get_3000_bins():
        precision, max_anomaly = resolve_bin_spec(_field(scale=1000.0), None, None)  # pressure-like, in Pa
        assert 2 * max_anomaly / precision == pytest.approx(3000)
        assert max_anomaly > 2000

    def precision_only_bin_count_follows():
        da = _field(scale=3.0)
        precision, max_anomaly = resolve_bin_spec(da, 0.01, None)
        assert precision == 0.01
        assert max_anomaly == pytest.approx(float(da.max()))

    def one_sided_max_anomaly_keeps_1000_bins():
        assert resolve_bin_spec(_field(), None, 40.0) == (pytest.approx(0.08), 40.0)

    for row in (
        both_supplied,
        precision_alone,
        max_anomaly_alone,
        neither,
        constant_field,
        all_nan_field,
        upper_tail_is_the_data_maximum,
        lower_tail_is_minus_the_data_minimum,
        large_units_get_3000_bins,
        precision_only_bin_count_follows,
        one_sided_max_anomaly_keeps_1000_bins,
    ):
        check(failures, row.__name__, row)
    assert_no_failures(failures)


def test_per_cell_estimate():
    """An outlier extreme is lowered by the per-cell estimate; a Gaussian field keeps its maximum."""
    da = _outlier_field()
    spec = _derive_bin_spec(da, None, None, None, 95, "upper", "time")
    assert spec.mode == "estimated"
    assert spec.cap == pytest.approx(400.0)
    assert spec.max_anomaly < 100  # far below the single 400 outlier
    assert spec.n_bins == pytest.approx(3000, abs=1)

    # 3 x (mean + 1.645 std) lies past a normal sample's own maximum: the estimate never binds.
    gaussian = _field(scale=2.0)
    spec = _derive_bin_spec(gaussian, None, None, None, 95, "upper", "time")
    assert spec.mode == "data"
    assert spec.max_anomaly == pytest.approx(float(gaussian.max()))


def test_bin_spec_is_rejected():
    """Degenerate bin counts, and every geometry that would realise more than 65535 bins."""
    failures = []
    rows = {f"n_bins={n}": (lambda n=n: resolve_bin_spec(_field(), None, None, n), "n_bins must be at least 2") for n in (1, 0, -5)}
    rows.update(
        {
            # Bin indices are uint16; above 65535 they wrap silently rather than fail.
            "n_bins=70000": (lambda: resolve_bin_spec(_field(), None, None, 70000), "n_bins must not exceed 65535"),
            "derived >65000 bins": (lambda: resolve_bin_spec(_field(scale=1000.0), 0.01, None), "above the 65000"),
            # precision=1e-4 over +/-5 is 100000 bins: the uint16 index wrapped and the threshold came back as 1e-4.
            "fine precision over a pinned range": (
                lambda: _seasonal(_alternating(), precision=1e-4, max_anomaly=5.0),
                "n_bins must not exceed 65535",
            ),
            "pinned one bin past the ceiling": (
                lambda: resolve_bin_spec(_alternating(), 1.0, 32768.0),
                "n_bins must not exceed 65535",
            ),
            # NaN's index 65536 would wrap to 0 and be counted in the coldest bin.
            "n_bins=65535 realises 65536": (
                lambda: _seasonal(_alternating(nan_steps=(4, 9)), n_bins=65535),
                "n_bins must not exceed 65535",
            ),
        }
    )
    for case, (call, match) in rows.items():

        def row(call=call, match=match):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                with pytest.raises(ConfigurationError, match=match):
                    call()

        check(failures, case, row)
    assert_no_failures(failures)


def test_bin_spec_warnings(marex_warnings):
    """Warn where the geometry is suspect, stay silent on its controls; never change the returned values."""
    failures = []

    def many_bins_warns():
        da = _field(scale=3.0)  # max ~12: 0.001 needs ~24000 bins
        with marex_warnings as cap:
            precision, _ = resolve_bin_spec(da, 0.001, None)
        assert precision == 0.001
        assert any("10000 bins" in m for m in cap.messages), cap.messages

    def few_bins_is_silent():
        with marex_warnings as cap:
            resolve_bin_spec(_field(scale=3.0), 0.01, None)
        assert not any("10000 bins" in m for m in cap.messages), cap.messages

    def heavy_tail_warns_and_keeps_the_derived_bins():
        # One outlier sets the derived range and so every bin: warn, never change the value.
        da = _heavy_or_gaussian(heavy=True)
        with marex_warnings as cap:
            precision, max_anomaly = resolve_bin_spec(da, None, None, 1000)
        assert any("Derived histogram bins are coarse" in m for m in cap.messages), cap.messages
        observed = float(da.max())  # the upper tail's own extreme, not max|anomaly|
        assert max_anomaly == pytest.approx(observed)
        assert precision == pytest.approx(2 * observed / 1000)

    def gaussian_is_silent():
        with marex_warnings as cap:
            resolve_bin_spec(_heavy_or_gaussian(heavy=False), None, None, 1000)
        assert not any("coarse" in m for m in cap.messages), cap.messages

    def explicit_precision_is_never_second_guessed():
        with marex_warnings as cap:
            resolve_bin_spec(_heavy_or_gaussian(heavy=True), 0.5, None, 1000)
        assert not any("coarse" in m for m in cap.messages), cap.messages

    for row in (
        many_bins_warns,
        few_bins_is_silent,
        heavy_tail_warns_and_keeps_the_derived_bins,
        gaussian_is_silent,
        explicit_precision_is_never_second_guessed,
    ):
        check(failures, row.__name__, row)
    assert_no_failures(failures)


def test_deprecated_max_anomaly_warns_and_still_pins():
    """The failure the derivation removes, still reachable when pinned explicitly: an error, and a FutureWarning."""
    da = _field(scale=15.0, seed=3)
    with pytest.warns(FutureWarning, match="`max_anomaly` is deprecated"):
        with pytest.raises(ConfigurationError, match="exceed expected range"):
            marEx.extremes.identify(
                da, method="global_percentile", threshold_percentile=95, precision=0.01, max_anomaly=5.0, dimensions=GRID_DIMS
            ).compute()


def test_deprecated_n_bins_replaces_the_target():
    da = _field(scale=3.0)
    with pytest.warns(FutureWarning, match="`n_bins` is deprecated"):
        ds = marEx.extremes.identify(da, method="global_percentile", n_bins=500, dimensions=GRID_DIMS)
    assert 2 * ds.attrs["max_anomaly"] / ds.attrs["precision"] == pytest.approx(500)


def test_derivation_runs_once_and_is_logged(marex_records):
    """Counted from the INFO line, not from ``dask.compute`` calls.

    ``dask`` is one module object, so patching ``marEx.extremes.base.dask`` also intercepts the
    histogram path's own bounds check: that count would read a legitimate compute as a double
    derivation. ``_extremes_core`` resolves once, and ``identify_extremes`` must find it done.
    """
    records = marex_records(logging.INFO)
    marEx.extremes.identify(_field(n_time=400), method="global_percentile", dimensions=GRID_DIMS)
    derived = [r.getMessage() for r in records if DERIVED in r.getMessage()]
    assert len(derived) == 1, derived

    records.clear()
    marEx.extremes.identify(_field(n_time=400), method="global_percentile", precision=0.01, max_anomaly=5.0, dimensions=GRID_DIMS)
    assert [r.getMessage() for r in records if DERIVED in r.getMessage()] == []


def test_the_derivation_fuses_its_min_and_max(monkeypatch):
    """One traversal for both, not one each.

    Safe to count ``dask.compute`` here because ``resolve_bin_spec`` is called in isolation:
    nothing else runs inside this block.
    """
    calls = []
    real = marEx.extremes.base.dask.compute

    def counting_compute(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(marEx.extremes.base.dask, "compute", counting_compute)
    resolve_bin_spec(_field(n_time=400), None, None, 1000)
    assert len(calls) == 1
    # min, max and the std behind the coarse-bin warning, all in that one call.
    assert len(calls[0]) == 3


def test_exact_path_takes_no_bins(marex_records):
    """The exact path rejects every bin argument, runs clean without one, claims no geometry and derives nothing.

    The sentinel check matters: ``precision != 0.01`` would fire on every derived run, and 0.01 / 5.0
    stopped being defaults, so passing them IS an explicit request. The slab-histogram rewrite's ``_extremes_core``
    replaced the caller's values with None, so ``identify`` ignored ``precision`` silently.
    """
    failures = []
    rows = {
        "precision": (
            lambda: marEx.extremes.identify_extremes(_field(n_time=400), method_percentile="exact", precision=0.02),
            "precision",
        ),
        "max_anomaly": (
            lambda: marEx.extremes.identify_extremes(_field(n_time=400), method_percentile="exact", max_anomaly=10.0),
            "max_anomaly",
        ),
        "precision through identify": (
            lambda: marEx.extremes.identify(
                _field(n_time=400), method="global_percentile", method_percentile="exact", precision=0.02, dimensions=GRID_DIMS
            ),
            "precision",
        ),
        "the historical default value": (
            lambda: marEx.extremes.identify_extremes(_field(n_time=400), method_percentile="exact", precision=0.01),
            "precision",
        ),
    }
    for case, (call, name) in rows.items():

        def row(call=call, name=name):
            with pytest.raises(ConfigurationError, match=f"Parameter '{name}' cannot be used"):
                call()

        check(failures, case, row)
    assert_no_failures(failures)

    extremes, _ = marEx.extremes.identify_extremes(
        _field(n_time=400), method_extreme="global_percentile", method_percentile="exact"
    )
    assert extremes.dtype == bool

    records = marex_records(logging.INFO)
    ds = marEx.extremes.identify(_field(n_time=400), method="global_percentile", method_percentile="exact", dimensions=GRID_DIMS)
    # Deriving here would cost a full pass over the anomaly for nothing.
    assert [r.getMessage() for r in records if DERIVED in r.getMessage()] == []
    # Nothing is binned, so nothing is claimed about bins. Serialised as the string "None":
    # `core/attrs.make_netcdf_safe_attrs` coerces None so the dataset stays writable to NetCDF.
    assert ds.attrs["precision"] == "None"
    assert ds.attrs["max_anomaly"] == "None"


def _regrow_run(da, monkeypatch, safety, **kwargs):
    monkeypatch.setattr(base, "_RANGE_SAFETY", safety)
    used = []
    extremes, thresholds = base.identify_extremes(
        da, method_extreme="global_percentile", threshold_percentile=99, bin_spec_out=used, **kwargs
    )
    return extremes.compute(), thresholds.compute(), used[0]


def test_regrow_equals_a_run_over_the_cap(monkeypatch, marex_warnings):
    """An estimated range a threshold reaches is regrown to the data's extreme, exactly; one it does not is left.

    At a fixed width the interior edges are the same floats whatever the range, so the range only
    matters where a threshold reaches its edge (the premise of the regrow). Each arm is compared
    with the same bin width over the cap from the start (estimate disabled).
    """
    # Saturated: 2.5 % of one cell's days at 150, so its p99 IS 150, past a 0.5x estimate.
    da = _outlier_field()
    data = da.values.copy()
    data[::40, 4, 4] = 150.0
    saturated = da.copy(data=data).chunk({"time": 200, "lat": 3, "lon": 3})
    with marex_warnings as cap:
        ext, thr, spec = _regrow_run(saturated, monkeypatch, 0.5)
    assert spec.mode == "data" and spec.max_anomaly == pytest.approx(400.0)
    assert any("recomputing the thresholds" in m for m in cap.messages), cap.messages
    ext_ref, thr_ref, spec_ref = _regrow_run(saturated, monkeypatch, 1e9, precision=spec.precision)
    assert spec_ref == spec
    xr.testing.assert_identical(thr, thr_ref)
    xr.testing.assert_identical(ext, ext_ref)

    # Unsaturated: the estimate stands, and is still bit-identical to the run over the cap.
    ext, thr, spec = _regrow_run(da, monkeypatch, 3.0)
    assert spec.mode == "estimated" and spec.max_anomaly < 100
    ext_ref, thr_ref, spec_ref = _regrow_run(da, monkeypatch, 1e9, precision=spec.precision)
    assert spec_ref.mode == "data" and spec_ref.max_anomaly == pytest.approx(400.0)
    xr.testing.assert_identical(thr, thr_ref)
    xr.testing.assert_identical(ext, ext_ref)


def test_a_seasonal_end_bin_crossing_below_the_inner_edge_is_regrown(monkeypatch):
    """The 2-D path interpolates between bin CENTRES, so a quantile crossing in the clipped end
    bin can land below that bin's inner edge. Here every window's end-bin mass is 2/30 of its
    samples, between (1-q) and 2(1-q) at q=0.95, so no threshold passes the inner edge: only a
    centre-based bound sees the saturation."""
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


def test_a_non_sst_field_gets_a_derived_range():
    """The reason the derivation exists: a variable that is not SST in kelvin (anomalies of tens, as mm/day).

    Against the EMPIRICAL per-cell percentile, not the analytic one: with 1500 samples per cell the
    sampling error of the 95th percentile is itself ~5 %. The histogram estimate must track the true
    percentile to a few of its own (derived, ~0.12-wide) bins. A Gaussian field's per-cell estimate
    (3 x 1.645 sigma) lies past the data's own maximum, so the range is that maximum and the width
    gives 3000 bins over it.
    """
    da = _field(scale=15.0, seed=3)
    ds = marEx.extremes.identify(da, method="global_percentile", threshold_percentile=95, dimensions=GRID_DIMS).compute()
    reference = np.percentile(da.compute().values, 95, axis=0)
    np.testing.assert_allclose(ds.thresholds.values, reference, atol=5 * ds.attrs["precision"])
    assert ds.attrs["max_anomaly"] > 40
    assert ds.attrs["max_anomaly"] == pytest.approx(float(da.max()), rel=1e-6)
    assert ds.attrs["precision"] == pytest.approx(2 * float(da.max()) / 3000, rel=1e-6)


# ── the edges themselves ────────────────────────────────────────────────


def test_symmetric_edge_construction():
    """Positive half bit for bit the legacy one, exactly symmetric, finite at both ends, twice the bins.

    ``arange`` evaluates ``start + i * step``, so ``-p + i*p`` is not the same float as ``p * (i - 1)``.
    The spec proposed ``precision * arange(-n, n+1)``; measured, that shifts the float32 edges by ~5e-7,
    enough to move a float32 threshold by a ULP and flip a ``>=``. If the arange-trap check ever fails,
    numpy's arange changed and the construction can be simplified.
    """
    n = int(round(MAX_ANOMALY / PRECISION))
    for dtype in (np.float64, np.float32):
        new = _symmetric_bin_edges(PRECISION, MAX_ANOMALY, dtype)
        old = legacy_bin_edges(dtype=dtype)
        zero = int(np.searchsorted(new, 0.0))
        assert np.array_equal(new[zero:], old[2:]), f"{np.dtype(dtype).name}: positive edges moved"
        assert new.dtype == old.dtype
        naive = (PRECISION * np.arange(-n, n + 1)).astype(dtype)
        assert not np.array_equal(naive, new), np.dtype(dtype).name
        assert np.allclose(naive, new, atol=1e-6), np.dtype(dtype).name
        assert np.array_equal(new, -new[::-1]), f"{np.dtype(dtype).name}: not symmetric"

    # A trailing +inf would make the top clip a no-op and the top centre infinite.
    edges = _symmetric_bin_edges(PRECISION, MAX_ANOMALY)
    assert np.isfinite(edges).all()
    assert edges[0] == pytest.approx(-MAX_ANOMALY)
    assert edges[-1] == pytest.approx(MAX_ANOMALY)
    # Task-count consequence: the tile budget halves with the bin count.
    legacy_positive_bins = len(legacy_bin_edges()) - 1 - 2  # drop the -inf bin and [-precision, 0)
    assert len(edges) - 1 == 2 * legacy_positive_bins


def test_guard_rails_and_end_clips():
    """The clamped cells keep their clamped VALUE; the bottom clip is a no-op on legacy edges, the top unchanged.

    ``bin_edges[3]`` on the legacy edges is the upper edge of the bin containing zero;
    ``_zero_bin_edges`` finds that same edge on either binning, and both come from the same
    ``arange`` element, so the clamp target does not move.
    """
    legacy = legacy_bin_edges()
    symmetric = _symmetric_bin_edges(PRECISION, MAX_ANOMALY)
    assert _zero_bin_edges(legacy)[0] == float(legacy[3])
    assert _zero_bin_edges(symmetric)[0] == _zero_bin_edges(legacy)[0]

    upper, lower = _zero_bin_edges(symmetric)
    assert lower < 0.0 < upper
    assert upper == pytest.approx(PRECISION)
    assert lower == pytest.approx(-PRECISION)

    assert _end_clips(legacy)[0] == -np.inf
    bottom, top = _end_clips(symmetric)
    assert bottom == pytest.approx(-MAX_ANOMALY + PRECISION / 2)
    assert top == pytest.approx(MAX_ANOMALY - PRECISION / 2)
    assert top == _end_clips(legacy)[1]


def test_realised_bin_count_boundary():
    """The boundary cases, derived from the edge function itself: pinned, and an explicit n_bins."""
    assert _realised(1.0, 32767.0) == CEILING - 1
    assert _realised(1.0, 32768.0) == CEILING + 1
    assert _realised(2 * 3.0 / 65534, 3.0) == 65534
    assert _realised(2 * 3.0 / 65535, 3.0) == CEILING + 1


def test_largest_geometries_run():
    """The largest pinned geometry tracks exact to a bin; n_bins=65534 runs and drops NaN, bit for bit."""
    precision, max_anomaly = 1.0, 32767.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert resolve_bin_spec(_alternating(), precision, max_anomaly) == (precision, max_anomaly)
    ds = _seasonal(_alternating(), precision=precision, max_anomaly=max_anomaly)
    # The exact-path thresholds: slots no window reaches are NaN on both paths.
    exact = marEx.extremes.identify(
        _alternating(),
        method="seasonal_percentile",
        threshold_percentile=95,
        window_days=11,
        method_percentile="exact",
        dimensions=GRID_DIMS,
    ).thresholds.compute()
    approx, exact = ds.thresholds.transpose(*exact.dims).values, exact.values
    np.testing.assert_array_equal(np.isfinite(approx), np.isfinite(exact))
    assert np.isfinite(approx).sum() > 0
    assert np.nanmax(np.abs(approx - exact)) <= precision

    # NaN steps are dropped, so the thresholds equal those of the series without them, bit for bit.
    da = _alternating(nan_steps=(4, 9))
    ds = _seasonal(da, n_bins=65534)
    assert ds.attrs["max_anomaly"] == pytest.approx(3.0)
    valid = np.isfinite(da.isel(lat=0, lon=0).values)
    without = _seasonal(da.isel(time=np.flatnonzero(valid)), n_bins=65534)
    assert without.attrs["precision"] == ds.attrs["precision"]
    assert np.isfinite(ds.thresholds.values).sum() > 0
    np.testing.assert_array_equal(ds.thresholds.values, without.thresholds.values)
