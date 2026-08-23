"""
``marEx.anomaly.compute`` as a standalone product.

The acceptance test for the reorganisation: someone who wants a smoothed
climatology and anomalies, and has no interest in extreme detection, must get a
complete and saveable dataset without ever touching the extremes stage.

That path was never exercised before the split -- the finalisation tail only
ever ran on a dataset that already carried ``thresholds`` and a cycle
dimension -- so these tests cover the shape it now has to handle: no
``thresholds`` and no ``dayofyear``.
"""

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError

METHODS = ["detrend_harmonic", "fixed_baseline", "detrend_fixed_baseline", "shifting_baseline"]

# shifting_baseline drops the first window_years years, so the series must be
# comfortably longer than the window used below.
N_YEARS = 8
WINDOW_YEARS = 3


def _gridded(nt=N_YEARS * 365, ny=6, nx=8, seed=0):
    """Daily gridded field with a seasonal cycle, a trend, and permanent land."""
    rng = np.random.default_rng(seed)
    time = xr.date_range("2000-01-01", periods=nt, freq="D")
    seasonal = 3 * np.sin(2 * np.pi * np.arange(nt) / 365.25)
    trend = np.linspace(0, 1.5, nt)
    values = rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.5 + 285
    values += (seasonal + trend).astype(np.float32)[:, None, None]
    values[:, 0, 0] = np.nan  # a permanently-invalid cell, i.e. land
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": np.linspace(-60, 60, ny, dtype=np.float32),
            "lon": np.linspace(-180, 150, nx, dtype=np.float32),
        },
        name="t2m",
    ).chunk({"time": 100, "lat": -1, "lon": -1})


@pytest.fixture(scope="module")
def da():
    return _gridded()


def _kwargs(method):
    extra = {"window_years": WINDOW_YEARS, "smooth_days": 11} if method == "shifting_baseline" else {}
    return {"method": method, "dask_chunks": {"time": 50}, **extra}


class TestStandaloneOutput:
    """What the anomaly stage returns on its own."""

    @pytest.mark.parametrize("method", METHODS)
    def test_returns_anomaly_and_mask_only(self, da, method):
        ds = marEx.anomaly.compute(da, **_kwargs(method))

        assert "dat_anomaly" in ds.data_vars
        assert "mask" in ds.data_vars
        # No detection ever ran, so none of its outputs may appear.
        for detection_var in ("extreme_events", "thresholds", "extreme_events_stn", "thresholds_stn"):
            assert detection_var not in ds.data_vars, f"{method}: unexpected detection output '{detection_var}'"
        # And no cycle-index dimension, which only a seasonal threshold creates.
        assert "dayofyear" not in ds.dims

    @pytest.mark.parametrize("method", METHODS)
    def test_anomaly_is_finite_where_the_mask_is_true(self, da, method):
        ds = marEx.anomaly.compute(da, **_kwargs(method)).compute()
        valid = ds.mask.values
        assert valid.any(), f"{method}: mask marks nothing valid"

        # Day-of-year 366 is excluded deliberately. A rolling climatology can only fill
        # that group from a leap year inside its own window, so with a short
        # window_years a leap day legitimately has no baseline and yields NaN. Measured
        # here: exactly one timestep, 2004-12-31, whose preceding three years
        # (2001-2003) contain no leap day. This is a property of the method, not of the
        # split -- the shipped default of window_years=15 always spans a leap year.
        not_leap_day = ds.dat_anomaly.time.dt.dayofyear.values != 366
        anomaly = ds.dat_anomaly.values[not_leap_day][:, valid]
        assert np.isfinite(anomaly).all(), f"{method}: non-finite anomaly inside the valid region"

        # The land cell stays masked out.
        assert not valid[0, 0]

    def test_short_window_leaves_only_the_leap_day_unbaselined(self, da):
        """Pin the exception above, so a broader NaN leak cannot hide behind it."""
        ds = marEx.anomaly.compute(da, **_kwargs("shifting_baseline")).compute()
        valid = ds.mask.values
        non_finite_rows = ~np.isfinite(ds.dat_anomaly.values[:, valid]).all(axis=1)

        affected = ds.dat_anomaly.time.dt.dayofyear.values[non_finite_rows]
        assert set(affected.tolist()) <= {366}, f"NaN on non-leap days: {sorted(set(affected.tolist()))}"

    @pytest.mark.parametrize("method", METHODS)
    def test_preprocessing_steps_describe_only_the_anomaly_stage(self, da, method):
        ds = marEx.anomaly.compute(da, **_kwargs(method))
        steps = ds.attrs["preprocessing_steps"]

        assert steps, f"{method}: no preprocessing_steps recorded"
        assert ds.attrs["method_anomaly"] == method
        assert "method_extreme" not in ds.attrs
        joined = " ".join(steps).lower()
        for detection_word in ("percentile", "threshold", "day-of-year"):
            assert detection_word not in joined, f"{method}: steps mention detection ('{detection_word}')"


class TestRoundTrip:
    """The finalisation tail has to produce something actually saveable."""

    @pytest.mark.parametrize("method", ["detrend_harmonic", "shifting_baseline"])
    def test_zarr_round_trip(self, da, method, tmp_path):
        ds = marEx.anomaly.compute(da, **_kwargs(method))
        store = tmp_path / f"anomaly_{method}.zarr"
        ds.to_zarr(store)

        reloaded = xr.open_zarr(store)
        np.testing.assert_array_equal(reloaded.dat_anomaly.values, ds.dat_anomaly.values)
        np.testing.assert_array_equal(reloaded.mask.values, ds.mask.values)

    @pytest.mark.parametrize("method", ["detrend_harmonic", "shifting_baseline"])
    def test_netcdf_round_trip(self, da, method, tmp_path):
        # Booleans and None in attrs survive Zarr but break to_netcdf, which is why
        # the tail coerces them. A standalone anomaly dataset must get that too.
        ds = marEx.anomaly.compute(da, **_kwargs(method))
        path = tmp_path / f"anomaly_{method}.nc"
        ds.to_netcdf(path)

        reloaded = xr.open_dataset(path)
        np.testing.assert_array_equal(reloaded.dat_anomaly.values, ds.dat_anomaly.values)


class TestChainingEquivalence:
    """The chainer must not be a different implementation of the same stage."""

    def test_anomaly_matches_the_full_pipeline_bit_for_bit(self, da):
        standalone = marEx.anomaly.compute(da, method="detrend_harmonic", dask_chunks={"time": 50}).compute()
        chained = marEx.preprocess_data(
            da,
            method_anomaly="detrend_harmonic",
            method_extreme="global_percentile",
            dask_chunks={"time": 50},
        ).compute()

        np.testing.assert_array_equal(standalone.dat_anomaly.values, chained.dat_anomaly.values)
        np.testing.assert_array_equal(standalone.mask.values, chained.mask.values)

    def test_identify_composes_onto_the_anomaly_dataset(self, da):
        # extremes.identify accepts the anomaly stage's Dataset directly; that
        # composition is what makes the two functions peers rather than stages.
        anomalies = marEx.anomaly.compute(da, method="detrend_harmonic", dask_chunks={"time": 50})
        events = marEx.extremes.identify(anomalies, method="global_percentile", dask_chunks={"time": 50})

        assert "extreme_events" in events.data_vars
        assert "thresholds" in events.data_vars
        # The anomaly stage's variables are carried through, not dropped.
        assert "dat_anomaly" in events.data_vars
        assert "mask" in events.data_vars


class TestRejectedConfigurations:
    """Failures the split turned from silent no-ops into errors."""

    @pytest.mark.parametrize("method", ["shifting_baseline", "fixed_baseline", "detrend_fixed_baseline"])
    def test_standardise_requires_the_harmonic_method(self, da, method):
        # Standardisation comes from the harmonic fit's rolling STD, so asking for it
        # elsewhere used to return a dataset silently missing dat_stn.
        with pytest.raises(ConfigurationError, match="standardise"):
            marEx.anomaly.compute(
                da, method=method, standardise=True, **{"window_years": WINDOW_YEARS} if method == "shifting_baseline" else {}
            )

    def test_reference_period_rejected_for_non_fixed_methods(self, da):
        with pytest.raises(ConfigurationError, match="reference_period"):
            marEx.anomaly.compute(da, method="detrend_harmonic", reference_period=(2001, 2003))

    def test_unknown_method_is_rejected(self, da):
        with pytest.raises(ConfigurationError, match="Unknown anomaly method"):
            marEx.anomaly.compute(da, method="not_a_method")

    def test_non_dask_input_is_rejected(self):
        with pytest.raises(Exception, match="Dask"):
            marEx.anomaly.compute(_gridded(nt=400).compute(), method="detrend_harmonic")
