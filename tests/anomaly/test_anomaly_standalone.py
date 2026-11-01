"""
``marEx.anomaly.compute`` as a standalone product.

The acceptance test for the reorganisation: someone who wants a smoothed climatology and
anomalies, and has no interest in extreme detection, must get a complete and saveable dataset
without ever touching the extremes stage.

That path was never exercised before the split -- the finalisation tail only ever ran on a
dataset that already carried ``thresholds`` and a cycle dimension -- so these tests cover the
shape it now has to handle: no ``thresholds`` and no ``dayofyear``.
"""

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError

from ..conftest import assert_no_failures

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


@pytest.mark.parametrize("method", METHODS)
def test_standalone_output(da, method):
    """What the anomaly stage returns on its own: the anomaly and the mask, nothing from detection.

    The anomaly is finite wherever the mask is true, including day of year 366, which takes every
    year's 31 December, so even a ``window_years`` with no leap year inside it (2001-2003 before
    2004) has a baseline for a leap year's last day. The land cell stays masked out, and the
    recorded steps describe only the anomaly stage.
    """
    ds = marEx.anomaly.compute(da, **_kwargs(method))

    assert "dat_anomaly" in ds.data_vars
    assert "mask" in ds.data_vars
    for detection_var in ("extreme_events", "thresholds", "extreme_events_stn", "thresholds_stn"):
        assert detection_var not in ds.data_vars, f"{method}: unexpected detection output '{detection_var}'"
    # No cycle-index dimension, which only a seasonal threshold creates.
    assert "dayofyear" not in ds.dims

    steps = ds.attrs["preprocessing_steps"]
    assert steps, f"{method}: no preprocessing_steps recorded"
    assert ds.attrs["method_anomaly"] == method
    assert "method_extreme" not in ds.attrs
    joined = " ".join(steps).lower()
    for detection_word in ("percentile", "threshold", "day-of-year"):
        assert detection_word not in joined, f"{method}: steps mention detection ('{detection_word}')"

    ds = ds.compute()
    valid = ds.mask.values
    assert valid.any(), f"{method}: mask marks nothing valid"
    assert np.isfinite(ds.dat_anomaly.values[:, valid]).all(), f"{method}: non-finite anomaly inside the valid region"
    assert not valid[0, 0]

    if method == "shifting_baseline":
        leap_day = ds.dat_anomaly.time.dt.strftime("%Y-%m-%d").values == "2004-12-31"
        assert leap_day.sum() == 1
        assert np.isfinite(ds.dat_anomaly.values[leap_day][:, valid]).all()


@pytest.mark.parametrize("method", [pytest.param("detrend_harmonic", marks=pytest.mark.smoke), "shifting_baseline"])
def test_round_trip(da, method, tmp_path):
    """The finalisation tail produces something actually saveable, to zarr and to netCDF.

    Booleans and None in attrs survive zarr but break ``to_netcdf``, which is why the tail
    coerces them; a standalone anomaly dataset must get that too.
    """
    ds = marEx.anomaly.compute(da, **_kwargs(method))
    anomaly, mask = ds.dat_anomaly.values, ds.mask.values

    store = tmp_path / f"anomaly_{method}.zarr"
    ds.to_zarr(store)
    reloaded = xr.open_zarr(store)
    np.testing.assert_array_equal(reloaded.dat_anomaly.values, anomaly)
    np.testing.assert_array_equal(reloaded.mask.values, mask)

    path = tmp_path / f"anomaly_{method}.nc"
    ds.to_netcdf(path)
    with xr.open_dataset(path) as reloaded:
        np.testing.assert_array_equal(reloaded.dat_anomaly.values, anomaly)


def test_anomaly_matches_the_full_pipeline_bit_for_bit(da):
    """The chainer must not be a different implementation of the same stage."""
    standalone = marEx.anomaly.compute(da, method="detrend_harmonic", dask_chunks={"time": 50}).compute()
    chained = marEx.preprocess_data(
        da,
        method_anomaly="detrend_harmonic",
        method_extreme="global_percentile",
        dask_chunks={"time": 50},
    ).compute()

    np.testing.assert_array_equal(standalone.dat_anomaly.values, chained.dat_anomaly.values)
    np.testing.assert_array_equal(standalone.mask.values, chained.mask.values)


def test_identify_composes_onto_the_anomaly_dataset(da):
    # extremes.identify accepts the anomaly stage's Dataset directly; that
    # composition is what makes the two functions peers rather than stages.
    anomalies = marEx.anomaly.compute(da, method="detrend_harmonic", dask_chunks={"time": 50})
    events = marEx.extremes.identify(anomalies, method="global_percentile", dask_chunks={"time": 50})

    assert "extreme_events" in events.data_vars
    assert "thresholds" in events.data_vars
    # The anomaly stage's variables are carried through, not dropped.
    assert "dat_anomaly" in events.data_vars
    assert "mask" in events.data_vars


def test_rejected_configurations(da):
    """Failures the split turned from silent no-ops into errors.

    Standardisation comes from the harmonic fit's rolling STD, so asking for it elsewhere used to
    return a dataset silently missing ``dat_stn``.
    """
    rows = [
        (
            f"standardise with {method}",
            lambda m=method: marEx.anomaly.compute(
                da, method=m, standardise=True, **({"window_years": WINDOW_YEARS} if m == "shifting_baseline" else {})
            ),
            ConfigurationError,
            "standardise",
        )
        for method in ("shifting_baseline", "fixed_baseline", "detrend_fixed_baseline")
    ]
    rows += [
        (
            "reference_period with detrend_harmonic",
            lambda: marEx.anomaly.compute(da, method="detrend_harmonic", reference_period=(2001, 2003)),
            ConfigurationError,
            "reference_period",
        ),
        ("unknown method", lambda: marEx.anomaly.compute(da, method="not_a_method"), ConfigurationError, "Unknown anomaly method"),
        ("numpy input", lambda: marEx.anomaly.compute(_gridded(nt=400).compute(), method="detrend_harmonic"), Exception, "Dask"),
    ]
    failures = []
    for case, call, exc, match in rows:
        try:
            call()
        except exc as e:
            if match not in str(e):
                failures.append(f"{case}: {type(e).__name__} without {match!r}: {e}")
        except Exception as e:  # noqa: BLE001 - a wrong exception type is a row failure
            failures.append(f"{case}: raised {type(e).__name__} instead of {exc.__name__}: {e}")
        else:
            failures.append(f"{case}: did not raise")
    assert_no_failures(failures)
