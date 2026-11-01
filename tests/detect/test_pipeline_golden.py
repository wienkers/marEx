"""Characterization (golden) snapshot test for the detect preprocessing pipeline.

This pins the *current* numerical behaviour of ``marEx.preprocess_data`` for the two
histogram-based code paths so that refactors of the detect stage can be verified to leave the
scientific outputs bit-for-bit unchanged.

Two configurations are captured, each exercising a distinct quantile path:

* ``A`` -- ``detrend_harmonic`` anomaly + ``global_percentile`` (the 1D histogram
  quantile path, ``_compute_histogram_quantile_1d``).
* ``B`` -- ``shifting_baseline`` anomaly + ``seasonal_percentile`` (the 2D per-day-of-year
  histogram quantile path, ``_compute_histogram_quantile_2d``; the package default).

Both use ``method_percentile='approximate'`` (the histogram approximation, the default).

Baselines are captured to zarr stores under ``tests/data/`` (zarr rather than NetCDF
because the pipeline emits boolean dataset attributes NetCDF cannot serialise). They were
regenerated from the histogram-quantile fixes (edge-based 1D interpolation, top-bin
clipping, unified NaN policy) and validated positively against ``np.percentile`` -- config A
threshold mean error ~0.0017 vs the true per-cell percentile. The input is the last
``N_GOLDEN_STEPS`` of the deterministic ``sst_gridded.zarr`` fixture (kept short so the golden
stores stay small), with one masked cell (``load_golden_sst`` in ``conftest.py``).

Determinism note: the detect pipeline is independent of the input chunking;
the histogram counts are exact integers and the quantile interpolation is a pure
function of the histogram, so the outputs are bit-reproducible across processes.

The loader's no-futures tripwire lives with every other detect loader's, in
``test_compute_mode_equivalence.py::test_loaders_carry_no_futures``.
"""

import numpy as np
import pytest
import xarray as xr

import marEx

from .conftest import DATA_DIR, GRID_DIMS, load_golden_sst

DASK_CHUNKS = {"time": 25}

# ``precision`` and ``max_anomaly`` are pinned EXPLICITLY, at what were their defaults
# when these baselines were captured. The bin range is now derived from the data
# when it is left unset, so pinning them here keeps the baselines a characterisation of
# the numerics rather than of the current default, and means a later change to the
# derivation cannot silently move them.
_BINS = {"precision": 0.01, "max_anomaly": 5.0}

CONFIGS = {
    "A_harm_global": {"method_anomaly": "detrend_harmonic", "method_extreme": "global_percentile", **_BINS},
    "B_shift_seasonal": {
        "method_anomaly": "shifting_baseline",
        "method_extreme": "seasonal_percentile",
        "window_years": 5,
        "smooth_days": 11,
        "window_days": 3,
        **_BINS,
    },
}

# The golden zarr stores keep their original names. Config A has never been regenerated.
# Config B's arrays were rewritten ONCE, 2026-09-16: the old values encoded the input's
# time chunk boundaries, and detect is now chunk-invariant. Only B moved (dat_anomaly max 7.6e-4
# at 543,528 of 585,600; extreme_events 59 of 585,600; thresholds one 0.01 bin at 9,493 of
# 292,800), and the rewritten values are bit-identical to the PRE-fix code run on a time-whole
# input, so the change is the chunk dependence and nothing else.
# Config B was rewritten AGAIN 2026-10-05: the smoothing's rolling mean moved from bottleneck's
# float32 running sum to xarray's numpy path on every xarray release (dat_anomaly max 7.3e-4 at 543,605 of
# 585,600; extreme_events 61; thresholds one 0.01 bin at 9,590 of 292,800; mask 0).
# Config B was rewritten again 2026-10-18: the seasonal spatial window wrapped longitude on any
# lat/lon grid, so on this 10-degree box the edge columns pooled samples from the opposite edge. It now
# wraps only a grid spanning 360 degrees. Only the outermost window_spatial // 2 columns on each side moved
# (lon 0, 1, 38, 39): thresholds at 28,505 of 292,800; extreme_events 1,350 of 585,600; dat_anomaly and
# mask 0.
# Config B was rewritten again 2026-10-21: the day-366 climatology now pools every year's 31 December
# instead of the leap years' alone, which left a leap year's 31 December with a one-sample climatology
# carrying the trend. Every moved element sits on the predicted day-366 set: dat_anomaly 799 of 585,600,
# all on 2020-12-31; thresholds 1,874 of 292,800, on day of year 365, 366 and 1 only; extreme_events 118
# of 585,600, on dates with those days of year only; mask 0. Never regenerate without a deliberate decision.
GOLDEN_STORE = {"A_harm_global": "A_harm_global", "B_shift_seasonal": "B_shift_hobday"}

# The histogram bins were changed from asymmetric to symmetric ones, so the 1-D path now
# accumulates its CDF over twice as many bins. Config A's float64 `thresholds` therefore
# differ from the baseline by pure summation round-off: MEASURED at 732 of 800 cells, max
# 1.13e-14, against a method whose own bin precision is 0.01. `extreme_events` is
# unaffected at all 2,044,000 elements, and config B -- the 2-D seasonal path, the package
# default -- is bit-for-bit identical on every variable.
#
# This is the ONLY tolerance in this module. Everything else is compared exactly, and
# nothing hides under 1e-14 in a field of this magnitude. The evidence that even this is
# the bins and not the code is `test_legacy_bins_reproduce_the_goldens` below, which is
# exact.
FLOAT_ATOL = {"A_harm_global": {"thresholds": 2e-14}}

# Variables whose raw arrays must match the golden baseline exactly.
GOLDEN_VARS = ["dat_anomaly", "mask", "extreme_events", "thresholds"]


def _run(tag):
    ds = marEx.preprocess_data(load_golden_sst(), dimensions=GRID_DIMS, dask_chunks=DASK_CHUNKS, **CONFIGS[tag])
    return ds.compute()


def _golden(tag):
    return xr.open_zarr(str(DATA_DIR / f"detect_golden_{GOLDEN_STORE[tag]}.zarr"))


@pytest.mark.parametrize(
    "tag",
    [pytest.param(t, marks=pytest.mark.smoke) if t == "A_harm_global" else t for t in CONFIGS],
)
def test_detect_outputs_identical(tag, client):
    """Every load-bearing detect output must match the golden baseline exactly, and the run records its methods."""
    golden = _golden(tag)
    result = _run(tag)

    for var in GOLDEN_VARS:
        assert var in result.data_vars, f"[{tag}] missing variable '{var}'"
        atol = FLOAT_ATOL.get(tag, {}).get(var, 0.0)
        if atol:
            np.testing.assert_allclose(
                result[var].values,
                golden[var].values,
                atol=atol,
                rtol=0,
                equal_nan=True,
                err_msg=f"[{tag}] variable '{var}' differs from golden baseline by more than round-off",
            )
        else:
            np.testing.assert_array_equal(
                result[var].values,
                golden[var].values,
                err_msg=f"[{tag}] variable '{var}' differs from golden baseline",
            )

    # Method-selection attributes must be preserved (asserted on the golden run, not a rerun).
    assert result.attrs["method_anomaly"] == CONFIGS[tag]["method_anomaly"]
    assert result.attrs["method_extreme"] == CONFIGS[tag]["method_extreme"]


@pytest.mark.parametrize("tag", list(CONFIGS))
def test_legacy_bins_reproduce_the_goldens(tag, client, monkeypatch):
    """The isolating test for the symmetric bins: bins changed, code did not.

    Forcing the old asymmetric edges back in must reproduce the stored
    baselines. Everything else in the extremes stage was rewritten around a
    ``tail`` parameter -- the guard rail became sign-aware, the bounds check moved
    into a shared helper, the histogram gained a bottom clip -- and this is what
    says none of that moved a value.

    Compared with ZERO tolerance, unlike `test_detect_outputs_identical`: the
    baselines were captured with the legacy bins, so putting the legacy bins back
    must reproduce them exactly. Measured, it does -- all four variables, both
    configs. A separate run is required: the bins are monkeypatched.
    """
    from marEx.extremes import histogram as H

    def legacy_bin_edges(precision, max_anomaly, dtype=np.float64):
        if dtype == np.float32:
            return np.concatenate(
                [[-np.inf], np.arange(-precision, max_anomaly + precision, precision, dtype=np.float32)],
                dtype=np.float32,
            )
        return np.concatenate([[-np.inf], np.arange(-precision, max_anomaly + precision, precision)])

    monkeypatch.setattr(H, "_symmetric_bin_edges", legacy_bin_edges)
    golden = _golden(tag)
    result = _run(tag)

    for var in GOLDEN_VARS:
        np.testing.assert_array_equal(
            result[var].values,
            golden[var].values,
            err_msg=f"[{tag}] '{var}' moved under the legacy bins -- a CODE change, not the bins",
        )
