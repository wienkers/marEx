"""Characterization (golden) snapshot test for the detect preprocessing pipeline.

This pins the *current* numerical behaviour of ``marEx.preprocess_data`` for the two
histogram-based code paths so that refactors of the detect package (e.g. the
histogram-quantile chunking fix and the removal of the temp-checkpoint machinery)
can be verified to leave the scientific outputs bit-for-bit unchanged.

Two configurations are captured, each exercising a distinct quantile path:

* ``A`` -- ``detrend_harmonic`` anomaly + ``global_percentile`` (the 1D histogram
  quantile path, ``_compute_histogram_quantile_1d``).
* ``B`` -- ``shifting_baseline`` anomaly + ``seasonal_percentile`` (the 2D per-day-of-year
  histogram quantile path, ``_compute_histogram_quantile_2d``; the package default).

Both use ``method_percentile='approximate'`` (the histogram approximation, the default).

Baselines are captured to zarr stores under ``tests/data/`` (zarr rather than NetCDF
because the pipeline emits boolean dataset attributes NetCDF cannot serialise). They were
regenerated from the histogram-quantile fixes (§3.x: edge-based 1D interpolation, top-bin
clipping, unified NaN policy) and validated positively against ``np.percentile`` -- config A
threshold mean error ~0.0017 vs the true per-cell percentile. The input is the last
``N_GOLDEN_STEPS`` of the deterministic ``sst_gridded.zarr`` fixture (kept short so the golden
stores stay small), with the same masked-NaN injection as ``test_gridded_preprocessing.py``.

Determinism note: the detect pipeline is deterministic for a fixed input chunking;
the histogram counts are exact integers and the quantile interpolation is a pure
function of the histogram, so the outputs are bit-reproducible across processes.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx

DATA_DIR = Path(__file__).parent / "data"
DIMENSIONS = {"time": "time", "x": "lon", "y": "lat"}
DASK_CHUNKS = {"time": 25}
# Baselines are captured over the last ~7 years of the fixture. That keeps the golden
# zarr stores small (the full 41-year outputs were ~61 MB of binaries in a public repo)
# while still exceeding config B's 5-year shifting-baseline window. Must match the slice
# used when the goldens were regenerated.
N_GOLDEN_STEPS = 7 * 365

# ``precision`` and ``max_anomaly`` are pinned EXPLICITLY, at what were their defaults
# when these baselines were captured. Phase D auto-derives ``max_anomaly`` from the data
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

# The golden zarr stores keep their original names, and -- through Phase D -- have still
# never been regenerated. That is what makes these comparisons mean anything.
GOLDEN_STORE = {"A_harm_global": "A_harm_global", "B_shift_seasonal": "B_shift_hobday"}

# Phase D replaced the asymmetric histogram bins with symmetric ones, so the 1-D path now
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


def _load_sst():
    sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.isel(time=slice(-N_GOLDEN_STEPS, None)).persist()
    # Match the masked-NaN injection used by test_gridded_preprocessing.
    sst = sst.where(~((sst.lat == sst.lat[1]) & (sst.lon == sst.lon[1])), np.nan)
    return sst


class TestDetectGolden:
    """Golden snapshot test for the detect preprocessing pipeline."""

    @classmethod
    def setup_class(cls):
        cls.sst = _load_sst()

    def _run(self, tag):
        ds = marEx.preprocess_data(self.sst, dimensions=DIMENSIONS, dask_chunks=DASK_CHUNKS, **CONFIGS[tag])
        return ds.compute()

    @pytest.mark.parametrize("tag", list(CONFIGS))
    def test_detect_outputs_identical(self, tag, dask_client_gridded):
        """Every load-bearing detect output must match the golden baseline exactly."""
        golden = xr.open_zarr(str(DATA_DIR / f"detect_golden_{GOLDEN_STORE[tag]}.zarr"))
        result = self._run(tag)

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

    @pytest.mark.parametrize("tag", list(CONFIGS))
    def test_legacy_bins_reproduce_the_goldens(self, tag, dask_client_gridded, monkeypatch):
        """The isolating gate for Phase D: bins changed, code did not.

        Forcing the pre-Phase-D asymmetric edges back in must reproduce the stored
        baselines. Everything else in the extremes stage was rewritten around a
        ``tail`` parameter -- the guard rail became sign-aware, the bounds check moved
        into a shared helper, the histogram gained a bottom clip -- and this is what
        says none of that moved a value.

        Compared with ZERO tolerance, unlike `test_detect_outputs_identical`: the
        baselines were captured with the legacy bins, so putting the legacy bins back
        must reproduce them exactly. Measured, it does -- all four variables, both
        configs.
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
        golden = xr.open_zarr(str(DATA_DIR / f"detect_golden_{GOLDEN_STORE[tag]}.zarr"))
        result = self._run(tag)

        for var in GOLDEN_VARS:
            np.testing.assert_array_equal(
                result[var].values,
                golden[var].values,
                err_msg=f"[{tag}] '{var}' moved under the legacy bins -- a CODE change, not the bins",
            )

    @pytest.mark.parametrize("tag", list(CONFIGS))
    def test_detect_key_attributes(self, tag, dask_client_gridded):
        """Method-selection attributes must be preserved."""
        result = self._run(tag)
        assert result.attrs["method_anomaly"] == CONFIGS[tag]["method_anomaly"]
        assert result.attrs["method_extreme"] == CONFIGS[tag]["method_extreme"]
