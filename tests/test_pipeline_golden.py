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

CONFIGS = {
    "A_harm_global": {"method_anomaly": "detrend_harmonic", "method_extreme": "global_percentile"},
    "B_shift_seasonal": {
        "method_anomaly": "shifting_baseline",
        "method_extreme": "seasonal_percentile",
        "window_years": 5,
        "smooth_days": 11,
        "window_days": 3,
    },
}

# The golden zarr stores keep their original names: the baselines are byte-for-byte
# unchanged by the reorganisation, and regenerating them would destroy the evidence
# that nothing moved.
GOLDEN_STORE = {"A_harm_global": "A_harm_global", "B_shift_seasonal": "B_shift_hobday"}

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
            np.testing.assert_array_equal(
                result[var].values,
                golden[var].values,
                err_msg=f"[{tag}] variable '{var}' differs from golden baseline",
            )

    @pytest.mark.parametrize("tag", list(CONFIGS))
    def test_detect_key_attributes(self, tag, dask_client_gridded):
        """Method-selection attributes must be preserved."""
        result = self._run(tag)
        assert result.attrs["method_anomaly"] == CONFIGS[tag]["method_anomaly"]
        assert result.attrs["method_extreme"] == CONFIGS[tag]["method_extreme"]
