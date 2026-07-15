"""Characterization (golden) snapshot test for the detect preprocessing pipeline.

This pins the *current* numerical behaviour of ``marEx.preprocess_data`` for the two
histogram-based code paths so that refactors of the detect package (e.g. the
histogram-quantile chunking fix and the removal of the temp-checkpoint machinery)
can be verified to leave the scientific outputs bit-for-bit unchanged.

Two configurations are captured, each exercising a distinct quantile path:

* ``A`` -- ``detrend_harmonic`` anomaly + ``global_extreme`` (the 1D histogram
  quantile path, ``_compute_histogram_quantile_1d``).
* ``B`` -- ``shifting_baseline`` anomaly + ``hobday_extreme`` (the 2D per-day-of-year
  histogram quantile path, ``_compute_histogram_quantile_2d``; the package default).

Both use ``method_percentile='approximate'`` (the histogram approximation, the default).

Baselines were captured (from the pre-refactor code) to zarr stores under
``tests/data/`` -- zarr rather than NetCDF because the pre-fix pipeline emits boolean
dataset attributes that NetCDF cannot serialise (one of the bugs this work fixes).
The synthetic input is the small deterministic ``sst_gridded.zarr`` fixture already
used by ``test_gridded_preprocessing.py``, with the same masked-NaN injection.

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

CONFIGS = {
    "A_harm_global": {"method_anomaly": "detrend_harmonic", "method_extreme": "global_extreme"},
    "B_shift_hobday": {
        "method_anomaly": "shifting_baseline",
        "method_extreme": "hobday_extreme",
        "window_year_baseline": 5,
        "smooth_days_baseline": 11,
        "window_days_hobday": 3,
    },
}

# Variables whose raw arrays must match the golden baseline exactly.
GOLDEN_VARS = ["dat_anomaly", "mask", "extreme_events", "thresholds"]


def _load_sst():
    sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.persist()
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
        golden = xr.open_zarr(str(DATA_DIR / f"detect_golden_{tag}.zarr"))
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
