"""Cross-mode bit-identity: lazy and streaming must equal persist, byte for byte.

This is the Phase-2 equivalence gate applied to the Phase-3 compute modes. Integer,
boolean and label outputs are compared with no tolerance, and so are the floats: nothing
in this phase reorders a reduction, so a float difference here is a bug, not rounding.

The modes differ only in *where* an intermediate lives -- cluster RAM, nowhere, or a
scratch zarr -- so any divergence points at a round-trip defect (a dtype demoted on write,
a coordinate lost, an encoding conflict) rather than at the numerics.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.detect.compute_mode import clear_staging

METHOD_COMBOS = [
    ("detrend_harmonic", "global_extreme"),
    ("detrend_harmonic", "hobday_extreme"),
    ("shifting_baseline", "hobday_extreme"),
    ("fixed_baseline", "global_extreme"),
    ("detrend_fixed_baseline", "hobday_extreme"),
]

COMPARED_VARS = ("dat_anomaly", "extreme_events", "thresholds", "mask")


def _kwargs(method_anomaly, method_extreme, dimensions, coordinates=None):
    kw = {
        "method_anomaly": method_anomaly,
        "method_extreme": method_extreme,
        "threshold_percentile": 95,
        "dimensions": dimensions,
        "dask_chunks": {"time": 25},
    }
    if coordinates is not None:
        kw["coordinates"] = coordinates
    if method_anomaly == "shifting_baseline":
        kw.update(window_year_baseline=5, smooth_days_baseline=11)
    if method_extreme == "hobday_extreme":
        kw.update(window_days_hobday=3)
    return kw


def _assert_identical(reference, candidate, label):
    """Every compared variable must match in dtype, shape, NaN pattern and value."""
    for name in COMPARED_VARS:
        if name not in reference.data_vars:
            continue
        assert name in candidate.data_vars, f"{label}: {name} missing from candidate"
        a = reference[name].compute().values
        b = candidate[name].compute().values
        assert a.dtype == b.dtype, f"{label}: {name} dtype moved {a.dtype} -> {b.dtype}"
        assert a.shape == b.shape, f"{label}: {name} shape moved {a.shape} -> {b.shape}"

        if a.dtype.kind == "f":
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            assert np.array_equal(nan_a, nan_b), f"{label}: {name} NaN pattern differs"
            assert np.array_equal(a[~nan_a], b[~nan_b]), f"{label}: {name} is not bit-identical"
        else:
            assert np.array_equal(a, b), f"{label}: {name} is not bit-identical"


class TestLazyEquivalence:
    """lazy must reproduce persist exactly, across every method combination."""

    @classmethod
    def setup_class(cls):
        path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        cls.sst = xr.open_zarr(str(path), chunks={}).to.persist()
        cls.dimensions = {"time": "time", "x": "lon", "y": "lat"}

    @pytest.mark.parametrize("method_anomaly,method_extreme", METHOD_COMBOS)
    def test_lazy_matches_persist(self, method_anomaly, method_extreme):
        kw = _kwargs(method_anomaly, method_extreme, self.dimensions)
        ref = marEx.preprocess_data(self.sst, compute_mode="persist", **kw)
        lazy = marEx.preprocess_data(self.sst, compute_mode="lazy", **kw)
        _assert_identical(ref, lazy, f"lazy/{method_anomaly}/{method_extreme}")


class TestStreamingEquivalence:
    """streaming must reproduce persist exactly, across every method combination."""

    @classmethod
    def setup_class(cls):
        path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        cls.sst = xr.open_zarr(str(path), chunks={}).to.persist()
        cls.dimensions = {"time": "time", "x": "lon", "y": "lat"}

    @pytest.mark.parametrize("method_anomaly,method_extreme", METHOD_COMBOS)
    def test_streaming_matches_persist(self, method_anomaly, method_extreme, tmp_path):
        kw = _kwargs(method_anomaly, method_extreme, self.dimensions)
        ref = marEx.preprocess_data(self.sst, compute_mode="persist", **kw)
        streamed = marEx.preprocess_data(self.sst, compute_mode="streaming", scratch_dir=str(tmp_path), **kw)
        try:
            _assert_identical(ref, streamed, f"streaming/{method_anomaly}/{method_extreme}")
        finally:
            clear_staging(streamed)


class TestUnstructuredEquivalence:
    """The unstructured path has its own coordinate handling; round-trip it too."""

    @classmethod
    def setup_class(cls):
        path = Path(__file__).parent / "data" / "sst_unstructured.zarr"
        cls.sst = xr.open_zarr(str(path), chunks={}).to.persist()

        # The fixture carries no lat/lon; attach mock ones exactly as
        # test_unstructured_preprocessing.py does, since the pipeline requires them.
        ncells = cls.sst.sizes["ncells"]
        cls.sst = cls.sst.assign_coords(
            lat=xr.DataArray(np.linspace(-90, 90, ncells), dims=["ncells"], name="lat"),
            lon=xr.DataArray(np.linspace(-180, 180, ncells), dims=["ncells"], name="lon"),
        )
        cls.dimensions = {"time": "time", "x": "ncells"}
        cls.coordinates = {"time": "time", "x": "lon", "y": "lat"}

    def test_lazy_matches_persist(self):
        kw = _kwargs("detrend_harmonic", "hobday_extreme", self.dimensions, self.coordinates)
        ref = marEx.preprocess_data(self.sst, compute_mode="persist", **kw)
        lazy = marEx.preprocess_data(self.sst, compute_mode="lazy", **kw)
        _assert_identical(ref, lazy, "lazy/unstructured")

    def test_streaming_matches_persist(self, tmp_path):
        kw = _kwargs("detrend_harmonic", "hobday_extreme", self.dimensions, self.coordinates)
        ref = marEx.preprocess_data(self.sst, compute_mode="persist", **kw)
        streamed = marEx.preprocess_data(self.sst, compute_mode="streaming", scratch_dir=str(tmp_path), **kw)
        try:
            _assert_identical(ref, streamed, "streaming/unstructured")
        finally:
            clear_staging(streamed)
