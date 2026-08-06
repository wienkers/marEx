"""Unit tests for the Materialiser and the compute_mode plumbing.

``compute_mode`` is the Phase 3 knob that decides how ``preprocess_data`` materialises its
intermediates. These tests cover the policy object itself and the wiring at the public
entry point; cross-mode bit-identity lives in ``test_compute_mode_equivalence.py``.
"""

import inspect
import logging
from pathlib import Path

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.detect.compute_mode import Materialiser, clear_staging, create_staging_dir
from marEx.exceptions import ConfigurationError


def _capture_pipeline_log(fn, logger_name="marEx.detect.pipeline", level=logging.INFO):
    """Run ``fn`` and return the messages its pipeline logger emitted.

    marEx configures its own logging handler, so pytest's ``caplog`` fixture captures
    nothing from it. Attaching a handler to the named logger is independent of both
    propagation and handler configuration.
    """

    class _Collector(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.DEBUG)
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    logger = logging.getLogger(logger_name)
    collector = _Collector()
    previous_level = logger.level
    logger.addHandler(collector)
    logger.setLevel(level)
    try:
        fn()
    finally:
        logger.removeHandler(collector)
        logger.setLevel(previous_level)
    return collector.messages


def _lazy_da(name="x"):
    return xr.DataArray(
        dsa.ones((10, 4), chunks=(5, 4), dtype=np.float32),
        dims=("time", "space"),
        coords={"time": np.arange(10), "space": np.arange(4)},
        name=name,
    )


class TestMaterialiserPersist:
    """The default mode materialises through both verbs."""

    def test_pin_materialises(self):
        m = Materialiser("persist")
        (out,) = m.pin(_lazy_da())
        assert out.chunks is not None
        assert float(out.sum()) == pytest.approx(40.0)

    def test_pin_one_returns_a_single_object(self):
        m = Materialiser("persist")
        out = m.pin_one(_lazy_da())
        assert isinstance(out, xr.DataArray)
        assert float(out.sum()) == pytest.approx(40.0)

    def test_pin_accepts_several_objects(self):
        m = Materialiser("persist")
        a, b = m.pin(_lazy_da("a"), _lazy_da("b"))
        assert isinstance(a, xr.DataArray) and isinstance(b, xr.DataArray)

    def test_stage_materialises_without_touching_disk(self, tmp_path):
        m = Materialiser("persist")
        out = m.stage(_lazy_da(), "anom")
        assert isinstance(out, xr.DataArray)
        assert list(tmp_path.iterdir()) == []

    def test_is_lazy_false(self):
        assert Materialiser("persist").is_lazy is False


class TestMaterialiserLazy:
    """Lazy mode is a no-op through both verbs."""

    def test_pin_returns_the_input_untouched(self):
        m = Materialiser("lazy")
        original = _lazy_da()
        (out,) = m.pin(original)
        assert out is original

    def test_stage_returns_the_input_untouched(self):
        m = Materialiser("lazy")
        original = _lazy_da()
        assert m.stage(original, "anom") is original

    def test_is_lazy_true(self):
        assert Materialiser("lazy").is_lazy is True


class TestMaterialiserValidation:
    def test_unknown_mode_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="compute_mode"):
            Materialiser("turbo")

    def test_streaming_without_staging_dir_raises(self):
        with pytest.raises(ConfigurationError, match="scratch_dir"):
            Materialiser("streaming", staging_dir=None)

    def test_streaming_with_staging_dir_is_accepted(self, tmp_path):
        m = Materialiser("streaming", staging_dir=tmp_path)
        assert m.is_lazy is True


class TestStagingDir:
    def test_create_staging_dir_is_unique_and_nested(self, tmp_path):
        a = create_staging_dir(tmp_path)
        b = create_staging_dir(tmp_path)
        assert a != b
        assert a.parent == tmp_path and a.is_dir()
        assert b.parent == tmp_path and b.is_dir()

    def test_create_staging_dir_creates_a_missing_parent(self, tmp_path):
        root = tmp_path / "does" / "not" / "exist"
        made = create_staging_dir(root)
        assert made.is_dir()

    def test_clear_staging_removes_the_directory(self, tmp_path):
        d = create_staging_dir(tmp_path)
        (d / "probe.txt").write_text("x")
        clear_staging(d)
        assert not d.exists()

    def test_clear_staging_is_idempotent(self, tmp_path):
        d = create_staging_dir(tmp_path)
        clear_staging(d)
        clear_staging(d)  # must not raise

    def test_clear_staging_on_a_dataset_without_the_attribute_is_a_noop(self):
        clear_staging(xr.Dataset())  # must not raise


class TestPipelinePlumbing:
    """compute_mode reaches preprocess_data and is validated there."""

    @classmethod
    def setup_class(cls):
        path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        cls.sst = xr.open_zarr(str(path), chunks={}).to.persist()
        cls.dimensions = {"time": "time", "x": "lon", "y": "lat"}

    def _run(self, **overrides):
        kwargs = {
            "method_anomaly": "detrend_harmonic",
            "method_extreme": "global_extreme",
            "dimensions": self.dimensions,
            "dask_chunks": {"time": 25},
        }
        kwargs.update(overrides)
        return marEx.preprocess_data(self.sst, **kwargs)

    def test_default_signature_values(self):
        sig = inspect.signature(marEx.preprocess_data)
        assert sig.parameters["compute_mode"].default == "persist"
        assert sig.parameters["scratch_dir"].default is None
        assert sig.parameters["validate"].default is True

    def test_unknown_compute_mode_raises(self):
        with pytest.raises(ConfigurationError, match="compute_mode"):
            self._run(compute_mode="turbo")

    def test_streaming_without_scratch_dir_raises(self):
        with pytest.raises(ConfigurationError, match="scratch_dir"):
            self._run(compute_mode="streaming")

    def test_persist_mode_still_produces_a_complete_dataset(self):
        ds = self._run(compute_mode="persist")
        for name in ("dat_anomaly", "extreme_events", "thresholds", "mask"):
            assert name in ds.data_vars


class _PersistRecorder:
    """Record the size of every object materialised during a call.

    A pure call *count* would be the wrong contract. Two persists are deliberately kept in
    every mode because they are bounded and buy a large saving: the harmonic model fit is
    ``(coeff, y, x)`` -- tens of MB -- and keeps the detrend a cheap elementwise
    expression, and the shifting-baseline climatology pins two 1-D coordinate arrays. What
    lazy mode must not do is pin anything that scales with the input or with
    ``366 x space``, so this records bytes and the tests bound the total.
    """

    def __init__(self, monkeypatch):
        self.sizes = []
        self._install(monkeypatch)

    def _record(self, obj):
        nbytes = getattr(obj, "nbytes", 0)
        if isinstance(nbytes, int):
            self.sizes.append(nbytes)
        return obj

    def _install(self, monkeypatch):
        import dask

        real_dask_persist = dask.persist
        real_da_persist = xr.DataArray.persist
        real_ds_persist = xr.Dataset.persist

        def dask_persist(*args, **kwargs):
            for a in args:
                self._record(a)
            return real_dask_persist(*args, **kwargs)

        def da_persist(obj, *args, **kwargs):
            self._record(obj)
            return real_da_persist(obj, *args, **kwargs)

        def ds_persist(obj, *args, **kwargs):
            self._record(obj)
            return real_ds_persist(obj, *args, **kwargs)

        monkeypatch.setattr(dask, "persist", dask_persist)
        monkeypatch.setattr(xr.DataArray, "persist", da_persist)
        monkeypatch.setattr(xr.Dataset, "persist", ds_persist)

    @property
    def total(self):
        return sum(self.sizes)

    @property
    def largest(self):
        return max(self.sizes, default=0)


class TestLazyMode:
    """lazy mode must pin nothing that scales with the input or with space."""

    @classmethod
    def setup_class(cls):
        path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        cls.sst = xr.open_zarr(str(path), chunks={}).to.persist()
        cls.dimensions = {"time": "time", "x": "lon", "y": "lat"}

    def _run(self, **overrides):
        kwargs = {
            "method_anomaly": "shifting_baseline",
            "method_extreme": "hobday_extreme",
            "window_year_baseline": 5,
            "smooth_days_baseline": 11,
            "window_days_hobday": 3,
            "dimensions": self.dimensions,
            "dask_chunks": {"time": 25},
        }
        kwargs.update(overrides)
        return marEx.preprocess_data(self.sst, **kwargs)

    def test_persist_mode_pins_at_least_the_output(self):
        """Calibrates the bound below: persist mode pins on the order of the output."""
        import pytest as _pytest

        monkeypatch = _pytest.MonkeyPatch()
        try:
            rec = _PersistRecorder(monkeypatch)
            ds = self._run(compute_mode="persist")
            assert rec.total >= ds.nbytes, f"persist pinned {rec.total} < output {ds.nbytes}"
        finally:
            monkeypatch.undo()

    def test_lazy_mode_pins_almost_nothing(self, monkeypatch):
        rec = _PersistRecorder(monkeypatch)
        ds = self._run(compute_mode="lazy")
        budget = 0.01 * ds.nbytes
        assert rec.total < budget, (
            f"lazy pinned {rec.total} bytes (largest single object {rec.largest}), "
            f"budget {budget:.0f} = 1% of the {ds.nbytes}-byte output"
        )

    def test_lazy_mode_returns_dask_backed_variables(self):
        from dask.base import is_dask_collection

        ds = self._run(compute_mode="lazy")
        for name in ("dat_anomaly", "extreme_events", "thresholds"):
            assert is_dask_collection(ds[name].data), f"{name} is not dask-backed"

    def test_lazy_mode_output_is_writable(self, tmp_path):
        ds = self._run(compute_mode="lazy")
        ds.to_zarr(tmp_path / "out.zarr", mode="w")
        assert (tmp_path / "out.zarr").is_dir()

    def test_lazy_mode_does_not_sum_extremes_at_info_level(self):
        """The INFO summary sum is a full pass over the field; lazy must not pay it.

        Captured with a handler attached directly to the pipeline logger rather than with
        ``caplog``: marEx installs its own handler, so caplog sees no records at all --
        which would make both assertions below pass vacuously.
        """
        messages = _capture_pipeline_log(lambda: self._run(compute_mode="lazy"))
        assert messages, "captured no log records; the capture helper is broken"
        assert not any("extreme events identified" in m for m in messages)
        assert any("compute_mode='lazy'" in m for m in messages)

    def test_persist_mode_does_sum_extremes_at_info_level(self):
        """The counterpart: persist mode still emits the count, so the gate is real."""
        messages = _capture_pipeline_log(lambda: self._run(compute_mode="persist"))
        assert any("extreme events identified" in m for m in messages)


class TestValidateFlag:
    @classmethod
    def setup_class(cls):
        path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        cls.sst = xr.open_zarr(str(path), chunks={}).to.persist()
        cls.dimensions = {"time": "time", "x": "lon", "y": "lat"}

    def _run(self, monkeypatch, validate):
        import marEx.detect.pipeline as pipeline_mod

        called = []
        monkeypatch.setattr(pipeline_mod, "_validate_data_values", lambda *a, **k: called.append(True))
        marEx.preprocess_data(
            self.sst,
            method_anomaly="detrend_harmonic",
            method_extreme="global_extreme",
            validate=validate,
            dimensions=self.dimensions,
            dask_chunks={"time": 25},
        )
        return called

    def test_validate_false_skips_the_validation_pass(self, monkeypatch):
        assert self._run(monkeypatch, validate=False) == []

    def test_validate_true_runs_the_validation_pass(self, monkeypatch):
        assert self._run(monkeypatch, validate=True) == [True]


class TestStreamingMode:
    @classmethod
    def setup_class(cls):
        path = Path(__file__).parent / "data" / "sst_gridded.zarr"
        cls.sst = xr.open_zarr(str(path), chunks={}).to.persist()
        cls.dimensions = {"time": "time", "x": "lon", "y": "lat"}

    def _run(self, tmp_path, **overrides):
        kwargs = {
            "method_anomaly": "detrend_harmonic",
            "method_extreme": "global_extreme",
            "compute_mode": "streaming",
            "scratch_dir": str(tmp_path),
            "dimensions": self.dimensions,
            "dask_chunks": {"time": 25},
        }
        kwargs.update(overrides)
        return marEx.preprocess_data(self.sst, **kwargs)

    def test_staging_dir_is_created_and_recorded(self, tmp_path):
        ds = self._run(tmp_path)
        recorded = ds.attrs["marex_staging_dir"]
        assert Path(recorded).is_dir()
        assert Path(recorded).parent == tmp_path
        clear_staging(ds)

    def test_anomaly_and_thresholds_are_staged_to_zarr(self, tmp_path):
        ds = self._run(tmp_path)
        staged = Path(ds.attrs["marex_staging_dir"])
        names = {p.name for p in staged.iterdir()}
        assert "dat_anomaly.zarr" in names
        assert "thresholds.zarr" in names
        clear_staging(ds)

    def test_staging_survives_return_so_output_is_writable(self, tmp_path):
        """The returned Dataset reads lazily from the staged store; it must still exist."""
        ds = self._run(tmp_path)
        out = tmp_path / "out.zarr"
        ds.to_zarr(out, mode="w")
        assert out.is_dir()
        clear_staging(ds)

    def test_clear_staging_removes_the_directory(self, tmp_path):
        ds = self._run(tmp_path)
        staged = Path(ds.attrs["marex_staging_dir"])
        clear_staging(ds)
        assert not staged.exists()

    def test_two_runs_do_not_collide(self, tmp_path):
        a = self._run(tmp_path)
        b = self._run(tmp_path)
        assert a.attrs["marex_staging_dir"] != b.attrs["marex_staging_dir"]
        clear_staging(a)
        clear_staging(b)

    def test_streaming_output_is_netcdf_safe(self, tmp_path):
        """attrs added by streaming must not break to_netcdf (booleans/None do)."""
        ds = self._run(tmp_path)
        ds.to_netcdf(tmp_path / "out.nc")
        clear_staging(ds)

    def test_streaming_mode_pins_almost_nothing(self, monkeypatch, tmp_path):
        rec = _PersistRecorder(monkeypatch)
        ds = self._run(tmp_path)
        budget = 0.01 * ds.nbytes
        assert rec.total < budget, f"streaming pinned {rec.total} bytes, budget {budget:.0f}"
        clear_staging(ds)
