"""The Materialiser and the compute_mode plumbing: what each mode pins, writes, logs and cleans up.

``compute_mode`` is the knob that decides how ``preprocess_data`` materialises its
intermediates. These tests cover the policy object itself, the wiring at the public entry
point, and the graph structure streaming hands the bounds check; cross-mode bit-identity lives
in ``test_compute_mode_equivalence.py``.

Several tests are tables (one behaviour, several inputs): every row runs, failures are
collected with the row's case id, and the test fails once at the end.

Laziness is asserted by counting the BYTES handed to ``persist`` (``PersistRecorder``), never by
``is_dask_collection``: two persists are deliberately kept in every mode because they are bounded
and buy a large saving (the harmonic fit, ``(coeff, y, x)``, and the shifting-baseline
climatology's two 1-D coordinate arrays). What lazy and streaming must not do is pin anything that
scales with the input or with ``366 x space``.
"""

import inspect
from pathlib import Path

import dask.array as dsa
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.base import is_dask_collection
from dask.core import istask

import marEx
import marEx.extremes.histogram as H
from marEx.core.compute_mode import Materialiser, clear_staging, create_staging_dir
from marEx.exceptions import ConfigurationError
from marEx.extremes.histogram import _compute_histogram_quantile_1d, _compute_histogram_quantile_2d

from ..conftest import assert_no_failures
from .conftest import GRID_DIMS, PersistRecorder, capture_pipeline_log, check, on_client, open_sst_gridded


def _lazy_da(name="x"):
    return xr.DataArray(
        dsa.ones((10, 4), chunks=(5, 4), dtype=np.float32),
        dims=("time", "space"),
        coords={"time": np.arange(10), "space": np.arange(4)},
        name=name,
    )


def _unmaterialised(obj):
    """Graph entries that are still tasks."""
    return [k for k, v in dict(obj.data.__dask_graph__()).items() if istask(v)]


def test_materialiser_verbs(tmp_path):
    """Table over persist / lazy / streaming x pin / pin_one / stage / is_lazy.

    persist materialises through every verb and stages in memory: the staged block holds no task
    and no graph layer reads it back from a store. lazy returns its input untouched. streaming
    with a staging directory is accepted and lazy.
    """
    failures = []

    def persist_pin():
        (out,) = Materialiser("persist").pin(_lazy_da())
        assert out.chunks is not None
        assert float(out.sum()) == pytest.approx(40.0)

    def persist_pin_one():
        out = Materialiser("persist").pin_one(_lazy_da())
        assert isinstance(out, xr.DataArray)
        assert float(out.sum()) == pytest.approx(40.0)

    def persist_pin_several():
        a, b = Materialiser("persist").pin(_lazy_da("a"), _lazy_da("b"))
        assert isinstance(a, xr.DataArray) and isinstance(b, xr.DataArray)

    def persist_stage():
        out = Materialiser("persist").stage(_lazy_da(), "anom")
        assert isinstance(out, xr.DataArray)
        assert float(out.sum()) == pytest.approx(40.0)
        unmaterialised = _unmaterialised(out)
        assert unmaterialised == [], f"staged block still holds tasks: {unmaterialised[:5]}"
        readers = [name for name in out.data.__dask_graph__().layers if "zarr" in name or "open_dataset" in name]
        assert readers == [], f"persist staging reads from a store: {readers}"

    def persist_is_lazy():
        assert Materialiser("persist").is_lazy is False

    def lazy_pin():
        original = _lazy_da()
        (out,) = Materialiser("lazy").pin(original)
        assert out is original

    def lazy_stage():
        original = _lazy_da()
        assert Materialiser("lazy").stage(original, "anom") is original

    def lazy_is_lazy():
        assert Materialiser("lazy").is_lazy is True

    def streaming_accepted():
        assert Materialiser("streaming", staging_dir=tmp_path).is_lazy is True

    for case, row in {
        "persist-pin": persist_pin,
        "persist-pin_one": persist_pin_one,
        "persist-pin-several": persist_pin_several,
        "persist-stage": persist_stage,
        "persist-is_lazy": persist_is_lazy,
        "lazy-pin": lazy_pin,
        "lazy-stage": lazy_stage,
        "lazy-is_lazy": lazy_is_lazy,
        "streaming-with-staging-dir": streaming_accepted,
    }.items():
        check(failures, case, row)
    assert_no_failures(failures)


def _ragged_time_da(name="x"):
    """A (time, space) DataArray with a ragged final time chunk, (2, 2, 2, 3).

    Built the way ``fill_time_gaps`` produces one in practice: concatenating two
    differently-chunked pieces leaves the boundary between them.
    """
    first = xr.DataArray(np.ones((6, 4), dtype=np.float32), dims=("time", "space")).chunk({"time": 2})
    second = xr.DataArray(np.ones((3, 4), dtype=np.float32), dims=("time", "space")).chunk({"time": 3})
    combined = xr.concat([first, second], dim="time")
    combined.name = name
    combined.coords["time"] = np.arange(9)
    combined.coords["space"] = np.arange(4)
    return combined


def test_stage_preserve_chunks(tmp_path):
    """``stage(..., preserve_chunks=True)`` restores chunk boundaries streaming moved.

    Streaming staging rechunks to the largest chunk per dimension before writing (zarr
    rejects ragged chunks), so a ragged input comes back with different boundaries. The flag
    is opt-in and defaults to False, so detect's documented staged-chunk-layout difference is
    untouched; only callers that need boundaries to hold (the tracker's merge loop) pass True.
    """
    m = Materialiser("streaming", staging_dir=tmp_path)

    ragged = _ragged_time_da()
    assert ragged.chunks[0] == (2, 2, 2, 3)  # the fixture really is ragged
    assert m.stage(ragged, "ragged_kept", preserve_chunks=True).chunks == ragged.chunks

    staged = m.stage(_ragged_time_da(), "ragged")  # preserve_chunks defaults to False
    assert staged.chunks != ragged.chunks
    assert staged.chunks[0] == (3, 3, 3)

    uniform = xr.DataArray(np.ones((9, 4), dtype=np.float32), dims=("time", "space")).chunk({"time": 3})
    uniform.name = "u"
    uniform.coords["time"] = np.arange(9)
    uniform.coords["space"] = np.arange(4)
    assert m.stage(uniform, "uniform_a", preserve_chunks=True).chunks == uniform.chunks
    assert m.stage(uniform, "uniform_b", preserve_chunks=False).chunks == uniform.chunks


def _pipeline_kwargs(**overrides):
    kwargs = {
        "method_anomaly": "detrend_harmonic",
        "method_extreme": "global_percentile",
        "dimensions": GRID_DIMS,
        "dask_chunks": {"time": 25},
    }
    kwargs.update(overrides)
    return kwargs


def test_mode_validation(tmp_path):
    """Table: what the Materialiser and preprocess_data reject, and the public signature's defaults."""
    sst = open_sst_gridded()
    failures = []

    def materialiser_unknown():
        with pytest.raises(ConfigurationError, match="compute_mode"):
            Materialiser("turbo")

    def materialiser_streaming_without_dir():
        with pytest.raises(ConfigurationError, match="scratch_dir"):
            Materialiser("streaming", staging_dir=None)

    def pipeline_unknown():
        with pytest.raises(ConfigurationError, match="compute_mode"):
            marEx.preprocess_data(sst, **_pipeline_kwargs(compute_mode="turbo"))

    def pipeline_streaming_without_dir():
        with pytest.raises(ConfigurationError, match="scratch_dir"):
            marEx.preprocess_data(sst, **_pipeline_kwargs(compute_mode="streaming"))

    def signature_defaults():
        sig = inspect.signature(marEx.preprocess_data)
        assert sig.parameters["compute_mode"].default == "persist"
        assert sig.parameters["scratch_dir"].default is None
        assert sig.parameters["validate"].default is True

    for case, row in {
        "Materialiser-unknown-mode": materialiser_unknown,
        "Materialiser-streaming-without-staging-dir": materialiser_streaming_without_dir,
        "preprocess_data-unknown-mode": pipeline_unknown,
        "preprocess_data-streaming-without-scratch_dir": pipeline_streaming_without_dir,
        "signature-defaults": signature_defaults,
    }.items():
        check(failures, case, row)
    assert_no_failures(failures)


def test_staging_dir_lifecycle(tmp_path):
    """create (unique, nested, missing parent made) -> clear (removes, idempotent) -> clear on a Dataset without one."""
    a = create_staging_dir(tmp_path)
    b = create_staging_dir(tmp_path)
    assert a != b
    assert a.parent == tmp_path and a.is_dir()
    assert b.parent == tmp_path and b.is_dir()

    made = create_staging_dir(tmp_path / "does" / "not" / "exist")
    assert made.is_dir()

    (a / "probe.txt").write_text("x")
    clear_staging(a)
    assert not a.exists()
    clear_staging(b)
    clear_staging(b)  # idempotent: must not raise
    assert not b.exists()

    clear_staging(xr.Dataset())  # no staging dir recorded: a no-op, must not raise


def _seasonal_kwargs(**overrides):
    return _pipeline_kwargs(
        method_anomaly="shifting_baseline",
        method_extreme="seasonal_percentile",
        window_years=5,
        smooth_days=11,
        window_days=3,
        **overrides,
    )


def test_lazy_run_pins_nothing_and_writes(tmp_path):
    """One lazy run (shifting_baseline / seasonal, the full record): it pins under 1% of its output,
    logs ``compute_mode='lazy'`` without paying the INFO event-count sum, returns dask-backed
    variables, and writes.

    The dask-backed check is not a laziness proof (the byte count is); it is the only check that
    catches an eager ``.compute()`` returning numpy, which no persist recorder sees.
    """
    sst = open_sst_gridded()
    with PersistRecorder() as rec:
        ds, messages = capture_pipeline_log(lambda: marEx.preprocess_data(sst, **_seasonal_kwargs(compute_mode="lazy")))

    budget = 0.01 * ds.nbytes
    assert rec.bytes < budget, (
        f"lazy pinned {rec.bytes} bytes (largest single object {rec.largest}), "
        f"budget {budget:.0f} = 1% of the {ds.nbytes}-byte output"
    )
    assert messages, "captured no log records; the capture helper is broken"
    assert not any("extreme events identified" in m for m in messages), "lazy paid the INFO event-count sum"
    assert any("compute_mode='lazy'" in m for m in messages)
    for name in ("dat_anomaly", "extreme_events", "thresholds"):
        assert is_dask_collection(ds[name].data), f"{name} is not dask-backed"

    ds.to_zarr(tmp_path / "out.zarr", mode="w")
    assert (tmp_path / "out.zarr").is_dir()


def test_persist_pins_the_output_and_logs_the_count():
    """The counterpart, and the calibration of the lazy bound: persist pins at least its output and logs the count."""
    sst = open_sst_gridded()
    with PersistRecorder() as rec:
        ds, messages = capture_pipeline_log(lambda: marEx.preprocess_data(sst, **_seasonal_kwargs(compute_mode="persist")))
    assert rec.bytes >= ds.nbytes, f"persist pinned {rec.bytes} < output {ds.nbytes}"
    assert any("extreme events identified" in m for m in messages)


def test_validate_flag(monkeypatch):
    """``validate=False`` skips the data-value pass; ``validate=True`` runs it once."""
    import marEx.anomaly.api as anomaly_api

    sst = open_sst_gridded()
    calls = {}
    for validate in (False, True):
        called = calls[validate] = []
        monkeypatch.setattr(anomaly_api, "_validate_data_values", lambda *a, _c=called, **k: _c.append(True))
        marEx.preprocess_data(sst, **_pipeline_kwargs(validate=validate))
    assert calls == {False: [], True: [True]}, calls


def test_streaming_run_stages_writes_and_clears(tmp_path, client):
    """One recorded streaming run carries every assertion; a second run checks that two runs do not collide.

    On the session client, so streaming's staging writes and the reads back go through
    distributed workers. The staging dir is created under ``scratch_dir`` and recorded on
    ``encoding`` (never ``attrs``, which a write would carry to disk after the dir is gone);
    the anomaly and thresholds are staged to zarr; the returned Dataset, which reads lazily from
    the staged store, writes to zarr and netCDF; the run pins under 1% of its output; and
    ``clear_staging`` removes the dir.
    """
    assert on_client(client), "not computing on the session client"
    sst = open_sst_gridded()
    scratch = tmp_path / "scratch"
    with PersistRecorder() as rec:
        ds = marEx.preprocess_data(sst, **_pipeline_kwargs(compute_mode="streaming", scratch_dir=str(scratch)))
    second = None
    try:
        budget = 0.01 * ds.nbytes
        assert rec.bytes < budget, f"streaming pinned {rec.bytes} bytes, budget {budget:.0f}"

        staged = Path(ds.encoding["marex_staging_dir"])
        assert staged.is_dir()
        assert staged.parent == scratch
        names = {p.name for p in staged.iterdir()}
        assert "dat_anomaly.zarr" in names
        assert "thresholds.zarr" in names

        out = tmp_path / "out.zarr"
        ds.to_zarr(out, mode="w")
        assert out.is_dir()
        ds.to_netcdf(tmp_path / "out.nc")  # attrs added by streaming must not break netCDF (booleans/None do)

        second = marEx.preprocess_data(sst, **_pipeline_kwargs(compute_mode="streaming", scratch_dir=str(scratch)))
        assert second.encoding["marex_staging_dir"] != ds.encoding["marex_staging_dir"], "two runs share a staging dir"

        clear_staging(ds)
        assert not staged.exists(), "clear_staging must actually remove the directory"
        reopened = xr.open_zarr(out)
        assert "marex_staging_dir" not in reopened.attrs, "the written store records a staging dir that no longer exists"
    finally:
        clear_staging(ds)
        if second is not None:
            clear_staging(second)


# ── streaming pins the histogram threshold before the bounds check ─────
#
# ``_apply_threshold_bounds`` makes one eager ``dask.compute`` over the threshold. ``pin_one`` is a
# no-op in streaming mode, so that compute ran the whole histogram graph on its own and the later
# ``stage`` of the thresholds ran it a second time; staging it to zarr instead still crashed at
# 731 x 120 x 1440 on 16 x 14 GB workers, where persisting the same graph completed. What is pinned
# here is graph STRUCTURE, which bit-identity tests are blind to.


def _anomaly_fixture(n_time=400, n_y=6, n_x=7, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(n_time, n_y, n_x)).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=n_time, freq="D"),
            "lat": np.arange(n_y, dtype=np.float32),
            "lon": np.arange(n_x, dtype=np.float32),
        },
        name="da",
    )
    return da.chunk({"time": -1, "lat": 3, "lon": 7})


def test_bounds_check_reads_a_pinned_threshold(monkeypatch, tmp_path):
    """Table over the 1-D and 2-D drivers in streaming mode: the array the bounds check receives holds
    materialised blocks, carrying none of the histogram graph."""
    seen = []
    original = H._apply_threshold_bounds

    def spy(threshold, *args, **kwargs):
        seen.append(_unmaterialised(threshold))
        return original(threshold, *args, **kwargs)

    monkeypatch.setattr(H, "_apply_threshold_bounds", spy)

    def drive_2d(mat):
        da = _anomaly_fixture()
        da = da.assign_coords(dayofyear=da.time.dt.dayofyear)
        _compute_histogram_quantile_2d(da, q=0.95, window_spatial=3, dimensions=GRID_DIMS, materialiser=mat)

    def drive_1d(mat):
        _compute_histogram_quantile_1d(_anomaly_fixture(), 0.95, dim="time", materialiser=mat)

    failures = []
    for case, drive in {"2d": drive_2d, "1d": drive_1d}.items():

        def row(case=case, drive=drive):
            seen.clear()
            drive(Materialiser("streaming", create_staging_dir(str(tmp_path / case))))
            assert len(seen) == 1, f"bounds check called {len(seen)} times"
            assert seen[0] == [], f"the bounds check reads an unpinned threshold ({len(seen[0])} tasks, e.g. {seen[0][:2]})"

        check(failures, case, row)
    assert_no_failures(failures)


def _synthetic(n_time):
    rng = np.random.default_rng(0)
    t = pd.date_range("2000-01-01", periods=n_time, freq="D")
    seasonal = 10 + 3 * np.sin(2 * np.pi * t.dayofyear.values / 365.25)[:, None, None]
    data = (seasonal + rng.normal(0, 1, (n_time, 12, 16))).astype(np.float32)
    coords = {"time": t, "lat": np.linspace(-10, 10, 12), "lon": np.linspace(0, 30, 16)}
    return xr.DataArray(data, dims=("time", "lat", "lon"), coords=coords, name="sst").chunk({"time": 30})


@pytest.mark.parametrize("method_extreme", ["seasonal_percentile", "global_percentile"])
def test_pipeline_streaming_pins_the_same_bytes_at_every_n_time(method_extreme, tmp_path):
    """Every persist route through a streaming ``preprocess_data`` (standardised, so both
    threshold runs), recorded at three record lengths: the total must not move with n_time.

    Pinned as invariance in n_time, not an absolute bound. Whether the anchor itself is reached
    is ``test_bounds_check_reads_a_pinned_threshold``.
    """
    totals = []
    for n_time in (730, 1460, 2920):
        with PersistRecorder() as rec:
            marEx.preprocess_data(
                _synthetic(n_time),
                method_anomaly="detrend_harmonic",
                method_extreme=method_extreme,
                window_years=2,
                standardise=True,
                compute_mode="streaming",
                scratch_dir=str(tmp_path / str(n_time)),
            )
        totals.append(rec.bytes)
    assert totals[0] > 0, "nothing pinned: the recorder is not wired"
    assert totals[0] == totals[1] == totals[2], totals
