"""Volumetric tracking across compute modes and time chunkings: exact equality and graph structure."""

import os

import dask.array.core as dask_array_core
import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.track.volumetric import tracker4d as _tracker4d
from tests._recorders import PersistRecorder

from .synthetic import make_inputs, scenarios
from .test_tracker4d import _assert_datasets_equal, _kwargs, _scenario_inputs

EVENT_VARS_INTEGER = ("ID_field", "merge_ledger", "presence", "global_ID")


def _run(name, tmp_path, mode="persist", time_chunk=None, binary=None, **overrides):
    """Track a synthetic scenario, returning ``(events, merges)`` with every array loaded."""
    kwargs = {} if time_chunk is None else {"time_chunk": time_chunk}
    scenario_binary, meta, (extremes, wet, bounds, areas) = _scenario_inputs(name, **kwargs)
    if binary is not None:
        extremes, wet, bounds, areas = make_inputs(binary, **{**meta["make_inputs"], **kwargs})
    if mode == "streaming":
        overrides.setdefault("temp_dir", str(tmp_path / "scratch"))
    t = marEx.tracker(extremes, wet, layer_bounds=bounds, cell_areas=areas, compute_mode=mode, **_kwargs(**overrides))
    events, merges = t.run(return_merges=True)
    return events, merges


def _loaded(ds):
    """A fully computed copy, safe to compare after the staging directory is cleared."""
    return ds.load()


# --------------------------------------------------------------------------------------
# V3: persist and streaming agree exactly
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(scenarios()))
def test_persist_vs_streaming_bit_identical(name, tmp_path):
    """Every variable of events and merges, integers and floats, is exactly equal in the two modes."""
    persist_events, persist_merges = _run(name, tmp_path)
    stream_events, stream_merges = _run(name, tmp_path, mode="streaming")
    try:
        assert stream_events.encoding.get("marex_staging_dir")
        # Names, dims, dtypes (via values), coords and attrs all identical, not only the data variables.
        xr.testing.assert_identical(_loaded(persist_events), _loaded(stream_events))
        xr.testing.assert_identical(_loaded(persist_merges), _loaded(stream_merges))
        _assert_datasets_equal(_loaded(persist_events), _loaded(stream_events), f"{name} events ")
        _assert_datasets_equal(_loaded(persist_merges), _loaded(stream_merges), f"{name} merges ")
        assert persist_events.attrs.keys() == stream_events.attrs.keys()
        for key, value in persist_events.attrs.items():
            np.testing.assert_array_equal(stream_events.attrs[key], value, err_msg=f"{name} attr {key}")
    finally:
        marEx.clear_staging(stream_events)


def test_staging_dir_on_encoding_and_cleared(tmp_path):
    """The staging path is on ``encoding`` (never ``attrs``) and ``clear_staging`` removes it."""
    events, _ = _run("one_merge", tmp_path, mode="streaming")
    staging = events.encoding.get("marex_staging_dir")
    assert staging and "marex_staging_dir" not in events.attrs
    assert str(staging).startswith(str(tmp_path / "scratch"))
    assert os.path.isdir(staging) and os.listdir(staging)
    # Still readable lazily before clearing.
    assert int(events["ID_field"].max()) >= 1
    marEx.clear_staging(events)
    assert not os.path.exists(staging)
    marEx.clear_staging(events)  # idempotent


# --------------------------------------------------------------------------------------
# V4: time-chunk invariance, exact IDs and ledger
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["persist", "streaming"])
@pytest.mark.parametrize("time_chunk", [1, 2, 3])
def test_split_records_do_not_depend_on_the_time_chunk(mode, time_chunk, tmp_path):
    nt = scenarios()["one_split"][0].shape[0]
    _, reference = _run("one_split", tmp_path, time_chunk=nt)
    _, merges = _run("one_split", tmp_path, mode=mode, time_chunk=time_chunk)
    assert reference.sizes["split_ID"] == 1
    _assert_datasets_equal(_loaded(reference), _loaded(merges), f"{mode} chunk {time_chunk} merges ")


@pytest.mark.parametrize("mode", ["persist", "streaming"])
@pytest.mark.parametrize("time_chunk", [1, 2, 3, 5, "nt"])
@pytest.mark.parametrize("name", ["cascade_at_chunk_boundary", "merge_then_split"])
def test_time_chunk_invariance(name, mode, time_chunk, tmp_path):
    """
    Exact ID_field, merge ledger and merge arrays for every chunking, incl. a short final chunk (3 -> 3,3,2).

    The cascade scenario puts a multi-parent merge at a chunk boundary; ``merge_then_split`` adds splits,
    whose consolidation reads the previous chunk's last step when the chunk is one step wide.
    """
    nt = scenarios()[name][0].shape[0]
    time_chunk = nt if time_chunk == "nt" else time_chunk
    reference_events, reference_merges = _run(name, tmp_path, time_chunk=nt)
    events, merges = _run(name, tmp_path, mode=mode, time_chunk=time_chunk)
    try:
        assert merges.sizes["merge_ID"] > 0
        _assert_datasets_equal(_loaded(reference_events), _loaded(events), f"{name} {mode} chunk {time_chunk} events ")
        _assert_datasets_equal(_loaded(reference_merges), _loaded(merges), f"{name} {mode} chunk {time_chunk} merges ")
        for var in EVENT_VARS_INTEGER:
            assert events[var].dtype == reference_events[var].dtype, var
    finally:
        if mode == "streaming":
            marEx.clear_staging(events)


def _merge_loop_chunks(monkeypatch):
    """Record the time-chunk sizes the merge loop receives, one tuple per call."""
    seen = []
    real = _tracker4d.split_and_merge_volumetric

    def spy(id_field, *args, **kwargs):
        seen.append(tuple(id_field.chunks[0]))
        return real(id_field, *args, **kwargs)

    monkeypatch.setattr(_tracker4d, "split_and_merge_volumetric", spy)
    return seen


def test_persist_eleven_plus_chunks(tmp_path, monkeypatch):
    """24 steps in 12 chunks of 2 (past the flush-every-10 boundary) equal one chunk of 24, exactly."""
    base, _ = scenarios()["cascade_at_chunk_boundary"]
    binary = np.concatenate([base, base[::-1], base], axis=0)
    assert binary.shape[0] == 24
    seen = _merge_loop_chunks(monkeypatch)
    one_events, one_merges = _run("cascade_at_chunk_boundary", tmp_path, time_chunk=24, binary=binary)
    many_events, many_merges = _run("cascade_at_chunk_boundary", tmp_path, time_chunk=2, binary=binary)
    # The merge loop really ran in 24 steps / 12 chunks, not just an input that was labelled so.
    assert seen == [(24,), (2,) * 12], seen
    assert one_merges.sizes["merge_ID"] >= 3
    _assert_datasets_equal(_loaded(one_events), _loaded(many_events), "events ")
    _assert_datasets_equal(_loaded(one_merges), _loaded(many_merges), "merges ")


# --------------------------------------------------------------------------------------
# V4: graph structure, pinned in nt rather than bounded absolutely
# --------------------------------------------------------------------------------------


def _tiled(name, nt):
    """The scenario's field repeated along time to ``nt`` steps (same grid)."""
    base, _ = scenarios()[name]
    return np.tile(base, (nt // base.shape[0], 1, 1, 1))


def _track_with(recorder_cm, name, nt, mode, tmp_path):
    with recorder_cm as recorder:
        events, _ = _run(name, tmp_path, mode=mode, time_chunk=4, binary=_tiled(name, nt))
        events.load()
    if mode == "streaming":
        marEx.clear_staging(events)
    return recorder


def test_streaming_pinned_bytes_flat_in_nt(tmp_path):
    """
    Streaming persists no voxel-sized intermediate, at any length; persist mode does, and its bytes grow with nt.

    Voxel intermediates are objects of at least one int32 timestep. The persist run is the control: it shows
    the recorder is wired to the merge loop, so a flat streaming figure is not a recorder that saw nothing.
    """
    name = "birth_death"
    _, _, (extremes, *_) = _scenario_inputs(name)
    voxel_step_bytes = int(np.prod(extremes.shape[1:])) * 4

    def voxel_bytes(recorder):
        return sum(size for sizes in recorder.calls for size in sizes if size >= voxel_step_bytes)

    streaming, persist = {}, {}
    for nt in (8, 32):
        streaming[nt] = voxel_bytes(_track_with(PersistRecorder(allow_empty=True), name, nt, "streaming", tmp_path))
        persist[nt] = voxel_bytes(_track_with(PersistRecorder(), name, nt, "persist", tmp_path))
    assert persist[32] > persist[8] > 0, persist
    assert streaming[8] == streaming[32], f"streaming pinned bytes grew with nt: {streaming}"
    assert streaming[32] < persist[8], (streaming, persist)


class _EagerWrapSpy:
    """
    Record every numpy array dask eagerly wraps (and therefore copies) through ``from_array``.

    Both the ``dask.array.core`` binding (used by ``apply_gufunc``) and the ``dask.array`` one
    (used by xarray's ``chunk``) are patched, since either site can densify an array at graph build.
    """

    def __init__(self):
        self.wrapped = []

    def __enter__(self):
        import dask.array as da

        self._orig = dask_array_core.from_array
        self._orig_public = da.from_array
        self._da = da

        def spy(x, *args, **kwargs):
            if isinstance(x, np.ndarray):
                self.wrapped.append((tuple(x.shape), str(x.dtype), int(x.nbytes)))
            return self._orig(x, *args, **kwargs)

        dask_array_core.from_array = spy
        da.from_array = spy
        return self

    def __exit__(self, *exc):
        dask_array_core.from_array = self._orig
        self._da.from_array = self._orig_public
        return False

    @property
    def total_bytes(self):
        return sum(nbytes for _, _, nbytes in self.wrapped)


def test_geometry_not_broadcast_over_time(tmp_path):
    """
    Geometry reaches dask as ``(z, y, x)`` / ``(y, x)`` arrays: the wrapped geometry bytes are equal at nt=8 and nt=32.

    Geometry is every wrapped array that is float64 (the ID chunks are int32, the offsets int32, the
    masks bool). None may carry a time axis, whatever its length. The graph of the cell volumes is a
    supplementary check: one key and one chunk at both lengths.
    """
    name = "birth_death"
    seen = {}
    for nt in (8, 32):
        _, meta, _ = _scenario_inputs(name)
        extremes, wet, bounds, areas = make_inputs(_tiled(name, nt), **meta["make_inputs"])
        with _EagerWrapSpy() as spy:
            t = marEx.tracker(extremes, wet, layer_bounds=bounds, cell_areas=areas, **_kwargs())
            events, _ = t.run(return_merges=True)
            events.load()
        cell_volume = t._cell_volume_da()
        spatial = tuple(extremes.shape[1:])
        geometry = [w for w in spy.wrapped if w[1] == "float64"]
        seen[nt] = (sum(w[2] for w in geometry), geometry, len(cell_volume.__dask_graph__()), cell_volume.chunks, spatial)
    assert seen[8][0] == seen[32][0], f"eagerly wrapped geometry grew with nt: {seen[8][0]} B vs {seen[32][0]} B"
    for nt, (_total, geometry, n_keys, chunks, spatial) in seen.items():
        assert geometry, "the spy saw no geometry wrap: it is not wired to the code under test"
        offenders = [w for w in geometry if w[0] not in (spatial, spatial[1:])]
        assert not offenders, f"geometry arrays with a time axis (or any other shape) wrapped at nt={nt}: {offenders}"
        assert chunks == tuple((n,) for n in spatial), chunks
        assert n_keys == 1, n_keys
    # The geometry arrays themselves are (z, y, x) only.
    for field in ("top", "bottom", "thickness", "midpoint", "cell_volume", "wet"):
        assert getattr(t.geom, field).shape == spatial, field
