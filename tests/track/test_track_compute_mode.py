"""``compute_mode`` on the tracker: selection, persist/streaming bit-identity, staging lifetime, pinned bytes,
persist graph shape and driver memory.

Laziness is asserted by BYTES handed to ``persist`` (``PersistRecorder``), never by
``is_dask_collection``: an array is still a dask collection after ``.persist()``. Every
streaming bound is paired with a persist control on the same recorder, so a recorder that sees
nothing cannot make a bound pass.

The graph-shape bounds are a tripwire for what values are blind to: an all-to-all rechunk once
shipped while every value test was green. If a change moves them, understand it and update them
deliberately, never just to pass.
"""

from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import dask
import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.core.compute_mode import Materialiser
from marEx.exceptions import ConfigurationError
from marEx.track import merge_split as _merge_split

from .._recorders import PersistRecorder
from ..conftest import released
from .conftest import (
    CHUNK_SIZE,
    MODE_KWARGS,
    THREADED,
    UNSTRUCTURED_KWARGS,
    UNSTRUCTURED_MERGING_KWARGS,
    assert_no_failures,
    check,
    on_client,
    open_gridded,
    open_unstructured,
    open_unstructured_merging,
    tile_to,
)

# Measured at dd43301 (464 rechunk keys) with a +2 margin. The transpose bound was raised from 50
# when global_ID and presence moved to arrays anchored on the cluster (one new transpose layer) and
# centroid's anchoring moved (two), each one anchored leaf per time chunk: measured 96, +2.
EXPECTED_MAX_RECHUNK = 466
EXPECTED_MAX_TRANSPOSE = 98


def _identical(stream, persist, label):
    """No tolerance: assert_identical on the events (attrs dropped) and on the merges."""
    stream_events, stream_merges = stream
    persist_events, persist_merges = persist
    xr.testing.assert_identical(stream_events.drop_attrs(deep=False), persist_events.drop_attrs(deep=False))
    xr.testing.assert_identical(stream_merges, persist_merges)


def _run_pair(data_bin, mask, tmp_path, label, **kwargs):
    """Persist and streaming on the SAME chunking, both computed: ((events, merges), (events, merges)).

    Matching chunking matters: the gridded merge loop's end-of-chunk consolidation is boundary
    dependent, so streaming at one chunking against persist at another could legitimately differ.
    """
    kwargs = {**MODE_KWARGS, **kwargs}
    events, merges = marEx.tracker(data_bin, mask, **kwargs).run(return_merges=True)
    persist = (events.compute(), merges.compute())
    del events, merges
    events, merges = marEx.tracker(data_bin, mask, compute_mode="streaming", temp_dir=str(tmp_path / label), **kwargs).run(
        return_merges=True
    )
    try:
        stream = (events.compute(), merges.compute())
    finally:
        marEx.clear_staging(events)
    return persist, stream


def test_compute_mode_selection(client, tmp_path):
    """persist is the default; streaming needs and builds a staging dir; unstructured is accepted under streaming."""
    ds = open_gridded()
    data_bin = ds.extreme_events.chunk(CHUNK_SIZE)
    failures = []

    def materialiser():
        assert Materialiser("streaming", tmp_path / "m").is_streaming is True
        assert Materialiser("persist").is_streaming is False
        assert Materialiser("lazy").is_streaming is False

    def default():
        tr = marEx.tracker(data_bin, ds.mask, **MODE_KWARGS)
        assert tr.compute_mode == "persist"
        assert tr.materialiser.is_streaming is False

    def streaming():
        tr = marEx.tracker(data_bin, ds.mask, compute_mode="streaming", temp_dir=str(tmp_path / "s"), **MODE_KWARGS)
        assert tr.materialiser.is_streaming is True
        assert tr.staging_dir is not None and Path(tr.staging_dir).exists()

    def unstructured():
        # Unstructured input used to be rejected under streaming; the merge/split core is threaded now.
        u = open_unstructured()
        tr = marEx.tracker(
            u.extreme_events.chunk({"time": 2, "ncells": -1}),
            u.mask,
            R_fill=2,
            area_filter_quartile=0.5,
            compute_mode="streaming",
            temp_dir=str(tmp_path / "u"),
            neighbours=u.neighbours,
            cell_areas=u.cell_areas,
            **{k: v for k, v in UNSTRUCTURED_KWARGS.items() if k != "regional_mode"},
        )
        assert tr.materialiser.is_streaming is True
        assert tr.unstructured_grid is True

    # Each row on the scheduler it ran on before the consolidation.
    for name, row in (("materialiser", materialiser), ("default", default), ("streaming", streaming)):
        with dask.config.set(THREADED):
            check(failures, name, row)
    check(failures, "unstructured", unstructured)
    assert_no_failures(failures)


def test_streaming_retiles_time_chunking(tmp_path):
    """(5, ..., 5, 2) is accepted as is; a genuinely ragged chunking is re-tiled once to a uniform width."""
    ds = open_gridded()
    n_time = ds.sizes["time"]
    uniform = ds.extreme_events.chunk({"time": 5, "lat": -1, "lon": -1})
    assert n_time % 5 != 0, "fixture must not divide evenly, so a smaller final chunk exists"
    tr = marEx.tracker(uniform, ds.mask, compute_mode="streaming", temp_dir=str(tmp_path / "u"), **MODE_KWARGS)
    assert tr.materialiser.is_streaming is True

    ragged = tuple([3, 5] * ((n_time // 8) + 1))[: n_time // 8 * 2]
    if n_time - sum(ragged) > 0:
        ragged = ragged + (n_time - sum(ragged),)
    assert sum(ragged) == n_time and len(ragged) > 1
    data_bin = ds.extreme_events.chunk({"lat": -1, "lon": -1}).chunk({"time": ragged})
    tr = marEx.tracker(data_bin, ds.mask, compute_mode="streaming", temp_dir=str(tmp_path / "r"), **MODE_KWARGS)
    time_chunks = tr.data_bin.chunks[tr.data_bin.dims.index("time")]
    assert set(time_chunks[:-1]) == {5} and time_chunks[-1] <= 5
    assert tr.timechunks == 5


@pytest.mark.slow
def test_streaming_staging_lifetime(gridded_mode_pair):
    """The staging dir outlives run(), holds the merge ledger, never reaches a written store, and clear_staging removes it.

    persist with a temp_dir stages nothing. The ledger carries real merges, else every cross-mode
    assert_identical over it compares -1 with -1.
    """
    facts = gridded_mode_pair["facts"]
    assert facts["staging_dir"] is not None, "streaming output must advertise its staging dir"
    assert facts["staging_exists_after_run"], "staging dir must OUTLIVE run(); the result reads from it"
    assert "merge_ledger.zarr" in facts["staging_holds"], f"merge_ledger not staged; staging dir holds {facts['staging_holds']}"
    assert not facts["written_attrs_carry_staging"], "the written store records a staging dir that clear_staging deletes"
    assert not facts["staging_exists_after_clear"], "clear_staging must actually remove the directory"
    assert not facts["persist_staging_attr"] and not facts["persist_staging_encoding"]
    assert not facts["persist_staged_ledgers"], f"persist mode staged {facts['persist_staged_ledgers']}"
    for mode in ("persist", "streaming"):
        ledger = gridded_mode_pair[mode][0].merge_ledger.values
        assert int((ledger > 0).sum()) > 0, f"{mode}: fixture produced no merges, the ledger is all fill"


@pytest.mark.slow
def test_streaming_matches_persist_exactly(gridded_mode_pair):
    """streaming is BIT-IDENTICAL to persist; integer label fields, no tolerance."""
    _identical(gridded_mode_pair["streaming"], gridded_mode_pair["persist"], "mode pair")
    assert gridded_mode_pair["persist"][1].sizes["merge_ID"] > 0, "no merges: the equivalence covers no merge"


@pytest.mark.slow
def test_streaming_pins_far_less_than_persist(gridded_mode_pair):
    """persist pins >= 2 whole int32 fields (the control); streaming stays under 2."""
    facts = gridded_mode_pair["facts"]
    field = facts["field_bytes"]
    assert facts["persist_pinned"] >= 2 * field, (
        f"persist pinned only {facts['persist_pinned']} bytes, less than 2x one int32 field ({field}): "
        "the recorder is not seeing persist's own pins"
    )
    assert facts["stream_pinned"] < 2 * field, (
        f"streaming pinned {facts['stream_pinned']} bytes, more than 2x one int32 field ({field}): "
        "the materialiser is not reaching every site"
    )


@pytest.mark.slow
def test_persist_graph_shape_is_stable(gridded_mode_pair):
    """A rise in n_rechunk is the signature of shredded chunk boundaries."""
    stats = gridded_mode_pair["facts"]["persist_graph"]
    assert stats["n_rechunk"] <= EXPECTED_MAX_RECHUNK, (
        f"rechunk key count rose to {stats['n_rechunk']} (limit {EXPECTED_MAX_RECHUNK}); investigate before updating this bound"
    )
    assert stats["n_transpose"] <= EXPECTED_MAX_TRANSPOSE, (
        f"transpose key count rose to {stats['n_transpose']} (limit {EXPECTED_MAX_TRANSPOSE})"
    )


def _size_one_tail(ds):
    data_bin = ds.extreme_events.isel(time=slice(0, 31)).chunk(CHUNK_SIZE)
    assert data_bin.chunks[0][-1] == 1, "the final chunk must be a single step"
    return data_bin, {}


def _chunk1(ds):
    # 32 chunks: the merge loop flushes at chunk_idx 10 and 20 (>= 21 chunks needed for two flushes).
    data_bin = ds.extreme_events.chunk({"time": 1, "lat": -1, "lon": -1})
    assert len(data_bin.chunks[0]) >= 21
    return data_bin, {}


def _whole(ds):
    data_bin = ds.extreme_events.chunk({"time": -1, "lat": -1, "lon": -1})
    assert len(data_bin.chunks[0]) == 1
    return data_bin, {}


def _trigger(ds):
    # 21 steps at chunk 5 with T_fill=4 (depth 2): dask_image's overlap rebalancing used to hand back
    # (5, 5, 5, 4, 2), which zarr rejects, and the streaming run died mid-merge.
    data_bin = tile_to(ds.extreme_events, 21).chunk({"time": 5, "lat": -1, "lon": -1})
    return data_bin, {"T_fill": 4}


@pytest.mark.slow
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(_size_one_tail, id="size_one_tail"),
        pytest.param(_chunk1, id="chunk1"),
        pytest.param(_whole, id="whole"),
        pytest.param(_trigger, id="trigger_21x5_T4"),
    ],
)
def test_streaming_matches_persist_across_time_chunkings(layout, client, tmp_path):
    """Bit-identity at a size-one final chunk, two periodic flushes, a single finalise write, and the realignment trigger."""
    assert on_client(client)
    ds = open_gridded()
    data_bin, overrides = layout(ds)
    persist, stream = _run_pair(data_bin, ds.mask, tmp_path, "stream", **overrides)
    assert stream[0].ID_field.sizes["time"] == data_bin.sizes["time"]
    _identical(stream, persist, layout.__name__)
    assert persist[1].sizes["merge_ID"] > 0, "no merges: the equivalence covers no merge"


@pytest.mark.slow
def test_absolute_filter_labels_once_and_matches_persist(client, tmp_path, monkeypatch):
    """The absolute filter labels every slice ONCE in every mode, and streaming matches persist on that path.

    ``filter_small_objects`` derives the keep-mask and the area census from one ``apply_ufunc`` and
    materialises both together; under streaming that was a no-op pin, so the census ran the pass and
    the stage of the filtered field ran it again. Calls are counted THROUGH A FILE: the wrapper is
    pickled into the ufunc closure, so an in-memory list would be counted on a worker's copy.
    """
    assert on_client(client)
    from marEx.track import morphology

    counter = tmp_path / "labels.count"
    orig = morphology.scipy_label

    def counting(*args, **kwargs):
        with open(counter, "ab") as handle:
            handle.write(b"x")
        return orig(*args, **kwargs)

    monkeypatch.setattr(morphology, "scipy_label", counting)
    ds = open_gridded()
    data_bin = ds.extreme_events.chunk(CHUNK_SIZE)
    kwargs = dict({k: v for k, v in MODE_KWARGS.items() if k != "area_filter_quartile"}, area_filter_absolute=400)

    events, merges = marEx.tracker(data_bin, ds.mask, **kwargs).run(return_merges=True)
    persist = (events.compute(), merges.compute())
    persist_calls = counter.stat().st_size if counter.exists() else 0
    counter.unlink(missing_ok=True)
    events, merges = marEx.tracker(data_bin, ds.mask, compute_mode="streaming", temp_dir=str(tmp_path / "s"), **kwargs).run(
        return_merges=True
    )
    try:
        stream = (events.compute(), merges.compute())
    finally:
        marEx.clear_staging(events)
    streaming_calls = counter.stat().st_size if counter.exists() else 0

    assert persist_calls >= data_bin.sizes["time"], (
        f"the counter saw only {persist_calls} labelling calls in persist mode, fewer than the "
        f"{data_bin.sizes['time']} slices: the patch is not reaching the ufunc body"
    )
    assert streaming_calls <= persist_calls, (
        f"streaming labelled {streaming_calls} slices against persist's {persist_calls}: the joint "
        "materialisation of (keep-mask, area census) is not shared under streaming"
    )
    _identical(stream, persist, "absolute filter")
    assert persist[1].sizes["merge_ID"] > 0, "no merges: the equivalence covers no merge"


@pytest.mark.slow
def test_no_merging_path_matches_and_pins_nothing(client, tmp_path):
    """``allow_merging=False`` (objects.py's ``elif time_connectivity:`` branch): identical, and streaming pins no field.

    N_events_final is an attr, which the identity check drops, and it is the one output this branch
    can move: streaming reads it as max(labels) off the staged store.
    """
    assert on_client(client)
    ds = open_gridded()
    data_bin = ds.extreme_events.chunk(CHUNK_SIZE)
    field = data_bin.size * 4
    kwargs = dict(MODE_KWARGS, allow_merging=False)

    with PersistRecorder() as persist_rec:
        lazy = marEx.tracker(data_bin, ds.mask, **kwargs).run()
    persist = lazy.compute()
    del lazy
    with PersistRecorder(allow_empty=True) as stream_rec:
        lazy = marEx.tracker(data_bin, ds.mask, compute_mode="streaming", temp_dir=str(tmp_path / "s"), **kwargs).run()
    try:
        stream = lazy.compute()
    finally:
        marEx.clear_staging(lazy)
    del lazy  # a failure's traceback would otherwise keep its futures past the module

    xr.testing.assert_identical(stream.drop_attrs(deep=False), persist.drop_attrs(deep=False))
    assert stream.attrs["N_events_final"] == persist.attrs["N_events_final"], (
        f"event count differs by mode: streaming {stream.attrs['N_events_final']} vs persist {persist.attrs['N_events_final']}"
    )
    assert persist_rec.bytes >= field, (
        f"persist pinned only {persist_rec.bytes} bytes, less than one int32 field ({field}): the recorder sees nothing"
    )
    assert stream_rec.bytes < 0.5 * field, (
        f"streaming with allow_merging=False pinned {stream_rec.bytes} bytes, more than half an int32 field ({field})"
    )


def _array(value):
    import dask.array as dask_array

    return xr.DataArray(dask_array.full((4, 3), value, chunks=(2, 3)), dims=("time", "x"))


def test_materialiser_label_reuse(tmp_path):
    """A staging label may not be reused for a different array.

    ``_stage_to_zarr`` writes ``<staging_dir>/<label>.zarr`` with ``mode="w"``; a second array under
    the same label would rewrite the store the first is still lazily reading.
    """
    failures = []

    def noop():
        mat = Materialiser("streaming", tmp_path / "noop")
        first = mat.stage(_array(1), "field")
        assert mat.stage(first, "field") is first, "re-anchoring the identical staged array must not rewrite the store"

    def rejected():
        mat = Materialiser("streaming", tmp_path / "rejected")
        staged = mat.stage(_array(1), "field")
        with pytest.raises(ConfigurationError, match="different array"):
            mat.stage(_array(2), "field")
        assert int(staged.sum().compute()) == 12, "the first array no longer reads its own values"

    def persist():
        mat = Materialiser("persist")
        mat.stage(_array(1), "field")
        mat.stage(_array(2), "field")  # persist writes no store, so it has no label to collide on

    for name, row in (("noop", noop), ("rejected", rejected), ("persist", persist)):
        check(failures, name, row)
    assert_no_failures(failures)


@pytest.mark.slow
@pytest.mark.smoke
def test_unstructured_streaming_matches_persist(client, tmp_path):
    """streaming is BIT-IDENTICAL to persist on the unstructured merge loop, over many time chunks.

    Many chunks drive the per-iteration staging and the post-merge anchor more than once, and give
    the merge writer both its write branch and its no-merge branch. The persist side carries the
    output invariants of a run that really merges, and the pins captured before the
    writer stopped returning the whole field.
    """
    assert on_client(client)
    data = open_unstructured_merging()
    data_bin = data.extreme_events.chunk({"time": 2, "ncells": -1})
    assert len(data_bin.chunks[0]) >= 20, "many time chunks: drives the merge loop repeatedly"
    common = dict(UNSTRUCTURED_MERGING_KWARGS, neighbours=data.neighbours, cell_areas=data.cell_areas)

    def run():
        # Inside ``released``: a failed run's traceback, and the futures it holds, never leave this frame.
        events, merges = marEx.tracker(data_bin, data.mask, temp_dir=str(tmp_path / "persist"), **common).run(return_merges=True)
        persist = (events.compute(), merges.compute())
        del events, merges
        streaming = marEx.tracker(data_bin, data.mask, compute_mode="streaming", temp_dir=str(tmp_path / "streaming"), **common)
        staging_dir = streaming.staging_dir
        try:
            events, merges = streaming.run(return_merges=True)
            del streaming
            return persist, (events.compute(), merges.compute())
        finally:
            marEx.clear_staging(staging_dir)  # also when the streaming run itself raises

    (persist_events, persist_merges), (stream_events, stream_merges) = released(run, "the persist/streaming pair")

    xr.testing.assert_identical(stream_events.drop_attrs(deep=False), persist_events.drop_attrs(deep=False))
    xr.testing.assert_identical(stream_merges, persist_merges)
    assert stream_events.attrs["N_events_final"] == persist_events.attrs["N_events_final"]

    ev = persist_events
    assert ev.attrs["N_events_final"] == 11
    assert int(ev.ID_field.sum()) == 15288
    assert ev.attrs["total_merges"] > 0 and persist_merges.sizes["merge_ID"] > 0, "the merging fixture produced no merges"
    assert ev.sizes["ID"] == ev.attrs["N_events_final"]
    assert np.issubdtype(ev.ID_field.dtype, np.integer)
    present = ev.presence.values
    assert np.array_equal(present, ev.global_ID.values != 0), "presence does not match global_ID != 0"
    assert (ev.area.values[present] > 0).all(), "an event has non-positive area where present"
    ever = present.any(axis=ev.presence.dims.index("time"))
    assert (ev.time_start.values[ever] <= ev.time_end.values[ever]).all(), "an event starts after it ends"
    centroid = ev.centroid.transpose(*ev.presence.dims, "component").values
    assert not np.isnan(centroid[present]).any(), "a centroid is NaN where its event is present"
    lat, lon = centroid[present][:, 0], centroid[present][:, 1]
    assert (lat >= float(data.lat.min())).all() and (lat <= float(data.lat.max())).all(), "centroid outside lat bounds"
    assert (lon >= float(data.lon.min())).all() and (lon <= float(data.lon.max())).all(), "centroid outside lon bounds"


# ---------------------------------------------------------------------------------------------
# Driver memory. The gridded merge loop computes one time chunk at a time to numpy on the driver;
# only that chunk and its predecessor are needed. Event metadata (time, ID) must stay dask-backed:
# never gathered to the driver as numpy, and under streaming never pinned.
# ---------------------------------------------------------------------------------------------

TIME_CHUNK = 2


def _root(arr):
    """The ndarray that owns ``arr``'s memory (views and slices share it)."""
    while isinstance(arr.base, np.ndarray):
        arr = arr.base
    return arr


class _ChunkCensus:
    """Weakref every chunk the serial merge loop loads; count the live ones at each load."""

    def __init__(self, n_time):
        self.refs, self.live = [], []
        self.active = False
        self.n_time = n_time

    def record(self, out):
        if not (self.active and isinstance(out, xr.DataArray) and isinstance(out.data, np.ndarray)):
            return
        if "time" not in out.dims or out.ndim != 3 or out.sizes["time"] >= self.n_time:
            return
        self.refs.append(weakref.ref(_root(out.data)))
        gc.collect()
        self.live.append(sum(r() is not None for r in self.refs))  # the chunk just loaded still counts


def _has_time_and_id(dims):
    return "time" in dims and "ID" in dims


class _MetadataCensus:
    """Record every (time, ID) array that reaches the driver as numpy, or is pinned, from the clustering step on."""

    def __init__(self):
        self.active = False
        self.entered = False
        self.driver_shapes, self.pinned_shapes, self.driver_named, self.pinned = [], [], [], []

    def array_pinned(self, shape):
        if self.active:
            self.pinned_shapes.append(shape)

    def numpy_result(self, out):
        if self.active and isinstance(out, np.ndarray) and out.ndim >= 2:
            self.driver_shapes.append(out.shape)

    def xr_result(self, out, where):
        if not self.active:
            return
        items = out.data_vars.values() if isinstance(out, xr.Dataset) else [out]
        for v in items:
            if isinstance(v, xr.DataArray) and _has_time_and_id(v.dims) and isinstance(v.data, np.ndarray):
                self.driver_named.append((where, v.name, v.dims, v.shape))

    def persisted(self, objs, where):
        if not self.active:
            return
        for o in objs:
            items = o.data_vars.values() if isinstance(o, xr.Dataset) else [o]
            for v in items:
                if isinstance(v, xr.DataArray) and _has_time_and_id(v.dims):
                    self.pinned.append((where, v.name, v.dims, v.shape))


def _census_run(data_bin, mask, monkeypatch, chunk_census=None, **kwargs):
    """One streaming run under the metadata census and, on the gridded leg, the merge-loop chunk census."""
    import dask.array as da
    from distributed import Client

    census = _MetadataCensus()
    orig = {
        "arr_compute": da.Array.compute,
        "arr_array": da.Array.__array__,
        "xr_compute": xr.DataArray.compute,
        "ds_compute": xr.Dataset.compute,
        "full_like": xr.full_like,
        "dask_persist": dask.persist,
        "da_persist": xr.DataArray.persist,
        "ds_persist": xr.Dataset.persist,
        # Every result the client brings back passes through Client._gather; every persist of a
        # dask Array rebuilds through __dask_postpersist__, including a raw Array.persist.
        "client_gather": Client._gather,
        "postpersist": da.Array.__dask_postpersist__,
    }

    async def client_gather(self, *a, **k):
        out = await orig["client_gather"](self, *a, **k)
        stack = [out]
        while stack:
            r = stack.pop()
            if isinstance(r, (list, tuple)):
                stack.extend(r)
            elif isinstance(r, dict):
                stack.extend(r.values())
            else:
                census.numpy_result(r)
        return out

    def postpersist(self):
        census.array_pinned(self.shape)
        return orig["postpersist"](self)

    def arr_compute(self, *a, **k):
        out = orig["arr_compute"](self, *a, **k)
        census.numpy_result(out)
        return out

    def arr_array(self, *a, **k):
        out = orig["arr_array"](self, *a, **k)
        census.numpy_result(out)
        return out

    def xr_compute(self, *a, **k):
        out = orig["xr_compute"](self, *a, **k)
        census.xr_result(out, "DataArray.compute")
        if chunk_census is not None:
            chunk_census.record(out)
        return out

    def ds_compute(self, *a, **k):
        out = orig["ds_compute"](self, *a, **k)
        census.xr_result(out, "Dataset.compute")
        return out

    def full_like(other, *a, **k):
        out = orig["full_like"](other, *a, **k)
        census.xr_result(out, "full_like")
        return out

    def dask_persist(*objs, **k):
        census.persisted(objs, "dask.persist")
        return orig["dask_persist"](*objs, **k)

    def da_persist(self, **k):
        census.persisted((self,), "DataArray.persist")
        return orig["da_persist"](self, **k)

    def ds_persist(self, **k):
        census.persisted((self,), "Dataset.persist")
        return orig["ds_persist"](self, **k)

    orig_cluster = _merge_split.cluster_rename_objects_and_props
    orig_merge = _merge_split.split_and_merge_objects

    def cluster(*a, **k):
        census.active = census.entered = True
        return orig_cluster(*a, **k)

    def merge(*a, **k):
        if chunk_census is not None:
            chunk_census.active = True
        try:
            return orig_merge(*a, **k)
        finally:
            if chunk_census is not None:
                chunk_census.active = False

    with monkeypatch.context() as m:
        m.setattr(da.Array, "compute", arr_compute)
        m.setattr(da.Array, "__array__", arr_array)
        m.setattr(xr.DataArray, "compute", xr_compute)
        m.setattr(xr.Dataset, "compute", ds_compute)
        m.setattr(xr, "full_like", full_like)
        m.setattr(dask, "persist", dask_persist)
        m.setattr(xr.DataArray, "persist", da_persist)
        m.setattr(xr.Dataset, "persist", ds_persist)
        m.setattr(Client, "_gather", client_gather)
        m.setattr(da.Array, "__dask_postpersist__", postpersist)
        m.setattr(_merge_split, "cluster_rename_objects_and_props", cluster)
        m.setattr(_merge_split, "split_and_merge_objects", merge)
        # `from dask import persist` binds the original at import time: rebind those names too.
        for name, mod in list(sys.modules.items()):
            if name.startswith("marEx") and getattr(mod, "persist", None) is orig["dask_persist"]:
                m.setattr(mod, "persist", dask_persist)
            if name.startswith("marEx") and getattr(mod, "full_like", None) is orig["full_like"]:
                m.setattr(mod, "full_like", full_like)
        events = marEx.tracker(data_bin, mask, **kwargs).run()
        census.active = False
        if chunk_census is not None:
            events.ID_field.compute()

    n_id, nt = events.sizes["ID"], events.sizes["time"]

    def time_by_id(s):  # inside the clustering step the ID axis carries one padding entry
        return nt in s and (n_id in s or n_id + 1 in s)

    gathered = [s for s in census.driver_shapes if time_by_id(s)]
    pinned = [s for s in census.pinned_shapes if time_by_id(s)]
    backing = {v: type(events[v].data).__name__ for v in events.data_vars if _has_time_and_id(events[v].dims)}
    marEx.clear_staging(events)
    return census, gathered, pinned, backing, n_id


@pytest.mark.parametrize("grid", ["gridded", "unstructured"])
def test_streaming_driver_memory_is_flat_in_nt(grid, client, tmp_path, monkeypatch):
    """At two record lengths: no (time, ID) metadata on the driver or pinned; the gridded loop holds <= 2 chunks."""
    assert on_client(client)
    worst = {}
    for nt in (16, 32) if grid == "gridded" else (50, 100):
        chunk_census = None
        if grid == "gridded":
            ds = open_gridded()
            data_bin = ds.extreme_events.isel(time=slice(0, nt)).chunk({"time": TIME_CHUNK, "lat": -1, "lon": -1})
            kwargs = dict(MODE_KWARGS, compute_mode="streaming", temp_dir=str(tmp_path / f"g{nt}"))
            chunk_census = _ChunkCensus(nt)
        else:
            ds = open_unstructured_merging()
            data_bin = ds.extreme_events.isel(time=slice(0, nt)).chunk({"time": 4, "ncells": -1})
            kwargs = dict(
                UNSTRUCTURED_MERGING_KWARGS,
                compute_mode="streaming",
                temp_dir=str(tmp_path / f"u{nt}"),
                neighbours=ds.neighbours,
                cell_areas=ds.cell_areas,
            )
        census, gathered, pinned, backing, n_id = _census_run(data_bin, ds.mask, monkeypatch, chunk_census, **kwargs)
        assert census.entered, "the clustering step never ran: the census is vacuous"
        assert n_id > 1, "no events: the census is vacuous"
        assert backing and all(b != "ndarray" for b in backing.values()), f"nt={nt}: (time, ID) outputs held as numpy: {backing}"
        assert not gathered, f"nt={nt}: (time, ID) arrays gathered to the driver: {gathered}"
        assert not census.driver_named, f"nt={nt}: numpy-backed (time, ID) built on the driver: {census.driver_named}"
        assert not census.pinned, f"nt={nt}: (time, ID) pinned under streaming: {census.pinned}"
        assert not pinned, f"nt={nt}: (time, ID) dask arrays persisted under streaming: {pinned}"
        if chunk_census is not None:
            assert len(chunk_census.live) == nt // TIME_CHUNK, f"nt={nt}: the loop did not load every chunk: {chunk_census.live}"
            worst[nt] = max(chunk_census.live)
    if grid == "gridded":
        assert worst[16] == worst[32] <= 2, f"live chunk buffers per load grew with nt: {worst}"


def _with_extra_property(object_props):
    ids = object_props["ID"]
    return object_props.assign(temperature=(ids * 0.25 + 1.0).astype(np.float32))


def test_extra_property_is_relabelled_by_lookup(client, tmp_path, monkeypatch):
    """A property other than area and centroid is relabelled by lookup: the value of the object the event held, NaN elsewhere."""
    assert on_client(client)
    orig_cluster = _merge_split.cluster_rename_objects_and_props

    def cluster(object_id_field_unique, object_props, *a, **k):
        return orig_cluster(object_id_field_unique, _with_extra_property(object_props), *a, **k)

    monkeypatch.setattr(_merge_split, "cluster_rename_objects_and_props", cluster)
    ds = open_gridded()
    data_bin = ds.extreme_events.isel(time=slice(0, 16)).chunk({"time": TIME_CHUNK, "lat": -1, "lon": -1})
    failures = []
    for mode in ("persist", "streaming"):

        def row(mode=mode):
            events = marEx.tracker(data_bin, ds.mask, compute_mode=mode, temp_dir=str(tmp_path / mode), **MODE_KWARGS).run()
            try:
                assert list(events.data_vars) == [
                    "ID_field",
                    "global_ID",
                    "area",
                    "centroid",
                    "temperature",
                    "presence",
                    "time_start",
                    "time_end",
                    "merge_ledger",
                ], f"variable order {list(events.data_vars)}"
                got = events["temperature"]
                assert got.dims == ("time", "ID") and got.dtype == np.float32
                global_id = events["global_ID"].values
                assert (global_id > 0).any(), "no event is present anywhere: the lookup is vacuous"
                want = np.where(global_id > 0, (global_id * 0.25 + 1.0).astype(np.float32), np.float32(np.nan))
                np.testing.assert_array_equal(got.values, want, err_msg=f"{mode}: temperature is not the looked-up value")
            finally:
                if mode == "streaming":
                    marEx.clear_staging(events)

        check(failures, mode, row)
    assert_no_failures(failures)
