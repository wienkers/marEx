"""The tracker's driver must not hold O(nt) of the ID field.

The gridded merge loop computes one time chunk at a time to numpy on the driver. Only the
chunk being processed and the one before it (the t-1/t-2 continuity source) are needed;
anything older is finished and belongs in the accumulator. Under streaming, these tests count
the chunk buffers still alive each time the loop loads a new chunk, at two record lengths, so
a retention that grows with nt fails even when every output value is right.
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.track import merge_split as _merge_split

TEST_DATA_DIR = Path(__file__).parent / "data"
TRACKER_KWARGS = {
    "R_fill": 8,
    "area_filter_quartile": 0.5,
    "T_fill": 2,
    "allow_merging": True,
    "overlap_threshold": 0.5,
    "nn_partitioning": True,
    "quiet": True,
}
TIME_CHUNK = 2


@pytest.fixture(scope="module")
def extremes():
    return xr.open_zarr(str(TEST_DATA_DIR / "extremes_gridded.zarr"), chunks={})


def _root(arr: np.ndarray) -> np.ndarray:
    """The ndarray that owns ``arr``'s memory (views and slices share it)."""
    while isinstance(arr.base, np.ndarray):
        arr = arr.base
    return arr


class _ChunkCensus:
    """Weakref every chunk the serial merge loop loads; count the live ones at each load."""

    def __init__(self) -> None:
        self.refs: list = []
        self.live: list = []
        self.active = False
        self.n_time = None

    def record(self, out) -> None:
        if not (self.active and isinstance(out, xr.DataArray) and isinstance(out.data, np.ndarray)):
            return
        if "time" not in out.dims or out.ndim != 3 or out.sizes["time"] >= self.n_time:
            return
        self.refs.append(weakref.ref(_root(out.data)))
        gc.collect()
        # The chunk just loaded is still referenced by ``out`` here, so it counts.
        self.live.append(sum(r() is not None for r in self.refs))


def _census_run(data_bin, mask, monkeypatch, **mode_kwargs):
    census = _ChunkCensus()
    census.n_time = data_bin.sizes["time"]

    orig_compute = xr.DataArray.compute

    def compute(self, *args, **kwargs):
        out = orig_compute(self, *args, **kwargs)
        census.record(out)
        return out

    orig_merge = _merge_split.split_and_merge_objects

    def merge(*args, **kwargs):
        census.active = True
        try:
            return orig_merge(*args, **kwargs)
        finally:
            census.active = False

    with monkeypatch.context() as m:
        m.setattr(xr.DataArray, "compute", compute)
        m.setattr(_merge_split, "split_and_merge_objects", merge)
        tr = marEx.tracker(data_bin, mask, **mode_kwargs, **TRACKER_KWARGS)
        events = tr.run()
        events.ID_field.compute()
    return census


def test_streaming_merge_loop_holds_at_most_two_chunks(extremes, tmp_path, dask_client, monkeypatch):
    """Live chunk buffers at every load under streaming: the new one and its predecessor, at any nt.

    Persist mode deliberately batches its flush (every tenth chunk): each flush there is a
    setitem + persist of the whole field, so flushing per chunk costs more than it frees.
    """
    worst = {}
    for nt in (16, 32):
        data_bin = extremes.extreme_events.isel(time=slice(0, nt)).chunk({"time": TIME_CHUNK, "lat": -1, "lon": -1})
        kwargs = {"compute_mode": "streaming", "temp_dir": str(tmp_path / f"nt{nt}")}
        census = _census_run(data_bin, extremes.mask, monkeypatch, **kwargs)
        # Non-vacuous: the loop loaded every chunk through the spied compute.
        assert len(census.live) == nt // TIME_CHUNK, census.live
        worst[nt] = max(census.live)
    assert worst[16] == worst[32] <= 2, f"live chunk buffers per load grew with nt: {worst}"


# ---------------------------------------------------------------------------------------------
# Event metadata: global_ID, presence, the merge ledger and the recomputed properties are all
# (time, ID). They must stay dask-backed: never gathered to the driver as numpy, and under
# streaming never pinned in memory.
# ---------------------------------------------------------------------------------------------

UNSTRUCTURED_KWARGS = {
    "R_fill": 1,
    "area_filter_absolute": 5,
    "T_fill": 2,
    "allow_merging": True,
    "overlap_threshold": 0.8,
    "nn_partitioning": True,
    "unstructured_grid": True,
    "dimensions": {"x": "ncells"},
    "coordinates": {"x": "lon", "y": "lat"},
    "regional_mode": False,
    "coordinate_units": "degrees",
    "quiet": True,
}


def _has_time_and_id(dims) -> bool:
    return "time" in dims and "ID" in dims


class _MetadataCensus:
    """Record every (time, ID) array that reaches the driver as numpy, or is pinned.

    Active from the clustering step on, where the event metadata is built. The per-slice
    area census the tracker gathers before merging is a separate (time, label) array.
    """

    def __init__(self) -> None:
        self.active = False
        self.entered = False  # the clustering step ran: without it every census below is vacuous
        self.driver_shapes: list = []  # numpy results of dask-level computes (shape-matched later)
        self.pinned_shapes: list = []  # dask arrays persisted at any level (shape-matched later)
        self.driver_named: list = []  # numpy-backed xarray objects with (time, ID) dims
        self.pinned: list = []  # (time, ID) objects handed to persist

    def array_pinned(self, shape) -> None:
        if self.active:
            self.pinned_shapes.append(shape)

    def numpy_result(self, out) -> None:
        if self.active and isinstance(out, np.ndarray) and out.ndim >= 2:
            self.driver_shapes.append(out.shape)

    def xr_result(self, out, where: str) -> None:
        if not self.active:
            return
        items = out.data_vars.values() if isinstance(out, xr.Dataset) else [out]
        for v in items:
            if isinstance(v, xr.DataArray) and _has_time_and_id(v.dims) and isinstance(v.data, np.ndarray):
                self.driver_named.append((where, v.name, v.dims, v.shape))

    def persisted(self, objs, where: str) -> None:
        if not self.active:
            return
        for o in objs:
            items = o.data_vars.values() if isinstance(o, xr.Dataset) else [o]
            for v in items:
                if isinstance(v, xr.DataArray) and _has_time_and_id(v.dims):
                    self.pinned.append((where, v.name, v.dims, v.shape))


def _metadata_run(data_bin, mask, monkeypatch, **kwargs):
    import sys

    import dask
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
        # Choke points. Every result the distributed client brings back to the driver passes
        # through `Client._gather` (the finalize step runs on a worker), whatever the entry point
        # (.compute, .load, dask.compute, __array__, Client.gather, Future.result); every persist
        # of a dask Array rebuilds through `__dask_postpersist__`, including a raw Array.persist.
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

    def cluster(*a, **k):
        census.active = census.entered = True
        return orig_cluster(*a, **k)

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
        # `from dask import persist` binds the original at import time: rebind those names too.
        for name, mod in list(sys.modules.items()):
            if name.startswith("marEx") and getattr(mod, "persist", None) is orig["dask_persist"]:
                m.setattr(mod, "persist", dask_persist)
            if name.startswith("marEx") and getattr(mod, "full_like", None) is orig["full_like"]:
                m.setattr(mod, "full_like", full_like)
        tr = marEx.tracker(data_bin, mask, **kwargs)
        events = tr.run()
        census.active = False

    n_id = events.sizes["ID"]
    nt = events.sizes["time"]

    # Inside the clustering step the ID axis carries one padding entry, dropped on return.
    def time_by_id(s):
        return nt in s and (n_id in s or n_id + 1 in s)

    gathered = [s for s in census.driver_shapes if time_by_id(s)]
    pinned = [s for s in census.pinned_shapes if time_by_id(s)]
    backing = {v: type(events[v].data).__name__ for v in events.data_vars if _has_time_and_id(events[v].dims)}
    return census, gathered, pinned, backing, n_id


def _gridded_case(extremes, nt, tmp_path):
    data_bin = extremes.extreme_events.isel(time=slice(0, nt)).chunk({"time": TIME_CHUNK, "lat": -1, "lon": -1})
    return data_bin, extremes.mask, dict(compute_mode="streaming", temp_dir=str(tmp_path / f"g{nt}"), **TRACKER_KWARGS)


@pytest.fixture(scope="module")
def merging_unstructured():
    return xr.open_zarr(str(TEST_DATA_DIR / "extremes_unstructured_merging.zarr"), chunks={})


def _unstructured_case(data, nt, tmp_path):
    data_bin = data.extreme_events.isel(time=slice(0, nt)).chunk({"time": 4, "ncells": -1})
    kwargs = dict(
        compute_mode="streaming",
        temp_dir=str(tmp_path / f"u{nt}"),
        neighbours=data.neighbours,
        cell_areas=data.cell_areas,
        **UNSTRUCTURED_KWARGS,
    )
    return data_bin, data.mask, kwargs


@pytest.mark.parametrize("grid", ["gridded", "unstructured"])
def test_streaming_keeps_event_metadata_off_the_driver(grid, extremes, merging_unstructured, tmp_path, dask_client, monkeypatch):
    sizes = (16, 32) if grid == "gridded" else (50, 100)
    for nt in sizes:
        if grid == "gridded":
            data_bin, mask, kwargs = _gridded_case(extremes, nt, tmp_path)
        else:
            data_bin, mask, kwargs = _unstructured_case(merging_unstructured, nt, tmp_path)
        census, gathered, pinned, backing, n_id = _metadata_run(data_bin, mask, monkeypatch, **kwargs)
        assert census.entered, "the clustering step never ran: the census is vacuous"
        assert n_id > 1, "no events: the census is vacuous"
        assert backing and all(b != "ndarray" for b in backing.values()), f"nt={nt}: (time, ID) outputs held as numpy: {backing}"
        assert not gathered, f"nt={nt}: (time, ID) arrays gathered to the driver: {gathered}"
        assert not census.driver_named, f"nt={nt}: numpy-backed (time, ID) built on the driver: {census.driver_named}"
        assert not census.pinned, f"nt={nt}: (time, ID) pinned under streaming: {census.pinned}"
        assert not pinned, f"nt={nt}: (time, ID) dask arrays persisted under streaming: {pinned}"


# ---------------------------------------------------------------------------------------------
# Properties other than area and centroid are relabelled by lookup: at each (time, event) the
# value of the object the event held there, NaN where it held none.
# ---------------------------------------------------------------------------------------------


def _with_extra_property(object_props: xr.Dataset) -> xr.Dataset:
    ids = object_props["ID"]
    return object_props.assign(temperature=(ids * 0.25 + 1.0).astype(np.float32))


@pytest.mark.parametrize("compute_mode", ["persist", "streaming"])
def test_extra_property_is_relabelled_by_lookup(compute_mode, extremes, tmp_path, dask_client, monkeypatch):
    orig_cluster = _merge_split.cluster_rename_objects_and_props

    def cluster(object_id_field_unique, object_props, *a, **k):
        return orig_cluster(object_id_field_unique, _with_extra_property(object_props), *a, **k)

    monkeypatch.setattr(_merge_split, "cluster_rename_objects_and_props", cluster)
    data_bin = extremes.extreme_events.isel(time=slice(0, 16)).chunk({"time": TIME_CHUNK, "lat": -1, "lon": -1})
    events = marEx.tracker(data_bin, extremes.mask, compute_mode=compute_mode, temp_dir=str(tmp_path), **TRACKER_KWARGS).run()

    # The input properties keep their order in the output, ahead of the derived variables.
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
    ]
    got = events["temperature"]
    assert got.dims == ("time", "ID") and got.dtype == np.float32
    global_id = events["global_ID"].values
    assert (global_id > 0).any(), "no event is present anywhere: the lookup is vacuous"
    want = np.where(global_id > 0, (global_id * 0.25 + 1.0).astype(np.float32), np.float32(np.nan))
    np.testing.assert_array_equal(got.values, want)
