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
