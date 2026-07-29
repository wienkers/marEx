"""Shared openers, configurations, helpers and module fixtures for the tracker tests.

Every opener returns a field rooted in a shipped zarr store, never a ``.persist()``ed one: a
persisted input holds futures on whichever client was current when it was built, and the tests
fail with ``lost dependencies`` once that client closes. ``test_tracker_config.py::
test_shared_fixtures_hold_no_futures`` runs every opener here under the session client.

The module fixtures (``golden_run``, ``gridded_basic_run``, ``gridded_mode_pair``) each run the
tracker once on the session ``client`` (where every consumer ran before the consolidation) and
return computed, numpy-backed datasets and recorded facts only, never a tracker or a lazy result.
"""

from __future__ import annotations

from pathlib import Path

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from distributed import Client

import marEx

from .._recorders import PersistRecorder
from ..conftest import released
from .mre_mesh import TRACKER_KWARGS as MESH_KWARGS
from .mre_mesh import tracker_inputs

DATA_DIR = Path(__file__).parent.parent / "data"

# The conftest pin a test that does not request the cluster runs under (tests/conftest.py).
THREADED = {"scheduler": "threads", "array.rechunk.method": "tasks"}

# Spatial dimensions must be contiguous for the tracker.
CHUNK_SIZE = {"time": 2, "lat": -1, "lon": -1}

# The golden configuration: merging with temporal gap filling and nearest-neighbour partitioning.
GOLDEN_KWARGS = {
    "area_filter_quartile": 0.5,
    "R_fill": 4,
    "T_fill": 2,
    "allow_merging": True,
    "overlap_threshold": 0.5,
    "nn_partitioning": True,
    "quiet": True,
}

# The compute-mode configuration (R_fill 8).
MODE_KWARGS = dict(GOLDEN_KWARGS, R_fill=8)

# The no-merge configuration of ``gridded_basic_run``.
BASIC_KWARGS = {"area_filter_quartile": 0.5, "R_fill": 4, "T_fill": 0, "allow_merging": False, "quiet": True}

UNSTRUCTURED_KWARGS = {
    "unstructured_grid": True,
    "dimensions": {"x": "ncells"},
    "coordinates": {"x": "lon", "y": "lat"},
    "regional_mode": False,
    "coordinate_units": "degrees",
    "quiet": True,
}

# The merging configuration on ``extremes_unstructured_merging.zarr``.
UNSTRUCTURED_MERGING_KWARGS = dict(
    UNSTRUCTURED_KWARGS,
    R_fill=1,
    area_filter_absolute=5,
    T_fill=2,
    allow_merging=True,
    overlap_threshold=0.8,
    nn_partitioning=True,
)


def open_store(name):
    """A shipped fixture store, lazily read and never persisted."""
    return xr.open_zarr(str(DATA_DIR / name), chunks={})


def open_gridded():
    """The shipped gridded extremes (32 steps x 180 x 360, time chunk 2 on disk)."""
    return open_store("extremes_gridded.zarr")


def open_gridded_blob():
    """The shipped steadily moving blob (3 degrees east per day from 170 E)."""
    return open_store("extremes_gridded_blob.zarr")


def open_unstructured():
    """The shipped unstructured extremes (100 steps x 405 cells)."""
    return open_store("extremes_unstructured.zarr")


def open_unstructured_merging():
    """The shipped unstructured extremes built to merge (100 steps x 405 cells)."""
    return open_store("extremes_unstructured_merging.zarr")


OPENERS = {
    "open_gridded": open_gridded,
    "open_gridded_blob": open_gridded_blob,
    "open_unstructured": open_unstructured,
    "open_unstructured_merging": open_unstructured_merging,
}


def polar_mask(ds):
    """``ds.mask`` with the north polar cap (lat >= 85) excluded."""
    return ds.mask.where((ds.lat < 85) & (ds.lat > -90), other=False)


def on_client(client):
    """True when dask's current scheduler is ``client``."""
    return getattr(dask.base.get_scheduler(), "__self__", None) is client


def check(failures, case, func):
    """Run one table row's assertions; record the first that fails, or the error the row raised, under its case id."""
    try:
        func()
    except (AssertionError, pytest.fail.Exception) as e:
        failures.append(f"{case}: {str(e).strip().splitlines()[0] if str(e).strip() else 'assertion failed'}")
    except Exception as e:  # noqa: BLE001 - a row that raises is a failing row; the others still run
        failures.append(f"{case}: raised {type(e).__name__}: {e}")


def assert_no_failures(failures):
    assert not failures, f"{len(failures)} row(s) failed: " + " | ".join(failures)


def graph_stats(dataset):
    """Task count and rechunk- and transpose-key counts of a dataset's combined graph."""
    graph = dataset.__dask_graph__()
    keys = [str(k[0]) if isinstance(k, tuple) else str(k) for k in graph.keys()]
    return {
        "n_tasks": len(keys),
        "n_rechunk": sum(1 for k in keys if "rechunk" in k or "shuffle" in k),
        "n_transpose": sum(1 for k in keys if "transpose" in k),
    }


def tile_to(da, nt, timedim="time"):
    """Repeat the record along time up to ``nt`` steps (values are irrelevant here)."""
    reps = int(np.ceil(nt / da.sizes[timedim]))
    out = xr.concat([da] * reps, dim=timedim).isel({timedim: slice(0, nt)})
    return out.assign_coords({timedim: np.arange(nt)})


def _capture_objects(tr):
    """Wrap ``tr``'s clustering step so the pre-cluster object field is kept as numpy."""
    captured = {}
    orig = tr.cluster_rename_objects_and_props

    def _capture(field, props, overlaps, merges):
        captured["objects"] = np.asarray(field.compute().values)
        return orig(field, props, overlaps, merges)

    tr.cluster_rename_objects_and_props = _capture
    return captured


def _track(binary_sq, nrow, ncol, time_chunk, temp_dir, serial=False):
    """Run the unstructured tracker on an ``mre_mesh`` scenario; return (event field, pre-cluster object field, n_merges).

    A tracker error is re-raised as ``AssertionError("the tracker run raised <type>: <message>")``.
    """

    def run():
        da, mask, neighbours, cell_areas = tracker_inputs(binary_sq, nrow, ncol, time_chunk)
        tr = marEx.tracker(da, mask, temp_dir=str(temp_dir), neighbours=neighbours, cell_areas=cell_areas, **MESH_KWARGS)
        if serial:
            tr.split_and_merge_objects_parallel = tr.split_and_merge_objects
        captured = _capture_objects(tr)
        events, merges = tr.run(return_merges=True)
        return np.asarray(events.ID_field.values), captured["objects"], int(merges.sizes.get("merge_ID", 0))

    return released(run, "the tracker run")


# The nine-cell equatorial chain: two objects join into one, which splits into a five-cell and a
# four-cell piece; the next object covers the four middle cells.
CHAIN_ROWS = [
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 0, 0],
]

CHAIN_KWARGS = dict(UNSTRUCTURED_KWARGS, R_fill=0, T_fill=0, area_filter_absolute=1, allow_merging=True, overlap_threshold=0.5)
CHAIN_KWARGS["nn_partitioning"] = False


def _chain_inputs(rows=CHAIN_ROWS):
    cells = np.array(rows, dtype=bool)
    nt, n = cells.shape
    lat = np.zeros(n, np.float32)
    lon = np.arange(-(n // 2), n - n // 2, dtype=np.float32)
    neighbours = np.zeros((3, n), np.int32)  # 1-based, 0 = none
    neighbours[0, 1:] = np.arange(1, n)
    neighbours[1, :-1] = np.arange(2, n + 1)
    spatial = {"lat": ("ncells", lat), "lon": ("ncells", lon)}
    times = pd.date_range("2000-01-01", periods=nt, freq="D")
    da = xr.DataArray(cells, dims=("time", "ncells"), coords={"time": times, **spatial}).chunk({"time": nt, "ncells": -1})
    mask = xr.DataArray(np.ones(n, bool), dims="ncells", coords=spatial).chunk({"ncells": -1})
    nb = xr.DataArray(neighbours, dims=("nv", "ncells")).chunk({"nv": -1, "ncells": -1})
    area = xr.DataArray(np.ones(n, np.float32), dims="ncells", coords=spatial).chunk({"ncells": -1})
    return da, mask, nb, area


def _track_chain(temp_dir, serial, rows=CHAIN_ROWS, **overrides):
    """Run the unstructured tracker on a chain of unit-area cells; return (event field, pre-cluster object field, n_merges).

    A tracker error is re-raised as ``AssertionError("the tracker run raised <type>: <message>")``.
    """

    def run():
        da, mask, nb, area = _chain_inputs(rows)
        tr = marEx.tracker(da, mask, temp_dir=str(temp_dir), neighbours=nb, cell_areas=area, **{**CHAIN_KWARGS, **overrides})
        if serial:
            tr.split_and_merge_objects_parallel = tr.split_and_merge_objects
        captured = _capture_objects(tr)
        events, merges = tr.run(return_merges=True)
        return np.asarray(events.ID_field.values), captured["objects"], int(merges.sizes.get("merge_ID", 0))

    return released(run, "the tracker run")


class _CapturingDask:
    """Stand-in for ``dask`` as ``merge_split`` sees it that keeps the ``process_chunk`` closure it is handed."""

    def __init__(self, real):
        self._real = real
        self.kernels = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def delayed(self, func, *args, **kwargs):
        if getattr(func, "__name__", None) == "process_chunk":
            self.kernels.append(func)
        return self._real.delayed(func, *args, **kwargs)


@pytest.fixture
def merge_kernel(client, tmp_path, monkeypatch):
    """The real ``process_chunk`` closure, captured from a tracker run on the nine-cell chain."""
    from marEx.track import merge_split

    capture = _CapturingDask(merge_split.dask)
    monkeypatch.setattr(merge_split, "dask", capture)
    _track_chain(tmp_path / "capture", serial=False)
    monkeypatch.setattr(merge_split, "dask", capture._real)
    assert capture.kernels, "the merge loop never reached process_chunk"
    return capture.kernels[0]


@pytest.fixture
def inprocess_client():
    """Threads in this process, so a patched partitioner is the one the merge tasks call.

    Function-scoped: while it is open it is dask's default client, so a module-scoped one would
    take over every later ``client`` test in the same module."""
    client = Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None)
    yield client
    client.close()


@pytest.fixture(scope="module")
def golden_run(client):
    """One tracker run in the golden configuration, computed before any client teardown: (events, merges)."""
    assert on_client(client), "golden_run is not computing on the session client"
    ds = open_gridded()
    tracker = marEx.tracker(ds.extreme_events.chunk(CHUNK_SIZE), polar_mask(ds), **GOLDEN_KWARGS)
    events, merges = tracker.run(return_merges=True)
    return events.compute(), merges.compute()


@pytest.fixture(scope="module")
def gridded_basic_run(client):
    """One no-merge tracker run (R_fill 4, quartile 0.5, T_fill 0, polar cap masked), computed."""
    assert on_client(client), "gridded_basic_run is not computing on the session client"
    ds = open_gridded()
    return marEx.tracker(ds.extreme_events, polar_mask(ds), **BASIC_KWARGS).run().compute()


@pytest.fixture(scope="module")
def gridded_mode_pair(client, tmp_path_factory):
    """One persist and one streaming run in the compute-mode configuration, with the pin recorder on both.

    Returns only computed outputs and recorded facts. The persist graph statistics are taken from
    the lazy persist result. The streaming result is computed, written and reopened, and its staging
    facts are recorded, before ``clear_staging``; no test reads the lazy streaming dataset.
    """
    assert on_client(client), "gridded_mode_pair is not computing on the session client"
    root = tmp_path_factory.mktemp("mode_pair")
    ds = open_gridded()
    data_bin = ds.extreme_events.chunk(CHUNK_SIZE)
    facts = {"field_bytes": data_bin.size * 4}  # one whole int32 field

    persist_dir = root / "persist"
    persist_dir.mkdir()
    with PersistRecorder() as persist_rec:
        events, merges = marEx.tracker(data_bin, ds.mask, temp_dir=str(persist_dir), **MODE_KWARGS).run(return_merges=True)
    facts["persist_graph"] = graph_stats(events)
    facts["persist_pinned"] = persist_rec.bytes
    facts["persist_staging_attr"] = "marex_staging_dir" in events.attrs
    facts["persist_staging_encoding"] = "marex_staging_dir" in events.encoding
    persist_events, persist_merges = events.compute(), merges.compute()
    facts["persist_staged_ledgers"] = sorted(str(p) for p in persist_dir.glob("**/merge_ledger.zarr"))

    stream_dir = root / "streaming"
    stream_dir.mkdir()
    with PersistRecorder(allow_empty=True) as stream_rec:
        lazy, lazy_merges = marEx.tracker(data_bin, ds.mask, compute_mode="streaming", temp_dir=str(stream_dir), **MODE_KWARGS).run(
            return_merges=True
        )
    facts["stream_pinned"] = stream_rec.bytes
    staged = lazy.encoding.get("marex_staging_dir")
    facts["staging_dir"] = staged
    facts["staging_exists_after_run"] = staged is not None and Path(staged).exists()
    facts["staging_holds"] = sorted(p.name for p in Path(staged).iterdir()) if facts["staging_exists_after_run"] else []
    stream_events, stream_merges = lazy.compute(), lazy_merges.compute()
    out = root / "out.zarr"
    lazy.to_zarr(out, mode="w")
    reopened = xr.open_zarr(out)
    facts["written_attrs_carry_staging"] = "marex_staging_dir" in reopened.attrs
    reopened.close()
    marEx.clear_staging(lazy)
    facts["staging_exists_after_clear"] = staged is not None and Path(staged).exists()
    del lazy, lazy_merges
    return {
        "persist": (persist_events, persist_merges),
        "streaming": (stream_events, stream_merges),
        "facts": facts,
    }
