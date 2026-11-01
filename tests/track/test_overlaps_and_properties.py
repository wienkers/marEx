"""Overlaps and object properties: the overlap payload and threshold, the child first-time lookup and its
cost, structured and unstructured object properties (values, laziness, label magnitude), the props store,
and the float64 accumulation rule for every unstructured area.

Value tests compare returned labels, so they cannot see an intermediate that is wider, larger or more often
shipped than it needs to be; the census tests here intercept the intermediates themselves, and assert
invariance in the record length or query size rather than an absolute bound.

Each test runs on the scheduler its pre-consolidation rows ran on: a test that requests a client wraps its
threaded-origin rows in the conftest pin (``THREADED``).
"""

from __future__ import annotations

import collections
import inspect
import json
import logging
import pickle
import subprocess
import sys
import textwrap
import tracemalloc

import dask
import dask.array as da
import dask.array.core as dask_array_core
import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.track import merge_split
from marEx.track import objects as _objects
from marEx.track import overlap as _overlap
from marEx.track.objects import ObjectPropsStore, calculate_object_properties, calculate_partitioned_child_properties
from marEx.track.overlap import check_overlap_slice

from .conftest import THREADED, UNSTRUCTURED_MERGING_KWARGS, assert_no_failures, check, on_client, open_unstructured_merging

# -- structured child properties and the props store --------------------------------------------------


def _structured_props(ids, regional_mode=False):
    """Full-slice structured ``calculate_object_properties`` (area, centroid) for a 2-D label array, by ID."""
    return calculate_object_properties(
        xr.DataArray(ids, dims=["lat", "lon"]),
        unstructured_grid=False,
        lat=None,
        lon=None,
        cell_area=None,
        timedim="time",
        regional_mode=regional_mode,
        ydim="lat",
        xdim="lon",
        properties=["area", "centroid"],
    ).sortby("ID")


def test_partitioned_child_properties_match_regionprops():
    """``calculate_partitioned_child_properties`` reproduces the full-slice structured properties (pixel-count
    area, pixel-coordinate centroid with the same antimeridian convention) from a label set's own pixels."""
    interior = np.zeros((30, 60), dtype=np.int32)
    interior[5:10, 5:12] = 7
    interior[20:25, 30:40] = 13
    interior[15:18, 45:50] = 21
    one_edge = np.zeros((30, 60), dtype=np.int32)
    one_edge[5:10, 0:8] = 3  # left edge only: no wrap adjustment
    one_edge[20:25, 52:60] = 9  # right edge only
    straddle = np.zeros((20, 200), dtype=np.int32)
    straddle[8:12, 0:5] = 4  # pixels near BOTH x-edges: wrap-adjusted x-centroid
    straddle[8:12, 196:200] = 4
    straddle[2:6, 90:100] = 8  # an ordinary interior blob alongside it
    regional = straddle.copy()
    regional[2:6, 90:100] = 0

    failures = []
    for case, ids, regional_mode in (
        ("interior", interior, False),
        ("one_edge", one_edge, False),
        ("antimeridian_straddle", straddle, False),
        ("regional", regional, True),
    ):

        def row(ids=ids, regional_mode=regional_mode):
            oracle = _structured_props(ids, regional_mode)
            y_idx, x_idx = np.nonzero(ids)
            helper = calculate_partitioned_child_properties(
                y_idx, x_idx, ids[y_idx, x_idx], Nx=ids.shape[1], regional_mode=regional_mode
            ).sortby("ID")
            np.testing.assert_array_equal(helper.ID.values, oracle.ID.values)
            np.testing.assert_array_equal(helper.area.values.astype(np.int64), oracle.area.values.astype(np.int64))
            max_diff = float(np.max(np.abs(helper.centroid.values - oracle.centroid.values)))
            assert max_diff < 1e-6, f"centroid diff {max_diff}"
            if regional_mode:
                # Sanity: the straddling blob's raw x-mean is mid-grid (~100), not wrap-adjusted.
                wrapped = _structured_props(ids, regional_mode=False)
                assert abs(float(wrapped.centroid.sel(ID=4).values[1]) - float(oracle.centroid.sel(ID=4).values[1])) > 50

        check(failures, case, row)
    assert_no_failures(failures)


def test_object_props_store_matches_xarray_semantics():
    """``ObjectPropsStore`` round-trips a properties dataset (IDs sorted), its lookups match ``.sel(ID=...)``
    order-preservingly, and set / drop / insert keep ``max_id`` and membership right."""
    ds = xr.Dataset(
        {
            "area": ("ID", np.array([100, 50, 7])),
            "centroid": (("component", "ID"), np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])),
        },
        coords={"ID": np.array([5, 2, 9], dtype=np.int32)},
    )
    store = ObjectPropsStore.from_dataset(ds)
    out = store.to_dataset()
    np.testing.assert_array_equal(out.ID.values, np.array([2, 5, 9], dtype=np.int32))
    ds_sorted = ds.sortby("ID")
    np.testing.assert_array_equal(out.area.values.astype(np.int64), ds_sorted.area.values.astype(np.int64))
    np.testing.assert_allclose(out.centroid.values, ds_sorted.centroid.values)

    ids = np.array([9, 5])
    np.testing.assert_array_equal(store.areas(ids), ds["area"].sel(ID=ids).values)
    np.testing.assert_allclose(store.centroids(ids), ds["centroid"].sel(ID=ids).values.T)  # (k, 2), [:, 0] = y
    np.testing.assert_allclose(store.centroid(2), ds["centroid"].sel(ID=2).values)
    assert store.area(5) == 100
    assert store.max_id() == 9
    assert 2 in store and 999 not in store

    store.set(2, 99, 5.0, 6.0)  # update existing
    assert store.area(2) == 99
    np.testing.assert_allclose(store.centroid(2), [5.0, 6.0])
    store.set(50, 7, 1.0, 2.0)  # insert new
    assert 50 in store and store.max_id() == 50
    store.drop(5)
    assert 5 not in store
    np.testing.assert_array_equal(store.to_dataset().ID.values, np.array([2, 9, 50], dtype=np.int32))


# -- the overlap threshold -----------------------------------------------------------------------------


def _small_tracker(ny, extent, **kwargs):
    data = xr.DataArray(
        np.random.default_rng(0).random((5, ny, ny)) > 0.7,
        dims=["time", "lat", "lon"],
        coords={"time": range(5), "lat": np.linspace(-2 * extent, 2 * extent, ny), "lon": np.linspace(-extent, extent, ny)},
    ).chunk({"time": 5})
    return marEx.tracker(
        data,
        xr.ones_like(data.isel(time=0), dtype=bool),
        R_fill=0,
        area_filter_quartile=0.5,
        regional_mode=True,
        coordinate_units="degrees",
        **kwargs,
    )


def test_enforce_overlap_threshold(client, marex_records):
    """``enforce_overlap_threshold`` keeps pairs whose overlap over the smaller area reaches the threshold.

    * No pair's IDs are known: an empty (0, 3) int32 array.
    * At threshold 0.1: 150 over min(100, 120) = 1.5 is kept and logged as an overlap fraction > 1;
      80 / 200 = 0.4 is kept; 5 / 100 = 0.05 is dropped.
    """
    assert on_client(client)
    failures = []

    def no_valid_ids():
        with dask.config.set(THREADED):
            tracker = _small_tracker(5, 5)
        overlaps = np.array([[999, 1000, 50.0], [1001, 1002, 60.0]], dtype=np.float32)
        store = ObjectPropsStore(area={1: 100.0, 2: 200.0}, cy={1: 0.0, 2: 1.0}, cx={1: 0.0, 2: 1.0})
        result = tracker.enforce_overlap_threshold(overlaps, store)
        assert result.shape == (0, 3), result.shape
        assert result.dtype == np.int32, result.dtype  # structured grid

    def fraction_above_one():
        tracker = _small_tracker(10, 10, overlap_threshold=0.1)
        records = marex_records(logging.WARNING)
        overlaps = np.array([[1, 2, 150], [3, 4, 80], [5, 6, 5]], dtype=np.int32)
        areas = {1: 100.0, 2: 120.0, 3: 200.0, 4: 300.0, 5: 100.0, 6: 100.0}
        store = ObjectPropsStore(area=areas, cy={k: 0.0 for k in areas}, cx={k: 0.0 for k in areas})
        result = tracker.enforce_overlap_threshold(overlaps, store)
        assert result.tolist() == [[1, 2, 150], [3, 4, 80]], result.tolist()
        messages = [r.getMessage() for r in records]
        assert any("Found 1 overlap fractions > 1.0" in m for m in messages), messages

    check(failures, "no_valid_ids", no_valid_ids)
    check(failures, "fraction_above_one", fraction_above_one)
    assert_no_failures(failures)


# -- the overlap payload -------------------------------------------------------------------------------


class _StopAtPersist(Exception):
    def __init__(self, obj):
        super().__init__("graph captured")
        self.obj = obj


def _overlap_graph(n_cells, monkeypatch, n_time=8, time_chunk=2):
    """Build (but never run) the unstructured overlap graph; return its task specs and the area bytes."""
    ids = xr.DataArray(da.zeros((n_time, n_cells), dtype=np.int32, chunks=(time_chunk, n_cells)), dims=("time", "ncells"))
    cell_area = xr.DataArray(np.linspace(1.0, 2.0, n_cells, dtype=np.float32), dims=("ncells",))

    def stop(self, *args, **kwargs):
        raise _StopAtPersist(self)

    with monkeypatch.context() as patched:
        patched.setattr(xr.DataArray, "persist", stop)
        with pytest.raises(_StopAtPersist) as captured:
            _overlap.find_overlapping_objects(ids, "time", True, None, "ncells", cell_area)
    return dict(captured.value.obj.__dask_graph__()), cell_area.nbytes


def test_overlap_area_is_not_embedded_per_task(monkeypatch):
    """The per-cell area is shipped once as a graph dependency, not inside every overlap task."""
    largest_other = {}
    for n_cells in (100_000, 200_000):
        graph, area_bytes = _overlap_graph(n_cells, monkeypatch)
        spec_bytes = {k: len(pickle.dumps(v)) for k, v in graph.items()}
        carriers = [k for k, b in spec_bytes.items() if b > area_bytes // 2]
        assert len(carriers) == 1, (
            f"{len(carriers)} task specs carry a copy of the {area_bytes} B area at {n_cells} cells "
            f"(max spec {max(spec_bytes.values())} B); expected exactly one shared area key"
        )
        # Every other spec stays small, whatever the mesh size.
        largest_other[n_cells] = max(b for k, b in spec_bytes.items() if k not in carriers)
        # The area key is a dependency of the overlap kernels, not a stray.
        (area_key,) = carriers
        dependents = [k for k, v in graph.items() if area_key in getattr(v, "dependencies", ())]
        assert len(dependents) >= 4, f"area key feeds {len(dependents)} tasks, expected one per time chunk"
    # Every other spec is independent of the mesh size.
    assert largest_other[200_000] <= largest_other[100_000] + 64, largest_other


def test_overlap_area_matches_cells_by_position():
    """A coordinate on the area's cell dimension is not used to align it to the field."""
    ids = np.array([[1, 1, 0, 2], [1, 1, 0, 2], [0, 0, 0, 0]], dtype=np.int32)
    field = xr.DataArray(ids, dims=("time", "ncells"), coords={"ncells": [0, 1, 2, 3]}).chunk({"time": 1})
    area = xr.DataArray(np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32), dims=("ncells",), coords={"ncells": [3, 2, 1, 0]})
    result = _overlap.find_overlapping_objects(field, "time", True, None, "ncells", area)
    np.testing.assert_array_equal(result, np.array([[1, 1, 3.0], [2, 2, 8.0]], dtype=np.float32))


# -- the child first-time lookup -------------------------------------------------------------------------


def _lifetime_field(n_time, n_cells=60):
    """IDs 2t+1 and 2t+2 are born at t and live three steps; returns the field and the children queried."""
    field = np.zeros((n_time, n_cells), dtype=np.int32)
    for t in range(n_time):
        for k, obj in enumerate((2 * t + 1, 2 * t + 2)):
            cells = slice((2 * (t % 5) + k) * 6, (2 * (t % 5) + k) * 6 + 6)
            field[t : t + 3, cells] = obj
    children = [2 * t + 1 for t in range(n_time)]
    return field, children


def _id_time_work(n_time, monkeypatch, time_chunk=5):
    """Per source block: bytes of every element-wise comparison built, and tasks run, while looking up each
    child's first time. Blockwise fusion hides a comparison inside one task, so the comparison is measured
    where it is built, not from task results."""
    from dask.callbacks import Callback

    field, children = _lifetime_field(n_time)
    field = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": time_chunk, "ncells": -1})
    compared, n_tasks = [], [0]
    original_eq = xr.DataArray.__eq__

    def eq_spy(self, other):
        out = original_eq(self, other)
        compared.append(out.size * out.dtype.itemsize)
        return out

    def count(key, result, dsk, state, id):
        n_tasks[0] += 1

    with monkeypatch.context() as patched:
        patched.setattr(xr.DataArray, "__eq__", eq_spy)
        with dask.config.set(scheduler="sync"), Callback(posttask=count):
            mapping = _overlap.compute_id_time_dict(
                field, children, 2 * n_time + 1, "time", True, None, "ncells", all_objects=False
            )

    n_source_blocks = n_time // time_chunk
    return sum(compared) / n_source_blocks, n_tasks[0] / n_source_blocks, mapping


def test_child_time_work_per_block_is_flat(monkeypatch):
    """The first-time lookup costs the same per source block whatever the record length and query size."""
    bytes_short, tasks_short, map_short = _id_time_work(20, monkeypatch)
    bytes_long, tasks_long, map_long = _id_time_work(40, monkeypatch)
    # Twice the record AND twice the queried children: a (time x buffer x children) comparison grows 4x in
    # total, 2x per source block, however it is chunked.
    assert bytes_long <= 1.25 * bytes_short + 1024, f"{bytes_short:.0f} -> {bytes_long:.0f} compared B per source block"
    assert tasks_long <= 1.25 * tasks_short, f"{tasks_short:.1f} -> {tasks_long:.1f} tasks per source block"
    assert map_short == {2 * t + 1: t for t in range(20)}
    assert map_long == {2 * t + 1: t for t in range(40)}


def test_child_first_time_matches_oracle():
    """Each queried ID maps to the first time it is present (absent IDs to 0), at time chunks 1, 3 and 7, for a
    query and for ``all_objects``."""
    failures = []
    for time_chunk in (1, 3, 7):
        rng = np.random.default_rng(time_chunk)
        field = rng.integers(0, 30, size=(17, 40)).astype(np.int32)
        field[field < 12] = 0
        field[:, 0] = 0  # every slice has background
        for all_objects in (False, True):

            def row(field=field, time_chunk=time_chunk, all_objects=all_objects):
                max_objects = 35  # IDs 30..34 never occur: absent IDs map to 0
                children = [33, 5, 29, 12, 31, 18]
                lazy = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": time_chunk, "ncells": -1})
                mapping = _overlap.compute_id_time_dict(
                    lazy, children, max_objects, "time", True, None, "ncells", all_objects=all_objects
                )
                oracle = {}
                for obj in range(max_objects) if all_objects else children:
                    present = np.nonzero((field == obj).any(axis=1))[0]
                    oracle[int(obj)] = int(present[0]) if present.size else 0
                assert mapping == oracle

            check(failures, f"time_chunk={time_chunk} all_objects={all_objects}", row)
    assert_no_failures(failures)


def test_background_id_follows_the_padded_slice_rule():
    """ID 0 is 'present' where a slice has background or fewer distinct IDs than the old buffer
    (buffer = int(101 / 101 * 100) = 100 distinct IDs per slice)."""
    failures = []
    for case, n_first in (("first_slice_full_buffer", 100), ("first_slice_padded", 1)):

        def row(n_first=n_first):
            n_time = max_objects = 101
            field = np.zeros((n_time, 100), dtype=np.int32)
            field[0] = np.arange(1, 101) if n_first == 100 else 1
            lazy = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": 10, "ncells": -1})
            mapping = _overlap.compute_id_time_dict(lazy, [], max_objects, "time", True, None, "ncells", all_objects=True)
            assert mapping[0] == (1 if n_first == 100 else 0), mapping[0]
            assert mapping[1] == 0 and mapping[100] == 0

        check(failures, case, row)
    assert_no_failures(failures)


def test_child_time_query_enters_the_graph_once(monkeypatch):
    """The query is one shared node: the per-block graph has the same shape whichever children are queried,
    and no block task carries its own copy of the query."""
    field, children = _lifetime_field(20)
    lazy = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": 5, "ncells": -1})
    shapes, block_task_bytes = [], []
    original = dask.compute

    def spy(*args, **kwargs):
        graph = {k: v for a in args if dask.is_dask_collection(a) for k, v in dict(a.__dask_graph__()).items()}
        names = [str(k[0] if isinstance(k, tuple) else k).rsplit("-", 1)[0] for k in graph]
        shapes.append(collections.Counter(names))
        block_task_bytes.append(sorted(len(pickle.dumps(v)) for k, v in graph.items() if "_first_time_in_block" in str(k)))
        return original(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(dask, "compute", spy)
        for query in (children[:3], children):
            _overlap.compute_id_time_dict(lazy, query, 41, "time", True, None, "ncells", all_objects=False)
    assert len(shapes) == 2 and shapes[0] == shapes[1], shapes
    assert shapes[0]["ndarray"] == 1, f"the query should be one shared node: {shapes[0]}"
    assert block_task_bytes[0] == block_task_bytes[1], block_task_bytes


def test_block_filter_allocation_does_not_follow_the_query():
    """Filtering a block to the query allocates by the block's IDs, not by the query's length."""
    block = np.arange(200, dtype=np.int32).reshape(4, 50)

    def peak(n_query):
        keep = np.arange(0, 2 * n_query, 2, dtype=np.int64)  # sorted, half the block's IDs
        tracemalloc.start()
        try:
            ids, _ = _overlap._first_time_in_block(block, 0, 10, keep)
            return tracemalloc.get_traced_memory()[1], ids
        finally:
            tracemalloc.stop()

    # tracemalloc also sees other threads' allocations, so keep the smallest of five runs.
    (small, ids_small), (large, ids_large) = (min((peak(n) for _ in range(5)), key=lambda r: r[0]) for n in (10_000, 1_000_000))
    assert np.array_equal(ids_small, ids_large) and np.array_equal(ids_small, np.arange(0, 200, 2))
    assert large <= 2 * small + 64 * 1024, f"block peak {small} B with a 1e4 query, {large} B with 1e6"


def test_gathered_summary_follows_the_query(monkeypatch):
    """With a fixed query, the bytes gathered to the driver do not grow with unrelated objects."""
    gathered = []
    original = dask.compute

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        gathered.append(sum(arr.nbytes for part in out for arr in part))
        return out

    def run(n_unrelated):
        n_time, n_cells = 20, 2 + n_unrelated
        field = np.zeros((n_time, n_cells), dtype=np.int32)
        field[3:, 0], field[7:, 1] = 1, 2  # the two queried objects
        # unrelated objects: a new ID in every cell at every step
        field[:, 2:] = 3 + np.arange(n_time * n_unrelated, dtype=np.int32).reshape(n_time, n_unrelated)
        lazy = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": 5, "ncells": -1})
        gathered.clear()
        with monkeypatch.context() as patched:
            patched.setattr(dask, "compute", spy)
            mapping = _overlap.compute_id_time_dict(
                lazy, [2, 1], int(field.max()) + 1, "time", True, None, "ncells", all_objects=False
            )
        assert mapping == {2: 7, 1: 3}
        return gathered[0]

    small, large = run(10), run(80)
    assert large == small, f"gathered {small} B with 10 unrelated objects per step, {large} B with 80"


def test_child_time_lookup_rejects_split_space():
    """Background and buffer membership are whole-slice properties, so a split slice is refused, naming the fix."""
    field = np.zeros((4, 6), dtype=np.int32)
    field[:, :3] = 1
    lazy = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": 2, "ncells": 3})
    with pytest.raises(ValueError, match=r"needs \['ncells'\] in a single chunk"):
        _overlap.compute_id_time_dict(lazy, [1], 2, "time", True, None, "ncells", all_objects=False)


def test_lookup_memory_does_not_depend_on_the_query():
    """Peak allocation inside each block does not change with the number of queried children, and the extra
    peak of a large query over a small one does not grow with the record.

    Measured in a fresh interpreter: tracemalloc counts every thread's allocations, and in the test process a
    Dask client's event loop and its nannies' monitors allocate and free at any moment, which moved single
    block peaks by up to a megabyte in either direction."""
    script = inspect.getsource(_lifetime_field) + textwrap.dedent(
        """
        import gc
        import json
        import tracemalloc

        import dask
        import numpy as np
        import xarray as xr

        from marEx.track import overlap as _overlap

        block_peaks = []
        original = _overlap._first_time_in_block

        def measured(block, t0, buffer, keep=None):
            tracemalloc.reset_peak()
            start = tracemalloc.get_traced_memory()[0]
            out = original(block, t0, buffer, keep)
            block_peaks.append(tracemalloc.get_traced_memory()[1] - start)
            return out

        _overlap._first_time_in_block = measured

        def run(n_time, n_children):
            field, _ = _lifetime_field(n_time)
            lazy = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": 5, "ncells": -1})
            query = list(range(1, n_children + 1))
            block_peaks.clear()
            gc.collect()
            tracemalloc.start()
            try:
                with dask.config.set(scheduler="sync"):
                    _overlap.compute_id_time_dict(lazy, query, 2 * n_time + 1, "time", True, None, "ncells", all_objects=False)
                whole = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            return list(block_peaks), whole

        run(20, 2)  # first-call allocations (lazy imports, caches) are not the lookup's
        print(json.dumps({n: {q: [run(n, q) for _ in range(3)] for q in (2, 2000)} for n in (20, 40)}))
        """
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    runs = json.loads(done.stdout.strip().splitlines()[-1])

    extra = {}
    for n_time in (20, 40):
        few, many = runs[str(n_time)]["2"], runs[str(n_time)]["2000"]
        assert all(len(blocks) == n_time // 5 for blocks, _ in few + many)
        median_few = float(np.median(few[0][0]))
        median_many = float(np.median(many[0][0]))
        # A (time x buffer x children) array in a block adds ~1 kB per queried child here.
        assert abs(median_many - median_few) <= 16 * 2000, (few[0][0], many[0][0])
        extra[n_time] = min(w for _, w in many) - min(w for _, w in few)
    # The lookup itself is O(queried children); a (time x buffer x children) array doubles here.
    assert extra[40] <= 1.25 * extra[20] + 256 * 2000, extra


# -- unstructured object properties -----------------------------------------------------------------------
#
# The per-cell geometry arrays (lat, lon, cell_area) used to be broadcast against the (time, xdim) ID field
# with ``xr.broadcast`` before ``xr.apply_ufunc``. The ID field is dask-backed and the geometry is not, so
# ``xr.broadcast`` returned a numpy zero-stride view, and ``dask.array.from_array`` copies any array-like
# unconditionally: the view was densified to (n_time, n_cells) float64 AT GRAPH-BUILD TIME, IN THE CLIENT --
# 122 GiB on the ICON R02B09 mesh at n_time=1096. The discriminating assertion is invariance in n_time, not
# an absolute byte bound.

XDIM = "ncells"
TIMEDIM = "time"
N_GEOMETRY_CELLS = 400


def _geometry(n_cells=N_GEOMETRY_CELLS, seed=0):
    """1-D per-cell geometry arrays, as the unstructured tracker holds them."""
    rng = np.random.default_rng(seed)
    lat = xr.DataArray(rng.uniform(-80.0, 80.0, n_cells).astype(np.float64), dims=(XDIM,), name="lat")
    lon = xr.DataArray(rng.uniform(-180.0, 180.0, n_cells).astype(np.float64), dims=(XDIM,), name="lon")
    cell_area = xr.DataArray(rng.uniform(1.0, 2.0, n_cells).astype(np.float64), dims=(XDIM,), name="cell_area")
    return lat, lon, cell_area


def _unique_id_field(n_time, n_cells=N_GEOMETRY_CELLS, seed=1, chunk_time=4):
    """Dask-backed unstructured ID field, IDs unique across time as the tracker makes them."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 4, size=(n_time, n_cells)).astype(np.int32)
    offsets = (np.arange(n_time, dtype=np.int32) * 3)[:, None]
    ids = np.where(ids > 0, ids + offsets, 0).astype(np.int32)
    return xr.DataArray(da.from_array(ids, chunks=(chunk_time, n_cells)), dims=(TIMEDIM, XDIM), name="object_id_field")


def _unstructured_props(object_id_field, lat, lon, cell_area):
    return calculate_object_properties(
        object_id_field, True, lat, lon, cell_area, TIMEDIM, False, None, XDIM, properties=["area", "centroid"]
    )


class _EagerWrapSpy:
    """Record every numpy array dask eagerly wraps (and therefore copies) via ``from_array``.

    ``apply_gufunc`` calls ``asarray`` on each argument, which resolves ``from_array`` inside
    ``dask.array.core``; patching the attribute on that module is what intercepts it (patching
    ``dask.array.from_array`` would miss the intra-module binding).
    """

    def __init__(self):
        self.wrapped = []

    def __enter__(self):
        self._orig = dask_array_core.from_array

        def spy(x, *args, **kwargs):
            if isinstance(x, np.ndarray):
                self.wrapped.append((tuple(x.shape), str(x.dtype), int(x.nbytes)))
            return self._orig(x, *args, **kwargs)

        dask_array_core.from_array = spy
        return self

    def __exit__(self, *exc):
        dask_array_core.from_array = self._orig
        return False

    @property
    def total_bytes(self):
        return sum(nbytes for _, _, nbytes in self.wrapped)


def test_unstructured_properties_copy_nothing_per_timestep():
    """At n_time 8 and 64: the bytes copied at graph-build time are equal, no copied array carries the time
    axis, at most one geometry array is copied (less than one int32 field), and the properties compute."""
    seen = {}
    for n_time in (8, 64):
        lat, lon, cell_area = _geometry()
        with _EagerWrapSpy() as spy:
            props = _unstructured_props(_unique_id_field(n_time), lat, lon, cell_area)
        props.compute()
        seen[n_time] = (spy.total_bytes, list(spy.wrapped))
    short_bytes, long_bytes = seen[8][0], seen[64][0]
    assert long_bytes == short_bytes, (
        f"eager materialisation grew with n_time: {short_bytes} B at n_time=8 vs {long_bytes} B at n_time=64 -- "
        f"a geometry array is being broadcast to the whole field before apply_ufunc"
    )
    for n_time, (total, wrapped) in seen.items():
        offenders = [w for w in wrapped if n_time in w[0]]
        assert not offenders, f"whole-field arrays eagerly copied by dask.from_array at n_time={n_time}: {offenders}"
        assert total <= N_GEOMETRY_CELLS * 8, f"n_time={n_time}: {total} B copied, more than one geometry array"
        assert total < n_time * N_GEOMETRY_CELLS * 4, f"n_time={n_time}: {total} B copied, a whole int32 field"


def test_unstructured_properties_correct_for_every_time_layout():
    """Areas are the exact (float32-rounded) sums of the member cells' areas, and the IDs are right, for a
    multi-step field, a size-1 time axis (the non-parallel branch, which once raised IndexError on a
    transposed broadcast), no time axis, and a split cell dimension with and without a size-1 time axis."""
    lat, lon, cell_area = _geometry()
    area_vals = cell_area.values

    def expected_area(cells):
        return np.float32(area_vals[cells].astype(np.float32).sum())

    def multi_step():
        ids = np.zeros((4, N_GEOMETRY_CELLS), dtype=np.int32)
        ids[0, 0:10], ids[1, 5:25], ids[2, 100:103] = 7, 11, 13
        props = _unstructured_props(
            xr.DataArray(da.from_array(ids, chunks=(2, N_GEOMETRY_CELLS)), dims=(TIMEDIM, XDIM)), lat, lon, cell_area
        )
        props = props.compute()
        assert sorted(props.ID.values.tolist()) == [7, 11, 13], props.ID.values.tolist()
        for oid, cells in ((7, slice(0, 10)), (11, slice(5, 25)), (13, slice(100, 103))):
            np.testing.assert_allclose(props.area.sel(ID=oid).values, expected_area(cells), rtol=1e-6)

    def size_one_time():
        ids = np.zeros((1, N_GEOMETRY_CELLS), dtype=np.int32)
        ids[0, 3:9] = 5
        props = _unstructured_props(
            xr.DataArray(da.from_array(ids, chunks=(1, N_GEOMETRY_CELLS)), dims=(TIMEDIM, XDIM)), lat, lon, cell_area
        )
        props = props.compute()
        assert props.ID.values.tolist() == [5]
        np.testing.assert_allclose(props.area.values[0], expected_area(slice(3, 9)), rtol=1e-6)

    def no_time():
        ids = np.zeros(N_GEOMETRY_CELLS, dtype=np.int32)
        ids[10:20] = 3
        props = _unstructured_props(xr.DataArray(da.from_array(ids, chunks=(N_GEOMETRY_CELLS,)), dims=(XDIM,)), lat, lon, cell_area)
        props = props.compute()
        assert props.ID.values.tolist() == [3]
        np.testing.assert_allclose(props.area.values[0], expected_area(slice(10, 20)), rtol=1e-6)

    def split_cells(with_time_axis):
        # The single-slice path takes unpadded output and does not need a whole cell dimension.
        ids = np.zeros(N_GEOMETRY_CELLS, dtype=np.int32)
        ids[:6] = [1, 1, 2, 3, 3, 3]
        field = xr.DataArray(da.from_array(ids, chunks=N_GEOMETRY_CELLS // 4), dims=(XDIM,))
        if with_time_axis:
            field = field.expand_dims(TIMEDIM)
        props = _unstructured_props(field, lat, lon, cell_area).compute()
        assert props["ID"].values.tolist() == [1, 2, 3], props["ID"].values.tolist()

    failures = []
    for case, func in (
        ("multi_step", multi_step),
        ("size_one_time", size_one_time),
        ("no_time", no_time),
        ("split_cells_size_one_time", lambda: split_cells(True)),
        ("split_cells_no_time", lambda: split_cells(False)),
    ):
        check(failures, case, func)
    assert_no_failures(failures)


def test_property_buffer_holds_the_busiest_timestep(monkeypatch):
    """All objects of a long record in its last timestep: the buffer must fit the peak, not the mean."""
    n_time, n_objects = 1096, 101
    widths = []
    original = xr.apply_ufunc

    def spy(*args, **kwargs):
        sizes = (kwargs.get("dask_gufunc_kwargs") or {}).get("output_sizes", {})
        if "out_id" in sizes:
            widths.append(sizes["out_id"])
        return original(*args, **kwargs)

    monkeypatch.setattr(xr, "apply_ufunc", spy)
    lat, lon, cell_area = _geometry()
    last = np.zeros(N_GEOMETRY_CELLS, dtype=np.int32)
    last[: 2 * n_objects] = np.repeat(np.arange(1, n_objects + 1, dtype=np.int32), 2)
    ids = np.zeros((n_time, N_GEOMETRY_CELLS), dtype=np.int32)
    ids[-1] = last
    field = xr.DataArray(da.from_array(ids, chunks=(25, N_GEOMETRY_CELLS)), dims=(TIMEDIM, XDIM))
    props = _unstructured_props(field, lat, lon, cell_area).compute()

    # Reference: the same busy timestep after one empty one.
    ref_field = xr.DataArray(da.from_array(ids[-2:], chunks=(1, N_GEOMETRY_CELLS)), dims=(TIMEDIM, XDIM))
    ref = _unstructured_props(ref_field, lat, lon, cell_area).compute()

    assert props.sizes["ID"] == n_objects
    xr.testing.assert_identical(props, ref)
    # The per-timestep buffer is exactly the busiest timestep's object count, never a count that grows with
    # the record (e.g. the total number of objects).
    assert widths == [n_objects, n_objects], widths


# -- structured properties and label magnitude --------------------------------------------------------------
#
# The tracker's structured labels are offset by the running object count, so a late slice holds IDs in the
# millions while containing only a few objects. The property step must cost what the objects present cost,
# and return the same values whatever the labels are.


def _labelled_slice(offset):
    """Three objects on a periodic 20 x 40 grid; one wraps across the x edge."""
    ids = np.zeros((20, 40), dtype=np.int32)
    ids[2:6, 3:9] = 1 + offset
    ids[10:14, 0:3] = 2 + offset  # wraps: also on the right edge
    ids[10:14, 37:40] = 2 + offset
    ids[15:18, 20:30] = 3 + offset
    return ids


def _props_of(field):
    return calculate_object_properties(
        field,
        unstructured_grid=False,
        lat=None,
        lon=None,
        cell_area=None,
        timedim="time",
        regional_mode=False,
        ydim="lat",
        xdim="lon",
        properties=["area", "centroid"],
    )


def test_structured_properties_ignore_label_values():
    """A single slice with labels near 1e6 gives the same properties as with labels near 1; two time steps
    through the parallel path keep their original (offset) labels and identical per-object values."""
    failures = []

    def single_slice_offset():
        small = _props_of(xr.DataArray(_labelled_slice(0), dims=["lat", "lon"]))
        large = _props_of(xr.DataArray(_labelled_slice(1_000_000), dims=["lat", "lon"]))
        assert list(large.ID.values) == [1_000_001, 1_000_002, 1_000_003], list(large.ID.values)
        assert large.ID.dtype == small.ID.dtype
        xr.testing.assert_identical(small.drop_vars("ID"), large.drop_vars("ID"))

    def two_slices_parallel():
        field = np.stack([_labelled_slice(0), _labelled_slice(3)])
        props = _props_of(xr.DataArray(field, dims=["time", "lat", "lon"]).chunk({"time": 1}))
        assert list(props.ID.values) == [1, 2, 3, 4, 5, 6], list(props.ID.values)
        np.testing.assert_array_equal(props.area.values[:3], props.area.values[3:])
        np.testing.assert_array_equal(props.centroid.values[:, :3], props.centroid.values[:, 3:])

    check(failures, "single_slice_offset", single_slice_offset)
    check(failures, "two_slices_parallel", two_slices_parallel)
    assert_no_failures(failures)


def test_allocation_does_not_follow_label_magnitude():
    def peak(offset):
        field = xr.DataArray(_labelled_slice(offset), dims=["lat", "lon"])
        _props_of(field)  # first-call imports and caches are not the measurement
        tracemalloc.start()
        try:
            _props_of(field)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    small, large = peak(0), peak(2_000_000)
    assert large <= 2 * small + 256 * 1024, f"peak {small} B with labels near 1, {large} B with labels near 2e6"


def test_small_labels_cost_no_more_than_regionprops():
    """Compacting is not free: with labels already small it must not run at all."""
    from skimage.measure import regionprops_table

    ids = np.zeros((1200, 2400), dtype=np.int32)
    rng = np.random.default_rng(0)
    for k in range(200):
        y, x = rng.integers(0, 1180), rng.integers(0, 2380)
        ids[y : y + 15, x : x + 15] = k + 1

    def peak(fn):
        fn()
        tracemalloc.start()
        try:
            fn()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    ours = peak(lambda: _props_of(xr.DataArray(ids, dims=["lat", "lon"])))
    reference = peak(lambda: regionprops_table(ids, properties=["label", "area", "centroid", "bbox"]))
    # One compaction pass over this 2.9M-cell slice peaks ~17 MB above regionprops alone.
    assert ours <= reference + 4 * 1024 * 1024, f"peak {ours} B, regionprops alone {reference} B"


# -- float64 accumulation of unstructured areas ---------------------------------------------------------------
#
# A float32 running sum over a large object drifts with the object's size, so a merge decision near the
# overlap threshold could depend on how big the objects are rather than on their overlap. Every unstructured
# area -- serial overlaps, object properties, the merge kernel's split consolidation and the recomputed event
# properties -- follows one rule: accumulate in float64, round once to float32, and divide the rounded values
# in float64. The serial and parallel paths then make the same decisions and return the same areas.

ROUNDING_CELLS = 2_000_000


def _serial_consolidation(prev, cur, area, threshold):
    n = prev.size
    lat = xr.DataArray(np.zeros(n, dtype=np.float32), dims="ncells")
    lon = xr.DataArray(np.linspace(0.0, 1.0, n, dtype=np.float32), dims="ncells")
    cell_area = xr.DataArray(area, dims="ncells")
    kwargs = {
        "unstructured_grid": True,
        "lat": lat,
        "lon": lon,
        "cell_area": cell_area,
        "timedim": "time",
        "regional_mode": False,
        "ydim": None,
        "xdim": "ncells",
    }
    props = [
        _objects.calculate_object_properties(xr.DataArray(field, dims="ncells"), properties=["area", "centroid"], **kwargs)
        for field in (prev, cur)
    ]
    store = _objects.ObjectPropsStore.from_dataset(xr.concat(props, dim="ID"))
    out, _ = _overlap.consolidate_object_ids(
        xr.DataArray(prev, dims="ncells"),
        xr.DataArray(cur.copy(), dims="ncells"),
        store,
        1,
        True,
        cell_area,
        threshold,
        lat,
        lon,
        "time",
        False,
        None,
        "ncells",
    )
    return np.asarray(out.values)


CONSOLIDATION_ROWS = [
    # Equal areas: 1/3 in float64 is just below the float32-rounded threshold, so only child 3 (whole overlap)
    # is accepted and nothing is consolidated. A float32 ratio rounds 1/3 up to the threshold and wrongly
    # relabels child 3 as 2.
    ("float32_ratio", [1, 1, 1, 0, 0], [2, 3, 0, 2, 2], [1.0] * 5, 0.33333334, [2, 3, 0, 2, 2]),
    # Unequal areas: child 2's overlap fraction is 0.7904288999387088 under the shared rule, below the
    # threshold; float32 accumulation gives 0.790428933398726 and accepts it.
    (
        "float32_accumulation",
        [1, 1, 1, 1, 0],
        [2, 2, 2, 3, 2],
        [1.40057528, 1.18298054, 0.68794185, 0.86739153, 1.46195507],
        0.7904289166687174,
        [2, 2, 2, 3, 2],
    ),
    # Exactly on the threshold: 2 x 0.1 over 4 x 0.1 is exactly 0.5 after float32 rounding, and the test is
    # inclusive, so both children are accepted and merged.
    ("on_threshold", [1, 1, 1, 1, 0, 0, 0, 0], [2, 2, 3, 3, 2, 2, 3, 3], [0.1] * 8, 0.5, [2] * 8),
    # Just above it: neither child is accepted.
    (
        "above_threshold",
        [1, 1, 1, 1, 0, 0, 0, 0],
        [2, 2, 3, 3, 2, 2, 3, 3],
        [0.1] * 8,
        float(np.nextafter(0.5, 1.0)),
        [2, 2, 3, 3, 2, 2, 3, 3],
    ),
]


def test_parallel_consolidation_equals_serial():
    """The unstructured kernel's split consolidation makes the serial path's decisions. Each row is decided
    one way by the shared rule and the other way by some other arithmetic, or sits exactly on the threshold."""
    failures = []
    for case, prev, cur, area, threshold, expected in CONSOLIDATION_ROWS:

        def row(prev=prev, cur=cur, area=area, threshold=threshold, expected=expected):
            prev, cur = np.asarray(prev, dtype=np.int32), np.asarray(cur, dtype=np.int32)
            area = np.asarray(area, dtype=np.float32)
            serial = _serial_consolidation(prev, cur, area, threshold)
            parallel = cur.copy()
            merge_split._consolidate_slice(prev, parallel, area, threshold)
            np.testing.assert_array_equal(serial, expected, err_msg="serial")
            np.testing.assert_array_equal(parallel, expected, err_msg="parallel")

        check(failures, case, row)
    assert_no_failures(failures)


def test_areas_are_rounded_float64_sums(inprocess_client, tmp_path):
    """Overlap areas, object areas and the area the tracker returns for each event and timestep are all the
    float64 sum rounded once to float32.

    The tracked-event row spreads the cell areas over six decades so that a float32 sum visibly drifts from
    the rounded float64 one even on this small mesh (on the shipped areas it rarely shows).
    """
    assert on_client(inprocess_client), "not computing on the in-process client"
    area = np.random.default_rng(0).uniform(0.5, 1.5, ROUNDING_CELLS).astype(np.float32)
    rounded = np.float32(area.astype(np.float64).sum())
    failures = []

    def overlap_area():
        ids = np.ones(ROUNDING_CELLS, dtype=np.int32)
        result = check_overlap_slice(ids, ids, True, area)
        assert result.shape == (1, 3), result.shape
        assert np.float32(result[0, 2]) == rounded, (result[0, 2], rounded)

    def object_area():
        lat = xr.DataArray(np.linspace(-60, 60, ROUNDING_CELLS, dtype=np.float32), dims="ncells")
        lon = xr.DataArray(np.linspace(-170, 170, ROUNDING_CELLS, dtype=np.float32), dims="ncells")
        with dask.config.set(THREADED):
            props = calculate_object_properties(
                xr.DataArray(np.ones((1, ROUNDING_CELLS), dtype=np.int32), dims=("time", "ncells")).chunk(
                    {"time": 1, "ncells": -1}
                ),
                unstructured_grid=True,
                lat=lat,
                lon=lon,
                cell_area=xr.DataArray(area, dims="ncells"),
                timedim="time",
                regional_mode=False,
                ydim=None,
                xdim="ncells",
                properties=["area", "centroid"],
            )
            dtype, values = props.area.dtype, props.area.values.tolist()
        assert dtype == np.float32, dtype
        assert values == [rounded], (values, rounded)

    def tracked_event_area():
        data = open_unstructured_merging()
        spread = 10 ** np.random.default_rng(0).uniform(-3, 3, data.sizes["ncells"])
        cell_areas = (data.cell_areas.values.astype(np.float64) * spread).astype(np.float32)
        cell_areas = xr.DataArray(cell_areas, dims=data.cell_areas.dims, coords=data.cell_areas.coords)
        tr = marEx.tracker(
            data.extreme_events.chunk({"time": 4, "ncells": -1}),
            data.mask,
            neighbours=data.neighbours,
            cell_areas=cell_areas,
            temp_dir=str(tmp_path),
            **UNSTRUCTURED_MERGING_KWARGS,
        )
        events = tr.run().compute()
        del tr
        id_field = events.ID_field.transpose("time", "ncells").values
        event_area = events.area.transpose("time", "ID").values
        assert event_area.dtype == np.float32, event_area.dtype
        checked, wrong = 0, []
        for t in range(id_field.shape[0]):
            for j, event_id in enumerate(events.ID.values):
                cells = id_field[t] == event_id
                if not cells.any():
                    continue
                checked += 1
                expected = np.float32(cell_areas.values[cells].astype(np.float64).sum())
                if event_area[t, j] != expected:
                    wrong.append((t, int(event_id), float(event_area[t, j]), float(expected)))
        assert checked > 50, f"only {checked} event-timesteps were checked"
        assert not wrong, f"{len(wrong)} of {checked} event areas differ from the rounded float64 sum, e.g. {wrong[:3]}"

    check(failures, "overlap_area", overlap_area)
    check(failures, "object_area", object_area)
    check(failures, "tracked_event_area", tracked_event_area)
    assert_no_failures(failures)
