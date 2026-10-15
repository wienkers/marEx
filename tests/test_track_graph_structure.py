"""Graph-structure guard for the default (`persist`) tracker path.

Values are guarded by tests/test_track_golden.py. This file guards what values are
blind to: the SHAPE of the dask graph. An all-to-all rechunk regression once shipped
while 440 value-based tests, the window harness and the coverage
tripwires were all green; only the full-scale run caught it. Changing the
materialisation strategy is exactly the kind of change that perturbs graph
shape without moving a single output value.

The absolute numbers here are not meaningful in themselves -- they are a tripwire.
If a change moves them, that change must be understood and the numbers updated
DELIBERATELY, with the reason recorded in the commit message. Never update them
just to make the test pass.
"""

from pathlib import Path

import pytest
import xarray as xr

import marEx

TEST_DATA_DIR = Path(__file__).parent / "data"

CHUNK_SIZE = {"time": 2, "lat": -1, "lon": -1}

TRACKER_KWARGS = {
    "R_fill": 8,
    "area_filter_quartile": 0.5,
    "T_fill": 2,
    "allow_merging": True,
    "overlap_threshold": 0.5,
    "nn_partitioning": True,
    "quiet": True,
}

# Measured on unmodified code at commit cbf21ff. Margin of +2 added to each measured
# value. (n_tasks was 678 at the same measurement; not asserted here, but useful as a
# reference point if these bounds ever need to be revisited.)
EXPECTED_MAX_RECHUNK = 466  # measured 464 at cbf21ff, +2 margin
EXPECTED_MAX_TRANSPOSE = 50  # measured 48 at cbf21ff, +2 margin


def _graph_stats(dataset):
    """Task count and rechunk-key count of a dataset's combined graph."""
    graph = dataset.__dask_graph__()
    keys = [str(k[0]) if isinstance(k, tuple) else str(k) for k in graph.keys()]
    return {
        "n_tasks": len(keys),
        "n_rechunk": sum(1 for k in keys if "rechunk" in k or "shuffle" in k),
        "n_transpose": sum(1 for k in keys if "transpose" in k),
    }


class TestTrackGraphStructure:
    """The default persist-mode graph must not change shape."""

    @classmethod
    def setup_class(cls):
        cls.ds = xr.open_zarr(str(TEST_DATA_DIR / "extremes_gridded.zarr"), chunks={})

    def _run(self):
        data_bin = self.ds.extreme_events.chunk(CHUNK_SIZE)
        tracker = marEx.tracker(data_bin, self.ds.mask, **TRACKER_KWARGS)
        return tracker.run()

    @pytest.mark.slow
    def test_default_mode_graph_shape_is_stable(self, dask_client):
        """Record/compare the persist-mode graph shape of the returned dataset.

        A rise in n_rechunk is the specific failure this exists to catch: it is the
        signature of chunk boundaries being shredded (the earlier regression).
        """
        events_ds = self._run()
        stats = _graph_stats(events_ds)

        # Recorded on unmodified code at commit cbf21ff. See module docstring before
        # changing these.
        assert stats["n_rechunk"] <= EXPECTED_MAX_RECHUNK, (
            f"rechunk key count rose to {stats['n_rechunk']} (limit {EXPECTED_MAX_RECHUNK}). "
            "This is the signature of shredded chunk boundaries -- investigate before "
            "updating this bound."
        )
        assert (
            stats["n_transpose"] <= EXPECTED_MAX_TRANSPOSE
        ), f"transpose key count rose to {stats['n_transpose']} (limit {EXPECTED_MAX_TRANSPOSE})"


class _StopAtPersist(Exception):
    def __init__(self, obj):
        super().__init__("graph captured")
        self.obj = obj


def _overlap_graph(n_cells, monkeypatch, n_time=8, time_chunk=2):
    """Build (but never run) the unstructured overlap graph; return its task specs and the area bytes."""
    import dask.array as dsa
    import numpy as np

    from marEx.track import overlap as _overlap

    ids = xr.DataArray(
        dsa.zeros((n_time, n_cells), dtype=np.int32, chunks=(time_chunk, n_cells)),
        dims=("time", "ncells"),
    )
    cell_area = xr.DataArray(np.linspace(1.0, 2.0, n_cells, dtype=np.float32), dims=("ncells",))

    def stop(self, *args, **kwargs):
        raise _StopAtPersist(self)

    monkeypatch.setattr(xr.DataArray, "persist", stop)
    with pytest.raises(_StopAtPersist) as captured:
        _overlap.find_overlapping_objects(ids, "time", True, None, "ncells", cell_area)
    monkeypatch.undo()
    return dict(captured.value.obj.__dask_graph__()), cell_area.nbytes


class TestOverlapAreaPayload:
    """The per-cell area is shipped once as a graph dependency, not inside every overlap task."""

    def test_area_is_not_embedded_per_task(self, monkeypatch):
        import pickle

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

    def test_area_matches_cells_by_position(self):
        """A coordinate on the area's cell dimension is not used to align it to the field."""
        import numpy as np

        from marEx.track import overlap as _overlap

        ids = np.array([[1, 1, 0, 2], [1, 1, 0, 2], [0, 0, 0, 0]], dtype=np.int32)
        field = xr.DataArray(ids, dims=("time", "ncells"), coords={"ncells": [0, 1, 2, 3]}).chunk({"time": 1})
        area = xr.DataArray(np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32), dims=("ncells",), coords={"ncells": [3, 2, 1, 0]})
        result = _overlap.find_overlapping_objects(field, "time", True, None, "ncells", area)
        np.testing.assert_array_equal(result, np.array([[1, 1, 3.0], [2, 2, 8.0]], dtype=np.float32))


def _lifetime_field(n_time, n_cells=60):
    """IDs 2t+1 and 2t+2 are born at t and live three steps; returns the field and the children queried."""
    import numpy as np

    field = np.zeros((n_time, n_cells), dtype=np.int32)
    for t in range(n_time):
        for k, obj in enumerate((2 * t + 1, 2 * t + 2)):
            cells = slice((2 * (t % 5) + k) * 6, (2 * (t % 5) + k) * 6 + 6)
            field[t : t + 3, cells] = obj
    children = [2 * t + 1 for t in range(n_time)]
    return field, children


def _id_time_work(n_time, monkeypatch, time_chunk=5):
    """Per source block: bytes of every element-wise comparison built, and tasks run, while looking
    up each child's first time. Blockwise fusion hides a comparison inside one task, so the
    comparison is measured where it is built, not from task results."""
    import dask
    from dask.callbacks import Callback

    from marEx.track import overlap as _overlap

    field, children = _lifetime_field(n_time)
    da = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": time_chunk, "ncells": -1})
    compared, n_tasks = [], [0]
    original_eq = xr.DataArray.__eq__

    def eq_spy(self, other):
        out = original_eq(self, other)
        compared.append(out.size * out.dtype.itemsize)
        return out

    def count(key, result, dsk, state, id):
        n_tasks[0] += 1

    monkeypatch.setattr(xr.DataArray, "__eq__", eq_spy)
    with dask.config.set(scheduler="sync"), Callback(posttask=count):
        mapping = _overlap.compute_id_time_dict(da, children, 2 * n_time + 1, "time", True, None, "ncells", all_objects=False)
    monkeypatch.undo()

    n_source_blocks = n_time // time_chunk
    return sum(compared) / n_source_blocks, n_tasks[0] / n_source_blocks, mapping


class TestChildTimeLookup:
    """The first-time lookup costs the same per source block whatever the record length and query size."""

    def test_work_per_block_does_not_grow_with_record_and_children(self, monkeypatch):
        bytes_short, tasks_short, map_short = _id_time_work(20, monkeypatch)
        bytes_long, tasks_long, map_long = _id_time_work(40, monkeypatch)
        # Twice the record AND twice the queried children: a (time x buffer x children)
        # comparison grows 4x in total, 2x per source block, however it is chunked.
        assert bytes_long <= 1.25 * bytes_short + 1024, f"{bytes_short:.0f} -> {bytes_long:.0f} compared B per source block"
        assert tasks_long <= 1.25 * tasks_short, f"{tasks_short:.1f} -> {tasks_long:.1f} tasks per source block"
        assert map_short == {2 * t + 1: t for t in range(20)}
        assert map_long == {2 * t + 1: t for t in range(40)}

    @pytest.mark.parametrize("time_chunk", [1, 3, 7])
    @pytest.mark.parametrize("all_objects", [False, True])
    def test_first_time_matches_oracle(self, time_chunk, all_objects):
        import numpy as np

        from marEx.track import overlap as _overlap

        rng = np.random.default_rng(time_chunk)
        field = rng.integers(0, 30, size=(17, 40)).astype(np.int32)
        field[field < 12] = 0
        field[:, 0] = 0  # every slice has background
        max_objects = 35  # IDs 30..34 never occur: absent IDs map to 0
        children = [33, 5, 29, 12, 31, 18]
        da = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": time_chunk, "ncells": -1})
        mapping = _overlap.compute_id_time_dict(da, children, max_objects, "time", True, None, "ncells", all_objects=all_objects)

        queried = range(max_objects) if all_objects else children
        oracle = {}
        for obj in queried:
            present = np.nonzero((field == obj).any(axis=1))[0]
            oracle[int(obj)] = int(present[0]) if present.size else 0
        assert mapping == oracle

    @pytest.mark.parametrize("n_first", [100, 1], ids=["first-slice-full-buffer", "first-slice-padded"])
    def test_background_id_follows_the_padded_slice_rule(self, n_first):
        """ID 0 is 'present' where a slice has background or fewer distinct IDs than the old buffer."""
        import numpy as np

        from marEx.track import overlap as _overlap

        n_time = max_objects = 101  # buffer = int(101 / 101 * 100) = 100 distinct IDs per slice
        field = np.zeros((n_time, 100), dtype=np.int32)
        field[0] = np.arange(1, 101) if n_first == 100 else 1
        da = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": 10, "ncells": -1})
        mapping = _overlap.compute_id_time_dict(da, [], max_objects, "time", True, None, "ncells", all_objects=True)
        assert mapping[0] == (1 if n_first == 100 else 0)
        assert mapping[1] == 0 and mapping[100] == 0

    def test_lookup_graph_does_not_depend_on_the_query(self, monkeypatch):
        """The per-block work is the same whichever children are queried: the query never enters it."""
        import dask

        from marEx.track import overlap as _overlap

        field, children = _lifetime_field(20)
        da = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": 5, "ncells": -1})
        graphs = []
        original = dask.compute

        def spy(*args, **kwargs):
            graphs.append(sorted(str(k) for a in args if dask.is_dask_collection(a) for k in dict(a.__dask_graph__())))
            return original(*args, **kwargs)

        monkeypatch.setattr(dask, "compute", spy)
        for query in (children[:3], children):
            _overlap.compute_id_time_dict(da, query, 41, "time", True, None, "ncells", all_objects=False)
        monkeypatch.undo()
        assert len(graphs) == 2 and graphs[0] and graphs[0] == graphs[1]

    def test_split_space_is_rejected(self):
        """Background and buffer membership are whole-slice properties, so a split slice is refused."""
        import numpy as np

        from marEx.track import overlap as _overlap

        field = np.zeros((4, 6), dtype=np.int32)
        field[:, :3] = 1
        da = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": 2, "ncells": 3})
        with pytest.raises(ValueError):
            _overlap.compute_id_time_dict(da, [1], 2, "time", True, None, "ncells", all_objects=False)

    def test_memory_does_not_depend_on_the_query(self, monkeypatch):
        """Peak allocation inside each block does not change with the number of queried children, and
        the extra peak of a large query over a small one does not grow with the record."""
        import gc
        import tracemalloc

        import dask
        import numpy as np

        from marEx.track import overlap as _overlap

        block_peaks = []
        original = _overlap._first_time_in_block

        def measured(block, t0, buffer):
            tracemalloc.reset_peak()
            start = tracemalloc.get_traced_memory()[0]
            out = original(block, t0, buffer)
            block_peaks.append(tracemalloc.get_traced_memory()[1] - start)
            return out

        monkeypatch.setattr(_overlap, "_first_time_in_block", measured)

        def run(n_time, n_children):
            field, _ = _lifetime_field(n_time)
            da = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": 5, "ncells": -1})
            query = list(range(1, n_children + 1))
            block_peaks.clear()
            gc.collect()
            tracemalloc.start()
            try:
                with dask.config.set(scheduler="sync"):
                    _overlap.compute_id_time_dict(da, query, 2 * n_time + 1, "time", True, None, "ncells", all_objects=False)
                whole = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            return list(block_peaks), whole

        run(20, 2)  # first-call allocations (lazy imports, caches) are not the lookup's
        # tracemalloc also sees other threads, so one block or one run can carry a stray
        # allocation: compare the median block and the smallest of three runs.
        extra = {}
        for n_time in (20, 40):
            few = [run(n_time, 2) for _ in range(3)]
            many = [run(n_time, 2000) for _ in range(3)]
            assert all(len(blocks) == n_time // 5 for blocks, _ in few + many)
            median_few = float(np.median(few[0][0]))
            median_many = float(np.median(many[0][0]))
            # A (time x buffer x children) array in a block adds ~1 kB per queried child here.
            assert abs(median_many - median_few) <= 16 * 2000, (few[0][0], many[0][0])
            extra[n_time] = min(w for _, w in many) - min(w for _, w in few)
        # The lookup itself is O(queried children); a (time x buffer x children) array doubles here.
        assert extra[40] <= 1.25 * extra[20] + 256 * 2000, extra
