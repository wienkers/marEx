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
