"""Graph-structure guard for the default (`persist`) tracker path.

Values are guarded by tests/test_track_golden.py. This file guards what values are
blind to: the SHAPE of the dask graph. Phase 2 shipped an all-to-all rechunk
regression while 440 value-based tests, the window harness and the coverage
tripwires were all green; only the full-scale run caught it. Phase 4 changes the
materialisation strategy, which is exactly the kind of change that perturbs graph
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
        signature of chunk boundaries being shredded (the Phase 2 regression).
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
