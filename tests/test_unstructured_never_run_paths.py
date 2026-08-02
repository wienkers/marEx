"""Two bugs that survived because their code paths had never been executed.

Both were found by executing `examples/unstructured data/02_id_track_events.ipynb`, which had
never been run: the notebook is committed with `execution_count: null` on every cell. Neither
is a Phase-2 regression -- `partition_centroid_unstructured` is byte-identical at `ccade8e`.

1. `partition_centroid_unstructured` allocated one label per *mesh* cell but the caller does
   `data_t[child_mask] = new_labels`, so the assignment raised ValueError whenever the child
   was smaller than the mesh -- i.e. always. Its own docstring and its sibling
   `partition_nn_unstructured` both promise one label per cell *in child_mask*.
2. `tracker.run()`'s single-use guard fired unconditionally when `data_bin` was None, which
   made the documented two-cluster pattern impossible:
   `run_preprocess(checkpoint="save")` -> close cluster -> `run(checkpoint="load")`.
   The load path returns straight from the zarr store and never reads `data_bin`.
3. `tracker.__init__` `persist()`-ed `lat`/`lon`/`lat_init`/`lon_init`/`cell_area`, binding
   them to whichever client was active at construction and replacing their graphs with
   futures. Closing that client -- what the two-cluster pattern does -- orphaned them, and
   tracking died with `FutureCancelledError: ... lost dependencies`. Invisible on gridded
   data, where those coords are small numpy arrays and `persist()` is a no-op.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import TrackingError
from marEx.track.partitioning import partition_centroid_unstructured


class TestPartitionCentroidUnstructured:
    """The return-length contract, which the caller depends on."""

    @staticmethod
    def _mesh():
        # 10-cell mesh; the child is 4 cells, split north/south around two parent centroids.
        n = 10
        child_mask = np.zeros(n, dtype=bool)
        child_mask[[2, 3, 7, 8]] = True
        lat = np.array([0, 0, 10, 11, 0, 0, 0, -10, -11, 0], dtype=np.float32)
        lon = np.zeros(n, dtype=np.float32)
        parent_centroids = np.array([[10.5, 0.0], [-10.5, 0.0]], dtype=np.float64)
        child_ids = np.array([101, 202], dtype=np.int32)
        return child_mask, parent_centroids, child_ids, lat, lon

    def test_returns_one_label_per_child_cell(self):
        """The bug: it returned one per mesh cell (14.9 M vs 201 k on the real mesh)."""
        child_mask, cent, ids, lat, lon = self._mesh()
        out = partition_centroid_unstructured(child_mask, cent, ids, lat, lon)
        assert len(out) == int(child_mask.sum()) == 4

    def test_assigns_each_child_cell_to_nearest_parent(self):
        child_mask, cent, ids, lat, lon = self._mesh()
        out = partition_centroid_unstructured(child_mask, cent, ids, lat, lon)
        np.testing.assert_array_equal(out, np.array([101, 101, 202, 202], dtype=np.int32))

    def test_result_is_assignable_the_way_the_caller_assigns_it(self):
        """`data_t[child_mask] = new_labels` in merge_split.py is the exact failure site."""
        child_mask, cent, ids, lat, lon = self._mesh()
        out = partition_centroid_unstructured(child_mask, cent, ids, lat, lon)
        data_t = np.zeros(len(child_mask), dtype=np.int32)
        data_t[child_mask] = out  # raised ValueError before the fix
        np.testing.assert_array_equal(data_t, [0, 0, 101, 101, 0, 0, 0, 202, 202, 0])

    def test_dtype_follows_child_ids(self):
        child_mask, cent, ids, lat, lon = self._mesh()
        out = partition_centroid_unstructured(child_mask, cent, ids, lat, lon)
        assert out.dtype == ids.dtype

    def test_whole_mesh_is_the_child(self):
        """Degenerate case: the old code only 'worked' when child_mask was all-True."""
        n = 6
        child_mask = np.ones(n, dtype=bool)
        lat = np.linspace(-20, 20, n).astype(np.float32)
        lon = np.zeros(n, dtype=np.float32)
        cent = np.array([[20.0, 0.0], [-20.0, 0.0]], dtype=np.float64)
        ids = np.array([7, 9], dtype=np.int32)
        out = partition_centroid_unstructured(child_mask, cent, ids, lat, lon)
        assert len(out) == n
        assert out[0] == 9 and out[-1] == 7  # southmost -> south parent, northmost -> north

    def test_single_parent_takes_everything(self):
        child_mask, _, _, lat, lon = self._mesh()
        cent = np.array([[0.0, 0.0]], dtype=np.float64)
        ids = np.array([42], dtype=np.int32)
        out = partition_centroid_unstructured(child_mask, cent, ids, lat, lon)
        assert set(np.unique(out)) == {42}
        assert len(out) == int(child_mask.sum())


class TestCheckpointLoadBypassesSingleUseGuard:
    """run(checkpoint='load') must not be rejected as 'already been run'."""

    @classmethod
    def setup_class(cls):
        path = Path(__file__).parent / "data" / "extremes_gridded.zarr"
        cls.extremes = xr.open_zarr(str(path), chunks={}).persist()

    def _tracker(self, **kw):
        return marEx.tracker(
            self.extremes.extreme_events,
            self.extremes.mask,
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=0,
            allow_merging=False,
            quiet=True,
            **kw,
        )

    def test_guard_still_fires_without_checkpoint(self):
        """The single-use protection itself must be preserved."""
        trk = self._tracker()
        trk.data_bin = None  # what run_preprocess() does after hole filling
        with pytest.raises(TrackingError, match="already been run"):
            trk.run()

    def test_guard_does_not_fire_for_checkpoint_load(self, tmp_path):
        """The regression: this raised 'already been run' instead of proceeding."""
        trk = self._tracker(temp_dir=str(tmp_path), checkpoint="save")
        trk.data_bin = None
        # No checkpoint files exist, so it must fail *later* (on the missing store) --
        # the point is that it gets past the guard rather than being rejected up front.
        with pytest.raises(Exception) as excinfo:
            trk.run(checkpoint="load")
        assert "already been run" not in str(excinfo.value)

    def test_instance_level_checkpoint_load_also_bypasses(self, tmp_path):
        """checkpoint may come from the constructor rather than the run() argument."""
        trk = self._tracker(temp_dir=str(tmp_path), checkpoint="load")
        trk.data_bin = None
        with pytest.raises(Exception) as excinfo:
            trk.run()
        assert "already been run" not in str(excinfo.value)


class TestTwoClusterPatternUnstructured:
    """The documented pattern: preprocess on one cluster, track on another.

    Unstructured only. On a gridded store `lat`/`lon` are small numpy coords, so the
    `persist()` that caused this was a silent no-op and the same test passes either way --
    it cannot discriminate. Here they are genuinely dask-backed (405 cells in the fixture,
    14 886 338 on ICON R02B09), which is what binds them to a client.
    """

    @pytest.mark.slow
    def test_run_checkpoint_load_survives_a_cluster_restart(self):
        from distributed import Client, LocalCluster

        data = xr.open_zarr(
            str(Path(__file__).parent / "data" / "extremes_unstructured.zarr"),
            chunks={"time": 2, "ncells": -1},
        )
        # The condition under test: these must really be dask-backed, or the test is vacuous.
        assert data.cell_areas.chunks is not None
        assert data.lat.chunks is not None

        tmp = tempfile.mkdtemp(prefix="marex_twocluster_")
        cluster_one = Client(LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="3GB", processes=True))
        try:
            trk = marEx.tracker(
                data.extreme_events,
                data.mask,
                R_fill=2,
                area_filter_quartile=0.1,
                T_fill=0,
                allow_merging=True,
                unstructured_grid=True,
                dimensions={"x": "ncells"},
                coordinates={"x": "lon", "y": "lat"},
                regional_mode=False,
                coordinate_units="degrees",
                quiet=True,
                neighbours=data.neighbours,
                cell_areas=data.cell_areas,
                temp_dir=tmp,
            )
            trk.run_preprocess(checkpoint="save")
        finally:
            cluster_one.close()

        cluster_two = Client(LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="3GB", processes=True))
        try:
            # Raised FutureCancelledError("... lost dependencies") before the fix.
            events = trk.run(checkpoint="load")
            assert int(events.ID.size) > 0
        finally:
            cluster_two.close()

    def test_coordinate_state_is_not_left_as_dask_futures(self):
        """Cheap structural guard for the same bug, with no cluster restart.

        If these come back dask-backed, they are again bound to a client and the
        two-cluster pattern is broken -- whether or not the slow test above ran.
        """
        from dask import is_dask_collection

        data = xr.open_zarr(
            str(Path(__file__).parent / "data" / "extremes_unstructured.zarr"),
            chunks={"time": 2, "ncells": -1},
        )
        tmp = tempfile.mkdtemp(prefix="marex_coordstate_")
        trk = marEx.tracker(
            data.extreme_events,
            data.mask,
            R_fill=2,
            area_filter_quartile=0.1,
            T_fill=0,
            allow_merging=False,
            unstructured_grid=True,
            dimensions={"x": "ncells"},
            coordinates={"x": "lon", "y": "lat"},
            regional_mode=False,
            coordinate_units="degrees",
            quiet=True,
            neighbours=data.neighbours,
            cell_areas=data.cell_areas,
            temp_dir=tmp,
        )
        for attr in ("lat", "lon", "lat_init", "lon_init", "cell_area"):
            value = getattr(trk, attr, None)
            if value is None:
                continue
            assert not is_dask_collection(getattr(value, "data", None)), f"{attr} is still dask-backed"
