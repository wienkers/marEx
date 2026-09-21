"""``_materialise_dask_coords`` -- the fix for a graph-construction bug in the xarray/dask/
distributed stack where a dask-backed coordinate that does NOT vary with the animation's time
dimension (so the identical coordinate task is shared by every per-frame ``dask.delayed`` call)
gets submitted to the scheduler with an unresolvable dependency, cancelling every frame with
``FutureCancelledError: ... cancelled for reason: lost dependencies`` -- instantly, with no
worker error, independent of array scale or chunking (D-120).

Isolated with a marEx-independent probe (jobs 27595891, 27595929): the trigger is the
``.where()`` CONDITION being sourced from a coordinate -- an otherwise-identical condition
sourced from a plain dask-backed data variable never fails, even on an array that still
carries other, untouched dask coordinates. ``_animate`` cannot avoid the upstream bug, but it
can avoid the trigger: ``dask.persist()`` (NOT ``.compute()``) any non-time-varying coordinate
before the frame loop. Persisting resolves the coordinate under its own key, ahead of time, so
a frame's graph never has to re-traverse the fragile lineage back through the original
``.where()`` -- while, under a real distributed client, keeping it a small, by-reference dask
array rather than a numpy literal that ``dask.delayed`` would embed in every one of the N
per-frame task specs. That embedding is not hypothetical: a falsifier round caught an earlier
``.compute()``-based version of this fix doing exactly that, reintroducing D-122's separate
large-per-frame-graph crash at ICON scale.
"""

from pathlib import Path

import dask
import numpy as np
import pytest
import xarray as xr

from marEx.plotX.animation import _materialise_dask_coords


def _build_zarr_da(tmp_path, name, seed=0, n_time=4, n_cells=100):
    """A real zarr-backed DataArray with one dask coordinate that varies with time and one
    (``lat``) that does not -- the exact shape that reproduces the bug. Must come from a real
    zarr store: xarray's zarr backend is what constructs the vulnerable coordinate variables.
    ``name``/``seed`` pick a distinct store path and distinct values, so two calls produce
    independently-tokenised dask coordinates (persisting one must not accidentally persist
    the other via dask's content-addressed task naming).
    """
    rng = np.random.default_rng(seed)
    ds = xr.Dataset(
        {"ID": (("time", "ncells"), rng.integers(0, 5, size=(n_time, n_cells)).astype("int32"))},
        coords={
            "lat": ("ncells", rng.uniform(-90, 90, size=n_cells)),
            "time": np.datetime64("2020-01-01") + np.arange(n_time),
        },
    )
    path = Path(tmp_path) / f"{name}.zarr"
    ds.chunk({"time": 1, "ncells": -1}).to_zarr(path, mode="w")
    return xr.open_zarr(path)["ID"]


@pytest.fixture
def zarr_da(tmp_path):
    return _build_zarr_da(tmp_path, "store", seed=0)


def _is_persisted(dask_array):
    """A persisted dask array's graph holds exactly one task per chunk (the already-resolved
    result) and nothing else -- no residual lineage back to whatever produced it lazily."""
    return len(dict(dask_array.__dask_graph__())) == dask_array.npartitions


class TestMaterialiseDaskCoords:
    def test_a_non_time_varying_dask_coord_is_persisted_not_computed(self, zarr_da):
        """Must stay a dask array (not become numpy): computing it eagerly turns it into a
        literal that dask.delayed EMBEDS in every per-frame task spec, which at ICON scale
        reintroduces D-122's large-graph crash (falsifier-caught in an earlier attempt:
        measured 0.01 MB -> 4.80 MB per frame at 300k cells). Persisting severs the graph
        lineage back to the original `.where()` (what D-120's bug loses track of) while
        keeping it a small, by-reference dask array."""
        assert hasattr(zarr_da.lat.data, "dask")
        assert not _is_persisted(zarr_da.lat.data)
        out = _materialise_dask_coords(zarr_da, time_dim="time")
        assert hasattr(out.lat.data, "dask")
        assert _is_persisted(out.lat.data)

    def test_coordinate_values_are_unchanged(self, zarr_da):
        before = zarr_da.lat.values
        out = _materialise_dask_coords(zarr_da, time_dim="time")
        np.testing.assert_array_equal(out.lat.values, before)

    def test_the_data_variable_itself_stays_lazy(self, zarr_da):
        """Only coordinates are touched -- the field being animated must stay lazy, since
        streaming/lazy compute_mode outputs can be larger than memory."""
        out = _materialise_dask_coords(zarr_da, time_dim="time")
        assert hasattr(out.data, "dask")

    def test_a_time_varying_coordinate_is_left_alone(self, zarr_da):
        """A coordinate that DOES vary with time_dim is sliced per frame (never shared across
        delayed tasks), so it is not part of the failure mode and must not be force-computed."""
        import dask.array as dsa

        da = zarr_da.assign_coords(frame_label=("time", dsa.arange(zarr_da.sizes["time"])))
        out = _materialise_dask_coords(da, time_dim="time")
        assert hasattr(out.frame_label.data, "dask")

    def test_no_dask_backed_coords_is_a_noop(self):
        da = xr.DataArray(np.zeros((2, 3)), dims=("time", "x"), coords={"x": [1, 2, 3]})
        out = _materialise_dask_coords(da, time_dim="time")
        assert out is da

    def test_a_per_frame_slice_does_not_grow_with_coord_size(self, tmp_path, dask_client):
        """The regression this whole function exists to avoid a second time: a per-frame
        slice's pickled size must stay near the unfixed baseline, not scale with the
        coordinate's byte size (which is what `.compute()` did). Needs a coordinate large
        enough for the difference to show above pickling overhead -- a falsifier round
        measured 0.01 MB -> 4.80 MB at 300k cells; this uses the same scale.

        Needs `dask_client` (an ACTIVE distributed client): `dask.persist()` only keeps the
        result out of the pickled literal when there is a separate scheduler/worker to hold
        it, addressable by key. On the default local/synchronous scheduler (no client),
        persist has nowhere else to put the data and embeds it exactly like `.compute()`
        would -- confirmed empirically (this test failed with fixed_size ~2.4 MB before
        `dask_client` was added). `_animate` always runs under a real distributed client in
        practice (that is the entire reason it batches frames through `dask.delayed`), so
        this is the scenario that matters, not a gap in the fix."""
        import cloudpickle

        da = _build_zarr_da(tmp_path, "big", seed=2, n_time=2, n_cells=300_000)

        unfixed_slice = da.isel(time=0)
        baseline = len(cloudpickle.dumps(unfixed_slice))

        fixed = _materialise_dask_coords(da, time_dim="time")
        fixed_slice = fixed.isel(time=0)
        fixed_size = len(cloudpickle.dumps(fixed_slice))

        computed_slice = da.assign_coords(lat=da.lat.compute()).isel(time=0)
        computed_size = len(cloudpickle.dumps(computed_slice))

        assert fixed_size < 2 * baseline, f"persisted slice grew {fixed_size} bytes vs baseline {baseline}"
        assert computed_size > 10 * fixed_size, "the .compute()-based regression this guards against did not reproduce"


@pytest.mark.slow
class TestMaterialiseDaskCoordsPreventsLostDependencies:
    """The actual failure mode, reproduced end-to-end under a real distributed client."""

    def test_unfixed_coord_derived_where_loses_dependencies(self, zarr_da, dask_client):
        import dask
        from distributed.client import FutureCancelledError

        cut = zarr_da.where(zarr_da.lat < 45.0, 0)

        @dask.delayed
        def touch(slice_):
            return slice_.values.sum()

        tasks = [touch(cut.isel(time=i)) for i in range(cut.sizes["time"])]
        with pytest.raises(FutureCancelledError):
            dask.compute(*tasks)

    def test_materialising_coords_first_prevents_it(self, zarr_da, dask_client):
        import dask

        cut = zarr_da.where(zarr_da.lat < 45.0, 0)
        fixed = _materialise_dask_coords(cut, time_dim="time")

        @dask.delayed
        def touch(slice_):
            return slice_.values.sum()

        tasks = [touch(fixed.isel(time=i)) for i in range(fixed.sizes["time"])]
        results = dask.compute(*tasks)
        assert len(results) == fixed.sizes["time"]

    def test_a_second_where_cut_array_needs_its_own_materialising(self, zarr_da, dask_client, tmp_path):
        """``_animate`` passes ``object_ids``/``centroids`` as SEPARATE arrays, each isel'd per
        frame into its own delayed task -- materialising the main field does not protect them
        (falsifier finding: an unfixed object_ids field still crashed a run with a fixed main
        field). Each array touched by the frame loop needs its own call.

        ``object_ids`` is built from an INDEPENDENT zarr store (own path, own values), not
        from ``zarr_da`` itself: persisting a dask array registers its result on the cluster
        under that array's own content-addressed token, so reusing the identical `zarr_da`
        object for both arrays would let `object_ids` ride on `main_field`'s persist for free
        and the test would stop discriminating anything.
        """
        import dask
        from distributed.client import FutureCancelledError

        main_field = _materialise_dask_coords(zarr_da, time_dim="time")
        object_ids_da = _build_zarr_da(tmp_path, "object_ids", seed=1)
        object_ids = object_ids_da.where(object_ids_da.lat < 45.0, 0)  # independently .where()-cut

        @dask.delayed
        def touch(slice_):
            return slice_.values.sum()

        unfixed_tasks = [touch(object_ids.isel(time=i)) for i in range(object_ids.sizes["time"])]
        with pytest.raises(FutureCancelledError):
            dask.compute(*unfixed_tasks)

        object_ids_fixed = _materialise_dask_coords(object_ids, time_dim="time")
        fixed_tasks = [touch(object_ids_fixed.isel(time=i)) for i in range(object_ids_fixed.sizes["time"])]
        results = dask.compute(*fixed_tasks)
        assert len(results) == object_ids_fixed.sizes["time"]
        assert main_field.sizes["time"] == object_ids_fixed.sizes["time"]


@pytest.mark.slow
class TestAnimateExecutesWithMaterialisedCoords:
    """Every test above exercises ``_materialise_dask_coords`` directly; none of them RUNS
    ``_animate`` itself, so the ``plotter.da`` -> ``da`` call-site rewiring, and the
    ``centroids``/``object_ids`` wiring, were checked by reading and grep, not by execution
    (flagged twice by falsifier review). This is the one test that actually calls
    ``_animate``, with ffmpeg/make_frame/subprocess stubbed so it stays fast and
    deterministic while still building and computing the real per-frame dask graph -- the
    exact region where the object_ids wiring bug lived and was found only by an adversarial
    probe, not by reading the code.
    """

    @staticmethod
    def _stub_animate_deps(monkeypatch):
        from marEx.plotX import animation as animation_mod

        monkeypatch.setattr(animation_mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")
        monkeypatch.setattr(animation_mod.subprocess, "run", lambda *a, **k: None)

        @dask.delayed
        def fake_make_frame(data_slice, time_ind, temp_dir, plot_params, grid_info):
            data_slice.values  # the exact touch that triggers the upstream bug in make_frame
            filename = f"time_{time_ind:04d}.jpg"
            (Path(temp_dir) / filename).write_bytes(b"\x00")
            # NOTE: cannot record `time_ind` into a shared Python list here -- these tasks may
            # run in separate WORKER PROCESSES (dask_client can use processes=True), whose
            # memory is not visible back in the test process. `dask.compute()` not raising,
            # and `_animate` returning non-None, is itself the proof every frame computed
            # (an exception from any one delayed task propagates out of `_animate`
            # uncaught -- there is no except clause around the `dask.compute` call).
            return filename

        monkeypatch.setattr(animation_mod, "make_frame", fake_make_frame)
        return animation_mod

    @staticmethod
    def _stub_plotter(da):
        class StubPlotter:
            def __init__(self, da):
                self.da = da
                self.dimensions = {"time": "time"}
                self.coordinates = {"time": "time", "y": "lat", "x": "lon"}

            def _setup_common_params(self, config):
                return "viridis", None, None, "", "both"

        return StubPlotter(da)

    def test_animate_runs_with_a_where_cut_main_field_and_object_ids(self, zarr_da, dask_client, monkeypatch, tmp_path):
        """The current, fixed code: must succeed."""
        from marEx.plotX.config import PlotConfig

        animation_mod = self._stub_animate_deps(monkeypatch)

        main_field = zarr_da.where(zarr_da.lat < 45.0, 0)
        object_ids = zarr_da.where(zarr_da.lat < 45.0, 0)  # separate array, its own where()-cut coord
        plotter = self._stub_plotter(main_field)

        result = animation_mod._animate(plotter, PlotConfig(), plot_dir=str(tmp_path), object_ids=object_ids)

        assert result is not None
        assert result.endswith(".mp4")

    def test_animate_fails_without_materialising_coords(self, zarr_da, dask_client, monkeypatch, tmp_path):
        """Discriminator: with materialisation stubbed to a no-op (simulating the pre-fix
        code), the same call must fail with the upstream error -- proving this test would
        have caught the bug, not just exercised the happy path."""
        from distributed.client import FutureCancelledError

        from marEx.plotX.config import PlotConfig

        animation_mod = self._stub_animate_deps(monkeypatch)
        monkeypatch.setattr(animation_mod, "_materialise_dask_coords", lambda da, time_dim: da)

        main_field = zarr_da.where(zarr_da.lat < 45.0, 0)
        object_ids = zarr_da.where(zarr_da.lat < 45.0, 0)
        plotter = self._stub_plotter(main_field)

        with pytest.raises(FutureCancelledError):
            animation_mod._animate(plotter, PlotConfig(), plot_dir=str(tmp_path), object_ids=object_ids)

    def test_animate_survives_multiple_batches(self, dask_client, monkeypatch, tmp_path):
        """`_animate` batches frames (`_FRAME_BATCH_SIZE`, one `dask.compute()` call per
        batch): every test above uses few enough frames for a SINGLE batch. A persisted
        coordinate is scheduler-held data referenced by key, not an inert literal like a
        computed one -- its survival across a SECOND, separate `dask.compute()` submission
        is exactly the kind of thing this bug class is about (the scheduler losing track of
        something it shouldn't). Shrink the batch size so 6 frames span 3 batches."""
        from marEx.plotX.config import PlotConfig

        animation_mod = self._stub_animate_deps(monkeypatch)
        monkeypatch.setattr(animation_mod, "_FRAME_BATCH_SIZE", 2)

        da = _build_zarr_da(tmp_path, "multibatch", seed=3, n_time=6)
        main_field = da.where(da.lat < 45.0, 0)
        object_ids = da.where(da.lat < 45.0, 0)
        plotter = self._stub_plotter(main_field)

        result = animation_mod._animate(plotter, PlotConfig(), plot_dir=str(tmp_path), object_ids=object_ids)

        assert result is not None
        assert result.endswith(".mp4")
