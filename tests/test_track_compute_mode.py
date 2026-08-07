"""compute_mode on the tracker: validation, wiring, equivalence and laziness."""

from pathlib import Path

import pytest
import xarray as xr

import marEx
from marEx.detect.compute_mode import Materialiser
from marEx.exceptions import ConfigurationError

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


@pytest.fixture(scope="module")
def extremes():
    return xr.open_zarr(str(TEST_DATA_DIR / "extremes_gridded.zarr"), chunks={})


@pytest.fixture(scope="module")
def extremes_unstructured():
    return xr.open_zarr(str(TEST_DATA_DIR / "extremes_unstructured.zarr"), chunks={})


class TestMaterialiserIsStreaming:
    def test_is_streaming_true_only_for_streaming(self, tmp_path):
        assert Materialiser("streaming", tmp_path).is_streaming is True
        assert Materialiser("persist").is_streaming is False
        assert Materialiser("lazy").is_streaming is False


class TestTrackerComputeModeValidation:
    def test_default_is_persist(self, extremes):
        tr = marEx.tracker(extremes.extreme_events.chunk(CHUNK_SIZE), extremes.mask, **TRACKER_KWARGS)
        assert tr.compute_mode == "persist"
        assert tr.materialiser.is_streaming is False

    def test_lazy_is_rejected(self, extremes):
        with pytest.raises(ConfigurationError, match="lazy"):
            marEx.tracker(
                extremes.extreme_events.chunk(CHUNK_SIZE),
                extremes.mask,
                compute_mode="lazy",
                **TRACKER_KWARGS,
            )

    def test_unknown_mode_is_rejected(self, extremes):
        with pytest.raises(ConfigurationError):
            marEx.tracker(
                extremes.extreme_events.chunk(CHUNK_SIZE),
                extremes.mask,
                compute_mode="nonsense",
                **TRACKER_KWARGS,
            )

    def test_streaming_requires_temp_dir(self, extremes):
        with pytest.raises(ConfigurationError, match="temp_dir"):
            marEx.tracker(
                extremes.extreme_events.chunk(CHUNK_SIZE),
                extremes.mask,
                compute_mode="streaming",
                **TRACKER_KWARGS,
            )

    def test_streaming_with_temp_dir_builds_a_staging_dir(self, extremes, tmp_path):
        tr = marEx.tracker(
            extremes.extreme_events.chunk(CHUNK_SIZE),
            extremes.mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **TRACKER_KWARGS,
        )
        assert tr.materialiser.is_streaming is True
        assert tr.staging_dir is not None
        assert Path(tr.staging_dir).exists()

    def test_streaming_is_rejected_for_unstructured_grid(self, extremes_unstructured, tmp_path, dask_client_unstructured):
        """Streaming stages the shared preprocessing stages but not the unstructured
        merge/split core (its own separate zarr writer), so the combination is rejected
        outright rather than silently streaming only part of the pipeline."""
        with pytest.raises(ConfigurationError, match="unstructured"):
            marEx.tracker(
                extremes_unstructured.extreme_events.chunk({"time": 2, "ncells": -1}),
                extremes_unstructured.mask,
                R_fill=2,
                area_filter_quartile=0.5,
                compute_mode="streaming",
                temp_dir=str(tmp_path),
                unstructured_grid=True,
                dimensions={"x": "ncells"},
                coordinates={"x": "lon", "y": "lat"},
                coordinate_units="degrees",
                neighbours=extremes_unstructured.neighbours,
                cell_areas=extremes_unstructured.cell_areas,
                quiet=True,
            )

    def test_streaming_accepts_uniform_time_chunking_with_smaller_final_chunk(self, extremes, tmp_path):
        """`.chunk({"time": k})` always yields (k, k, ..., r) with r <= k -- this is the
        normal case for nearly every real input and must NOT be rejected."""
        data_bin = extremes.extreme_events.chunk({"time": 5, "lat": -1, "lon": -1})
        assert data_bin.sizes["time"] % 5 != 0, "fixture must not divide evenly, so a smaller final chunk exists"
        tr = marEx.tracker(
            data_bin,
            extremes.mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **TRACKER_KWARGS,
        )
        assert tr.materialiser.is_streaming is True

    def test_streaming_rejects_ragged_time_chunking(self, extremes, tmp_path):
        """A genuinely ragged chunking (e.g. from open_mfdataset over uneven per-year
        files) must be rejected at construction time, not fail inside the zarr write."""
        n_time = extremes.sizes["time"]
        ragged = tuple([3, 5] * ((n_time // 8) + 1))[: n_time // 8 * 2]
        remainder = n_time - sum(ragged)
        if remainder > 0:
            ragged = ragged + (remainder,)
        assert sum(ragged) == n_time
        assert len(ragged) > 1
        data_bin = extremes.extreme_events.chunk({"lat": -1, "lon": -1}).chunk({"time": ragged})

        with pytest.raises(ConfigurationError, match="uniform"):
            marEx.tracker(
                data_bin,
                extremes.mask,
                compute_mode="streaming",
                temp_dir=str(tmp_path),
                **TRACKER_KWARGS,
            )


class TestStagingLifetime:
    @pytest.mark.slow
    def test_streaming_output_advertises_its_staging_dir(self, extremes, tmp_path, dask_client):
        tr = marEx.tracker(
            extremes.extreme_events.chunk(CHUNK_SIZE),
            extremes.mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **TRACKER_KWARGS,
        )
        events = tr.run()
        staged = events.attrs.get("marex_staging_dir")
        assert staged is not None, "streaming output must advertise its staging dir"
        assert Path(staged).exists(), "staging dir must OUTLIVE run(); the result reads from it"
        # The result must still be readable -- this is the trap clear_staging-on-return causes.
        assert int(events.ID_field.max().compute()) >= 0
        marEx.clear_staging(events)
        assert not Path(staged).exists()


class TestCrossModeEquivalence:
    """streaming must be BIT-IDENTICAL to persist. Integer label fields; no tolerance."""

    @staticmethod
    def _run_both(data_bin, mask, tmp_path):
        """Run persist and streaming on the SAME chunking and return their computed outputs.

        Comparing at matching chunking matters: the gridded merge loop's end-of-chunk
        consolidation is boundary-dependent (that is the entire premise of
        ``preserve_chunks``), so comparing streaming-at-chunking-X against
        persist-at-chunking-Y could legitimately differ for reasons unrelated to the mode.
        """
        persist_tr = marEx.tracker(data_bin, mask, **TRACKER_KWARGS)
        persist_events, persist_merges = persist_tr.run(return_merges=True)
        persist_events = persist_events.compute()
        persist_merges = persist_merges.compute()

        stream_tr = marEx.tracker(
            data_bin,
            mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **TRACKER_KWARGS,
        )
        stream_events, stream_merges = stream_tr.run(return_merges=True)
        stream_events = stream_events.compute()
        stream_merges = stream_merges.compute()
        return persist_events, persist_merges, stream_events, stream_merges

    @pytest.mark.slow
    def test_streaming_matches_persist_exactly(self, extremes, tmp_path, dask_client):
        data_bin = extremes.extreme_events.chunk(CHUNK_SIZE)

        persist_tr = marEx.tracker(data_bin, extremes.mask, **TRACKER_KWARGS)
        persist_events, persist_merges = persist_tr.run(return_merges=True)
        persist_events = persist_events.compute()
        persist_merges = persist_merges.compute()

        stream_tr = marEx.tracker(
            data_bin,
            extremes.mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **TRACKER_KWARGS,
        )
        stream_events, stream_merges = stream_tr.run(return_merges=True)
        stream_events = stream_events.compute()
        stream_merges = stream_merges.compute()

        # assert_identical, not assert_allclose: no tolerance is granted in Phase 4.
        xr.testing.assert_identical(stream_events.drop_attrs(deep=False), persist_events.drop_attrs(deep=False))
        xr.testing.assert_identical(stream_merges, persist_merges)

    @pytest.mark.slow
    def test_streaming_matches_persist_odd_ntime(self, extremes, tmp_path, dask_client):
        """Odd-length series equivalence.

        The fixture is 32 steps (even), and every OTHER case in this module chunks time
        with a plain ``.chunk({"time": k})``. For any such call, ``chunks = (k,)*(n//k) +
        (n % k,)`` and ``max(chunks) == k`` whenever ``n >= k`` -- so re-chunking by that
        max reproduces the exact same tuple, regardless of whether ``n`` is odd or even.
        Verified directly: instrumenting ``Materialiser.stage`` at all four
        ``preserve_chunks=True`` call sites in ``marEx/track`` (data_bin_filled,
        data_bin_filtered, object_id_field, relabeled_id_field) with n_time=25 and
        n_time=31 under CHUNK_SIZE shows before/after chunks IDENTICAL at every site --
        the uniform-rechunk-to-max never moves a boundary here, so this case does NOT
        exercise the ``preserve_chunks`` guard (contrary to an earlier assumption).

        It still earns its keep as equivalence coverage for an odd-length series, which
        the always-even 32-step fixture otherwise never exercises.

        A genuinely ragged input chunking (e.g. ``(5, 3, 5, 3, 5, 3, 5, 3)``, which DOES
        move a boundary under the uniform-rechunk-to-max: verified it becomes
        ``(5, 5, 5, 5, 5, 5, 2)``) was tried as a way to make the guard load-bearing and
        testable. It does not reach the guard at all: ``ObjectIDRegionWriter._initialise``
        (``marEx/track/region_writer.py``) writes the *preserved* (ragged) chunking
        straight to zarr as the store's chunk grid, and zarr rejects interior ragged
        chunks outright -- ``ValueError: Zarr requires uniform chunk sizes except for
        final chunk`` -- independent of ``preserve_chunks``'s value. persist mode has no
        such constraint (no zarr write) and completes fine on the identical ragged input.
        So streaming mode cannot process a genuinely ragged time chunking at all, which
        means ``preserve_chunks=True`` at the four ``marEx/track`` call sites has no input
        under which it is reachable: inputs shaped like the ones streaming can actually
        run on never trigger it (this test and its siblings), and inputs that would
        trigger it crash streaming before the guard matters. Not fixed here -- this is a
        test-only task; see the task report for the full finding.
        """
        data_bin = extremes.extreme_events.isel(time=slice(0, 31)).chunk(CHUNK_SIZE)
        assert data_bin.sizes["time"] == 31

        persist_events, persist_merges, stream_events, stream_merges = self._run_both(data_bin, extremes.mask, tmp_path)
        xr.testing.assert_identical(stream_events.drop_attrs(deep=False), persist_events.drop_attrs(deep=False))
        xr.testing.assert_identical(stream_merges, persist_merges)

    @pytest.mark.slow
    def test_streaming_matches_persist_many_time_chunks(self, extremes, tmp_path, dask_client):
        """>= 21 time chunks: two consecutive periodic flushes in the merge loop.

        The merge loop flushes when ``chunk_idx % 10 == 0`` (and more than one chunk is
        retained). With time chunk width 1 over the 32-step fixture there are 32 chunks,
        so real flushes occur at chunk_idx 10 (writing chunks 0-9, retaining chunk 10) and
        20 (writing chunks 10-19, retaining chunk 20) -- chunk 10 is written by the LATER
        flush, not its own. With the default CHUNK_SIZE (16 chunks) only chunk_idx 0 occurs
        (and is a no-op, since a single retained chunk never satisfies
        ``len(updated_chunks) > 1``), so this path is otherwise untested.
        """
        chunk = {"time": 1, "lat": -1, "lon": -1}
        data_bin = extremes.extreme_events.chunk(chunk)
        assert len(data_bin.chunks[0]) >= 21

        persist_events, persist_merges, stream_events, stream_merges = self._run_both(data_bin, extremes.mask, tmp_path)
        xr.testing.assert_identical(stream_events.drop_attrs(deep=False), persist_events.drop_attrs(deep=False))
        xr.testing.assert_identical(stream_merges, persist_merges)

    @pytest.mark.slow
    def test_streaming_matches_persist_single_time_chunk(self, extremes, tmp_path, dask_client):
        """Single time chunk: finalise() is called with exactly one write."""
        chunk = {"time": -1, "lat": -1, "lon": -1}
        data_bin = extremes.extreme_events.chunk(chunk)
        assert len(data_bin.chunks[0]) == 1

        persist_events, persist_merges, stream_events, stream_merges = self._run_both(data_bin, extremes.mask, tmp_path)
        xr.testing.assert_identical(stream_events.drop_attrs(deep=False), persist_events.drop_attrs(deep=False))
        xr.testing.assert_identical(stream_merges, persist_merges)


class TestSharedLabellingPass:
    """The absolute-threshold filter labels every slice ONCE, in every mode.

    ``morphology.filter_small_objects`` derives the keep-mask and the area census from a
    single ``apply_ufunc`` and materialises both together so the labelling pass is shared.
    Under streaming that joint materialisation was a no-op ``pin``, so the census's
    ``.compute()`` ran the pass and the caller's ``stage`` of the filtered field ran it
    again -- a whole extra per-slice labelling of the entire field.
    """

    ABSOLUTE_KWARGS = {k: v for k, v in TRACKER_KWARGS.items() if k != "area_filter_quartile"}

    def _count_labels(self, monkeypatch, counter_path):
        """Count labelling calls THROUGH A FILE, not an in-memory list.

        The counting wrapper is captured by cloudpickle into the nested ufunc closure and
        shipped to the worker, so a list in its closure is pickled BY VALUE -- the worker
        appends to a copy and the client sees zero. Verified: the list version reported 0
        calls against 32 real slices. A path pickles fine and the file is shared, which
        works under threads and processes alike.
        """
        from marEx.track import morphology

        orig = morphology.scipy_label

        def counting(*args, **kwargs):
            with open(counter_path, "ab") as handle:
                handle.write(b"x")
            return orig(*args, **kwargs)

        monkeypatch.setattr(morphology, "scipy_label", counting)

    @staticmethod
    def _count(counter_path):
        return counter_path.stat().st_size if counter_path.exists() else 0

    @pytest.mark.slow
    def test_streaming_labels_no_more_slices_than_persist(self, extremes, tmp_path, dask_client, monkeypatch):
        data_bin = extremes.extreme_events.chunk(CHUNK_SIZE)
        kwargs = dict(self.ABSOLUTE_KWARGS, area_filter_absolute=400)

        counter = tmp_path / "labels.count"
        self._count_labels(monkeypatch, counter)
        marEx.tracker(data_bin, extremes.mask, **kwargs).run()
        persist_calls = self._count(counter)

        counter.unlink(missing_ok=True)
        marEx.tracker(
            data_bin,
            extremes.mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **kwargs,
        ).run()
        streaming_calls = self._count(counter)

        # Control: a patch that saw nothing would make the bound below vacuous.
        assert persist_calls >= data_bin.sizes["time"], (
            f"the counter saw only {persist_calls} labelling calls in persist mode, fewer "
            f"than the {data_bin.sizes['time']} slices in the fixture -- the monkeypatch is "
            f"not reaching the ufunc body, so the streaming bound proves nothing."
        )
        assert streaming_calls <= persist_calls, (
            f"streaming labelled {streaming_calls} slices against persist's {persist_calls}. "
            f"The joint materialisation of (keep-mask, area census) is not shared under "
            f"streaming, so the whole field is labelled twice."
        )

    @pytest.mark.slow
    def test_streaming_matches_persist_with_absolute_filter(self, extremes, tmp_path, dask_client):
        """Equivalence on the absolute-threshold path, which `stage_many` rewrote.

        Every other equivalence test here uses ``area_filter_quartile``, which routes
        through the two-pass branch and never reaches the joint materialisation.
        """
        data_bin = extremes.extreme_events.chunk(CHUNK_SIZE)
        kwargs = dict(self.ABSOLUTE_KWARGS, area_filter_absolute=400)

        persist_tr = marEx.tracker(data_bin, extremes.mask, **kwargs)
        persist_events, persist_merges = persist_tr.run(return_merges=True)
        persist_events, persist_merges = persist_events.compute(), persist_merges.compute()

        stream_tr = marEx.tracker(
            data_bin,
            extremes.mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **kwargs,
        )
        stream_events, stream_merges = stream_tr.run(return_merges=True)
        stream_events, stream_merges = stream_events.compute(), stream_merges.compute()

        # assert_identical, not assert_allclose: no tolerance is granted in Phase 4.
        xr.testing.assert_identical(stream_events.drop_attrs(deep=False), persist_events.drop_attrs(deep=False))
        xr.testing.assert_identical(stream_merges, persist_merges)


class TestNoMergingPath:
    """``allow_merging=False``: the ``elif time_connectivity:`` branch of objects.py.

    Every other test in this module runs with ``allow_merging=True``, which routes
    through ``split_and_merge``. With merging off the tracker instead calls
    ``identify_objects(time_connectivity=True)``, a 13th whole-field pin site that the
    original Phase 4 profile (job 26764480) missed and that no bit-identity gate covered.
    """

    def _run_both(self, data_bin, mask, tmp_path):
        kwargs = dict(TRACKER_KWARGS, allow_merging=False)
        persist_tr = marEx.tracker(data_bin, mask, **kwargs)
        persist_events = persist_tr.run().compute()

        stream_tr = marEx.tracker(
            data_bin,
            mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **kwargs,
        )
        stream_events = stream_tr.run().compute()
        return persist_events, stream_events

    @pytest.mark.slow
    def test_streaming_matches_persist_without_merging(self, extremes, tmp_path, dask_client):
        data_bin = extremes.extreme_events.chunk(CHUNK_SIZE)
        persist_events, stream_events = self._run_both(data_bin, extremes.mask, tmp_path)

        # assert_identical, not assert_allclose: no tolerance is granted in Phase 4.
        xr.testing.assert_identical(stream_events.drop_attrs(deep=False), persist_events.drop_attrs(deep=False))

        # N_events_final is an ATTR, so the comparison above drops it -- and it is the one
        # output this branch's fix can move: streaming derives the count as max(labels) off
        # the staged store instead of from dask_image.label's num_features. Verified those
        # agree (max == num_features == distinct-nonzero, with wrap merging active), but the
        # equivalence assertion above cannot see it, so compare it explicitly.
        assert stream_events.attrs["N_events_final"] == persist_events.attrs["N_events_final"], (
            f"event count differs by mode: streaming {stream_events.attrs['N_events_final']} vs "
            f"persist {persist_events.attrs['N_events_final']}. objects.py's streaming branch "
            f"reads N_objects as max(labels) off the staged field; if dask_image.label ever "
            f"stops numbering objects 1..N contiguously, pin its num_features instead."
        )

    @pytest.mark.slow
    def test_streaming_without_merging_pins_far_less_than_persist(self, extremes, tmp_path, dask_client, monkeypatch):
        """The no-merging path must stage the 3D-labelled field, not pin it.

        The bound is tighter than the merging test's ``2 x field_bytes``: with merging
        off there is no merge loop, so streaming has no whole-field pin left to make at
        all and anything approaching one field means the branch is still un-threaded.
        """
        data_bin = extremes.extreme_events.chunk(CHUNK_SIZE)
        field_bytes = data_bin.size * 4  # int32 whole field

        pinned = TestBytesPinned()._recorder(monkeypatch)
        tr = marEx.tracker(
            data_bin,
            extremes.mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **dict(TRACKER_KWARGS, allow_merging=False),
        )
        tr.run()
        total = sum(pinned)

        assert total < 0.5 * field_bytes, (
            f"streaming with allow_merging=False pinned {total} bytes, more than half a "
            f"whole int32 field ({field_bytes}). objects.py's `elif time_connectivity:` "
            f"branch is still pinning the whole labelled field."
        )

    @pytest.mark.slow
    def test_persist_without_merging_pins_a_whole_field(self, extremes, dask_client, monkeypatch):
        """Control for the test above: a recorder that sees nothing would pass it trivially."""
        data_bin = extremes.extreme_events.chunk(CHUNK_SIZE)
        field_bytes = data_bin.size * 4

        pinned = TestBytesPinned()._recorder(monkeypatch)
        tr = marEx.tracker(data_bin, extremes.mask, **dict(TRACKER_KWARGS, allow_merging=False))
        tr.run()
        total = sum(pinned)

        assert total >= field_bytes, (
            f"persist with allow_merging=False pinned only {total} bytes, less than one "
            f"whole int32 field ({field_bytes}). The recorder is not seeing the pins, so "
            f"the streaming bound above is meaningless."
        )


class TestBytesPinned:
    """Assert laziness by BYTES PINNED, not is_dask_collection.

    An array is STILL a dask collection after .persist() -- that check passes on a fully
    materialised dataset and proves nothing. Only a byte count is evidence.

    NOTE the instrumentation gap this repo has already been bitten by: marEx/track's
    modules do `from dask import persist`, which binds the ORIGINAL function at import
    time, so patching dask.persist alone misses those call sites entirely. The recorder
    below rebinds the module-level names too.
    """

    def _recorder(self, monkeypatch):
        import sys

        import dask

        pinned = []
        orig_dask = dask.persist
        orig_da = xr.DataArray.persist
        orig_ds = xr.Dataset.persist
        depth = {"n": 0}

        def _record(objs):
            if depth["n"]:
                return
            for o in objs:
                nb = getattr(o, "nbytes", 0)
                if isinstance(nb, int):
                    pinned.append(nb)

        def wrap(record_objs, call):
            _record(record_objs)
            depth["n"] += 1
            try:
                return call()
            finally:
                depth["n"] -= 1

        def dask_persist(*a, **k):
            return wrap(a, lambda: orig_dask(*a, **k))

        def da_persist(self, **k):
            return wrap((self,), lambda: orig_da(self, **k))

        def ds_persist(self, **k):
            return wrap((self,), lambda: orig_ds(self, **k))

        monkeypatch.setattr(dask, "persist", dask_persist)
        monkeypatch.setattr(xr.DataArray, "persist", da_persist)
        monkeypatch.setattr(xr.Dataset, "persist", ds_persist)
        for name, mod in list(sys.modules.items()):
            if name.startswith("marEx") and getattr(mod, "persist", None) is orig_dask:
                monkeypatch.setattr(mod, "persist", dask_persist)
        return pinned

    @pytest.mark.slow
    def test_persist_mode_pins_at_least_two_fields(self, extremes, dask_client, monkeypatch):
        """Control for the streaming test below.

        A recorder that silently records nothing would make the streaming assertion
        (``total < 2 * field_bytes``) pass trivially -- that would be evidence the
        instrument is broken, not that streaming is lazy. Run the SAME recorder against
        persist mode and confirm it sees at least as much as the streaming bound rules
        out. persist is documented to pin >= 7 whole int32 fields plus 5 bool fields
        (measured at scale, job 26764480); on this tiny fixture the same shapes must still
        clear 2 whole int32 fields.
        """
        data_bin = extremes.extreme_events.chunk(CHUNK_SIZE)
        field_bytes = data_bin.size * 4  # int32 whole field

        pinned = self._recorder(monkeypatch)
        tr = marEx.tracker(data_bin, extremes.mask, **TRACKER_KWARGS)
        tr.run()
        total = sum(pinned)

        assert total >= 2 * field_bytes, (
            f"persist pinned only {total} bytes, less than 2x one whole int32 field "
            f"({field_bytes}). The recorder is not seeing persist's own pins -- the "
            f"streaming test's 'far less than persist' comparison would be meaningless."
        )

    @pytest.mark.slow
    def test_streaming_pins_far_less_than_persist(self, extremes, tmp_path, dask_client, monkeypatch):
        data_bin = extremes.extreme_events.chunk(CHUNK_SIZE)
        field_bytes = data_bin.size * 4  # int32 whole field

        pinned = self._recorder(monkeypatch)
        tr = marEx.tracker(
            data_bin,
            extremes.mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **TRACKER_KWARGS,
        )
        tr.run()
        total = sum(pinned)

        # persist mode pins >= 7 whole int32 fields plus 5 bool fields (measured at
        # scale, job 26764480). Streaming must stay far under a single field.
        assert total < 2 * field_bytes, (
            f"streaming pinned {total} bytes, more than 2x one whole int32 field "
            f"({field_bytes}). The materialiser is not reaching every site."
        )
