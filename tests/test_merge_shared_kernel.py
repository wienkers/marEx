"""
The unstructured merge loop executes ``process_chunk`` ONCE per time chunk per iteration.

``merge_objects_parallel_iteration`` derives eight arrays from a single ``apply_ufunc``
call. All eight are ``getitem``s on one shared blockwise task -- the ``process_chunk``
invocation -- so materialising a strict subset of them lets the scheduler release that
shared task, and anchoring the remaining one afterwards re-runs the whole merge kernel.

That is what the loop used to do: seven arrays went through ``persist`` and
``updates_array`` was anchored on the line below. The cost is not marginal. ``process_chunk``
contains the BFS partitioner that instrumentation measured at 93 % of this stage's CPU, so
the loop was doing exactly twice the necessary work in its dominant stage. It was visible in
the instrumented ICON runs all along and read as normal: 80 invocations at n_time=32
(5 iterations x 8 time chunks = 40 expected) and 160 at n_time=64 (5 x 16 = 80).

The tests are in two layers, because the call-site fix depends on a dask behaviour that is
not obvious and that a future dask release could change:

* :class:`TestSharedTaskSemantics` pins the dask semantics themselves on a toy kernel,
  including the near-miss: routing the eighth array through a deferred
  ``to_zarr(compute=False)`` submitted in the SAME call does not share the task either,
  because ``to_zarr`` re-optimises its source graph and renames the shared keys.
* :class:`TestMergeLoopSharesKernel` measures the real call site against a control that
  restores the old wiring, and gates the outputs as bit-identical.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dask import persist as dask_persist

import marEx
from marEx.track import merge_split


@pytest.fixture(scope="module")
def unstructured_merging_data(dask_client_unstructured):
    """The merging-specific unstructured fixture, the one that actually reaches the loop.

    Deliberately NOT persisted, and deliberately dependent on the client fixture.

    This test module previously flaked under ``-n 4`` with
    ``FutureCancelledError: ... cancelled for reason: lost dependencies``, raised from the
    FIRST compute of the shipped leg (``validate_inputs``' ``mask.any().compute()``), long
    before the merge wiring this module is about. That error has exactly one source:
    ``Scheduler._find_lost_dependencies`` fires when a submitted graph references a key that
    is neither in the submitted graph nor in ``scheduler.tasks``. A dead worker does not
    produce it -- the key stays in ``scheduler.tasks`` and dask recomputes. Futures whose
    scheduler no longer knows their keys do.

    ``.persist()`` is what put futures in this graph, and it bought nothing: the store is
    309 KB, 100 timesteps x 405 cells. Dropping it removes the failure class outright --
    a graph rooted in the zarr store is recomputable by any scheduler, so there is no key
    that can go missing. That is the load-bearing half of this fix.

    Which client ended up minting those futures is NOT established, and is deliberately not
    claimed here. The fixture did not request the client, so being module-scoped like
    ``dask_client_per_module`` it was set up first (equal scope, argument order decides) and
    persisted against whatever ``_get_global_client()`` returned -- and that function skips
    only clients whose status is already ``closed``. A neighbouring module's client is the
    candidate, since ``-n 4`` selects xdist's ``--dist load``, which interleaves tests from
    different modules on one worker.

    That requires a client to have survived its own teardown, and it does: running this
    module prints ``Warning: Error during Dask client cleanup`` from
    ``dask_client_per_module``, because ``client.restart()`` raises
    ``AssertionError: assert not self.tasks`` inside ``Scheduler.restart``. The teardown
    calls ``restart()`` BEFORE ``close()`` inside one ``try``, so that exception -- swallowed
    by design -- skips ``close()`` and leaves a client that never reaches ``closed``. What is
    still not established is that this is what happened in the failing suite run
    (job 27115003); the chain is observed link by link, not end to end. It does not need to
    be: requesting the client fixture pins the ordering, and holding no futures at all makes
    the question moot.
    """
    path = Path(__file__).parent / "data" / "extremes_unstructured_merging.zarr"
    return xr.open_zarr(str(path), chunks={})


# Copied from `test_advanced_unstructured_tracking_with_merging`, which is the configuration
# this fixture's artificially-merging blobs were built for. It matters that merges actually
# occur: with no merge event the partitioner is never called and the ratio below would be
# 0 == 2 * 0, vacuously true. The test asserts a non-zero count for exactly that reason.
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


class TestSharedTaskSemantics:
    """Why the fix has the shape it has, pinned against dask itself.

    A toy two-output ``apply_ufunc`` stands in for ``process_chunk``. The kernel counts its
    own invocations, so the assertions are about executions, not about graph structure.
    """

    @staticmethod
    def _build(counter_path):
        def kernel(block):
            with open(counter_path, "ab") as handle:
                handle.write(b"x")
            return block * 2, block * 3

        data = xr.DataArray(np.arange(400).reshape(100, 4), dims=["t", "x"]).chunk({"t": 10})
        small, big = xr.apply_ufunc(
            kernel,
            data,
            input_core_dims=[["x"]],
            output_core_dims=[["x"], ["x"]],
            output_dtypes=[np.int64, np.int64],
            dask="parallelized",
        )
        return small, big

    @staticmethod
    def _count(counter_path):
        return counter_path.stat().st_size if counter_path.exists() else 0

    def test_persisting_a_subset_re_runs_the_shared_task(self, tmp_path):
        """The bug: materialise one output, then the other, and the kernel runs twice."""
        counter = tmp_path / "subset.count"
        small, big = self._build(counter)

        (small,) = dask_persist(small)
        small.compute()
        after_first = self._count(counter)

        big.compute()
        after_second = self._count(counter)

        n_chunks = len(small.chunks[0])
        assert after_first == n_chunks, f"control: expected one kernel call per chunk, saw {after_first} for {n_chunks} chunks"
        assert after_second == 2 * n_chunks, (
            f"expected the second consumer to re-run all {n_chunks} chunks (total {2 * n_chunks}), saw {after_second}. "
            f"If dask has started keeping the shared task alive, the merge-loop fix is now a no-op rather than wrong, "
            f"but the comment at its call site needs updating."
        )

    def test_persisting_together_shares_the_task(self, tmp_path):
        """The fix: name both outputs in ONE call and the kernel runs once."""
        counter = tmp_path / "together.count"
        small, big = self._build(counter)

        small, big = dask_persist(small, big)
        small.compute()
        big.compute()

        n_chunks = len(small.chunks[0])
        assert self._count(counter) == n_chunks, (
            f"expected {n_chunks} kernel calls when both outputs are persisted in one call, " f"saw {self._count(counter)}"
        )

    def test_deferred_zarr_write_does_not_share_the_task(self, tmp_path):
        """The near-miss, recorded so it is not re-attempted.

        Submitting ``to_zarr(compute=False)`` alongside the other output looks like one
        graph, and would avoid pinning the whole field. It does not share: ``to_zarr``
        re-optimises its source and renames the shared keys, so the kernel still runs twice.
        """
        counter = tmp_path / "deferred.count"
        small, big = self._build(counter)

        deferred = big.to_dataset(name="v").to_zarr(tmp_path / "deferred.zarr", mode="w", consolidated=True, compute=False)
        small, written = dask_persist(small, deferred)
        written.compute()
        small.compute()

        n_chunks = len(small.chunks[0])
        assert self._count(counter) == 2 * n_chunks, (
            f"a deferred to_zarr shared the task ({self._count(counter)} calls for {n_chunks} chunks). "
            f"If dask now preserves keys across to_zarr's optimisation, the merge loop could stage "
            f"updates_array without the transient whole-field pin -- worth revisiting."
        )


@pytest.mark.slow
class TestMergeLoopSharesKernel:
    """The real call site, against a control that restores the pre-fix wiring."""

    @staticmethod
    def _count_partition_calls(monkeypatch, counter_path):
        """Count partitioner calls THROUGH A FILE.

        ``process_chunk`` is a closure shipped to the workers by value, so a counter held in
        a list would be pickled by value too and the client would see zero. A path pickles
        fine and the file is shared, under threads and processes alike -- the same reason
        ``TestSharedLabellingPass`` counts this way.

        Both partitioners are wrapped, so the count does not depend on ``nn_partitioning``.
        """
        for name in ("partition_nn_unstructured_optimised", "partition_centroid_unstructured"):
            original = getattr(merge_split, name)

            def counting(*args, _original=original, **kwargs):
                with open(counter_path, "ab") as handle:
                    handle.write(b"x")
                return _original(*args, **kwargs)

            monkeypatch.setattr(merge_split, name, counting)

    @staticmethod
    def _restore_old_wiring(monkeypatch):
        """Reproduce the pre-fix call exactly: persist seven of the eight, leave the sixth.

        The sixth positional argument is ``updates_array``, which the shipped code names in
        the persist call and the old code did not.
        """

        def persist_all_but_updates_array(*objs, **kwargs):
            if len(objs) != 8:
                return dask_persist(*objs, **kwargs)
            kept = objs[5]
            persisted = list(dask_persist(*(objs[:5] + objs[6:]), **kwargs))
            persisted.insert(5, kept)
            return tuple(persisted)

        monkeypatch.setattr(merge_split, "persist", persist_all_but_updates_array)

    def _run(self, data, temp_dir):
        tracker = marEx.tracker(
            data.extreme_events,
            data.mask,
            temp_dir=str(temp_dir),
            neighbours=data.neighbours,
            cell_areas=data.cell_areas,
            **UNSTRUCTURED_KWARGS,
        )
        events, merges = tracker.run(return_merges=True)
        return events.compute(), merges.compute()

    def test_input_fixture_carries_no_futures(self, unstructured_merging_data):
        """Tripwire: re-adding ``.persist()`` to the fixture reintroduces the flake.

        A persisted collection carries ``distributed.Future`` objects in its graph, and a
        future is only meaningful to the scheduler that minted it. That is the whole
        mechanism behind the ``lost dependencies`` cancellation documented on the fixture,
        and it is invisible to every value-based assertion in this module -- the outputs are
        identical right up until the client changes underneath them.

        ``futures_of`` is the predicate, deliberately, rather than scanning graph values for
        ``Future`` instances: where futures sit in a materialised graph is a dask
        representation detail that has already moved once inside this package's supported
        dask range, and a scan that stops matching would leave a tripwire that passes on
        exactly what it exists to catch.
        """
        from distributed.client import futures_of

        held = futures_of(unstructured_merging_data)
        assert not held, (
            f"the fixture holds {len(held)} future(s), so it is pinned to one client's scheduler "
            f"and dies with it. Keep it rooted in the zarr store (no `.persist()`) -- see the "
            f"fixture docstring."
        )

    def test_kernel_runs_once_per_chunk_and_output_is_unchanged(
        self, unstructured_merging_data, tmp_path, dask_client_unstructured, monkeypatch
    ):
        counter = tmp_path / "partition.count"

        with monkeypatch.context() as patched:
            self._count_partition_calls(patched, counter)
            shipped_events, shipped_merges = self._run(unstructured_merging_data, tmp_path / "shipped")
        shipped_calls = counter.stat().st_size if counter.exists() else 0

        # Control for the control: a patch that reached nothing would make the bound vacuous.
        # Checked BEFORE the control leg runs, so a dead monkeypatch fails in one run's time
        # rather than two.
        assert shipped_calls > 0, (
            "the counter saw no partitioner calls at all, so the monkeypatch is not reaching "
            "the merge kernel and the ratio below proves nothing"
        )

        counter.unlink(missing_ok=True)
        with monkeypatch.context() as patched:
            self._count_partition_calls(patched, counter)
            self._restore_old_wiring(patched)
            control_events, control_merges = self._run(unstructured_merging_data, tmp_path / "control")
        control_calls = counter.stat().st_size if counter.exists() else 0

        # Absolute counts, not just their ratio: the ratio is what is asserted, but a change
        # in `shipped_calls` itself means the graph feeding the merge kernel moved, which is
        # worth seeing even on a green run. Visible under `pytest -s`.
        print(f"\npartitioner calls: shipped={shipped_calls} control={control_calls}")

        assert control_calls == 2 * shipped_calls, (
            f"expected the pre-fix wiring to run the merge kernel exactly twice "
            f"({2 * shipped_calls} calls against the shipped {shipped_calls}), saw {control_calls}. "
            f"The shared-task hazard this guards may have changed shape."
        )

        # The fix is a materialisation change only: no value may move.
        xr.testing.assert_identical(shipped_events, control_events)
        xr.testing.assert_identical(shipped_merges, control_merges)
