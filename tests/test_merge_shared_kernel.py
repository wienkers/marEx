"""
The unstructured merge loop executes ``process_chunk`` ONCE per dirty time chunk per iteration.

History. The loop used to derive eight arrays from a single ``apply_ufunc`` call; all eight were
``getitem``s on one shared blockwise task, so materialising a strict subset let the scheduler
release that task and anchoring the remaining one re-ran the whole kernel (80 invocations where
40 were expected in the instrumented ICON runs). That wiring is gone: design R (2026-09, D-084)
runs one ``dask.delayed`` task per dirty chunk and writes each chunk's labels to a zarr region.

The tests are in two layers:

* :class:`TestSharedTaskSemantics` pins the dask semantics behind the old hazard on a toy
  kernel, including the near-miss: routing an output through a deferred
  ``to_zarr(compute=False)`` submitted in the SAME call does not share the task either,
  because ``to_zarr`` re-optimises its source graph and renames the shared keys. Kept so the
  multi-output wiring is not rebuilt.
* :class:`TestMergeLoopSharesKernel` counts the real call site's kernel tasks, built and
  executed, and checks the batch structure of the loop.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dask import persist as dask_persist

import marEx
from marEx.track import merge_split

from .mre_mesh import same_partition


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


class _CountingDask:
    """Stand-in for the ``dask`` module as ``merge_split`` sees it.

    Only ``merge_split``'s own name is replaced, so nothing else in the process is touched.
    Every ``dask.compute`` call is recorded as one batch, and every ``process_chunk`` task
    records the chunk's first timestep twice: on the client when the task is BUILT, and
    through a file when the task EXECUTES. ``process_chunk`` is a closure shipped to the
    workers by value, so a list-held execution counter would be pickled and stay empty on
    the client; a path pickles fine and the file is shared under threads and processes.
    """

    def __init__(self, real, counter_path):
        self._real = real
        self._counter_path = str(counter_path)
        self._pending = []
        self.batches = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def delayed(self, func, *args, **kwargs):
        if getattr(func, "__name__", None) != "process_chunk":
            return self._real.delayed(func, *args, **kwargs)
        counter_path = self._counter_path

        def process_chunk(*call_args, **call_kwargs):
            with open(counter_path, "a") as handle:
                handle.write(f"{int(call_args[5])}\n")
            return func(*call_args, **call_kwargs)

        real_delayed = self._real.delayed(process_chunk)

        def build(*call_args, **call_kwargs):
            self._pending.append(int(call_args[5]))
            return real_delayed(*call_args, **call_kwargs)

        return build

    def compute(self, *args, **kwargs):
        if self._pending:
            self.batches.append(self._pending)
            self._pending = []
        return self._real.compute(*args, **kwargs)

    def executed(self):
        path = Path(self._counter_path)
        if not path.exists():
            return []
        return [int(line) for line in path.read_text().split()]


@pytest.mark.slow
class TestMergeLoopSharesKernel:
    """The real call site: every submitted chunk task executes exactly once.

    The merge loop (design R, 2026-09, D-084) runs ``process_chunk`` as one ``dask.delayed``
    task per DIRTY time chunk, in two batches per iteration (even chunks, then odd chunks,
    so no chunk reads a boundary slice that its own batch writes). Iteration 1 runs every
    chunk; a later iteration runs only chunks whose predecessor's last slice or forwarded
    queue changed. The previous wiring (eight arrays off one ``apply_ufunc``, where
    persisting a subset re-ran the kernel) no longer exists; ``TestSharedTaskSemantics``
    stays as the record of why that hazard is not to be rebuilt.
    """

    # The fixture is one 100-step chunk on disk. At 20 chunks of 5 a boundary changes and the loop
    # reruns two chunks in iteration 2 (observed 2026-09-13: batches 10, 10, 1, 1); at 10 or 20
    # steps it converges in one iteration and the rerun path would go unexercised.
    TIME_CHUNK = 5

    def _run(self, data, temp_dir, time_chunk):
        data = data.chunk({"time": time_chunk})
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
        and it is invisible to every value-based assertion in this module.

        ``futures_of`` is the predicate, deliberately, rather than scanning graph values for
        ``Future`` instances: where futures sit in a materialised graph is a dask
        representation detail that has already moved once inside this package's supported
        dask range.
        """
        from distributed.client import futures_of

        held = futures_of(unstructured_merging_data)
        assert not held, (
            f"the fixture holds {len(held)} future(s), so it is pinned to one client's scheduler "
            f"and dies with it. Keep it rooted in the zarr store (no `.persist()`) -- see the "
            f"fixture docstring."
        )

    def test_kernel_runs_once_per_dirty_chunk(self, unstructured_merging_data, tmp_path, dask_client_unstructured, monkeypatch):
        n_time = unstructured_merging_data.sizes["time"]
        chunk_starts = list(range(0, n_time, self.TIME_CHUNK))
        n_chunks = len(chunk_starts)

        counting = _CountingDask(merge_split.dask, tmp_path / "kernel.count")
        with monkeypatch.context() as patched:
            patched.setattr(merge_split, "dask", counting)
            chunked_events, chunked_merges = self._run(unstructured_merging_data, tmp_path / "chunked", self.TIME_CHUNK)
        batches = counting.batches
        executed = counting.executed()
        built = [t0 for batch in batches for t0 in batch]

        # Visible under `pytest -s`: the shape of the loop on this fixture.
        print(f"\nkernel batches (chunk first timesteps): {batches}; executions: {len(executed)}")

        # Control for the control: a patch that reached nothing would make every bound vacuous.
        assert built, "no process_chunk task was built through the patched dask; the patch is not reaching the loop"
        assert chunked_merges.sizes.get("merge_ID", 0) > 0, "the fixture produced no merge, so the kernel was never exercised"

        # Every task that was built ran exactly once: no re-execution, no dropped task.
        assert sorted(executed) == sorted(built), f"executions {sorted(executed)} != submitted tasks {sorted(built)}"

        # Iteration 1 is the first two batches and covers every chunk once, by parity.
        assert len(batches) >= 2, f"expected at least the two parity batches of iteration 1, saw {batches}"
        assert batches[0] == chunk_starts[0::2] and batches[1] == chunk_starts[1::2], batches[:2]

        # No batch runs a chunk twice, and no batch mixes parities (a chunk would then read a
        # boundary slice written by its own batch).
        for batch in batches:
            assert len(batch) == len(set(batch)), f"a chunk ran twice in one batch: {batch}"
            parities = {chunk_starts.index(t0) % 2 for t0 in batch}
            assert len(parities) == 1, f"batch mixes chunk parities: {batch}"

        # The rerun path must be exercised, and a rerun is a strict subset of the chunks.
        assert len(batches) > 2, f"the loop converged in one iteration, so no rerun was exercised: {batches}"
        assert all(len(batch) < n_chunks // 2 + 1 for batch in batches[2:]), batches

        # The loop is not a fixed n_iterations x n_chunks sweep: beyond iteration 1 it only
        # reruns chunks downstream of a change, so the total is bounded by a serial cascade.
        assert len(built) <= n_chunks * n_chunks, f"{len(built)} kernel runs for {n_chunks} chunks"

        # Chunking must not move the result (the invariance itself is pinned in
        # test_merge_chunk_invariance.py; this checks the fixture the count ran on).
        single_events, single_merges = self._run(unstructured_merging_data, tmp_path / "single", n_time)
        assert int(chunked_merges.sizes["merge_ID"]) == int(single_merges.sizes["merge_ID"])
        assert same_partition(np.asarray(chunked_events.ID_field.values), np.asarray(single_events.ID_field.values))
