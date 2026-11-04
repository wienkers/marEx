"""The unstructured merge loop: chunk and path invariance, the ledger, the compaction retry, the
kernel's record widths and hand-computed overlaps, int32 ID ranges, real parent centroids and the
region writer.

``split_and_merge_objects_parallel`` processes each time chunk from the iteration's input field and
defers cascades that cross a chunk boundary. Before the 2026-09 repair a deferred boundary object was
partitioned against stale parents and never repaired, so lineages that were only ever bridged for a
few days fused into one event, and the result changed with the chunk width. The oracle is the serial
(gridded) path: the single-chunk parallel run matches it on merge counts, events AND object IDs, and
both paths are independent of the time chunking, including one-timestep chunks.

Two design records that used to be tests of NumPy and dask themselves (they imported no marEx code,
so no marEx change could fail them) are kept here as text:

* The kernel tests a candidate parent at the child's own ascending cell indices,
  ``overlap_cells = child_cells[parent_mask[child_cells]]``, rather than intersecting two whole-mesh
  masks. That is O(child) instead of O(ncells) (14.9 M cells on ICON R02B09), and it gathers the same
  elements in the same ascending order as ``area[parent_mask & child_mask]``, so the sum is
  byte-identical. A single ULP moves a comparison against ``overlap_threshold``, so the order matters:
  ``test_merge_kernel_overlap_area_is_the_ascending_sum`` drives the real kernel on areas where any
  other order rounds differently; ``test_merge_kernel_forward_overlap_is_the_ascending_sum`` does the same
  for the forward search's overlap with a next-step object.
* The kernel runs as one ``dask.delayed`` task per dirty chunk that writes its own zarr region. The
  previous wiring derived eight arrays from one shared ``apply_ufunc`` task; materialising a strict
  subset let the scheduler release that task, so the next consumer re-ran the whole kernel (80 kernel
  calls where 40 were expected on ICON). Persisting the outputs together in one call shares the task,
  and a deferred ``to_zarr(compute=False)`` submitted alongside does NOT: ``to_zarr`` re-optimises its
  source graph and renames the shared keys. Do not rebuild the multi-output wiring.
  ``test_kernel_runs_once_per_dirty_chunk`` counts the real executions.

Every test that tracks runs on the session ``client`` (where it ran before the consolidation), except
``test_merge_loop_passes_only_real_parent_centroids``, which needs an in-process client so that the
patched partitioners are the ones the merge tasks call.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import dask
import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import TrackingError
from marEx.track import merge_split
from marEx.track import overlap as _overlap
from marEx.track.merge_split import _area_sum, _overlap_fraction
from marEx.track.objects import ObjectPropsStore, calculate_object_properties
from marEx.track.partitioning import partition_centroid_unstructured, partition_nn_unstructured_optimised
from marEx.track.region_writer import ObjectIDRegionWriter
from marEx.track.tracker import tracker as _Tracker

from .conftest import (
    CHAIN_KWARGS,
    UNSTRUCTURED_KWARGS,
    UNSTRUCTURED_MERGING_KWARGS,
    _chain_inputs,
    _track,
    _track_chain,
    assert_no_failures,
    check,
    on_client,
    open_unstructured,
    open_unstructured_merging,
    released,
)
from .mre_mesh import (
    NT,
    TRACKER_KWARGS,
    cells_to_squares,
    fused_events,
    random_scenario,
    render,
    same_partition,
    scenarios,
    tracker_inputs,
)

CASES = ["S1", "S2", "S3", "S4", "seed5", "seed15"]
WIDTHS = [5, 4, 8]
# S4 has no merges on either path (it merged, 20 times, only on the pre-fix serial path); every other
# scenario merges, so equal merge counts there are not a 0 == 0 match. S4's own truth is zero merges.
NON_MERGING = {"S4"}


def _assert_merges(n_merges, message, merging=True):
    if merging:
        assert n_merges > 0, message
    else:
        assert n_merges == 0, message


def _scenario(name):
    if name.startswith("seed"):
        return random_scenario(int(name[4:]))
    return scenarios()[name]


@pytest.fixture(scope="module")
def reference(client, tmp_path_factory):
    """Single-chunk parallel runs (the oracle), computed on the session client the first time a test asks.

    Returns ``get(name) -> (scenario, binary, lineage, bridge_day, (events, objects, n_merges))``, all numpy.
    """
    cache = {}

    def get(name):
        if name not in cache:
            assert on_client(client), "the scenario references are not computing on the session client"
            sc = _scenario(name)
            binary, lineage, bridge_day = render(sc["nrow"], sc["ncol"], NT, sc["blobs"], sc["bridges"])
            run = _track(binary, sc["nrow"], sc["ncol"], NT, tmp_path_factory.mktemp(name))
            cache[name] = (sc, binary, lineage, bridge_day, run)
        return cache[name]

    return get


@pytest.mark.parametrize("name", CASES)
def test_scenario_is_chunk_and_path_invariant(reference, name, client, tmp_path):
    """The single-chunk run recovers the truth; every chunk width and the serial path reproduce it.

    The truth is independent of the code: no event fuses two lineages that were only bridged, and the
    event count equals the scenario's. Widths 5, 4 and 8 put chunk boundaries at different distances
    from the bridges, so all three are kept.
    """
    assert on_client(client)
    sc, binary, lineage, bridge_day, (ref_events, ref_objects, ref_merges) = reference(name)
    failures = []

    def truth():
        fused = fused_events(cells_to_squares(ref_events, sc["nrow"], sc["ncol"]), lineage, bridge_day)
        assert fused == {}, f"{name}: the single-chunk run fused lineages {fused}"
        n_events = len(np.unique(ref_events[ref_events > 0]))
        assert n_events == sc["truth"], f"{name}: {n_events} events, the scenario has {sc['truth']}"

    check(failures, "truth", truth)
    merging = name not in NON_MERGING
    expected = "made no merges" if merging else f"made {ref_merges} merges, the scenario has none"
    check(failures, "merges", lambda: _assert_merges(ref_merges, f"{name}: the single-chunk run {expected}", merging))

    for width in WIDTHS:

        def chunked(width=width):
            events, objects, n_merges = _track(binary, sc["nrow"], sc["ncol"], width, tmp_path / f"w{width}")
            assert n_merges == ref_merges, f"{name} width={width}: {n_merges} merges, single chunk {ref_merges}"
            assert same_partition(objects, ref_objects), f"{name} width={width}: object partition differs from the single chunk"
            assert same_partition(events, ref_events), f"{name} width={width}: event partition differs from the single chunk"

        check(failures, f"width={width}", chunked)

    def serial():
        events, objects, n_merges = _track(binary, sc["nrow"], sc["ncol"], NT, tmp_path / "serial", serial=True)
        assert n_merges == ref_merges, f"{name} serial: {n_merges} merges, parallel {ref_merges}"
        assert same_partition(objects, ref_objects), f"{name} serial: object partition differs from the parallel path"
        assert np.array_equal(events, ref_events), f"{name} serial: event IDs differ from the parallel path"

    check(failures, "serial", serial)
    assert_no_failures(failures)


@pytest.mark.parametrize("name", ["S4", "seed0", "seed6"])
def test_serial_path_is_chunk_invariant(name, client, tmp_path):
    """(3, ..., 3, 1) and all-ones. The serial path skipped end-of-chunk consolidation for a one-timestep
    chunk; these three cases changed object IDs (S4 also merges, 20 vs 0) before the fix."""
    assert on_client(client)
    sc = _scenario(name)
    binary, _, _ = render(sc["nrow"], sc["ncol"], NT, sc["blobs"], sc["bridges"])
    _, ref_objects, ref_merges = _track(binary, sc["nrow"], sc["ncol"], NT, tmp_path / "one", serial=True)
    merging = name not in NON_MERGING
    expected = "made no merges" if merging else f"made {ref_merges} merges, the scenario has none"
    _assert_merges(ref_merges, f"{name}: the one-chunk serial run {expected}", merging)
    failures = []
    for width in (3, 1):

        def chunked(width=width):
            _, objects, n_merges = _track(binary, sc["nrow"], sc["ncol"], width, tmp_path / f"w{width}", serial=True)
            assert n_merges == ref_merges, f"{name} width={width}: {n_merges} merges, one chunk {ref_merges}"
            assert same_partition(objects, ref_objects), f"{name} width={width}: object partition differs from one chunk"

        check(failures, f"width={width}", chunked)
    assert_no_failures(failures)


def _run_dataset(binary_sq, nrow, ncol, time_chunks, temp_dir, **extra):
    """A parallel run at an explicit time chunking; computed (events, merges)."""
    da, mask, neighbours, cell_areas = tracker_inputs(binary_sq, nrow, ncol, NT)
    tr = marEx.tracker(
        da.chunk({"time": time_chunks}),
        mask,
        temp_dir=str(temp_dir),
        neighbours=neighbours,
        cell_areas=cell_areas,
        **{**TRACKER_KWARGS, **extra},
    )
    try:
        events, merges = tr.run(return_merges=True)
        return events.compute(), merges.compute()
    except Exception as e:  # noqa: BLE001 - re-raised below without its traceback
        failure = f"{type(e).__name__}: {e}"
    # A failed run's traceback holds the tracker's persisted intermediates, which would also leak futures
    # past the module; report the error without them.
    del tr
    raise AssertionError(f"the tracker run raised {failure}")


def test_size_one_tail_chunk_with_merges(reference, client, tmp_path):
    """(3, ..., 3, 1): the last chunk holds one timestep. Crashed before 2026-09."""
    assert on_client(client)
    sc, binary, _, _, (ref_events, _, ref_merges) = reference("S3")
    assert NT % 3 == 1
    assert ref_merges > 0, "the S3 reference made no merges"
    events, merges = _run_dataset(binary, sc["nrow"], sc["ncol"], 3, tmp_path)
    assert int(merges.sizes["merge_ID"]) == ref_merges
    assert same_partition(np.asarray(events.ID_field.values), ref_events)


def test_ragged_time_chunks_are_retiled(reference, client, tmp_path):
    """An interior-ragged chunking is re-tiled once at construction; the result is unchanged."""
    assert on_client(client)
    sc, binary, _, _, (ref_events, _, ref_merges) = reference("S3")
    ragged = (2, 5, 5, 5, 5, 5, 5, 5, 3)
    assert sum(ragged) == NT
    da, mask, neighbours, cell_areas = tracker_inputs(binary, sc["nrow"], sc["ncol"], NT)
    tr = marEx.tracker(
        da.chunk({"time": ragged}), mask, temp_dir=str(tmp_path), neighbours=neighbours, cell_areas=cell_areas, **TRACKER_KWARGS
    )
    retiled, timechunks = tr.data_bin.chunks[tr.data_bin.dims.index("time")], tr.timechunks
    events, merges = tr.run(return_merges=True)
    n_merges, id_field = int(merges.sizes["merge_ID"]), np.asarray(events.ID_field.values)
    del tr, events, merges
    assert retiled == (5, 5, 5, 5, 5, 5, 5, 5)
    assert timechunks == 5
    assert ref_merges > 0, "the S3 reference made no merges"
    assert n_merges == ref_merges
    assert same_partition(id_field, ref_events)


def test_merge_ledger_is_written_positionally(client, tmp_path):
    """The ledger holds, at (time, ID, sibling), that parent's own event ID (the pre-2026-09 label-based
    write produced exactly this); it is written by position so a shared pandas index engine can never race.

    Also the catcher for the default record widths: S2 merges at width 5, so a width too small for its
    merges raises rather than leaving this ledger empty.
    """
    assert on_client(client)
    sc = scenarios()["S2"]
    binary, _, _ = render(sc["nrow"], sc["ncol"], NT, sc["blobs"], sc["bridges"])
    events, merges = _run_dataset(binary, sc["nrow"], sc["ncol"], 5, tmp_path)
    ledger = events.merge_ledger.transpose("time", "ID", "sibling_ID").values
    ids = events.ID.values
    filled = ledger != -1
    assert filled.any()
    rows = np.broadcast_to(ids[None, :, None], ledger.shape)
    assert np.array_equal(ledger[filled], rows[filled])
    # every merge time carries at least one parent row
    merge_times = np.unique(merges.merge_time.values)
    time_has_row = filled.any(axis=(1, 2))
    assert time_has_row[np.isin(events.time.values, merge_times)].all()


def test_compaction_survives_a_task_retry(reference, client, tmp_path, monkeypatch):
    """A worker restart makes dask re-run a compaction task whose first attempt already wrote its region.
    Minted IDs used to share a range with the compacted ones, so the retry remapped IDs a second time and
    the run crashed or relabelled cells. Every region is compacted once up front, then the real compute runs
    over the already-compacted store. Scenarios S3 and seed5."""
    assert on_client(client)
    failures = []
    for name in ("S3", "seed5"):

        def retried(name=name):
            sc, binary, _, _, _ = reference(name)
            ref_events, ref_objects, ref_merges = _track(binary, sc["nrow"], sc["ncol"], 5, tmp_path / f"{name}_once")
            assert ref_merges > 0, f"{name}: the run without a retry made no merges"
            real_compute = dask.compute
            first_attempts = []

            def _compute_with_retry(*args, **kwargs):
                regions = [a for a in args if str(getattr(a, "key", "")).startswith("_compact_region")]
                if regions:
                    first_attempts.extend(real_compute(*regions, scheduler="synchronous"))
                return real_compute(*args, **kwargs)

            with monkeypatch.context() as patched:
                patched.setattr(merge_split.dask, "compute", _compute_with_retry)
                events, objects, n_merges = _track(binary, sc["nrow"], sc["ncol"], 5, tmp_path / f"{name}_retried")

            assert first_attempts, f"{name}: no compaction task was re-executed, so nothing was tested"
            assert n_merges == ref_merges, f"{name}: {n_merges} merges after a retry, {ref_merges} without"
            assert np.array_equal(objects, ref_objects), f"{name}: a compaction retry changed object IDs"
            assert np.array_equal(events, ref_events), f"{name}: a compaction retry changed event IDs"

        check(failures, name, retried)
    assert_no_failures(failures)


# -- the captured kernel -------------------------------------------------------------------------

N_CELLS = 12


def _row(spans, n=N_CELLS):
    """A one-timestep label row on an ``n``-cell chain: ``{id: (first cell, last cell)}``."""
    out = np.zeros(n, np.int32)
    for ident, (lo, hi) in spans.items():
        out[lo : hi + 1] = ident
    return out


def test_kernel_record_width_guards(merge_kernel, monkeypatch, tmp_path):
    """Drive the guards on the real kernel, on a twelve-cell chain with unit areas and threshold 0.5.

    Previous timestep: P1 = cells 0-3 (4), P2 = cells 4-8 (5), P3 = cells 9-11 (3). The record width
    is lowered to two parents; the guard logic does not depend on its value.

    (a) Child 5 = cells 2-9: P1 overlaps 2 / 4 (accepted, on the threshold), P2 5 / 5 (accepted),
        P3 1 / 3 (rejected). The reject has the largest id, so it is scanned after the arrays are
        already full: a guard that counts candidates rather than accepted parents raises here.
    (b) Child 5 = cells 2-10: P3 now overlaps 2 / 3, so three parents are accepted and must raise.
    (c) The default parent width with no merges allowed: (a) must raise on the merge count.
    (d) ``parent_masks_uint`` is uint8 with 255 as "no parent", so the width must stay <= 255.
    """
    previous = _row({1: (0, 3), 2: (4, 8), 3: (9, 11)})
    following = _row({7: (3, 6), 8: (9, 11), 9: (0, 2)})

    def run(current, tag):
        store = str(tmp_path / f"{tag}.zarr")
        xr.DataArray(current[None, :], dims=("time", "ncells"), name="temp").to_zarr(store, mode="w")
        return merge_kernel(
            current[None, :],
            previous,
            following,
            [[5]],
            np.array([100, 101], dtype=np.int64),
            0,
            np.zeros(N_CELLS, np.float32),
            np.arange(N_CELLS, dtype=np.float32),
            np.ones(N_CELLS, np.float32),
            np.zeros((3, N_CELLS), np.int32),
            store,
            False,
        )

    full_plus_reject = _row({5: (2, 9)})
    default_parents = merge_split.MAX_PARENTS

    monkeypatch.setattr(merge_split, "MAX_PARENTS", 2)
    records, _, _, _, max_parents, _, _, _ = run(full_plus_reject, "a")
    assert records[0][2].tolist() == [1, 2]
    assert max_parents == 2

    with pytest.raises(TrackingError, match="Too many parent objects"):
        run(_row({5: (2, 10)}), "b")

    monkeypatch.setattr(merge_split, "MAX_PARENTS", default_parents)
    monkeypatch.setattr(merge_split, "MAX_MERGES", 0)
    with pytest.raises(TrackingError, match="Too many merge operations"):
        run(full_plus_reject, "c")

    assert default_parents <= 255


def test_width_guard_propagates_from_a_tracker_run(client, monkeypatch, tmp_path):
    """S2 has two-parent merges, so a one-parent record width must raise out of a real run (through dask)."""
    assert on_client(client)
    sc = scenarios()["S2"]
    binary, _, _ = render(sc["nrow"], sc["ncol"], NT, sc["blobs"], sc["bridges"])
    monkeypatch.setattr(merge_split, "MAX_PARENTS", 1)
    try:
        _track(binary, sc["nrow"], sc["ncol"], 5, tmp_path)
    except AssertionError as e:  # ``_track`` re-raises a tracker error as an AssertionError carrying its type and message
        message = str(e)
    else:
        message = None
    assert message is not None, "a one-parent record width did not raise on S2's two-parent merges"
    assert "raised TrackingError:" in message, message
    assert "Too many parent objects" in message, message


def test_merge_kernel_hand_computed_overlaps(merge_kernel, tmp_path):
    """Drive the real ``process_chunk`` on a twelve-cell chain whose every overlap fraction is worked out by hand.

    ``_overlap_fraction`` first: the float64 ratio of the overlap to the smaller area.

    The threshold is 0.5 and every cell has unit area and sits one degree east of the last.

    Previous timestep:  P1 = cells 0-3 (4), P2 = cells 4-8 (5), P3 = cells 9-11 (3).
    Child 5 at this timestep: cells 2-9 (8).
      parent search, overlap / min(parent area, child area):
        P1: cells 2, 3        -> 2 / min(4, 8) = 0.5     accepted (exactly on the threshold)
        P2: cells 4-8         -> 5 / min(5, 8) = 1.0     accepted
        P3: cell 9            -> 1 / min(3, 8) = 1/3     rejected
      two parents merge: child ids [5, 100]. Cells go to the nearer parent centroid (P1 at 1.5, P2 at 6):
        piece 5 = cells 2, 3 (area 2); piece 100 = cells 4-9 (area 6).
    Next timestep:  A = cells 3-6 (4), B = cells 9-11 (3), C = cells 0-2 (3).
      forward search, overlap / min(piece area, candidate area), kept when >= 0.5:
        A: piece 5 -> cell 3 -> 1 / min(2, 4) = 0.5      kept (on the threshold); piece 100 -> 3 / min(6, 4) = 0.75
        B: piece 100 -> cell 9 -> 1 / min(6, 3) = 1/3    rejected
        C: piece 5 -> cell 2 -> 1 / min(2, 3) = 0.5      kept
      so the forwarded objects are A and C, and not B.
    """
    got = _overlap_fraction(np.array([2, 5, 1], dtype=np.float32), np.array([4, 5, 3], dtype=np.float32))
    assert got.dtype == np.float64
    assert got[0] == 0.5 and got[1] == 1.0 and got[2] == 1.0 / 3.0

    previous = _row({1: (0, 3), 2: (4, 8), 3: (9, 11)})
    current = _row({5: (2, 9)})
    following = _row({7: (3, 6), 8: (9, 11), 9: (0, 2)})

    store = str(tmp_path / "accumulator.zarr")
    xr.DataArray(current[None, :], dims=("time", "ncells"), name="temp").to_zarr(store, mode="w")
    records, forwarded, _, _, max_parents, wrote, _, _ = merge_kernel(
        current[None, :],
        previous,
        following,
        [[5]],
        np.array([100], dtype=np.int64),
        0,
        np.zeros(N_CELLS, np.float32),
        np.arange(N_CELLS, dtype=np.float32),
        np.ones(N_CELLS, np.float32),
        np.zeros((3, N_CELLS), np.int32),
        store,
        False,
    )

    assert len(records) == 1
    t_abs, child_ids, parent_ids, overlap_areas = records[0]
    assert t_abs == 0
    assert child_ids.tolist() == [5, 100]
    assert parent_ids.tolist() == [1, 2]
    assert overlap_areas.tolist() == [2.0, 5.0]
    assert max_parents == 2 and wrote
    assert forwarded == [7, 9]
    assert xr.open_zarr(store)["temp"].values[0].tolist() == [0, 0, 5, 5, 100, 100, 100, 100, 100, 100, 0, 0]


def test_merge_kernel_overlap_area_is_the_ascending_sum(merge_kernel, tmp_path):
    """A parent's recorded overlap area is the float64 sum, rounded once, of its cells in ascending order:
    exactly what the whole-mesh boolean form ``area[parent_mask & child_mask]`` gives.

    The areas are built so that the order is observable. P1's cells hold 2**24, 1, 2**-29, 2**-29: in
    ascending order the two small addends are each half an ULP of 2**24 + 1 in float64 and round away,
    and 2**24 + 1 then rounds (to even) to 2**24 in float32; summed from the other end they first add up
    to a full ULP and the result rounds to 2**24 + 2. On random areas a reordered float64 sum essentially
    never changes the float32 result, which is why this input is engineered rather than random.

    Child 5 covers P1 (cells 0-3) and P2 (cells 4-8, unit areas) whole, so both are accepted and the
    merge is recorded with both overlap areas.
    """
    n = 10
    area = np.ones(n, np.float32)
    area[:4] = [2.0**24, 1.0, 2.0**-29, 2.0**-29]
    previous = _row({1: (0, 3), 2: (4, 8)}, n)
    current = _row({5: (0, 8)}, n)
    following = np.zeros(n, np.int32)
    p1_cells = np.nonzero((previous == 1) & (current == 5))[0]
    boolean_form = _area_sum(area[(previous == 1) & (current == 5)])
    # The input must make the order matter, or the check below is vacuous.
    assert _area_sum(area[p1_cells[::-1]]) != boolean_form, "the areas no longer make the summation order observable"

    store = str(tmp_path / "accumulator.zarr")
    xr.DataArray(current[None, :], dims=("time", "ncells"), name="temp").to_zarr(store, mode="w")
    records, *_ = merge_kernel(
        current[None, :],
        previous,
        following,
        [[5]],
        np.array([100], dtype=np.int64),
        0,
        np.zeros(n, np.float32),
        np.arange(n, dtype=np.float32),
        area,
        np.zeros((3, n), np.int32),
        store,
        False,
    )
    assert len(records) == 1
    _, _, parent_ids, overlap_areas = records[0]
    assert parent_ids.tolist() == [1, 2]
    assert overlap_areas[0] == boolean_form, f"P1 overlap area {overlap_areas[0]!r}, boolean form {boolean_form!r}"
    assert overlap_areas[1] == np.float32(5.0)


def test_merge_kernel_forward_overlap_is_the_ascending_sum(merge_kernel, tmp_path):
    """The forward search sums a piece's overlap with a next-step object over the piece's cells in ascending
    order too, and here that decides whether the object is forwarded: its overlap fraction is exactly the
    threshold (0.5) in ascending order and falls below it in any other.

    Previous timestep: P1 = cells 0-2, P2 = cells 3-5 (unit areas). Child 5 = cells 0-10 covers both whole,
    so both are accepted; the centroid partition gives piece 5 = cells 0-2 and piece 100 = cells 3-10.
    Next timestep: object 7 = cells 6-9 and 11, areas 2**-29, 2**-29, 1, 2**24 and 2**24 + 2. Its overlap
    with piece 100 is cells 6-9: ascending, the two small addends first make 1 + 2**-28, so the float64 sum
    sits just above 2**24 + 1 and rounds up to 2**24 + 2 in float32; from the other end they are each half an
    ULP and round away, and 2**24 + 1 rounds (to even) to 2**24. Object 7's own area is 2**25 + 4 (cell 10,
    2**25, makes the piece the larger of the two), so the fraction is exactly 0.5 only in ascending order.
    """
    n = 14
    area = np.ones(n, np.float32)
    area[6:12] = [2.0**-29, 2.0**-29, 1.0, 2.0**24, 2.0**25, 2.0**24 + 2]
    previous = _row({1: (0, 2), 2: (3, 5)}, n)
    current = _row({5: (0, 10)}, n)
    following = _row({7: (6, 9)}, n)
    following[11] = 7
    pieces = _row({5: (0, 2), 100: (3, 10)}, n)

    # The order must decide the threshold comparison, or the check below is vacuous: the forward search's
    # own index arrays, the piece's cells ascending and then the candidate's cells among them.
    parent_cells = np.nonzero(pieces == 100)[0].astype(np.int32)
    candidate = following == 7
    overlap_cells = parent_cells[candidate[parent_cells]]
    min_area = min(_area_sum(area[parent_cells]), _area_sum(area[candidate]))
    threshold = CHAIN_KWARGS["overlap_threshold"]  # the captured kernel's threshold
    assert _overlap_fraction(_area_sum(area[overlap_cells]), min_area) == threshold, "the ascending overlap is not on the threshold"
    assert _overlap_fraction(_area_sum(area[overlap_cells[::-1]]), min_area) < threshold, (
        "the summation order is no longer observable"
    )

    store = str(tmp_path / "accumulator.zarr")
    xr.DataArray(current[None, :], dims=("time", "ncells"), name="temp").to_zarr(store, mode="w")
    records, forwarded, *_ = merge_kernel(
        current[None, :],
        previous,
        following,
        [[5]],
        np.array([100], dtype=np.int64),
        0,
        np.zeros(n, np.float32),
        np.arange(n, dtype=np.float32),
        area,
        np.zeros((3, n), np.int32),
        store,
        False,
    )
    assert len(records) == 1
    _, child_ids, parent_ids, _ = records[0]
    assert child_ids.tolist() == [5, 100] and parent_ids.tolist() == [1, 2]
    assert xr.open_zarr(store)["temp"].values[0].tolist() == pieces.tolist(), "the pieces are not the ones the areas were built for"
    assert forwarded == [7], f"object 7 overlaps piece 100 exactly on the threshold but forwarded is {forwarded}"


def test_exact_threshold_overlap_continues_the_cascade(client, tmp_path):
    """After a merge partitions a child, the forward search queues every next-step object that one of the
    new pieces overlaps by at least ``overlap_threshold``. A strict comparison skipped a piece whose overlap
    sat exactly on the threshold, so a cascade the serial path follows was cut short.

    On the nine-cell chain (unit areas, threshold 0.5) two objects join into one, which splits into a
    five-cell and a four-cell piece; the next object covers the four middle cells, two inside each piece,
    so each piece overlaps it by exactly 2 / 4.
    """
    assert on_client(client)
    events, objects, n_merges = _track_chain(tmp_path / "parallel", serial=False)
    ref_events, ref_objects, ref_merges = _track_chain(tmp_path / "serial", serial=True)

    # The serial path partitions the last object between the two pieces: two merges, two events.
    assert ref_merges == 2
    assert np.array_equal(objects[2] > 0, ref_objects[2] > 0)
    assert len(np.unique(ref_objects[2][ref_objects[2] > 0])) == 2

    assert n_merges == ref_merges, "the forward search dropped a child that overlaps a new piece exactly at the threshold"
    assert np.array_equal(objects, ref_objects)
    assert np.array_equal(events, ref_events)


# A field on the same chain where, after splitting and merging, one object overlaps two distinct parents.
# Found by search over small random fields; at a 0.1 threshold the final overlap list holds child 7 with
# parents 5 and 8 (areas 3 and 2, wholly overlapped), so the multi-parent report fires.
MULTI_PARENT_ROWS = [
    [1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]


def test_a_child_with_several_parents_is_reported(client, tmp_path, caplog):
    """The multi-parent validation logs one summary, one detail line with the overlap details, and no
    consistency warning."""
    assert on_client(client)
    # The package root logger neither propagates to pytest's handler nor keeps handlers added before the
    # tracker is built (configuring it clears them), so attach caplog's to the module's own logger.
    module_logger = logging.getLogger("marEx.track.merge_split")
    module_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG, logger="marEx.track.merge_split"):
            _track_chain(tmp_path, serial=True, rows=MULTI_PARENT_ROWS, overlap_threshold=0.1, quiet=False)
    finally:
        module_logger.removeHandler(caplog.handler)
    records = [r for r in caplog.records if r.name == "marEx.track.merge_split"]

    summary = [r for r in records if r.levelno == logging.INFO and "more than one parent after splitting/merging" in r.getMessage()]
    assert len(summary) == 1, "the multi-parent summary was not logged exactly once"
    assert summary[0].getMessage().startswith("1 children have more than one parent")

    detail = [
        r.getMessage()
        for r in records
        if r.levelno == logging.DEBUG and r.getMessage().startswith("Child ") and "parent areas" in r.getMessage()
    ]
    assert len(detail) == 1, detail
    assert "parents [5.0, 8.0]" in detail[0] and "parent areas [3.0, 2.0]" in detail[0]
    assert "overlap fractions [1.0, 1.0]" in detail[0]
    assert any(r.levelno == logging.DEBUG and "appears at timesteps" in r.getMessage() for r in records)
    assert not any(r.levelno == logging.INFO and "Validation passed" in r.getMessage() for r in records)
    # Each overlap is exactly 1.0 of its parent and together they cover 5 of the child's 8 cells, so neither
    # consistency warning (a fraction above 1, or overlaps exceeding the child's area) applies.
    consistency = [
        r.getMessage()
        for r in records
        if r.levelno == logging.WARNING and ("overlap fraction" in r.getMessage() or "exceed its area" in r.getMessage())
    ]
    assert not consistency, consistency


# -- int32 ID ranges ------------------------------------------------------------------------------


class _Stop(Exception):
    """Raised by a patched step to end a run once the checked range has been accepted."""


def _id_range_tracker(ds, temp_dir, allow_merging):
    return marEx.tracker(
        ds.extreme_events.chunk({"time": 25, "ncells": -1}),
        ds.mask,
        R_fill=2,
        area_filter_quartile=0.5,
        temp_dir=str(temp_dir),
        T_fill=0,
        allow_merging=allow_merging,
        neighbours=ds.neighbours,
        cell_areas=ds.cell_areas,
        **UNSTRUCTURED_KWARGS,
    )


def test_int32_id_ranges(client, tmp_path, monkeypatch):
    """Both int32 ranges are checked before they are used, with the last accepted value as the boundary.

    * Unique IDs: the per-slice labels are offset by the running maximum, and the unique field keeps the
      int32 dtype. A total of int32 max - 1 is accepted (max ID + 1 still fits); one more raises.
    * Merge temporaries: minted above two reserved ranges of ``n_time * id_stride`` IDs past the largest
      object ID. The largest ID that leaves the top temporary ID + 1 inside int32 is accepted; one more raises.
    """
    assert on_client(client)
    ds = open_unstructured()
    limit = np.iinfo(np.int32).max
    n_cells = ds.sizes["ncells"]
    id_stride = merge_split.MAX_MERGES * (merge_split.MAX_PARENTS - 1)

    def field(first, second):
        out = np.zeros((2, n_cells), dtype=np.int32)
        out[0, 0], out[1, 1] = first, second
        return xr.DataArray(out, dims=("time", "ncells")).chunk({"time": 1})

    def unique_offsets(excess):
        first = 2**30
        object_id_field = field(first, limit - 1 - first + excess)  # total = limit - 1 (accepted) or limit (rejected)

        def stop(self, data_bin):
            raise _Stop(data_bin)

        with monkeypatch.context() as patched:
            tr = _id_range_tracker(ds, tmp_path / f"unique{excess}", allow_merging=False)
            patched.setattr(tr, "identify_objects", lambda *a, **k: (object_id_field, None, None))
            patched.setattr(_Tracker, "refresh_dask_graph", stop)
            try:
                tr.track_objects(None)
            except (_Stop, ValueError) as e:
                raised = e
            else:
                raised = None
            del tr
        if excess:
            assert isinstance(raised, ValueError), f"a total of int32 max was not rejected: {raised!r}"
            assert "int32 ID range" in str(raised), str(raised)
        else:
            assert isinstance(raised, _Stop), f"a total of int32 max - 1 was not accepted: {raised!r}"
            unique = raised.args[0]
            dtype, largest = unique.dtype, int(unique.max())
            del unique, raised
            assert dtype == np.int32, dtype
            assert largest == limit - 1, largest

    def merge_temporaries(excess):
        n_time = 2
        largest = limit - 1 - 2 * n_time * id_stride + excess
        # Slice 1's label is offset by slice 0's maximum (1) when IDs are made unique across time.
        object_id_field = field(1, largest - 1)

        def stop(*args, **kwargs):
            raise _Stop

        with monkeypatch.context() as patched:
            tr = _id_range_tracker(ds, tmp_path / f"merge{excess}", allow_merging=True)
            patched.setattr(tr, "identify_objects", lambda *a, **k: (object_id_field, None, None))
            patched.setattr(_overlap, "find_overlapping_objects", stop)
            try:
                tr.track_objects(None)
            except (_Stop, ValueError) as e:
                raised = e
            else:
                raised = None
            del tr
        if excess:
            assert isinstance(raised, ValueError), f"a temporary range past int32 was not rejected: {raised!r}"
            assert "int32 ID range" in str(raised), str(raised)
        else:
            assert isinstance(raised, _Stop), f"the largest fitting temporary range was not accepted: {raised!r}"

    failures = []
    check(failures, "unique_offsets_largest_accepted", lambda: unique_offsets(0))
    check(failures, "unique_offsets_first_rejected", lambda: unique_offsets(1))
    check(failures, "merge_temporaries_largest_accepted", lambda: merge_temporaries(0))
    check(failures, "merge_temporaries_first_rejected", lambda: merge_temporaries(1))
    assert_no_failures(failures)


# -- real parent centroids (in-process client) ----------------------------------------------------

# Module-level spies: dask pickles a test-local closure by value, so its appends would land in a copy.
# Functions and lists at module level are pickled by reference and stay shared.
_CALLS: list = []


def _spy_nn(child_mask, parent_masks, parent_centroids, *args, **kwargs):
    n_parents = int(parent_masks[parent_masks < 255].max()) + 1
    _CALLS.append((len(parent_centroids), n_parents))
    return partition_nn_unstructured_optimised(child_mask, parent_masks, parent_centroids, *args, **kwargs)


def _spy_centroid(child_mask, parent_centroids, child_ids, lat, lon):
    _CALLS.append((len(parent_centroids), len(child_ids)))
    return partition_centroid_unstructured(child_mask, parent_centroids, child_ids, lat, lon)


# (N_events_final, total_merges, ID_field.sum()) of the nn_partitioning=False run at time chunk 4: the only
# unstructured run that partitions children by centroid with merges.
CENTROID_RUN_PINS = (11, 9, 15291)


def test_merge_loop_passes_only_real_parent_centroids(inprocess_client, monkeypatch, tmp_path):
    """The merge loop collects parents into fixed-width ``MAX_PARENTS`` arrays whose unused centroid rows
    hold a far-off sentinel. Both unstructured partitioners treat every row as a parent, so the loop must
    pass only the real ones: on a sphere a sentinel is just another point and can win."""
    assert on_client(inprocess_client), "not computing on the in-process client"
    monkeypatch.setattr(merge_split, "partition_nn_unstructured_optimised", _spy_nn)
    monkeypatch.setattr(merge_split, "partition_centroid_unstructured", _spy_centroid)
    failures = []
    for nn_partitioning in (True, False):

        def run(nn_partitioning=nn_partitioning):
            _CALLS.clear()
            data = open_unstructured_merging()
            tr = marEx.tracker(
                data.extreme_events.chunk({"time": 4, "ncells": -1}),
                data.mask,
                neighbours=data.neighbours,
                cell_areas=data.cell_areas,
                temp_dir=str(tmp_path / f"nn{nn_partitioning}"),
                **dict(UNSTRUCTURED_MERGING_KWARGS, nn_partitioning=nn_partitioning),
            )
            events = tr.run().compute()
            del tr
            calls = list(_CALLS)
            assert calls, "the merge loop never partitioned a child: the check is vacuous"
            wrong = [c for c in calls if c[0] != c[1]]
            assert not wrong, f"{len(wrong)} of {len(calls)} partitions got padded centroids, e.g. (rows, parents) = {wrong[0]}"
            if not nn_partitioning:
                got = (int(events.attrs["N_events_final"]), int(events.attrs["total_merges"]), int(events.ID_field.sum()))
                assert got == CENTROID_RUN_PINS, f"(events, merges, ID_field sum) moved: {got} != {CENTROID_RUN_PINS}"

        check(failures, f"nn_partitioning={nn_partitioning}", run)
    assert_no_failures(failures)


# -- kernel executions per dirty chunk -------------------------------------------------------------


class _CountingDask:
    """Stand-in for the ``dask`` module as ``merge_split`` sees it.

    Only ``merge_split``'s own name is replaced, so nothing else in the process is touched. Every
    ``dask.compute`` call is recorded as one batch, and every ``process_chunk`` task records the chunk's
    first timestep twice: on the client when the task is BUILT, and through a file when the task EXECUTES.
    ``process_chunk`` is a closure shipped to the workers by value, so a list-held execution counter would
    be pickled and stay empty on the client; a path pickles fine and the file is shared under threads and
    processes.
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


def _merging_run(temp_dir, time_chunk):
    data = open_unstructured_merging().chunk({"time": time_chunk})
    tracker = marEx.tracker(
        data.extreme_events,
        data.mask,
        temp_dir=str(temp_dir),
        neighbours=data.neighbours,
        cell_areas=data.cell_areas,
        **UNSTRUCTURED_MERGING_KWARGS,
    )
    events, merges = tracker.run(return_merges=True)
    return events.compute(), merges.compute()


@pytest.mark.slow
def test_kernel_runs_once_per_dirty_chunk(client, tmp_path, monkeypatch):
    """Every submitted chunk task executes exactly once.

    The merge loop runs ``process_chunk`` as one ``dask.delayed`` task per DIRTY time chunk, in two batches
    per iteration (even chunks, then odd chunks, so no chunk reads a boundary slice that its own batch
    writes). Iteration 1 runs every chunk; a later iteration runs only chunks whose predecessor's last slice
    or forwarded queue changed. The store is one 100-step chunk on disk: at 20 chunks of 5 a boundary
    changes and the loop reruns two chunks in iteration 2 (observed 2026-09-13: batches 10, 10, 1, 1); at 10
    or 20 steps it converges in one iteration and the rerun path would go unexercised.
    """
    assert on_client(client)
    time_chunk = 5
    n_time = open_unstructured_merging().sizes["time"]
    chunk_starts = list(range(0, n_time, time_chunk))
    n_chunks = len(chunk_starts)

    counting = _CountingDask(merge_split.dask, tmp_path / "kernel.count")
    with monkeypatch.context() as patched:
        patched.setattr(merge_split, "dask", counting)
        chunked_events, chunked_merges = _merging_run(tmp_path / "chunked", time_chunk)
    batches = counting.batches
    executed = counting.executed()
    built = [t0 for batch in batches for t0 in batch]

    # Visible under `pytest -s`: the shape of the loop on this store.
    print(f"\nkernel batches (chunk first timesteps): {batches}; executions: {len(executed)}")

    # Control for the control: a patch that reached nothing would make every bound vacuous.
    assert built, "no process_chunk task was built through the patched dask; the patch is not reaching the loop"
    assert chunked_merges.sizes.get("merge_ID", 0) > 0, "the store produced no merge, so the kernel was never exercised"

    # Every task that was built ran exactly once: no re-execution, no dropped task.
    assert sorted(executed) == sorted(built), f"executions {sorted(executed)} != submitted tasks {sorted(built)}"

    # Iteration 1 is the first two batches and covers every chunk once, by parity.
    assert len(batches) >= 2, f"expected at least the two parity batches of iteration 1, saw {batches}"
    assert batches[0] == chunk_starts[0::2] and batches[1] == chunk_starts[1::2], batches[:2]

    # No batch runs a chunk twice, and no batch mixes parities (a chunk would then read a boundary slice
    # written by its own batch).
    for batch in batches:
        assert len(batch) == len(set(batch)), f"a chunk ran twice in one batch: {batch}"
        parities = {chunk_starts.index(t0) % 2 for t0 in batch}
        assert len(parities) == 1, f"batch mixes chunk parities: {batch}"

    # The rerun path must be exercised, and a rerun is a strict subset of the chunks.
    assert len(batches) > 2, f"the loop converged in one iteration, so no rerun was exercised: {batches}"
    assert all(len(batch) < n_chunks // 2 + 1 for batch in batches[2:]), batches

    # The loop is not a fixed n_iterations x n_chunks sweep: beyond iteration 1 it only reruns chunks
    # downstream of a change, so the total is bounded by a serial cascade.
    assert len(built) <= n_chunks * n_chunks, f"{len(built)} kernel runs for {n_chunks} chunks"

    # Chunking must not move the result on the store the count ran on.
    single_events, single_merges = _merging_run(tmp_path / "single", n_time)
    assert int(chunked_merges.sizes["merge_ID"]) == int(single_merges.sizes["merge_ID"])
    assert same_partition(np.asarray(chunked_events.ID_field.values), np.asarray(single_events.ID_field.values))


# -- the region writer ------------------------------------------------------------------------------


def _field(n_time=8, ny=4, nx=5, chunk=2):
    """A chunked int32 field with coords, shaped like object_id_field_unique."""
    data = dsa.from_array(np.arange(n_time * ny * nx, dtype=np.int32).reshape(n_time, ny, nx), chunks=(chunk, ny, nx))
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(n_time), "lat": np.linspace(-10, 10, ny), "lon": np.linspace(0, 40, nx)},
        name="ID_field",
    )


WRITER_ROUNDTRIPS = [
    ("in_order", 8, 2, [(0, 2), (2, 4), (4, 6), (6, 8)]),
    ("out_of_order", 8, 2, [(4, 6), (0, 2), (6, 8), (2, 4)]),
    ("ragged_final_chunk", 7, 3, [(0, 3), (3, 6), (6, 7)]),
    ("one_region", 8, 2, [(0, 8)]),
]


def test_region_writer_roundtrip(tmp_path):
    """Writing every region (in any order, with a ragged final chunk) then finalising returns exactly what was
    written: values, dtype, dims and coords. The result is read lazily from disk, and no region write probes
    for consolidated metadata (the staging store is unconsolidated)."""
    failures = []
    for case, n_time, chunk, regions in WRITER_ROUNDTRIPS:

        def roundtrip(case=case, n_time=n_time, chunk=chunk, regions=regions):
            template = _field(n_time=n_time, chunk=chunk)
            writer = ObjectIDRegionWriter(template, tmp_path / f"{case}.zarr", "time")
            source = template.compute()
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=".*consolidated metadata.*")
                for start, end in regions:
                    writer.write(start, end, source.isel(time=slice(start, end)))
                out = writer.finalise()
            assert out.chunks is not None, f"{case}: finalise() returned a materialised array"
            # assert_identical compares values, dims, coords and attrs, but not the dtype.
            assert out.dtype == np.int32, f"{case}: finalise() returned {out.dtype}, not int32"
            xr.testing.assert_identical(out.compute(), source)

        check(failures, case, roundtrip)
    assert_no_failures(failures)


def test_region_writer_rejects_bad_coverage(tmp_path):
    """Finalising raises when no region was written, when a region is missing (a gap would silently leave
    zarr fill values), and when regions overlap."""
    failures = []
    for case, regions, match in (
        ("no_write", [], "no regions"),
        ("gap", [(0, 2), (4, 6), (6, 8)], "gap"),
        ("overlap", [(0, 4), (2, 6), (6, 8)], "overlap"),
    ):

        def rejected(case=case, regions=regions, match=match):
            template = _field()
            writer = ObjectIDRegionWriter(template, tmp_path / f"{case}.zarr", "time")
            source = template.compute()
            for start, end in regions:
                writer.write(start, end, source.isel(time=slice(start, end)))
            with pytest.raises(RuntimeError, match=match):
                writer.finalise()

        check(failures, case, rejected)
    assert_no_failures(failures)


def test_unstructured_consolidation_reports_its_renames():
    """Parent 1 overlaps children 2 and 3; 3 is renamed onto 2 and the pair is handed back."""
    prev = np.array([1, 1, 1, 1, 0, 0], dtype=np.int32)
    cur = np.array([2, 2, 3, 3, 0, 0], dtype=np.int32)
    renames = []
    merge_split._consolidate_slice(prev, cur, np.ones(6, dtype=np.float32), 0.5, renames)
    assert renames == [(3, 2)]
    np.testing.assert_array_equal(cur, [2, 2, 2, 2, 0, 0])


def test_unstructured_loop_hands_its_renames_to_the_clusterer(client, tmp_path, monkeypatch):
    """One object separates into two: the loop renames one child onto the other, and the clusterer receives that pair."""
    seen = []
    real = merge_split._add_event_ids

    def spy(merge_events, id_to_event):
        seen.append((merge_events["_renamed_from"].values.copy(), merge_events["_renamed_to"].values.copy()))
        real(merge_events, id_to_event)

    monkeypatch.setattr(merge_split, "_add_event_ids", spy)
    rows = [[1] * 9, [1, 1, 1, 0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1]]
    _, objects, _ = _track_chain(tmp_path, serial=False, rows=rows)
    assert len(seen) == 1
    renamed, kept = seen[0]
    assert renamed.size > 0
    final_ids = np.unique(np.asarray(objects))
    assert not np.isin(renamed, final_ids).any()
    assert np.isin(kept, final_ids).all()


# --------------------------------------------------------------------------------------
# Split records
# --------------------------------------------------------------------------------------


def _chain_neighbours(n):
    """``(3, n)`` neighbour table of a chain of ``n`` cells, -1 where absent."""
    cells = np.arange(n, dtype=np.int32)
    left, right = cells - 1, np.where(cells == n - 1, -1, cells + 1)
    return np.stack([left, right, np.full(n, -1, dtype=np.int32)])


@pytest.mark.parametrize(
    "prev, cur, expected",
    [
        pytest.param([1] * 8, [2, 2, 2, 2, 3, 3, 3, 3], [], id="adjacent_pieces_of_a_partition"),
        pytest.param([1, 1, 1, 0, 0, 0, 1, 1, 1], [2, 2, 2, 0, 0, 0, 3, 3, 3], [], id="parent_holding_two_pieces"),
        pytest.param([1, 1, 1, 1, 1, 0, 0, 0, 1, 1], [2, 2, 0, 3, 3, 0, 0, 0, 4, 4], [(1, [2, 3, 4])], id="one_piece_separates"),
        pytest.param(
            [6, 6, 6, 5, 5, 5, 5, 5, 5, 5], [1, 0, 2, 2, 0, 0, 0, 3, 3, 3], [(5, [2, 3]), (6, [1, 2])], id="kept_id_renamed_later"
        ),
    ],
)
def test_unstructured_split_rule(prev, cur, expected):
    prev, cur = np.array(prev, dtype=np.int32), np.array(cur, dtype=np.int32)
    splits = []
    area = np.ones(prev.size, dtype=np.float32)
    merge_split._consolidate_slice(prev, cur, area, 0.5, None, _chain_neighbours(prev.size), splits)
    assert splits == expected


def test_no_unstructured_split_without_the_neighbours():
    prev, cur = np.ones(5, dtype=np.int32), np.array([2, 0, 0, 0, 3], dtype=np.int32)
    splits = []
    merge_split._consolidate_slice(prev, cur, np.ones(5, dtype=np.float32), 0.5, None, None, splits)
    assert splits == []


def _gridded_separates(parent, child, wrap_x, active=None):
    active = child if active is None else active
    return _overlap.object_separates_gridded(parent, child, _overlap.water_pieces_gridded(active, wrap_x), wrap_x)


def test_gridded_split_rule_uses_eight_connectivity_and_the_seam():
    parent = np.ones((3, 6), dtype=bool)
    diagonal = np.zeros((3, 6), dtype=bool)
    diagonal[0, 1] = diagonal[1, 2] = True
    assert not _gridded_separates(parent, diagonal, False)

    across_seam = np.zeros((3, 6), dtype=bool)
    across_seam[1, 0] = across_seam[1, 5] = True
    assert _gridded_separates(parent, across_seam, False)
    assert not _gridded_separates(parent, across_seam, True)

    two_pieces = np.zeros((3, 6), dtype=bool)
    two_pieces[0, 0] = two_pieces[0, 3] = True
    assert _gridded_separates(parent, two_pieces, True)
    held = two_pieces.copy()
    parent_in_two = two_pieces.copy()
    assert not _gridded_separates(parent_in_two, held, True)
    assert not _gridded_separates(np.zeros((3, 6), dtype=bool), np.zeros((3, 6), dtype=bool), True)


def test_gridded_pieces_joined_by_another_object_are_not_a_split():
    """The other object's row lies outside the parent's and the child's bounding box."""
    parent = np.zeros((3, 6), dtype=bool)
    parent[0] = True
    child = np.zeros((3, 6), dtype=bool)
    child[0, 0] = child[0, 5] = True
    active = child.copy()
    active[1] = True
    assert not _gridded_separates(parent, child, False, active)
    assert _gridded_separates(parent, child, False)


def test_gridded_box_does_not_wrap_a_parent_away_from_the_seam():
    parent = np.zeros((1, 20), dtype=bool)
    parent[0, [5, 6, 9, 10]] = True
    assert not _gridded_separates(parent, parent, True)


def test_consolidation_labels_gridded_water_once(monkeypatch):
    """Two parents separate in one slice; the water is labelled from all active cells, once."""
    ids = np.array([[[1, 1, 1, 1, 1, 0, 4, 4, 4, 4, 4]], [[2, 2, 0, 3, 3, 0, 5, 5, 0, 6, 6]]], dtype=np.int32)
    coords = {"lat": [0.0], "lon": np.arange(11.0)}
    prev, cur = (xr.DataArray(ids[t], dims=("lat", "lon"), coords=coords) for t in range(2))
    calls = []
    water_pieces = _overlap.water_pieces_gridded
    monkeypatch.setattr(
        _overlap, "water_pieces_gridded", lambda active, wrap_x: calls.append(active.sum()) or water_pieces(active, wrap_x)
    )
    cell_area = xr.ones_like(cur, dtype=np.float64)
    kwargs = dict(
        unstructured_grid=False,
        lat=cur.lat,
        lon=cur.lon,
        cell_area=cell_area,
        timedim="time",
        regional_mode=True,
        ydim="lat",
        xdim="lon",
    )
    props = [calculate_object_properties(field, properties=["area", "centroid"], **kwargs) for field in (prev, cur)]
    store = ObjectPropsStore.from_dataset(xr.concat(props, dim="ID"))
    splits = []
    _overlap.consolidate_object_ids(
        prev, cur, store, 7, False, cell_area, 0.5, cur.lat, cur.lon, "time", True, "lat", "lon", splits=splits
    )
    assert [(t, p) for t, p, _ in splits] == [(7, 1), (7, 4)]
    assert calls == [8]


def _consolidate_row(prev_row, cur_row, regional, threshold=0.5):
    """``consolidate_object_ids`` on one row of cells, returning its split records."""
    ids = np.array([[prev_row], [cur_row]], dtype=np.int32)
    coords = {"lat": [0.0], "lon": np.arange(float(len(prev_row)))}
    prev, cur = (xr.DataArray(ids[t], dims=("lat", "lon"), coords=coords) for t in range(2))
    cell_area = xr.ones_like(cur, dtype=np.float64)
    kwargs = dict(
        unstructured_grid=False,
        lat=cur.lat,
        lon=cur.lon,
        cell_area=cell_area,
        timedim="time",
        regional_mode=regional,
        ydim="lat",
        xdim="lon",
    )
    props = [calculate_object_properties(field, properties=["area", "centroid"], **kwargs) for field in (prev, cur)]
    store = ObjectPropsStore.from_dataset(xr.concat(props, dim="ID"))
    splits = []
    _overlap.consolidate_object_ids(
        prev, cur, store, 7, False, cell_area, threshold, cur.lat, cur.lon, "time", regional, "lat", "lon", splits=splits
    )
    return splits


WATER_ACROSS_SEAM = ([1] * 20, [2] * 9 + [0] + [3] * 10)
PARENT_ACROSS_SEAM = ([1, 1, 1] + [0] * 15 + [1, 1], [0, 2, 2] + [0] * 15 + [3, 3])


@pytest.mark.parametrize(
    "rows, regional, expected",
    [
        (WATER_ACROSS_SEAM, False, []),
        (WATER_ACROSS_SEAM, True, [(7, 1, [2, 3])]),
        (PARENT_ACROSS_SEAM, False, [(7, 1, [2, 3])]),
        (PARENT_ACROSS_SEAM, True, []),
    ],
    ids=["water-wrapped", "water-regional", "parent-wrapped", "parent-regional"],
)
def test_consolidation_applies_the_seam_to_water_and_parent(rows, regional, expected):
    """A global grid joins both the water and the parent across the seam; a regional grid joins neither."""
    assert _consolidate_row(*rows, regional) == expected


def test_gridded_parent_piece_uses_eight_connectivity():
    """The parent's two runs touch only at a corner, and each child cell lies under a different run."""
    parent = np.zeros((2, 6), dtype=bool)
    parent[0, [0, 1]] = parent[1, [2, 3]] = True
    child = np.zeros((2, 6), dtype=bool)
    child[0, 0] = child[1, 3] = True
    assert _gridded_separates(parent, child, False)


def test_consolidation_follows_a_kept_id_renamed_through_two_later_parents():
    """Parent 1 keeps child 4, parent 2 renames 4 onto 5, parent 3 renames 5 onto 6: the test of parent 1 must reach 6."""
    prev = [3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    cur = [4, 4, 4, 0, 5, 5, 5, 0, 6, 6, 6, 0, 7, 7, 7]
    assert _consolidate_row(prev, cur, True, threshold=0.25) == [(7, 1, [6, 7]), (7, 2, [5, 6]), (7, 3, [4, 5])]


def test_consolidation_follows_a_kept_id_renamed_by_a_later_parent():
    """Parent 1 keeps child 4, then parent 2 renames 4 onto 3; the test of parent 1 must look at 3."""
    prev = [2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    cur = [4, 4, 4, 3, 4, 4, 4, 4, 5, 5, 5, 0, 5, 5, 5, 5]
    assert _consolidate_row(prev, cur, True, threshold=0.25) == [(7, 1, [4, 5])]


def _bridged_mesh():
    """A chain of seven cells, with cells 7, 8, 9 forming a path from cell 1 to cell 5 around it."""
    edges = [(i, i + 1) for i in range(6)] + [(1, 7), (7, 8), (8, 9), (9, 5)]
    table = [[] for _ in range(10)]
    for a, b in edges:
        table[a].append(b)
        table[b].append(a)
    return np.array([row + [-1] * (3 - len(row)) for row in table], dtype=np.int32).T


def test_unstructured_pieces_joined_by_another_object_are_not_a_split():
    prev = np.array([1] * 7 + [0] * 3, dtype=np.int32)
    area = np.ones(10, dtype=np.float32)
    bridged = np.array([2, 2, 0, 0, 0, 3, 3, 7, 7, 7], dtype=np.int32)
    splits = []
    merge_split._consolidate_slice(prev, bridged, area, 0.5, None, _bridged_mesh(), splits)
    assert splits == []
    open_water = np.array([2, 2, 0, 0, 0, 3, 3, 0, 0, 0], dtype=np.int32)
    merge_split._consolidate_slice(prev, open_water, area, 0.5, None, _bridged_mesh(), splits)
    assert splits == [(1, [2, 3])]


def test_unstructured_water_is_labelled_once_per_slice(monkeypatch):
    prev = np.array([1, 1, 1, 0, 4, 4, 4], dtype=np.int32)
    cur = np.array([2, 0, 3, 0, 5, 0, 6], dtype=np.int32)
    calls = []
    pieces = _overlap._cell_pieces
    monkeypatch.setattr(_overlap, "_cell_pieces", lambda cells, nb: calls.append(cells.size) or pieces(cells, nb))
    splits = []
    merge_split._consolidate_slice(prev, cur, np.ones(7, dtype=np.float32), 0.5, None, _chain_neighbours(7), splits)
    assert [p for p, _ in splits] == [1, 4]
    assert calls.count(4) == 1  # the four active cells; each parent's three cells are labelled apart


def _chain_merges(temp_dir, serial, rows, time_chunk=None):
    da, mask, nb, area = _chain_inputs(rows)
    if time_chunk is not None:
        da = da.chunk({"time": time_chunk})

    def run():
        tracker = marEx.tracker(da, mask, temp_dir=str(temp_dir), neighbours=nb, cell_areas=area, **CHAIN_KWARGS)
        if serial:
            tracker.split_and_merge_objects_parallel = tracker.split_and_merge_objects
        events, merges = tracker.run(return_merges=True)
        return events.compute(), merges.compute()

    return released(run, "the tracker run")


@pytest.mark.parametrize("serial", [False, True], ids=["parallel", "serial"])
def test_a_split_that_persists_is_recorded_once(client, tmp_path, serial):
    rows = [[1] * 9, [1, 1, 1, 0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1]]
    events, merges = _chain_merges(tmp_path, serial, rows)
    assert merges.sizes["split_ID"] == 1
    np.testing.assert_array_equal(merges["split_time"].values, events["time"].values[[1]])
    np.testing.assert_array_equal(merges["n_split_children"].values, [2])
    np.testing.assert_array_equal(merges["split_event_IDs"].values, [1])
    assert merges["split_parent_IDs"].dtype == np.int32 and merges["split_child_IDs"].dtype == np.int32
    assert events.attrs["total_splits"] == 1
    assert "_renamed_from" not in merges


@pytest.mark.parametrize("serial", [False, True], ids=["parallel", "serial"])
def test_two_events_that_merge_and_separate_are_not_a_split(client, tmp_path, serial):
    rows = [[1, 1, 1, 0, 0, 0, 1, 1, 1], [1] * 9, [1, 1, 1, 0, 0, 0, 1, 1, 1]]
    events, merges = _chain_merges(tmp_path, serial, rows)
    assert merges.sizes["merge_ID"] > 0
    assert merges.sizes["split_ID"] == 0
    assert merges["split_time"].dtype == events["time"].dtype
    assert events.attrs["total_splits"] == 0


def test_a_minted_parent_id_in_a_split_row_resolves_to_its_event(client, tmp_path):
    """A parent minted by a merge is renumbered in the split row, and the row names the event that holds it."""
    rows = [
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1] * 16,
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    ]
    _, merges = _chain_merges(tmp_path, False, rows)
    assert merges.sizes["split_ID"] == 1
    assert int(merges["split_parent_IDs"].values[0]) in merges["child_IDs"].values, "the parent is the ID the merge minted"
    assert (merges["split_event_IDs"].values > 0).all(), "the split row names an event"


@pytest.mark.parametrize("time_chunk", [1, 2, 5])
def test_split_records_survive_chunk_reruns(client, tmp_path, time_chunk):
    """A merge at t1 changes the next chunk's boundary, so later chunks run again: their records replace the first run's."""
    rows = [[1, 1, 1, 0, 0, 0, 1, 1, 1], [1] * 9, [1] * 9, [1, 0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 1, 1]]
    _, reference = _chain_merges(tmp_path / "whole", False, rows)
    _, merges = _chain_merges(tmp_path / f"chunk{time_chunk}", False, rows, time_chunk=time_chunk)
    assert reference.sizes["split_ID"] == 1
    assert (reference["split_event_IDs"].values > 0).all(), "the split row must resolve to an event"
    for name in ("split_time", "split_parent_IDs", "split_child_IDs", "n_split_children", "split_event_IDs"):
        np.testing.assert_array_equal(merges[name].values, reference[name].values, err_msg=name)
