"""The unstructured merge loop must not depend on the input time chunking.

``split_and_merge_objects_parallel`` processes each time chunk from the iteration's input field and
defers cascades that cross a chunk boundary. Before the 2026-09 repair a deferred boundary object was
partitioned against stale parents and never repaired, so lineages that were only ever bridged for a
few days fused into one event, and the result changed with the chunk width (Q8, D-082).

The oracle is the serial (gridded) path. Since the kernel consolidates split pieces the way the
serial path does (D-087), the single-chunk parallel run matches it on merge counts, events AND object
IDs, and both paths are independent of the time chunking, including one-timestep chunks.
"""

from __future__ import annotations

import numpy as np
import pytest

import marEx

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


def _track(binary_sq, nrow, ncol, time_chunk, temp_dir, serial=False):
    """Run the unstructured tracker; return (event field, pre-cluster object field, n_merges)."""
    da, mask, neighbours, cell_areas = tracker_inputs(binary_sq, nrow, ncol, time_chunk)
    tr = marEx.tracker(da, mask, temp_dir=str(temp_dir), neighbours=neighbours, cell_areas=cell_areas, **TRACKER_KWARGS)
    if serial:
        tr.split_and_merge_objects_parallel = tr.split_and_merge_objects
    captured = {}
    orig = tr.cluster_rename_objects_and_props

    def _capture(field, props, overlaps, merges):
        captured["objects"] = np.asarray(field.compute().values)
        return orig(field, props, overlaps, merges)

    tr.cluster_rename_objects_and_props = _capture
    events, merges = tr.run(return_merges=True)
    return np.asarray(events.ID_field.values), captured["objects"], int(merges.sizes.get("merge_ID", 0))


def _scenario(name):
    if name.startswith("seed"):
        return random_scenario(int(name[4:]))
    return scenarios()[name]


CASES = ["S1", "S2", "S3", "S4", "seed5", "seed15"]
WIDTHS = [5, 4, 8]


@pytest.fixture(scope="module")
def reference(dask_client_unstructured, tmp_path_factory):
    """Single-chunk runs of every case (the oracle)."""
    out = {}
    for name in CASES:
        sc = _scenario(name)
        binary, lineage, bridge_day = render(sc["nrow"], sc["ncol"], NT, sc["blobs"], sc["bridges"])
        out[name] = (sc, binary, lineage, bridge_day, _track(binary, sc["nrow"], sc["ncol"], NT, tmp_path_factory.mktemp(name)))
    return out


@pytest.mark.parametrize("name", CASES)
def test_single_chunk_recovers_the_truth(reference, name):
    sc, binary, lineage, bridge_day, (events, objects, n_merges) = reference[name]
    idf_sq = cells_to_squares(events, sc["nrow"], sc["ncol"])
    assert fused_events(idf_sq, lineage, bridge_day) == {}
    assert len(np.unique(events[events > 0])) == sc["truth"]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("name", CASES)
def test_chunked_run_matches_single_chunk(reference, name, width, tmp_path):
    sc, binary, lineage, bridge_day, (ref_events, ref_objects, ref_merges) = reference[name]
    events, objects, n_merges = _track(binary, sc["nrow"], sc["ncol"], width, tmp_path)
    assert n_merges == ref_merges
    assert same_partition(objects, ref_objects), "object-level partition differs from the single-chunk run"
    assert same_partition(events, ref_events), "event partition differs from the single-chunk run"


@pytest.mark.parametrize("name", CASES)
def test_serial_path_is_the_oracle(reference, name, tmp_path):
    sc, binary, lineage, bridge_day, (ref_events, ref_objects, ref_merges) = reference[name]
    events, objects, n_merges = _track(binary, sc["nrow"], sc["ncol"], NT, tmp_path, serial=True)
    assert n_merges == ref_merges
    assert same_partition(objects, ref_objects), "object-level partition differs from the serial path"
    assert np.array_equal(events, ref_events), "a compaction retry changed event IDs"


@pytest.mark.parametrize("width", [3, 1])
@pytest.mark.parametrize("name", ["S4", "seed0", "seed6"])
def test_serial_path_is_chunk_invariant(name, width, dask_client_unstructured, tmp_path):
    """(3, ..., 3, 1) and all-ones. The serial path skipped end-of-chunk consolidation for a
    one-timestep chunk; these three cases changed object IDs (S4 also merges, 20 vs 0) before
    the fix (D-087)."""
    sc = _scenario(name)
    binary, _, _ = render(sc["nrow"], sc["ncol"], NT, sc["blobs"], sc["bridges"])
    (tmp_path / "one").mkdir()
    (tmp_path / "chunked").mkdir()
    _, ref_objects, ref_merges = _track(binary, sc["nrow"], sc["ncol"], NT, tmp_path / "one", serial=True)
    _, objects, n_merges = _track(binary, sc["nrow"], sc["ncol"], width, tmp_path / "chunked", serial=True)
    assert n_merges == ref_merges
    assert same_partition(objects, ref_objects)


def _run_dataset(binary_sq, nrow, ncol, time_chunks, temp_dir, **extra):
    da, mask, neighbours, cell_areas = tracker_inputs(binary_sq, nrow, ncol, NT)
    da = da.chunk({"time": time_chunks})
    tr = marEx.tracker(
        da, mask, temp_dir=str(temp_dir), neighbours=neighbours, cell_areas=cell_areas, **{**TRACKER_KWARGS, **extra}
    )
    events, merges = tr.run(return_merges=True)
    return events, merges


def test_size_one_tail_chunk_with_merges(reference, tmp_path):
    """(3, ..., 3, 1): the last chunk holds one timestep. Crashed before 2026-09 (D-077)."""
    sc, binary, lineage, bridge_day, (ref_events, _, ref_merges) = reference["S3"]
    assert NT % 3 == 1
    events, merges = _run_dataset(binary, sc["nrow"], sc["ncol"], 3, tmp_path)
    assert int(merges.sizes["merge_ID"]) == ref_merges
    assert same_partition(np.asarray(events.ID_field.values), ref_events)


def test_ragged_time_chunks_are_retiled(reference, tmp_path):
    """An interior-ragged chunking is re-tiled once at construction; the result is unchanged."""
    sc, binary, lineage, bridge_day, (ref_events, _, ref_merges) = reference["S3"]
    ragged = (2, 5, 5, 5, 5, 5, 5, 5, 3)
    assert sum(ragged) == NT
    da, mask, neighbours, cell_areas = tracker_inputs(binary, sc["nrow"], sc["ncol"], NT)
    tr = marEx.tracker(
        da.chunk({"time": ragged}), mask, temp_dir=str(tmp_path), neighbours=neighbours, cell_areas=cell_areas, **TRACKER_KWARGS
    )
    assert tr.data_bin.chunks[tr.data_bin.dims.index("time")] == (5, 5, 5, 5, 5, 5, 5, 5)
    assert tr.timechunks == 5
    events, merges = tr.run(return_merges=True)
    assert int(merges.sizes["merge_ID"]) == ref_merges
    assert same_partition(np.asarray(events.ID_field.values), ref_events)


def test_merge_ledger_is_written_positionally(reference, tmp_path):
    """The ledger holds, at (time, ID, sibling), that parent's own event ID (the pre-2026-09
    label-based write produced exactly this); it is written by position so a shared pandas
    index engine can never race (D-079)."""
    sc, binary, lineage, bridge_day, (ref_events, _, ref_merges) = reference["S2"]
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


@pytest.mark.parametrize("name", ["S3", "seed5"])
def test_compaction_survives_a_task_retry(reference, name, tmp_path, monkeypatch):
    """A worker restart makes dask re-run a compaction task whose first attempt already wrote its
    region. Minted IDs used to share a range with the compacted ones, so the retry remapped IDs a
    second time and the run crashed or relabelled cells (D-093). Every region is compacted once up
    front, then the real compute runs over the already-compacted store."""
    import dask

    from marEx.track import merge_split

    sc, binary, lineage, bridge_day, _ = reference[name]
    (tmp_path / "once").mkdir()
    (tmp_path / "retried").mkdir()
    ref_events, ref_objects, ref_merges = _track(binary, sc["nrow"], sc["ncol"], 5, tmp_path / "once")
    real_compute = dask.compute
    first_attempts = []

    def _compute_with_retry(*args, **kwargs):
        regions = [a for a in args if str(getattr(a, "key", "")).startswith("_compact_region")]
        if regions:
            first_attempts.extend(real_compute(*regions, scheduler="synchronous"))
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(merge_split.dask, "compute", _compute_with_retry)
    events, objects, n_merges = _track(binary, sc["nrow"], sc["ncol"], 5, tmp_path / "retried")
    monkeypatch.undo()

    assert first_attempts, "no compaction task was re-executed, so nothing was tested"
    assert n_merges == ref_merges
    assert np.array_equal(objects, ref_objects), "a compaction retry changed object IDs"
    assert np.array_equal(events, ref_events), "a compaction retry changed event IDs"
