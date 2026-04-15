"""
Unit tests for marEx.genealogy — the merge/split genealogy post-processing module.

Exercises build_genealogy on a hand-constructed synthetic tracker output that
deliberately contains one Partitioned Merge (PM), one Absorptive Merge (AM),
and one Partitioned Split (PS). Asserts that the derived edge list matches
expectations and that per-event, family, and global statistics are consistent.
"""

import numpy as np
import pandas as pd
import xarray as xr

import marEx


def _make_synthetic_inputs():
    """Build one (events_ds, genealogy_ds) pair with 6 events and 6 timesteps.

    Layout:
        e1 + e2 --PM--> e3   at t=2
        e4 --AM--> e5        at t=3 (e4 dies touching persistent e5)
        e5 ‖-PS-‖ e6         separating between t=4 and t=5
    """
    times = pd.date_range("2020-01-01", periods=6, freq="D").values
    n_times = len(times)
    event_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    n_events = len(event_ids)

    # Presence matrix (time, ID).
    presence = np.zeros((n_times, n_events), dtype=bool)
    #      e1 e2 e3 e4 e5 e6
    presence[0] = [1, 1, 0, 1, 1, 1]
    presence[1] = [1, 1, 0, 1, 1, 1]
    presence[2] = [1, 1, 1, 1, 1, 1]  # PM at t=2 produces e3
    presence[3] = [0, 0, 1, 1, 1, 1]  # e1, e2 ended
    presence[4] = [0, 0, 1, 0, 1, 1]  # e4 ended (AM at t=3)
    presence[5] = [0, 0, 1, 0, 1, 1]

    # Area (time, ID) — positive only where present.
    area = np.where(presence, 10.0, 0.0).astype(np.float32)

    # time_start / time_end (per event)
    time_start = np.array([times[0], times[0], times[2], times[0], times[0], times[0]], dtype="datetime64[ns]")
    time_end = np.array([times[2], times[2], times[5], times[3], times[5], times[5]], dtype="datetime64[ns]")

    # Stub ID_field — not consumed by build_genealogy, but required by validation.
    id_field = np.zeros((n_times, 2, 2), dtype=np.int32)

    events_ds = xr.Dataset(
        data_vars={
            "ID_field": (("time", "lat", "lon"), id_field),
            "area": (("time", "ID"), area),
            "presence": (("time", "ID"), presence),
            "time_start": (("ID",), time_start),
            "time_end": (("ID",), time_end),
        },
        coords={
            "time": times,
            "ID": event_ids,
            "lat": np.array([0.0, 1.0], dtype=np.float32),
            "lon": np.array([0.0, 1.0], dtype=np.float32),
        },
    )

    # --- Genealogy dataset ---
    # PM: one merge record, parents [1, 2], child [3] at t=2.
    parent_IDs = np.array([[1, 2]], dtype=np.int32)
    child_IDs = np.array([[3]], dtype=np.int32)
    overlap_areas = np.array([[10.0, 15.0]], dtype=np.float32)
    merge_time = np.array([times[2]], dtype="datetime64[ns]")
    n_parents = np.array([2], dtype=np.int32)
    n_children = np.array([1], dtype=np.int32)

    # Adjacency edges:
    #   (t=3, 4, 5, len=5)  →  AM: e4 ends here, e5 persists
    #   (t=4, 5, 6, len=3)  →  PS: pair disappears between t=4 and t=5, both persist
    adj_time = np.array([times[3], times[4]], dtype="datetime64[ns]")
    adj_a = np.array([4, 5], dtype=np.int32)
    adj_b = np.array([5, 6], dtype=np.int32)
    adj_len = np.array([5, 3], dtype=np.int32)

    genealogy_ds = xr.Dataset(
        data_vars={
            "parent_IDs": (("merge_ID", "parent_idx"), parent_IDs),
            "child_IDs": (("merge_ID", "child_idx"), child_IDs),
            "overlap_areas": (("merge_ID", "parent_idx"), overlap_areas),
            "merge_time": (("merge_ID",), merge_time),
            "n_parents": (("merge_ID",), n_parents),
            "n_children": (("merge_ID",), n_children),
            "adj_time": (("edge",), adj_time),
            "adj_id_a": (("edge",), adj_a),
            "adj_id_b": (("edge",), adj_b),
            "adj_boundary_length": (("edge",), adj_len),
        },
    )

    return events_ds, genealogy_ds


class TestBuildGenealogyRoundTrip:
    """Round-trip tests against a synthetic dataset with known PM/AM/PS edges."""

    def test_edge_counts(self):
        events_ds, genealogy_ds = _make_synthetic_inputs()
        out = marEx.build_genealogy(events_ds, genealogy_ds)

        edge_type = out["edge_type"].values.astype("U2")
        # 2 PM edges (parent 1→3, parent 2→3), 1 AM edge, 1 PS edge.
        assert (edge_type == "PM").sum() == 2
        assert (edge_type == "AM").sum() == 1
        assert (edge_type == "PS").sum() == 1

    def test_partitioned_merge_edges(self):
        events_ds, genealogy_ds = _make_synthetic_inputs()
        out = marEx.build_genealogy(events_ds, genealogy_ds)

        mask = out["edge_type"].values.astype("U2") == "PM"
        src = out["source_id"].values[mask]
        tgt = out["target_id"].values[mask]
        pm_pairs = set(zip(src.tolist(), tgt.tolist()))
        assert pm_pairs == {(1, 3), (2, 3)}

        # Overlap areas come straight from the merge record.
        overlap = out["overlap_area"].values[mask]
        assert sorted(overlap.tolist()) == [10.0, 15.0]

    def test_absorptive_merge_edge(self):
        events_ds, genealogy_ds = _make_synthetic_inputs()
        out = marEx.build_genealogy(events_ds, genealogy_ds)

        mask = out["edge_type"].values.astype("U2") == "AM"
        src = int(out["source_id"].values[mask][0])
        tgt = int(out["target_id"].values[mask][0])
        # e4 is absorbed into e5.
        assert src == 4
        assert tgt == 5

    def test_partitioned_split_edge(self):
        events_ds, genealogy_ds = _make_synthetic_inputs()
        out = marEx.build_genealogy(events_ds, genealogy_ds)

        mask = out["edge_type"].values.astype("U2") == "PS"
        src = int(out["source_id"].values[mask][0])
        tgt = int(out["target_id"].values[mask][0])
        assert {src, tgt} == {5, 6}

    def test_per_event_counts(self):
        events_ds, genealogy_ds = _make_synthetic_inputs()
        out = marEx.build_genealogy(events_ds, genealogy_ds)

        event_id = out["event_id"].values
        # Build lookup event_id → row index.
        pos = {int(e): i for i, e in enumerate(event_id)}

        n_anc_PM = out["n_ancestors_PM"].values
        n_anc_AM = out["n_ancestors_AM"].values
        n_desc_PM = out["n_descendants_PM"].values
        n_desc_PS = out["n_descendants_PS"].values

        # PM: parents 1, 2 contribute one descendant each; child 3 has two ancestors.
        assert n_desc_PM[pos[1]] == 1
        assert n_desc_PM[pos[2]] == 1
        assert n_anc_PM[pos[3]] == 2

        # AM: e4 is absorbed once (bookkeeping stored on the absorbed event).
        assert n_anc_AM[pos[4]] == 1

        # PS: both endpoints get a "split" descendant tally.
        assert n_desc_PS[pos[5]] == 1
        assert n_desc_PS[pos[6]] == 1

    def test_family_clustering(self):
        events_ds, genealogy_ds = _make_synthetic_inputs()
        out = marEx.build_genealogy(events_ds, genealogy_ds)

        event_id = out["event_id"].values
        family_id = out["family_id"].values
        fid = {int(e): int(f) for e, f in zip(event_id, family_id)}

        # PM connects 1-2-3; AM+PS chains in 4-5-6 via e5.
        assert fid[1] == fid[2] == fid[3]
        assert fid[4] == fid[5] == fid[6]
        # The two families are distinct.
        assert fid[1] != fid[4]

    def test_global_statistics(self):
        events_ds, genealogy_ds = _make_synthetic_inputs()
        out = marEx.build_genealogy(events_ds, genealogy_ds)
        stats = marEx.compute_global_statistics(out)

        assert stats["n_events"] == 6
        assert stats["n_edges_PM"] == 2
        assert stats["n_edges_AM"] == 1
        assert stats["n_edges_PS"] == 1
        assert stats["n_families"] == 2

    def test_family_statistics_frame(self):
        events_ds, genealogy_ds = _make_synthetic_inputs()
        out = marEx.build_genealogy(events_ds, genealogy_ds)
        fam_df = marEx.compute_family_statistics(out)

        # Two rows, one per family.
        assert len(fam_df) == 2
        assert set(fam_df["n_events"].tolist()) == {3}

    def test_event_statistics_dataframe(self):
        events_ds, genealogy_ds = _make_synthetic_inputs()
        out = marEx.build_genealogy(events_ds, genealogy_ds)
        df = marEx.compute_event_statistics(out)

        assert len(df) == 6
        # Lifetime of e3 is 4 days (t=2..t=5 inclusive).
        row = df[df["event_id"] == 3].iloc[0]
        assert row["lifetime_days"] == 4

    def test_partitioned_merge_dedup_consecutive_rows(self):
        """Contiguous partition calls with the same parent/child sets must
        collapse into one PM episode. The tracker re-invokes partition every
        timestep a merged blob still straddles its parents, so the raw ledger
        contains one row per timestep — genealogy must emit a single episode
        anchored at the earliest time. A time gap ≥2 starts a new episode."""
        events_ds, genealogy_ds = _make_synthetic_inputs()

        times = events_ds["time"].values

        # Replace the single-row PM ledger with three contiguous rows at
        # t=2,3,4 (one episode) plus an isolated row at a different group-key
        # — none here; test focuses on the contiguous-run collapse.
        parent_IDs = np.array([[1, 2], [1, 2], [1, 2]], dtype=np.int32)
        child_IDs = np.array([[3], [3], [3]], dtype=np.int32)
        overlap_areas = np.array([[10.0, 15.0]] * 3, dtype=np.float32)
        merge_time = np.array([times[2], times[3], times[4]], dtype="datetime64[ns]")
        n_parents = np.array([2, 2, 2], dtype=np.int32)
        n_children = np.array([1, 1, 1], dtype=np.int32)

        genealogy_ds = genealogy_ds.drop_dims("merge_ID")
        genealogy_ds = genealogy_ds.assign(
            parent_IDs=(("merge_ID", "parent_idx"), parent_IDs),
            child_IDs=(("merge_ID", "child_idx"), child_IDs),
            overlap_areas=(("merge_ID", "parent_idx"), overlap_areas),
            merge_time=(("merge_ID",), merge_time),
            n_parents=(("merge_ID",), n_parents),
            n_children=(("merge_ID",), n_children),
        )

        out = marEx.build_genealogy(events_ds, genealogy_ds)
        edge_type = out["edge_type"].values.astype("U2")
        pm_mask = edge_type == "PM"

        # Three contiguous rows collapse to a single episode → 2 parents × 1 child = 2 PM edges.
        assert pm_mask.sum() == 2

        pm_times = np.unique(out["edge_time"].values[pm_mask])
        # Episode anchored at t=2 (earliest row in the run).
        assert len(pm_times) == 1
        assert pm_times[0] == times[2]
