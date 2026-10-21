"""The merge loop's forward search must accept an overlap exactly at the threshold.

After a merge partitions a child, the unstructured merge loop queues every object at the next
timestep that one of the new pieces overlaps by at least ``overlap_threshold``, the rule every
other acceptance in the tracker uses. A strict comparison there skipped a piece whose overlap
sat exactly on the threshold, so a cascade the serial path follows was cut short.

The case is a nine-cell equatorial chain with unit areas and ``overlap_threshold=0.5``: two
objects join into one, which splits into a five-cell and a four-cell piece; the next object
covers the four middle cells, two inside each piece, so each piece overlaps it by exactly 2 / 4.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

import marEx
from marEx.track import merge_split
from marEx.track.merge_split import _overlap_fraction

ROWS = [
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 0, 0],
]

KWARGS = {
    "R_fill": 0,
    "T_fill": 0,
    "area_filter_absolute": 1,
    "allow_merging": True,
    "overlap_threshold": 0.5,
    "nn_partitioning": False,
    "unstructured_grid": True,
    "dimensions": {"x": "ncells"},
    "coordinates": {"x": "lon", "y": "lat"},
    "regional_mode": False,
    "coordinate_units": "degrees",
    "quiet": True,
}


def _inputs(rows=ROWS):
    cells = np.array(rows, dtype=bool)
    nt, n = cells.shape
    lat = np.zeros(n, np.float32)
    lon = np.arange(-(n // 2), n - n // 2, dtype=np.float32)
    neighbours = np.zeros((3, n), np.int32)  # 1-based, 0 = none
    neighbours[0, 1:] = np.arange(1, n)
    neighbours[1, :-1] = np.arange(2, n + 1)
    spatial = {"lat": ("ncells", lat), "lon": ("ncells", lon)}
    times = pd.date_range("2000-01-01", periods=nt, freq="D")
    da = xr.DataArray(cells, dims=("time", "ncells"), coords={"time": times, **spatial}).chunk({"time": nt, "ncells": -1})
    mask = xr.DataArray(np.ones(n, bool), dims="ncells", coords=spatial).chunk({"ncells": -1})
    nb = xr.DataArray(neighbours, dims=("nv", "ncells")).chunk({"nv": -1, "ncells": -1})
    area = xr.DataArray(np.ones(n, np.float32), dims="ncells", coords=spatial).chunk({"ncells": -1})
    return da, mask, nb, area


def _track(temp_dir, serial, rows=ROWS, **overrides):
    da, mask, nb, area = _inputs(rows)
    tr = marEx.tracker(da, mask, temp_dir=str(temp_dir), neighbours=nb, cell_areas=area, **{**KWARGS, **overrides})
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


def test_exact_threshold_overlap_continues_the_cascade(dask_client_unstructured, tmp_path):
    (tmp_path / "parallel").mkdir()
    (tmp_path / "serial").mkdir()
    events, objects, n_merges = _track(tmp_path / "parallel", serial=False)
    ref_events, ref_objects, ref_merges = _track(tmp_path / "serial", serial=True)

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


def test_a_child_with_several_parents_is_reported_with_its_overlap_details(dask_client_unstructured, tmp_path, caplog):
    # The package root logger neither propagates to pytest's handler nor keeps handlers added before the
    # tracker is built (configuring it clears them), so attach caplog's to the module's own logger.
    module_logger = logging.getLogger("marEx.track.merge_split")
    module_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG, logger="marEx.track.merge_split"):
            _track(tmp_path, serial=True, rows=MULTI_PARENT_ROWS, overlap_threshold=0.1, quiet=False)
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


class _CapturingDask:
    """Stand-in for ``dask`` as ``merge_split`` sees it that keeps the ``process_chunk`` closure it is handed."""

    def __init__(self, real):
        self._real = real
        self.kernels = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def delayed(self, func, *args, **kwargs):
        if getattr(func, "__name__", None) == "process_chunk":
            self.kernels.append(func)
        return self._real.delayed(func, *args, **kwargs)


def test_overlap_fraction_is_the_float64_ratio_of_overlap_to_the_smaller_area():
    got = _overlap_fraction(np.array([2, 5, 1], dtype=np.float32), np.array([4, 5, 3], dtype=np.float32))
    assert got.dtype == np.float64
    assert got[0] == 0.5 and got[1] == 1.0 and got[2] == 1.0 / 3.0


def test_merge_kernel_accepts_parents_and_forwards_children_by_hand_computed_overlaps(
    dask_client_unstructured, tmp_path, monkeypatch
):
    """Drive the real ``process_chunk`` on a twelve-cell chain whose every overlap fraction is worked out by hand.

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
    capture = _CapturingDask(merge_split.dask)
    monkeypatch.setattr(merge_split, "dask", capture)
    _track(tmp_path, serial=False)
    monkeypatch.undo()
    assert capture.kernels, "the merge loop never reached process_chunk"
    process_chunk = capture.kernels[0]

    n = 12

    def row(spans):
        out = np.zeros(n, np.int32)
        for ident, (lo, hi) in spans.items():
            out[lo : hi + 1] = ident
        return out

    previous = row({1: (0, 3), 2: (4, 8), 3: (9, 11)})
    current = row({5: (2, 9)})
    following = row({7: (3, 6), 8: (9, 11), 9: (0, 2)})

    store = str(tmp_path / "accumulator.zarr")
    xr.DataArray(current[None, :], dims=("time", "ncells"), name="temp").to_zarr(store, mode="w")
    zeros3 = np.zeros((3, n), np.int32)
    records, forwarded, _, _, max_parents, wrote = process_chunk(
        current[None, :],
        previous,
        following,
        [[5]],
        np.array([100], dtype=np.int64),
        0,
        np.zeros(n, np.float32),
        np.arange(n, dtype=np.float32),
        np.ones(n, np.float32),
        zeros3,
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
