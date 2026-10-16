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

import numpy as np
import pandas as pd
import xarray as xr

import marEx

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


def _inputs():
    cells = np.array(ROWS, dtype=bool)
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


def _track(temp_dir, serial):
    da, mask, nb, area = _inputs()
    tr = marEx.tracker(da, mask, temp_dir=str(temp_dir), neighbours=nb, cell_areas=area, **KWARGS)
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
