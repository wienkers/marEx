"""The volumetric merge loop without merges: chunk handling, split consolidation, links, streaming."""

import gc
import json
import logging
import re
import subprocess
import sys
import tracemalloc
import weakref
from contextlib import contextmanager
from pathlib import Path

import dask.array as dsa
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from marEx.core.compute_mode import Materialiser
from marEx.track.region_writer import ObjectIDRegionWriter
from marEx.track.volumetric.geometry import build_vertical_geometry
from marEx.track.volumetric.ids import IDAllocator
from marEx.track.volumetric.merge_loop import (
    MergeLedger,
    SplitGeometry,
    consolidate_volumes,
    final_links,
    merge_loop_client_bytes,
    split_and_merge_volumetric,
)
from marEx.track.volumetric.overlap import link_mask, volume_overlaps
from marEx.track.volumetric.partition import ScaledCoords
from marEx.track.volumetric.store import ObjectIndex, VolumeStore

from .conftest import VDIMS, build_case, store_from_field
from .synthetic import Ball, paint, scenarios
from .test_partition import brute_nearest

THRESHOLD = 0.5
MODES = [("persist", 4), ("persist", 1), ("persist", 3), ("streaming", 3), ("streaming", 1)]


def _materialiser(mode, tmp_path):
    if mode == "streaming":
        return Materialiser("streaming", staging_dir=tmp_path / "stage")
    return Materialiser("persist")


def _run(case, mode, tmp_path):
    materialiser = _materialiser(mode, tmp_path)
    path = str(tmp_path / "merged.zarr") if mode == "streaming" else None
    field, store, ledger = split_and_merge_volumetric(
        case.id_field,
        case.store,
        case.allocator,
        case.geom,
        case.coords,
        THRESHOLD,
        case.dims,
        materialiser=materialiser,
        id_field_path=path,
    )
    links = final_links(field, store, case.cell_volume, THRESHOLD, case.dims.time)
    return field, store, ledger, links


def _chains(links):
    """Connected components of the link graph, as sorted tuples of IDs."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in links.tolist():
        parent[find(a)] = find(b)
    groups = {}
    for a in list(parent):
        groups.setdefault(find(a), []).append(a)
    return sorted(tuple(sorted(g)) for g in groups.values())


# --------------------------------------------------------------------------------------
# Region writer: a 4-D field through the existing 2-D writer
# --------------------------------------------------------------------------------------


def _region(start, end, shape):
    """Deterministic int32 content of time steps [start, end)."""
    rng = np.random.default_rng(start * 1000 + end)
    return rng.integers(0, 2**31 - 1, size=(end - start,) + shape, dtype=np.int32)


def _write_regions(path, nt, spatial, tchunk):
    """Write a 4-D field region by region through the writer; return what the client saw."""
    computed = []

    def poison(block):
        computed.append(block.shape)
        raise AssertionError("the template's values must never be computed")

    base = dsa.zeros((nt,) + spatial, dtype=np.int32, chunks=(tchunk, -1, -1, -1))
    template = xr.DataArray(
        base.map_blocks(poison, dtype=np.int32, meta=np.empty((0, 0, 0, 0), dtype=np.int32)),
        dims=("time", "depth", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=nt, freq="D"),
            "depth": np.arange(spatial[0], dtype=float) * 10.0 + 5.0,
            "lat": np.linspace(30, 40, spatial[1]),
            "lon": np.linspace(-170, -160, spatial[2]),
        },
        name="ID_field",
    )
    writer = ObjectIDRegionWriter(template, path, "time")
    # [2, 6) spans the chunk boundary at 4; the rest are chunk-sized or smaller.
    regions = [(0, 2), (2, 6), (6, 8)] + [(s, s + tchunk) for s in range(8, nt, tchunk)]
    # One trace over the whole loop. Each region is built inside it and released after its
    # write, so the bytes still held after a write are what the writer (or anything it
    # calls) keeps from the regions it was handed.
    held, alive = [], []
    tracemalloc.start()
    try:
        for start, end in regions:
            values = _region(start, end, spatial)
            ref = weakref.ref(values)
            data = template.isel(time=slice(start, end)).copy(data=values)
            writer.write(start, end, data)
            del data, values
            gc.collect()
            alive.append(ref() is not None)
            held.append(tracemalloc.get_traced_memory()[0])
    finally:
        tracemalloc.stop()
    out = writer.finalise()
    expected = np.concatenate([_region(s, e, spatial) for s, e in regions], axis=0)
    return out, expected, computed, held, alive


def _smoke_facts(tmp_path, spatial, tchunk):
    """Write 4-D regions for two lengths and collect the facts the smoke test asserts, as plain data."""
    facts = {}
    for nt in (16, 64):
        out, expected, computed, held, alive = _write_regions(Path(tmp_path) / f"ids{nt}.zarr", nt, spatial, tchunk)
        facts[nt] = {
            "computed": [list(c) for c in computed],
            "alive": alive,
            "held": held,
            "dtype": str(out.dtype),
            "dims": list(out.dims),
            "chunks": [list(c) for c in out.chunks],
            "equal": bool(np.array_equal(out.values, expected)),
        }
    return facts


def _assert_smoke_facts(facts, spatial, tchunk):
    chunk_bytes = tchunk * int(np.prod(spatial)) * 4
    for nt, f in facts.items():
        nt = int(nt)
        # The template's values are never computed.
        assert f["computed"] == []
        # No region array outlives its write, and the client memory held after the last write
        # exceeds that after the first by at most one chunk (keeping every region would add nt / tchunk chunks).
        assert not any(f["alive"])
        assert f["held"][-1] - f["held"][0] <= chunk_bytes, (nt, f["held"][0], f["held"][-1], chunk_bytes)
        assert f["dtype"] == "int32" and f["dims"] == ["time", "depth", "lat", "lon"]
        # Re-opened lazily with the template's chunk structure.
        assert f["chunks"] == [[tchunk] * (nt // tchunk), [spatial[0]], [spatial[1]], [spatial[2]]]
        assert f["equal"]


_SMOKE_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import marEx
assert marEx.__file__.startswith(sys.argv[1]), marEx.__file__
from tests.track.volumetric.test_merge_loop import _smoke_facts
print(json.dumps(_smoke_facts(sys.argv[2], (4, 60, 60), 4)))
"""


def test_region_writer_4d_smoke(tmp_path):
    """A (time, z, y, x) int32 field round-trips exactly through the 2-D region writer, one region at a time."""
    # tracemalloc counts every thread, and a dask cluster left running by another test allocates
    # in the background, so the write loop is measured in a fresh interpreter.
    spatial, tchunk = (4, 60, 60), 4
    root = str(Path(__file__).resolve().parents[3])
    proc = subprocess.run(
        [sys.executable, "-c", _SMOKE_SCRIPT, root, str(tmp_path)], cwd=root, capture_output=True, text=True, check=True
    )
    facts = json.loads(proc.stdout.strip().splitlines()[-1])
    assert set(facts) == {"16", "64"}
    _assert_smoke_facts(facts, spatial, tchunk)


# --------------------------------------------------------------------------------------
# Client memory model
# --------------------------------------------------------------------------------------


def _resident_float64_bytes(geom) -> int:
    """Bytes of the float64 ``(z, y, x)`` arrays the geometry keeps resident (counted from the real object)."""
    return sum(
        value.nbytes
        for value in vars(geom).values()
        if isinstance(value, np.ndarray) and value.dtype == np.float64 and value.ndim == 3
    )


def test_client_bytes_model(volumetric_case):
    # The static term is what the vertical geometry really holds: five float64 (z, y, x) arrays
    # (layer top, bottom, thickness, midpoint depth and cell volume), counted from the object.
    binary, meta = scenarios()["one_merge"]
    case = volumetric_case(binary, meta)
    n_voxels = case.cell_volume.size
    assert _resident_float64_bytes(case.geom) == 5 * n_voxels * 8
    static = _resident_float64_bytes(case.geom)
    for mode, retained in (("streaming", 2), ("persist", 11)):
        assert merge_loop_client_bytes(n_voxels, 3, mode) == retained * 3 * n_voxels * 4 + static
    assert merge_loop_client_bytes(7_560_000, 30, "streaming") == 2 * 30 * 7_560_000 * 4 + 5 * 7_560_000 * 8
    assert merge_loop_client_bytes(7_560_000, 30, "persist") == 11 * 30 * 7_560_000 * 4 + 5 * 7_560_000 * 8
    # At the 361 x 601 x 36 box the static term is ~312 MB, not the ~125 MB two arrays would give.
    assert merge_loop_client_bytes(361 * 601 * 36, 0, "streaming") == 5 * 361 * 601 * 36 * 8
    with pytest.raises(Exception, match="compute_mode"):
        merge_loop_client_bytes(10, 1, "lazy")


# --------------------------------------------------------------------------------------
# Synthetic scenarios describe themselves correctly
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(scenarios()))
def test_scenario_counts_and_merge_flags(name):
    binary, meta = scenarios()[name]
    case = build_case(binary, meta)
    counts = [np.unique(case.ids[t][case.ids[t] > 0]).size for t in range(binary.shape[0])]
    assert counts == meta["n_objects"]
    flat = case.ids.reshape(binary.shape[0], -1)
    cv = case.cell_volume.ravel()
    max_parents = 0
    for t in range(1, binary.shape[0]):
        pairs, ov = volume_overlaps(flat[t - 1], flat[t], cv)
        linked = pairs[link_mask(pairs, ov, case.store, THRESHOLD)]
        if linked.size:
            max_parents = max(max_parents, np.unique(linked[:, 1], return_counts=True)[1].max())
    assert (max_parents > 1) == meta["has_merge"]


# --------------------------------------------------------------------------------------
# No-merge scenarios through the loop
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode,time_chunk", MODES)
def test_persistence_growth_decay_one_event(tmp_path, mode, time_chunk):
    binary, meta = scenarios()["persist_grow_decay"]
    case = build_case(binary, meta, time_chunk=time_chunk)
    field, store, ledger, links = _run(case, mode, tmp_path)

    # t=0 and t=7 are empty; timesteps 1..6 hold one object each, IDs 1..6.
    expected = binary.astype(np.int32) * np.arange(binary.shape[0], dtype=np.int32)[:, None, None, None]
    assert field.dtype == np.int32 and field.dims == VDIMS.internal_order()
    np.testing.assert_array_equal(field.values, expected)
    np.testing.assert_array_equal(links, [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    assert links.dtype == np.int64
    assert _chains(links) == [(1, 2, 3, 4, 5, 6)]
    assert ledger.to_dataset().sizes["merge_ID"] == 0
    # Volumes are the identified ones, unchanged.
    reference = store_from_field(expected, case.cell_volume)
    ids = np.arange(1, 7)
    np.testing.assert_array_equal(store.volumes(ids), reference.volumes(ids))
    assert len(store) == 6


@pytest.mark.parametrize("mode,time_chunk", MODES)
def test_birth_death_separate_events(tmp_path, mode, time_chunk):
    binary, meta = scenarios()["birth_death"]
    case = build_case(binary, meta, time_chunk=time_chunk)
    field, store, ledger, links = _run(case, mode, tmp_path)

    nt, nz, ny, nx = binary.shape
    a = paint(nt, nz, ny, nx, [Ball(0, 4, z=1.0, y=2.0, x=2.0, rz=1.0, ry=1.5, rx=1.5)])
    b = paint(nt, nz, ny, nx, [Ball(2, 7, z=2.0, y=7.0, x=9.0, rz=1.0, ry=1.5, rx=1.5)])
    # A (shallower, first in C order) takes the lower ID while both exist.
    id_a = [1, 2, 3, 5, 7, 0, 0, 0]
    id_b = [0, 0, 4, 6, 8, 9, 10, 11]
    expected = np.zeros(binary.shape, dtype=np.int32)
    for t in range(nt):
        expected[t][a[t]] = id_a[t]
        expected[t][b[t]] = id_b[t]
    np.testing.assert_array_equal(field.values, expected)
    assert _chains(links) == [(1, 2, 3, 5, 7), (4, 6, 8, 9, 10, 11)]
    assert links.shape == (9, 2)
    assert ledger.to_dataset().sizes["merge_ID"] == 0


@pytest.mark.parametrize("mode,time_chunk", MODES + [("persist", 2), ("streaming", 2)])
def test_empty_and_all_foreground_slices(tmp_path, mode, time_chunk):
    nt, nz, ny, nx = 6, 2, 4, 5
    binary = np.zeros((nt, nz, ny, nx), dtype=bool)
    binary[[1, 2, 4]] = True  # t=0, t=3 and the last timestep t=5 are empty
    meta = {"make_inputs": {"thickness": [10.0, 20.0]}, "max_gap_m": 0.0}
    case = build_case(binary, meta, time_chunk=time_chunk)
    field, store, ledger, links = _run(case, mode, tmp_path)

    expected = np.zeros(binary.shape, dtype=np.int32)
    expected[1], expected[2], expected[4] = 1, 2, 3
    np.testing.assert_array_equal(field.values, expected)
    np.testing.assert_array_equal(links, [[1, 2]])
    reference = store_from_field(expected, case.cell_volume)
    np.testing.assert_array_equal(store.volumes([1, 2, 3]), reference.volumes([1, 2, 3]))
    assert len(store) == 3


@pytest.mark.parametrize("mode", ["persist", "streaming"])
def test_long_run_flushes_mid_loop(tmp_path, mode):
    """A persistent split over 24 one-step chunks: persist mode flushes consolidated chunks mid-loop (chunks 10, 20)."""
    nt, nz, ny, nx = 24, 1, 1, 5
    binary = np.zeros((nt, nz, ny, nx), dtype=bool)
    binary[0] = True  # one object at t=0 ...
    binary[1:, 0, 0, [0, 1, 3, 4]] = True  # ... then two pieces at every later step
    meta = {"make_inputs": {"thickness": [10.0]}, "max_gap_m": 0.0}
    case = build_case(binary, meta, time_chunk=1)
    field, store, ledger, links = _run(case, mode, tmp_path)

    # Identified IDs at t >= 1 are (2t, 2t + 1); every second piece is consolidated onto 2t.
    ids_kept = np.array([1] + [2 * t for t in range(1, nt)], dtype=np.int32)
    expected = binary.astype(np.int32) * ids_kept[:, None, None, None]
    np.testing.assert_array_equal(field.values, expected)
    np.testing.assert_array_equal(links, np.stack([ids_kept[:-1], ids_kept[1:]], axis=1))
    assert not any(2 * t + 1 in store for t in range(1, nt))


@pytest.mark.parametrize("mode,time_chunk", MODES + [("persist", 2), ("persist", 6)])
def test_split_is_consolidated_across_chunk_boundaries(tmp_path, mode, time_chunk):
    """A split keeps one ID: each t-1 slice is consolidated against t-2, whatever the chunking."""
    binary, meta = scenarios()["one_split"]
    case = build_case(binary, meta, time_chunk=time_chunk)
    field, store, ledger, links = _run(case, mode, tmp_path)

    nt = binary.shape[0]
    # Identified IDs: t=0..2 -> 1, 2, 3; then two pieces per step: (4, 5), (6, 7), (8, 9).
    # Consolidation renames each second piece onto the first.
    expected = binary.astype(np.int32) * np.array([1, 2, 3, 4, 6, 8], dtype=np.int32)[:, None, None, None]
    np.testing.assert_array_equal(field.values, expected)
    # The output keeps the input's chunk structure (streaming re-opens the written regions).
    assert field.chunks == case.id_field.chunks
    np.testing.assert_array_equal(links, [[1, 2], [2, 3], [3, 4], [4, 6], [6, 8]])
    for gone in (5, 7, 9):
        assert gone not in store
    reference = store_from_field(expected, case.cell_volume)
    kept = np.array([4, 6, 8])
    np.testing.assert_array_equal(store.volumes(kept), reference.volumes(kept))
    assert ledger.to_dataset().sizes["merge_ID"] == 0
    assert field.sizes["time"] == nt


# --------------------------------------------------------------------------------------
# Consolidation refreshes the volume that later link decisions read
# --------------------------------------------------------------------------------------


def _unit_case(ids, cell_volume=None):
    """Inputs on a grid of unit-volume cells (or ``cell_volume``) for a hand-built ``(time, 1, ny, nx)`` ID array."""
    nt, nz, ny, nx = ids.shape
    lat, lon = np.linspace(0.0, 1.0, ny), np.linspace(0.0, 1.0, nx)
    wet = xr.DataArray(np.ones((nz, ny, nx), bool), dims=("depth", "lat", "lon"))
    bounds = xr.DataArray(np.array([[0.0, 1.0]] * nz), dims=("depth", "nv"))
    geom = build_vertical_geometry(
        np.full(nz, 0.5),
        {"units": "m"},
        wet,
        zdim="depth",
        ydim="lat",
        xdim="lon",
        lat_deg=lat,
        lon_deg=lon,
        layer_bounds=bounds,
        layer_thickness=None,
        cell_volumes=xr.DataArray(np.ones((nz, ny, nx)) if cell_volume is None else cell_volume, dims=("depth", "lat", "lon")),
        cell_areas=xr.DataArray(np.ones((ny, nx)), dims=("lat", "lon"), attrs={"units": "m2"}),
        grid_resolution=None,
    )
    coords = ScaledCoords(lat, lon, geom.midpoint, *geom.default_scales)
    field = xr.DataArray(
        ids,
        dims=VDIMS.internal_order(),
        coords={"time": pd.date_range("2000-01-01", periods=nt, freq="D")},
        name="ID_field",
    )
    return geom, coords, field


def _consolidation_ids():
    ids = np.zeros((3, 1, 2, 10), dtype=np.int32)
    ids[0, 0, 0, :] = 1  # parent P, volume 10
    ids[1, 0, 0, 0] = 2  # child A, volume 1 (lower ID: the one kept)
    ids[1, 0, 0, 2:] = 3  # child B, 8 cells of row 0 ...
    ids[1, 0, 1, 9] = 3  # ... plus one of row 1: volume 9
    ids[2, 0, 1, 0:9] = 4  # T, volume 10: 9 cells of row 1 ...
    ids[2, 0, 0, 0] = 4  # ... plus A's cell, the only overlap with A + B
    return ids


def test_consolidation_updates_volume_and_link_decision(tmp_path):
    ids = _consolidation_ids()
    cv = np.ones(ids.shape[1:], dtype=np.float64)

    # Directly: B is renamed onto A and the store holds the consolidated volume.
    store = store_from_field(ids, cv)
    assert store.volumes([2])[0] == 1.0
    out = consolidate_volumes(ids[0], ids[1], store, cv, THRESHOLD)
    assert out.dtype == np.int32 and out.shape == ids[1].shape
    expected_t1 = np.where(ids[1] > 0, 2, 0).astype(np.int32)
    np.testing.assert_array_equal(out, expected_t1)
    np.testing.assert_array_equal(ids[1][ids[1] == 3].size, 9)  # the input is not modified
    assert store.volumes([2])[0] == 10.0
    assert 3 not in store
    # C_V(A+B, T) = 1 / min(10, 10) = 0.1 < 0.5: no link (a stale volume 1 would give 1.0).
    pairs, ov = volume_overlaps(out.ravel(), ids[2].ravel(), cv.ravel())
    np.testing.assert_array_equal(pairs, [[2, 4]])
    assert ov[0] / min(store.volumes([2])[0], store.volumes([4])[0]) == 0.1
    assert not link_mask(pairs, ov, store, THRESHOLD).any()

    # Through the loop, for every chunking that puts t-1 and t-2 in different chunks or the same one.
    for mode, time_chunk in [("persist", 3), ("persist", 1), ("persist", 2), ("streaming", 1)]:
        geom, coords, field = _unit_case(ids)
        store = store_from_field(ids, cv)
        sub = tmp_path / f"{mode}{time_chunk}"
        sub.mkdir()
        result, store, ledger = split_and_merge_volumetric(
            field.chunk({"time": time_chunk}),
            store,
            IDAllocator(5),
            geom,
            coords,
            THRESHOLD,
            VDIMS,
            materialiser=_materialiser(mode, sub),
            id_field_path=str(sub / "merged.zarr") if mode == "streaming" else None,
        )
        np.testing.assert_array_equal(result.values[1], expected_t1)
        np.testing.assert_array_equal(result.values[[0, 2]], ids[[0, 2]])
        assert store.volumes([2])[0] == 10.0 and 3 not in store
        links = final_links(result, store, cv, THRESHOLD, "time")
        np.testing.assert_array_equal(links, [[1, 2]])
        assert ledger.to_dataset().sizes["merge_ID"] == 0


def test_final_links_reads_volumes_from_the_store():
    """final_links decides with the store's volumes: a larger stored volume suppresses a link."""
    ids = np.zeros((2, 1, 1, 4), dtype=np.int32)
    ids[0, 0, 0, :2] = 1
    ids[1, 0, 0, 1:] = 2
    cv = np.ones((1, 1, 4))
    field = xr.DataArray(ids, dims=VDIMS.internal_order()).chunk({"time": 1})
    store = store_from_field(ids, cv)
    np.testing.assert_array_equal(final_links(field, store, cv, THRESHOLD, "time"), [[1, 2]])  # 1 / 2
    store.set(1, 4.0)
    store.set(2, 4.0)
    assert final_links(field, store, cv, THRESHOLD, "time").shape == (0, 2)  # 1 / 4


def test_ledger_dataset_schema():
    from marEx.track.volumetric.merge_loop import MergeLedger

    empty = MergeLedger().to_dataset()
    assert empty.sizes["merge_ID"] == 0 and empty.attrs == {"fill_value": -1}
    assert empty.sizes["split_ID"] == 0 and empty["split_time"].dtype == np.dtype("datetime64[ns]")
    t = np.datetime64("2000-01-03", "ns")
    ledger = MergeLedger()
    ledger.times.append(t)
    ledger.parent_ids.append(np.array([3, 5, 7], dtype=np.int32))
    ledger.child_ids.append(np.array([9, 10], dtype=np.int32))
    ledger.overlap_volumes.append(np.array([1.5, 2.5, 3.5]))
    ds = ledger.to_dataset()
    assert ds.parent_IDs.dtype == np.int32 and ds.child_IDs.dtype == np.int32
    assert ds.overlap_volumes.dtype == np.float64
    assert ds.n_parents.dtype == np.int32 and ds.n_children.dtype == np.int32
    np.testing.assert_array_equal(ds.parent_IDs.values, [[3, 5, 7]])
    np.testing.assert_array_equal(ds.child_IDs.values, [[9, 10]])
    np.testing.assert_array_equal(ds.overlap_volumes.values, [[1.5, 2.5, 3.5]])
    assert ds.merge_time.values[0] == t and ds.n_parents.item() == 3 and ds.n_children.item() == 2
    assert ds.parent_IDs.dims == ("merge_ID", "parent_idx") and ds.child_IDs.dims == ("merge_ID", "child_idx")


# --------------------------------------------------------------------------------------
# Merges: partition dispatch, ID allocation, convergence guard, ledger
# --------------------------------------------------------------------------------------

LOGGER_NAME = "marEx.track.volumetric.merge_loop"


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextmanager
def _debug_messages():
    """Collect the merge loop's DEBUG messages (the marEx root logger does not propagate to caplog)."""
    log = logging.getLogger(LOGGER_NAME)
    handler, old_level = _Records(), log.level
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    try:
        yield handler.messages
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)


def _iterations_by_timestep(messages):
    out = {}
    for msg in messages:
        m = re.match(r"Timestep (\d+): merges resolved in (\d+) iterations", msg)
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


def _oracle_labels(coords, child_vox, parent_vox, parent_ids, child_ids):
    """Label of every child voxel from the brute-force nearest parent voxel (ties: smallest parent ID)."""
    parent_pts = coords.points(np.concatenate(parent_vox))
    owner = np.repeat(np.arange(len(parent_vox)), [len(p) for p in parent_vox])
    idx = brute_nearest(coords.points(child_vox), parent_pts, owner, np.asarray(parent_ids))
    return np.asarray(child_ids, dtype=np.int32)[idx]


def _run_field(field, store, allocator, geom, coords, mode, tmp_path, **kwargs):
    sub = tmp_path / f"{mode}-{len(list(tmp_path.iterdir()))}"
    sub.mkdir()
    return split_and_merge_volumetric(
        field,
        store,
        allocator,
        geom,
        coords,
        THRESHOLD,
        VDIMS,
        materialiser=_materialiser(mode, sub),
        id_field_path=str(sub / "merged.zarr") if mode == "streaming" else None,
        **kwargs,
    )


def _merge_ids():
    """Two parents (volumes 2 and 3) at t=0, both covered by one 7-cell child at t=1."""
    ids = np.zeros((2, 1, 1, 7), dtype=np.int32)
    ids[0, 0, 0, 0:2] = 1
    ids[0, 0, 0, 4:7] = 2
    ids[1, 0, 0, :] = 3
    return ids


@pytest.mark.parametrize("mode,time_chunk", [("persist", 2), ("persist", 1), ("streaming", 1)])
def test_one_merge_partitions_and_records_ledger(tmp_path, mode, time_chunk):
    ids = _merge_ids()
    cv = np.ones(ids.shape[1:], dtype=np.float64)
    geom, coords, field = _unit_case(ids)
    store = store_from_field(ids, cv)
    allocator = IDAllocator(4)
    result, store, ledger = _run_field(field.chunk({"time": time_chunk}), store, allocator, geom, coords, mode, tmp_path)

    # Cells 0-2 are nearest parent 1 (the lower parent ID keeps the child ID 3), cells 3-6 parent 2 (new ID 4).
    expected = ids.copy()
    expected[1, 0, 0, :] = [3, 3, 3, 4, 4, 4, 4]
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result.values, expected)
    assert allocator.next_id == 5
    np.testing.assert_array_equal(store.volumes([3, 4]), [3.0, 4.0])

    ds = ledger.to_dataset()
    assert ds.sizes["merge_ID"] == 1
    assert ds.n_parents.dtype == np.int32 and ds.n_children.dtype == np.int32
    np.testing.assert_array_equal(ds.n_parents.values, np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(ds.n_children.values, np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(ds.parent_IDs.values, [[1, 2]])
    np.testing.assert_array_equal(ds.child_IDs.values, [[3, 4]])
    # Analytic overlaps: parent 1 shares 2 unit cells with the child, parent 2 shares 3.
    assert ds.overlap_volumes.dtype == np.float64
    np.testing.assert_array_equal(ds.overlap_volumes.values, [[2.0, 3.0]])
    assert ds.merge_time.values[0] == field.time.values[1]
    # After the partition each piece links to one parent only.
    np.testing.assert_array_equal(final_links(result, store, cv, THRESHOLD, "time"), [[1, 3], [2, 4]])


@pytest.mark.parametrize("mode,time_chunk", [("persist", 4), ("persist", 3), ("persist", 1), ("streaming", 2)])
@pytest.mark.parametrize("name", ["one_split", "merge_then_split"])
def test_one_split_and_merge_then_split(tmp_path, name, mode, time_chunk):
    binary, meta = scenarios()[name]
    case = build_case(binary, meta, time_chunk=time_chunk)
    ids0 = case.ids.copy()
    field, store, ledger = _run_field(case.id_field, case.store, case.allocator, case.geom, case.coords, mode, tmp_path)
    nt = binary.shape[0]
    flat_in = ids0.reshape(nt, -1)
    expected = ids0.copy().reshape(nt, -1)

    if name == "one_split":
        # No merge: identified IDs, every second piece consolidated onto the first.
        expected = binary.astype(np.int32) * np.array([1, 2, 3, 4, 6, 8], dtype=np.int32)[:, None, None, None]
        np.testing.assert_array_equal(field.values, expected)
        assert ledger.to_dataset().sizes["merge_ID"] == 0
        return

    # merge_then_split: identified IDs t0..t2 (1, 2), (3, 4), (5, 6); t3, t4, t5: 7, 8, 9;
    # t6..t8 (10, 11), (12, 13), (14, 15). The merged object is partitioned at t=3 (new ID 16), and
    # its continuation at t=4 and t=5 overlaps both pieces, so it is partitioned again (17, 18).
    assert [np.unique(flat_in[t][flat_in[t] > 0]).tolist() for t in range(nt)] == [
        [1, 2],
        [3, 4],
        [5, 6],
        [7],
        [8],
        [9],
        [10, 11],
        [12, 13],
        [14, 15],
    ]
    merges = [(3, [5, 6], [7, 16]), (4, [7, 16], [8, 17]), (5, [8, 17], [9, 18])]
    for t, parents, children in merges:
        child_vox = np.flatnonzero(flat_in[t] == children[0])
        parent_vox = [np.flatnonzero(expected[t - 1] == p) for p in parents]
        expected[t, child_vox] = _oracle_labels(case.coords, child_vox, parent_vox, parents, children)
    np.testing.assert_array_equal(field.values.reshape(nt, -1), expected)
    # The two halves stay apart at t=3..5: each half holds voxels.
    for t, _, children in merges:
        assert all((expected[t] == c).any() for c in children)

    ds = ledger.to_dataset()
    np.testing.assert_array_equal(ds.parent_IDs.values, [m[1] for m in merges])
    np.testing.assert_array_equal(ds.child_IDs.values, [m[2] for m in merges])
    np.testing.assert_array_equal(ds.merge_time.values, field.time.values[[3, 4, 5]])
    cv = case.cell_volume.ravel()
    # Summed left to right in flat voxel order (the order the loop's float64 sums use).
    overlaps = [
        [sum(cv[(expected[t - 1] == p) & (flat_in[t] == children[0])].tolist()) for p in parents] for t, parents, children in merges
    ]
    np.testing.assert_array_equal(ds.overlap_volumes.values, overlaps)
    # The store holds exactly the volumes of the final field.
    reference = store_from_field(expected.reshape(binary.shape), case.cell_volume)
    final_ids = np.unique(expected[expected > 0])
    np.testing.assert_array_equal(store.volumes(final_ids), reference.volumes(final_ids))
    assert len(store) == final_ids.size


@pytest.mark.parametrize("mode,time_chunk", [("persist", 2), ("streaming", 1)])
def test_many_parents_130_int32_counts(tmp_path, mode, time_chunk):
    binary, meta = scenarios()["many_parents_130"]
    case = build_case(binary, meta, time_chunk=time_chunk)
    ids0 = case.ids.reshape(2, -1).copy()
    np.testing.assert_array_equal(np.unique(ids0[0][ids0[0] > 0]), np.arange(1, 131))
    np.testing.assert_array_equal(np.unique(ids0[1][ids0[1] > 0]), [131])
    field, store, ledger = _run_field(case.id_field, case.store, case.allocator, case.geom, case.coords, mode, tmp_path)
    ds = ledger.to_dataset()
    assert ds.n_parents.dtype == np.int32 and ds.n_children.dtype == np.int32
    np.testing.assert_array_equal(ds.n_parents.values, np.array([130], dtype=np.int32))
    assert ds.n_parents.values[0] > np.iinfo(np.int8).max  # a 2-D int8 count would overflow
    np.testing.assert_array_equal(ds.n_children.values, np.array([130], dtype=np.int32))
    np.testing.assert_array_equal(ds.parent_IDs.values[0], np.arange(1, 131))
    children = np.r_[131, np.arange(132, 132 + 129)].astype(np.int32)
    np.testing.assert_array_equal(ds.child_IDs.values[0], children)
    assert case.allocator.next_id == 132 + 129

    child_vox = np.flatnonzero(ids0[1] == 131)
    parent_vox = [np.flatnonzero(ids0[0] == p) for p in range(1, 131)]
    expected = ids0.copy()
    expected[1, child_vox] = _oracle_labels(case.coords, child_vox, parent_vox, np.arange(1, 131), children)
    np.testing.assert_array_equal(field.values.reshape(2, -1), expected)
    # The child fills the plane, and every parent keeps a piece: 130 IDs at t=1.
    np.testing.assert_array_equal(np.unique(expected[1]), children)


def _exhaustion_ids():
    """Three parents at t=0 (2, 3 and 2 cells), all covered by one 11-cell child at t=1.

    No child cell is equidistant from two parents: cells 0-2 are nearest parent 1, 3-7 parent 2, 8-10 parent 3.
    """
    ids = np.zeros((2, 1, 1, 11), dtype=np.int32)
    ids[0, 0, 0, 0:2] = 1
    ids[0, 0, 0, 4:7] = 2
    ids[0, 0, 0, 9:11] = 3
    ids[1, 0, 0, :] = 4
    return ids


def test_id_exhaustion_raises_before_mutating(tmp_path):
    from marEx.exceptions import TrackingError
    from marEx.track.volumetric.ids import INT32_MAX
    from marEx.track.volumetric.merge_loop import MergeLedger, _resolve_timestep

    ids = _exhaustion_ids()
    cv = np.ones(ids.shape[1:], dtype=np.float64)
    geom, coords, field = _unit_case(ids)
    prev, cur = ids[0].ravel().copy(), ids[1].ravel().copy()
    flat_cv = cv.ravel()

    def attempt(next_id):
        store = store_from_field(ids, cv)
        allocator = IDAllocator(next_id)
        ledger = MergeLedger()
        slice_t = cur.copy()
        before = dict(store._vol)
        try:
            out = _resolve_timestep(
                prev, slice_t, store, allocator, coords, flat_cv, THRESHOLD, ledger, field.time.values[1], 1, 100
            )
        except TrackingError as err:
            return err, slice_t, store, before, allocator, ledger
        return out, slice_t, store, before, allocator, ledger

    # At INT32_MAX one more ID is legal, the merge needs two: it raises and nothing changed.
    err, slice_t, store, before, allocator, ledger = attempt(INT32_MAX)
    assert isinstance(err, TrackingError) and "exhausted" in str(err)
    np.testing.assert_array_equal(slice_t, cur)
    assert dict(store._vol) == before
    assert allocator.next_id == INT32_MAX
    assert ledger.parent_ids == [] and ledger.child_ids == [] and ledger.overlap_volumes == [] and ledger.times == []

    # Success control: from INT32_MAX - 1 the same merge mints exactly the last two int32 IDs.
    out, slice_t, store, before, allocator, ledger = attempt(INT32_MAX - 1)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, [4] * 3 + [2147483646] * 5 + [2147483647] * 3)
    np.testing.assert_array_equal(ledger.child_ids[0], np.array([4, 2147483646, 2147483647], dtype=np.int32))
    assert ledger.child_ids[0].dtype == np.int32
    assert allocator.next_id == INT32_MAX + 1
    np.testing.assert_array_equal(store.volumes([4, 2147483646, 2147483647]), [3.0, 5.0, 3.0])

    # Two merging children in one pass (5 over parents 1, 2; 6 over parents 3, 4) need two new IDs
    # together. At INT32_MAX only one is legal: the pass raises before either child is touched.
    ids2 = np.zeros((2, 1, 1, 16), dtype=np.int32)
    ids2[0, 0, 0, 0:2], ids2[0, 0, 0, 4:7] = 1, 2
    ids2[0, 0, 0, 9:11], ids2[0, 0, 0, 13:16] = 3, 4
    ids2[1, 0, 0, 0:7], ids2[1, 0, 0, 9:16] = 5, 6
    cv2 = np.ones(ids2.shape[1:], dtype=np.float64)
    _, coords2, field2 = _unit_case(ids2)
    prev2, cur2 = ids2[0].ravel().copy(), ids2[1].ravel().copy()

    def attempt_two(next_id):
        store = store_from_field(ids2, cv2)
        allocator, ledger, slice_t = IDAllocator(next_id), MergeLedger(), cur2.copy()
        before = dict(store._vol)
        try:
            _resolve_timestep(
                prev2, slice_t, store, allocator, coords2, cv2.ravel(), THRESHOLD, ledger, field2.time.values[1], 1, 100
            )
        except TrackingError as err:
            return err, slice_t, store, before, allocator, ledger
        return None, slice_t, store, before, allocator, ledger

    err, slice_t, store, before, allocator, ledger = attempt_two(INT32_MAX)
    assert isinstance(err, TrackingError) and "exhausted" in str(err)
    np.testing.assert_array_equal(slice_t, cur2)
    assert dict(store._vol) == before
    assert allocator.next_id == INT32_MAX
    assert ledger.parent_ids == [] and ledger.child_ids == [] and ledger.overlap_volumes == [] and ledger.times == []

    # Success control: children in ascending order mint 2147483646 (child 5) and 2147483647 (child 6).
    err, slice_t, store, before, allocator, ledger = attempt_two(INT32_MAX - 1)
    assert err is None
    np.testing.assert_array_equal(slice_t, [5] * 3 + [2147483646] * 4 + [0, 0] + [6] * 3 + [2147483647] * 4)
    np.testing.assert_array_equal(ledger.child_ids[0], [5, 2147483646])
    np.testing.assert_array_equal(ledger.child_ids[1], [6, 2147483647])
    assert allocator.next_id == INT32_MAX + 1

    # Through the loop: the raise propagates and the input field is untouched.
    store = store_from_field(ids, cv)
    original = field.values.copy()
    with pytest.raises(TrackingError, match="exhausted"):
        _run_field(field.chunk({"time": 1}), store, IDAllocator(INT32_MAX), geom, coords, "persist", tmp_path)
    np.testing.assert_array_equal(field.values, original)


def _cascade_case(time_chunk=None):
    binary, meta = scenarios()["cascade_11_iterations"]
    return build_case(binary, meta, time_chunk=time_chunk)


def test_convergence_guard_raises(tmp_path):
    from marEx.exceptions import TrackingError

    # The cascade needs 11 passes at timestep 2; after one pass the column (ID 25) still has two parents.
    case = _cascade_case(time_chunk=4)
    with pytest.raises(TrackingError) as err:
        _run_field(case.id_field, case.store, case.allocator, case.geom, case.coords, "persist", tmp_path, max_iterations=1)
    assert err.value.context == {"timestep": 2, "unresolved_children": [25]}
    assert "timestep 2" in str(err.value) and "[25]" in str(err.value)

    # Ten passes are one too few; eleven are enough.
    case = _cascade_case(time_chunk=4)
    with pytest.raises(TrackingError, match="after 10 iterations"):
        _run_field(case.id_field, case.store, case.allocator, case.geom, case.coords, "persist", tmp_path, max_iterations=10)
    case = _cascade_case(time_chunk=4)
    _run_field(case.id_field, case.store, case.allocator, case.geom, case.coords, "persist", tmp_path, max_iterations=11)


# Column labels after the cascade, top to bottom: B (parent 13) keeps the column ID 25 on
# the top two levels; Q11 (parent 14, level 2) gets ID 36, and each deeper Qk gets its own
# level plus the empty level above it, with the ID minted on the pass it linked (Q1: 26).
CASCADE_COLUMN = [25, 25, 36] + [36 - i for i in range(1, 11) for _ in range(2)]


@pytest.mark.parametrize("mode,time_chunk", [("persist", 4), ("persist", 2), ("persist", 1), ("streaming", 2)])
def test_cascade_over_ten_iterations_converges(tmp_path, mode, time_chunk):
    case = _cascade_case(time_chunk=time_chunk)
    ids0 = case.ids.copy()
    nt, nz = ids0.shape[:2]
    with _debug_messages() as messages:
        field, store, ledger = _run_field(case.id_field, case.store, case.allocator, case.geom, case.coords, mode, tmp_path)
    # The loop's own count: timestep 2 needed 11 passes (more than the 2-D loop's 10).
    assert _iterations_by_timestep(messages) == {2: 11}

    out = field.values
    np.testing.assert_array_equal(out[:2], ids0[:2])
    expected = np.zeros_like(ids0[2])
    expected[:, 2, 2] = CASCADE_COLUMN
    np.testing.assert_array_equal(out[2], expected)

    ds = ledger.to_dataset()
    # Pass k partitions the remaining column between B (13) and Q_k (parent 25 - k), minting 25 + k.
    np.testing.assert_array_equal(ds.parent_IDs.values, [[13, 25 - k] for k in range(1, 12)])
    np.testing.assert_array_equal(ds.child_IDs.values, [[25, 25 + k] for k in range(1, 12)])
    np.testing.assert_array_equal(ds.n_parents.values, np.full(11, 2, dtype=np.int32))
    assert (ds.merge_time.values == field.time.values[2]).all()
    # Overlap volumes from the thresholded rows: all of B's voxel, and Q_k's one column voxel.
    cv = case.cell_volume
    col = cv[:, 2, 2]
    q_levels = [2 + 2 * (11 - k) for k in range(1, 12)]
    np.testing.assert_array_equal(ds.overlap_volumes.values, [[col[0], col[lvl]] for lvl in q_levels])
    assert case.allocator.next_id == 37
    reference = store_from_field(out, cv)
    final = np.arange(25, 37)
    np.testing.assert_array_equal(store.volumes(final), reference.volumes(final))


def test_partition_volumes_conserved_in_loop(tmp_path, monkeypatch):
    from marEx.track.volumetric import merge_loop

    calls = []
    real = merge_loop._resolve_merge

    def recording(slice_t, index_t, index_prev, child_id, parent_ids, overlap_rows, store, allocator, coords, cell_volume):
        before_vox = index_t.voxels(child_id).copy()
        before_vol = store.volumes([child_id])[0]
        np.testing.assert_array_equal(before_vox, np.flatnonzero(slice_t == child_id))
        child_ids, overlap = real(
            slice_t, index_t, index_prev, child_id, parent_ids, overlap_rows, store, allocator, coords, cell_volume
        )
        pieces = [index_t.voxels(c) for c in child_ids]
        vols = [store.volumes([c])[0] for c in child_ids if c in store]
        calls.append((before_vox, before_vol, pieces, vols, slice_t.copy(), child_ids))
        return child_ids, overlap

    monkeypatch.setattr(merge_loop, "_resolve_merge", recording)

    # Binary-exact voxel volumes (powers of two) on the three-parent case: the pieces' summed
    # volumes equal the child's exactly.
    ids = _exhaustion_ids()
    cv = (2.0 ** (np.arange(11) % 4 - 1)).reshape(ids.shape[1:])
    geom, coords, field = _unit_case(ids, cell_volume=cv)
    store = store_from_field(ids, cv)
    _run_field(field.chunk({"time": 1}), store, IDAllocator(5), geom, coords, "persist", tmp_path)
    # And the 11 merges of the cascade (volumes from the sphere, not binary-exact).
    case = _cascade_case(time_chunk=2)
    _run_field(case.id_field, case.store, case.allocator, case.geom, case.coords, "persist", tmp_path)

    assert len(calls) == 1 + 11
    for i, (before_vox, before_vol, pieces, vols, slice_after, child_ids) in enumerate(calls):
        joined = np.concatenate(pieces)
        assert joined.size == np.unique(joined).size  # disjoint
        np.testing.assert_array_equal(np.sort(joined), before_vox)  # union is the child
        for c, vox in zip(child_ids, pieces):
            np.testing.assert_array_equal(vox, np.flatnonzero(slice_after == c))
        if i == 0:
            assert len(vols) == 3 and sum(vols) == before_vol == float(cv.sum())


def _allocation_inputs(domain, n_parents):
    """A child and its parents with fixed voxel counts (480 each) inside a (z, y, x) domain."""
    nz, ny, nx = domain
    prev = np.zeros(domain, dtype=np.int32)
    cur = np.zeros(domain, dtype=np.int32)
    width = 24 // n_parents
    for p in range(n_parents):
        prev[0:2, 0:10, p * width : (p + 1) * width] = p + 1
    cur[0:2, 0:10, 0:24] = 100
    prev, cur = prev.ravel(), cur.ravel()
    cv = np.ones(prev.size, dtype=np.float64)
    lat, lon = np.linspace(30.0, 40.0, ny), np.linspace(-170.0, -160.0, nx)
    midpoint = np.broadcast_to((np.arange(nz) * 10.0 + 5.0)[:, None, None], domain).copy()
    coords = ScaledCoords(lat, lon, midpoint, 1.0e4, 10.0)
    store = VolumeStore()
    store.update_from_slice(prev, cv)
    store.update_from_slice(cur, cv)
    parent_ids = np.arange(1, n_parents + 1, dtype=np.int32)
    overlap_rows = np.full(n_parents, 480.0 / n_parents)
    return prev, cur, cv, coords, store, parent_ids, overlap_rows


def _merge_peak(domain, n_parents):
    from marEx.track.volumetric.merge_loop import _resolve_merge

    def once():
        prev, cur, cv, coords, store, parent_ids, overlap_rows = _allocation_inputs(domain, n_parents)
        index_prev, index_t = ObjectIndex.from_slice(prev), ObjectIndex.from_slice(cur)
        allocator = IDAllocator(1000)
        gc.collect()
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            child_ids, _ = _resolve_merge(cur, index_t, index_prev, 100, parent_ids, overlap_rows, store, allocator, coords, cv)
            peak = tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()
        assert len(child_ids) == n_parents and (cur[cur > 0] != 100).sum() == 480 - 480 // n_parents
        return peak

    once()  # warm up first-call allocations (imports, caches)
    return min(once() for _ in range(3))


_PEAKS_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from tests.track.volumetric.test_merge_loop import _merge_peak
print(json.dumps([[list(d), n, _merge_peak(d, n)] for d in [(4, 40, 60), (4, 80, 120)] for n in (2, 8)]))
"""


def test_per_merge_allocations_bounded():
    # Asserts invariance of the per-merge peak, not an absolute size. tracemalloc counts every
    # thread, and a dask cluster left running by another test allocates in the background, so
    # the peaks are measured in a fresh interpreter.
    root = str(Path(__file__).resolve().parents[3])
    proc = subprocess.run([sys.executable, "-c", _PEAKS_SCRIPT, root], cwd=root, capture_output=True, text=True, check=True)
    peaks = {(tuple(d), n): peak for d, n, peak in json.loads(proc.stdout.strip().splitlines()[-1])}
    # Neither the domain (9600 -> 38400 voxels) nor the parent count (2 -> 8) moves the peak. One
    # full-domain bool mask of the larger domain is 38400 B; the allowance is a fraction of it.
    allowance = 4096
    small, large = (4, 40, 60), (4, 80, 120)
    for n in (2, 8):
        assert peaks[(large, n)] - peaks[(small, n)] <= allowance, peaks
    for d in (small, large):
        assert peaks[(d, 8)] - peaks[(d, 2)] <= allowance, peaks


# --------------------------------------------------------------------------------------
# Renames and split records
# --------------------------------------------------------------------------------------


def _consolidate_row(prev2, prev1, **kwargs):
    """Consolidate one row of IDs (a two-timestep field) with split labelling on; returns the ledger's dataset."""
    ids = np.array([prev2, prev1], dtype=np.int32)[:, np.newaxis, np.newaxis, :]
    cv = np.ones(ids.shape[1:], dtype=np.float64)
    ledger = MergeLedger()
    consolidate_volumes(
        ids[0],
        ids[1],
        store_from_field(ids, cv),
        cv,
        THRESHOLD,
        ledger=ledger,
        time_value=np.datetime64("2000-01-02", "ns"),
        split_geometry=SplitGeometry(np.ones(cv.shape, dtype=bool), cv, 0.0, False),
        **kwargs,
    )
    return ledger.to_dataset()


def test_consolidation_records_the_rename_and_the_split():
    ids = _consolidation_ids()
    cv = np.ones(ids.shape[1:], dtype=np.float64)
    ledger = MergeLedger()
    day = np.datetime64("2000-01-02", "ns")
    geometry = SplitGeometry(np.ones(cv.shape, dtype=bool), cv, 0.0, False)
    consolidate_volumes(
        ids[0], ids[1], store_from_field(ids, cv), cv, THRESHOLD, ledger=ledger, time_value=day, split_geometry=geometry
    )
    ds = ledger.to_dataset()
    np.testing.assert_array_equal(ds["_renamed_from"].values, [3])
    np.testing.assert_array_equal(ds["_renamed_to"].values, [2])
    np.testing.assert_array_equal(ds["split_time"].values, [day])
    np.testing.assert_array_equal(ds["split_parent_IDs"].values, [1])
    np.testing.assert_array_equal(ds["split_child_IDs"].values, [[2, 3]])
    np.testing.assert_array_equal(ds["n_split_children"].values, [2])


def test_no_split_is_recorded_without_the_geometry():
    ids = _consolidation_ids()
    cv = np.ones(ids.shape[1:], dtype=np.float64)
    ledger = MergeLedger()
    consolidate_volumes(
        ids[0], ids[1], store_from_field(ids, cv), cv, THRESHOLD, ledger=ledger, time_value=np.datetime64("2000-01-02", "ns")
    )
    ds = ledger.to_dataset()
    np.testing.assert_array_equal(ds["_renamed_from"].values, [3])
    assert ds.sizes["split_ID"] == 0


def test_pieces_of_one_partition_are_renamed_but_are_not_a_split():
    """A parent overlapping two adjacent pieces of a partitioned child did not separate."""
    ds = _consolidate_row([1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 3, 3, 3, 3])
    np.testing.assert_array_equal(ds["_renamed_from"].values, [3])
    np.testing.assert_array_equal(ds["_renamed_to"].values, [2])
    assert ds.sizes["split_ID"] == 0


def test_a_parent_holding_two_pieces_is_not_a_split():
    """Each piece of a two-piece parent continues as one child piece, as after a merge the day before."""
    ds = _consolidate_row([1, 1, 1, 0, 0, 0, 1, 1, 1], [2, 2, 2, 0, 0, 0, 3, 3, 3])
    np.testing.assert_array_equal(ds["_renamed_from"].values, [3])
    assert ds.sizes["split_ID"] == 0


def test_one_piece_of_a_two_piece_parent_separating_is_a_split():
    ds = _consolidate_row([1, 1, 1, 1, 1, 0, 0, 0, 1, 1], [2, 2, 0, 3, 3, 0, 0, 0, 4, 4])
    np.testing.assert_array_equal(ds["split_parent_IDs"].values, [1])
    np.testing.assert_array_equal(ds["split_child_IDs"].values, [[2, 3, 4]])


def test_split_follows_a_kept_id_that_a_later_parent_renames():
    """Parent 5 keeps 2, then parent 6 renames 2 onto 1: the children of parent 5 are read as survivor 1."""
    ds = _consolidate_row([6, 6, 6, 5, 5, 5, 5, 5, 5, 5], [1, 0, 2, 2, 0, 0, 0, 3, 3, 3])
    np.testing.assert_array_equal(ds["_renamed_from"].values, [3, 2])
    np.testing.assert_array_equal(ds["_renamed_to"].values, [2, 1])
    np.testing.assert_array_equal(ds["split_parent_IDs"].values, [5, 6])
    np.testing.assert_array_equal(ds["split_child_IDs"].values, [[2, 3], [1, 2]])


def test_consolidation_without_a_ledger_is_unchanged():
    ids = _consolidation_ids()
    cv = np.ones(ids.shape[1:], dtype=np.float64)
    a = consolidate_volumes(ids[0], ids[1], store_from_field(ids, cv), cv, THRESHOLD)
    b = consolidate_volumes(
        ids[0], ids[1], store_from_field(ids, cv), cv, THRESHOLD, ledger=MergeLedger(), time_value=np.datetime64("2000-01-02", "ns")
    )
    np.testing.assert_array_equal(a, b)
