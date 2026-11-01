"""Object identification: time-connected labelling, per-slice identification, the small-object filters
(``prefilter_min_cells`` and the quartile / absolute area filters), the area census, hole filling's chunk
structure, and time-gap filling's chunk structure.

Every oracle here is written independently of the implementation (scipy labelling, plain BFS, bincounts).
Only the two tracker-integration tests run on the session ``client``, where they ran before the
consolidation; the rest are threaded.
"""

from __future__ import annotations

import sys
import warnings

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.array.core import PerformanceWarning
from dask_image.ndmeasure import label as dask_image_label
from scipy.ndimage import label as scipy_label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import marEx
import marEx.track as track
from marEx.core.encoding import clear_store_encoding
from marEx.track.grid import build_sparse_dilation_matrix
from marEx.track.labelling import label_time_connected
from marEx.track.morphology import _EIGHT_CONNECTIVITY, _merge_lon_seam, drop_small_components, fill_holes, fill_time_gaps
from marEx.track.objects import identify_objects

from .conftest import assert_no_failures, check, on_client, open_gridded, open_unstructured, tile_to
from .mre_mesh import build_mesh

# -- sparse boolean powers (the R_fill closing on a mesh) --------------------------------------------


def _csr(rows):
    """CSR (data, indices, indptr) of a boolean adjacency given as one neighbour list per node."""
    indices = np.array([j for row in rows for j in row], dtype=np.int32)
    indptr = np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int32)
    return np.ones(indices.size, dtype=bool), indices, indptr


def test_sparse_bool_power():
    """``sparse_bool_power(vec, A, e)`` is ``vec @ A**e`` over booleans, one row per vector."""
    identity = _csr([[i] for i in range(5)])
    chain = _csr([[1], [0, 2], [1]])  # 0-1-2
    pairs = _csr([[1], [0], [3], [2]])  # 0-1 and 2-3, disconnected
    rows = [
        ("identity^1", identity, [[1, 0, 1, 0, 1]], 1, [[1, 0, 1, 0, 1]]),
        ("identity^2", identity, [[1, 0, 1, 0, 1]], 2, [[1, 0, 1, 0, 1]]),
        ("identity^3", identity, [[1, 0, 1, 0, 1]], 3, [[1, 0, 1, 0, 1]]),
        ("chain_one_step", chain, [[1, 0, 0]], 1, [[0, 1, 0]]),
        ("chain_two_steps", chain, [[1, 0, 0]], 2, [[1, 0, 1]]),
        ("exponent_zero", _csr([[1], [0]]), [[1, 0]], 0, [[1, 0]]),
        ("several_vectors", _csr([[i] for i in range(3)]), np.eye(3).tolist(), 1, np.eye(3).tolist()),
        ("disconnected", pairs, [[1, 0, 1, 0]], 1, [[0, 1, 0, 1]]),
    ]
    failures = []
    for case, (data, indices, indptr), vec, exponent, expected in rows:

        def row(data=data, indices=indices, indptr=indptr, vec=vec, exponent=exponent, expected=expected):
            got = track.sparse_bool_power(np.array(vec, dtype=bool), data, indices, indptr, exponent)
            assert np.array_equal(got, np.array(expected, dtype=bool)), got.astype(int).tolist()

        check(failures, case, row)
    assert_no_failures(failures)


# -- time-connected labelling ------------------------------------------------------------------------
#
# ``label_time_connected`` must give dask-image's labels and count exactly, with a task graph linear in the
# number of time chunks. ``dask_image.ndmeasure.label`` is the reference here, never on the library path:
# its graph grows roughly quadratically in the number of chunks (2,037 tasks at 8 time chunks of the field
# below, 6,517 at 16, and about 82,000 at 64, where the exact count varies with the hash seed).

FULL = np.ones((3, 3, 3))


def _random_label_cases():
    cases = []
    for seed in range(24):
        rng = np.random.default_rng(seed)
        nt, ny, nx = int(rng.integers(3, 30)), int(rng.integers(2, 12)), int(rng.integers(2, 16))
        density = float(rng.choice([0.02, 0.1, 0.3, 0.6]))
        field = rng.random((nt, ny, nx)) < density
        cuts = np.sort(rng.choice(np.arange(1, nt), size=min(nt - 1, int(rng.integers(0, 8))), replace=False))
        chunks = tuple(np.diff(np.concatenate(([0], cuts, [nt]))).tolist())
        cases.append((f"random{seed}", field, chunks, bool(seed % 2)))
    return cases


def _edge_label_cases():
    empty = np.zeros((6, 4, 5), dtype=bool)
    full = np.ones((6, 4, 5), dtype=bool)
    # Seam contacts across a time boundary, both diagonals: (t, y, x=-1) touches (t+1, y+-1, x=0).
    corner = np.zeros((6, 5, 7), dtype=bool)
    corner[2, 1, -1] = corner[3, 2, 0] = True  # crosses the boundary between chunks (3,) and (3,)
    corner[2, 4, 0] = corner[3, 3, -1] = True
    # Two objects that only meet through a chain spanning three chunks.
    chain = np.zeros((9, 3, 3), dtype=bool)
    chain[0:9, 0, 0] = True
    chain[4, 2, 2] = chain[5, 1, 1] = True
    return [
        ("empty", empty, (2, 2, 2), True),
        ("all_true", full, (2, 2, 2), True),
        ("all_true_one_chunk", full, (6,), False),
        ("seam_corner_wrapped", corner, (3, 3), True),
        ("seam_corner_regional", corner, (3, 3), False),
        ("seam_corner_time_chunk_1", corner, (1,) * 6, True),
        ("chain_across_chunks", chain, (3, 3, 3), True),
        ("chain_time_chunk_1", chain, (1,) * 9, False),
    ]


@pytest.mark.parametrize("table", ["random", "edge_cases"])
def test_label_time_connected_matches_dask_image(table):
    """Labels and count equal dask-image's on 24 random fields with random time chunkings, and on named edge
    cases (empty, all-true, seam corners across a time boundary, chains spanning chunks), with the input's
    chunks and an int32 dtype."""
    cases = _random_label_cases() if table == "random" else _edge_label_cases()
    failures = []
    for case, field, time_chunks, wrap_x in cases:

        def row(field=field, time_chunks=time_chunks, wrap_x=wrap_x):
            data = da.from_array(field, chunks=(time_chunks, -1, -1))
            ref_labels, ref_n = dask_image_label(data, structure=FULL, wrap_axes=(2,) if wrap_x else None)
            labels, n = label_time_connected(data, wrap_x)
            assert labels.chunks == data.chunks, labels.chunks
            assert labels.dtype == np.int32, labels.dtype
            assert int(n.compute()) == int(ref_n.compute()), (int(n.compute()), int(ref_n.compute()))
            np.testing.assert_array_equal(np.asarray(labels.compute()), np.asarray(ref_labels.compute()))

        check(failures, case, row)
    assert_no_failures(failures)


def _n_label_tasks(n_chunks):
    field = da.from_array(np.zeros((4 * n_chunks, 8, 12), dtype=bool), chunks=(4, -1, -1))
    labels, n = label_time_connected(field, wrap_x=True)
    return len(set(dict(labels.__dask_graph__())) | set(dict(n.__dask_graph__())))


def test_label_time_connected_task_count_is_linear():
    counts = {b: _n_label_tasks(b) for b in (8, 16, 32, 64)}
    per_chunk = {b: (counts[b] - counts[8]) / (b - 8) for b in (16, 32, 64)}
    # Constant tasks per added chunk: linear, with no term that grows with the chunk count.
    assert len(set(per_chunk.values())) == 1, counts
    assert per_chunk[64] <= 10, counts


def test_resolving_task_receives_no_time_planes():
    """The resolving task must get counts and contacts only, never a chunk's whole label planes.

    Checked on what ``_resolve`` is actually called with, not on graph key names: a plane routed through
    any intermediate task would pass a dependency-name check.
    """
    module = sys.modules[label_time_connected.__module__]
    original, received = module._resolve, []

    def spy(*args):
        received.append(args)
        return original(*args)

    module._resolve = spy
    try:
        field = da.from_array(np.random.default_rng(0).random((16, 8, 12)) < 0.2, chunks=(4, -1, -1))
        labels, n_objects = label_time_connected(field, wrap_x=True)
        dask.compute(labels, n_objects, scheduler="synchronous")
    finally:
        module._resolve = original
    assert len(received) == 1
    leaves = [leaf for arg in received[0] for leaf in (arg if isinstance(arg, (list, tuple)) else [arg])]
    pairs = [leaf for leaf in leaves if isinstance(leaf, np.ndarray)]
    assert pairs and sum(len(p) for p in pairs) > 0, "the field must produce contacts for this check to bite"
    for leaf in leaves:
        if isinstance(leaf, np.ndarray):
            assert leaf.ndim == 2 and leaf.shape[1] == 2 and leaf.dtype == np.int64, (leaf.shape, leaf.dtype)
        else:
            assert isinstance(leaf, (int, np.integer)), type(leaf)


def test_label_time_connected_rejects_spatial_chunks():
    data = da.zeros((4, 6, 6), dtype=bool, chunks=(2, 3, -1))
    with pytest.raises(ValueError, match="whole in y and x"):
        label_time_connected(data, wrap_x=True)


# -- per-slice identification -------------------------------------------------------------------------


def _identify(field, regional_mode, time_chunk=1):
    data_bin = xr.DataArray(field, dims=("time", "lat", "lon")).chunk({"time": time_chunk, "lat": -1, "lon": -1})
    mask = xr.DataArray(np.ones(field.shape[1:], dtype=bool), dims=("lat", "lon"))
    ids, _, n_objects = identify_objects(data_bin, False, False, mask, None, "lon", regional_mode)
    return ids.values, n_objects


def test_foreground_only_slices_are_objects():
    """A gridded slice with no background cell is one object, not background.

    Per-slice labelling compacts the local labels to 1..k. A slice that is foreground everywhere has a single
    label and no 0, so a remap that assumes background is present would send its only object to 0 and the
    whole slice would vanish. Rows: an all-foreground record, and a partial / full / partial record at time
    chunks 1, 2 and 3, each regional and global.
    """
    partial_full_partial = np.zeros((3, 3, 4), dtype=bool)
    partial_full_partial[0, 0:2, 1:3] = True  # one object, interior so no seam merge
    partial_full_partial[1] = True  # no background at all
    partial_full_partial[2, 1, 0] = True  # two objects, not joined across the longitude seam either
    partial_full_partial[2, 1, 2] = True

    failures = []
    for regional_mode in (True, False):
        mode = "regional" if regional_mode else "global"

        def all_foreground(regional_mode=regional_mode):
            ids, n_objects = _identify(np.ones((2, 3, 4), dtype=bool), regional_mode)
            assert int((ids > 0).sum()) == 24, int((ids > 0).sum())
            assert n_objects == 2, n_objects
            assert np.unique(ids).tolist() == [1, 2], np.unique(ids).tolist()
            assert (ids[0] == 1).all() and (ids[1] == 2).all()

        check(failures, f"all_foreground-{mode}", all_foreground)
        for time_chunk in (1, 2, 3):

            def partial(regional_mode=regional_mode, time_chunk=time_chunk):
                field = partial_full_partial
                ids, n_objects = _identify(field, regional_mode, time_chunk)
                np.testing.assert_array_equal(ids > 0, field)
                assert (ids[0][field[0]] == 1).all()
                assert (ids[1] == 2).all()
                assert sorted(np.unique(ids[2][field[2]]).tolist()) == [3, 4]
                assert n_objects == 4, n_objects

            check(failures, f"partial_full_partial-{mode}-chunk{time_chunk}", partial)
    assert_no_failures(failures)


# -- prefilter_min_cells: dropping small components before the closing ------------------------------

EIGHT = np.ones((3, 3), dtype=bool)


def oracle_unstructured(b, nb, min_cells):
    """Reference speck filter over a neighbour graph (nb: (3, ncells), 0-based, -1 = none)."""
    out = np.zeros_like(b)
    n = b.shape[-1]
    for t in range(b.shape[0]):
        x = b[t]
        idx = np.flatnonzero(x)
        if idx.size == 0:
            continue
        lut = -np.ones(n, np.int64)
        lut[idx] = np.arange(idx.size)
        cols = nb[:, idx].T.ravel()
        rows = np.repeat(np.arange(idx.size), 3)
        ok = (cols >= 0) & (cols < n)
        ok[ok] &= x[cols[ok]]
        g = coo_matrix((np.ones(ok.sum(), bool), (rows[ok], lut[cols[ok]])), shape=(idx.size, idx.size)).tocsr()
        _, lab = connected_components(g, directed=False)
        keep = np.bincount(lab) >= min_cells
        out[t][idx[keep[lab]]] = True
    return out


def oracle_gridded(b, min_cells, periodic):
    """Reference speck filter on a lat/lon slice stack. Periodic: label a 3x-tiled copy in longitude, then
    union the three copies of every physical cell, which gives exact cylinder connectivity (including objects
    joined only by going all the way round)."""
    out = np.zeros_like(b)
    nx = b.shape[-1]
    for t in range(b.shape[0]):
        x = np.concatenate([b[t]] * 3, axis=-1) if periodic else b[t]
        lab, n = scipy_label(x, structure=EIGHT)
        if periodic:
            parent = np.arange(n + 1)

            def find(i, parent=parent):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for k in (0, 2):
                a_, b_ = lab[:, nx : 2 * nx], lab[:, k * nx : (k + 1) * nx]
                for u, v in set(zip(a_[a_ > 0].tolist(), b_[a_ > 0].tolist())):
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[ru] = rv
            roots = np.array([find(i) for i in range(n + 1)])
            mid = roots[lab[:, nx : 2 * nx]]
            mid[lab[:, nx : 2 * nx] == 0] = 0
            counts = np.bincount(mid.ravel(), minlength=n + 1)
            keep = counts >= min_cells
            keep[0] = False
            out[t] = keep[mid]
        else:
            counts = np.bincount(lab.ravel())
            keep = counts >= min_cells
            keep[0] = False
            out[t] = keep[lab]
    return out


def _gridded_da(arr):
    return xr.DataArray(arr, dims=("time", "lat", "lon")).chunk({"time": 2, "lat": -1, "lon": -1})


def test_drop_small_components_hand_cases():
    """A speck goes, a blob of exactly ``min_cells`` stays (ties kept), the longitude seam joins on a global
    grid but not a regional one, and diagonal neighbours are connected."""
    failures = []

    def speck_and_tie():
        a = np.zeros((1, 10, 12), bool)
        a[0, 1, 1] = True  # 1-cell speck
        a[0, 4:6, 4:6] = True  # 4-cell blob, exactly min_cells
        a[0, 7:9, 7:10] = True  # 6-cell blob
        out = drop_small_components(_gridded_da(a), 4, False, "lon", "lat", regional_mode=True).values
        assert not out[0, 1, 1]
        assert out[0, 4:6, 4:6].all() and out[0, 7:9, 7:10].all()
        assert out.sum() == 10, out.sum()

    def seam():
        a = np.zeros((1, 6, 12), bool)
        a[0, 2:4, 0] = True  # 2 cells on the west edge
        a[0, 2:4, -1] = True  # 2 cells on the east edge: one 4-cell object across the seam
        glob = drop_small_components(_gridded_da(a), 4, False, "lon", "lat", regional_mode=False).values
        reg = drop_small_components(_gridded_da(a), 4, False, "lon", "lat", regional_mode=True).values
        assert glob.sum() == 4, glob.sum()
        assert reg.sum() == 0, reg.sum()

    def diagonal():
        a = np.zeros((1, 6, 6), bool)
        a[0, 1, 1] = a[0, 2, 2] = True
        out = drop_small_components(_gridded_da(a), 2, False, "lon", "lat", regional_mode=True).values
        assert out.sum() == 2, out.sum()

    for case, func in (("speck_and_tie", speck_and_tie), ("seam", seam), ("diagonal", diagonal)):
        check(failures, case, func)
    assert_no_failures(failures)


def test_drop_small_components_matches_oracle():
    """Random 6 x 40 x 60 fields, periodic and regional: the filter equals the oracle, and the result is lazy
    with the input's chunks and a boolean dtype."""
    rng = np.random.default_rng(7)
    a = rng.random((6, 40, 60)) < 0.35
    failures = []
    for periodic in (True, False):

        def row(periodic=periodic):
            data = _gridded_da(a)
            out = drop_small_components(data, 9, False, "lon", "lat", regional_mode=not periodic)
            assert out.chunks == data.chunks, out.chunks
            assert out.dtype == bool, out.dtype
            np.testing.assert_array_equal(out.values, oracle_gridded(a, 9, periodic))

        check(failures, f"periodic={periodic}", row)
    assert_no_failures(failures)


def test_drop_small_components_unstructured_matches_oracle():
    """On the shipped mesh: random fields at three densities and four ``min_cells`` equal the oracle, and
    ``min_cells=1`` returns the shipped field unchanged."""
    ds = open_unstructured()
    nb = (ds.neighbours.astype(np.int32) - 1).load()
    rng = np.random.default_rng(3)
    n = ds.sizes["ncells"]
    failures = []
    for p in (0.2, 0.45, 0.7):
        a = rng.random((5, n)) < p
        data = xr.DataArray(a, dims=("time", "ncells")).chunk({"time": 2, "ncells": -1})
        for m in (1, 2, 5, 23):

            def row(a=a, data=data, m=m):
                got = drop_small_components(data, m, True, "ncells", None, False, nb).values
                np.testing.assert_array_equal(got, oracle_unstructured(a, nb.values, m))

            check(failures, f"p={p} m={m}", row)

    def identity():
        shipped = ds.extreme_events.values
        data = xr.DataArray(shipped, dims=("time", "ncells")).chunk({"time": 7, "ncells": -1})
        np.testing.assert_array_equal(drop_small_components(data, 1, True, "ncells", None, False, nb).values, shipped)

    check(failures, "min_cells=1 on the shipped field", identity)
    assert_no_failures(failures)


PREFILTER_GRIDDED = {
    "R_fill": 4,
    "area_filter_absolute": 30,
    "T_fill": 2,
    "allow_merging": True,
    "nn_partitioning": True,
    "quiet": True,
}


def test_tracker_prefilter_gridded(client, tmp_path):
    """``prefilter_min_cells`` is exactly 'filter the input, then track' -- nothing else moves -- in persist
    and in streaming."""
    assert on_client(client)
    ds = open_gridded()
    ev = ds.extreme_events.chunk({"time": 4, "lat": -1, "lon": -1})
    pre = xr.DataArray(oracle_gridded(ev.values, 6, periodic=True), coords=ev.coords, dims=ev.dims).chunk(ev.chunks)
    assert 0 < int(pre.sum()) < int(ev.sum()), "the store must actually lose some specks"

    prefiltered = marEx.tracker(ev, ds.mask, prefilter_min_cells=6, **PREFILTER_GRIDDED).run().compute()
    tracked_pre = marEx.tracker(pre.astype(bool), ds.mask, **PREFILTER_GRIDDED).run().compute()
    lazy = marEx.tracker(
        ev, ds.mask, prefilter_min_cells=6, compute_mode="streaming", temp_dir=str(tmp_path), **PREFILTER_GRIDDED
    ).run()
    streamed = np.asarray(lazy.ID_field.values)
    marEx.clear_staging(lazy)
    del lazy

    np.testing.assert_array_equal(prefiltered.ID_field.values, tracked_pre.ID_field.values)
    assert prefiltered.attrs["prefilter_min_cells"] == 6
    assert "prefilter_min_cells" not in tracked_pre.attrs
    np.testing.assert_array_equal(prefiltered.ID_field.values, streamed, err_msg="streaming differs from persist")


def test_tracker_prefilter_unstructured(client, tmp_path):
    """The same identity on the mesh: tracking with ``prefilter_min_cells=3`` equals tracking the
    oracle-filtered input."""
    assert on_client(client)
    ds = open_unstructured()
    ev = ds.extreme_events.chunk({"time": 10, "ncells": -1})
    nb = (ds.neighbours.astype(np.int32) - 1).values
    pre = xr.DataArray(oracle_unstructured(ev.values, nb, 3), coords=ev.coords, dims=ev.dims).chunk(ev.chunks)
    assert 0 < int(pre.sum()) < int(ev.sum()), "the store must actually lose some specks"
    kwargs = {
        "R_fill": 2,
        "area_filter_absolute": 4,
        "T_fill": 2,
        "allow_merging": True,
        "nn_partitioning": True,
        "unstructured_grid": True,
        "dimensions": {"x": "ncells"},
        "coordinates": {"x": "lon", "y": "lat"},
        "coordinate_units": "degrees",
        "neighbours": ds.neighbours,
        "cell_areas": ds.cell_areas,
        "quiet": True,
    }
    a = marEx.tracker(ev, ds.mask, prefilter_min_cells=3, temp_dir=str(tmp_path / "a"), **kwargs).run().compute()
    b = marEx.tracker(pre, ds.mask, temp_dir=str(tmp_path / "b"), **kwargs).run().compute()
    np.testing.assert_array_equal(a.ID_field.values, b.ID_field.values)
    assert a.attrs["prefilter_min_cells"] == 3


# -- the gridded area census -------------------------------------------------------------------------


def _census_field(n_time, ny, nx):
    """Objects in four slices spread over three time chunks; none after t=13, so the count is fixed."""
    field = np.zeros((n_time, ny, nx), dtype=bool)
    field[0, 1:3, 1:3] = True
    field[1, 3:5, 4:7] = True
    field[1, 0, 0] = True
    field[7, 0:2, 5:8] = True
    field[7, 4, 1] = True
    field[13, 2:6, 0:2] = True
    field[13, 0, 7] = True
    return field


def _census_oracle(field, threshold):
    """Per-slice 8-connected areas in labelling order (regional, no seam), and the kept mask."""
    structure = np.ones((3, 3), dtype=int)
    areas, kept = [], np.zeros_like(field)
    for t, frame in enumerate(field):
        labels, _ = scipy_label(frame, structure=structure)
        counts = np.bincount(labels.ravel())
        areas.extend(counts[1:][counts[1:] > 0].tolist())
        keep = counts >= threshold
        keep[0] = False
        kept[t] = keep[labels]
    return areas, kept


def _gathered_bytes_in_filter(n_time, absolute, monkeypatch):
    """Bytes returned to the driver by every outermost compute inside gridded filter_small_objects."""
    from marEx.track import morphology

    ny, nx = 6, 8
    field = _census_field(n_time, ny, nx)
    data_bin = xr.DataArray(field, dims=("time", "lat", "lon")).chunk({"time": 5, "lat": -1, "lon": -1})
    lat = xr.DataArray(np.linspace(-10, 10, ny), dims="lat")
    lon = xr.DataArray(np.linspace(0, 20, nx), dims="lon")
    mask = xr.DataArray(np.ones((ny, nx), dtype=bool), dims=("lat", "lon"))

    gathered = []
    depth = [0]

    def nbytes(obj):
        if isinstance(obj, (tuple, list)):
            return sum(nbytes(o) for o in obj)
        return int(getattr(obj, "nbytes", 0))

    def wrap(fn):
        def spy(*args, **kwargs):
            depth[0] += 1
            try:
                out = fn(*args, **kwargs)
            finally:
                depth[0] -= 1
            if depth[0] == 0:
                gathered.append(nbytes(out))
            return out

        return spy

    with monkeypatch.context() as patched:
        patched.setattr(xr.DataArray, "compute", wrap(xr.DataArray.compute))
        patched.setattr(xr.Dataset, "compute", wrap(xr.Dataset.compute))
        patched.setattr(da.Array, "compute", wrap(da.Array.compute))
        patched.setattr(dask, "compute", wrap(dask.compute))
        patched.setattr(dask.base, "compute", wrap(dask.base.compute))
        patched.setattr(da, "compute", wrap(da.compute))
        out = morphology.filter_small_objects(
            data_bin, False, "lon", absolute, 2, 0.5, mask, None, True, lat, lon, None, "time", "lat"
        )
    return sum(gathered), out


def test_area_census_gather_is_flat_in_record(monkeypatch):
    """The gridded small-object census gathers compact areas, never the padded per-slice buffer: the same
    objects in twice the record gather no more bytes, and the census is exact and time-major across time
    chunks. Quartile and absolute filtering."""
    failures = []
    for absolute in (False, True):

        def row(absolute=absolute):
            short, out_short = _gathered_bytes_in_filter(20, absolute, monkeypatch)
            long, out_long = _gathered_bytes_in_filter(40, absolute, monkeypatch)
            # Same objects, twice the record: what reaches the driver must not double.
            assert long <= short + 1024, f"gathered {short} B at nt=20 but {long} B at nt=40"
            assert short < 20 * 8192 * 8, f"gathered {short} B: the padded census reached the driver"
            for n_time, out in ((20, out_short), (40, out_long)):
                filtered, threshold, areas, n_unfiltered, n_filtered = out
                oracle_areas, _ = _census_oracle(_census_field(n_time, 6, 8), 0)
                assert areas.dtype == np.int64
                assert areas.values.tolist() == oracle_areas == [4, 1, 6, 6, 1, 1, 8], areas.values.tolist()
                assert threshold == (2 if absolute else np.percentile(oracle_areas, 50.0)), threshold
                assert n_unfiltered == 7, n_unfiltered
                assert n_filtered == sum(a >= threshold for a in oracle_areas), n_filtered
                _, oracle_kept = _census_oracle(_census_field(n_time, 6, 8), threshold)
                np.testing.assert_array_equal(filtered.values, oracle_kept)

        check(failures, "absolute" if absolute else "quartile", row)
    assert_no_failures(failures)


# -- the absolute area filter --------------------------------------------------------------------------
#
# The mode the example notebooks and the full-scale runs use (area_filter_absolute=600), and the branch that
# fuses the area census and the keep-mask into one labelling pass. The reference re-derives the outputs from
# scipy labelling + np.bincount per slice.


def _absolute_field(n_time=3, ny=24, nx=40, seed=0):
    """Binary field with a spread of object sizes, including some crossing the seam."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n_time, ny, nx), dtype=bool)
    for t in range(n_time):
        for size in (2, 3, 5, 7):
            y = int(rng.integers(0, ny - size))
            x = int(rng.integers(0, nx - size))
            data[t, y : y + size, x : x + size] = True
        # one object straddling the antimeridian seam, so the seam merge is exercised
        data[t, 4:8, :3] = True
        data[t, 4:8, -3:] = True
    return data


def _absolute_reference(data, threshold, regional_mode):
    """Per-slice 8-connected labelling, pixel counts, keep >= threshold."""
    keep_field = np.zeros_like(data)
    all_areas = []
    for t in range(data.shape[0]):
        labels, n_labels = scipy_label(data[t], structure=_EIGHT_CONNECTIVITY)
        if not regional_mode:
            labels = _merge_lon_seam(labels, n_labels)
        counts = np.bincount(labels.ravel())
        all_areas.append(counts[1:][counts[1:] > 0])
        keep = counts >= threshold
        keep[0] = False
        keep_field[t] = keep[labels]
    areas_np = np.concatenate(all_areas)
    return keep_field, areas_np, int(areas_np.size), int(np.sum(areas_np >= threshold))


def _absolute_filter(data, threshold, regional_mode, chunks=1):
    from marEx.track.morphology import filter_small_objects

    n_time, ny, nx = data.shape
    field = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(n_time), "lat": np.linspace(-10, 10, ny), "lon": np.linspace(0, 359, nx)},
    ).chunk({"time": chunks})
    mask = xr.DataArray(np.ones((ny, nx), dtype=bool), dims=("lat", "lon"), coords={"lat": field.lat, "lon": field.lon})
    cell_area = xr.DataArray(np.ones((ny, nx), dtype=np.float32), dims=("lat", "lon"), coords={"lat": field.lat, "lon": field.lon})
    return filter_small_objects(
        data_bin=field,
        unstructured_grid=False,
        xdim="lon",
        use_absolute_filtering=True,
        area_filter_absolute=threshold,
        area_filter_quartile=0.0,
        mask=mask,
        neighbours_int=None,
        regional_mode=regional_mode,
        lat=field.lat,
        lon=field.lon,
        cell_area=cell_area,
        timedim="time",
        ydim="lat",
    )


def test_absolute_filter_matches_reference():
    """The single-pass absolute branch reproduces label + bincount + threshold at thresholds 4, 9 and 25,
    regional and global; and an object of exactly the threshold size is kept (>=)."""
    data = _absolute_field()
    failures = []
    for threshold in (4, 9, 25):
        for regional_mode in (False, True):

            def row(threshold=threshold, regional_mode=regional_mode):
                filtered, area_threshold, object_areas, n_unfiltered, n_filtered = _absolute_filter(data, threshold, regional_mode)
                exp_field, exp_areas, exp_unfiltered, exp_filtered = _absolute_reference(data, threshold, regional_mode)
                np.testing.assert_array_equal(filtered.values, exp_field, err_msg="filtered field differs from the reference")
                np.testing.assert_array_equal(np.sort(object_areas.values), np.sort(exp_areas), err_msg="object areas differ")
                assert area_threshold == threshold, area_threshold
                assert n_unfiltered == exp_unfiltered, (n_unfiltered, exp_unfiltered)
                assert n_filtered == exp_filtered, (n_filtered, exp_filtered)

            check(failures, f"threshold={threshold} regional={regional_mode}", row)

    def tie():
        field = np.zeros((1, 12, 12), dtype=bool)
        field[0, 2:4, 2:4] = True  # exactly 4 cells
        field[0, 7:8, 7:8] = True  # 1 cell, must be dropped
        filtered, _, object_areas, n_unfiltered, n_filtered = _absolute_filter(field, 4, regional_mode=True)
        assert filtered.values.sum() == 4, "the 4-cell object must survive an area_filter_absolute of 4"
        assert (n_unfiltered, n_filtered) == (2, 1), (n_unfiltered, n_filtered)
        assert sorted(object_areas.values.tolist()) == [1, 4]

    check(failures, "tie", tie)
    assert_no_failures(failures)


def test_absolute_filter_is_chunk_invariant():
    """Chunking the time axis must not change what the filter keeps."""
    data = _absolute_field(n_time=4, seed=3)
    one_chunk = _absolute_filter(data, 9, regional_mode=False, chunks=4)
    many_chunks = _absolute_filter(data, 9, regional_mode=False, chunks=1)

    np.testing.assert_array_equal(one_chunk[0].values, many_chunks[0].values)
    np.testing.assert_array_equal(np.sort(one_chunk[2].values), np.sort(many_chunks[2].values))
    assert one_chunk[3] == many_chunks[3]
    assert one_chunk[4] == many_chunks[4]


# -- hole filling's sparse matrix --------------------------------------------------------------------


def test_sparse_matrix_core_dims_do_not_multiply_chunks():
    """The unstructured hole filling passes the sparse dilation matrix to apply_ufunc as core dimensions.

    Left to dask's automatic chunking, a large mesh splits those arrays (``indices`` is ~0.8 GB at ICON
    R02B09), and apply_gufunc has to merge them back while re-splitting the loop dimension, raising
    "Increasing number of chunks". A small ``array.chunk-size`` reproduces that on a test mesh.
    """
    nb, _, _, _ = build_mesh(20, 70)
    neighbours = (xr.DataArray(nb, dims=("nv", "ncells")).astype(np.int32) - 1).chunk(-1)  # 0-based, as the tracker holds it
    ncells = nb.shape[1]
    rng = np.random.default_rng(0)
    data = xr.DataArray(rng.random((10, ncells)) > 0.6, dims=("time", "ncells")).chunk({"time": 5, "ncells": -1})
    mask = xr.DataArray(np.ones(ncells, bool), dims="ncells").chunk(-1)
    with dask.config.set({"array.chunk-size": "2KiB"}), warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        filled = fill_holes(data, 1, True, build_sparse_dilation_matrix(neighbours), "ncells", mask, False, None)
    assert filled.chunks == data.chunks
    with dask.config.set(scheduler="synchronous"):
        reference = fill_holes(data, 1, True, build_sparse_dilation_matrix(neighbours), "ncells", mask, False, None)
        np.testing.assert_array_equal(filled.values, reference.values)


# -- time-gap filling keeps the input's time chunking -------------------------------------------------
#
# ``fill_time_gaps`` pads the record by ``T_fill + 1`` and hands it to dask_image's ``binary_closing``,
# whose overlap step calls ``ensure_minimum_chunksize``: any chunk shorter than the kernel depth is refilled
# from its neighbour, and nothing realigns it afterwards. The trigger is arithmetic,
# ``0 < n_time % time_chunk < (T_fill + 1) // 2``, and which ILLEGAL layout comes back depends on the width
# (T_fill=4, depth 2, on the gridded store):
#
#     n_time=21,  chunk=5   ->  (5, 5, 5, 4, 2)      ragged INTERIOR
#     n_time=101, chunk=25  ->  (25, 25, 25, 24, 2)  ragged INTERIOR
#     n_time=13,  chunk=3   ->  (3, 3, 3, 4)         oversized FINAL
#
# Zarr accepts a short final chunk and nothing else, so both are fatal: ``ObjectIDRegionWriter._initialise``
# writes the full-length template with ``to_zarr``, and a streaming run dies mid-merge. The production crash
# was the ragged-interior form (n_time=951, time chunk 25, T_fill=4). Realign to the input's own boundaries
# rather than trusting what an overlap operation hands back.

# (case, n_time, time_chunk, T_fill, does the borrow fire?)
GAP_FILL_LAYOUTS = [
    ("ragged_interior_trigger", 21, 5, 4, True),
    ("oversized_final_trigger", 13, 3, 4, True),
    ("no_remainder", 20, 5, 4, False),
    ("remainder_at_depth", 22, 5, 4, False),
    ("deeper_kernel", 22, 5, 8, True),  # T_fill=8 widens the depth to 4, so remainders 1..3 all trigger
]


def _fill(data_bin, mask, t_fill):
    return fill_time_gaps(
        data_bin,
        T_fill=t_fill,
        R_fill=8,
        timedim="time",
        ydim="lat",
        unstructured_grid=False,
        dilate_sparse=None,
        xdim="lon",
        mask=mask,
        regional_mode=False,
    )


def test_fill_time_gaps_preserves_time_chunking(tmp_path):
    """The filled field has exactly the input's time chunking on every row, and the field filled on each
    triggering row writes to zarr: asserted against zarr itself, not only against a model of what it allows.

    ``got == expected`` is the load-bearing assertion: a check that the result is merely zarr-legal passes on
    the pre-fix code at 13 / 3, whose (3, 3, 3, 4) has a uniform interior.
    """
    ds = open_gridded()
    failures = []
    for case, nt, chunk, t_fill, triggers in GAP_FILL_LAYOUTS:
        data_bin = tile_to(ds.extreme_events, nt).chunk({"time": chunk, "lat": -1, "lon": -1})
        expected = data_bin.chunks[data_bin.dims.index("time")]

        def label(nt=nt, chunk=chunk, t_fill=t_fill, triggers=triggers):
            assert (0 < nt % chunk < (t_fill + 1) // 2) is triggers, "the row no longer matches its trigger label"

        def chunking(case=case, data_bin=data_bin, t_fill=t_fill, expected=expected):
            filled = _fill(data_bin, ds.mask, t_fill)
            got = filled.chunks[filled.dims.index("time")]
            assert got == expected, f"not realigned: {got} != {expected}"

        check(failures, f"{case} ({nt}/{chunk}) label", label)
        check(failures, f"{case} ({nt}/{chunk}) chunking", chunking)
        if triggers and t_fill == 4:

            def writes(case=case, data_bin=data_bin, t_fill=t_fill):
                out = _fill(data_bin, ds.mask, t_fill).rename("filled").to_dataset()
                # The store is format 2: its codecs ride along in `.encoding` and zarr 3 refuses them in a format-3
                # store. That is not what this checks, so drop them as any caller writing an intermediate must.
                clear_store_encoding(out)
                out.to_zarr(str(tmp_path / f"{case}.zarr"), mode="w")

            check(failures, f"{case} ({nt}/{chunk}) to_zarr", writes)
    assert_no_failures(failures)
