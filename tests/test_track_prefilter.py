"""`prefilter_min_cells`: dropping small components before the morphological closing."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import marEx
from marEx.exceptions import ConfigurationError
from marEx.track.morphology import drop_small_components

TEST_DATA_DIR = Path(__file__).parent / "data"
EIGHT = np.ones((3, 3), dtype=bool)


@pytest.fixture(scope="module")
def extremes():
    return xr.open_zarr(str(TEST_DATA_DIR / "extremes_gridded.zarr"), chunks={})


@pytest.fixture(scope="module")
def extremes_unstructured():
    return xr.open_zarr(str(TEST_DATA_DIR / "extremes_unstructured.zarr"), chunks={})


# -- Oracles: written independently of the implementation ------------------------------


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
        g = coo_matrix(
            (np.ones(ok.sum(), bool), (rows[ok], lut[cols[ok]])),
            shape=(idx.size, idx.size),
        ).tocsr()
        _, lab = connected_components(g, directed=False)
        keep = np.bincount(lab) >= min_cells
        out[t][idx[keep[lab]]] = True
    return out


def oracle_gridded(b, min_cells, periodic):
    """Reference speck filter on a lat/lon slice stack. Periodic: label a 3x-tiled copy in
    longitude, then union the three copies of every physical cell, which gives exact cylinder
    connectivity (including objects joined only by going all the way round)."""
    out = np.zeros_like(b)
    nx = b.shape[-1]
    for t in range(b.shape[0]):
        x = np.concatenate([b[t]] * 3, axis=-1) if periodic else b[t]
        lab, n = label(x, structure=EIGHT)
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


# -- drop_small_components --------------------------------------------------------------


class TestDropSmallComponentsGridded:
    def _da(self, arr):
        return xr.DataArray(arr, dims=("time", "lat", "lon")).chunk({"time": 2, "lat": -1, "lon": -1})

    def test_drops_speck_keeps_blob_and_tie(self):
        a = np.zeros((1, 10, 12), bool)
        a[0, 1, 1] = True  # 1-cell speck
        a[0, 4:6, 4:6] = True  # 4-cell blob, exactly min_cells
        a[0, 7:9, 7:10] = True  # 6-cell blob
        out = drop_small_components(self._da(a), 4, False, "lon", "lat", regional_mode=True).values
        assert not out[0, 1, 1]
        assert out[0, 4:6, 4:6].all() and out[0, 7:9, 7:10].all()
        assert out.sum() == 10

    def test_seam_joins_on_global_grid_but_not_regional(self):
        a = np.zeros((1, 6, 12), bool)
        a[0, 2:4, 0] = True  # 2 cells on the west edge
        a[0, 2:4, -1] = True  # 2 cells on the east edge: one 4-cell object across the seam
        da = self._da(a)
        glob = drop_small_components(da, 4, False, "lon", "lat", regional_mode=False).values
        reg = drop_small_components(da, 4, False, "lon", "lat", regional_mode=True).values
        assert glob.sum() == 4
        assert reg.sum() == 0

    def test_diagonal_is_connected(self):
        a = np.zeros((1, 6, 6), bool)
        a[0, 1, 1] = a[0, 2, 2] = True
        out = drop_small_components(self._da(a), 2, False, "lon", "lat", regional_mode=True).values
        assert out.sum() == 2

    @pytest.mark.parametrize("periodic", [True, False])
    def test_matches_oracle_on_random_fields(self, periodic):
        rng = np.random.default_rng(7)
        a = rng.random((6, 40, 60)) < 0.35
        got = drop_small_components(self._da(a), 9, False, "lon", "lat", regional_mode=not periodic).values
        np.testing.assert_array_equal(got, oracle_gridded(a, 9, periodic))

    def test_lazy_and_chunks_preserved(self):
        a = np.zeros((6, 8, 8), bool)
        out = drop_small_components(self._da(a), 3, False, "lon", "lat", regional_mode=False)
        assert out.chunks == self._da(a).chunks
        assert out.dtype == bool


class TestDropSmallComponentsUnstructured:
    def test_matches_oracle_on_fixture_mesh(self, extremes_unstructured):
        nb = (extremes_unstructured.neighbours.astype(np.int32) - 1).load()
        rng = np.random.default_rng(3)
        n = extremes_unstructured.sizes["ncells"]
        for p in (0.2, 0.45, 0.7):
            a = rng.random((5, n)) < p
            da = xr.DataArray(a, dims=("time", "ncells")).chunk({"time": 2, "ncells": -1})
            for m in (1, 2, 5, 23):
                got = drop_small_components(da, m, True, "ncells", None, False, nb).values
                np.testing.assert_array_equal(got, oracle_unstructured(a, nb.values, m), err_msg=f"p={p} m={m}")

    def test_min_cells_one_is_identity(self, extremes_unstructured):
        nb = (extremes_unstructured.neighbours.astype(np.int32) - 1).load()
        a = extremes_unstructured.extreme_events.values
        da = xr.DataArray(a, dims=("time", "ncells")).chunk({"time": 7, "ncells": -1})
        np.testing.assert_array_equal(drop_small_components(da, 1, True, "ncells", None, False, nb).values, a)


# -- tracker integration ----------------------------------------------------------------

GRIDDED_KW = {"R_fill": 4, "area_filter_absolute": 30, "T_fill": 2, "allow_merging": True, "nn_partitioning": True, "quiet": True}


def _unstructured_kw(ds, temp_dir):
    return {
        "temp_dir": str(temp_dir),
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


class TestTrackerPrefilter:
    @pytest.mark.parametrize("bad", [0, -3, 2.5, True, "23"])
    def test_rejects_invalid(self, extremes, bad):
        with pytest.raises(ConfigurationError, match="prefilter_min_cells"):
            marEx.tracker(
                extremes.extreme_events.chunk({"time": 4}),
                extremes.mask,
                prefilter_min_cells=bad,
                **GRIDDED_KW,
            )

    def test_is_keyword_only(self):
        import inspect

        p = inspect.signature(marEx.tracker.__init__).parameters["prefilter_min_cells"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None

    def test_gridded_equals_tracking_a_prefiltered_input(self, extremes, dask_client):
        """The option must be exactly 'filter the input, then track' -- nothing else moves."""
        ev = extremes.extreme_events.chunk({"time": 4, "lat": -1, "lon": -1})
        pre = xr.DataArray(oracle_gridded(ev.values, 6, periodic=True), coords=ev.coords, dims=ev.dims).chunk(ev.chunks)
        assert 0 < int(pre.sum()) < int(ev.sum()), "fixture must actually lose some specks"
        a = marEx.tracker(ev, extremes.mask, prefilter_min_cells=6, **GRIDDED_KW).run()
        b = marEx.tracker(pre.astype(bool), extremes.mask, **GRIDDED_KW).run()
        np.testing.assert_array_equal(a.ID_field.values, b.ID_field.values)
        assert a.attrs["prefilter_min_cells"] == 6
        assert "prefilter_min_cells" not in b.attrs

    def test_gridded_persist_equals_streaming(self, extremes, dask_client, tmp_path):
        ev = extremes.extreme_events.chunk({"time": 4, "lat": -1, "lon": -1})
        a = marEx.tracker(ev, extremes.mask, prefilter_min_cells=6, **GRIDDED_KW).run()
        s = marEx.tracker(
            ev,
            extremes.mask,
            prefilter_min_cells=6,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **GRIDDED_KW,
        ).run()
        np.testing.assert_array_equal(a.ID_field.values, s.ID_field.values)
        marEx.clear_staging(s)

    def test_unstructured_equals_tracking_a_prefiltered_input(self, extremes_unstructured, dask_client_unstructured, tmp_path):
        ds = extremes_unstructured
        ev = ds.extreme_events.chunk({"time": 10, "ncells": -1})
        nb = (ds.neighbours.astype(np.int32) - 1).values
        pre = xr.DataArray(oracle_unstructured(ev.values, nb, 3), coords=ev.coords, dims=ev.dims).chunk(ev.chunks)
        assert 0 < int(pre.sum()) < int(ev.sum()), "fixture must actually lose some specks"
        a = marEx.tracker(ev, ds.mask, prefilter_min_cells=3, **_unstructured_kw(ds, tmp_path / "a")).run()
        b = marEx.tracker(pre, ds.mask, **_unstructured_kw(ds, tmp_path / "b")).run()
        np.testing.assert_array_equal(a.ID_field.values, b.ID_field.values)
        assert a.attrs["prefilter_min_cells"] == 3
