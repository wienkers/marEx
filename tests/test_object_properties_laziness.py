"""
Regression tests for ``calculate_object_properties`` on the unstructured path.

The bug these pin: the per-cell geometry arrays (``lat``, ``lon``, ``cell_area``) were
broadcast against the ``(time, xdim)`` object ID field with ``xr.broadcast`` before being
handed to ``xr.apply_ufunc``. Because the ID field is dask-backed and the geometry arrays
are not, ``xr.broadcast`` returns a **numpy** zero-stride view rather than a dask array,
and ``dask.array.from_array`` does an unconditional ``x.copy()`` on any array-like. That
copy densifies the view to ``(n_time, n_cells)`` float64 **at graph-build time, in the
client process** -- 122 GiB on the ICON R02B09 mesh at ``n_time=1096``.

The discriminating assertion is therefore *invariance in n_time*, not an absolute byte
bound: the buggy version materialises ``3 x n_time x n_cells x 8 B``, which passes any
bound generous enough for a short series and fails at scale.
"""

import dask.array as da
import dask.array.core as dask_array_core
import numpy as np
import pytest
import xarray as xr

from marEx.track.objects import calculate_object_properties

XDIM = "ncells"
TIMEDIM = "time"
N_CELLS = 400


def _geometry(n_cells=N_CELLS, seed=0):
    """1-D per-cell geometry arrays, as the unstructured tracker holds them."""
    rng = np.random.default_rng(seed)
    lat = xr.DataArray(rng.uniform(-80.0, 80.0, n_cells).astype(np.float64), dims=(XDIM,), name="lat")
    lon = xr.DataArray(rng.uniform(-180.0, 180.0, n_cells).astype(np.float64), dims=(XDIM,), name="lon")
    cell_area = xr.DataArray(rng.uniform(1.0, 2.0, n_cells).astype(np.float64), dims=(XDIM,), name="cell_area")
    return lat, lon, cell_area


def _id_field(n_time, n_cells=N_CELLS, seed=1, chunk_time=4):
    """Dask-backed unstructured ID field, IDs unique across time as the tracker makes them."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 4, size=(n_time, n_cells)).astype(np.int32)
    # Make IDs globally unique across time (mirrors the cumsum step in track_objects)
    offsets = (np.arange(n_time, dtype=np.int32) * 3)[:, None]
    ids = np.where(ids > 0, ids + offsets, 0).astype(np.int32)
    return xr.DataArray(
        da.from_array(ids, chunks=(chunk_time, n_cells)),
        dims=(TIMEDIM, XDIM),
        name="object_id_field",
    )


def _call(object_id_field, lat, lon, cell_area):
    return calculate_object_properties(
        object_id_field,
        True,  # unstructured_grid
        lat,
        lon,
        cell_area,
        TIMEDIM,
        False,  # regional_mode
        None,  # ydim
        XDIM,
        properties=["area", "centroid"],
    )


class _EagerWrapSpy:
    """Record every numpy array dask eagerly wraps (and therefore copies) via ``from_array``.

    ``dask.array.gufunc.apply_gufunc`` calls ``asarray`` on each argument, which resolves
    ``from_array`` inside ``dask.array.core``; patching the attribute on that module is what
    intercepts it (patching ``dask.array.from_array`` would miss the intra-module binding --
    the same import-binding trap that bites ``dask.persist``).
    """

    def __init__(self):
        self.wrapped = []

    def __enter__(self):
        self._orig = dask_array_core.from_array

        def spy(x, *args, **kwargs):
            if isinstance(x, np.ndarray):
                self.wrapped.append((tuple(x.shape), str(x.dtype), int(x.nbytes)))
            return self._orig(x, *args, **kwargs)

        dask_array_core.from_array = spy
        return self

    def __exit__(self, *exc):
        dask_array_core.from_array = self._orig
        return False

    @property
    def total_bytes(self):
        return sum(nbytes for _, _, nbytes in self.wrapped)


def _bytes_materialised(n_time):
    lat, lon, cell_area = _geometry()
    field = _id_field(n_time)
    with _EagerWrapSpy() as spy:
        props = _call(field, lat, lon, cell_area)
    return spy.total_bytes, spy.wrapped, props


def test_eager_materialisation_is_invariant_in_n_time():
    """Bytes eagerly copied at graph-build time must not grow with the length of the series.

    This is the assertion that discriminates the bug: the broadcast version materialised
    ``3 x n_time x n_cells x 8 B``, so an 8x longer series copied 8x more.
    """
    short_bytes, _, _ = _bytes_materialised(8)
    long_bytes, _, _ = _bytes_materialised(64)

    assert long_bytes == short_bytes, (
        f"eager materialisation grew with n_time: {short_bytes} B at n_time=8 vs "
        f"{long_bytes} B at n_time=64 -- a geometry array is being broadcast to the "
        f"whole field before apply_ufunc"
    )


def test_no_whole_field_array_is_eagerly_wrapped():
    """No eagerly-copied array may carry the time axis."""
    n_time = 64
    _, wrapped, _ = _bytes_materialised(n_time)

    offenders = [(shape, dtype, nbytes) for shape, dtype, nbytes in wrapped if n_time in shape]
    assert not offenders, f"whole-field arrays eagerly copied by dask.from_array: {offenders}"


def test_eager_materialisation_stays_below_one_field():
    """Sanity bound: less than a single whole int32 field is copied up front."""
    n_time = 64
    total, _, _ = _bytes_materialised(n_time)
    whole_field_bytes = n_time * N_CELLS * 4
    assert total < whole_field_bytes, f"{total} B eagerly copied, a whole int32 field is only {whole_field_bytes} B"


def test_properties_are_correct_on_a_known_field():
    """Areas must be the exact sum of the member cells' areas."""
    n_time = 4
    lat, lon, cell_area = _geometry()
    ids = np.zeros((n_time, N_CELLS), dtype=np.int32)
    ids[0, 0:10] = 7
    ids[1, 5:25] = 11
    ids[2, 100:103] = 13
    field = xr.DataArray(da.from_array(ids, chunks=(2, N_CELLS)), dims=(TIMEDIM, XDIM))

    props = _call(field, lat, lon, cell_area).compute()

    assert sorted(props.ID.values.tolist()) == [7, 11, 13]
    area_vals = cell_area.values
    for oid, sl in ((7, slice(0, 10)), (11, slice(5, 25)), (13, slice(100, 103))):
        expected = np.float32(area_vals[sl].astype(np.float32).sum())
        got = props.area.sel(ID=oid).values
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_single_timestep_with_size_one_time_axis():
    """A size-1 time axis takes the non-parallel branch and must not raise.

    Under the broadcast version this branch raised ``IndexError``: ``xr.broadcast`` ordered
    the result ``(xdim, time)`` while the ID field is ``(time, xdim)``, so the boolean mask
    and the geometry array had transposed shapes.
    """
    lat, lon, cell_area = _geometry()
    ids = np.zeros((1, N_CELLS), dtype=np.int32)
    ids[0, 3:9] = 5
    field = xr.DataArray(da.from_array(ids, chunks=(1, N_CELLS)), dims=(TIMEDIM, XDIM))

    props = _call(field, lat, lon, cell_area).compute()

    assert props.ID.values.tolist() == [5]
    expected = np.float32(cell_area.values[3:9].astype(np.float32).sum())
    np.testing.assert_allclose(props.area.values[0], expected, rtol=1e-6)


def test_single_timestep_without_time_dimension():
    """A field with no time dimension at all must also work."""
    lat, lon, cell_area = _geometry()
    ids = np.zeros(N_CELLS, dtype=np.int32)
    ids[10:20] = 3
    field = xr.DataArray(da.from_array(ids, chunks=(N_CELLS,)), dims=(XDIM,))

    props = _call(field, lat, lon, cell_area).compute()

    assert props.ID.values.tolist() == [3]
    expected = np.float32(cell_area.values[10:20].astype(np.float32).sum())
    np.testing.assert_allclose(props.area.values[0], expected, rtol=1e-6)


@pytest.mark.parametrize("n_time", [8, 32])
def test_geometry_core_dims_are_single_chunk(n_time):
    """Core dimensions must reach apply_gufunc in one chunk.

    ``asarray``'s ``chunks="auto"`` splits an array above dask's 128 MiB default; the ICON
    geometry arrays sit at 119 MB, 7% under that cliff. Passing them explicitly single-chunk
    removes the dependence on that margin. Here we assert the call succeeds and that nothing
    the spy sees is multi-chunk-able, i.e. the geometry arrives already dask-backed.
    """
    lat, lon, cell_area = _geometry()
    field = _id_field(n_time)
    with _EagerWrapSpy() as spy:
        props = _call(field, lat, lon, cell_area)
    props.compute()
    geometry_bytes = N_CELLS * 8
    assert spy.total_bytes <= geometry_bytes, f"expected at most one geometry array copied, got {spy.total_bytes} B"
