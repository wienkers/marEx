"""Compare spatial closing and opening with the original dilation operators."""

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import binary_closing, binary_opening, distance_transform_edt
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix

from marEx.exceptions import ConfigurationError
from marEx.track import morphology
from marEx.track.overlap import sparse_bool_power


def _edt_reference(plane, radius, regional):
    if radius == 0:
        return plane
    halo = 4 * radius
    lon_padded = np.pad(plane, ((0, 0), (halo, halo)), mode="edge" if regional else "wrap")
    padded = np.pad(lon_padded, ((halo, halo), (0, 0)), mode="edge")
    limit = radius * radius
    dilated = distance_transform_edt(~padded) ** 2 <= limit
    closed = distance_transform_edt(dilated) ** 2 > limit
    eroded = distance_transform_edt(closed) ** 2 > limit
    opened = distance_transform_edt(~eroded) ** 2 <= limit
    return opened[halo:-halo, halo:-halo]


def _structured(data, wet, radius, regional):
    return morphology.fill_holes(data, radius, False, None, "lon", wet, regional, "lat")


def _disk_reference(plane, radius, regional):
    offsets = np.arange(-radius, radius + 1)
    disk = offsets[:, None] ** 2 + offsets**2 <= radius**2
    halo = 4 * radius
    padded = np.pad(plane, ((0, 0), (halo, halo)), mode="edge" if regional else "wrap")
    padded = np.pad(padded, ((halo, halo), (0, 0)), mode="edge")
    if radius <= 1:
        closed = binary_closing(padded, structure=disk)
        opened = binary_opening(closed, structure=disk)
        return opened[halo : halo + plane.shape[0], halo : halo + plane.shape[1]]

    # Count footprint hits for each pass to avoid the large offset tables
    # built by ndimage for wide disks. Valid convolution consumes one halo per pass.
    footprint_size = disk.sum()
    for minimum in (1, footprint_size, footprint_size, 1):
        padded = np.rint(fftconvolve(padded, disk, mode="valid")) >= minimum
    return padded


@pytest.mark.parametrize("nx", [1, 5, 9, 63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize("radius", [0, 1, 31, 32, 33, 40, 63, 64, 65, 96, 128, 130])
@pytest.mark.parametrize("regional", [False, True])
def test_structured_disk_footprint_across_word_boundaries(nx, radius, regional):
    scale = max(1, radius)
    y, x = np.ogrid[: 4 * scale + 9, :nx]
    edge = scale + 4 + (scale // 2 + 1) * np.cos(2 * np.pi * x / nx)
    plane = y < edge
    plane[1, nx // 2] = False
    plane[-3, nx // 2] = True

    expected = _disk_reference(plane, radius, regional)
    assert expected.any() and not expected.all()
    if radius:
        assert expected[1, nx // 2] and not expected[-3, nx // 2]
    data = xr.DataArray(plane[None], dims=("time", "lat", "lon")).chunk({"time": 1})
    wet = xr.ones_like(data.isel(time=0, drop=True))
    result = _structured(data, wet, radius, regional)
    assert result.chunks == data.chunks
    np.testing.assert_array_equal(result.compute().values[0], expected)


@pytest.mark.parametrize("half_width", [63, 64, 65, 96, 128, 130, 513])
@pytest.mark.parametrize("position", ["left", "middle", "right"])
def test_wide_chord_clips_to_valid_packed_columns(half_width, position):
    nx = 2 * half_width + 71
    centre = {"left": 0, "middle": nx // 2, "right": nx - 1}[position]
    n_words = (nx + 63) // 64
    last_mask = np.uint64((1 << (nx - 64 * (n_words - 1))) - 1)
    source = np.zeros((1, n_words), dtype=np.uint64)
    source[0, centre // 64] = np.uint64(1) << np.uint64(centre % 64)
    result, horizontal, scratch = (np.empty_like(source) for _ in range(3))

    # A single wide chord forces a shift cap much sooner than a full disk does.
    morphology._dilate_disk_words(source, result, horizontal, scratch, np.array([half_width]), 0, 1, last_mask)

    columns = np.arange(n_words * 64)
    expected = (np.abs(columns - centre) <= half_width) & (columns < nx)
    result_bytes = result.astype("<u8", copy=False).view(np.uint8)
    np.testing.assert_array_equal(np.unpackbits(result_bytes, bitorder="little"), expected)


@pytest.mark.parametrize("default_radius, radius", [(-2, None), (0, -2)])
def test_fill_holes_rejects_negative_radius(default_radius, radius):
    data = xr.DataArray(np.zeros((4, 3, 5), dtype=bool), dims=("time", "lat", "lon")).chunk({"time": 2})
    wet = xr.ones_like(data.isel(time=0, drop=True))
    with pytest.raises(ConfigurationError, match="R_fill must be non-negative"):
        morphology.fill_holes(data, default_radius, False, None, "lon", wet, True, "lat", R_fill=radius)


@pytest.mark.parametrize("nx", [1, 7, 15, 63, 64, 65, 128, 145])
@pytest.mark.parametrize("ny", [1, 8, 64, 90])
@pytest.mark.parametrize("radius", [0, 1, 2, 4, 8, 13, 33])
@pytest.mark.parametrize("regional", [False, True])
def test_structured_matches_scipy(nx, ny, radius, regional):
    rng = np.random.default_rng(1000 * ny + nx)
    values = rng.random((1, ny, nx)) < 0.55
    wet = rng.random((ny, nx)) < 0.8
    data = xr.DataArray(values, dims=("time", "lat", "lon")).chunk({"time": 1})
    mask = xr.DataArray(wet, dims=("lat", "lon"))
    result = _structured(data, mask, radius, regional)
    assert result.chunks == data.chunks
    expected = _edt_reference(values[0], radius, regional) & wet
    np.testing.assert_array_equal(result.compute().values, expected[None])


@pytest.mark.parametrize("regional", [False, True])
@pytest.mark.parametrize("radius", [1, 13, 32, 33])
def test_structured_empty_full_and_partial_word(regional, radius):
    values = np.zeros((3, 73, 65), dtype=bool)
    values[1:] = True
    values[2, 36, -2] = False
    data = xr.DataArray(values, dims=("time", "lat", "lon"))
    wet = xr.ones_like(data.isel(time=0))
    result = _structured(data, wet, radius, regional)
    expected = np.stack([_edt_reference(plane, radius, regional) for plane in values])
    np.testing.assert_array_equal(result.values, expected)
    assert not result.values[0].any()
    assert result.values[1:].all()


def test_structured_wraps_longitude_and_extends_poles():
    values = np.zeros((2, 28, 65), dtype=bool)
    values[0, 8:20, 1:10] = values[0, 8:20, -9:] = True
    values[1, 1:10, 20:40] = values[1, -9:, 20:40] = True
    data = xr.DataArray(values, dims=("time", "lat", "lon"))
    wet = xr.ones_like(data.isel(time=0))
    for regional in (False, True):
        result = _structured(data, wet, 2, regional)
        expected = np.stack([_edt_reference(plane, 2, regional) for plane in values])
        np.testing.assert_array_equal(result.values, expected)
        assert bool(result.values[0, 14, 0]) == (not regional)
        assert not result.values[1, 0, 30]


@pytest.mark.parametrize("regional", [False, True])
@pytest.mark.parametrize("radius", [0, 4, 33])
def test_structured_rank_four_preserves_layout_and_level_mask(regional, radius):
    rng = np.random.default_rng(87)
    values = rng.random((3, 3, 19, 65)) < rng.uniform(0.1, 0.9, size=(3, 3, 1, 1))
    wet = rng.random((3, 19, 65)) < 0.65
    dims = ("time", "depth", "lat", "lon")
    coords = {dim: np.arange(length) for dim, length in zip(dims, values.shape)}
    coords["time"] = 2**53 + np.arange(values.shape[0], dtype=np.int64)
    coords["flag"] = ("time", np.array([-1, 0, 1], dtype=np.int8))
    data = xr.DataArray(values, dims=dims, coords=coords).chunk({"time": (2, 1), "depth": (2, 1)})
    mask = xr.DataArray(wet, dims=dims[1:], coords={dim: coords[dim] for dim in dims[1:]})
    result = _structured(data, mask, radius, regional)
    assert result.chunks == data.chunks
    assert result.dims == data.dims
    assert result.time.dtype == np.int64
    assert result.flag.dtype == np.int8
    xr.testing.assert_identical(result.coords.to_dataset(), data.coords.to_dataset())
    expected = np.array([[_edt_reference(plane, radius, regional) for plane in frame] for frame in values])
    np.testing.assert_array_equal(result.compute().values, expected & wet)


def _gather_reference(values, graph, wet, radius):
    def dilate(bitmap):
        return sparse_bool_power(bitmap, graph.data, graph.indices, graph.indptr, radius)

    dilated = dilate(values)
    dilated[:, ~wet] = True
    closed = ~dilate(~dilated)
    closed[:, ~wet] = True
    return dilate(~dilate(~closed))


def _assert_unstructured(values, graph, wet, radius):
    data = xr.DataArray(values, dims=("time", "ncells")).chunk({"time": (2, 1)})
    mask = xr.DataArray(wet, dims=("ncells",))
    result = morphology.fill_holes(data, radius, True, graph, "ncells", mask, False, None)
    assert result.chunks == data.chunks
    expected = _gather_reference(values, graph, wet, radius)
    np.testing.assert_array_equal(result.compute().values, expected)


@pytest.mark.parametrize("radius", [0, 1, 2, 4, 8, 16])
def test_unstructured_directed_graphs(radius):
    rng = np.random.default_rng(20261123)
    for _ in range(100):
        n = rng.integers(3, 57)
        adjacency = rng.random((n, n)) < rng.uniform(0.15, 2) / n
        np.fill_diagonal(adjacency, True)
        adjacency[0, 1] = True
        adjacency[1, 0] = False
        # Explicit zeros can appear in CSR, including the disabled reverse edge.
        stored = adjacency | (rng.random((n, n)) < 0.5 / n)
        stored[1, 0] = True
        rows, cols = np.nonzero(stored)
        graph = csr_matrix((adjacency[rows, cols], (rows, cols)), shape=(n, n))
        values = rng.random((3, n)) < rng.uniform(0.05, 0.95, size=(3, 1))
        wet = rng.random(n) < 0.85
        _assert_unstructured(values, graph, wet, radius)


@pytest.mark.parametrize("radius", [0, 1, 2, 4, 8, 16])
def test_unstructured_triangular_mesh_with_coast(monkeypatch, radius):
    # Alternating triangles share two same-row neighbours and one across rows.
    ny, nx = 8, 12
    y, x = np.indices((ny, nx))
    cells = y * nx + x
    across = ((y + (-1) ** ((y + x) % 2)) % ny) * nx + x
    edge_targets = np.stack([cells, y * nx + (x + 1) % nx, y * nx + (x - 1) % nx, across])
    graph = csr_matrix((np.ones(edge_targets.size, bool), (np.tile(cells.ravel(), 4), edge_targets.ravel())))
    wet = ((x > 1) | (y > 3)).ravel()
    values = np.stack([((x < 5) & (y < 6)).ravel(), (x > 7).ravel(), wet])

    def unexpected_copy(*args, **kwargs):
        pytest.fail("reciprocal triangular mesh was transposed")

    monkeypatch.setattr(morphology, "csr_matrix", unexpected_copy)
    _assert_unstructured(values, graph, wet, radius)


@pytest.mark.parametrize("radius", [0, 1, 2, 4, 8, 16])
def test_unstructured_missing_or_disabled_self_loop(monkeypatch, radius):
    # Row 0 has no stored loop. Row 1 stores a loop, but the edge is false.
    indices = np.array([1, 2, 1, 2, 0, 3, 3], dtype=np.int32)
    indptr = np.array([0, 1, 3, 6, 7], dtype=np.int32)
    flags = np.array([True, True, False, True, True, True, True])
    graph = csr_matrix((flags, indices, indptr), shape=(4, 4))
    values = np.eye(4, dtype=bool)[:3]
    wet = np.array([True, True, False, True])

    def unexpected_bfs(*args, **kwargs):
        pytest.fail("a graph with missing self-loops was passed to bounded BFS")

    monkeypatch.setattr(morphology, "_graph_close_open", unexpected_bfs)
    _assert_unstructured(values, graph, wet, radius)
    # Restore row 0's loop to isolate the effect of the explicit false edge.
    graph = graph.tolil()
    graph[0, 0] = True
    _assert_unstructured(values, graph.tocsr(), wet, radius)
