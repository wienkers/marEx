"""Tests for volumetric preprocessing: per-level R_fill, rank-4 T_fill and the volume filter."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import marEx.track.volumetric.labelling as labelling_module
import marEx.track.volumetric.preprocess as preprocess_module
from marEx.core.compute_mode import Materialiser, create_staging_dir
from marEx.exceptions import ConfigurationError, TrackingError
from marEx.track.morphology import fill_holes, fill_time_gaps
from marEx.track.volumetric.config import VolumetricDims
from marEx.track.volumetric.geometry import VerticalGeometry
from marEx.track.volumetric.preprocess import VolumeFilterResult, filter_small_volumes

from .oracles import voxel_graph_labels

DIMS = VolumetricDims(time="time", z="depth", y="lat", x="lon")
INTERNAL = DIMS.internal_order()


# ---------------------------------------------------------------------------
# R_fill and T_fill: the 4-D result is the 2-D / 3-D result applied level by level
# ---------------------------------------------------------------------------


def _blob_field(seed, shape, p=0.45):
    """Random boolean field with spatial structure (smoothed noise) so closing/opening act."""
    from scipy.ndimage import uniform_filter

    rng = np.random.default_rng(seed)
    noise = rng.random(shape)
    smooth = uniform_filter(noise, size=(1, 1, 3, 3), mode="wrap")
    return smooth < np.quantile(smooth, p)


def _wet_mask(seed, nz, ny, nx):
    """Wet mask that differs per level (deeper levels lose more of the domain)."""
    rng = np.random.default_rng(seed)
    wet = np.ones((nz, ny, nx), dtype=bool)
    for k in range(1, nz):
        wet[k] &= rng.random((ny, nx)) > 0.15 * k
    wet[:, :2, :] = False  # a dry polar band at every level
    return wet


@pytest.mark.parametrize("regional", [False, True])
@pytest.mark.parametrize("R_fill", [0, 2])
def test_fill_holes_is_per_level(R_fill, regional):
    nt, nz, ny, nx = 3, 3, 24, 28
    values = _blob_field(11, (nt, nz, ny, nx))
    wet = _wet_mask(12, nz, ny, nx)
    data = xr.DataArray(values, dims=INTERNAL).chunk({"time": 1, "depth": -1})
    mask = xr.DataArray(wet, dims=["depth", "lat", "lon"])

    def call(field, field_mask):
        return fill_holes(
            field,
            default_R_fill=R_fill,
            unstructured_grid=False,
            dilate_sparse=None,
            xdim="lon",
            mask=field_mask,
            regional_mode=regional,
            ydim="lat",
        ).compute()

    out4 = call(data, mask)
    assert out4.dims == INTERNAL
    if R_fill:  # the fill must actually act, or the comparison is vacuous
        assert not np.array_equal(out4.values, values & wet[None])
    for k in range(nz):
        out2 = call(data.isel(depth=k), mask.isel(depth=k))
        np.testing.assert_array_equal(out4.isel(depth=k).values, out2.values)


@pytest.mark.parametrize("regional", [False, True])
def test_fill_time_gaps_rank4_equals_per_level(regional):
    nt, nz, ny, nx = 14, 3, 20, 24
    rng = np.random.default_rng(5)
    base = _blob_field(21, (1, nz, ny, nx), p=0.3)[0]
    # An event that blinks: off for one or two steps in places, so closing in time has work to do.
    on = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    values = base[None] & on[:, None, None, None]
    values |= rng.random((nt, nz, ny, nx)) < 0.02
    wet = _wet_mask(22, nz, ny, nx)
    data = xr.DataArray(values, dims=INTERNAL).chunk({"time": 5, "depth": -1})
    mask = xr.DataArray(wet, dims=["depth", "lat", "lon"])

    def call(field, field_mask):
        return fill_time_gaps(
            field,
            T_fill=2,
            R_fill=2,
            timedim="time",
            ydim="lat",
            unstructured_grid=False,
            dilate_sparse=None,
            xdim="lon",
            mask=field_mask,
            regional_mode=regional,
        )

    out4 = call(data, mask)
    assert out4.dims == INTERNAL
    assert out4.chunks[0] == data.chunks[0]  # the realignment to the input's time chunks still applies
    out4 = out4.compute()
    assert not np.array_equal(out4.values, values & wet[None])  # gaps were filled
    for k in range(nz):
        out3 = call(data.isel(depth=k), mask.isel(depth=k)).compute()
        np.testing.assert_array_equal(out4.isel(depth=k).values, out3.values)


# ---------------------------------------------------------------------------
# Volume filter
# ---------------------------------------------------------------------------


def _geometry(wet, thickness, area):
    """VerticalGeometry from a wet mask, layer thickness (z, y, x) and cell areas (y, x)."""
    thickness = np.where(wet, thickness, 0.0)
    volume = np.where(wet, thickness * area[None], 0.0)
    return VerticalGeometry(
        top=np.zeros_like(thickness),
        bottom=thickness.copy(),
        thickness=thickness,
        midpoint=thickness / 2.0,
        cell_area_m2=area,
        cell_volume=volume,
        wet=wet,
        source="bounds",
        flip_z=False,
        default_scales=(1.0, 1.0),
    )


def _unit_geometry(shape):
    wet = np.ones(shape, dtype=bool)
    return _geometry(wet, np.ones(shape), np.ones(shape[1:]))


def _field(active):
    """Dask-backed boolean field, one time chunk per step."""
    return xr.DataArray(active, dims=INTERNAL).chunk({"time": 1, "depth": -1, "lat": -1, "lon": -1})


def retained_oracle(active, geom, max_gap_m, wrap_x, *, quantile=None, absolute=None):
    """
    Exact retained mask, volumes and threshold from the voxel-graph labelling oracle.

    Objects are keyed by ``(time, label)``; their volumes are summed voxel by voxel.
    """
    keep = np.zeros(active.shape, dtype=bool)
    census = []  # (t, label, volume) in (time, label) order
    all_labels = []
    for t in range(active.shape[0]):
        labels = voxel_graph_labels(active[t], geom.wet, geom.thickness, max_gap_m, wrap_x)
        all_labels.append(labels)
        for label in range(1, int(labels.max()) + 1):
            volume = 0.0
            for z, y, x in np.argwhere(labels == label).tolist():
                volume += float(geom.cell_volume[z, y, x])
            census.append((t, label, volume))
    volumes = np.array([v for _, _, v in census], dtype=np.float64)
    if absolute is not None:
        threshold = float(absolute)
    else:
        threshold = float(np.percentile(volumes, quantile * 100.0))
    for t, label, volume in census:
        if volume >= threshold:
            keep[t] |= all_labels[t] == label
    return keep, volumes, threshold, len(census), int(np.sum(volumes >= threshold))


def _run(active, geom, *, quantile=None, absolute=None, gap=0.0, wrap_x=False, materialiser=None):
    return filter_small_volumes(
        _field(active), geom, gap, wrap_x, DIMS, quantile=quantile, absolute=absolute, materialiser=materialiser
    )


def _assert_matches_oracle(result, active, geom, gap, wrap_x, **kwargs):
    keep, volumes, threshold, n_before, n_after = retained_oracle(active, geom, gap, wrap_x, **kwargs)
    assert isinstance(result, VolumeFilterResult)
    np.testing.assert_array_equal(result.filtered.compute().values, keep)
    np.testing.assert_array_equal(np.sort(result.volumes), np.sort(volumes))
    assert result.threshold == threshold
    assert (result.n_before, result.n_after) == (n_before, n_after)
    return keep


def test_vertically_joined_object_passes_though_each_level_fails():
    active = np.zeros((1, 3, 6, 8), dtype=bool)
    active[0, :, 1, 1] = True  # A: three levels deep, one cell per level
    active[0, 0, 4, 5] = True  # B: a single voxel
    geom = _unit_geometry((3, 6, 8))
    # A single level of A holds volume 1, below the threshold; the whole column holds 3.
    result = _run(active, geom, absolute=2.5)
    keep = _assert_matches_oracle(result, active, geom, 0.0, False, absolute=2.5)
    assert keep[0, :, 1, 1].all()
    assert not keep[0, 0, 4, 5]
    assert (result.n_before, result.n_after) == (2, 1)


def test_equal_voxel_counts_unequal_volumes():
    shape = (2, 6, 10)
    area = np.ones(shape[1:])
    area[:, 5:] = 4.0  # the right half of the domain has 4x larger cells
    geom = _geometry(np.ones(shape, dtype=bool), np.ones(shape), area)
    active = np.zeros((1, *shape), dtype=bool)
    active[0, 0, 1, 1:3] = True  # two voxels, volume 2
    active[0, 0, 4, 6:8] = True  # two voxels, volume 8
    result = _run(active, geom, absolute=4.0)
    keep = _assert_matches_oracle(result, active, geom, 0.0, False, absolute=4.0)
    assert not keep[0, 0, 1].any()
    assert keep[0, 0, 4, 6:8].all()
    # A voxel-count filter would keep neither or both; the volume filter separates them.
    assert sorted(result.volumes.tolist()) == [2.0, 8.0]


@pytest.mark.parametrize("mode", ["quantile", "absolute"])
def test_local_labels_reused_across_timesteps(mode):
    shape = (1, 8, 12)
    geom = _unit_geometry(shape)
    active = np.zeros((2, *shape), dtype=bool)
    active[0, 0, 0:3, 0:3] = True  # t0 label 1: volume 9
    active[1, 0, 0:1, 0:1] = True  # t1 label 1: volume 1, the same local label
    active[1, 0, 5:7, 5:8] = True  # t1 label 2: volume 6
    kwargs = {"absolute": 5.0} if mode == "absolute" else {"quantile": 0.5}
    result = _run(active, geom, **kwargs)
    keep = _assert_matches_oracle(result, active, geom, 0.0, False, **kwargs)
    assert keep[0, 0, 0:3, 0:3].all()  # t0 label 1 kept while t1 label 1 (same id) is not
    assert not keep[1, 0, 0, 0]
    assert keep[1, 0, 5:7, 5:8].all()


@pytest.mark.parametrize("mode", ["quantile", "absolute"])
def test_threshold_ties_kept(mode):
    shape = (1, 6, 14)
    geom = _unit_geometry(shape)
    active = np.zeros((1, *shape), dtype=bool)
    active[0, 0, 0, 0:1] = True  # volume 1
    active[0, 0, 2, 0:2] = True  # volume 2
    active[0, 0, 4, 0:2] = True  # volume 2 (tie with the previous)
    active[0, 0, 4, 5:8] = True  # volume 3
    kwargs = {"absolute": 2.0} if mode == "absolute" else {"quantile": 0.5}
    result = _run(active, geom, **kwargs)
    keep = _assert_matches_oracle(result, active, geom, 0.0, False, **kwargs)
    assert result.threshold == 2.0
    assert not keep[0, 0, 0].any()
    assert keep[0, 0, 2, 0:2].all() and keep[0, 0, 4, 0:2].all() and keep[0, 0, 4, 5:8].all()
    assert (result.n_before, result.n_after) == (4, 3)


def test_pooled_quantile_example():
    shape = (2, 12, 14)
    geom = _unit_geometry(shape)
    active = np.zeros((2, *shape), dtype=bool)
    active[0, 0, 0:10, 0:10] = True  # volume 100
    active[0, 0, 11, 0:10] = True  # volume 10, separated from the block by an empty row
    active[1, 0, 5, 5] = True  # volume 1
    result = _run(active, geom, quantile=0.5)
    keep = _assert_matches_oracle(result, active, geom, 0.0, False, quantile=0.5)
    assert result.threshold == 10.0
    assert (result.n_before, result.n_after) == (3, 2)
    assert keep[0].sum() == 110 and not keep[1].any()


def test_absolute_m3_threshold():
    shape = (3, 6, 8)
    thickness = np.array([2.0, 4.0, 8.0])[:, None, None] * np.ones(shape)
    area = np.full(shape[1:], 0.5)
    geom = _geometry(np.ones(shape, dtype=bool), thickness, area)
    active = np.zeros((1, *shape), dtype=bool)
    active[0, 0, 1, 1] = True  # volume 1
    active[0, 0:2, 4, 5] = True  # volume 1 + 2 = 3
    active[0, 2, 1, 6] = True  # volume 4
    result = _run(active, geom, absolute=3.0)
    keep = _assert_matches_oracle(result, active, geom, 0.0, False, absolute=3.0)
    assert result.threshold == 3.0
    assert not keep[0, 0, 1, 1]
    assert keep[0, 0:2, 4, 5].all() and keep[0, 2, 1, 6]


def test_all_removed_raises():
    geom = _unit_geometry((2, 6, 8))
    active = np.zeros((1, 2, 6, 8), dtype=bool)
    active[0, 0, 1, 1] = True
    with pytest.raises(TrackingError, match="removed every object"):
        _run(active, geom, absolute=5.0)


def test_no_objects_raises():
    geom = _unit_geometry((2, 6, 8))
    with pytest.raises(TrackingError, match="No objects"):
        _run(np.zeros((2, 2, 6, 8), dtype=bool), geom, quantile=0.5)


def test_census_overflow_raises(monkeypatch):
    monkeypatch.setattr(preprocess_module, "_OBJECT_BUFFER", 2)
    geom = _unit_geometry((1, 6, 12))
    active = np.zeros((1, 1, 6, 12), dtype=bool)
    active[0, 0, 0, 0] = active[0, 0, 2, 0] = active[0, 0, 4, 0] = True  # three objects in one step
    with pytest.raises(TrackingError, match="overflow"):
        _run(active, geom, quantile=0.5)
    with pytest.raises(TrackingError, match="overflow"):
        _run(active, geom, absolute=1.0)


def test_argument_validation():
    geom = _unit_geometry((1, 4, 4))
    active = np.ones((1, 1, 4, 4), dtype=bool)
    with pytest.raises(ConfigurationError):
        _run(active, geom)
    with pytest.raises(ConfigurationError):
        _run(active, geom, quantile=0.5, absolute=1.0)
    with pytest.raises(ConfigurationError):
        _run(active, geom, quantile=1.5)
    split = xr.DataArray(active, dims=INTERNAL).chunk({"lat": 2})
    with pytest.raises(ConfigurationError, match="single chunk"):
        filter_small_volumes(split, geom, 0.0, False, DIMS, quantile=0.5, absolute=None)
    reordered = xr.DataArray(active, dims=INTERNAL).transpose("depth", "time", "lat", "lon")
    with pytest.raises(ConfigurationError, match="internal order"):
        filter_small_volumes(reordered, geom, 0.0, False, DIMS, quantile=0.5, absolute=None)


def _random_case(seed, shape=(3, 4, 12, 16), p_active=0.3):
    """Random timesteps over a partially dry domain with irregular dyadic thickness and cell areas."""
    rng = np.random.default_rng(seed)
    nt, nz, ny, nx = shape
    active = rng.random(shape) < p_active
    level_thickness = rng.choice([1.0, 2.0, 4.0], size=nz)
    thickness = np.broadcast_to(level_thickness[:, None, None], (nz, ny, nx)).copy()
    n_wet = rng.integers(1, nz + 1, size=(ny, nx))
    wet = np.arange(nz)[:, None, None] < n_wet[None]
    wet &= ~(rng.random((nz, ny, nx)) < 0.05)
    area = rng.choice([0.5, 1.0, 2.0], size=(ny, nx))
    return active, _geometry(wet, thickness, area)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("wrap_x", [False, True])
@pytest.mark.parametrize("gap", [0.0, 4.0])
@pytest.mark.parametrize("mode", ["quantile", "absolute"])
def test_random_fields_match_the_oracle(seed, wrap_x, gap, mode):
    active, geom = _random_case(100 + seed)
    kwargs = {"absolute": 6.0} if mode == "absolute" else {"quantile": 0.4}
    result = _run(active, geom, gap=gap, wrap_x=wrap_x, **kwargs)
    _assert_matches_oracle(result, active, geom, gap, wrap_x, **kwargs)
    assert result.n_before > result.n_after >= 1  # the filter both removes and keeps something


def test_object_volumes_sum_in_voxel_order_exactly():
    """Non-dyadic cell volumes: volumes and threshold equal a sequential sum in (depth, lat, lon) order."""
    rng = np.random.default_rng(5)
    shape = (3, 5, 8, 10)
    _, nz, ny, nx = shape
    active = rng.random(shape) < 0.7
    thickness = np.broadcast_to(rng.choice([0.1, 0.3, 1.0 / 3.0], size=(nz, ny, nx)), (nz, ny, nx)).copy()
    area = rng.choice([0.1, 0.7, 1.0 / 3.0, 2.0 / 3.0], size=(ny, nx))
    geom = _geometry(np.ones((nz, ny, nx), dtype=bool), thickness, area)
    result = _run(active, geom, quantile=0.4)
    keep, volumes, threshold, n_before, n_after = retained_oracle(active, geom, 0.0, False, quantile=0.4)
    # Guard: reversing the summation order must change at least one object's volume, otherwise
    # the exact comparison below could not tell the orders apart.
    reversed_sums = []
    for t in range(shape[0]):
        labels = voxel_graph_labels(active[t], geom.wet, geom.thickness, 0.0, False)
        for label in range(1, int(labels.max()) + 1):
            total = 0.0
            for z, y, x in reversed(np.argwhere(labels == label).tolist()):
                total += float(geom.cell_volume[z, y, x])
            reversed_sums.append(total)
    assert np.any(np.array(reversed_sums) != volumes)
    np.testing.assert_array_equal(result.volumes, volumes)
    assert result.threshold == threshold
    np.testing.assert_array_equal(result.filtered.compute().values, keep)
    assert (result.n_before, result.n_after) == (n_before, n_after)


def test_bridged_cells_add_no_volume():
    """A gap-bridged column joins two voxels into one object whose volume excludes the bridge."""
    shape = (3, 4, 4)
    geom = _unit_geometry(shape)
    active = np.zeros((1, *shape), dtype=bool)
    active[0, 0, 1, 1] = active[0, 2, 1, 1] = True
    result = _run(active, geom, absolute=1.0, gap=1.0)
    assert result.volumes.tolist() == [2.0]
    assert not result.filtered.compute().values[0, 1, 1, 1]
    separate = _run(active, geom, absolute=1.0, gap=0.0)
    assert sorted(separate.volumes.tolist()) == [1.0, 1.0]


@pytest.mark.parametrize("mode", ["persist", "lazy", "streaming"])
@pytest.mark.parametrize("threshold_kind", ["quantile", "absolute"])
def test_every_compute_mode_equals_the_oracle(mode, threshold_kind, tmp_path):
    staging = create_staging_dir(tmp_path) if mode == "streaming" else None
    active, geom = _random_case(7, shape=(4, 3, 10, 12))
    kwargs = {"absolute": 6.0} if threshold_kind == "absolute" else {"quantile": 0.4}
    result = _run(active, geom, materialiser=Materialiser(mode, staging), **kwargs)
    _assert_matches_oracle(result, active, geom, 0.0, False, **kwargs)


def test_output_layout_matches_the_input():
    active, geom = _random_case(3, shape=(5, 3, 8, 8))
    data = xr.DataArray(active, dims=INTERNAL).chunk({"time": 2, "depth": -1, "lat": -1, "lon": -1})
    for kwargs in ({"quantile": 0.5}, {"absolute": 3.0}):
        result = filter_small_volumes(
            data, geom, 0.0, False, DIMS, materialiser=None, **{"quantile": None, "absolute": None, **kwargs}
        )
        assert result.filtered.dims == data.dims
        assert result.filtered.dtype == bool
        assert result.filtered.chunks == data.chunks


def test_labelling_passes(monkeypatch):
    """The absolute threshold labels each step once; the quantile needs the census first, so twice."""
    calls = []
    original = labelling_module.label_volume

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(labelling_module, "label_volume", counting)
    active, geom = _random_case(9, shape=(4, 3, 8, 8))
    _run(active, geom, absolute=3.0, materialiser=Materialiser("persist")).filtered.compute()
    absolute_calls = len(calls)
    calls.clear()
    _run(active, geom, quantile=0.5, materialiser=Materialiser("persist")).filtered.compute()
    assert absolute_calls == active.shape[0]
    assert len(calls) == 2 * active.shape[0]


def test_numpy_backed_input():
    active, geom = _random_case(4, shape=(2, 3, 8, 8))
    result = filter_small_volumes(xr.DataArray(active, dims=INTERNAL), geom, 0.0, False, DIMS, quantile=0.5, absolute=None)
    _assert_matches_oracle(result, active, geom, 0.0, False, quantile=0.5)
    assert isinstance(da.asarray(result.filtered.data), da.Array)
