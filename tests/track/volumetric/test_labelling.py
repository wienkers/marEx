"""Tests for per-timestep volumetric labelling."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import label as scipy_label

from marEx.exceptions import ConfigurationError
from marEx.track.objects import identify_objects
from marEx.track.volumetric.geometry import VerticalGeometry
from marEx.track.volumetric.labelling import _column_contacts, label_volume, label_volumes_lazy

from .oracles import voxel_graph_labels

SHAPE = (6, 12, 16)
N_TRIALS = 30


def _canonical(labels):
    """Renumber a label field 1..n by first voxel in C order (0 stays background)."""
    flat = labels.ravel()
    fg = flat > 0
    uniq, first = np.unique(flat[fg], return_index=True)
    order = uniq[np.argsort(first)]
    lut = np.zeros(int(flat.max()) + 1 if flat.size else 1, dtype=np.int32)
    lut[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return lut[labels]


DYADIC_LEVELS = (2.5, 5.0, 7.5, 10.0)
# Not exactly representable: gap sums depend on the summation order.
NON_DYADIC_LEVELS = (0.1, 0.2, 0.3, 0.7)


def _random_case(seed, shape=SHAPE, p_active=0.35, all_wet=False, levels=DYADIC_LEVELS):
    """Random active field with irregular, partial-cell thickness and non-monotone wet columns.

    With the default dyadic thicknesses gap sums land exactly on round thresholds; with
    ``NON_DYADIC_LEVELS`` they carry rounding that depends on the summation order.
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    active = rng.random(shape) < p_active
    level_thickness = rng.choice(levels, size=nz)
    thickness = np.broadcast_to(level_thickness[:, None, None], shape).copy()
    if all_wet:
        return active, np.ones(shape, dtype=bool), thickness
    n_wet = rng.integers(1, nz + 1, size=(ny, nx))
    wet = np.arange(nz)[:, None, None] < n_wet[None]
    # Partial bottom cells: the deepest wet cell of a column is shortened from below.
    fraction = rng.choice([0.25, 0.5, 0.75, 1.0], size=(ny, nx))
    yy, xx = np.indices((ny, nx))
    thickness[n_wet - 1, yy, xx] *= fraction
    # Dry cells in the middle of wet columns (a wet cell can sit below a dry one).
    wet &= ~(rng.random(shape) < 0.08)
    thickness = np.where(wet, thickness, 0.0)
    return active, wet, thickness


def _column(active_levels, wet_levels=None, thickness_levels=None):
    """A single (nz, 1, 1) column from per-level lists."""
    active = np.array(active_levels, dtype=bool)[:, None, None]
    nz = active.shape[0]
    wet = np.ones(nz, dtype=bool) if wet_levels is None else np.array(wet_levels, dtype=bool)
    thk = np.full(nz, 10.0) if thickness_levels is None else np.array(thickness_levels, dtype=np.float64)
    thk = np.where(wet, thk, 0.0)
    return active, wet[:, None, None], thk[:, None, None]


# Vertical faces plus the full 3x3 in-plane neighbourhood.
_GAP_ZERO_STRUCTURE = np.zeros((3, 3, 3), dtype=bool)
_GAP_ZERO_STRUCTURE[1] = True
_GAP_ZERO_STRUCTURE[0, 1, 1] = _GAP_ZERO_STRUCTURE[2, 1, 1] = True


def test_gap_zero_matches_scipy_label():
    for trial in range(N_TRIALS):
        active, wet, thickness = _random_case(100 + trial, all_wet=True)
        labels, n = label_volume(active, wet, thickness, max_gap_m=0.0, wrap_x=False)
        expected, n_expected = scipy_label(active, structure=_GAP_ZERO_STRUCTURE)
        assert labels.dtype == np.int32
        assert n == n_expected
        np.testing.assert_array_equal(labels, _canonical(expected))


@pytest.mark.parametrize("wrap_x", [False, True])
@pytest.mark.parametrize("max_gap_m", [0.0, 9.999, 10.0, 10.001, 50.0])
def test_matches_voxel_graph_oracle(max_gap_m, wrap_x):
    for trial in range(N_TRIALS):
        active, wet, thickness = _random_case(200 + trial)
        labels, n = label_volume(active, wet, thickness, max_gap_m=max_gap_m, wrap_x=wrap_x)
        expected = voxel_graph_labels(active, wet, thickness, max_gap_m, wrap_x)
        assert labels.dtype == np.int32
        np.testing.assert_array_equal(labels, expected)
        assert n == int(expected.max())


NON_DYADIC_SHAPE = (10, 12, 16)
# 0.6 == 0.3 + 0.2 + 0.1 (bottom-up) and 0.6000000000000001 == 0.1 + 0.2 + 0.3 (top-down).
NON_DYADIC_GAPS = [0.3, 0.6, 0.6000000000000001, 1.0]


def _non_dyadic_case(trial):
    return _random_case(600 + trial, shape=NON_DYADIC_SHAPE, p_active=0.25, levels=NON_DYADIC_LEVELS)


@pytest.mark.parametrize("wrap_x", [False, True])
@pytest.mark.parametrize("max_gap_m", NON_DYADIC_GAPS)
def test_matches_voxel_graph_oracle_non_dyadic_thickness(max_gap_m, wrap_x):
    for trial in range(N_TRIALS):
        active, wet, thickness = _non_dyadic_case(trial)
        labels, n = label_volume(active, wet, thickness, max_gap_m=max_gap_m, wrap_x=wrap_x)
        expected = voxel_graph_labels(active, wet, thickness, max_gap_m, wrap_x)
        np.testing.assert_array_equal(labels, expected)
        assert n == int(expected.max())


def test_non_dyadic_trials_depend_on_summation_order():
    """Some eligible gap in the non-dyadic trials must flip at a threshold when summed bottom-up."""
    flips = 0
    for trial in range(N_TRIALS):
        active, wet, thickness = _non_dyadic_case(trial)
        fg = active & wet
        nz, ny, nx = fg.shape
        for y in range(ny):
            for x in range(nx):
                levels = np.flatnonzero(fg[:, y, x])
                for k1, k2 in zip(levels[:-1], levels[1:]):
                    between = range(k1 + 1, k2)
                    if not all(wet[m, y, x] for m in between):
                        continue
                    down = up = 0.0
                    for m in between:
                        down += float(thickness[m, y, x])
                    for m in reversed(between):
                        up += float(thickness[m, y, x])
                    flips += sum((down <= g) != (up <= g) for g in NON_DYADIC_GAPS)
    assert flips > 0


def test_oracle_trials_exercise_the_threshold_and_barriers():
    """The random trials must discriminate: equality at the threshold, dry holes, active-on-dry."""
    differs_at_threshold = False
    has_mid_column_dry = False
    has_active_on_dry = False
    for trial in range(N_TRIALS):
        active, wet, thickness = _random_case(200 + trial)
        below = voxel_graph_labels(active, wet, thickness, 9.999, False)
        equal = voxel_graph_labels(active, wet, thickness, 10.0, False)
        differs_at_threshold |= int(below.max()) != int(equal.max())
        has_mid_column_dry |= bool((~wet[:-1] & wet[1:]).any())
        has_active_on_dry |= bool((active & ~wet).any())
    assert differs_at_threshold
    assert has_mid_column_dry
    assert has_active_on_dry


def test_gap_threshold_below_equal_above():
    active, wet, thk = _column([1, 0, 1], thickness_levels=[5.0, 10.0, 5.0])
    joined, n_joined = label_volume(active, wet, thk, max_gap_m=10.0, wrap_x=False)
    assert n_joined == 1
    assert joined[0, 0, 0] == joined[2, 0, 0] == 1
    split, n_split = label_volume(active, wet, thk, max_gap_m=9.999, wrap_x=False)
    assert n_split == 2
    assert split[0, 0, 0] == 1 and split[2, 0, 0] == 2
    above, n_above = label_volume(active, wet, thk, max_gap_m=10.001, wrap_x=False)
    assert n_above == 1


def test_gap_sums_partial_cells():
    # Two intervening cells, 7.5 + 2.5 = 10 exactly (a partial cell below a full one).
    active, wet, thk = _column([1, 0, 0, 1], thickness_levels=[5.0, 7.5, 2.5, 5.0])
    assert label_volume(active, wet, thk, max_gap_m=10.0, wrap_x=False)[1] == 1
    assert label_volume(active, wet, thk, max_gap_m=9.999, wrap_x=False)[1] == 2


def test_gap_is_summed_top_down():
    # 0.1 + 0.2 + 0.3 == 0.6000000000000001 > 0.6, while 0.3 + 0.2 + 0.1 == 0.6.
    active, wet, thk = _column([1, 0, 0, 0, 1], thickness_levels=[1.0, 0.1, 0.2, 0.3, 1.0])
    assert label_volume(active, wet, thk, max_gap_m=0.6, wrap_x=False)[1] == 2
    active, wet, thk = _column([1, 0, 0, 0, 1], thickness_levels=[1.0, 0.3, 0.2, 0.1, 1.0])
    assert label_volume(active, wet, thk, max_gap_m=0.6, wrap_x=False)[1] == 1


def test_bridged_cells_stay_zero():
    active, wet, thk = _column([1, 0, 1], thickness_levels=[5.0, 10.0, 5.0])
    labels, n = label_volume(active, wet, thk, max_gap_m=10.0, wrap_x=False)
    assert n == 1
    assert labels[1, 0, 0] == 0
    assert np.count_nonzero(labels) == 2


def test_dry_cell_mid_column_blocks_and_resets():
    active, wet, thk = _column([1, 0, 1, 1], wet_levels=[1, 0, 1, 1])
    for gap in (0.0, 10.0, 1.0e6, np.inf):
        labels, n = label_volume(active, wet, thk, max_gap_m=gap, wrap_x=False)
        assert n == 2
        assert labels[0, 0, 0] == 1
        assert labels[1, 0, 0] == 0
        # Below the dry cell the wet cells still join each other.
        assert labels[2, 0, 0] == labels[3, 0, 0] == 2

    # A dry cell resets the running gap: thickness above it is not carried below it.
    active, wet, thk = _column([1, 0, 0, 0, 1], wet_levels=[1, 1, 0, 1, 1], thickness_levels=[5.0, 1.0, 1.0, 1.0, 5.0])
    assert label_volume(active, wet, thk, max_gap_m=100.0, wrap_x=False)[1] == 2
    # And a run that starts below the dry cell measures its gap from there.
    active, wet, thk = _column([0, 0, 1, 0, 1], wet_levels=[1, 0, 1, 1, 1], thickness_levels=[1.0, 1.0, 5.0, 4.0, 5.0])
    labels, n = label_volume(active, wet, thk, max_gap_m=4.0, wrap_x=False)
    assert n == 1
    assert labels[2, 0, 0] == labels[4, 0, 0] == 1


def test_true_on_dry_cells_is_ignored():
    # Vertically: an active dry cell between two active wet cells neither labels nor bridges.
    active, wet, thk = _column([1, 1, 1], wet_levels=[1, 0, 1])
    labels, n = label_volume(active, wet, thk, max_gap_m=1.0e6, wrap_x=False)
    assert n == 2
    assert labels[1, 0, 0] == 0

    # Horizontally: an active dry cell between two active wet cells on one level does not join them.
    active = np.array([[[1, 1, 1]]], dtype=bool)
    wet = np.array([[[1, 0, 1]]], dtype=bool)
    thk = np.where(wet, 10.0, 0.0)
    labels, n = label_volume(active, wet, thk, max_gap_m=1.0e6, wrap_x=False)
    assert n == 2
    np.testing.assert_array_equal(labels[0, 0], [1, 0, 2])

    # An all-dry active field labels nothing.
    labels, n = label_volume(np.ones((2, 3, 3), bool), np.zeros((2, 3, 3), bool), np.zeros((2, 3, 3)), 10.0, True)
    assert n == 0
    assert not labels.any()


def test_transitive_path_changes_column():
    active = np.zeros((2, 1, 2), dtype=bool)
    active[0, 0, 0] = active[0, 0, 1] = active[1, 0, 1] = True
    wet = np.ones_like(active)
    labels, n = label_volume(active, wet, np.full(active.shape, 10.0), max_gap_m=0.0, wrap_x=False)
    assert n == 1
    assert labels[0, 0, 0] == labels[0, 0, 1] == labels[1, 0, 1] == 1


def test_cross_depth_diagonal_stays_disconnected():
    active = np.zeros((2, 1, 2), dtype=bool)
    active[0, 0, 0] = active[1, 0, 1] = True
    wet = np.ones_like(active)
    labels, n = label_volume(active, wet, np.full(active.shape, 10.0), max_gap_m=1.0e6, wrap_x=False)
    assert n == 2
    assert labels[0, 0, 0] == 1 and labels[1, 0, 1] == 2


@pytest.mark.parametrize("dy", [-1, 0, 1])
def test_seam_contact(dy):
    nz, ny, nx = 2, 5, 8
    active = np.zeros((nz, ny, nx), dtype=bool)
    active[1, 2, 0] = True  # left edge
    active[1, 2 + dy, nx - 1] = True  # right edge, same row or diagonal across the seam
    wet = np.ones_like(active)
    thk = np.full(active.shape, 10.0)
    assert label_volume(active, wet, thk, max_gap_m=0.0, wrap_x=False)[1] == 2
    labels, n = label_volume(active, wet, thk, max_gap_m=0.0, wrap_x=True)
    assert n == 1
    assert labels[1, 2, 0] == labels[1, 2 + dy, nx - 1] == 1
    # Two rows apart is not a seam contact.
    active[1, 2 + dy, nx - 1] = False
    active[1, 4, nx - 1] = True
    assert label_volume(active, wet, thk, max_gap_m=0.0, wrap_x=True)[1] == 2


@pytest.mark.parametrize("regional", [False, True])
def test_ids_first_voxel_c_order_and_nz1_equals_2d(regional):
    ny, nx = 12, 16
    for trial in range(20):
        rng = np.random.default_rng(300 + trial)
        slice_2d = rng.random((ny, nx)) < 0.4
        data_bin = xr.DataArray(slice_2d[None], dims=("time", "lat", "lon"))
        mask = xr.DataArray(np.ones((ny, nx), dtype=bool), dims=("lat", "lon"))
        field_2d, _, n_2d = identify_objects(
            data_bin,
            time_connectivity=False,
            unstructured_grid=False,
            mask=mask,
            neighbours_int=None,
            xdim="lon",
            regional_mode=regional,
        )
        expected = field_2d.values[0]
        labels, n = label_volume(
            slice_2d[None], np.ones((1, ny, nx), bool), np.ones((1, ny, nx)), max_gap_m=0.0, wrap_x=not regional
        )
        assert labels.dtype == expected.dtype == np.int32
        np.testing.assert_array_equal(labels[0], expected)
        assert n == n_2d

    # In 3-D the order is by first voxel over (z, y, x): an object starting on the top level
    # outranks one starting lower in the column but earlier in (y, x).
    active = np.zeros((2, 3, 3), dtype=bool)
    active[1, 0, 0] = True  # earliest (y, x), deeper level
    active[0, 2, 2] = True  # later (y, x), top level
    labels, n = label_volume(active, np.ones_like(active), np.ones(active.shape), 0.0, False)
    assert n == 2
    assert labels[0, 2, 2] == 1 and labels[1, 0, 0] == 2


def test_empty_and_all_foreground():
    shape = (3, 4, 5)
    wet = np.ones(shape, dtype=bool)
    thk = np.full(shape, 10.0)
    labels, n = label_volume(np.zeros(shape, bool), wet, thk, max_gap_m=10.0, wrap_x=True)
    assert n == 0
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(labels, np.zeros(shape, dtype=np.int32))

    labels, n = label_volume(np.ones(shape, bool), wet, thk, max_gap_m=0.0, wrap_x=False)
    assert n == 1
    np.testing.assert_array_equal(labels, np.ones(shape, dtype=np.int32))


def test_invalid_inputs_raise():
    active, wet, thk = _column([1, 0, 1])
    with pytest.raises(ConfigurationError):
        label_volume(active, wet, thk, max_gap_m=-1.0, wrap_x=False)
    with pytest.raises(ConfigurationError):
        label_volume(active, wet, thk, max_gap_m=np.nan, wrap_x=False)
    with pytest.raises(ConfigurationError):
        label_volume(active, wet[:2], thk, max_gap_m=1.0, wrap_x=False)


def test_column_contacts_python_semantics_equal_jitted():
    for trial in range(20):
        active, wet, thickness = _random_case(400 + trial)
        level_ids = np.where(active & wet, np.arange(1, active.size + 1).reshape(active.shape), 0).astype(np.int64)
        for gap in (0.0, 10.0, 25.0):
            jitted = _column_contacts(level_ids, wet, thickness, gap)
            python = _column_contacts.py_func(level_ids, wet, thickness, gap)
            assert jitted.dtype == np.int64 and jitted.shape[1] == 2
            np.testing.assert_array_equal(jitted, python)


def _geometry(wet, thickness):
    area = np.ones(wet.shape[1:])
    return VerticalGeometry(
        top=np.zeros_like(thickness),
        bottom=thickness.copy(),
        thickness=thickness,
        midpoint=thickness / 2.0,
        cell_area_m2=area,
        cell_volume=thickness * area,
        wet=wet,
        source="bounds",
        flip_z=False,
        default_scales=(1.0, 1.0),
    )


@pytest.mark.parametrize(
    "dims",
    [("time", "depth", "lat", "lon"), ("depth", "lat", "lon", "time")],
)
def test_label_volumes_lazy_matches_per_timestep(dims):
    nt = 5
    _, wet, thickness = _random_case(500)
    rng = np.random.default_rng(501)
    active = rng.random((nt,) + SHAPE) < 0.35
    data = xr.DataArray(active, dims=("time", "depth", "lat", "lon")).transpose(*dims).chunk({"time": 2})
    out = label_volumes_lazy(data, _geometry(wet, thickness), 10.0, True, "depth", "lat", "lon")
    assert isinstance(out.data, da.Array)
    assert out.dtype == np.int32
    assert out.dims == data.dims
    values = out.transpose("time", "depth", "lat", "lon").values
    for t in range(nt):
        expected, _ = label_volume(active[t], wet, thickness, 10.0, True)
        np.testing.assert_array_equal(values[t], expected)


def test_label_volumes_lazy_rejects_mismatched_geometry():
    _, wet, thickness = _random_case(500)
    data = xr.DataArray(np.zeros((2, 6, 12, 15), bool), dims=("time", "depth", "lat", "lon")).chunk({"time": 1})
    with pytest.raises(ConfigurationError):
        label_volumes_lazy(data, _geometry(wet, thickness), 10.0, True, "depth", "lat", "lon")
