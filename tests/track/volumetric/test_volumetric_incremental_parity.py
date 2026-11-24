"""Bitwise checks for incremental overlaps, split certificates and depth-only geometry."""

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.core.compute_mode import Materialiser
from marEx.track.volumetric.diagnostics import _event_slice
from marEx.track.volumetric.ids import IDAllocator
from marEx.track.volumetric.merge_loop import (
    MergeLedger,
    SplitGeometry,
    _resolve_timestep,
    _WaterComponents,
    consolidate_volumes,
    final_links,
    split_and_merge_volumetric,
)
from marEx.track.volumetric.overlap import OverlapTable, volume_overlaps
from marEx.track.volumetric.partition import ScaledCoords, partition_child
from marEx.track.volumetric.store import ObjectIndex

from .conftest import VDIMS, build_case, store_from_field
from .synthetic import make_inputs, scenarios
from .test_tracker4d import _kwargs


def _assert_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype
    if actual.dtype == np.float64:
        actual, expected = actual.view(np.uint64), expected.view(np.uint64)
    np.testing.assert_array_equal(actual, expected)


class _FullOverlapTable(OverlapTable):
    """Reference that rebuilds every boundary table after every pass."""

    def refresh(self, previous, current, cell_volume, changed):
        self.pairs, self.volumes = volume_overlaps(previous, current, cell_volume)


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("mode", ["persist", "streaming"])
def test_cascade_matches_full_redetection(width, mode, tmp_path, monkeypatch):
    binary, meta = scenarios()["cascade_11_iterations"]
    binary = np.concatenate((binary, binary[::-1], binary[:1]))
    case = build_case(binary, meta, time_chunk=width)
    cv = case.cell_volume.ravel()
    ids = case.ids.copy()
    volume_store = store_from_field(ids, cv)
    allocator = IDAllocator(case.allocator.next_id)
    ledger = MergeLedger(time_dtype=case.id_field.time.dtype)
    geometry = SplitGeometry(case.geom.wet, case.geom.thickness, 0.0, False)
    raw_tables = 0

    def check_refresh(self, previous, current, cell_volume, changed):
        nonlocal raw_tables
        refresh(self, previous, current, cell_volume, changed)
        expected_pairs, expected_volumes = volume_overlaps(previous, current, cell_volume)
        _assert_bits(self.pairs, expected_pairs)
        _assert_bits(self.volumes, expected_volumes)
        raw_tables += 1

    refresh = OverlapTable.refresh
    monkeypatch.setattr(OverlapTable, "refresh", check_refresh)
    for t in range(1, len(ids)):
        previous, current = ids[t - 1].ravel(), ids[t].ravel()
        time = case.id_field.time.values[t]
        _resolve_timestep(
            previous,
            current,
            volume_store,
            allocator,
            case.coords,
            cv,
            0.5,
            ledger,
            time,
            t,
            100,
            overlaps=_FullOverlapTable(previous, current, cv),
        )
        ids[t] = consolidate_volumes(
            ids[t - 1], ids[t], volume_store, case.cell_volume, 0.5, ledger=ledger, time_value=time, split_geometry=geometry
        )
    assert ledger.times.count(case.id_field.time.values[2]) == 11
    expected = case.id_field.copy(data=ids)
    links = final_links(expected, volume_store, cv, 0.5, "time")
    blocks = []
    stage = tmp_path / "stage" if mode == "streaming" else None
    actual, store, events = split_and_merge_volumetric(
        case.id_field,
        case.store,
        case.allocator,
        case.geom,
        case.coords,
        0.5,
        VDIMS,
        materialiser=Materialiser(mode, stage),
        id_field_path=None,
        max_gap_m=0.0,
        wrap_x=False,
        pristine_labels=True,
        links_out=blocks,
    )
    assert raw_tables >= 11
    assert actual.chunks == case.id_field.chunks
    xr.testing.assert_identical(actual.compute(), expected)
    _assert_bits(np.concatenate(blocks), links)
    _assert_bits(np.concatenate(blocks), final_links(actual, store, cv, 0.5, "time"))
    assert set(store._vol) == set(volume_store._vol)
    alive = sorted(store._vol)
    _assert_bits(store.volumes(alive), volume_store.volumes(alive))
    assert case.allocator.next_id == allocator.next_id
    xr.testing.assert_identical(events.to_dataset(), ledger.to_dataset())
    _assert_bits(events.to_dataset().overlap_volumes, ledger.to_dataset().overlap_volumes)


def test_chained_consolidation_recomputes_complete_survivor():
    previous = np.zeros(90, dtype=np.int32)
    current = previous.copy()
    previous[:15] = np.repeat([3, 2, 1], 5)
    current[:15] = [4, 4, 5, 0, 5, 5, 5, 0, 6, 6, 6, 0, 7, 7, 7]
    cv = np.ones(current.size)
    cv[1] = 1e16
    cv[5] = 1e16
    cv[10] = 1e16
    # Use the unit-volume case to select the chained renames, then compare the
    # raw overlap and survivor against adversarial float64 volume weights.
    store = store_from_field(np.stack((previous, current)), np.ones_like(cv))
    table = OverlapTable(previous, current, np.ones_like(cv))
    ledger = MergeLedger()
    out = consolidate_volumes(previous, current, store, np.ones_like(cv), 0.25, ledger=ledger, overlaps=table)
    assert ledger.renamed == [(7, 6), (6, 5), (5, 4)]
    _assert_bits(table.pairs, volume_overlaps(previous, out, cv)[0])

    weighted = OverlapTable(previous, current, cv)
    old_pair_sum = weighted.volumes[weighted.pairs[:, 0] == 3].sum()
    weighted.current_index = ObjectIndex.from_slice(out)
    weighted.refresh(previous, out, cv, {4, 5, 6, 7})
    expected_pairs, expected_volumes = volume_overlaps(previous, out, cv)
    _assert_bits(weighted.pairs, expected_pairs)
    _assert_bits(weighted.volumes, expected_volumes)
    assert expected_volumes[-1] != old_pair_sum


def test_consolidation_activates_previously_unlinked_pair():
    previous = np.zeros(64, dtype=np.int32)
    current = previous.copy()
    previous[:12] = [1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2]
    current[:10] = np.repeat([3, 4], 5)
    volume = np.ones(previous.size)
    store = store_from_field(np.stack((previous, current)), volume)
    table = OverlapTable(previous, current, volume)
    _assert_bits(table.linked(store, 0.5)[0], np.array([[1, 3], [1, 4]], dtype=np.int64))
    out = consolidate_volumes(previous, current, store, volume, 0.5, overlaps=table)
    pairs, sums = volume_overlaps(previous, out, volume)
    _assert_bits(table.pairs, pairs)
    _assert_bits(table.volumes, sums)
    _assert_bits(table.linked(store, np.nextafter(4 / 6, 0.0))[0], np.array([[1, 3], [2, 3]], dtype=np.int64))
    _assert_bits(table.linked(store, np.nextafter(4 / 6, 1.0))[0], np.array([[1, 3]], dtype=np.int64))


@pytest.mark.parametrize(
    "previous,pristine,current,expected,geometric_calls",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [3, 3, 0, 4, 4], True, 0),
        ([2, 1, 1, 1, 2], [1, 1, 1, 1, 1], [3, 0, 0, 0, 4], False, 1),
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [3, 3, 4, 4, 4], False, 0),
    ],
    ids=["whole-parent", "partial-parent", "common-water"],
)
def test_component_certificate(previous, pristine, current, expected, geometric_calls, monkeypatch):
    previous, pristine, current = (np.array(x, dtype=np.int32) for x in (previous, pristine, current))
    previous_water = _WaterComponents(pristine)
    if not np.array_equal(previous, pristine):
        ids, counts = np.unique(previous, return_counts=True)
        previous_water.partition(np.arange(pristine.size), ids, counts)
    # In the common-water case current IDs are two pieces of one pristine component.
    current_water = _WaterComponents(current if current[2] == 0 else np.full_like(current, 3))
    if current[2] != 0:
        ids, counts = np.unique(current, return_counts=True)
        current_water.partition(np.arange(current.size), ids, counts)
    geometry = SplitGeometry(np.ones((1, 1, 5), dtype=bool), np.ones(1), 0.0, False)
    original = SplitGeometry.separates
    calls = []

    def count(self, *args):
        calls.append(1)
        return original(self, *args)

    def no_water_label(*args):
        pytest.fail("Split handling relabelled the whole identified slice")

    monkeypatch.setattr(SplitGeometry, "separates", count)
    monkeypatch.setattr(SplitGeometry, "water_pieces", no_water_label)
    cv = np.ones(current.size)
    store = store_from_field(np.stack((previous, current)), cv)
    ledger = MergeLedger()
    consolidate_volumes(
        previous,
        current,
        store,
        cv,
        0.5,
        ledger=ledger,
        time_value=np.datetime64("2000-01-02"),
        split_geometry=geometry,
        overlaps=OverlapTable(previous, current, cv),
        water_components=(previous_water, current_water),
    )
    assert ledger.renamed == [(4, 3)]
    assert bool(ledger.split_parents) == expected
    assert len(calls) == geometric_calls


def test_consolidation_can_restore_ownership_of_a_whole_component():
    labels = np.array([1, 1, 1, 1, 0, 4, 4], dtype=np.int32)
    partitioned = np.array([1, 1, 3, 3, 0, 4, 4], dtype=np.int32)
    parent = _WaterComponents(labels)
    parent.partition(np.arange(4), np.array([1, 3]), np.array([2, 2]))
    children = _WaterComponents(np.array([5, 0, 6, 6, 0, 8, 8], dtype=np.int32))
    assert parent.certifies_split(1, np.array([0]), children) is False
    index = ObjectIndex.from_slice(partitioned)
    parent.consolidate(1, [4], index)
    index.rename([4], 1)
    assert parent.certifies_split(1, np.array([0, 5, 6]), children) is False
    parent.consolidate(1, [3], index)
    assert parent.certifies_split(1, np.array([0, 2, 3, 5, 6]), children) is True


@pytest.mark.parametrize("mode,width", [("persist", 2), ("streaming", 4)])
def test_depth_only_and_dense_geometry_are_bitwise_identical(mode, width, tmp_path):
    binary, meta = scenarios()["merge_then_split"]
    dry = np.zeros(binary.shape[1:], dtype=bool)
    dry[1:, 0, :3] = True
    data, wet, bounds, areas = make_inputs(binary, **{**meta["make_inputs"], "time_chunk": width, "dry": dry})
    dense = bounds.broadcast_like(wet).transpose("depth", "lat", "lon", "nv")
    options = _kwargs(temp_dir=str(tmp_path / "stage"))
    native = marEx.tracker(data, wet, layer_bounds=bounds, cell_areas=areas, compute_mode=mode, **options)
    grid = marEx.tracker(data, wet, layer_bounds=dense, cell_areas=areas, compute_mode="persist", **options)
    for name in ("top", "bottom", "midpoint", "thickness"):
        layer = getattr(native.geom, name)
        assert layer.shape == (binary.shape[1],)
        _assert_bits(np.where(wet, layer[:, None, None], 0.0), getattr(grid.geom, name))
    _assert_bits(native.geom.cell_volume, grid.geom.cell_volume)
    _assert_bits(native.geom.default_scales, grid.geom.default_scales)

    indices = np.arange(wet.size)
    coords = [ScaledCoords(data.lat, data.lon, g.midpoint, *g.default_scales, wet=g.wet) for g in (native.geom, grid.geom)]
    _assert_bits(coords[0].points(indices), coords[1].points(indices))
    events, merges = native.run(return_merges=True)
    expected_events, expected_merges = grid.run(return_merges=True)
    try:
        for actual, expected in ((events, expected_events), (merges, expected_merges)):
            xr.testing.assert_identical(actual.compute(), expected.compute())
            for name in actual.data_vars:
                _assert_bits(actual[name].values, expected[name].values)
    finally:
        marEx.clear_staging(events)


def test_scaled_coordinates_preserve_near_tie():
    lat = np.array([0.0])
    lon = np.array([-1.0, 0.0, 1.0 - 8 * np.spacing(1.0)])
    midpoint = np.array([0.7])
    dense = midpoint[:, None, None] * np.ones((1, 1, 3))
    coords = [ScaledCoords(lat, lon, m, 13.0, 0.2) for m in (midpoint, dense)]
    offsets = coords[0].points(np.arange(3)) - coords[1].points(np.array([1]))
    distances = np.sum(offsets**2, axis=1)
    assert 0 < distances[0] - distances[2] < 32 * np.spacing(distances[0])
    parents = [np.array([0]), np.array([2])]
    for scaled in coords:
        _assert_bits(scaled.points(np.arange(3)), coords[1].points(np.arange(3)))
        np.testing.assert_array_equal(partition_child(np.array([1]), parents, np.array([7, 11]), scaled), [1])


def test_depth_diagnostics_preserve_dry_cell_mask():
    wet = np.array([[[True, False]]])
    labels = np.ones(wet.shape, dtype=np.int32)
    area = np.ones((1, 2))
    cv = np.where(wet, 2.0, 0.0)
    layers = [np.array([x]) for x in (2.0, 1.0, 3.0)]
    dense = [np.where(wet, layer[:, None, None], 0.0) for layer in layers]
    args = (labels, np.ones(1, dtype=bool), np.array([1], dtype=np.int32), cv)
    actual = _event_slice(*args, *layers, wet, area, area, area, area, 0.0)
    expected = _event_slice(*args, *dense, wet, area, area, area, area, 0.0)
    for got, want in zip(actual, expected):
        _assert_bits(got, want)
