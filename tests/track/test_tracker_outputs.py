"""End-to-end tracker values on both grids.

The golden run is pinned bit for bit against ``tests/data/track_golden_{events,merges}.nc``. The
configuration enables merging with temporal gap filling and nearest-neighbour partitioning, so
both merge/split core routines run: ``split_and_merge_objects`` (the partitioned-merge ledger, with
``n_parents > 1`` and ``n_children > 1`` rows) and ``cluster_rename_objects_and_props`` (global
event IDs, per-event areas and centroids). It was verified bit-exact across separate processes
for every output variable, including the float32 ``centroid`` and the merge ledger's row order.
The full 32-step extent is required: 12- and 20-step slices give ``total_merges == 0``.

The other tests pin statistics per configuration, and prove that dimension names, spatial
chunking and an omitted mask are transparent by comparing ID fields exactly.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.track.tracker import tracker as _Tracker

from ..conftest import assert_count_in_reasonable_range, assert_reasonable_bounds
from .conftest import (
    BASIC_KWARGS,
    DATA_DIR,
    GOLDEN_KWARGS,
    UNSTRUCTURED_KWARGS,
    UNSTRUCTURED_MERGING_KWARGS,
    assert_no_failures,
    check,
    on_client,
    open_gridded,
    open_gridded_blob,
    open_unstructured,
    open_unstructured_merging,
    polar_mask,
)

EVENT_VARS = {"ID_field", "global_ID", "area", "centroid", "presence", "time_start", "time_end", "merge_ledger"}


def _open_golden():
    events = xr.open_dataset(str(DATA_DIR / "track_golden_events.nc"))
    merges = xr.open_dataset(str(DATA_DIR / "track_golden_merges.nc"))
    return events.load(), merges.load()


@pytest.mark.smoke
def test_golden_run_exercises_merge_and_split(golden_run):
    """Guards the golden test itself: a configuration that stopped merging would make identity trivial."""
    events, merges = golden_run
    assert merges.sizes["merge_ID"] > 0, "Configuration produced no partitioned merges"
    assert bool((merges.n_parents.values > 1).any()), "No merge (n_parents > 1) recorded by split_and_merge_objects"
    assert bool((merges.n_children.values > 1).any()), "No split (n_children > 1) recorded by split_and_merge_objects"
    # Tolerant statistics, so the smoke legs on other platforms check the merge path without bit identity.
    assert_reasonable_bounds(events.attrs["preprocessed_area_fraction"], 0.9010, tolerance_absolute=0.02)
    assert_count_in_reasonable_range(events.attrs["N_objects_prefiltered"], 516, tolerance=2)
    assert_count_in_reasonable_range(events.attrs["N_objects_filtered"], 259, tolerance=2)
    assert_count_in_reasonable_range(events.attrs["N_events_final"], 21, tolerance=1)
    assert_count_in_reasonable_range(events.attrs["total_merges"], 29, tolerance=2)


def test_outputs_identical_to_golden(golden_run):
    """The events and merges datasets, every load-bearing array and every attr match the baseline exactly.

    Values, dims, coords and attrs: ``xr.testing.assert_identical`` and ``assert_array_equal`` do not compare dtype."""
    events, merges = golden_run
    golden_events, golden_merges = _open_golden()
    failures = []
    check(failures, "events dataset", lambda: xr.testing.assert_identical(events, golden_events))
    check(failures, "merges dataset", lambda: xr.testing.assert_identical(merges, golden_merges))
    for name in ("ID_field", "global_ID", "presence", "area", "centroid", "merge_ledger"):
        check(
            failures,
            name,
            lambda name=name: np.testing.assert_array_equal(
                events[name].values, golden_events[name].values, err_msg=f"{name} array differs from golden baseline"
            ),
        )
    for name in ("parent_IDs", "child_IDs", "overlap_areas", "n_parents", "n_children"):
        check(
            failures,
            f"merges {name}",
            lambda name=name: np.testing.assert_array_equal(
                merges[name].values, golden_merges[name].values, err_msg=f"merges {name} array differs from golden baseline"
            ),
        )

    def attrs():
        for key, expected in golden_events.attrs.items():
            assert key in events.attrs, f"Missing attribute '{key}' in tracker output"
            actual = events.attrs[key]
            if isinstance(expected, (float, np.floating)):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected), err_msg=f"Attribute '{key}' differs")
            else:
                assert actual == expected, f"Attribute '{key}' differs: {actual!r} != {expected!r}"

    check(failures, "attrs", attrs)
    assert_no_failures(failures)


@pytest.mark.smoke
def test_no_merge_tracking_statistics(gridded_basic_run):
    """The only absolute check of the gridded no-merge path."""
    ds = gridded_basic_run
    assert "ID_field" in ds.data_vars
    assert set(ds.ID_field.dims) == {"time", "lat", "lon"}
    assert ds.attrs["allow_merging"] == 0
    for key in ("N_events_final", "R_fill", "T_fill"):
        assert key in ds.attrs
    max_id = int(ds.ID_field.max())
    assert max_id > 0, "No events were tracked"
    assert max_id == ds.attrs["N_events_final"], "Max ID doesn't match reported event count"
    assert int(ds.ID_field.min()) == 0
    assert np.issubdtype(ds.ID_field.dtype, np.integer), "ID_field should be integer type"
    assert_reasonable_bounds(ds.attrs["preprocessed_area_fraction"], 0.9724, tolerance_absolute=0.02)
    assert_count_in_reasonable_range(ds.attrs["N_objects_prefiltered"], 549, tolerance=2)
    assert_count_in_reasonable_range(ds.attrs["N_objects_filtered"], 274, tolerance=2)
    assert_count_in_reasonable_range(ds.attrs["N_events_final"], 24, tolerance=1)


def _cell_areas(ds):
    lat, lon = ds.lat.values, ds.lon.values
    lat_mesh, _ = np.meshgrid(lat, lon, indexing="ij")
    values = 111320 * 111320 * np.cos(np.radians(lat_mesh))
    return xr.DataArray(values.astype(np.float32), coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="cell_areas")


def _assert_area_is_the_cell_sum(tracked, cell_area, label):
    """Per (time, event): the event's area is the float sum of the cell areas its ID_field cells cover."""
    ids = tracked.ID_field.transpose("time", "lat", "lon").values
    area = tracked.area.transpose("time", "ID").values
    checked = 0
    for t in range(ids.shape[0]):
        for j, event in enumerate(tracked.ID.values):
            cells = ids[t] == event
            if cells.any():
                expected = cell_area[cells].astype(np.float64).sum()
                np.testing.assert_allclose(area[t, j], expected, rtol=1e-6, err_msg=f"{label}: area of event {event} at t={t}")
                checked += 1
    assert checked > 0, f"{label}: no event cells to check"


def test_low_threshold_gridded_merging_validates_multi_parent_children(client, marex_records, monkeypatch):
    """At a 0.15 overlap threshold the gridded merge path ends with children that have several parents, so
    its final validation runs through the per-child report; every other gridded run here uses 0.3 or more and
    never reaches it. The run must complete with ``quiet=False`` (the validation's normal-mode logging).
    It starts from quiet mode, so normal mode and INFO afterwards can only come from ``quiet=False``.

    ``quiet=False`` reconfigures the package logger globally, so the whole prior logging state is put back
    and checked: a later ``tracker(quiet=None)`` on the same worker must not inherit INFO."""
    from marEx import logging_config

    assert on_client(client)
    ds = open_gridded()
    package = logging.getLogger("marEx")
    saved = (
        logging_config._current_verbosity_level,
        logging_config._logger_configured,
        package.level,
        package.propagate,
        list(package.handlers),
    )
    # Normal mode's level comes from MAREX_LOG_LEVEL, and MAREX_VERBOSE turns quiet=False into verbose mode:
    # both are read at import, so pin them here rather than depend on the caller's environment.
    monkeypatch.setattr(logging_config, "MAREX_LOG_LEVEL", "INFO")
    monkeypatch.setattr(logging_config, "MAREX_VERBOSE", False)
    records = marex_records(logging.INFO)
    failure = None
    prior, mode, level = (None, None), None, None  # bound before the try, so a failure inside it is not masked
    try:
        # A quiet prior, so the mode and level asserts below can only pass if quiet=False changed them.
        logging_config.configure_logging(quiet=True)
        prior = (marEx.get_verbosity_level(), package.getEffectiveLevel())
        tracker = marEx.tracker(
            ds.extreme_events,
            ds.mask.where((ds.lat < 85) & (ds.lat > -90), other=False),
            area_filter_quartile=0.4,
            R_fill=8,
            T_fill=2,
            allow_merging=True,
            overlap_threshold=0.15,
            quiet=False,
        )
        mode, level = marEx.get_verbosity_level(), package.getEffectiveLevel()
        try:
            attrs = dict(tracker.run().attrs)
        except Exception as e:  # noqa: BLE001 - reported below without the traceback's persisted intermediates
            failure = f"{type(e).__name__}: {e}"
        del tracker
    finally:
        logging_config._current_verbosity_level, logging_config._logger_configured = saved[:2]
        package.setLevel(saved[2])
        package.propagate = saved[3]
        package.handlers = saved[4]
    restored = (
        logging_config._current_verbosity_level,
        logging_config._logger_configured,
        package.level,
        package.propagate,
        list(package.handlers),
    )
    assert restored == saved, f"the logging state was not restored: {restored} != {saved}"
    assert prior[0] == "quiet" and prior[1] > logging.INFO, f"the quiet prior did not take: {prior}"
    assert mode == "normal" and level <= logging.INFO, f"quiet=False did not take effect: mode {mode!r}, level {level}"
    assert failure is None, f"the low-threshold gridded run raised {failure}"
    summary = [r.getMessage() for r in records if "more than one parent after splitting/merging" in r.getMessage()]
    assert len(summary) == 1, "the run never reached the multi-parent validation, so the row tests nothing"
    assert attrs["N_events_final"] > 0 and attrs["total_merges"] > 0 and attrs["allow_merging"] == 1, attrs


def test_merging_without_nn_and_physical_areas(client):
    """Merging without nearest-neighbour partitioning, with unit and with physical cell areas."""
    assert on_client(client)
    ds = open_gridded()
    kwargs = {"area_filter_quartile": 0.5, "R_fill": 4, "T_fill": 2, "allow_merging": True, "quiet": True}
    cell_areas = _cell_areas(ds)

    tracker_counts = marEx.tracker(ds.extreme_events, polar_mask(ds), **kwargs)
    tracker_physical = marEx.tracker(ds.extreme_events, polar_mask(ds), cell_areas=cell_areas, **kwargs)
    assert tracker_counts.mean_cell_area == 1.0, (
        f"Default tracker should have mean_cell_area=1.0, got {tracker_counts.mean_cell_area}"
    )
    expected_mean = float(cell_areas.mean())
    assert abs(tracker_physical.mean_cell_area - expected_mean) < 1e-6, (
        f"Mean cell area mismatch: {tracker_physical.mean_cell_area} vs {expected_mean}"
    )
    counts = tracker_counts.run().compute()
    physical = tracker_physical.run().compute()

    assert counts.attrs["total_merges"] > 0, "the configuration produced no merges"
    assert counts.presence.equals(counts.global_ID != 0), "Presence doesn't match global_ID"
    present = counts.presence.values
    assert (counts.area.values[present] > 0).all(), "Some events have non-positive area"
    lat_c = counts.centroid.sel(component=0).values[present]
    lon_c = counts.centroid.sel(component=1).values[present]
    lat_c, lon_c = lat_c[~np.isnan(lat_c)], lon_c[~np.isnan(lon_c)]
    assert (lat_c >= float(ds.lat.min())).all() and (lat_c <= float(ds.lat.max())).all(), "centroid outside lat bounds"
    assert (lon_c >= float(ds.lon.min())).all() and (lon_c <= float(ds.lon.max())).all(), "centroid outside lon bounds"
    assert_reasonable_bounds(counts.attrs["preprocessed_area_fraction"], 0.9010, tolerance_absolute=0.02)
    assert_count_in_reasonable_range(counts.attrs["N_objects_prefiltered"], 516, tolerance=2)
    assert_count_in_reasonable_range(counts.attrs["N_objects_filtered"], 259, tolerance=2)
    assert_count_in_reasonable_range(counts.attrs["N_events_final"], 21, tolerance=1)
    assert_count_in_reasonable_range(counts.attrs["total_merges"], 16, tolerance=2)

    _assert_area_is_the_cell_sum(counts, np.ones(cell_areas.shape, np.float32), "unit areas")
    assert physical.attrs["total_merges"] > 0, "the physical-area run produced no merges"
    _assert_area_is_the_cell_sum(physical, cell_areas.values, "physical areas")


# (row id, tracker kwargs, expected statistics: area fraction, prefiltered, filtered, events)
GRIDDED_FILTER_ROWS = [
    ("q0", {"area_filter_quartile": 0.0, "R_fill": 2, "T_fill": 0}, (1.0524, 1046, 1046, 152)),
    ("q0.8", {"area_filter_quartile": 0.8, "R_fill": 2, "T_fill": 0}, (1.5218, 1046, 210, 21)),
    ("q0.5_T0", {"area_filter_quartile": 0.5, "R_fill": 2, "T_fill": 0}, (1.1650, 1046, 522, 54)),
    ("q0.5_T4", {"area_filter_quartile": 0.5, "R_fill": 2, "T_fill": 4}, (1.0080, 1041, 522, 38)),
]


def _assert_stats(ds, expected):
    fraction, prefiltered, filtered, events = expected
    assert_reasonable_bounds(ds.attrs["preprocessed_area_fraction"], fraction, tolerance_absolute=0.02)
    assert_count_in_reasonable_range(ds.attrs["N_objects_prefiltered"], prefiltered, tolerance=2)
    assert_count_in_reasonable_range(ds.attrs["N_objects_filtered"], filtered, tolerance=2)
    assert_count_in_reasonable_range(ds.attrs["N_events_final"], events, tolerance=1)


def test_filter_and_fill_parameters_gridded(client):
    """Area quartile and temporal gap filling, with pinned statistics for every configuration."""
    assert on_client(client)
    ds = open_gridded()
    failures, results = [], {}
    for name, kwargs, expected in GRIDDED_FILTER_ROWS:

        def row(name=name, kwargs=kwargs, expected=expected):
            out = marEx.tracker(ds.extreme_events, ds.mask, allow_merging=False, quiet=True, **kwargs).run()
            results[name] = out.attrs
            assert out.attrs["T_fill"] == kwargs["T_fill"]
            assert int(out.ID_field.max()) == out.attrs["N_events_final"] > 0
            _assert_stats(out, expected)

        check(failures, name, row)

    def ordering():
        assert results["q0.8"]["N_events_final"] <= results["q0"]["N_events_final"], "High filtering should produce fewer events"

    check(failures, "ordering", ordering)
    assert_no_failures(failures)


def _custom_grid(ds):
    """``ds``'s events and mask renamed to dims (t, y, x) and coords (T, latitude, longitude)."""
    events = xr.DataArray(
        ds.extreme_events.values,
        dims=["t", "y", "x"],
        coords={"T": ("t", ds.time.values), "latitude": ("y", ds.lat.values), "longitude": ("x", ds.lon.values)},
    ).chunk({"t": 2, "y": -1, "x": -1})
    mask = xr.DataArray(
        ds.mask.values, dims=["y", "x"], coords={"latitude": ("y", ds.lat.values), "longitude": ("x", ds.lon.values)}
    )
    return events, mask.where((mask.latitude < 85) & (mask.latitude > -90), other=False)


def _compare_custom(standard, custom, merging):
    for stat in ("N_events_final", "N_objects_prefiltered", "N_objects_filtered", "allow_merging", "T_fill", "R_fill"):
        assert standard.attrs[stat] == custom.attrs[stat], f"{stat} differs: {standard.attrs[stat]} vs {custom.attrs[stat]}"
    np.testing.assert_allclose(standard.attrs["preprocessed_area_fraction"], custom.attrs["preprocessed_area_fraction"], rtol=1e-12)
    np.testing.assert_array_equal(
        standard.ID_field.transpose("time", "lat", "lon").values,
        custom.ID_field.transpose("t", "y", "x").values,
        err_msg="ID_field values differ between standard and custom dimensions",
    )
    if merging:
        for name in ("global_ID", "area", "presence", "time_start", "time_end"):
            np.testing.assert_array_equal(standard[name].values, custom[name].values, err_msg=f"{name} differs")
        assert (
            standard.centroid.transpose("time", "ID", "component").shape == custom.centroid.transpose("t", "ID", "component").shape
        )
        np.testing.assert_array_equal(
            standard.merge_ledger.transpose("time", "ID", "sibling_ID").values,
            custom.merge_ledger.transpose("t", "ID", "sibling_ID").values,
            err_msg="merge_ledger differs",
        )
    for std_name, custom_name in (("time", "T"), ("lat", "latitude"), ("lon", "longitude")):
        np.testing.assert_array_equal(
            standard[std_name].values, custom[custom_name].values, err_msg=f"{std_name} coordinate differs"
        )


def test_custom_dimension_names_match_standard_gridded(gridded_basic_run, golden_run, client):
    """Custom dims/coords give the standard result: no-merge vs ``gridded_basic_run``, merge vs the golden run."""
    assert on_client(client)
    ds = open_gridded()
    events, mask = _custom_grid(ds)
    custom_kwargs = {
        "dimensions": {"time": "t", "x": "x", "y": "y"},
        "coordinates": {"time": "T", "x": "longitude", "y": "latitude"},
    }

    no_merge = marEx.tracker(events, mask, **custom_kwargs, **BASIC_KWARGS).run().compute()
    assert {"t", "y", "x"} <= set(no_merge.ID_field.dims)
    assert {"T", "latitude", "longitude"} <= set(no_merge.coords)
    _compare_custom(gridded_basic_run, no_merge, merging=False)

    merge = marEx.tracker(events, mask, **custom_kwargs, **GOLDEN_KWARGS).run().compute()
    assert set(merge.data_vars) == EVENT_VARS
    assert merge.attrs["total_merges"] > 0, "the merge configuration produced no merges"
    _compare_custom(golden_run[0], merge, merging=True)
    present = merge.presence.any(dim="t").values
    assert (merge.time_start.values[present] <= merge.time_end.values[present]).all(), "an event starts after it ends"


def test_centroid_follows_a_moving_blob(client):
    """The blob moves 3 degrees east per day from 170 E along the equator; its centroid must follow."""
    assert on_client(client)
    blob = open_gridded_blob().chunk({"time": 2, "lat": -1, "lon": -1})
    tracked = marEx.tracker(
        blob.extreme_events,
        polar_mask(blob),
        area_filter_quartile=0.0,
        R_fill=0,
        T_fill=0,
        allow_merging=True,
        overlap_threshold=0.3,
        quiet=True,
    ).run()
    assert "centroid" in tracked.data_vars
    assert tracked.attrs["allow_merging"] == 1
    durations = tracked.presence.sum(dim="time")
    longest = int(durations.argmax().values)
    assert durations[longest] > 5, "No long-duration event found (expected moving blob)"

    present = tracked.presence.isel(ID=longest).compute()
    times = tracked.time[present]
    lat_c = tracked.centroid.sel(component=0).isel(ID=longest)[present].values
    lon_c = tracked.centroid.sel(component=1).isel(ID=longest)[present].values
    valid = ~(np.isnan(lat_c) | np.isnan(lon_c))
    times, lat_c, lon_c = times[valid], lat_c[valid], lon_c[valid]
    assert len(times) > 5, "Not enough valid centroid measurements"

    days = (times - blob.time.min()).dt.days.values
    expected_lon = 170.0 + days * 3.0
    expected_lon = np.where(expected_lon > 180, expected_lon - 360, expected_lon)
    assert np.max(np.abs(lat_c)) <= 0.25, f"latitude centroids range [{lat_c.min():.2f}, {lat_c.max():.2f}], expected 0"
    max_lon_error = np.max(np.abs(lon_c - expected_lon))
    assert max_lon_error <= 0.5, (
        f"Maximum longitude error {max_lon_error:.3f} deg exceeds 0.5; expected [{expected_lon.min():.1f}, "
        f"{expected_lon.max():.1f}], got [{lon_c.min():.1f}, {lon_c.max():.1f}]"
    )


def test_spatially_chunked_inputs_warn_and_match(gridded_basic_run, client):
    """Spatially chunked events, mask and coords each warn, and the rechunk is transparent to the result."""
    assert on_client(client)
    ds = open_gridded()
    chunked = ds.copy()
    chunked["lat"] = chunked.lat.chunk({"lat": 4})
    chunked["lon"] = chunked.lon.chunk({"lon": 6})
    chunked["extreme_events"] = ds.extreme_events.chunk({"time": 2, "lat": 5, "lon": 8})
    chunked["mask"] = polar_mask(ds).chunk({"lat": 7, "lon": 11})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracker = marEx.tracker(chunked.extreme_events, chunked.mask, **BASIC_KWARGS)
    messages = [str(w.message) for w in caught]
    for dim in ("lat", "lon"):
        assert any(dim in m and "multiple chunks" in m and "apply_ufunc" in m for m in messages), (
            f"no events {dim} warning: {messages}"
        )
        assert any(f"Mask spatial dimension '{dim}'" in m and "multiple chunks" in m for m in messages), (
            f"no mask {dim} warning: {messages}"
        )

    tracked = tracker.run().compute()
    assert tracked.attrs["N_events_final"] == gridded_basic_run.attrs["N_events_final"]
    np.testing.assert_array_equal(
        tracked.ID_field.transpose("time", "lat", "lon").values,
        gridded_basic_run.ID_field.transpose("time", "lat", "lon").values,
        err_msg="spatially chunked input changed the ID field",
    )


def _unstructured(events, mask, ds, temp_dir, **kwargs):
    return marEx.tracker(
        events,
        mask,
        temp_dir=str(temp_dir),
        neighbours=ds.neighbours,
        cell_areas=ds.cell_areas,
        **{**UNSTRUCTURED_KWARGS, **kwargs},
    )


# (row id, tracker kwargs, time chunk or None for the store's, expected: area fraction, prefiltered, filtered, events)
UNSTRUCTURED_FILTER_ROWS = [
    ("q0.1", {"R_fill": 2, "area_filter_quartile": 0.1, "T_fill": 0}, None, (1.4, 15, 15, 3)),
    ("q0.9", {"R_fill": 2, "area_filter_quartile": 0.9, "T_fill": 0}, None, None),
    ("q0.5_T0_int32", {"R_fill": 2, "area_filter_quartile": 0.5, "T_fill": 0}, 25, (2.2, 15, 7, 2)),
    ("q0.5_T4", {"R_fill": 2, "area_filter_quartile": 0.5, "T_fill": 4}, None, None),
    ("R4_q0.5", {"R_fill": 4, "area_filter_quartile": 0.5, "T_fill": 0}, None, None),
]
# Exact counts captured on the current tree (pins/track/pins_misc.out): (prefiltered, filtered, events).
UNSTRUCTURED_FILTER_PINS = {
    "q0.1": (15, 13, 3),
    "q0.9": (15, 2, 1),
    "q0.5_T0_int32": (15, 8, 2),
    "q0.5_T4": (15, 8, 2),
    "R4_q0.5": (12, 6, 4),
}
_PIN_KEYS = ("N_objects_prefiltered", "N_objects_filtered", "N_events_final")


@pytest.mark.slow
@pytest.mark.smoke
def test_filter_and_fill_parameters_unstructured(client, tmp_path, monkeypatch):
    """Area quartile, gap filling and R_fill on the mesh, with pinned statistics for every configuration.

    The int32 row spies on ``refresh_dask_graph``: the globally unique ID field keeps the int32
    dtype of the per-slice labels.
    """
    assert on_client(client)
    ds = open_unstructured()
    failures, results = [], {}
    for name, kwargs, time_chunk, expected in UNSTRUCTURED_FILTER_ROWS:

        def row(name=name, kwargs=kwargs, time_chunk=time_chunk, expected=expected):
            events = ds.extreme_events if time_chunk is None else ds.extreme_events.chunk({"time": time_chunk, "ncells": -1})
            seen = []
            with monkeypatch.context() as m:
                if name.endswith("int32"):
                    original = _Tracker.refresh_dask_graph

                    def spy(self, data_bin):
                        seen.append((data_bin.dtype, data_bin.data.dtype))
                        return original(self, data_bin)

                    m.setattr(_Tracker, "refresh_dask_graph", spy)
                out = _unstructured(events, ds.mask, ds, tmp_path / name, allow_merging=False, **kwargs).run()
            results[name] = out.attrs
            assert out.attrs["allow_merging"] == 0 and out.attrs["T_fill"] == kwargs["T_fill"]
            assert np.issubdtype(out.ID_field.dtype, np.integer)
            assert int(out.ID_field.min()) == 0
            max_id = int(out.ID_field.max())
            assert max_id > 0, "no events tracked"
            assert max_id == out.attrs["N_events_final"], f"max ID {max_id} != N_events_final {out.attrs['N_events_final']}"
            got = tuple(int(out.attrs[k]) for k in _PIN_KEYS)
            assert got == UNSTRUCTURED_FILTER_PINS[name], f"{_PIN_KEYS} moved: {got} != {UNSTRUCTURED_FILTER_PINS[name]}"
            if expected is not None:
                fraction, prefiltered, filtered, n_events = expected
                assert_reasonable_bounds(out.attrs["preprocessed_area_fraction"], fraction, tolerance_relative=0.2)
                assert_count_in_reasonable_range(out.attrs["N_objects_prefiltered"], prefiltered, tolerance=5)
                assert_count_in_reasonable_range(out.attrs["N_objects_filtered"], filtered, tolerance=2)
                assert_count_in_reasonable_range(out.attrs["N_events_final"], n_events, tolerance=1)
            if name.endswith("int32"):
                assert seen, "refresh_dask_graph was never called on the unstructured path"
                for xr_dtype, block_dtype in seen:
                    assert xr_dtype == np.int32 and block_dtype == np.int32, f"refreshed ID field is {xr_dtype}/{block_dtype}"

        check(failures, name, row)

    def ordering():
        assert results["q0.9"]["N_events_final"] <= results["q0.1"]["N_events_final"], "High filtering should produce fewer events"

    check(failures, "ordering", ordering)
    assert_no_failures(failures)


@pytest.mark.slow
def test_spatially_chunked_unstructured_input_matches(client, tmp_path):
    """A mesh split into 100-cell chunks (the store has 405 cells) tracks to the single-chunk result."""
    assert on_client(client)
    ds = open_unstructured()
    kwargs = {"R_fill": 2, "area_filter_quartile": 0.5, "T_fill": 4, "allow_merging": False}
    whole = _unstructured(ds.extreme_events, ds.mask, ds, tmp_path / "whole", **kwargs).run().compute()
    events = ds.extreme_events.chunk({"time": 1, "ncells": 100})
    assert len(events.chunks[1]) > 1, "precondition: the mesh must actually be split"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        split = _unstructured(events, ds.mask.chunk({"ncells": 100}), ds, tmp_path / "split", **kwargs).run().compute()
    assert whole.attrs["N_events_final"] > 0
    assert split.attrs["N_events_final"] == whole.attrs["N_events_final"]
    np.testing.assert_array_equal(split.ID_field.values, whole.ID_field.values, err_msg="spatial chunking changed the ID field")


@pytest.mark.slow
def test_custom_dimension_names_match_standard_unstructured(client, tmp_path):
    """Custom dims (t, cell) and coords (latitude, longitude) give the standard result, without and with merging.

    The merge row runs the merging store at its own single time chunk (100 steps), the one run of
    the unstructured merge loop at that chunking, and pins its statistics there (captured on the
    current tree: pins/track/pins_fr1.out).
    """
    assert on_client(client)
    custom_names = {"dimensions": {"time": "t", "x": "cell"}, "coordinates": {"x": "longitude", "y": "latitude"}}
    rows = {
        "no_merge": (
            open_unstructured().isel(time=slice(0, 200)),
            dict(UNSTRUCTURED_KWARGS, R_fill=2, area_filter_quartile=0.1, T_fill=0, allow_merging=False),
        ),
        "merge": (open_unstructured_merging(), UNSTRUCTURED_MERGING_KWARGS),
    }
    failures = []
    for name, (data, kwargs) in rows.items():

        def row(name=name, data=data, kwargs=kwargs):
            renamed = data.rename({"time": "t", "ncells": "cell"}).rename({"lat": "latitude", "lon": "longitude"})
            original = marEx.tracker(
                data.extreme_events,
                data.mask,
                temp_dir=str(tmp_path / f"o_{name}"),
                neighbours=data.neighbours,
                cell_areas=data.cell_areas,
                **kwargs,
            ).run()
            original = original.compute()
            custom = marEx.tracker(
                renamed.extreme_events,
                renamed.mask,
                temp_dir=str(tmp_path / f"c_{name}"),
                neighbours=renamed.neighbours,
                cell_areas=renamed.cell_areas,
                **{**kwargs, **custom_names},
            ).run()
            custom = custom.compute()
            assert set(custom.ID_field.dims) == {"t", "cell"}
            assert original.attrs["N_events_final"] == custom.attrs["N_events_final"] > 0
            np.testing.assert_array_equal(original.ID_field.values, custom.ID_field.values, err_msg=f"{name}: ID_field differs")
            if kwargs["allow_merging"]:
                assert set(custom.data_vars) == EVENT_VARS
                for var in ("global_ID", "area", "presence", "time_start", "time_end", "merge_ledger"):
                    np.testing.assert_array_equal(original[var].values, custom[var].values, err_msg=f"{name}: {var} differs")
                assert data.extreme_events.chunks[0] == (100,), "the merge row must run at the store's single time chunk"
                stats = tuple(
                    int(original.attrs[k])
                    for k in ("N_objects_prefiltered", "N_objects_filtered", "N_events_final", "total_merges")
                )
                assert stats == (98, 98, 11, 9), f"(prefiltered, filtered, events, merges) at time chunk 100 moved: {stats}"
                assert int(original.ID_field.sum()) == 15288
                np.testing.assert_allclose(original.attrs["preprocessed_area_fraction"], 1.0052703490682484, rtol=1e-12)

        check(failures, name, row)
    assert_no_failures(failures)


def test_omitted_mask_is_an_all_true_mask(client):
    """No mask means exactly an all-True mask; a real mask is still honoured.

    The derived mask feeds chunking validation, the morphological fill and the area filter, so a
    shape, chunking or coordinate difference would move labels without moving values.
    """
    assert on_client(client)
    events = open_gridded().extreme_events
    kwargs = {"R_fill": 4, "area_filter_quartile": 0.4, "T_fill": 2, "allow_merging": True}
    all_true = xr.ones_like(events.isel(time=0, drop=True), dtype=bool)

    omitted_tracker = marEx.tracker(events, **kwargs)
    assert omitted_tracker.mask.dtype == bool
    assert "time" not in omitted_tracker.mask.dims
    assert bool(omitted_tracker.mask.all().compute())

    explicit = marEx.tracker(events, all_true, **kwargs).run().compute()
    omitted = omitted_tracker.run().compute()
    del omitted_tracker
    shared = sorted(set(explicit.data_vars) & set(omitted.data_vars))
    assert shared, "the two runs produced no comparable variables"
    for name in shared:
        a, b = explicit[name].values, omitted[name].values
        assert a.dtype == b.dtype, f"{name}: dtype moved"
        np.testing.assert_array_equal(a, b, err_msg=f"'{name}' differs between an omitted and an all-True mask")

    xdim = [d for d in events.dims if d != "time"][-1]
    half = all_true.copy(data=all_true.values)
    half[{xdim: slice(0, events.sizes[xdim] // 2)}] = False
    half = half.chunk(dict(zip(half.dims, [-1] * half.ndim)))
    masked = marEx.tracker(events, half, **kwargs).run()
    assert int(masked.ID_field.sum()) != int(omitted.ID_field.sum()), "a half-domain mask changed nothing"
