"""End-to-end runs: detect -> write -> reload -> track -> write, on both grid types, plus the store-encoding write paths.

The gridded run uses a synthetic field with one injected warm blob, so the expected event is known by
hand: a 3-year daily record on a global 9-degree grid (20 x 40 cells) with a seasonal cycle and iid
noise (sd 0.3 K), plus +5 K on a 6 x 6 block of cells on days 400-409. At the 99th percentile each
cell's threshold is exceeded on about 11 of its 1095 days; the ten blob days are each cell's top ten
(5 K is about 17 noise sd), so all 360 blob cell-days are extreme, while the noise extremes form specks
of at most a few cells. With ``R_fill=1``, ``T_fill=0`` and a 12-cell area filter, the tracker returns
exactly one event: days 400-409, inside the block grown by ``R_fill``.

Every helper that touches a lazy or persisted result returns computed, numpy-backed data only, and
re-raises a failure as an ``AssertionError`` carrying the error's message. A persisted result kept
alive by a fixture value or a traceback would hold futures on the session client past the module.
"""

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from distributed.client import futures_of

import marEx
from marEx.core.encoding import STORE_ENCODING_KEYS, clear_store_encoding, write_zarr
from marEx.track import morphology

from .conftest import assert_no_failures, check, on_client, released

DATA_DIR = Path(__file__).parent / "data"
PACKAGE_DIR = Path(marEx.__file__).parent

# The injected blob: day and cell-index ranges on the synthetic grid.
N_YEARS, NY, NX = 3, 20, 40
BLOB_DAYS, BLOB_LAT, BLOB_LON = range(400, 410), range(7, 13), range(17, 23)
R_FILL = 1

DETECT_KWARGS = {
    "method_anomaly": "detrend_harmonic",
    "method_extreme": "global_percentile",
    # (1 - 0.99) * 1095 = 11 exceedances per cell, enough to hold all ten blob days. At 99.5 only
    # about 5 are, and the event breaks into pieces.
    "threshold_percentile": 99,
    "dimensions": {"x": "lon", "y": "lat"},  # "time" is left to its default
    "dask_chunks": {"time": 25},
}
SHIFTING_KWARGS = {
    "method_anomaly": "shifting_baseline",
    "method_extreme": "seasonal_percentile",
    "threshold_percentile": 85,
    "window_years": 2,
    "smooth_days": 7,
    "window_days": 21,
    "window_spatial": 7,
    "dimensions": {"x": "lon", "y": "lat"},
    "dask_chunks": {"time": 25},
}
# Noise specks are at most a few cells; the blob is 36 cells on every one of its days.
BLOB_TRACKER_KWARGS = {"R_fill": R_FILL, "T_fill": 0, "area_filter_absolute": 12, "allow_merging": False, "quiet": True}

# Event cells on each blob day: the 6 x 6 block after the tracker's closing and opening at R_fill.
EVENT_CELLS_PER_DAY = [34, 32, 34, 32, 36, 32, 32, 34, 35, 35]

INPUT_ATTRS = {"units": "degC", "standard_name": "sea_surface_temperature"}


def _write_and_reload(ds, tmp_path, name):
    """Write the lazy ``ds`` to zarr and to netcdf; return both reloaded, in memory."""
    ds.to_zarr(tmp_path / f"{name}.zarr", mode="w")
    ds.to_netcdf(tmp_path / f"{name}.nc")
    reloaded = {"zarr": xr.open_zarr(tmp_path / f"{name}.zarr").load()}
    with xr.open_dataset(tmp_path / f"{name}.nc") as nc:
        reloaded["netcdf"] = nc.load()
    return reloaded


def _check_roundtrips(failures, reloaded, reference):
    """One row per format: every variable and coordinate survives exactly, with its dims and method attributes."""
    for fmt, got in reloaded.items():

        def row(got=got):
            assert set(got.data_vars) == set(reference.data_vars), (
                f"data_vars {sorted(got.data_vars)}, wrote {sorted(reference.data_vars)}"
            )
            for var in reference.data_vars:
                np.testing.assert_array_equal(got[var].values, reference[var].values, err_msg=f"variable '{var}' changed")
                assert got[var].dims == reference[var].dims, f"dims of '{var}' changed: {got[var].dims}"
            for coord in reference.coords:
                np.testing.assert_array_equal(got[coord].values, reference[coord].values, err_msg=f"coord '{coord}' changed")
            for key in ("method_anomaly", "method_extreme"):
                assert got.attrs.get(key) == reference.attrs.get(key), f"attribute '{key}' changed"

        check(failures, f"round trip {fmt}", row)


def _blob_field():
    """The synthetic 3-year field with the injected blob, dask-backed and holding no futures."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2000-01-01", periods=N_YEARS * 365, freq="D")
    lat = np.linspace(-85.5, 85.5, NY)  # 9-degree spacing, pole to pole
    lon = np.arange(NX) * 9.0  # 0 to 351: a full turn, endpoint-exclusive
    seasonal = 2.0 * np.cos(2 * np.pi * (time.dayofyear.values - 30) / 365.25)
    field = 20.0 - 0.1 * np.abs(lat)[None, :, None] + seasonal[:, None, None] + rng.normal(0.0, 0.3, (time.size, NY, NX))
    field[BLOB_DAYS.start : BLOB_DAYS.stop, BLOB_LAT.start : BLOB_LAT.stop, BLOB_LON.start : BLOB_LON.stop] += 5.0
    sst = xr.DataArray(
        field.astype(np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="sst",
        attrs=dict(INPUT_ATTRS),
    )
    return sst.chunk({"time": 365, "lat": -1, "lon": -1})


@pytest.mark.nocov
@pytest.mark.slow
@pytest.mark.integration
def test_gridded_detect_save_track_save(client, tmp_path):
    """The injected blob is found, written and reloaded exactly, and tracked as exactly one event.

    The tracker is given the pipeline's own lazy ``extreme_events`` and ``mask``. A shifting-baseline
    run on the same input then drops exactly its first ``window_years`` calendar years, and its
    day-of-year thresholds vary at every cell.
    """
    assert on_client(client)
    sst = _blob_field()
    assert not futures_of(sst), "the synthetic input holds futures"
    assert sst.attrs == INPUT_ATTRS

    def run():
        # xarray >= 2025.11 keeps attrs by default; force that here so the attrs row holds on any xarray.
        with xr.set_options(keep_attrs=True):
            ex = marEx.preprocess_data(sst, **DETECT_KWARGS)
        reference = ex.compute()
        reloaded = _write_and_reload(ex, tmp_path, "gridded")
        events = marEx.tracker(ex.extreme_events, ex.mask, temp_dir=str(tmp_path / "track"), **BLOB_TRACKER_KWARGS).run()
        events = events.compute()
        shifted = marEx.preprocess_data(sst, **SHIFTING_KWARGS)
        return reference, reloaded, events, shifted.time.values, shifted.thresholds.compute()

    reference, reloaded, events, shifted_time, thresholds = released(run)
    failures = []

    def blob():
        n_blob_extremes = int(
            reference.extreme_events.isel(
                time=slice(BLOB_DAYS.start, BLOB_DAYS.stop),
                lat=slice(BLOB_LAT.start, BLOB_LAT.stop),
                lon=slice(BLOB_LON.start, BLOB_LON.stop),
            ).sum()
        )
        assert n_blob_extremes == 360, f"{n_blob_extremes} of the 360 blob cell-days are extreme"

    def attrs():
        for name in ("dat_anomaly", "extreme_events"):
            assert reference[name].attrs == {}, f"{name} inherited the input's attrs: {reference[name].attrs}"

    def event():
        ids = events.ID_field.values
        assert events.attrs["N_events_final"] == 1, f"N_events_final = {events.attrs['N_events_final']}, the blob is one event"
        found = sorted(int(i) for i in np.unique(ids[ids > 0]))
        assert found == [1], f"event IDs {found}, expected [1]"
        t, y, x = np.nonzero(ids)
        assert sorted(set(t.tolist())) == list(BLOB_DAYS), f"the event spans days {sorted(set(t.tolist()))}, the blob 400-409"
        lat_ok = range(BLOB_LAT.start - R_FILL, BLOB_LAT.stop + R_FILL)
        lon_ok = range(BLOB_LON.start - R_FILL, BLOB_LON.stop + R_FILL)
        outside = sorted({(int(j), int(i)) for j, i in zip(y, x) if j not in lat_ok or i not in lon_ok})
        assert not outside, f"event cells (lat, lon index) outside the blob grown by R_fill: {outside[:10]}"
        # The upper bound above does not see an event that shrinks; the per-day size is pinned (seeded input).
        sizes = [int((ids[d] == 1).sum()) for d in BLOB_DAYS]
        assert sizes == EVENT_CELLS_PER_DAY, f"event cells per blob day {sizes}, expected {EVENT_CELLS_PER_DAY}"

    def shifting():
        # The first two calendar years are trimmed: 2000 (leap, 366 days) and 2001 (365 days).
        n_trimmed = int((sst.time.dt.year < 2000 + SHIFTING_KWARGS["window_years"]).sum())
        assert n_trimmed == 731
        assert shifted_time.size == sst.sizes["time"] - n_trimmed, (
            f"{shifted_time.size} steps after the trim, expected {sst.sizes['time']} - {n_trimmed}"
        )
        assert shifted_time[0] == np.datetime64("2002-01-01"), f"the trimmed record starts {shifted_time[0]}"
        assert thresholds.dims[-1] == "dayofyear", thresholds.dims
        n_nan = int(np.isnan(thresholds.values).sum())
        assert n_nan == 0, f"{n_nan} NaN thresholds"
        variance = thresholds.var("dayofyear").values
        assert (variance > 0).all(), f"{int((variance <= 0).sum())} cells have thresholds constant in day of year"

    check(failures, "blob detected", blob)
    check(failures, "attrs", attrs)
    _check_roundtrips(failures, reloaded, reference)
    check(failures, "tracked event", event)
    check(failures, "shifting baseline", shifting)
    assert_no_failures(failures)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.smoke
def test_unstructured_detect_save_roundtrip(client, tmp_path):
    """Unstructured detect on the shipped mesh: one spatial dim, events found, mesh attached, round trips exact.

    ``sst_unstructured.zarr`` is the same 405 cells, in the same order, as ``extremes_unstructured.zarr``,
    so that store's coordinates, neighbours and cell areas describe this field.
    """
    assert on_client(client)
    mesh = xr.open_zarr(str(DATA_DIR / "extremes_unstructured.zarr"), chunks={})[["lat", "lon", "neighbours", "cell_areas"]].load()
    sst = xr.open_zarr(str(DATA_DIR / "sst_unstructured.zarr"), chunks={}).to.isel(time=slice(0, 4 * 365))
    assert not futures_of(sst), "the input holds futures; keep it rooted in the zarr store"
    sst = sst.assign_coords(lat=("ncells", mesh.lat.values), lon=("ncells", mesh.lon.values))
    sst = sst.where(sst.ncells != 2)  # one all-NaN cell, so the mask has a False

    def run():
        ex = marEx.preprocess_data(
            sst,
            method_anomaly="detrend_harmonic",
            method_extreme="global_percentile",
            dimensions={"time": "time", "x": "ncells"},
            coordinates={"time": "time", "x": "lon", "y": "lat"},
            dask_chunks={"time": 25},
            neighbours=mesh.neighbours,
            cell_areas=mesh.cell_areas,
        )
        return ex.compute(), _write_and_reload(ex, tmp_path, "unstructured")

    reference, reloaded = released(run)
    failures = []

    def structure():
        assert reference.extreme_events.dims == ("time", "ncells"), reference.extreme_events.dims
        n_extremes = int(reference.extreme_events.sum())
        assert n_extremes > 0, "no extreme events found in the unstructured data"
        assert not bool(reference.mask.values[2]), "the all-NaN cell is not masked out"
        np.testing.assert_array_equal(reference.neighbours.values, mesh.neighbours.values, err_msg="neighbours changed")
        np.testing.assert_array_equal(reference.cell_areas.values, mesh.cell_areas.values, err_msg="cell_areas changed")

    check(failures, "structure", structure)
    _check_roundtrips(failures, reloaded, reference)
    assert_no_failures(failures)


def test_tracked_events_roundtrip(client, tmp_path):
    """The events dataset of one merging tracker run writes to zarr and netcdf and reloads exactly."""
    assert on_client(client)
    ex = xr.open_zarr(str(DATA_DIR / "extremes_gridded.zarr"), chunks={})
    assert not futures_of(ex), "the input holds futures; keep it rooted in the zarr store"
    data_bin = ex.extreme_events.chunk({"time": 2, "lat": -1, "lon": -1})
    mask = ex.mask.where((ex.lat < 85) & (ex.lat > -90), other=False)

    def run():
        tracker = marEx.tracker(
            data_bin,
            mask,
            area_filter_quartile=0.5,
            R_fill=4,
            T_fill=2,
            allow_merging=True,
            overlap_threshold=0.5,
            nn_partitioning=True,
            temp_dir=str(tmp_path / "track"),
            quiet=True,
        )
        events, merges = tracker.run(return_merges=True)
        return events.compute(), int(merges.sizes.get("merge_ID", 0)), _write_and_reload(events, tmp_path, "events")

    reference, n_merges, reloaded = released(run)
    assert n_merges > 0, "the tracker run made no merges"
    failures = []
    _check_roundtrips(failures, reloaded, reference)
    assert_no_failures(failures)


def test_checkpoint_save_then_load(client, tmp_path):
    """``run_preprocess(checkpoint='save')`` writes a format-2 input's coordinates; ``'load'`` reads back the same."""
    assert on_client(client)
    ex = xr.open_zarr(str(DATA_DIR / "extremes_gridded.zarr"), chunks={})
    held = futures_of(ex)
    assert not held, f"the input holds {len(held)} future(s); keep it rooted in the zarr store"
    events = ex.extreme_events.chunk({"time": 2, "lat": -1, "lon": -1})
    kwargs = {"area_filter_quartile": 0.5, "R_fill": 2, "T_fill": 2, "temp_dir": str(tmp_path), "quiet": True}

    def run():
        saved, _ = marEx.tracker(events, ex.mask, **kwargs).run_preprocess(checkpoint="save")
        saved = saved.load()
        loaded, _ = marEx.tracker(events, ex.mask, **kwargs).run_preprocess(checkpoint="load")
        return saved, loaded.load()

    saved, loaded = released(run)
    xr.testing.assert_identical(saved, loaded)


@pytest.fixture
def scratch():
    d = tempfile.mkdtemp(prefix="marex_test_store_enc_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_format2_inputs_write_under_zarr3(scratch):
    """Writes of variables built from a format-2 store succeed under zarr-python 3.

    A format-2 store's ``numcodecs`` codecs ride along in ``.encoding`` and zarr 3 refuses them in a
    format-3 store ("Expected a BytesBytesCodec"). Rows: the two public climatology helpers through a
    plain ``to_zarr``, and ``refresh_dask_graph``'s internal write on both shipped extremes stores.
    """
    sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.isel(time=slice(0, 3 * 365))
    failures = []

    for helper in ("rolling_climatology", "smoothed_rolling_climatology"):

        def climatology(helper=helper):
            clim = getattr(marEx.anomaly, helper)(sst, window_years=2, dimensions={"time": "time", "x": "lon", "y": "lat"})
            store = f"{scratch}/{helper}.zarr"
            clim.rename("clim").to_dataset().to_zarr(store, mode="w")
            assert xr.open_zarr(store).clim.shape == clim.shape, f"{helper}: the written shape differs"

        check(failures, helper, climatology)

    for store, chunks in (("extremes_gridded.zarr", {}), ("extremes_unstructured.zarr", {"time": 2, "ncells": -1})):

        def refresh(store=store, chunks=chunks):
            ex = xr.open_zarr(str(DATA_DIR / store), chunks=chunks)
            oid = xr.where(ex.extreme_events, 1, 0)
            refreshed = morphology.refresh_dask_graph(oid, f"{scratch}/refresh_{store}")
            assert (refreshed.values == oid.values).all(), f"refresh_dask_graph changed {store}"

        check(failures, f"refresh_dask_graph {store}", refresh)
    assert_no_failures(failures)


def _codec_keys(obj):
    return {name: sorted(k for k in v.encoding if k in STORE_ENCODING_KEYS) for name, v in obj.variables.items()}


def _cf_keys(obj):
    return {
        name: {k: v for k, v in var.encoding.items() if k in ("units", "calendar", "dtype")} for name, var in obj.variables.items()
    }


def test_store_encoding_is_cleared_on_a_copy(scratch):
    """``clear_store_encoding`` drops the codecs and keeps the CF keys; ``write_zarr`` clears a copy, not the caller's."""
    failures = []

    def clear():
        ds = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={})
        assert any(_codec_keys(ds).values()), "the store no longer carries codecs; the row is vacuous"
        cf_before = _cf_keys(ds)
        clear_store_encoding(ds)
        left = {name: keys for name, keys in _codec_keys(ds).items() if keys}
        assert not left, f"codecs left after clear_store_encoding: {left}"
        assert _cf_keys(ds) == cf_before, "clear_store_encoding changed the CF encoding"
        ds.isel(time=slice(0, 4)).to_zarr(f"{scratch}/cleared.zarr", mode="w")

    def caller_untouched():
        da = xr.open_zarr(str(DATA_DIR / "extremes_gridded.zarr"), chunks={}).extreme_events
        before = _codec_keys(da.to_dataset())
        assert any(before.values()), "the store no longer carries codecs; the row is vacuous"
        write_zarr(da, f"{scratch}/written.zarr", mode="w")
        assert _codec_keys(da.to_dataset()) == before, "write_zarr cleared the caller's encoding"

    check(failures, "clear_store_encoding", clear)
    check(failures, "write_zarr", caller_untouched)
    assert_no_failures(failures)


def test_every_to_zarr_goes_through_write_zarr():
    """Tripwire: a write that bypasses the helper is how three sites were missed twice.

    Under zarr 2 a bypassing write still succeeds on a format-2 input, so this source check is the only
    guard there.
    """
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if path.name == "encoding.py" and path.parent.name == "core":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if ">>>" in code or code.lstrip().startswith("... "):
                continue
            if re.search(r"\.to_zarr\(", code):
                offenders.append(f"{path.relative_to(PACKAGE_DIR)}:{lineno}: {line.strip()}")
    assert not offenders, "route these writes through marEx.core.encoding.write_zarr:\n" + "\n".join(offenders)
