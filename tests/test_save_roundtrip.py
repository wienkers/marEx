"""Save -> reload round-trip tests for ``marEx.preprocess_data`` output.

These tests guard the saving path that real/large runs depend on: the dataset
returned by ``preprocess_data`` (lazy, Dask-backed, persisted) must be writable to
*both* Zarr and NetCDF under a distributed scheduler and reload bit-for-bit, for
gridded and unstructured grids alike. The existing golden/preprocessing tests only
ever ``.compute()`` the result, so they never exercise the writer (``dask.array.store``)
path where the historical "tuple"/serialisation and NetCDF bool-attr bugs lived.

Each test saves the *lazy* dataset (not a pre-computed copy) so the Dask store path is
actually exercised, then compares every data variable element-for-element against the
in-memory result.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx

DATA_DIR = Path(__file__).parent / "data"


def _assert_roundtrip(reloaded: xr.Dataset, reference: xr.Dataset):
    """Every variable/coord must survive the save -> reload round-trip exactly."""
    ref = reference.compute()
    got = reloaded.compute()

    assert set(got.data_vars) == set(ref.data_vars), "data_vars differ after round-trip"
    for var in ref.data_vars:
        np.testing.assert_array_equal(got[var].values, ref[var].values, err_msg=f"variable '{var}' changed on round-trip")
        assert got[var].dims == ref[var].dims, f"dims of '{var}' changed on round-trip"
    for coord in ref.coords:
        np.testing.assert_array_equal(got[coord].values, ref[coord].values, err_msg=f"coord '{coord}' changed on round-trip")
    # Method-selection attributes must persist (and be NetCDF-safe).
    for key in ("method_anomaly", "method_extreme"):
        assert got.attrs.get(key) == ref.attrs.get(key)


@pytest.fixture(scope="module")
def gridded_extremes_lazy(dask_client_gridded):
    """A small lazy gridded extremes dataset (1D-histogram / global path)."""
    sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to
    sst = sst.isel(time=slice(0, 4 * 365)).persist()
    sst = sst.where(~((sst.lat == sst.lat[1]) & (sst.lon == sst.lon[1])), np.nan)
    return marEx.preprocess_data(
        sst,
        method_anomaly="detrend_harmonic",
        method_extreme="global_percentile",
        dimensions={"time": "time", "x": "lon", "y": "lat"},
        dask_chunks={"time": 25},
    )


@pytest.fixture(scope="module")
def unstructured_extremes_lazy(dask_client_gridded):
    """A small lazy unstructured extremes dataset (exercises neighbours/cell_areas/coords)."""
    ds = xr.open_zarr(str(DATA_DIR / "sst_unstructured.zarr"), chunks={}).persist()
    sst = ds.to.where(~(ds.to.ncells == 2), np.nan)
    ncells = sst.sizes["ncells"]
    sst = sst.assign_coords(
        lat=xr.DataArray(np.linspace(-90, 90, ncells), dims=["ncells"], name="lat"),
        lon=xr.DataArray(np.linspace(-180, 180, ncells), dims=["ncells"], name="lon"),
    )
    neighbours = xr.DataArray(np.random.default_rng(0).integers(0, ncells, (3, ncells)), dims=["nv", "ncells"])
    cell_areas = xr.DataArray(np.ones(ncells) * 1000.0, dims=["ncells"])
    return marEx.preprocess_data(
        sst,
        method_anomaly="detrend_harmonic",
        method_extreme="global_percentile",
        dimensions={"time": "time", "x": "ncells"},
        coordinates={"time": "time", "x": "lon", "y": "lat"},
        dask_chunks={"time": 25},
        neighbours=neighbours,
        cell_areas=cell_areas,
    )


@pytest.mark.parametrize("fmt", ["zarr", "netcdf"])
def test_roundtrip_gridded(fmt, gridded_extremes_lazy, tmp_path):
    ds = gridded_extremes_lazy
    if fmt == "zarr":
        target = tmp_path / "gridded.zarr"
        ds.to_zarr(target, mode="w")
        reloaded = xr.open_zarr(target)
    else:
        target = tmp_path / "gridded.nc"
        ds.to_netcdf(target)
        reloaded = xr.open_dataset(target)
    _assert_roundtrip(reloaded, ds)


@pytest.mark.parametrize("fmt", ["zarr", "netcdf"])
def test_roundtrip_unstructured(fmt, unstructured_extremes_lazy, tmp_path):
    ds = unstructured_extremes_lazy
    if fmt == "zarr":
        target = tmp_path / "unstructured.zarr"
        ds.to_zarr(target, mode="w")
        reloaded = xr.open_zarr(target)
    else:
        target = tmp_path / "unstructured.nc"
        ds.to_netcdf(target)
        reloaded = xr.open_dataset(target)
    _assert_roundtrip(reloaded, ds)


@pytest.fixture(scope="module")
def tracked_events_lazy(dask_client_gridded):
    """A small lazy tracked-events dataset (the tracker's user-facing output)."""
    ex = xr.open_zarr(str(DATA_DIR / "extremes_gridded.zarr"), chunks={}).persist()
    data_bin = ex.extreme_events.chunk({"time": 2, "lat": -1, "lon": -1})
    mask = ex.mask.where((ex.lat < 85) & (ex.lat > -90), other=False)
    tracker = marEx.tracker(
        data_bin,
        mask,
        area_filter_quartile=0.5,
        R_fill=4,
        T_fill=2,
        allow_merging=True,
        overlap_threshold=0.5,
        nn_partitioning=True,
        quiet=True,
    )
    events_ds, _ = tracker.run(return_merges=True)
    return events_ds


@pytest.mark.parametrize("fmt", ["zarr", "netcdf"])
def test_roundtrip_tracked_events(fmt, tracked_events_lazy, tmp_path):
    """The tracker's lazy events_ds output must save+reload exactly (zarr and netcdf)."""
    ds = tracked_events_lazy
    if fmt == "zarr":
        target = tmp_path / "events.zarr"
        ds.to_zarr(target, mode="w")
        reloaded = xr.open_zarr(target)
    else:
        target = tmp_path / "events.nc"
        ds.to_netcdf(target)
        reloaded = xr.open_dataset(target)
    _assert_roundtrip(reloaded, ds)


def test_clear_store_encoding_drops_codecs_and_keeps_cf_keys(tmp_path):
    """A format-2 input's codecs must not reach a write; CF encoding must survive.

    zarr-python 3 refuses to write a format-2 ``numcodecs`` compressor into a format-3
    store ("Expected a BytesBytesCodec"), so a variable opened from a format-2 store and
    saved with the default ``to_zarr`` failed before ``clear_store_encoding`` existed.
    """
    from marEx.core.encoding import STORE_ENCODING_KEYS, clear_store_encoding

    ds = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={})
    assert any("compressor" in v.encoding or "compressors" in v.encoding for v in ds.variables.values())
    cf_before = {
        name: {k: v for k, v in var.encoding.items() if k in ("units", "calendar", "dtype")} for name, var in ds.variables.items()
    }

    clear_store_encoding(ds)

    for name, var in ds.variables.items():
        assert not set(STORE_ENCODING_KEYS) & set(var.encoding), name
        assert {k: v for k, v in var.encoding.items() if k in ("units", "calendar", "dtype")} == cf_before[name]
    ds.isel(time=slice(0, 4)).to_zarr(tmp_path / "out.zarr", mode="w")


def test_detect_outputs_do_not_inherit_input_attrs(gridded_extremes_lazy):
    """xarray >= 2025.11 keeps attrs by default: the input SST's standard_name, units and
    valid_min/valid_max must not reappear on the anomaly or the boolean event mask."""
    assert DATA_DIR.joinpath("sst_gridded.zarr").exists()
    for name in ("dat_anomaly", "extreme_events"):
        assert gridded_extremes_lazy[name].attrs == {}, name


@pytest.mark.parametrize("helper", ["rolling_climatology", "smoothed_rolling_climatology"])
def test_climatology_helpers_write_to_zarr(helper, tmp_path):
    """The public climatology helpers return a variable built from a format-2 input; the default
    to_zarr must accept it under zarr-python 3 (it raised "Expected a BytesBytesCodec")."""
    sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.isel(time=slice(0, 3 * 365))
    clim = getattr(marEx.anomaly, helper)(sst, window_years=2, dimensions={"time": "time", "x": "lon", "y": "lat"})
    clim.rename("clim").to_dataset().to_zarr(tmp_path / "clim.zarr", mode="w")
    assert xr.open_zarr(tmp_path / "clim.zarr").clim.shape == clim.shape
