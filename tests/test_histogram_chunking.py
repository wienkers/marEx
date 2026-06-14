"""Tiling invariance of the 1D histogram-quantile kernel (the memory-safety fix).

``_compute_histogram_quantile_1d`` tiles the spatial dimensions (via
``_chunk_spatial_for_histogram``) so the histogram reduction runs at full resolution
without exhausting worker memory. Because each cell's full reduced-axis time series stays
within a single tile, the computed quantile must be bit-for-bit identical regardless of the
tiling. This test pins that invariant -- it is the empirical backing for the "memory-safe
*and* bit-exact" guarantee that replaced the removed temp-checkpoint workaround.
"""

from pathlib import Path

import numpy as np
import xarray as xr

import marEx
import marEx.extremes.histogram as H

DATA_DIR = Path(__file__).parent / "data"


def _single_tile(da, dim, **kwargs):
    return da.chunk({d: -1 for d in da.dims})


def _many_tiles(da, dim, **kwargs):
    return da.chunk({**{d: 7 for d in da.dims if d != dim}, dim: -1})


def test_histogram_quantile_tiling_invariant(dask_client_gridded, monkeypatch):
    """The 1D histogram quantile is identical under a single spatial chunk vs many small tiles."""
    sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.isel(time=slice(0, 4 * 365))
    sst = sst.chunk({"time": 30, "lat": -1, "lon": -1}).persist()
    anom = marEx.anomaly.compute_normalised_anomaly(
        sst, method_anomaly="detrend_harmonic", dimensions={"time": "time", "x": "lon", "y": "lat"}
    ).dat_anomaly
    anom.name = "dat_anomaly"

    # Single spatial chunk (one tile covering the whole field).
    monkeypatch.setattr(H, "_chunk_spatial_for_histogram", _single_tile)
    res_single = H._compute_histogram_quantile_1d(anom, 0.95, dim="time").compute()

    # Many small spatial tiles (7x7 cells -> multiple tiles for the 20x40 fixture).
    monkeypatch.setattr(H, "_chunk_spatial_for_histogram", _many_tiles)
    res_multi = H._compute_histogram_quantile_1d(anom, 0.95, dim="time").compute()

    np.testing.assert_array_equal(
        res_single.values,
        res_multi.values,
        err_msg="histogram-quantile threshold changed with spatial tiling",
    )


def _small_tiles_3d(da, dim, **kwargs):
    return da.chunk({**{d: 3 for d in da.dims if d != dim}, dim: -1})


def test_histogram_quantile_tiling_invariant_with_an_extra_dim(dask_client_gridded, monkeypatch):
    """The tiling invariance holds with a depth axis, and the tile covers it.

    ``_chunk_spatial_for_histogram`` derives its spatial dims from ``da.dims``, so it
    was already rank-agnostic. This pins that: a field with an extra dimension must
    tile that dimension too, and the quantile must not move when it does.
    """
    sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.isel(time=slice(0, 3 * 365))
    da3 = xr.concat([sst, sst * 0.8 - 1.5, sst * 1.3 + 2.25], dim="depth")
    da3 = da3.assign_coords(depth=np.arange(3, dtype=np.float32)).transpose("time", "depth", "lat", "lon")
    da3 = da3.chunk({"time": 30, "depth": -1, "lat": -1, "lon": -1}).persist()

    anom = marEx.anomaly.compute_normalised_anomaly(
        da3, method_anomaly="detrend_harmonic", dimensions={"time": "time", "x": "lon", "y": "lat"}
    ).dat_anomaly
    anom.name = "dat_anomaly"

    # The default tiling must cover every spatial dim, depth included.
    tiled = H._chunk_spatial_for_histogram(anom, "time")
    assert set(tiled.chunksizes) == {"time", "depth", "lat", "lon"}
    assert tiled.chunksizes["time"] == (anom.sizes["time"],), "the reduced axis must stay whole"

    monkeypatch.setattr(H, "_chunk_spatial_for_histogram", _single_tile)
    res_single = H._compute_histogram_quantile_1d(anom, 0.95, dim="time").compute()

    monkeypatch.setattr(H, "_chunk_spatial_for_histogram", _small_tiles_3d)
    res_multi = H._compute_histogram_quantile_1d(anom, 0.95, dim="time").compute()

    assert set(res_single.dims) == {"depth", "lat", "lon"}
    np.testing.assert_array_equal(
        res_single.values,
        res_multi.values,
        err_msg="histogram-quantile threshold changed with spatial tiling on a 3D+time field",
    )
