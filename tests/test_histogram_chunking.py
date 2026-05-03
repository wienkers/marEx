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
import marEx.detect.extremes.histogram as H

DATA_DIR = Path(__file__).parent / "data"


def _single_tile(da, dim, **kwargs):
    return da.chunk({d: -1 for d in da.dims})


def _many_tiles(da, dim, **kwargs):
    return da.chunk({**{d: 7 for d in da.dims if d != dim}, dim: -1})


def test_histogram_quantile_tiling_invariant(dask_client_gridded, monkeypatch):
    """The 1D histogram quantile is identical under a single spatial chunk vs many small tiles."""
    sst = xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to.isel(time=slice(0, 4 * 365))
    sst = sst.chunk({"time": 30, "lat": -1, "lon": -1}).persist()
    anom = marEx.compute_normalised_anomaly(
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
