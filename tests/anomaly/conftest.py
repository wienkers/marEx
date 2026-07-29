"""Shared inputs for the anomaly-stage tests.

Every builder returns a field whose values live in memory (numpy, optionally wrapped in a dask
graph by ``.chunk``), never a ``.persist()``ed one: a persisted fixture holds futures on
whichever client was current when it was built.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = Path(__file__).parent.parent / "data"


GRID_DIMS = {"time": "time", "x": "lon", "y": "lat"}
MESH_DIMS = {"time": "time", "x": "ncells"}
MESH_COORDS = {"time": "time", "x": "lon", "y": "lat"}


def gridded_sst():
    """The shipped gridded SST fixture (40 years daily, 20 x 40 cells), lazily read from its store."""
    return xr.open_zarr(str(DATA_DIR / "sst_gridded.zarr"), chunks={}).to


def trend(start="2000-01-01", end="2011-12-31", n_cells=3):
    """A pure linear trend in time, so every expected climatology is an exact mean of known values."""
    time = pd.date_range(start, end, freq="D")
    values = np.arange(time.size, dtype=np.float64)[:, None, None] * np.ones((1, n_cells))
    return xr.DataArray(
        values.astype(np.float32),
        coords={"time": time, "lat": [0.0], "lon": np.arange(n_cells, dtype=float)},
        dims=("time", "lat", "lon"),
        name="sst",
    ).chunk({"time": 365})


def dec31_values(da, years):
    """Values of cell (0, 0) on 31 December of each of ``years``."""
    t = da.time.to_index()
    return np.array([float(da.values[t.get_loc(pd.Timestamp(f"{y}-12-31")), 0, 0]) for y in years])
