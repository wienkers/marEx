"""Graph-shape pins for ``method_percentile='exact'``.

Values cannot catch these: both changes are pure layout, bit-identical by construction.
What they bound is how a task grows with the series length.
"""

import numpy as np
import pandas as pd
import xarray as xr

import marEx.extremes.global_percentile as global_percentile
import marEx.extremes.seasonal_percentile as seasonal_percentile


def _mesh(n_time, n_cells=20000):
    t = pd.date_range("2000-01-01", periods=n_time, freq="D")
    data = np.zeros((n_time, n_cells), dtype=np.float32)
    return xr.DataArray(data, dims=("time", "ncells"), coords={"time": t}, name="a").chunk({"time": 10})


def _quantile_cell_tile(monkeypatch, da):
    seen = {}
    real = xr.DataArray.quantile

    def spy(self, *args, **kwargs):
        seen["tile"] = max(self.chunks[self.dims.index("ncells")])
        return real(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "quantile", spy)
    global_percentile._identify_extremes_constant(da, 95, "exact", {"time": "time", "x": "ncells"})
    return seen["tile"]


def test_mesh_exact_tile_shrinks_with_the_series(monkeypatch):
    """Twice the series, half the cells: the per-task element count is invariant in
    n_time. 50,000 cells give a 300-cell tile from n_cells alone, which is what both lengths got before."""
    monkeypatch.setattr(global_percentile, "TASK_ELEMENTS", 24_000)
    short = _quantile_cell_tile(monkeypatch, _mesh(100, n_cells=50_000))
    long = _quantile_cell_tile(monkeypatch, _mesh(200, n_cells=50_000))
    assert (short, long) == (240, 120)


def test_mesh_exact_tile_keeps_its_floor(monkeypatch):
    """A budget smaller than one 100-cell tile cannot shred the mesh into single cells."""
    monkeypatch.setattr(global_percentile, "TASK_ELEMENTS", 1_000)
    assert _quantile_cell_tile(monkeypatch, _mesh(200)) == 100


def test_seasonal_exact_graph_carries_the_cycle_index_not_a_mask_table(monkeypatch):
    """The ufunc kwargs must scale with n_time, not with cycle.length x n_time."""
    captured = {}
    real = seasonal_percentile.xr.apply_ufunc

    def spy(func, *args, **kwargs):
        captured.update(kwargs.get("kwargs", {}))
        return real(func, *args, **kwargs)

    monkeypatch.setattr(seasonal_percentile.xr, "apply_ufunc", spy)
    n_time = 3653
    t = pd.date_range("2000-01-01", periods=n_time, freq="D")
    da = xr.DataArray(
        np.zeros((n_time, 2, 3), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": t, "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
        name="a",
    ).chunk({"time": -1})
    seasonal_percentile._identify_extremes_seasonal(
        da, 95, method_percentile="exact", dimensions={"time": "time", "y": "lat", "x": "lon"}, coordinates={"time": "time"}
    )
    payload = sum(v.nbytes for v in captured.values() if isinstance(v, np.ndarray))
    payload += sum(a.nbytes for v in captured.values() if isinstance(v, list) for a in v if isinstance(a, np.ndarray))
    assert payload <= 4 * n_time, f"{payload} B of arrays ride in the graph (a 366 x n_time table is {366 * n_time} B)"
