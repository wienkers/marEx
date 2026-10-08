"""The slab engine of the seasonal histogram path returns the dense engine's thresholds.

``_compute_histogram_quantile_2d`` builds the same integer (cycle x n_bins) counts per cell two
ways: ``engine="dense"`` with flox over a whole spatial tile, ``engine="slab"`` (the default) one
cell at a time from the transposed uint16 bin index. Both feed ``_rolling_histogram_quantile``, so
the thresholds must be IDENTICAL -- zero tolerance, including dims, coords and dtype.

The structure tests pin what the slab engine is for: its tile is budgeted on what a task reads,
so it does not shrink with the cycle length or the bin count, where the dense tile does.
"""

import dask
import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from marEx.core.time_axis import SeasonalCycle, resolve_cycle
from marEx.extremes import histogram as H

GRID_DIMS = {"time": "time", "y": "lat", "x": "lon"}
MESH_DIMS = {"time": "time", "x": "ncells"}


def _field(freq="D", periods=365 * 6 + 1, mesh=False, seed=0, shape=(6, 7)):
    t = pd.date_range("2000-01-01", periods=periods, freq=freq)
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1, (t.size,) + shape).astype(np.float32)
    values[:, 0, :2] = np.nan  # an all-NaN strip
    values[::5, 3, 3] = np.nan  # a cell with gaps
    if mesh:
        da = xr.DataArray(values.reshape(t.size, 42), dims=("time", "ncells"), coords={"time": t}, name="anom")
    else:
        da = xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            name="anom",
            coords={"time": t, "lat": np.linspace(-5, 5, shape[0]), "lon": np.linspace(0, 12, shape[1])},
        )
    cycle = resolve_cycle(da, "time")
    return da.assign_coords({cycle.index_name: cycle.index_of(da["time"]).compute()}), cycle


def _both(da, cycle, dims, tchunk, **kw):
    da = da.chunk({"time": tchunk})
    common = dict(dimensions=dims, cycle=cycle, window_steps=cycle.window_steps(11), precision=0.05, max_anomaly=5.0, **kw)
    dense = H._compute_histogram_quantile_2d(da, engine="dense", **common)
    slab = H._compute_histogram_quantile_2d(da, engine="slab", **common)
    return dense, slab


@pytest.mark.parametrize("tchunk", [-1, 30, 7])
@pytest.mark.parametrize("tail, q", [("upper", 0.95), ("lower", 0.05), ("upper", 0.9)])
@pytest.mark.parametrize("window_spatial", [None, 1, 3, 5])
def test_slab_matches_dense_gridded_daily(tchunk, tail, q, window_spatial):
    da, cycle = _field()
    dense, slab = _both(da, cycle, GRID_DIMS, tchunk, q=q, tail=tail, window_spatial=window_spatial)
    xr.testing.assert_identical(dense, slab)


@pytest.mark.parametrize("tail, q", [("upper", 0.95), ("lower", 0.05)])
@pytest.mark.parametrize("tchunk", [-1, 30])
def test_slab_matches_dense_mesh(tchunk, tail, q):
    da, cycle = _field(mesh=True)
    dense, slab = _both(da, cycle, MESH_DIMS, tchunk, q=q, tail=tail)
    xr.testing.assert_identical(dense, slab)


@pytest.mark.parametrize("freq, periods", [("6h", 4 * 365 * 3), ("MS", 12 * 20)])
@pytest.mark.parametrize("tail, q", [("upper", 0.95), ("lower", 0.05)])
def test_slab_matches_dense_other_cadences(freq, periods, tail, q):
    # 6-hourly: a 1464-slot cycle. Monthly: a one-step window, the kernel's pad_size == 0 branch.
    da, cycle = _field(freq=freq, periods=periods)
    dense, slab = _both(da, cycle, GRID_DIMS, 30, q=q, tail=tail, window_spatial=3)
    xr.testing.assert_identical(dense, slab)


def test_slab_matches_dense_under_the_legacy_asymmetric_bins():
    # The legacy edges open at -inf (bin centre 0 substituted); upper tail only.
    da, cycle = _field()
    edges = np.concatenate([[-np.inf], np.arange(-0.05, 5.0 + 0.05, 0.05, dtype=np.float32)], dtype=np.float32)
    common = {
        "dimensions": GRID_DIMS,
        "cycle": cycle,
        "window_steps": 11,
        "bin_edges": edges,
        "max_anomaly": 5.0,
        "q": 0.95,
        "window_spatial": 3,
    }
    dense = H._compute_histogram_quantile_2d(da.chunk({"time": 30}), engine="dense", **common)
    slab = H._compute_histogram_quantile_2d(da.chunk({"time": 30}), engine="slab", **common)
    xr.testing.assert_identical(dense, slab)


def test_slab_matches_dense_with_extra_dim():
    da, cycle = _field()
    da3 = xr.concat([da, da * 0.5, -da], dim="depth").transpose("time", "depth", "lat", "lon")
    dense, slab = _both(da3, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=3)
    xr.testing.assert_identical(dense, slab)


def test_slab_matches_dense_when_a_trailing_tile_is_narrower_than_the_halo(monkeypatch):
    """overlap merges a trailing chunk narrower than the halo into its neighbour (13 = 6+6+1 -> 6+7).

    The slab output must declare the blocks the kernel actually returns: a mis-declared chunk
    passes ``.values`` but breaks the NaN-mask ``.where`` and anything else that aligns by chunk.

    Blocks are computed on the current scheduler: under a live distributed client the rechunk is
    P2P, which a forced ``scheduler="synchronous"`` cannot run.
    """
    da, cycle = _field(shape=(13, 13))
    ntime = da.sizes["time"]
    monkeypatch.setattr(H, "_HISTOGRAM_TASK_ELEMENTS", 36 * ntime)  # tile side 6, remainder 1 < halo 2
    assert H._slab_tile_chunks(da, GRID_DIMS, 5, cycle.length)["lat"] == 6
    dense, slab = _both(da, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=5)
    xr.testing.assert_identical(dense, slab)
    raw = H._slab_seasonal_threshold(
        da.chunk({"time": 30}),
        *_edges_and_centres(),
        dimensions=GRID_DIMS,
        cycle=cycle,
        window_steps=cycle.window_steps(11),
        window_spatial=5,
        q=0.95,
        q_mirror=0.05,
        tail="upper",
    ).data
    assert len(raw.chunks[0]) > 1 and len(raw.chunks[1]) > 1
    for index in np.ndindex(*raw.numblocks):
        declared = tuple(raw.chunks[axis][i] for axis, i in enumerate(index))
        assert raw.blocks[index].compute().shape == declared


@pytest.mark.parametrize("ny, window_spatial", [(1, 5), (2, 7), (3, 9), (1, 3), (2, 5)])
def test_slab_matches_dense_when_y_is_narrower_than_the_halo(ny, window_spatial):
    # The dense window sum zero-pads y; overlap refuses a depth beyond the array, so the slab
    # engine clips the y halo to the y size (one chunk, every real row still in each window).
    da, cycle = _field(shape=(6, 12))
    da = da.isel(lat=slice(2, 2 + ny))
    dense, slab = _both(da, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=window_spatial)
    xr.testing.assert_identical(dense, slab)


@pytest.mark.parametrize("window_spatial", [1, 5])
def test_slab_matches_dense_when_a_time_is_nat(window_spatial):
    # A NaT step has a NaN cycle slot, which flox's expected groups drop.
    da, _ = _field()
    times = da["time"].values.copy()
    times[100] = np.datetime64("NaT")
    da = da.assign_coords(time=times)
    cycle = resolve_cycle(da, "time")
    da = da.assign_coords({cycle.index_name: cycle.index_of(da["time"]).compute()})
    dense, slab = _both(da, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=window_spatial)
    xr.testing.assert_identical(dense, slab)


def test_slab_matches_dense_when_slots_exceed_the_cycle():
    # Day 366 under an explicit 365-slot cycle is outside the expected groups and dropped.
    da, _ = _field()
    cycle = SeasonalCycle("dayofyear", 365, 1.0)
    da = da.assign_coords({cycle.index_name: cycle.index_of(da["time"]).compute()})
    assert int(da[cycle.index_name].max()) == 366
    dense, slab = _both(da, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=5)
    xr.testing.assert_identical(dense, slab)


def test_slab_matches_dense_when_edge_rounding_realises_65536_bins(monkeypatch):
    # n_bins=65535 passes the guard, but float32 edges realise 65536 bins: the y pad index
    # n_bins no longer fits uint16.
    monkeypatch.setattr(H, "_HISTOGRAM_TASK_ELEMENTS", 10**10)
    da, cycle = _field(freq="MS", periods=240, shape=(4, 5))
    common = {
        "dimensions": GRID_DIMS,
        "cycle": cycle,
        "window_steps": cycle.window_steps(11),
        "window_spatial": 3,
        "precision": 2 * 5.0 / 65535,
        "max_anomaly": 5.0,
        "q": 0.9,
        "tail": "upper",
    }
    da = da.chunk({"time": 30})
    dense = H._compute_histogram_quantile_2d(da, engine="dense", **common)
    slab = H._compute_histogram_quantile_2d(da, engine="slab", **common)
    xr.testing.assert_identical(dense, slab)


def _edges_and_centres(precision=0.05, max_anomaly=5.0):
    edges = H._symmetric_bin_edges(precision, max_anomaly, np.float32)
    return edges, ((edges[1:] + edges[:-1]) / 2).astype(np.float32)


def test_slab_matches_dense_with_samples_clipped_into_the_top_bin():
    # Values beyond max_anomaly are clipped into the top bin; both engines must count them.
    da, cycle = _field()
    da = da.where(~((da["time"].dt.dayofyear % 20 == 0) & (da["lat"] == da["lat"][2])), 7.0)
    dense, slab = _both(da, cycle, GRID_DIMS, 30, q=0.9, tail="upper", window_spatial=3)
    xr.testing.assert_identical(dense, slab)
    _, no_clip = _both(da.where(da < 5.0), cycle, GRID_DIMS, 30, q=0.9, tail="upper", window_spatial=3)
    assert not np.array_equal(slab.values, no_clip.values, equal_nan=True)


def test_one_extra_count_breaks_identity(monkeypatch):
    """Discriminator: the identity tests above can fail. One extra count per cell moves thresholds.

    Pinned to the synchronous scheduler: a session-scoped distributed client (conftest) runs the
    blocks in worker processes, where this process's monkeypatch does not reach.
    """
    monkeypatch.setitem(dask.config.config, "scheduler", "synchronous")
    da, cycle = _field()
    original = H._rolling_histogram_quantile

    def plus_one(hist, *a, **k):
        hist = hist.copy()
        hist[0, hist.shape[1] // 2] += 1
        return original(hist, *a, **k)

    dense, _ = _both(da, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=1)
    monkeypatch.setattr(H, "_rolling_histogram_quantile", plus_one)
    da2 = da.chunk({"time": 30})
    slab = H._compute_histogram_quantile_2d(
        da2,
        engine="slab",
        dimensions=GRID_DIMS,
        cycle=cycle,
        window_steps=11,
        precision=0.05,
        max_anomaly=5.0,
        q=0.95,
        tail="upper",
        window_spatial=1,
    )
    assert not np.array_equal(dense.values, slab.values, equal_nan=True)


def _cells(tile):
    return tile["lat"] * tile["lon"]


@pytest.mark.parametrize("cycle_length", [366, 1464, 8784])
def test_slab_tile_does_not_shrink_with_the_cycle_or_the_bins(cycle_length):
    # Lazy: only the sizes are read (a numpy array of this shape is 30 GB).
    da = xr.DataArray(dask.array.zeros((7305, 720, 1440), chunks=(30, 720, 1440), dtype=np.float32), dims=("time", "lat", "lon"))
    slab = _cells(H._slab_tile_chunks(da, GRID_DIMS, None, cycle_length))
    # Budgeted on max(ntime, cycle): at hourly the cycle (8784) passes ntime (7305) and binds, mildly.
    assert slab >= 0.8 * _cells(H._slab_tile_chunks(da, GRID_DIMS, None, 366))
    dense = _cells(H._histogram_tile_chunks(da, GRID_DIMS, 1000, None, cycle_length))
    assert _cells(H._histogram_tile_chunks(da, GRID_DIMS, 2000, None, cycle_length)) < dense or dense == 1
    assert slab >= 10 * dense


def _crowded_slot_field(n_low=70_000, n_high=20_000):
    """Every step in cycle slot 1, so one (cell, slot, bin) holds ``n_low`` > 65535 samples."""
    t = pd.date_range("2000-01-01", periods=n_low + n_high, freq="h")
    values = np.zeros((t.size, 2, 3), dtype=np.float32)
    values[n_low:] = 1.0
    values[n_low::7, 1, 2] = 2.0  # one cell differs, so cells are not all the same
    cycle = SeasonalCycle("month", 12, 30.0)
    da = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        name="anom",
        coords={"time": t, "lat": [-1.0, 1.0], "lon": [0.0, 1.0, 2.0], cycle.index_name: ("time", np.ones(t.size, dtype=np.int64))},
    )
    return da, cycle


def _int64_reference(da, cycle, q, count_dtype=np.int64):
    """Thresholds from int64 per-cell counts, digitized as both engines do; window_spatial=1."""
    edges, centres = _edges_and_centres()
    bottom, top = H._end_clips(edges)
    idx = np.digitize(np.clip(da.values, bottom, top), edges) - 1
    slots = da[cycle.index_name].values
    out = np.full((da.sizes["lat"], da.sizes["lon"], cycle.length), np.nan, dtype=np.float32)
    max_count = 0
    for j, i in np.ndindex(*out.shape[:2]):
        counts = np.zeros((cycle.length, len(edges) - 1), dtype=np.int64)
        np.add.at(counts, (slots - 1, idx[:, j, i]), 1)
        max_count = max(max_count, int(counts.max()))
        out[j, i] = H._rolling_histogram_quantile(counts.astype(count_dtype), cycle.window_steps(11), q, centres)
    return out, max_count


def test_slab_counts_past_uint16_match_an_int64_reference():
    """A (cell, slot, bin) with more than 65535 samples. The old uint16 flox count wrapped it."""
    da, cycle = _crowded_slot_field()
    reference, max_count = _int64_reference(da, cycle, q=0.9)
    assert max_count > np.iinfo(np.uint16).max  # the premise: the wrap regime is reached
    wrapped, _ = _int64_reference(da, cycle, q=0.9, count_dtype=np.uint16)
    assert not np.array_equal(reference, wrapped, equal_nan=True)  # a wrapping count would fail this test

    dense, slab = _both(da, cycle, GRID_DIMS, 5000, q=0.9, tail="upper", window_spatial=1)
    np.testing.assert_array_equal(slab.transpose("lat", "lon", cycle.index_name).values, reference)
    xr.testing.assert_identical(dense, slab)


def test_dense_window_sum_past_uint16_matches_slab():
    # window_spatial=3 pools up to 9 cells of > 65535 samples each into one count.
    da, cycle = _crowded_slot_field()
    dense, slab = _both(da, cycle, GRID_DIMS, 5000, q=0.9, tail="upper", window_spatial=3)
    xr.testing.assert_identical(dense, slab)


@pytest.mark.parametrize("window_spatial", [1, 5])
def test_slab_matches_dense_when_slots_are_fractional(window_spatial):
    # flox's integer expected groups drop a fractional slot; the slab engine must drop it too.
    da, cycle = _field()
    slots = da[cycle.index_name].values.astype(np.float64)
    slots[::3] += 0.5
    da = da.assign_coords({cycle.index_name: ("time", slots)})
    dense, slab = _both(da, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=window_spatial)
    xr.testing.assert_identical(dense, slab)
