"""The slab engine equals the dense engine bit for bit, and its tile and graph scale with the mesh, not the record.

``_compute_histogram_quantile_2d`` builds the same integer (cycle x n_bins) counts per cell two
ways: ``engine="dense"`` with flox over a whole spatial tile, ``engine="slab"`` (the default) one
cell at a time from the transposed uint16 bin index. Both feed ``_rolling_histogram_quantile``, so
the thresholds must be IDENTICAL (``xr.testing.assert_identical``): zero tolerance on values, dims, coords
and attrs. ``assert_identical`` does not compare dtype.

The kernel (``_slab_quantile_block``) builds each cell's counts over its occupied bin span only:
bins below a cell's lowest sample and above its highest carry no counts, so it slices them off
(keeping one empty bin below, which the interpolation reads). The kernel tests call it directly on
numpy blocks and compare it with the full-width kernel it replaced, on the raw float32 bits (NaN
sign included).

The slab tile is budgeted on what a task reads, so it does not shrink with the cycle length or
the bin count, where the dense tile does; it is capped at ``_SLAB_TILE_CELLS`` cells, which bounds
a task's wall, and is cut from the input's own spatial chunks so the transpose never makes a tile
straddle two input chunks.
"""

import json
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import cloudpickle
import dask
import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from marEx.core.time_axis import DAILY_CYCLE, SeasonalCycle, resolve_cycle
from marEx.extremes import histogram as H
from marEx.helper.dask_config import DEFAULT_DASK_CONFIG

from ..conftest import assert_no_failures
from .conftest import (
    GLOBAL_LON,
    GRID_DIMS,
    MESH_DIMS,
    REGIONAL_LON,
    both_engines,
    check,
    edges_and_centres,
    lon_field,
    slab_field,
)

# ── the kernel against the kernels it replaced ────────────────────────


def _full_width_block(block, slots, n_slots, n_bins, halo, window_steps, q, q_mirror, mirror, bin_centers):
    """The kernel before trimming: every cell's counts span all ``n_bins``."""
    core_shape = tuple(block.shape[i + 1] - 2 * h for i, h in enumerate(halo))
    out = np.full(core_shape + (n_slots,), np.nan, dtype=np.float32)
    ntime = block.shape[0]
    in_cycle = slots >= 1
    slot_offset = (slots.astype(np.int64) - 1) * n_bins
    for idx in np.ndindex(*core_shape):
        window = block[(slice(None),) + tuple(slice(i, i + 2 * h + 1) for i, h in zip(idx, halo))].reshape(ntime, -1)
        valid = (window < n_bins) & in_cycle[:, None]
        keys = (slot_offset[:, None] + window)[valid]
        counts = np.bincount(keys, minlength=n_slots * n_bins).reshape(n_slots, n_bins)
        if mirror:
            out[idx] = -H._rolling_histogram_quantile(counts[:, ::-1], window_steps, q_mirror, bin_centers)
        else:
            out[idx] = H._rolling_histogram_quantile(counts, window_steps, q, bin_centers)
    return out


def _centres(precision=0.05, max_anomaly=5.0):
    return edges_and_centres(precision, max_anomaly)[1]


def _q_mirror(q):
    return (100.0 - round(q * 100.0, 10)) / 100.0


def _cells(n_bins, ntime, rng):
    """Columns of bin indices, one per cell, covering every edge of the trim; ``n_bins`` is a NaN sample."""
    mid = n_bins // 2
    cols = []
    for _ in range(40):  # narrow spans away from both ends: the ordinary case
        centre, width = rng.integers(10, n_bins - 10), rng.integers(1, 8)
        cols.append(np.clip(np.round(rng.normal(centre, width, ntime)), 1, n_bins - 2))
    cols.append(np.clip(np.round(rng.normal(3, 4, ntime)), 0, n_bins - 1))  # touches bin 0
    cols.append(np.clip(np.round(rng.normal(n_bins - 4, 4, ntime)), 0, n_bins - 1))  # touches the top bin
    cols.append(np.where(rng.random(ntime) < 0.5, 0, n_bins - 1))  # clipped into both end bins
    cols.append(np.full(ntime, mid))  # a single occupied bin
    cols.append(np.full(ntime, 1))  # a single bin next to bin 0
    cols.append(np.full(ntime, n_bins - 1))  # a single bin at the top
    cols.append(np.full(ntime, n_bins))  # no samples: all NaN
    sparse = np.full(ntime, n_bins)
    sparse[rng.choice(ntime, 5, replace=False)] = rng.integers(20, n_bins - 20, 5)
    cols.append(sparse)  # a few samples, most slots empty
    gappy = np.clip(np.round(rng.normal(mid, 3, ntime)), 0, n_bins - 1)
    gappy[: ntime // 2] = n_bins
    cols.append(gappy)  # the first half of the record is NaN
    return np.stack(cols, axis=1).astype(np.uint16)


def _slots(ntime, n_slots, rng, with_dropped=True):
    slots = (np.arange(ntime) % n_slots + 1).astype(np.int64)
    if with_dropped:
        slots[rng.choice(ntime, 7, replace=False)] = 0  # out-of-cycle steps the kernel drops
    return slots


def _assert_bitwise(new, ref):
    assert new.shape == ref.shape and new.dtype == ref.dtype == np.float32
    np.testing.assert_array_equal(new.view(np.uint32), ref.view(np.uint32))


QS = [0.95, 0.9, 0.5, 0.07, 0.32, 0.34, 0.05, 0.0, 1.0, 1.0 - 1e-12]


def test_trimmed_kernel_is_bitwise_the_full_width_kernel():
    """q (the ulp-sensitive percentiles, 0 and 1 included) x mirror x (n_slots, window): one behaviour, pure numpy."""
    failures = []
    for q, mirror, (n_slots, window_steps) in [(q, m, g) for q in QS for m in (False, True) for g in ((366, 11), (12, 1), (12, 3))]:

        def row(q=q, mirror=mirror, n_slots=n_slots, window_steps=window_steps):
            rng = np.random.default_rng(7)
            centres = _centres()
            n_bins = centres.size
            ntime = 3 * n_slots + 5
            block = _cells(n_bins, ntime, rng)
            args = (_slots(ntime, n_slots, rng), n_slots, n_bins, [0], window_steps, q, _q_mirror(q), mirror, centres)
            _assert_bitwise(H._slab_quantile_block(block, *args), _full_width_block(block, *args))

        check(failures, f"q={q} mirror={mirror} n_slots={n_slots} window={window_steps}", row)
    assert_no_failures(failures)


def test_trimmed_kernel_is_bitwise_with_a_halo():
    """A 2-D block with a (1, 2) halo: each core cell pools a 3 x 5 window, whose union sets the span.

    The pad index n_bins (y rows beyond the edge) is a dropped sample, as in the driver.
    """
    failures = []
    for q, mirror in [(q, m) for q in (0.95, 0.07, 1.0) for m in (False, True)]:

        def row(q=q, mirror=mirror):
            rng = np.random.default_rng(11)
            centres = _centres()
            n_bins = centres.size
            n_slots, ntime = 366, 2 * 366 + 3
            block = _cells(n_bins, ntime, rng)[:, : 7 * 7].reshape(ntime, 7, 7)
            block[:, 0, :] = n_bins  # a padded row
            args = (_slots(ntime, n_slots, rng), n_slots, n_bins, [1, 2], 11, q, _q_mirror(q), mirror, centres)
            _assert_bitwise(H._slab_quantile_block(block, *args), _full_width_block(block, *args))

        check(failures, f"q={q} mirror={mirror}", row)
    assert_no_failures(failures)


def test_trimmed_kernel_is_bitwise_on_many_random_cells():
    """Broad random spans and offsets, upper and lower, daily cycle, on a 3000-bin geometry derived from real ICON data."""
    rng = np.random.default_rng(3)
    centres = _centres(0.0131818, 19.7727)
    n_bins = centres.size
    n_slots, ntime = 366, 2 * 365 + 1
    centre = rng.integers(0, n_bins, 300)
    width = rng.choice([0.5, 5, 50, 400], 300)
    block = np.clip(np.round(rng.normal(centre, width, (ntime, 300))), 0, n_bins - 1).astype(np.uint16)
    block[rng.random((ntime, 300)) < 0.1] = n_bins
    slots = _slots(ntime, n_slots, rng)
    for q, mirror in [(0.95, False), (0.9, False), (0.05, True), (0.1, True)]:
        args = (slots, n_slots, n_bins, [0], 11, q, _q_mirror(q), mirror, centres)
        _assert_bitwise(H._slab_quantile_block(block, *args), _full_width_block(block, *args))


def test_kernel_counts_only_the_occupied_span():
    """Structure, not values: a cell on bins 100..104 of 3000 is counted over 6 bins (one empty below), on both mirrors.

    The bitwise tests cannot tell a trimmed kernel from an untrimmed one.
    """
    centres = _centres(0.0131818, 19.7727)
    n_bins = centres.size
    ntime = 400
    block = (100 + np.arange(ntime) % 5).astype(np.uint16)[:, None]
    slots = _slots(ntime, 366, np.random.default_rng(0), False)
    for mirror in (False, True):
        _, widths = H._slab_quantile_cells(block, slots, 366, n_bins, [0], 11, 0.95, 0.05, mirror, centres)
        assert widths.tolist() == [6], f"mirror={mirror}: {widths.tolist()}"


def _peak_bytes(core, halo, n_bins=20, ntime=2):
    block = np.zeros((ntime, core + 2 * halo, core + 2 * halo), dtype=np.uint16)
    centres = np.arange(n_bins, dtype=np.float32)
    slots = np.arange(1, ntime + 1, dtype=np.int64)
    tracemalloc.start()
    H._slab_quantile_cells(block, slots, ntime, n_bins, [halo, halo], 1, 0.95, 0.05, False, centres)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


_PEAKS_IN_A_FRESH_PROCESS = """
import json, sys
sys.path[:0] = sys.argv[1:]  # the trees this process imported, ahead of the child's cwd
from tests.extremes.test_slab_engine import _peak_bytes
_peak_bytes(4, 25)  # compile outside the measurement
print(json.dumps([_peak_bytes(core, 25) for core in (16, 32)]))
"""


def test_pooling_metadata_does_not_scale_with_cells_times_window():
    """The pooled positions are cells + window integers, not a cells x window table.

    At a 51-wide window, quadrupling the core cells quadrupled a per-cell neighbourhood table
    (~21 MB at 32 x 32); here the peak grows only with the block itself.

    ``tracemalloc`` also counts what every other thread allocates meanwhile, and a distributed
    client left running by an earlier test adds ~0.3 MB to some calls, so the peaks are taken in a
    fresh interpreter, which has no other threads.
    """
    trees = [str(Path(H.__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2])]
    run = subprocess.run([sys.executable, "-c", _PEAKS_IN_A_FRESH_PROCESS, *trees], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    small, large = json.loads(run.stdout.strip().splitlines()[-1])
    assert large < 2 * small, f"{small} -> {large} bytes"


def test_kernel_refuses_what_it_would_index_out_of_bounds():
    """The compiled kernel does not bounds-check, so the wrapper refuses these; the NumPy kernel raised on all three."""
    failures = []
    for case in ("slot beyond the cycle", "too few bin centres", "no core cells"):

        def row(case=case):
            block = np.random.default_rng(0).integers(0, 20, (40, 3)).astype(np.uint16)
            slots = (np.arange(40) % 12 + 1).astype(np.int64)
            centres = np.linspace(-1, 1, 20, dtype=np.float32)
            halo = [0]
            if case == "slot beyond the cycle":
                slots[5] = 13
            elif case == "too few bin centres":
                centres = centres[:10]
            else:
                halo = [2]
            with pytest.raises(ValueError):
                H._slab_quantile_block(block, slots, 12, 20, halo, 3, 0.9, 0.1, False, centres)

        check(failures, case, row)
    assert_no_failures(failures)


def _numpy_kernel(block, slots, n_slots, n_bins, halo, window_steps, q, q_mirror, mirror, bin_centers):
    """The per-cell NumPy kernel the compiled one replaced, kept so its speed-up stays measurable.

    It built each cell's trimmed counts with ``np.bincount`` and ran ``_rolling_histogram_quantile``
    on them, about 1.6 ms per cell on a real record, which left most threads of a node idle behind
    the GIL.
    """
    core_shape = tuple(block.shape[i + 1] - 2 * h for i, h in enumerate(halo))
    out = np.full(core_shape + (n_slots,), np.nan, dtype=np.float32)
    ntime = block.shape[0]
    in_cycle = slots >= 1
    slot_index = slots.astype(np.int64) - 1
    q_used = q_mirror if mirror else q
    trim_top = q_used < 1.0 - 1e-6
    for idx in np.ndindex(*core_shape):
        window = block[(slice(None),) + tuple(slice(i, i + 2 * h + 1) for i, h in zip(idx, halo))].reshape(ntime, -1)
        valid = (window < n_bins) & in_cycle[:, None]
        samples = window[valid].astype(np.int64)
        if mirror:
            samples = (n_bins - 1) - samples
        sample_slots = np.broadcast_to(slot_index[:, None], window.shape)[valid]
        lo, hi = (int(samples.min()), int(samples.max())) if samples.size else (0, 0)
        start = lo - 1 if lo > 0 else 0
        stop = hi + 1 if trim_top else n_bins
        width = stop - start
        counts = np.bincount(sample_slots * width + (samples - start), minlength=n_slots * width).reshape(n_slots, width)
        threshold = H._rolling_histogram_quantile(counts, window_steps, q_used, bin_centers[start:stop])
        out[idx] = -threshold if mirror else threshold
    return out


@pytest.mark.slow
def test_kernel_matches_numpy_bitwise():
    """The compiled kernel equals the NumPy kernel it replaced, bit for bit, on an ICON-like block, both tails.

    400 cells x 1096 daily steps over the 3000-bin ICON geometry, ~400 bins occupied per cell. The
    speed ratio (single-threaded, best of 3) is printed, never asserted: a timing assertion on shared
    hardware fails for reasons that are not the code's.
    """
    numba = pytest.importorskip("numba")
    centres = _centres(0.0131818, 19.7727)
    n_bins = centres.size
    rng = np.random.default_rng(0)
    centre = rng.integers(400, n_bins - 400, 400)
    block = np.clip(np.round(rng.normal(centre, 60.0, (1096, 400))), 0, n_bins - 1).astype(np.uint16)
    block[rng.random((1096, 400)) < 0.02] = n_bins  # NaN samples
    slots = (np.arange(1096) % 365 + 1).astype(np.int64)

    def best_of(fn, repeats=3):
        best, result = np.inf, None
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fn()
            best = min(best, time.perf_counter() - t0)
        return best, result

    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        for q, mirror in ((0.95, False), (0.05, True)):
            args = (slots, 366, n_bins, [0], 11, q, _q_mirror(q), mirror, centres)
            H._slab_quantile_block(block[:, :2], *args)  # compile outside the timing
            new_time, new = best_of(lambda args=args: H._slab_quantile_block(block, *args))
            ref_time, ref = best_of(lambda args=args: _numpy_kernel(block, *args))
            _assert_bitwise(new, ref)
            print(f"q={q} mirror={mirror}: kernel {new_time:.3f} s vs NumPy {ref_time:.3f} s ({ref_time / new_time:.2f}x)")
    finally:
        numba.set_num_threads(previous)


# ── the engine against the dense engine ───────────────────────────────


def _assert_engines_identical(da, cycle, dims, tchunk, **kw):
    dense, slab = both_engines(da, cycle, dims, tchunk, **kw)
    xr.testing.assert_identical(dense, slab)
    return dense, slab


def _monthly_mesh(n_years=40, n_cells=2, seed=0):
    """Monthly data whose level depends on the month, so pooling one month twice moves the quantile."""
    t = pd.date_range("1981-01-01", periods=12 * n_years, freq="MS")
    rng = np.random.default_rng(seed)
    level = 0.2 * t.month.values[:, None]
    values = (level + 0.3 * rng.normal(size=(t.size, n_cells))).astype(np.float32)
    cells = np.arange(n_cells, dtype=np.float32)
    da = xr.DataArray(
        values, dims=("time", "ncells"), coords={"time": t, "lon": ("ncells", cells), "lat": ("ncells", cells)}, name="dat_anomaly"
    ).chunk({"time": -1})
    cycle = resolve_cycle(da, "time")
    return da.assign_coords({cycle.index_name: cycle.index_of(da["time"]).compute()}), cycle


def _rows_gridded_daily():
    da, cycle = slab_field()
    rows = {}
    for tchunk in (-1, 30, 7):
        for tail, q in (("upper", 0.95), ("lower", 0.05), ("upper", 0.9)):
            for ws in (None, 1, 3, 5):
                rows[f"tchunk={tchunk} {tail} q={q} ws={ws}"] = lambda tchunk=tchunk, tail=tail, q=q, ws=ws: (
                    _assert_engines_identical(da, cycle, GRID_DIMS, tchunk, q=q, tail=tail, window_spatial=ws)
                )
    return rows


def _rows_mesh():
    da, cycle = slab_field(mesh=True)
    return {
        f"tchunk={tchunk} {tail}": (
            lambda tchunk=tchunk, tail=tail, q=q: _assert_engines_identical(da, cycle, MESH_DIMS, tchunk, q=q, tail=tail)
        )
        for tchunk in (-1, 30)
        for tail, q in (("upper", 0.95), ("lower", 0.05))
    }


def _rows_cadences():
    # 6-hourly: a 1464-slot cycle. Monthly: a one-step window, the kernel's pad_size == 0 branch.
    rows = {}
    for freq, periods in (("6h", 4 * 365 * 3), ("MS", 12 * 20)):
        for tail, q in (("upper", 0.95), ("lower", 0.05)):

            def row(freq=freq, periods=periods, tail=tail, q=q):
                da, cycle = slab_field(freq=freq, periods=periods)
                _assert_engines_identical(da, cycle, GRID_DIMS, 30, q=q, tail=tail, window_spatial=3)

            rows[f"{freq} {tail}"] = row

    def monthly_window_past_the_cycle():
        # A 13-slot window on a 12-month cycle pools each month once: one threshold per cell.
        da, cycle = _monthly_mesh()
        common = {"q": 0.9, "window_steps": 13, "dimensions": MESH_DIMS, "cycle": cycle, "precision": 0.01, "max_anomaly": 5.0}
        dense = H._compute_histogram_quantile_2d(da, engine="dense", **common)
        xr.testing.assert_identical(dense, H._compute_histogram_quantile_2d(da, engine="slab", **common))
        for cell in range(da.sizes["ncells"]):
            assert np.unique(dense.isel(ncells=cell).values).size == 1

    rows["monthly window 13 past the 12-slot cycle"] = monthly_window_past_the_cycle
    return rows


def _rows_edge_inputs(monkeypatch):
    rows = {}

    def legacy_asymmetric_bins():
        # The legacy edges open at -inf (bin centre 0 substituted); upper tail only.
        da, cycle = slab_field()
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
        xr.testing.assert_identical(
            H._compute_histogram_quantile_2d(da.chunk({"time": 30}), engine="dense", **common),
            H._compute_histogram_quantile_2d(da.chunk({"time": 30}), engine="slab", **common),
        )
        # ``bin_edges`` is honoured: explicit 0.25-wide edges reproduce precision=0.25 exactly,
        # and that answer sits up to 0.27 from the default 0.01 bins (> 0.1 asserted), so a
        # driver that ignored the edges cannot pass.
        common.pop("bin_edges")
        coarse = H._compute_histogram_quantile_2d(
            da.chunk({"time": 30}), engine="dense", bin_edges=H._symmetric_bin_edges(0.25, 5.0, np.float32), **common
        )
        xr.testing.assert_identical(
            coarse, H._compute_histogram_quantile_2d(da.chunk({"time": 30}), engine="dense", precision=0.25, **common)
        )
        default = H._compute_histogram_quantile_2d(da.chunk({"time": 30}), engine="dense", **common)
        assert float(abs(coarse - default).max()) > 0.1, "0.25-wide edges gave the default-bin answer"

    def extra_dim():
        da, cycle = slab_field()
        da3 = xr.concat([da, da * 0.5, -da], dim="depth").transpose("time", "depth", "lat", "lon")
        _assert_engines_identical(da3, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=3)

    rows["legacy asymmetric bins"] = legacy_asymmetric_bins
    rows["extra dim"] = extra_dim

    for ws in (1, 5):

        def nat(ws=ws):
            # A NaT step has a NaN cycle slot, which flox's expected groups drop.
            da, _ = slab_field()
            times = da["time"].values.copy()
            times[100] = np.datetime64("NaT")
            da = da.assign_coords(time=times)
            cycle = resolve_cycle(da, "time")
            da = da.assign_coords({cycle.index_name: cycle.index_of(da["time"]).compute()})
            _assert_engines_identical(da, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=ws)

        rows[f"NaT time ws={ws}"] = nat

    def slots_past_the_cycle():
        # Day 366 under an explicit 365-slot cycle is outside the expected groups and dropped.
        da, _ = slab_field()
        cycle = SeasonalCycle("dayofyear", 365, 1.0)
        da = da.assign_coords({cycle.index_name: cycle.index_of(da["time"]).compute()})
        assert int(da[cycle.index_name].max()) == 366
        _assert_engines_identical(da, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=5)

    rows["slots past the cycle"] = slots_past_the_cycle

    for ws in (1, 5):

        def fractional(ws=ws):
            # flox's integer expected groups drop a fractional slot; the slab engine must drop it too.
            da, cycle = slab_field()
            slots = da[cycle.index_name].values.astype(np.float64)
            slots[::3] += 0.5
            da = da.assign_coords({cycle.index_name: ("time", slots)})
            _assert_engines_identical(da, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=ws)

        rows[f"fractional slots ws={ws}"] = fractional

    for ny, ws in ((1, 5), (2, 7), (3, 9), (1, 3), (2, 5)):

        def narrow_y(ny=ny, ws=ws):
            # The dense window sum zero-pads y; overlap refuses a depth beyond the array, so the slab
            # engine clips the y halo to the y size (one chunk, every real row still in each window).
            da, cycle = slab_field(shape=(6, 12))
            _assert_engines_identical(da.isel(lat=slice(2, 2 + ny)), cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=ws)

        rows[f"y={ny} narrower than the halo of ws={ws}"] = narrow_y

    def edge_rounding_realises_65536_bins():
        # n_bins=65535 passes the guard, but float32 edges realise 65536 bins: the y pad index n_bins no longer fits uint16.
        with monkeypatch.context() as m:
            m.setattr(H, "_HISTOGRAM_TASK_ELEMENTS", 10**10)
            da, cycle = slab_field(freq="MS", periods=240, shape=(4, 5))
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
            xr.testing.assert_identical(
                H._compute_histogram_quantile_2d(da, engine="dense", **common),
                H._compute_histogram_quantile_2d(da, engine="slab", **common),
            )

    rows["edge rounding realises 65536 bins"] = edge_rounding_realises_65536_bins
    return rows


def _rows_small_tile_cap(monkeypatch):
    """Caps small enough to cut the fixtures into tiles narrower than the halo, so the overlap has to stitch them."""
    rows = {}
    for ws in (None, 3, 5):
        for tail, q in (("upper", 0.95), ("lower", 0.05)):

            def gridded(ws=ws, tail=tail, q=q):
                # The 6 x 7 fixture is regional (no x wrap), so the edge pad is pinned.
                with monkeypatch.context() as m:
                    m.setattr(H, "_SLAB_TILE_CELLS", 4)
                    da, cycle = slab_field(shape=(6, 7))
                    tiles = H._slab_tile_chunks(da, GRID_DIMS, ws, cycle.length)
                    assert tiles["lat"] < 6 and tiles["lon"] < 7
                    _assert_engines_identical(da, cycle, GRID_DIMS, 30, q=q, tail=tail, window_spatial=ws)

            rows[f"gridded cap 4 ws={ws} {tail}"] = gridded

    for spatial_chunk in (-1, 21, 6):

        def mesh(spatial_chunk=spatial_chunk):
            with monkeypatch.context() as m:
                m.setattr(H, "_SLAB_TILE_CELLS", 8)
                da, cycle = slab_field(mesh=True)
                da = da.chunk({"ncells": spatial_chunk})
                _assert_engines_identical(da, cycle, MESH_DIMS, 30, q=0.95, tail="upper")
                assert H._slab_tile_chunks(da, MESH_DIMS, None, cycle.length)["ncells"] <= 8

        rows[f"mesh cap 8 chunk={spatial_chunk}"] = mesh

    for grid, lon in (("regional", REGIONAL_LON), ("global", GLOBAL_LON)):

        def x_tiles(lon=lon):
            # The wrap (global) and the dropped-index pad (regional) both sit on tile boundaries.
            with monkeypatch.context() as m:
                m.setattr(H, "_SLAB_TILE_CELLS", 4)
                da, cycle = lon_field(lon)
                assert H._slab_tile_chunks(da, GRID_DIMS, 5, cycle.length)["lon"] < lon.size
                common = {
                    "q": 0.95,
                    "dimensions": GRID_DIMS,
                    "cycle": cycle,
                    "window_steps": cycle.window_steps(11),
                    "window_spatial": 5,
                    "precision": 0.05,
                }
                xr.testing.assert_identical(
                    H._compute_histogram_quantile_2d(da, engine="dense", max_anomaly=5.0, **common),
                    H._compute_histogram_quantile_2d(da, engine="slab", max_anomaly=5.0, **common),
                )

        rows[f"{grid} lon cap 4"] = x_tiles
    return rows


TABLES = {
    "gridded-daily": lambda monkeypatch: _rows_gridded_daily(),
    "mesh": lambda monkeypatch: _rows_mesh(),
    "cadences": lambda monkeypatch: _rows_cadences(),
    "edge-inputs": _rows_edge_inputs,
    "small-tile-cap": _rows_small_tile_cap,
}


@pytest.mark.parametrize("table", list(TABLES))
def test_slab_matches_dense(table, monkeypatch):
    """Zero tolerance, dims, coords and attrs (``assert_identical``, which does not compare dtype), on every row.

    - gridded-daily: time chunk {whole, 30, 7} x (tail, q) x ``window_spatial`` {None, 1, 3, 5};
      tchunk 7 with ws 5 puts the halo across a short time chunk.
    - mesh: time chunk {whole, 30} x tail.
    - cadences: 6-hourly (a 1464-slot cycle) and monthly x tail, and a monthly window past the cycle.
    - edge-inputs: legacy bins, an extra dim, NaT steps, slots past the cycle, fractional slots,
      y narrower than the halo, and edge rounding that realises 65536 bins.
    - small-tile-cap: the cell cap cut down so tiles are narrower than the halo.
    """
    failures = []
    for case, row in TABLES[table](monkeypatch).items():
        check(failures, case, row)
    assert_no_failures(failures)


def test_slab_declares_the_blocks_it_returns(monkeypatch):
    """overlap merges a trailing chunk narrower than the halo into its neighbour (13 = 6+6+1 -> 6+7).

    The slab output must declare the blocks the kernel actually returns: a mis-declared chunk passes
    ``.values`` but breaks the NaN-mask ``.where`` and anything else that aligns by chunk.
    """
    da, cycle = slab_field(shape=(13, 13))
    monkeypatch.setattr(H, "_HISTOGRAM_TASK_ELEMENTS", 36 * da.sizes["time"])  # tile side 6, remainder 1 < halo 2
    assert H._slab_tile_chunks(da, GRID_DIMS, 5, cycle.length)["lat"] == 6
    _assert_engines_identical(da, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=5)
    raw = H._slab_seasonal_threshold(
        da.chunk({"time": 30}),
        *edges_and_centres(),
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


def test_slab_matches_dense_with_samples_clipped_into_the_top_bin():
    """Values beyond max_anomaly are clipped into the top bin; both engines count them, which a clip-free field shows."""
    da, cycle = slab_field()
    da = da.where(~((da["time"].dt.dayofyear % 20 == 0) & (da["lat"] == da["lat"][2])), 7.0)
    _, slab = _assert_engines_identical(da, cycle, GRID_DIMS, 30, q=0.9, tail="upper", window_spatial=3)
    _, no_clip = both_engines(da.where(da < 5.0), cycle, GRID_DIMS, 30, q=0.9, tail="upper", window_spatial=3)
    assert not np.array_equal(slab.values, no_clip.values, equal_nan=True)


def test_one_extra_count_breaks_identity(monkeypatch):
    """Discriminator: the identity table can fail. One extra count per cell moves thresholds.

    The extra count is one more sample per cell, in the first slot at the middle bin, handed to the
    slab kernel with its block. Pinned to the synchronous scheduler: a distributed client runs the
    blocks in worker processes, where this process's monkeypatch does not reach.
    """
    monkeypatch.setitem(dask.config.config, "scheduler", "synchronous")
    da, cycle = slab_field()
    original = H._slab_quantile_cells

    def plus_one(block, slots, n_slots, n_bins, *a, **k):
        extra = np.full((1,) + block.shape[1:], n_bins // 2, dtype=block.dtype)
        return original(np.concatenate([block, extra]), np.append(slots, 1), n_slots, n_bins, *a, **k)

    dense, _ = both_engines(da, cycle, GRID_DIMS, 30, q=0.95, tail="upper", window_spatial=1)
    monkeypatch.setattr(H, "_slab_quantile_cells", plus_one)
    slab = H._compute_histogram_quantile_2d(
        da.chunk({"time": 30}),
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
    edges, centres = edges_and_centres()
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


def test_counts_past_uint16_match_an_int64_reference():
    """A (cell, slot, bin) with more than 65535 samples; the old uint16 flox count wrapped it.

    window_spatial=1 against an int64 reference; window_spatial=3 pools up to 9 such cells into one
    dense window-sum count, which must still match the slab engine.
    """
    da, cycle = _crowded_slot_field()
    reference, max_count = _int64_reference(da, cycle, q=0.9)
    assert max_count > np.iinfo(np.uint16).max  # the premise: the wrap regime is reached
    wrapped, _ = _int64_reference(da, cycle, q=0.9, count_dtype=np.uint16)
    assert not np.array_equal(reference, wrapped, equal_nan=True)  # a wrapping count would fail this test

    _, slab = _assert_engines_identical(da, cycle, GRID_DIMS, 5000, q=0.9, tail="upper", window_spatial=1)
    np.testing.assert_array_equal(slab.transpose("lat", "lon", cycle.index_name).values, reference)
    _assert_engines_identical(da, cycle, GRID_DIMS, 5000, q=0.9, tail="upper", window_spatial=3)


# ── tile and graph structure (lazy: only sizes are read) ──────────────


def _lazy(shape, chunks, dims):
    data = dask.array.zeros(shape, chunks=chunks, dtype=np.float32)
    slots = (np.arange(shape[0]) % 365 + 1).astype(np.int64)
    return xr.DataArray(data, dims=dims, coords={"dayofyear": ("time", slots)})


def _threshold(da, dims, window_spatial=None):
    return H._slab_seasonal_threshold(
        da,
        *edges_and_centres(),
        dimensions=dims,
        cycle=DAILY_CYCLE,
        window_steps=11,
        window_spatial=window_spatial,
        q=0.95,
        q_mirror=0.05,
        tail="upper",
    ).data


def _threshold_blocks(da, dims, window_spatial=None):
    return _threshold(da, dims, window_spatial).chunks[:-1]


def _boundaries(chunks):
    return set(np.cumsum(chunks).tolist())


def _area(tile):
    return tile["lat"] * tile["lon"]


def test_slab_tile_rules():
    """The slab tile: capped in cells, linear in the mesh, aligned to the input's chunks, constant in the record and the bins."""
    failures = []
    cap = H._SLAB_TILE_CELLS

    def count_is_linear_in_the_mesh():
        # 0.5 M cells used to be 4 tasks; the count must now be linear in the mesh, not fixed by the budget.
        counts = {}
        for ncells in (1_000_000, 4_000_000):
            (tiles,) = _threshold_blocks(_lazy((365, ncells), (21, 25_000), ("time", "ncells")), MESH_DIMS)
            assert max(tiles) <= cap
            assert len(tiles) == ncells // max(tiles), ncells
            counts[ncells] = len(tiles)
        assert counts[4_000_000] == 4 * counts[1_000_000]

    def tiles_never_straddle_an_input_chunk():
        for in_chunk in (25_000, 1_000, 1_536, 600):
            da = _lazy((365, 300_000), (21, in_chunk), ("time", "ncells"))
            (tiles,) = _threshold_blocks(da, MESH_DIMS)
            inputs, outputs = _boundaries(da.chunks[1]), _boundaries(tiles)
            # Either every input boundary is a tile boundary (tiles split chunks) or the reverse (tiles merge them).
            assert inputs <= outputs or outputs <= inputs, in_chunk
            assert max(tiles) <= cap, in_chunk

    def gridded_tile_is_capped_and_aligned():
        # A spatially whole 0.25 deg globe: 720 x 1440 in one chunk is split into tiles that divide it.
        lat, lon = _threshold_blocks(_lazy((365, 720, 1440), (21, 720, 1440), ("time", "lat", "lon")), GRID_DIMS)
        assert max(lat) * max(lon) <= cap
        assert 720 % max(lat) == 0 and 1440 % max(lon) == 0
        assert len(lat) * len(lon) >= 720 * 1440 // cap

    def extra_dim_does_not_restack_into_a_starved_tile():
        # A small mesh with many levels: the leftover budget stacks levels only up to the cell cap.
        chunks = H._slab_tile_chunks(_lazy((365, 50, 1_000), (21, 1, 1_000), ("time", "depth", "ncells")), MESH_DIMS, None, 366)
        assert chunks["ncells"] * chunks["depth"] <= cap
        assert chunks["ncells"] == 1_000

    def tile_does_not_shrink_with_the_cycle_or_the_bins():
        # Budgeted on max(ntime, cycle): at hourly the cycle (8784) passes ntime (7305) and binds, mildly.
        da = xr.DataArray(
            dask.array.zeros((7305, 720, 1440), chunks=(30, 720, 1440), dtype=np.float32), dims=("time", "lat", "lon")
        )
        for cycle_length in (366, 1464, 8784):
            slab = _area(H._slab_tile_chunks(da, GRID_DIMS, None, cycle_length))
            assert slab >= 0.8 * _area(H._slab_tile_chunks(da, GRID_DIMS, None, 366)), cycle_length
            dense = _area(H._histogram_tile_chunks(da, GRID_DIMS, 1000, None, cycle_length))
            assert _area(H._histogram_tile_chunks(da, GRID_DIMS, 2000, None, cycle_length)) < dense or dense == 1, cycle_length
            assert slab >= 10 * dense, cycle_length

    def tile_is_constant_in_the_series_length():
        # Budgeted on what a task reads and writes, never on n_bins, under the cell cap.
        budget = H._HISTOGRAM_TASK_ELEMENTS

        def cells(ntime):
            da = xr.DataArray(dask.array.zeros((ntime, 10_000), chunks=(ntime, 1_000), dtype=np.float32), dims=("time", "ncells"))
            chunks = H._slab_tile_chunks(da, MESH_DIMS, None, 366)
            assert chunks["time"] == -1
            return chunks["ncells"]

        # Cut from the 1000-cell input chunks, so the cap binds as the largest multiple below it.
        assert cells(1_825) == cells(5_475)
        assert cap // 2 < cells(5_475) <= cap
        huge = budget // 100
        assert cells(huge) < cells(5_475)
        for ntime in (10, 366, 5_475, 100_000, huge, huge * 10):
            assert max(ntime, 366) * cells(ntime) <= budget, f"slab over budget at ntime={ntime}"

    for row in (
        count_is_linear_in_the_mesh,
        tiles_never_straddle_an_input_chunk,
        gridded_tile_is_capped_and_aligned,
        extra_dim_does_not_restack_into_a_starved_tile,
        tile_does_not_shrink_with_the_cycle_or_the_bins,
        tile_is_constant_in_the_series_length,
    ):
        check(failures, row.__name__, row)
    assert_no_failures(failures)


def _mesh_graph(ntime, ncells, chunk_size=DEFAULT_DASK_CONFIG["array.chunk-size"]):
    """The raw threshold graph on a mesh chunked as the ICON examples ship it.

    Returns the total tasks; the tasks outside the layers that run once per source block (the
    rechunk and the kernel); the per-source-block layers marEx adds to the input's own; and the
    pickled size of one kernel task, which carries everything the kernel callable captured.

    Built by default at the ``array.chunk-size`` marEx's cluster helpers ship, not the suite's: the
    rechunk planner stages the transpose through intermediate chunks below some chunk size, so the
    graph a user gets depends on it.
    """
    da = _lazy((ntime, ncells), (21, 25_000), ("time", "ncells"))
    with dask.config.set({"array.chunk-size": chunk_size}):
        threshold = _threshold(da, MESH_DIMS)
    graph = threshold.__dask_graph__()
    n_source = da.data.npartitions
    source_layers = [layer for layer in graph.layers.values() if len(layer) == n_source]
    input_layers = [layer for layer in da.data.__dask_graph__().layers.values() if len(layer) == n_source]
    kernel = graph.layers[threshold.name]
    return {
        "tasks": len(graph),
        "tile_side": len(graph) - sum(len(layer) for layer in source_layers),
        "source_blocks": n_source,
        "added_source_layers": len(source_layers) - len(input_layers),
        "kernel_task_bytes": len(cloudpickle.dumps(kernel[min(kernel.keys())])),
    }


def test_mesh_graph_structure():
    """Four graphs (record 365/730 x mesh 25k/50k), each assertion on every pair it applies to.

    - Twice the mesh is twice the source blocks and twice the tiles, so exactly twice the graph.
      Without the cell cap both meshes become one tile and the graph stops doubling.
    - A longer record adds source blocks, each adding one task per elementwise layer; the transpose
      may be staged, but it must add fewer tile-side tasks than source blocks were added (a split of
      the source blocks, or a rechunk cascade, adds at least one per block).
    - Clip, digitize, the -1 offset and the uint16 cast run once per source block; nothing else may.
    - A kernel task's payload is independent of the mesh (a callable that captured a mesh-sized
      array would double here and still pass the task counts).
    """
    graphs = {(nt, nc): _mesh_graph(nt, nc) for nt in (365, 730) for nc in (25_000, 50_000)}
    for nt in (365, 730):
        small, large = graphs[(nt, 25_000)], graphs[(nt, 50_000)]
        assert large["source_blocks"] == 2 * small["source_blocks"], nt
        assert large["tasks"] == 2 * small["tasks"], nt
        assert large["kernel_task_bytes"] == small["kernel_task_bytes"], nt
        assert small["added_source_layers"] == 4, (nt, small["added_source_layers"])
    for nc in (25_000, 50_000):
        short, long = graphs[(365, nc)], graphs[(730, nc)]
        added_blocks = long["source_blocks"] - short["source_blocks"]
        assert added_blocks > 0
        assert long["tile_side"] - short["tile_side"] < added_blocks, nc


def test_unstaged_transpose_does_not_grow_with_the_record():
    """Where the planner moves each tile in one stage (dask's default chunk size), the transpose and
    the kernel are per tile and must not grow with the record at all, on both mesh sizes."""
    for ncells in (25_000, 50_000):
        short, long = _mesh_graph(365, ncells, "128MiB"), _mesh_graph(730, ncells, "128MiB")
        assert long["source_blocks"] > short["source_blocks"], ncells
        assert long["tile_side"] == short["tile_side"], ncells
