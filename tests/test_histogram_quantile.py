"""Numerics tests for the approximate histogram-quantile kernels.

Targets the §3 findings the gridded golden fixture does *not* exercise:
- §3.1 the 1D CDF interpolation must land strictly between bin centres (was dead code
       collapsing to the crossing bin centre).
- §3.2 out-of-range-high values must be counted in the top bin, not dropped (which
       renormalised the CDF over a truncated total and biased thresholds low).
- §3.3 the spatial-tiling change must be value-neutral (both quantile paths).
- §3.4 a cell that is NaN only at some timesteps must still receive a real threshold.
"""

from pathlib import Path

import numpy as np
import xarray as xr

import marEx.extremes.histogram as H
from marEx.extremes.base import identify_extremes

DIMENSIONS = {"time": "time", "x": "x"}
PRECISION = 0.01
MAX_ANOMALY = 5.0


def _bin_centers(precision=PRECISION, max_anomaly=MAX_ANOMALY):
    edges = np.concatenate([[-np.inf], np.arange(-precision, max_anomaly + precision, precision)])
    centers = (edges[1:] + edges[:-1]) / 2
    centers[0] = 0.0
    return centers


def _series(values_2d, name="dat_anomaly"):
    """Wrap a (time, x) numpy array as a dask-backed DataArray with a time coord."""
    nt, nx = values_2d.shape
    time = np.arange(nt)
    da = xr.DataArray(values_2d.astype(np.float32), dims=["time", "x"], coords={"time": time}, name=name)
    return da.chunk({"time": -1, "x": -1})


# ── §3.1 interpolation is live AND accurate (edge-based) ─────────────────────
def test_1d_quantile_edge_interpolation_matches_true_percentile():
    """§3.1: the fixed within-bin edge interpolation must track the true percentile to well
    inside one bin. This accuracy assertion discriminates the correct edge interpolation
    from both the old bin-centre snap and the (systematically biased) centre interpolation:
    edge-based lands within ~precision/3, the others do not."""
    rng = np.random.default_rng(0)
    # Continuous normal columns: the true 95th percentile falls between bin centres.
    data = rng.normal(0.0, 1.0, size=(30000, 16))
    da = _series(data)

    thr = H._compute_histogram_quantile_1d(da, 0.95, dim="time", precision=PRECISION, max_anomaly=MAX_ANOMALY).compute()
    thr_vals = thr.values
    finite = np.isfinite(thr_vals)
    true_95 = np.percentile(data, 95, axis=0)

    max_err = np.max(np.abs(thr_vals[finite] - true_95[finite]))
    assert max_err < PRECISION / 3, f"edge interpolation not accurate: max_err={max_err:.5f}"

    # It must genuinely interpolate (not snap to bin centres).
    centers = _bin_centers()
    dist = np.min(np.abs(thr_vals[finite][:, None] - centers[None, :]), axis=1)
    assert np.mean(dist < 1e-9) < 0.5, "thresholds still snap to bin centres"


# ── §3.2 out-of-range mass is counted ────────────────────────────────────────
def test_1d_quantile_counts_out_of_range_high_mass():
    """§3.2: counting out-of-range-high mass (the fix) yields a strictly higher threshold
    than dropping it (the bug). A/B on the same data isolates the effect from the
    interpolation shape."""
    rng = np.random.default_rng(1)
    nt, nx = 5000, 6
    base = rng.normal(0.0, 1.0, size=(nt, nx)).astype(np.float32)
    outliers = base.copy()
    # Push the top 8% of every column far above max_anomaly (=5).
    k = int(0.08 * nt)
    for j in range(nx):
        idx = np.argsort(base[:, j])[-k:]
        outliers[idx, j] = 20.0

    # A: outliers present -> the fix clips them into the top bin and counts them.
    thr_counted = (
        H._compute_histogram_quantile_1d(_series(outliers), 0.95, dim="time", precision=PRECISION, max_anomaly=MAX_ANOMALY)
        .compute()
        .values
    )
    # B: the same outliers removed (NaN) -> emulates the old drop-and-renormalise behaviour.
    dropped = outliers.copy()
    for j in range(nx):
        dropped[dropped[:, j] > MAX_ANOMALY, j] = np.nan
    thr_dropped = (
        H._compute_histogram_quantile_1d(_series(dropped), 0.95, dim="time", precision=PRECISION, max_anomaly=MAX_ANOMALY)
        .compute()
        .values
    )

    # Counting the high mass must not bias the threshold low: it should be >= the
    # drop-and-renormalise result, and here strictly higher.
    assert np.all(thr_counted > thr_dropped + PRECISION), f"OOR mass not counted: {thr_counted} vs {thr_dropped}"


# ── §3.4 partial-NaN cell keeps a real threshold ─────────────────────────────
def test_1d_quantile_partial_nan_cell_gets_finite_threshold():
    """§3.4: a cell NaN only at t=0 (finite afterwards) must get a finite threshold, not a
    permanent NaN from the old any-NaN-in-time policy."""
    rng = np.random.default_rng(2)
    data = rng.normal(0.0, 1.0, size=(3000, 3)).astype(np.float32)
    data[0, 1] = np.nan  # column 1: NaN only at t=0
    da = _series(data)

    thr = H._compute_histogram_quantile_1d(da, 0.95, dim="time", precision=PRECISION, max_anomaly=MAX_ANOMALY).compute()
    assert np.isfinite(thr.values[1]), "partial-NaN cell got a NaN threshold (§3.4 not fixed)"
    # A fully-NaN cell must still be masked.
    data2 = data.copy()
    data2[:, 2] = np.nan
    thr2 = H._compute_histogram_quantile_1d(_series(data2), 0.95, dim="time").compute()
    assert np.isnan(thr2.values[2]), "all-NaN cell should be masked"


# ── §3.3 tiling is value-neutral (2D per-doy path with smoothing) ────────────
def _doy_series(nt_years=3):
    """A (time, y, x) dask array with a dayofyear coord for the 2D quantile path."""
    rng = np.random.default_rng(3)
    time = xr.date_range("2001-01-01", periods=365 * nt_years, freq="D", use_cftime=False)
    ny, nx = 6, 8
    vals = rng.normal(0.0, 1.0, size=(time.size, ny, nx)).astype(np.float32)
    da = xr.DataArray(vals, dims=["time", "y", "x"], coords={"time": time}, name="dat_anomaly")
    da = da.assign_coords(dayofyear=da["time"].dt.dayofyear)
    return da.chunk({"time": -1, "y": -1, "x": -1})


def test_2d_quantile_tiling_value_neutral_with_smoothing():
    """§3.3: the 2D per-doy quantile (with spatial smoothing) is identical regardless of the
    spatial tile size chosen internally."""
    da = _doy_series()
    dims = {"time": "time", "x": "x", "y": "y"}

    def run(target):
        orig = H._HISTOGRAM_TASK_ELEMENTS
        try:
            H._HISTOGRAM_TASK_ELEMENTS = target
            return H._compute_histogram_quantile_2d(
                da,
                0.9,
                window_steps=3,
                window_spatial=3,
                dimensions=dims,
                precision=PRECISION,
                max_anomaly=MAX_ANOMALY,
            ).compute()
        finally:
            H._HISTOGRAM_TASK_ELEMENTS = orig

    big = run(50_000_000)  # one tile covering the whole field
    # A budget that tiles into several chunks while staying >= the smoothing window.
    small = run(366 * len(_bin_centers()) * 16)  # ~4x4 tiles over the 6x8 field
    np.testing.assert_array_equal(big.values, small.values, err_msg="2D quantile changed with tiling (§3.3)")


# ── §9.2 the shipped tiling function is exercised multi-tile ─────────────────
def test_1d_quantile_real_chunker_multitile_matches_single():
    """§9.2: run the *real* _chunk_spatial_for_histogram (not a monkeypatched stub) with a
    small element budget so it actually tiles, and confirm it matches the single-tile run."""
    rng = np.random.default_rng(4)
    da = _series(rng.normal(0.0, 1.0, size=(2000, 60)))

    orig = H._HISTOGRAM_TASK_ELEMENTS
    try:
        H._HISTOGRAM_TASK_ELEMENTS = 10**12  # one tile
        big = H._compute_histogram_quantile_1d(da, 0.95, dim="time").compute()
        H._HISTOGRAM_TASK_ELEMENTS = 2000 * 8  # ~8 cells/tile -> many tiles
        small = H._compute_histogram_quantile_1d(da, 0.95, dim="time").compute()
    finally:
        H._HISTOGRAM_TASK_ELEMENTS = orig

    # Verify the small budget really produced multiple spatial tiles.
    tiled = H._chunk_spatial_for_histogram(da, "time", target_elements=2000 * 8)
    assert len(tiled.chunks[tiled.dims.index("x")]) > 1, "real chunker did not multi-tile"
    np.testing.assert_array_equal(big.values, small.values, err_msg="real-chunker tiling changed the quantile")


# ── §3.9 small unstructured grid + exact global must not crash ───────────────
def test_global_percentile_exact_small_unstructured_no_zero_chunk():
    """§3.9: for a small unstructured grid (< ~4445 cells) the exact-path rechunk size must
    be clamped to >= 1 (was 0 -> invalid zero-size chunk)."""
    rng = np.random.default_rng(5)
    ncells = 200  # < 4445, where the old formula rounded to 0
    anom = xr.DataArray(
        rng.normal(0.0, 1.0, size=(400, ncells)).astype(np.float32),
        dims=["time", "cell"],
        coords={"time": np.arange(400), "cell": np.arange(ncells)},
        name="dat_anomaly",
    ).chunk({"time": -1, "cell": -1})

    extremes, thresholds = identify_extremes(
        anom,
        method_extreme="global_percentile",
        method_percentile="exact",
        dimensions={"time": "time", "x": "cell"},
        coordinates={"time": "time", "x": "cell"},
    )
    thr = thresholds.compute()
    assert np.isfinite(thr.values).any()


def _anomaly_fixture_1d(n_time=400, n_y=6, n_x=7, seed=0):
    """A deterministic dask-backed anomaly field for the 1D quantile bit-identity oracle.

    Deliberately spans several spatial chunks with time unchunked, which is the shape the
    1D driver reduces over.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(n_time, n_y, n_x)).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(n_time),
            "lat": np.arange(n_y, dtype=np.float32),
            "lon": np.arange(n_x, dtype=np.float32),
        },
        name="da",
    )
    return da.chunk({"time": -1, "lat": 3, "lon": 7})


_REFERENCE_NPY = Path(__file__).parent / "data" / "histogram_quantile_1d_reference.npy"


def _legacy_bin_edges(precision, max_anomaly, dtype=np.float64):
    """The pre-Phase-D asymmetric edges: one bin for every negative value."""
    if dtype == np.float32:
        return np.concatenate(
            [[-np.inf], np.arange(-precision, max_anomaly + precision, precision, dtype=np.float32)], dtype=np.float32
        )
    return np.concatenate([[-np.inf], np.arange(-precision, max_anomaly + precision, precision)])


def test_1d_quantile_matches_captured_reference_under_the_legacy_bins(monkeypatch):
    """Bit-identity oracle for the 1D quantile ARITHMETIC. No tolerance.

    The reference was captured from the two-phase (persist-the-CDF) implementation before
    that path was restructured, and with the pre-Phase-D asymmetric bins. Forcing those
    bins back in isolates the arithmetic from the binning: any change to the quantile
    computation -- notably a cumsum that drops from float64 to float32 -- moves these
    values, and this is the gate that catches it.
    """
    monkeypatch.setattr(H, "_symmetric_bin_edges", _legacy_bin_edges)
    got = H._compute_histogram_quantile_1d(_anomaly_fixture_1d(), q=0.95, dim="time").compute().values
    expected = np.load(_REFERENCE_NPY)
    assert got.dtype == expected.dtype, f"dtype moved {expected.dtype} -> {got.dtype}"
    assert got.shape == expected.shape
    sentinel = -999.0
    np.testing.assert_array_equal(
        np.nan_to_num(got, nan=sentinel),
        np.nan_to_num(expected, nan=sentinel),
    )


def test_1d_quantile_symmetric_bins_move_the_reference_only_by_round_off():
    """Phase D's symmetric bins accumulate the CDF over twice as many bins.

    That reorders a float64 cumulative sum, so the same quantiles come back with
    last-bit differences and nothing more: MEASURED at 28 of 42 cells, max 1.33e-15,
    against a method whose own bin precision is 0.01. The bound here is the entire
    observable effect of the bin change on this path -- if it grows, something other
    than summation order moved.
    """
    got = H._compute_histogram_quantile_1d(_anomaly_fixture_1d(), q=0.95, dim="time").compute().values
    expected = np.load(_REFERENCE_NPY)
    np.testing.assert_allclose(got, expected, atol=2e-14, rtol=0, equal_nan=True)


def test_spatial_tile_budget_bounds_the_output_not_just_the_input():
    """The tile must be budgeted against the array produced, not only the one consumed.

    A task reads ``ntime x tile_area`` and writes ``output_elements_per_cell x tile_area``.
    Sizing on ``ntime`` alone bounds only the read, and since ``tile_area = target //
    ntime`` the write is then ``target * output_per_cell / ntime`` -- over budget exactly
    when the series is shorter than the per-cell output, and growing as it gets shorter.
    """
    target = 1_000_000
    ntime = 100
    output_per_cell = 500  # e.g. n_bins; deliberately > ntime

    da = xr.DataArray(
        np.zeros((ntime, 200, 200), dtype=np.float32),
        dims=("time", "lat", "lon"),
    ).chunk({"time": -1, "lat": -1, "lon": -1})

    tiled = H._chunk_spatial_for_histogram(da, "time", target_elements=target, output_elements_per_cell=output_per_cell)
    cells = tiled.chunks[1][0] * tiled.chunks[2][0]

    # Both sides of the reduction stay within the budget. The tolerance absorbs the
    # integer rounding of the tile side (round(sqrt(area))**2 can exceed area slightly);
    # it is a few percent, not the 5x the input-only budget gives here.
    tol = 1.1
    assert ntime * cells <= target * tol, "input slab over budget"
    assert output_per_cell * cells <= target * tol, "produced array over budget"

    # And the old input-only budget genuinely violated it, so this test can fail.
    input_only = H._chunk_spatial_for_histogram(da, "time", target_elements=target)
    old_cells = input_only.chunks[1][0] * input_only.chunks[2][0]
    assert output_per_cell * old_cells > target * tol


def test_spatial_tile_budget_unchanged_for_long_series():
    """ntime >> output_per_cell is the normal case: the tiling must not move."""
    da = xr.DataArray(
        np.zeros((5000, 180, 360), dtype=np.float32),
        dims=("time", "lat", "lon"),
    ).chunk({"time": -1, "lat": -1, "lon": -1})

    before = H._chunk_spatial_for_histogram(da, "time").chunks
    after = H._chunk_spatial_for_histogram(da, "time", output_elements_per_cell=502).chunks
    assert before == after


def test_output_budget_does_not_change_quantile_values():
    """The budget change is a pure rechunk, so thresholds must be bit-identical."""
    rng = np.random.default_rng(20260807)
    ntime = 120  # short on purpose: this is where the two budgets diverge
    da = xr.DataArray(
        rng.normal(size=(ntime, 240)).astype(np.float32),
        dims=("time", "x"),
        coords={"time": xr.date_range("2000-01-01", periods=ntime, freq="D")},
        name="dat",
    ).chunk({"time": -1, "x": 60})

    orig = H._HISTOGRAM_TASK_ELEMENTS
    try:
        H._HISTOGRAM_TASK_ELEMENTS = 10**12  # one tile: the unchunked reference
        reference = H._compute_histogram_quantile_1d(da, 0.95, dim="time").compute()
        H._HISTOGRAM_TASK_ELEMENTS = ntime * 40  # small budget -> the output bound binds
        tiled = H._compute_histogram_quantile_1d(da, 0.95, dim="time").compute()
    finally:
        H._HISTOGRAM_TASK_ELEMENTS = orig

    np.testing.assert_array_equal(tiled.values, reference.values)


def test_2d_tile_budget_is_constant_in_series_length():
    """The 2-D path holds time whole, so its tile must also be budgeted against ntime.

    Sizing only on the ``366 x n_bins`` output leaves the slab each task READS growing
    linearly with the length of the series -- a per-task working set that scales with input
    size. Budgeting on ``max(ntime, 366 * n_bins)`` makes it constant in both directions.
    """
    n_bins = 502
    out_per_cell = 366 * n_bins

    def cells(ntime):
        return max(1, H._HISTOGRAM_TASK_ELEMENTS // max(ntime, out_per_cell))

    # Realistic series: the output term dominates, so tiling is unchanged.
    assert cells(1_825) == cells(5_475) == cells(20_000)
    assert cells(5_475) == H._HISTOGRAM_TASK_ELEMENTS // out_per_cell

    # Absurdly long series: the tile shrinks so the read slab stays inside the budget.
    huge = out_per_cell * 4
    assert cells(huge) < cells(5_475)
    assert huge * cells(huge) <= H._HISTOGRAM_TASK_ELEMENTS

    # The bound holds across the whole range: neither side ever exceeds the budget.
    for ntime in (10, 366, 5_475, 100_000, out_per_cell, out_per_cell * 10):
        c = cells(ntime)
        assert ntime * c <= H._HISTOGRAM_TASK_ELEMENTS, f"read slab over budget at ntime={ntime}"
        assert out_per_cell * c <= H._HISTOGRAM_TASK_ELEMENTS, f"histogram over budget at ntime={ntime}"


def test_single_step_window_does_not_duplicate_the_cycle():
    """A one-step window must be a no-op smoothing, not a wrap-pad of the whole array.

    ``pad_size = window_steps // 2`` is 0 there, and ``hist[-0:]`` is ``hist[0:]`` -- the
    WHOLE array, not an empty one -- so the naive concatenate returns ``2 x n_slots``
    rows and the downstream fancy-indexing raises. Unreachable on daily data (the
    smallest legal ``window_days`` is 1 day and any larger odd window pads by at least
    1), but it is the normal case on a monthly axis, where any ``window_days`` under
    ~45 clamps to a single month.
    """
    rng = np.random.default_rng(7)
    hist = rng.integers(0, 50, size=(12, 40)).astype(np.int32)
    centers = np.linspace(-1.0, 5.0, 40)

    out = H._rolling_histogram_quantile(hist, 1, 0.9, centers)
    assert out.shape == (12,), "a one-step window changed the number of cycle slots"

    # With no smoothing, each slot's threshold depends on that slot's counts alone.
    for slot in range(12):
        one_row = H._rolling_histogram_quantile(hist[slot : slot + 1], 1, 0.9, centers)
        np.testing.assert_array_equal(out[slot : slot + 1], one_row)
