"""Numerics tests for the approximate histogram-quantile kernels.

Targets the §3 findings the gridded golden fixture does *not* exercise:
- §3.1 the 1D CDF interpolation must land strictly between bin centres (was dead code
       collapsing to the crossing bin centre).
- §3.2 out-of-range-high values must be counted in the top bin, not dropped (which
       renormalised the CDF over a truncated total and biased thresholds low).
- §3.3 the spatial-tiling change must be value-neutral (both quantile paths).
- §3.4 a cell that is NaN only at some timesteps must still receive a real threshold.
"""

import numpy as np
import xarray as xr

import marEx.detect.extremes.histogram as H
from marEx.detect.extremes.base import identify_extremes

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
                window_days_hobday=3,
                window_spatial_hobday=3,
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
def test_global_extreme_exact_small_unstructured_no_zero_chunk():
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
        method_extreme="global_extreme",
        method_percentile="exact",
        dimensions={"time": "time", "x": "cell"},
        coordinates={"time": "time", "x": "cell"},
    )
    thr = thresholds.compute()
    assert np.isfinite(thr.values).any()
