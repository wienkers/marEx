"""1-D and 2-D histogram-quantile numerics, the bounds check, and the 2-D path's windows.

What the gridded goldens do not exercise:

- the 1-D CDF interpolation must land strictly between bin centres (it was dead code collapsing
  to the crossing bin centre), and out-of-range-high values must be counted in the top bin, not
  dropped (which renormalised the CDF over a truncated total and biased thresholds low);
- a cell that is NaN only at some timesteps must still receive a real threshold;
- the bounds check makes one fused round trip, raises when a pinned range saturates, warns when
  a derived one does, and clamps thresholds off the zero guard rail;
- the cycle window sum keeps its integer dtype and the caller's chunking (summing shifted slices
  naively shreds the tiling into width-1 slivers, an all-to-all rechunk that OOM-killed the
  full-scale gridded run while every value test stayed green);
- the spatial window wraps longitude only on a grid that spans the globe;
- a window wider than the cycle pools each slot once.
"""

import logging

import dask
import dask.array as dsa
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

import marEx
import marEx.extremes.histogram as H
from marEx.core.compute_mode import Materialiser
from marEx.exceptions import ConfigurationError

from ..conftest import assert_no_failures
from .conftest import (
    DATA_DIR,
    GLOBAL_LON,
    GRID_DIMS,
    REGIONAL_LON,
    anomaly_fixture,
    check,
    legacy_bin_edges,
    lon_field,
    warnings_as_errors,
    with_dayofyear,
)

PRECISION = 0.01
MAX_ANOMALY = 5.0
REFERENCE_NPY = DATA_DIR / "histogram_quantile_1d_reference.npy"


def _series(values_2d, name="dat_anomaly"):
    """Wrap a (time, x) numpy array as a dask-backed DataArray with a time coord."""
    nt, nx = values_2d.shape
    da = xr.DataArray(values_2d.astype(np.float32), dims=["time", "x"], coords={"time": np.arange(nt)}, name=name)
    return da.chunk({"time": -1, "x": -1})


def _legacy_bin_centers(precision=PRECISION, max_anomaly=MAX_ANOMALY):
    edges = legacy_bin_edges(precision, max_anomaly)
    centers = (edges[1:] + edges[:-1]) / 2
    centers[0] = 0.0
    return centers


# ── 1-D numerics ──────────────────────────────────────────────────────


def test_1d_quantile_edge_interpolation_matches_true_percentile():
    """The within-bin edge interpolation tracks the true percentile to well inside one bin.

    This discriminates the edge interpolation from both the old bin-centre snap and the
    (systematically biased) centre interpolation: edge-based lands within ~precision/3, the others
    do not. At q=0.99 the tail is thin (300 of 30000 samples) and the measured error is 0.0049, so
    that row is held to one bin, the tolerance of the multi-quantile test it replaces.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, size=(30000, 16))  # continuous columns: the true quantile falls between centres
    da = _series(data)
    for q, bound in ((0.6, PRECISION / 3), (0.9, PRECISION / 3), (0.95, PRECISION / 3), (0.99, PRECISION)):
        thr = H._compute_histogram_quantile_1d(da, q, dim="time", precision=PRECISION, max_anomaly=MAX_ANOMALY).compute().values
        assert np.isfinite(thr).all(), q
        max_err = np.max(np.abs(thr - np.percentile(data, 100 * q, axis=0)))
        assert max_err < bound, f"q={q}: edge interpolation not accurate: max_err={max_err:.5f}"
        if q == 0.95:
            # It must genuinely interpolate, not snap to bin centres.
            dist = np.min(np.abs(thr[:, None] - _legacy_bin_centers()[None, :]), axis=1)
            assert np.mean(dist < 1e-9) < 0.5, "thresholds still snap to bin centres"


def test_1d_quantile_counts_out_of_range_high_mass():
    """Counting out-of-range-high mass (the fix) yields a strictly higher threshold than dropping it (the bug).

    A/B on the same data isolates the effect from the interpolation shape. The top 4 % of every
    column is pushed far above max_anomaly: under 1 - q, so the counted 95th percentile stays inside
    the range (a share above 5 % would put it in the clipped top bin, which is an error).
    """
    rng = np.random.default_rng(1)
    nt, nx = 5000, 6
    base = rng.normal(0.0, 1.0, size=(nt, nx)).astype(np.float32)
    outliers = base.copy()
    k = int(0.04 * nt)
    for j in range(nx):
        outliers[np.argsort(base[:, j])[-k:], j] = 20.0

    def quantile(values):
        return H._compute_histogram_quantile_1d(_series(values), 0.95, dim="time", precision=PRECISION, max_anomaly=MAX_ANOMALY)

    thr_counted = quantile(outliers).compute().values
    dropped = outliers.copy()
    dropped[dropped > MAX_ANOMALY] = np.nan  # the old drop-and-renormalise behaviour
    thr_dropped = quantile(dropped).compute().values
    assert np.all(thr_counted > thr_dropped + PRECISION), f"OOR mass not counted: {thr_counted} vs {thr_dropped}"


def test_1d_quantile_partial_nan_cell_gets_finite_threshold():
    """A cell NaN only at t=0 gets a finite threshold, not a permanent NaN; a fully-NaN cell stays masked."""
    rng = np.random.default_rng(2)
    data = rng.normal(0.0, 1.0, size=(3000, 3)).astype(np.float32)
    data[0, 1] = np.nan
    thr = H._compute_histogram_quantile_1d(_series(data), 0.95, dim="time", precision=PRECISION, max_anomaly=MAX_ANOMALY).compute()
    assert np.isfinite(thr.values[1]), "partial-NaN cell got a NaN threshold"
    data2 = data.copy()
    data2[:, 2] = np.nan
    thr2 = H._compute_histogram_quantile_1d(_series(data2), 0.95, dim="time").compute()
    assert np.isnan(thr2.values[2]), "all-NaN cell should be masked"


def test_1d_quantile_reference_oracle(monkeypatch):
    """Bit-identity oracle for the 1-D quantile ARITHMETIC, and the whole effect of the symmetric bins.

    The reference was captured from the two-phase (persist-the-CDF) implementation, with the legacy
    asymmetric bins. Forcing those bins back isolates the arithmetic from the binning: a cumsum that
    drops from float64 to float32 moves these values. No tolerance.

    The symmetric bins accumulate the CDF over twice as many bins, which reorders a float64
    cumulative sum: measured at 28 of 42 cells, max 1.33e-15, against a 0.01 bin. If the 2e-14
    bound grows, something other than summation order moved.
    """
    fixture = anomaly_fixture(dates=False)
    expected = np.load(REFERENCE_NPY)

    got = H._compute_histogram_quantile_1d(fixture, q=0.95, dim="time").compute().values
    np.testing.assert_allclose(got, expected, atol=2e-14, rtol=0, equal_nan=True)

    with monkeypatch.context() as m:
        m.setattr(H, "_symmetric_bin_edges", legacy_bin_edges)
        got = H._compute_histogram_quantile_1d(fixture, q=0.95, dim="time").compute().values
    assert got.dtype == expected.dtype, f"dtype moved {expected.dtype} -> {got.dtype}"
    assert got.shape == expected.shape
    np.testing.assert_array_equal(np.nan_to_num(got, nan=-999.0), np.nan_to_num(expected, nan=-999.0))


def test_1d_quantile_custom_edges_and_tiny_inputs():
    """The only 1-D test of an explicit ``bin_edges``; a constant series and a three-sample series."""

    def single(values):
        n = len(values)
        return xr.DataArray(
            np.asarray(values, dtype=np.float64)[np.newaxis, :, np.newaxis],
            dims=["time", "x", "y"],
            coords={"time": [0], "x": np.arange(n), "y": [0]},
            name="test_data",
        )

    data = np.random.RandomState(42).uniform(-2, 3, 1000)
    edges = np.linspace(-3, 4, 100)
    for q in (0.5, 0.9):
        result = H._compute_histogram_quantile_1d(single(data), q, dim="x", bin_edges=edges)
        assert np.isclose(result.values[0, 0], np.percentile(data, 100 * q), atol=0.05)
        # The empirical CDF at THESE edges, linearly interpolated: the default 0.01 bins land
        # 4e-4 (q=0.5) and 4e-3 (q=0.9) away, so ignoring ``bin_edges`` cannot meet 1e-10.
        cdf = np.array([(data < e).mean() for e in edges])
        k = int(np.argmax(cdf >= q))
        expected = edges[k - 1] + (q - cdf[k - 1]) / (cdf[k] - cdf[k - 1]) * (edges[k] - edges[k - 1])
        assert abs(result.values[0, 0] - expected) < 1e-10, (
            f"q={q}: {result.values[0, 0]!r} is not on the custom edges ({expected!r})"
        )

    result = H._compute_histogram_quantile_1d(single(np.full(100, 2.5)), 0.95, dim="x")
    assert np.isclose(result.values[0, 0], 2.5, atol=0.01)

    result = H._compute_histogram_quantile_1d(single([1.0, 2.0, 3.0]), 0.5, dim="x")
    assert np.isclose(result.values[0, 0], 2.0, atol=0.01)


def test_single_step_window_does_not_duplicate_the_cycle():
    """A one-step window is a no-op smoothing, not a wrap-pad of the whole array.

    ``pad_size = window_steps // 2`` is 0 there, and ``hist[-0:]`` is the WHOLE array, so the naive
    concatenate returns ``2 x n_slots`` rows. The normal case on a monthly axis, where any
    ``window_days`` under ~45 clamps to a single month.
    """
    rng = np.random.default_rng(7)
    hist = rng.integers(0, 50, size=(12, 40)).astype(np.int32)
    centers = np.linspace(-1.0, 5.0, 40)
    out = H._rolling_histogram_quantile(hist, 1, 0.9, centers)
    assert out.shape == (12,), "a one-step window changed the number of cycle slots"
    for slot in range(12):
        np.testing.assert_array_equal(out[slot : slot + 1], H._rolling_histogram_quantile(hist[slot : slot + 1], 1, 0.9, centers))


# ── the bounds check ──────────────────────────────────────────────────


class CountComputes(Callback):
    """Count top-level dask scheduler round trips (compute and persist both fire ``_start``)."""

    def __init__(self):
        self.n = 0

    def _start(self, dsk):
        self.n += 1


def test_bounds_check_round_trips(monkeypatch):
    """The bounds check costs one fused round trip, not one per predicate.

    ``if too_high.any():`` computes, and before the fusion this path made up to four separate
    round trips per threshold, each re-executing the whole upstream graph whenever the anomaly is
    not pinned (the lazy and streaming modes). Pinned to the synchronous scheduler: a distributed
    client never fires ``Callback._start``, so under one every count would read 0 and pass any
    upper bound. Hence the lower bound of 1 too.

    - 1-D persist: measured 5 originally, 4 after the fusion, 2 now (the threshold persist and one
      fused check; the CDF persist and the ``idx_upper`` compute went with the fused apply_ufunc).
    - 1-D lazy: nothing is pinned, so only the fused check remains. A re-introduced CDF persist
      (space-scaled, ~30 GB on the ICON mesh) would show here.
    - 2-D: measured 5 before the fusion, 4 after; it computes its ``nan_mask`` separately.
    """
    monkeypatch.setitem(dask.config.config, "scheduler", "synchronous")
    failures = []
    rows = {
        "1-D persist": (lambda: H._compute_histogram_quantile_1d(anomaly_fixture(), q=0.95, dim="time"), 1, 2),
        "1-D lazy": (
            lambda: H._compute_histogram_quantile_1d(anomaly_fixture(), q=0.95, dim="time", materialiser=Materialiser("lazy")),
            1,
            1,
        ),
        "2-D": (
            lambda: H._compute_histogram_quantile_2d(
                with_dayofyear(anomaly_fixture()), q=0.95, dimensions=GRID_DIMS, window_steps=3, window_spatial=None
            ),
            1,
            4,
        ),
    }
    for case, (call, low, high) in rows.items():

        def row(call=call, low=low, high=high):
            with CountComputes() as cb:
                call()
            assert low <= cb.n <= high, f"expected {low}-{high} scheduler round trips, got {cb.n}"

        check(failures, case, row)
    assert_no_failures(failures)


def test_reports_and_clamps_below_the_lower_bound(marex_records):
    """A constant-zero cell drives its quantile below the guard rail: reported at INFO and clamped onto it."""
    da = anomaly_fixture().load()
    da[:, 0, 0] = 0.0
    records = marex_records(logging.INFO)
    result = H._compute_histogram_quantile_1d(da.chunk({"time": -1, "lat": 3, "lon": 7}), q=0.95, dim="time")
    assert any("wrong side of zero anomaly" in r.getMessage() for r in records)
    assert float(result.isel(lat=0, lon=0).values) == pytest.approx(float(legacy_bin_edges()[3]))


def test_saturated_tail_raises_and_control_stays_silent():
    """A threshold inside the outermost bin of a pinned range is an error on both tails and both drivers.

    Each raising row has a control on the same fixture with the default range, so the check is
    shown to be able to stay silent (N(0, 1) quantiles sit far inside the bins). The 2-D control
    has ~3 samples per day-of-year slot, so the separate, deliberately non-fatal zero-guard
    UserWarning may fire; only the range check must stay silent, which ``isfinite`` shows.
    """
    failures = []
    for tail, q, match in (("upper", 0.95, "exceed expected range"), ("lower", 0.05, "below expected range")):

        def raises_1d(tail=tail, q=q, match=match):
            with pytest.raises(ConfigurationError, match=match):
                H._compute_histogram_quantile_1d(anomaly_fixture(), q=q, dim="time", precision=0.01, max_anomaly=0.5, tail=tail)

        def raises_2d(tail=tail, q=q, match=match):
            with pytest.raises(ConfigurationError, match=match):
                H._compute_histogram_quantile_2d(
                    with_dayofyear(anomaly_fixture()),
                    q=q,
                    window_steps=3,
                    dimensions=GRID_DIMS,
                    precision=0.01,
                    max_anomaly=0.5,
                    tail=tail,
                )

        def silent_1d(tail=tail, q=q):
            with warnings_as_errors():
                result = H._compute_histogram_quantile_1d(anomaly_fixture(), q=q, dim="time", tail=tail)
            assert np.isfinite(result.values).all()

        def silent_2d(tail=tail, q=q):
            result = H._compute_histogram_quantile_2d(
                with_dayofyear(anomaly_fixture()), q=q, window_steps=3, dimensions=GRID_DIMS, tail=tail
            )
            assert np.isfinite(result.values).all()

        for row in (raises_1d, raises_2d, silent_1d, silent_2d):
            check(failures, f"{row.__name__}[{tail}]", row)
    assert_no_failures(failures)


def test_derived_range_warns_where_a_pinned_one_raises():
    """With the range DERIVED, its edge is the data's own extreme and nothing is clipped: reaching
    the outermost bin is a small-sample warning. The same field with the same range PINNED raises.

    One case per driver; the fixtures were found by search (few samples per window put the
    interpolated quantile of the cell holding the global max in the top bin).
    """
    cases = {
        "global": ({"n_time": 8, "seed": 1}, {"method": "global_percentile"}),
        "seasonal": ({"n_time": 40, "seed": 1}, {"method": "seasonal_percentile", "window_days": 3, "window_spatial": 1}),
    }
    failures = []
    for case, (fixture, kwargs) in cases.items():

        def row(fixture=fixture, kwargs=kwargs):
            with pytest.warns(UserWarning, match="Nothing was clipped"):
                ds = marEx.extremes.identify(anomaly_fixture(**fixture), threshold_percentile=95, dimensions=GRID_DIMS, **kwargs)
            assert np.isfinite(ds.thresholds.values).any()
            da = anomaly_fixture(**fixture)
            with pytest.raises(ConfigurationError, match="exceed expected range"):
                marEx.extremes.identify(
                    da, threshold_percentile=95, max_anomaly=float(np.abs(da).max()), dimensions=GRID_DIMS, **kwargs
                )

        check(failures, case, row)
    assert_no_failures(failures)


# ── the cycle and spatial window sums ─────────────────────────────────


def _counts(shape, chunks, seed=0):
    rng = np.random.default_rng(seed)
    return xr.DataArray(dsa.from_array(rng.integers(0, 99, shape).astype(np.uint32), chunks=chunks), dims=("s", "b"))


def _rolling_reference(da, window, periodic):
    if not periodic:
        return da.rolling({"s": window}, center=True, min_periods=1).sum().values
    pad = window // 2
    return (
        da.pad({"s": (pad, pad)}, mode="wrap")
        .rolling({"s": window}, center=True, min_periods=1)
        .sum()
        .isel({"s": slice(pad, pad + da.sizes["s"])})
        .values
    )


@pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "non-periodic"])
def test_shifted_window_sum_matches_the_rolling_sum(periodic):
    """Equal to the (wrap-padded) ``min_periods=1`` rolling sum, in uint32, on the input's own chunks.

    Zero padding is what makes the non-periodic case exact: a full window over a zero-padded array
    is the partial window at the edges. Rows: window {3, 5, 7, 9} x length {13, 40, 97} x chunk
    {5, 7, whole}; production tiles that leave a remainder chunk, lengths {50, 97, 43, 22, 21} at
    chunk 20 (21 leaves a final chunk narrower than the window, where the wrap pad feeds in); and
    a (60, chunk 20) array. Every row asserts values, dtype (bottleneck's rolling promotes uint to
    float64) and ``out.chunks[0] == da.chunks[0]`` (unaligned shifted adds come back as
    (1, 1, 16, 1, 1, ...)).
    """
    rows = [(w, n, (n if c == -1 else c, 4), 0) for w in (3, 5, 7, 9) for n in (13, 40, 97) for c in (5, 7, -1)]
    rows += [(w, n, (20, 3), 3) for w in (3, 5, 7, 9) for n in (50, 97, 43, 22, 21)]
    rows += [(w, 60, (20, 4), 0) for w in (3, 5, 7, 9)]
    failures = []
    for window, length, chunks, seed in rows:

        def row(window=window, length=length, chunks=chunks, seed=seed):
            da = _counts((length, chunks[1]), chunks, seed=seed)
            out = _shifted_window_sum(da, window, periodic)
            assert out.dtype == np.uint32, out.dtype
            assert out.chunks[0] == da.chunks[0], f"tiling changed to {out.chunks[0]}"
            np.testing.assert_array_equal(out.values, _rolling_reference(da, window, periodic).astype(out.dtype))

        check(failures, f"window={window} length={length} chunk={chunks[0]} seed={seed}", row)
    assert_no_failures(failures)


def _shifted_window_sum(da, window, periodic):
    return H._shifted_window_sum(da, "s", window, periodic=periodic)


def test_two_dim_smoothing_does_not_explode_the_graph():
    """Both spatial passes together, on the shape the gridded seasonal path builds.

    The lat pass compounds whatever the lon pass did, so this is where the blow-up showed: 2654
    tasks / 1783 rechunk keys from a 6-chunk input before the fix. The bound is a blow-up
    detector, not a pin.
    """
    da = xr.DataArray(dsa.ones((8, 60, 40, 60), dtype=np.uint32, chunks=(8, 60, 20, 20)), dims=("doy", "bin", "lat", "lon"))
    out = H._shifted_window_sum(H._shifted_window_sum(da, "lon", 5, periodic=True), "lat", 5, periodic=False)
    assert out.chunks[2] == da.chunks[2]
    assert out.chunks[3] == da.chunks[3]
    rechunk_keys = sum(1 for k in out.data.dask if "rechunk" in str(k[0] if isinstance(k, tuple) else k))
    assert rechunk_keys < 400, f"{rechunk_keys} rechunk keys suggests chunk realignment"


def test_numpy_backed_input_still_works():
    """There are no target chunks for a non-dask array; the realignment must simply be skipped."""
    da = xr.DataArray(np.arange(20, dtype=np.uint32).reshape(10, 2), dims=("s", "b"))
    got = H._shifted_window_sum(da, "s", 3, periodic=False).values
    np.testing.assert_array_equal(got, da.rolling({"s": 3}, center=True, min_periods=1).sum().values.astype(got.dtype))


# The slab engine is the production path; the dense engine is the reference and the path even windows take.
ENGINES = [("slab", 5), ("dense", 5), ("dense", 4)]


def test_spatial_window_wraps_only_a_global_grid():
    """Changing ONLY the eastern-most columns leaves the western thresholds alone on a regional box, moves them on a global one.

    Longitude is periodic only when the grid closes on itself; on a regional box the western edge
    has no neighbour beyond it. Rows: both engines (slab and dense at window 5, dense at the even
    window 4) x {regional, global}; each base run is shared by its perturbed run.
    """
    failures = []
    for (engine, window_spatial), (grid, lon) in [
        (e, g) for e in ENGINES for g in (("regional", REGIONAL_LON), ("global", GLOBAL_LON))
    ]:

        def row(engine=engine, window_spatial=window_spatial, grid=grid, lon=lon):
            def thresholds(shift_east):
                da, cycle = lon_field(lon, shift_east=shift_east)
                out = H._compute_histogram_quantile_2d(
                    da,
                    q=0.95,
                    dimensions=GRID_DIMS,
                    cycle=cycle,
                    window_steps=cycle.window_steps(11),
                    window_spatial=window_spatial,
                    precision=0.05,
                    max_anomaly=5.0,
                    engine=engine,
                )
                return out.transpose("lat", "lon", ...).values

            base, moved = thresholds(0.0), thresholds(2.0)
            half = window_spatial // 2
            if grid == "regional":
                # The perturbation is large enough to move the thresholds it reaches...
                assert not np.array_equal(base[:, -half:], moved[:, -half:], equal_nan=True), "perturbation never reached"
                # ...and it does not reach across a regional box.
                np.testing.assert_array_equal(base[:, :half], moved[:, :half])
            else:
                assert not np.array_equal(base[:, :half], moved[:, :half], equal_nan=True), "global west edge did not wrap"

        check(failures, f"{engine}-{window_spatial}-{grid}", row)
    assert_no_failures(failures)


def test_x_is_periodic():
    """Longitude wraps when its columns make a full turn; units are read from the attribute, never guessed."""

    def with_lon(lon, units=None):
        da = xr.DataArray(np.zeros((2, lon.size)), dims=("lat", "lon"), coords={"lon": lon})
        if units:
            da["lon"].attrs["units"] = units
        return da

    bare = xr.DataArray(np.zeros((2, 8)), dims=("lat", "lon"))
    radians = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    rows = {
        "regional": (with_lon(REGIONAL_LON), False),
        "global 45": (with_lon(GLOBAL_LON), True),
        "-180..180": (with_lon(np.arange(-180.0, 180.0, 0.25)), True),
        "descending": (with_lon(np.arange(360.0, 0.0, -1.0)), True),
        "rolled across the seam": (with_lon(np.roll(np.arange(0.0, 360.0, 2.5), 50)), True),
        "repeated seam column": (with_lon(np.arange(0.0, 360.5, 0.5)), True),
        "two steps short": (with_lon(np.arange(0.0, 358.0, 1.0)), False),
        "hemisphere": (with_lon(np.arange(0.0, 180.0, 0.5)), False),
        "6.25 degrees, near 2 pi": (with_lon(10.0 + 0.25 * np.arange(25)), False),
        "radians unlabelled": (with_lon(radians), False),  # read as 0..6.2 degrees
        "radians labelled": (with_lon(radians, units="radians"), True),
        # No coordinate, or a 2-D one, gives no span to measure: keep the periodic behaviour.
        "no coordinate": (bare, True),
        "curvilinear": (bare.assign_coords(lon=(("lat", "lon"), np.tile(REGIONAL_LON, (2, 1)))), True),
    }
    failures = []
    for case, (da, expected) in rows.items():

        def row(da=da, expected=expected):
            assert H._x_is_periodic(da, "lon") is expected

        check(failures, case, row)
    assert_no_failures(failures)


def _uneven_counts(n_slots, n_bins, seed):
    """Per-slot counts whose slots differ in size and in where they sit, so pooling a slot twice moves the quantile."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_slots, n_bins), dtype=np.int64)
    for slot in range(n_slots):
        centre = rng.integers(n_bins // 4, 3 * n_bins // 4)
        samples = np.clip(rng.normal(centre, n_bins / 10, size=rng.integers(20, 200)).round(), 0, n_bins - 1)
        counts[slot] = np.bincount(samples.astype(np.int64), minlength=n_bins)
    return counts


def test_a_window_past_the_cycle_pools_each_slot_once():
    """A window wider than the cycle pools every slot once, so every slot gets the whole cycle's quantile.

    Estimator rows: (n_slots, window) in {(12, 13), (12, 23), (12, 25), (366, 401)} x q {0.9, 0.5}.
    Compiled slab-kernel rows: windows {13, 23, 25} x mirror, against the estimator on the same
    counts, bit for bit, as at every in-range window.
    """
    failures = []
    centres = np.linspace(-5, 5, 200, dtype=np.float32)
    for (n_slots, window), q in [(nw, q) for nw in ((12, 13), (12, 23), (12, 25), (366, 401)) for q in (0.9, 0.5)]:

        def estimator(n_slots=n_slots, window=window, q=q):
            counts = _uneven_counts(n_slots, 200, seed=window)
            got = H._rolling_histogram_quantile(counts, window, q, centres)
            whole_cycle = H._rolling_histogram_quantile(counts.sum(axis=0, keepdims=True), 1, q, centres)
            assert got.shape == (n_slots,)
            np.testing.assert_array_equal(got, np.full(n_slots, whole_cycle[0]))

        check(failures, f"estimator n_slots={n_slots} window={window} q={q}", estimator)

    for window, mirror in [(w, m) for w in (13, 23, 25) for m in (False, True)]:

        def kernel(window=window, mirror=mirror):
            rng = np.random.default_rng(window)
            n_slots, n_bins, ntime, n_cells = 12, 200, 12 * 30, 3
            slots = np.tile(np.arange(1, n_slots + 1, dtype=np.int64), ntime // n_slots)
            block = np.clip(rng.normal(80 + 5 * slots[:, None], 15, size=(ntime, n_cells)).round(), 0, n_bins - 1).astype(np.uint16)
            got, _ = H._slab_quantile_cells(block, slots, n_slots, n_bins, [0], window, 0.9, 0.1, mirror, centres)
            for cell in range(n_cells):
                counts = np.zeros((n_slots, n_bins), dtype=np.int64)
                np.add.at(counts, (slots - 1, block[:, cell].astype(np.int64)), 1)
                if mirror:
                    want = -H._rolling_histogram_quantile(counts[:, ::-1], window, 0.1, centres)
                else:
                    want = H._rolling_histogram_quantile(counts, window, 0.9, centres)
                assert np.unique(want).size == 1
                np.testing.assert_array_equal(got[cell], want)

        check(failures, f"slab kernel window={window} mirror={mirror}", kernel)
    assert_no_failures(failures)
