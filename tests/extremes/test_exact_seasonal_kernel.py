"""Bitwise controls for the exact seasonal rank kernel and its block dispatch."""

import json
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest
import xarray as xr

from marEx.core.compute_mode import Materialiser
from marEx.core.time_axis import resolve_cycle
from marEx.extremes import _exact_seasonal as exact
from marEx.extremes.seasonal_percentile import _identify_extremes_seasonal

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _mask_reference(values, slots, cycle, window, percentile):
    result = np.full(values.shape[:-1] + (cycle,), np.nan, dtype=np.float32)
    half = window // 2
    for slot in range(cycle):
        targets = (slot + np.arange(-half, half + 1)) % cycle + 1
        take = np.isin(slots, targets)
        if take.any():
            result[..., slot] = np.nanpercentile(values[..., take], percentile, axis=-1)
    return result


def _assert_bits(actual, expected):
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape
    missing = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), missing)
    np.testing.assert_array_equal(actual[~missing].view(np.uint32), expected[~missing].view(np.uint32))


def _compare(values, slots, cycle, window, percentile):
    actual = exact._doy_percentiles(values, slots, cycle, window, percentile, _backend="compiled")
    reference = exact._doy_percentiles(values, slots, cycle, window, percentile, _backend="numpy")
    _assert_bits(actual, reference)
    _assert_bits(reference, _mask_reference(values, slots, cycle, window, percentile))
    return actual


@pytest.fixture(scope="module", params=[np.float32, np.float64])
def mixed(request):
    rng = np.random.default_rng(82617)
    slots = rng.integers(-2, 14, size=139, dtype=np.int32)
    values = rng.standard_normal((7, slots.size)).astype(request.param)
    values[0] = np.nan
    values[1, ::5] = np.nan
    values[2, ::4] = 0.0
    values[3, ::13] = -0.0
    values[4, ::21] = np.inf
    values[5, ::11] = -np.inf
    values[6, ::2] = 0.625
    return values, slots


@pytest.mark.parametrize("window", [1, 3, 11, 13, 25])
@pytest.mark.parametrize("q", [0.0, 0.7, 5.2, 50.0, 73.5, 94.4, 95.0, 99.9, 100.0])
def test_calendar_masks_and_ties(mixed, window, q):
    values, slots = mixed
    _compare(values, slots, 12, window, values.dtype.type(q))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("window,q", [(1, 0), (5, 5), (7, 73.5), (13, 100)])
def test_rank_controls(dtype, window, q):
    values = np.random.default_rng(194900).normal(size=(7, 75)).astype(dtype)
    values[1] = np.nan
    values[2, 1::3] = 0.0
    values[3, ::4] = -0.0
    values[4, :10] = np.inf
    values[5, 2::11] = -np.inf
    slots = (np.arange(75, dtype=np.int32) % 11) + 1
    slots[::4] = 0
    _compare(values, slots, 12, window, dtype(q))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fractional_rank(dtype):
    values = np.array([0.0, 1.0], dtype=dtype)
    _compare(values, np.ones(2, dtype=np.int32), 12, 3, dtype(5.2))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("window", [1, 3, 13])
@pytest.mark.parametrize("q", [0.0, 5.0, 95.0, 100.0])
def test_finite_range_overflow(dtype, window, q):
    limit = np.finfo(dtype).max
    values = np.full((2, 21), -limit, dtype=dtype)
    values[:, -1] = limit
    values[1] *= -1
    _compare(values, np.ones(21, dtype=np.int32), 12, window, dtype(q))


def test_calendar_rotation(mixed):
    values, slots = mixed
    rotated = np.where((slots >= 1) & (slots <= 12), (slots - 1 + 5) % 12 + 1, slots)
    q = values.dtype.type(5.2)
    original = _compare(values, slots, 12, 5, q)
    shifted = _compare(values[..., ::-1], rotated[::-1], 12, 5, q)
    _assert_bits(shifted, np.roll(original, 5, axis=-1))


@pytest.mark.parametrize("q", [np.float32(5.2), np.float64(5.2), np.int32(5)])
def test_scalar_percentile_promotion(q):
    _compare(np.array([[0.0, 1.0, np.nan]], dtype=np.float32), np.ones(3, dtype=np.int32), 12, 3, q)


@pytest.mark.parametrize(
    "dtype", [np.float16, np.int16, np.dtype(np.float32).newbyteorder("S"), np.dtype(np.float64).newbyteorder("S")]
)
def test_reference_fallback_dtypes(dtype):
    _compare(np.array([[0, 1, 2]], dtype=dtype), np.ones(3, dtype=np.int32), 12, 3, 5.2)


@pytest.mark.parametrize(
    "dtype,n_time,cycle,window,q",
    [
        (np.float32, 5479, 366, 11, 95),
        (np.float32, 5479, 366, 41, 5),
        (np.float32, 4384, 1464, 45, 95),
        (np.float64, 144, 12, 3, 5),
    ],
)
@pytest.mark.parametrize("fractional", [False, True])
def test_daily_subdaily_and_monthly_tiles(dtype, n_time, cycle, window, q, fractional):
    rng = np.random.default_rng(9433)
    values = rng.standard_normal((n_time, 2, 2)).astype(dtype).transpose(2, 1, 0)
    values[0, 0, ::23] = np.nan
    slots = np.arange(n_time, dtype=np.int32) % cycle + 1
    slots[::197] = 0
    _compare(values, slots, cycle, window, dtype(5.2 if fractional else q))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("q", [0.0, 5.2, 50.0, 94.4, 100.0])
def test_subnormals_singletons_and_invalid_slots(dtype, q):
    subnormal = np.nextafter(dtype(0), dtype(1))
    values = np.array([[subnormal, -subnormal, 0.0, np.inf, 1.0], [np.nan, 2.0, np.nan, 4.0, 5.0]], dtype=dtype)
    slots = np.array([1, 1, 3, 2**32 + 1, -1], dtype=np.int64)
    _compare(values, slots, 12, 1, dtype(q))


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (5, 5)])
def test_empty_samples(shape):
    _compare(np.empty(shape, dtype=np.float32), np.zeros(shape[-1], dtype=np.int32), 12, 3, np.float32(5.2))


@pytest.mark.parametrize("all_exceptional", [False, True])
def test_only_exceptional_rows_use_numpy(monkeypatch, all_exceptional):
    values = np.arange(30, dtype=np.float32).reshape(5, 6)
    values[1, 1] = -0.0
    values[3, 4] = np.inf
    if all_exceptional:
        values[:, 0] = -np.inf
    slots = np.arange(1, 7, dtype=np.int32)
    seen = []
    numpy_backend = exact._numpy_doy_percentiles

    def record(rows, *args):
        seen.extend(rows[:, -1])
        return numpy_backend(rows, *args)

    monkeypatch.setattr(exact, "_numpy_doy_percentiles", record)
    result = exact._doy_percentiles(values, slots, 6, 3, np.float32(5.2), _backend="compiled")
    _assert_bits(result, _mask_reference(values, slots, 6, 3, np.float32(5.2)))
    np.testing.assert_array_equal(seen, values[:, -1] if all_exceptional else values[[1, 3], -1])


def test_auto_workload_crossover(monkeypatch):
    monkeypatch.setattr(exact, "_MIN_COMPILED_WORK", 12)
    calls = []
    kernel = exact._rolling_rank_percentiles

    def record(*args):
        calls.append(True)
        return kernel(*args)

    monkeypatch.setattr(exact, "_rolling_rank_percentiles", record)
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    slots = np.arange(1, 5, dtype=np.int32)
    for rows in (2, 3):
        actual = exact._doy_percentiles(values[:rows], slots, 4, 3, np.float32(5.2))
        _assert_bits(actual, _mask_reference(values[:rows], slots, 4, 3, np.float32(5.2)))
        assert len(calls) == rows - 2
    exact._doy_percentiles(values, slots, 4, 11, np.float32(5.2))
    assert len(calls) == 1
    exact._doy_percentiles(values, np.ones_like(slots), 60, 1, np.float32(5.2))
    assert len(calls) == 1


def test_cold_first_call():
    script = """
import json
import runpy
import sys
from time import perf_counter

import numpy as np

namespace = runpy.run_path(sys.argv[1])
percentiles = namespace['_doy_percentiles']
kernel = namespace['_rolling_rank_percentiles']
slots = np.arange(1, 9, dtype=np.int32)
values = np.arange(16, dtype=np.float32).reshape(2, -1)
percentiles(values, slots, 8, 3, np.float32(5.2))
assert not kernel.signatures
start = perf_counter()
percentiles(values, slots, 8, 3, np.float32(5.2), _backend='compiled')
elapsed = perf_counter() - start
percentiles(values.astype(np.float64), slots, 8, 3, np.float64(5.2), _backend='compiled')
print(json.dumps([elapsed, len(kernel.signatures)]))
"""
    cold = subprocess.run([sys.executable, "-c", script, exact.__file__], check=True, capture_output=True, text=True, timeout=30)
    elapsed, signatures = json.loads(cold.stdout)
    assert signatures == 1
    assert elapsed <= 1.5, f"cold compile and first call took {elapsed:.3f} s"


def test_strided_tile_uses_bounded_strips():
    n_time, n_slots, n_depth, n_columns = 1024, 8, 192, 256
    plane = np.ones((n_time, n_depth + n_columns - 1), dtype=np.float32)
    plane[0, -1] = -0.0  # Only the last row of the last level sees this element.
    time_stride, row_stride = plane.strides
    values = np.lib.stride_tricks.as_strided(
        plane, shape=(n_depth, n_columns, n_time), strides=(row_stride, row_stride, time_stride), writeable=False
    )
    assert values.size > 50_000_000 and not values.flags.c_contiguous
    slots = np.arange(n_time, dtype=np.int32) % n_slots + 1
    q = np.float32(94.4)
    exact._doy_percentiles(values[:1, :1], slots, n_slots, 3, q, _backend="compiled")
    tracemalloc.start()
    try:
        result = exact._doy_percentiles(values, slots, n_slots, 3, q, _backend="compiled")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak - result.nbytes < 8 * 2**20
    np.testing.assert_array_equal(result.reshape(-1, n_slots)[:-1], 1.0)
    reference = _mask_reference(values[-1, -1], slots, n_slots, 3, q)
    _assert_bits(result[-1, -1], reference)


@pytest.mark.parametrize("tail", ["upper", "lower"])
def test_exact_integration_preserves_chunks_clamp_and_flags(monkeypatch, tail):
    dims = {"time": "time", "x": "lon", "y": "lat"}
    time = xr.date_range("2000-01-01", periods=36, freq="MS")
    values = np.random.default_rng(41).normal(size=(36, 2, 3, 4)).astype(np.float32)
    values[:, 0, 0, 0] = 0
    values[:, 1, 0, 0] = np.nan
    data = xr.DataArray(values, dims=("time", "depth", "lat", "lon"), coords={"time": time}).chunk((9, 1, 2, 2))
    cycle = resolve_cycle(data, "time")
    options = dict(
        threshold_percentile=np.float32(5.2 if tail == "lower" else 94.4),
        method_percentile="exact",
        dimensions=dims,
        coordinates=dims,
        materialiser=Materialiser("lazy"),
        cycle=cycle,
        tail=tail,
    )
    monkeypatch.setattr(exact, "_MIN_COMPILED_WORK", sys.maxsize)
    ref_flags, ref_thresholds = _identify_extremes_seasonal(data, **options)
    flag_chunks, threshold_chunks = ref_flags.chunksizes, ref_thresholds.chunksizes
    ref_flags, ref_thresholds = ref_flags.compute(), ref_thresholds.compute()
    monkeypatch.setattr(exact, "_MIN_COMPILED_WORK", 0)
    flags, thresholds = _identify_extremes_seasonal(data, **options)
    assert flags.chunksizes == flag_chunks
    assert thresholds.chunksizes == threshold_chunks
    thresholds = thresholds.compute()
    _assert_bits(thresholds.values, ref_thresholds.values)
    xr.testing.assert_identical(flags.compute(), ref_flags)
    tiny = np.finfo(np.float32).tiny
    np.testing.assert_array_equal(thresholds.isel(depth=0, lat=0, lon=0), tiny if tail == "upper" else -tiny)
    assert thresholds.isel(depth=1, lat=0, lon=0).isnull().all()
