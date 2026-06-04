"""Tests for the integer-preserving centred window sum used by the hobday histogram.

`_shifted_window_sum` (review finding 3.6) replaces `.rolling().sum()` on the bin-resolved
histogram so the counts never leave their integer dtype. Two properties matter and only the
first was ever checked:

1. **Values** -- it must equal `.rolling({dim: window}, center=True, min_periods=1).sum()`
   exactly for odd windows, with a wrap-pad first in the periodic case.
2. **Chunk structure** -- it must not disturb the caller's spatial tiling. Summing shifted
   slices naively makes dask `unify_chunks` to the common refinement of every shifted
   boundary set, shredding the tiles into width-1 slivers and turning the smoothing into an
   all-to-all rechunk. That is invisible to any value-based test and it OOM-killed the
   full-scale gridded hobday run (sbatch #3, job 26614089) while leaving every unit test,
   golden and coverage tripwire green.
"""

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

from marEx.detect.extremes.histogram import _shifted_window_sum

ODD_WINDOWS = [3, 5, 7, 9]


def _counts(shape, chunks, seed=0):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        dsa.from_array(rng.integers(0, 99, shape).astype(np.uint32), chunks=chunks),
        dims=("s", "b"),
    )


@pytest.mark.parametrize("window", ODD_WINDOWS)
@pytest.mark.parametrize("length", [13, 40, 97])
@pytest.mark.parametrize("chunk", [5, 7, -1])
def test_matches_rolling_sum_non_periodic(window, length, chunk):
    """Zero-padded full window == min_periods=1 partial window at the edges."""
    da = _counts((length, 4), (length if chunk == -1 else chunk, 4))
    got = _shifted_window_sum(da, "s", window, periodic=False).values
    expected = da.rolling({"s": window}, center=True, min_periods=1).sum().values
    np.testing.assert_array_equal(got, expected.astype(got.dtype))


@pytest.mark.parametrize("window", ODD_WINDOWS)
@pytest.mark.parametrize("length", [13, 40, 97])
@pytest.mark.parametrize("chunk", [5, 7, -1])
def test_matches_rolling_sum_periodic(window, length, chunk):
    """Wrap-padded full window == the periodic rolling sum."""
    da = _counts((length, 4), (length if chunk == -1 else chunk, 4))
    pad = window // 2
    got = _shifted_window_sum(da, "s", window, periodic=True).values
    expected = (
        da.pad({"s": (pad, pad)}, mode="wrap")
        .rolling({"s": window}, center=True, min_periods=1)
        .sum()
        .isel({"s": slice(pad, pad + length)})
        .values
    )
    np.testing.assert_array_equal(got, expected.astype(got.dtype))


@pytest.mark.parametrize("window", ODD_WINDOWS)
@pytest.mark.parametrize("periodic", [True, False])
def test_preserves_integer_dtype(window, periodic):
    """The whole point of 3.6: bottleneck's rolling promotes uint to float64."""
    da = _counts((40, 4), (7, 4))
    assert _shifted_window_sum(da, "s", window, periodic=periodic).dtype == np.uint32


@pytest.mark.parametrize("window", ODD_WINDOWS)
@pytest.mark.parametrize("periodic", [True, False])
def test_preserves_input_chunking(window, periodic):
    """Regression guard: the tiling must survive the shifted adds.

    Without putting each shifted slice back on the input's boundaries, dask unifies to the
    common refinement and the chunks come back as (1, 1, 16, 1, 1, ...).
    """
    da = _counts((60, 4), (20, 4))
    out = _shifted_window_sum(da, "s", window, periodic=periodic)
    assert out.chunks[0] == da.chunks[0], f"tiling shredded to {out.chunks[0]}"


@pytest.mark.parametrize("window", ODD_WINDOWS)
@pytest.mark.parametrize("length", [50, 97, 43, 22, 21])
@pytest.mark.parametrize("periodic", [True, False])
def test_uneven_final_chunk(window, length, periodic):
    """Production tiles do not divide the grid evenly, so the last chunk is a remainder.

    `chunk_dict[d] = min(da.sizes[d], tile_side)` leaves e.g. (20, 20, 10) for a length-50
    dim, and the remainder can be *narrower than the window* (length 21 -> (20, 1)). The
    periodic case is the sharp one: the wrap pad feeds into that short chunk.
    """
    da = _counts((length, 3), (20, 3), seed=3)
    out = _shifted_window_sum(da, "s", window, periodic=periodic)

    assert out.chunks[0] == da.chunks[0], f"tiling changed to {out.chunks[0]}"
    if periodic:
        pad = window // 2
        expected = (
            da.pad({"s": (pad, pad)}, mode="wrap")
            .rolling({"s": window}, center=True, min_periods=1)
            .sum()
            .isel({"s": slice(pad, pad + length)})
            .values
        )
    else:
        expected = da.rolling({"s": window}, center=True, min_periods=1).sum().values
    np.testing.assert_array_equal(out.values, expected.astype(out.dtype))


def test_two_dim_smoothing_does_not_explode_the_graph():
    """Both spatial passes together, on the shape the gridded hobday path actually builds.

    The lat pass compounds whatever the lon pass did, so this is where the blow-up showed:
    2654 tasks / 1783 rechunk keys from a 6-chunk input before the fix.
    """
    da = xr.DataArray(
        dsa.ones((8, 60, 40, 60), dtype=np.uint32, chunks=(8, 60, 20, 20)),
        dims=("doy", "bin", "lat", "lon"),
    )
    out = _shifted_window_sum(_shifted_window_sum(da, "lon", 5, periodic=True), "lat", 5, periodic=False)

    assert out.chunks[2] == da.chunks[2]
    assert out.chunks[3] == da.chunks[3]
    rechunk_keys = sum(1 for k in out.data.dask if "rechunk" in str(k[0] if isinstance(k, tuple) else k))
    # Aligned adds need a bounded local shift per slice; the shredding failure produced
    # >1700 here. The bound is deliberately loose -- it is a blow-up detector, not a pin.
    assert rechunk_keys < 400, f"{rechunk_keys} rechunk keys suggests chunk realignment"


def test_numpy_backed_input_still_works():
    """target_chunks is None for a non-dask array; the rechunk must simply be skipped."""
    da = xr.DataArray(np.arange(20, dtype=np.uint32).reshape(10, 2), dims=("s", "b"))
    got = _shifted_window_sum(da, "s", 3, periodic=False).values
    expected = da.rolling({"s": 3}, center=True, min_periods=1).sum().values
    np.testing.assert_array_equal(got, expected.astype(got.dtype))
