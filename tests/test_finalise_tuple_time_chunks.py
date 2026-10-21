"""
finalise_dataset with explicit per-block time chunks and a seasonal cycle axis.

A tuple time entry lists block sizes that sum to the time length, so it cannot be applied to
the cycle axis (``dayofyear`` has 366 entries). The time axis takes the tuple as given; the
cycle axis takes the largest block as its step count, capped at the cycle length.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from marEx.core.finalise import finalise_dataset

DIMS = {"time": "time", "x": "lon", "y": "lat"}


def _seasonal_ds(n_time=10):
    time = xr.date_range("2001-01-01", periods=n_time, freq="D")
    return xr.Dataset(
        {
            "extreme_events": (("time", "lat", "lon"), np.zeros((n_time, 3, 4), dtype=bool)),
            "thresholds": (("dayofyear", "lat", "lon"), np.zeros((366, 3, 4), dtype=np.float32)),
        },
        coords={"time": time, "dayofyear": np.arange(1, 367), "lat": np.arange(3.0), "lon": np.arange(4.0)},
    ).chunk({"time": 1})


@pytest.mark.parametrize("time_chunks, cycle_chunks", [((4, 6), (6,) * 61), ((3, 3, 4), (4,) * 91 + (2,)), ([4, 6], (6,) * 61)])
def test_tuple_time_chunk_with_seasonal_cycle_axis(time_chunks, cycle_chunks):
    out = finalise_dataset(_seasonal_ds(), DIMS, DIMS, {"time": time_chunks}, SimpleNamespace(mode="lazy"))

    assert out.extreme_events.chunks == (tuple(time_chunks), (3,), (4,))
    assert out.thresholds.chunks == (cycle_chunks, (3,), (4,))


def test_integer_time_chunk_still_drives_the_cycle_axis():
    out = finalise_dataset(_seasonal_ds(), DIMS, DIMS, {"time": 5}, SimpleNamespace(mode="lazy"))

    assert out.extreme_events.chunks == ((5, 5), (3,), (4,))
    assert out.thresholds.chunks[0] == (5,) * 73 + (1,)
