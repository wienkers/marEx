"""
A gridded time slice with no background cell is one object, not background.

Per-slice labelling compacts the local labels to 1..k. A slice that is foreground
everywhere has a single label and no 0, so a remap that assumes background is present
would send its only object to 0 and the whole slice would vanish.
"""

import numpy as np
import pytest
import xarray as xr

from marEx.track.objects import identify_objects


def _identify(field, regional_mode, time_chunk=1):
    data_bin = xr.DataArray(field, dims=("time", "lat", "lon")).chunk({"time": time_chunk, "lat": -1, "lon": -1})
    mask = xr.DataArray(np.ones(field.shape[1:], dtype=bool), dims=("lat", "lon"))
    ids, _, n_objects = identify_objects(data_bin, False, False, mask, None, "lon", regional_mode)
    return ids.values, n_objects


@pytest.mark.parametrize("regional_mode", [True, False], ids=["regional", "global"])
def test_fully_foreground_record_keeps_every_cell(regional_mode):
    field = np.ones((2, 3, 4), dtype=bool)
    ids, n_objects = _identify(field, regional_mode)
    assert int((ids > 0).sum()) == 24
    assert n_objects == 2
    assert np.unique(ids).tolist() == [1, 2]
    assert (ids[0] == 1).all() and (ids[1] == 2).all()


@pytest.mark.parametrize("regional_mode", [True, False], ids=["regional", "global"])
@pytest.mark.parametrize("time_chunk", [1, 2, 3])
def test_partial_full_partial_record(regional_mode, time_chunk):
    field = np.zeros((3, 3, 4), dtype=bool)
    field[0, 0:2, 1:3] = True  # one object, interior so no seam merge
    field[1] = True  # no background at all
    field[2, 1, 0] = True  # two objects, not joined across the longitude seam either
    field[2, 1, 2] = True
    ids, n_objects = _identify(field, regional_mode, time_chunk)

    np.testing.assert_array_equal(ids > 0, field)
    assert (ids[0][field[0]] == 1).all()
    assert (ids[1] == 2).all()
    assert sorted(np.unique(ids[2][field[2]]).tolist()) == [3, 4]
    assert n_objects == 4
