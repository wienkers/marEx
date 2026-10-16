"""
The unstructured merge step's temporary IDs must fit the int32 range it writes them in.

Temporary IDs are minted above two reserved ranges of ``n_time * id_stride`` IDs past the
largest object ID. An input whose object IDs fit int32 can still overflow there, so the
whole range is checked before merging starts.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.track import merge_split
from marEx.track import overlap as _overlap

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def unstructured_extremes():
    return xr.open_zarr(str(DATA / "extremes_unstructured.zarr"), chunks={})


class _ReachedOverlap(Exception):
    pass


@pytest.mark.parametrize("excess", [0, 1], ids=["largest-accepted", "first-rejected"])
def test_temporary_id_range_is_checked_before_merging(
    excess, unstructured_extremes, tmp_path, monkeypatch, dask_client_unstructured
):
    """The last accepted largest ID leaves the top temporary ID + 1 inside int32; one more raises."""
    ds = unstructured_extremes
    n_time = 2
    id_stride = merge_split.MAX_MERGES * (merge_split.MAX_PARENTS - 1)
    largest = np.iinfo(np.int32).max - 1 - 2 * n_time * id_stride + excess
    # Slice 1's label is offset by slice 0's maximum (1) when IDs are made unique across time.
    field = np.zeros((n_time, ds.sizes["ncells"]), dtype=np.int32)
    field[0, 0], field[1, 1] = 1, largest - 1
    object_id_field = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": 1})

    def stop(*args, **kwargs):
        raise _ReachedOverlap

    tr = marEx.tracker(
        ds.extreme_events.chunk({"time": 25, "ncells": -1}),
        ds.mask,
        R_fill=2,
        area_filter_quartile=0.5,
        temp_dir=str(tmp_path),
        T_fill=0,
        allow_merging=True,
        unstructured_grid=True,
        dimensions={"x": "ncells"},
        coordinates={"x": "lon", "y": "lat"},
        regional_mode=False,
        coordinate_units="degrees",
        quiet=True,
        neighbours=ds.neighbours,
        cell_areas=ds.cell_areas,
    )
    monkeypatch.setattr(tr, "identify_objects", lambda *a, **k: (object_id_field, None, None))
    monkeypatch.setattr(_overlap, "find_overlapping_objects", stop)
    if excess:
        with pytest.raises(ValueError, match="int32 ID range"):
            tr.track_objects(None)
    else:
        with pytest.raises(_ReachedOverlap):
            tr.track_objects(None)
