"""
Payload and dtype checks on the tracker's intermediate graphs.

Value tests compare the returned labels, so they cannot see an intermediate that is
wider, larger or more often shipped than it needs to be. These tests intercept the
intermediates themselves.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.track.tracker import tracker as _Tracker

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def unstructured_extremes():
    # Rooted in the zarr store, not persisted: futures would tie the fixture to whichever
    # client was current when it was built, and that client may close before these tests run.
    return xr.open_zarr(str(DATA / "extremes_unstructured.zarr"), chunks={})


def _unstructured_tracker(ds, temp_dir, **kwargs):
    params = {
        "R_fill": 2,
        "area_filter_quartile": 0.5,
        "temp_dir": str(temp_dir),
        "T_fill": 0,
        "allow_merging": False,
        "unstructured_grid": True,
        "dimensions": {"x": "ncells"},
        "coordinates": {"x": "lon", "y": "lat"},
        "regional_mode": False,
        "coordinate_units": "degrees",
        "quiet": True,
        "neighbours": ds.neighbours,
        "cell_areas": ds.cell_areas,
    }
    params.update(kwargs)
    return marEx.tracker(ds.extreme_events.chunk({"time": 25, "ncells": -1}), ds.mask, **params)


def test_fixture_carries_no_futures(dask_client_unstructured, unstructured_extremes):
    """Re-adding ``.persist()`` to the fixture pins it to whichever client was current when it was
    built, and these tests fail with ``lost dependencies`` once that client closes."""
    from distributed.client import futures_of

    held = futures_of(unstructured_extremes)
    assert not held, f"the fixture holds {len(held)} future(s); keep it rooted in the zarr store"


class TestUniqueIdOffsets:
    """The globally unique ID field keeps the int32 dtype of the per-slice labels."""

    def test_refreshed_id_field_stays_int32(self, unstructured_extremes, tmp_path, monkeypatch, dask_client_unstructured):
        seen = []
        original = _Tracker.refresh_dask_graph

        def spy(self, data_bin):
            seen.append((data_bin.dtype, data_bin.data.dtype))
            return original(self, data_bin)

        monkeypatch.setattr(_Tracker, "refresh_dask_graph", spy)
        tracked = _unstructured_tracker(unstructured_extremes, tmp_path).run()

        assert seen, "refresh_dask_graph was never called on the unstructured path"
        for xr_dtype, block_dtype in seen:
            assert xr_dtype == np.int32, f"refreshed ID field is {xr_dtype}, expected int32"
            assert block_dtype == np.int32, f"refreshed ID field blocks are {block_dtype}, expected int32"
        assert int(tracked.ID_field.max()) == tracked.attrs["N_events_final"]

    @pytest.mark.parametrize("excess", [0, 1], ids=["largest-accepted", "first-rejected"])
    def test_int32_id_range_boundary(self, excess, unstructured_extremes, tmp_path, monkeypatch, dask_client_unstructured):
        """The last accepted total keeps max ID + 1 inside int32; one more object raises."""
        limit = np.iinfo(np.int32).max
        first = 2**30
        second = limit - 1 - first + excess  # total = limit - 1 (accepted) or limit (rejected)
        field = np.zeros((2, unstructured_extremes.sizes["ncells"]), dtype=np.int32)
        field[0, 0], field[1, 1] = first, second
        object_id_field = xr.DataArray(field, dims=("time", "ncells")).chunk({"time": 1})

        class Refreshed(Exception):
            pass

        def stop(self, data_bin):
            raise Refreshed(data_bin)

        tr = _unstructured_tracker(unstructured_extremes, tmp_path)
        monkeypatch.setattr(tr, "identify_objects", lambda *a, **k: (object_id_field, None, None))
        monkeypatch.setattr(_Tracker, "refresh_dask_graph", stop)
        if excess:
            with pytest.raises(ValueError, match="int32 ID range"):
                tr.track_objects(None)
        else:
            with pytest.raises(Refreshed) as refreshed:
                tr.track_objects(None)
            unique = refreshed.value.args[0]
            assert unique.dtype == np.int32
            assert int(unique.max()) == limit - 1
