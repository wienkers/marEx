"""
Unstructured areas are summed in float64, stored as float32, and compared in float64.

A float32 running sum over a large object drifts with the object's size, so a merge decision
near the overlap threshold could depend on how big the objects are rather than on their
overlap. Every unstructured area -- serial overlaps, object properties, the merge kernel's
split consolidation and the recomputed event properties -- follows one rule: accumulate in
float64, round once to float32, and divide the rounded values in float64. The serial and
parallel paths then make the same decisions and return the same areas.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dask.distributed import Client

import marEx
from marEx.track import merge_split
from marEx.track import objects as _objects
from marEx.track import overlap as _overlap
from marEx.track.objects import calculate_object_properties
from marEx.track.overlap import check_overlap_slice

N_CELLS = 2_000_000
TEST_DATA_DIR = Path(__file__).parent / "data"


def _cell_area():
    return np.random.default_rng(0).uniform(0.5, 1.5, N_CELLS).astype(np.float32)


def test_overlap_area_is_the_rounded_float64_sum():
    area = _cell_area()
    ids = np.ones(N_CELLS, dtype=np.int32)
    result = check_overlap_slice(ids, ids, True, area)
    assert result.shape == (1, 3)
    assert np.float32(result[0, 2]) == np.float32(area.astype(np.float64).sum())


def test_object_area_is_the_rounded_float64_sum():
    area = _cell_area()
    ids = np.ones((1, N_CELLS), dtype=np.int32)
    lat = xr.DataArray(np.linspace(-60, 60, N_CELLS, dtype=np.float32), dims="ncells")
    lon = xr.DataArray(np.linspace(-170, 170, N_CELLS, dtype=np.float32), dims="ncells")
    props = calculate_object_properties(
        xr.DataArray(ids, dims=("time", "ncells")).chunk({"time": 1, "ncells": -1}),
        unstructured_grid=True,
        lat=lat,
        lon=lon,
        cell_area=xr.DataArray(area, dims="ncells"),
        timedim="time",
        regional_mode=False,
        ydim=None,
        xdim="ncells",
        properties=["area", "centroid"],
    )
    assert props.area.dtype == np.float32
    assert props.area.values.tolist() == [np.float32(area.astype(np.float64).sum())]


class TestConsolidationMatchesSerial:
    """The unstructured kernel's split consolidation must make the serial path's decisions.

    Near the threshold the decision depends on the arithmetic, not only on the rule. Each case
    below is decided one way by the shared rule and the other way by some other arithmetic
    (a float32 ratio, or float32 accumulation), or sits exactly on the threshold.
    """

    @staticmethod
    def _serial(prev, cur, area, threshold):
        n = prev.size
        lat = xr.DataArray(np.zeros(n, dtype=np.float32), dims="ncells")
        lon = xr.DataArray(np.linspace(0.0, 1.0, n, dtype=np.float32), dims="ncells")
        cell_area = xr.DataArray(area, dims="ncells")
        kwargs = {
            "unstructured_grid": True,
            "lat": lat,
            "lon": lon,
            "cell_area": cell_area,
            "timedim": "time",
            "regional_mode": False,
            "ydim": None,
            "xdim": "ncells",
        }
        props = [
            _objects.calculate_object_properties(xr.DataArray(field, dims="ncells"), properties=["area", "centroid"], **kwargs)
            for field in (prev, cur)
        ]
        store = _objects.ObjectPropsStore.from_dataset(xr.concat(props, dim="ID"))
        out, _ = _overlap.consolidate_object_ids(
            xr.DataArray(prev, dims="ncells"),
            xr.DataArray(cur.copy(), dims="ncells"),
            store,
            1,
            True,
            cell_area,
            threshold,
            lat,
            lon,
            "time",
            False,
            None,
            "ncells",
        )
        return np.asarray(out.values)

    @pytest.mark.parametrize(
        "prev, cur, area, threshold, expected",
        [
            # Equal areas: 1/3 in float64 is just below the float32-rounded threshold, so only
            # child 3 (whole overlap) is accepted and nothing is consolidated. A float32 ratio
            # rounds 1/3 up to the threshold and wrongly relabels child 3 as 2.
            ([1, 1, 1, 0, 0], [2, 3, 0, 2, 2], [1.0, 1.0, 1.0, 1.0, 1.0], 0.33333334, [2, 3, 0, 2, 2]),
            # Unequal areas: child 2's overlap fraction is 0.7904288999387088 under the shared
            # rule, below the threshold; float32 accumulation gives 0.790428933398726 and accepts it.
            (
                [1, 1, 1, 1, 0],
                [2, 2, 2, 3, 2],
                [1.40057528, 1.18298054, 0.68794185, 0.86739153, 1.46195507],
                0.7904289166687174,
                [2, 2, 2, 3, 2],
            ),
            # Exactly on the threshold: 2 x 0.1 over 4 x 0.1 is exactly 0.5 after float32
            # rounding, and the test is inclusive, so both children are accepted and merged.
            ([1, 1, 1, 1, 0, 0, 0, 0], [2, 2, 3, 3, 2, 2, 3, 3], [0.1] * 8, 0.5, [2, 2, 2, 2, 2, 2, 2, 2]),
            # Just above it: neither child is accepted.
            (
                [1, 1, 1, 1, 0, 0, 0, 0],
                [2, 2, 3, 3, 2, 2, 3, 3],
                [0.1] * 8,
                float(np.nextafter(0.5, 1.0)),
                [2, 2, 3, 3, 2, 2, 3, 3],
            ),
        ],
    )
    def test_parallel_consolidation_equals_serial(self, prev, cur, area, threshold, expected):
        prev = np.asarray(prev, dtype=np.int32)
        cur = np.asarray(cur, dtype=np.int32)
        area = np.asarray(area, dtype=np.float32)
        serial = self._serial(prev, cur, area, threshold)
        parallel = cur.copy()
        merge_split._consolidate_slice(prev, parallel, area, threshold)
        np.testing.assert_array_equal(serial, expected)
        np.testing.assert_array_equal(parallel, expected)


# The configuration the merging fixture's blobs were built for (test_merge_shared_kernel.py).
UNSTRUCTURED_KWARGS = {
    "R_fill": 1,
    "area_filter_absolute": 5,
    "T_fill": 2,
    "allow_merging": True,
    "overlap_threshold": 0.8,
    "nn_partitioning": True,
    "unstructured_grid": True,
    "dimensions": {"x": "ncells"},
    "coordinates": {"x": "lon", "y": "lat"},
    "regional_mode": False,
    "coordinate_units": "degrees",
    "quiet": True,
}


@pytest.fixture(scope="module")
def inprocess_client():
    client = Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None)
    yield client
    client.close()


def test_tracked_event_area_is_the_rounded_float64_sum(inprocess_client, tmp_path):
    """The area the tracker returns for each event and timestep is the rounded float64 sum.

    The cell areas are spread over six decades so that a float32 sum visibly drifts from the
    rounded float64 one even on this small mesh (on the shipped areas it rarely shows).
    """
    data = xr.open_zarr(str(TEST_DATA_DIR / "extremes_unstructured_merging.zarr"), chunks={})
    rng = np.random.default_rng(0)
    spread = 10 ** rng.uniform(-3, 3, data.sizes["ncells"])
    cell_areas = (data.cell_areas.values.astype(np.float64) * spread).astype(np.float32)
    cell_areas = xr.DataArray(cell_areas, dims=data.cell_areas.dims, coords=data.cell_areas.coords)
    tr = marEx.tracker(
        data.extreme_events.chunk({"time": 4, "ncells": -1}),
        data.mask,
        neighbours=data.neighbours,
        cell_areas=cell_areas,
        temp_dir=str(tmp_path),
        **UNSTRUCTURED_KWARGS,
    )
    events = tr.run().compute()

    id_field = events.ID_field.transpose("time", "ncells").values
    area = events.area.transpose("time", "ID").values
    assert area.dtype == np.float32
    checked, wrong = 0, []
    for t in range(id_field.shape[0]):
        for j, event_id in enumerate(events.ID.values):
            cells = id_field[t] == event_id
            if not cells.any():
                continue
            checked += 1
            expected = np.float32(cell_areas.values[cells].astype(np.float64).sum())
            if area[t, j] != expected:
                wrong.append((t, int(event_id), float(area[t, j]), float(expected)))
    assert checked > 50, f"only {checked} event-timesteps were checked"
    assert not wrong, f"{len(wrong)} of {checked} event areas differ from the rounded float64 sum, e.g. {wrong[:3]}"
