"""
User-facing error messages from the tracker: each names the actual problem and a route that works.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import CoordinateError, TrackingError
from marEx.track.grid import unify_coordinates

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def extremes_gridded(dask_client):
    return xr.open_zarr(str(DATA_DIR / "extremes_gridded.zarr"), chunks={}).isel(time=slice(0, 10))


def _regional_field(ncells=50):
    return xr.DataArray(
        np.zeros((2, ncells), dtype=bool),
        dims=("time", "ncells"),
        coords={"time": np.arange(2), "lon": ("ncells", np.linspace(10.0, 60.0, ncells)), "lat": ("ncells", np.zeros(ncells))},
    )


class TestUndetectableCoordinateUnits:
    """A longitude range that is neither a full turn in degrees nor in radians."""

    def test_unstructured_suggests_coordinate_units_not_regional_mode(self):
        # regional_mode raises NotImplementedError on an unstructured grid, so suggesting it there
        # sends the user into a second error. coordinate_units alone is the route that works.
        with pytest.raises(CoordinateError) as excinfo:
            unify_coordinates(_regional_field(), False, None, "lon", "lat", unstructured_grid=True)
        suggestions = " ".join(excinfo.value.suggestions)
        assert "regional_mode" not in suggestions
        assert "coordinate_units='degrees'" in suggestions

    def test_structured_still_suggests_regional_mode(self):
        with pytest.raises(CoordinateError) as excinfo:
            unify_coordinates(_regional_field(), False, None, "lon", "lat")
        assert any("regional_mode=True" in s for s in excinfo.value.suggestions)


def test_area_filter_removing_every_event_raises_a_clear_error(extremes_gridded):
    """An area threshold above every object's area leaves nothing to track: say so, not ZeroDivisionError."""
    tracker = marEx.tracker(
        extremes_gridded.extreme_events,
        extremes_gridded.mask,
        R_fill=0,
        T_fill=0,
        area_filter_absolute=10**9,
        quiet=True,
    )
    with pytest.raises(TrackingError, match="area_filter_absolute"):
        tracker.run()


@pytest.fixture(scope="module")
def extremes_unstructured(dask_client):
    return xr.open_zarr(str(DATA_DIR / "extremes_unstructured.zarr"), chunks={}).isel(time=slice(0, 10))


def _unstructured_tracker(ds, cell_areas, **kwargs):
    return marEx.tracker(
        ds.extreme_events,
        ds.mask,
        cell_areas=cell_areas,
        neighbours=ds.neighbours,
        R_fill=1,
        T_fill=0,
        allow_merging=False,
        unstructured_grid=True,
        dimensions={"x": "ncells"},
        coordinates={"x": "lon", "y": "lat"},
        coordinate_units="degrees",
        quiet=True,
        **kwargs,
    )


def test_unstructured_prefilter_leaving_nothing_names_the_setting(extremes_unstructured, tmp_path):
    """On a short subset every object is at most a few cells, below the unstructured pre-filter cutoff."""
    tracker = _unstructured_tracker(
        extremes_unstructured, extremes_unstructured.cell_areas, area_filter_absolute=10**9, temp_dir=str(tmp_path)
    )
    with pytest.raises(TrackingError, match="area_filter_absolute=1000000000"):
        tracker.run()


def test_zero_cell_areas_raise_a_clear_error(dask_client, tmp_path):
    ds = xr.open_zarr(str(DATA_DIR / "extremes_unstructured.zarr"), chunks={})
    tracker = _unstructured_tracker(ds, ds.cell_areas * 0, area_filter_absolute=6, temp_dir=str(tmp_path))
    with pytest.raises(TrackingError, match="zero total area"):
        tracker.run()
