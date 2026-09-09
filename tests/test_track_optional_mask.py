"""``mask`` is optional: a field with no invalid region needs no mask.

``mask`` marks which cells hold data, and for SST that is the land-sea mask. An
atmospheric field has no equivalent -- every cell is valid -- so requiring one made
every atmospheric caller construct an all-``True`` array by hand.

The load-bearing test here is :meth:`TestNoMaskMatchesAnAllTrueMask.test_every_output_is_identical`.
"No mask" must mean *exactly* an all-``True`` mask, not approximately: the derived
mask feeds `validate_spatial_chunking`, the morphological fill and the area filter, so
a shape, chunking or coordinate difference would move labels without moving values.

The two guard tests exist because ``mask`` precedes ``R_fill`` positionally, so
``R_fill`` needs a signature default it must never use. Left unguarded, a positional
``tracker(data_bin, 8)`` binds 8 to ``mask`` and reaches coordinate unification, which
reports an undetectable coordinate range -- a symptom naming neither mistake.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.exceptions import ConfigurationError

DATA_DIR = Path(__file__).parent / "data"

TRACK_KWARGS = {"R_fill": 4, "area_filter_quartile": 0.4, "T_fill": 2, "allow_merging": True}


@pytest.fixture(scope="module")
def extremes_gridded(dask_client):
    return xr.open_zarr(str(DATA_DIR / "extremes_gridded.zarr"), chunks={}).persist()


class TestNoMaskMatchesAnAllTrueMask:
    def test_every_output_is_identical(self, extremes_gridded, dask_client):
        """Bit-identical across every variable both runs produce, integer labels included."""
        events = extremes_gridded.extreme_events
        all_true = xr.ones_like(events.isel(time=0, drop=True), dtype=bool)

        explicit = marEx.tracker(events, all_true, **TRACK_KWARGS).run()
        derived = marEx.tracker(events, **TRACK_KWARGS).run()

        shared = sorted(set(explicit.data_vars) & set(derived.data_vars))
        assert shared, "the two runs produced no comparable variables"
        for name in shared:
            a, b = explicit[name].values, derived[name].values
            assert a.dtype == b.dtype, f"{name}: dtype moved"
            np.testing.assert_array_equal(a, b, err_msg=f"'{name}' differs between an omitted and an all-True mask")

    def test_the_derived_mask_is_all_true_and_carries_no_time_dimension(self, extremes_gridded, dask_client):
        tracked = marEx.tracker(extremes_gridded.extreme_events, **TRACK_KWARGS)
        assert tracked.mask.dtype == bool
        assert "time" not in tracked.mask.dims
        assert bool(tracked.mask.all().compute())

    def test_a_real_mask_is_still_honoured(self, extremes_gridded, dask_client):
        """The default must not quietly replace a supplied mask: a mask that blanks half
        the domain has to produce a different result from no mask at all."""
        events = extremes_gridded.extreme_events
        xdim = [d for d in events.dims if d != "time"][-1]
        half = xr.ones_like(events.isel(time=0, drop=True), dtype=bool)
        half = half.copy(data=half.values)
        half[{xdim: slice(0, events.sizes[xdim] // 2)}] = False
        half = half.chunk(dict(zip(half.dims, [-1] * half.ndim)))

        masked = marEx.tracker(events, half, **TRACK_KWARGS).run()
        unmasked = marEx.tracker(events, **TRACK_KWARGS).run()
        assert int(masked.ID_field.sum()) != int(unmasked.ID_field.sum())


class TestTheGuardsNameTheRightMistake:
    def test_r_fill_passed_positionally_is_caught_as_a_mask(self, extremes_gridded):
        with pytest.raises(ConfigurationError, match="mask must be an xarray.DataArray"):
            marEx.tracker(extremes_gridded.extreme_events, 8)

    def test_a_missing_r_fill_is_rejected_by_name(self, extremes_gridded):
        with pytest.raises(ConfigurationError, match="R_fill is required"):
            marEx.tracker(extremes_gridded.extreme_events)

    def test_regional_tracker_still_demands_coordinate_units(self, extremes_gridded):
        with pytest.raises(ConfigurationError, match="coordinate_units is required"):
            marEx.regional_tracker(extremes_gridded.extreme_events, R_fill=4)
