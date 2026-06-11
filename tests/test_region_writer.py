"""Unit tests for ObjectIDRegionWriter -- the merge loop's incremental zarr writer."""

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

from marEx.track.region_writer import ObjectIDRegionWriter


def _field(n_time=8, ny=4, nx=5, chunk=2):
    """A chunked int32 field with coords, shaped like object_id_field_unique."""
    data = dsa.from_array(
        np.arange(n_time * ny * nx, dtype=np.int32).reshape(n_time, ny, nx),
        chunks=(chunk, ny, nx),
    )
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(n_time),
            "lat": np.linspace(-10, 10, ny),
            "lon": np.linspace(0, 40, nx),
        },
        name="ID_field",
    )


class TestObjectIDRegionWriter:
    def test_roundtrip_reproduces_the_written_values(self, tmp_path):
        """Writing every region then finalising returns exactly what was written."""
        template = _field()
        writer = ObjectIDRegionWriter(template, tmp_path / "f.zarr", "time")
        source = template.compute()
        for start in range(0, 8, 2):
            writer.write(start, start + 2, source.isel(time=slice(start, start + 2)))
        out = writer.finalise()
        xr.testing.assert_identical(out.compute(), source)

    def test_regions_may_be_written_with_a_ragged_final_chunk(self, tmp_path):
        """n_time not divisible by the chunk size must still round-trip."""
        template = _field(n_time=7, chunk=3)
        writer = ObjectIDRegionWriter(template, tmp_path / "f.zarr", "time")
        source = template.compute()
        for start, end in [(0, 3), (3, 6), (6, 7)]:
            writer.write(start, end, source.isel(time=slice(start, end)))
        xr.testing.assert_identical(writer.finalise().compute(), source)

    def test_result_is_lazy_not_materialised(self, tmp_path):
        """finalise() must return a dask-backed array read from disk."""
        template = _field()
        writer = ObjectIDRegionWriter(template, tmp_path / "f.zarr", "time")
        source = template.compute()
        writer.write(0, 8, source)
        out = writer.finalise()
        assert out.chunks is not None, "finalise() returned a materialised array"

    def test_dtype_and_coords_survive(self, tmp_path):
        template = _field()
        writer = ObjectIDRegionWriter(template, tmp_path / "f.zarr", "time")
        writer.write(0, 8, template.compute())
        out = writer.finalise()
        assert out.dtype == np.int32
        assert list(out.dims) == ["time", "lat", "lon"]
        np.testing.assert_array_equal(out.lat.values, template.lat.values)

    def test_writing_before_finalise_does_not_require_ordered_calls(self, tmp_path):
        """Regions are disjoint; the writer must not assume call order."""
        template = _field()
        writer = ObjectIDRegionWriter(template, tmp_path / "f.zarr", "time")
        source = template.compute()
        for start in [4, 0, 6, 2]:
            writer.write(start, start + 2, source.isel(time=slice(start, start + 2)))
        xr.testing.assert_identical(writer.finalise().compute(), source)

    def test_finalise_without_any_write_raises(self, tmp_path):
        writer = ObjectIDRegionWriter(_field(), tmp_path / "f.zarr", "time")
        with pytest.raises(RuntimeError, match="no regions"):
            writer.finalise()

    def test_full_coverage_passes(self, tmp_path):
        """Regions that exactly tile [0, n_time) must finalise without error."""
        template = _field()
        writer = ObjectIDRegionWriter(template, tmp_path / "f.zarr", "time")
        source = template.compute()
        for start in range(0, 8, 2):
            writer.write(start, start + 2, source.isel(time=slice(start, start + 2)))
        out = writer.finalise()
        xr.testing.assert_identical(out.compute(), source)

    def test_gap_in_coverage_raises(self, tmp_path):
        """A dropped region must raise rather than silently leave zarr fill values."""
        template = _field()
        writer = ObjectIDRegionWriter(template, tmp_path / "f.zarr", "time")
        source = template.compute()
        # Skip [2, 4): write [0, 2), [4, 6), [6, 8).
        writer.write(0, 2, source.isel(time=slice(0, 2)))
        writer.write(4, 6, source.isel(time=slice(4, 6)))
        writer.write(6, 8, source.isel(time=slice(6, 8)))
        with pytest.raises(RuntimeError, match="gap"):
            writer.finalise()

    def test_overlap_in_coverage_raises(self, tmp_path):
        """A region written twice (or overlapping another) must raise."""
        template = _field()
        writer = ObjectIDRegionWriter(template, tmp_path / "f.zarr", "time")
        source = template.compute()
        writer.write(0, 4, source.isel(time=slice(0, 4)))
        writer.write(2, 6, source.isel(time=slice(2, 6)))
        writer.write(6, 8, source.isel(time=slice(6, 8)))
        with pytest.raises(RuntimeError, match="overlap"):
            writer.finalise()
