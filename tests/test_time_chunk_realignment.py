"""The tracker must not let dask_image's overlap rebalancing move a time-chunk boundary.

`fill_time_gaps` pads the record by `T_fill + 1` and hands it to dask_image's
`binary_closing`, whose overlap step calls `ensure_minimum_chunksize`: any chunk shorter
than the kernel depth is refilled from its neighbour, and nothing realigns it afterwards.
The trigger is arithmetic: ``0 < n_time % time_chunk < (T_fill + 1) // 2``.

Which ILLEGAL layout comes back depends on the chunk width, and both were observed
directly (T_fill=4, so depth 2, on the gridded fixture):

    n_time=21,  chunk=5   ->  (5, 5, 5, 4, 2)      ragged INTERIOR
    n_time=101, chunk=25  ->  (25, 25, 25, 24, 2)  ragged INTERIOR
    n_time=13,  chunk=3   ->  (3, 3, 3, 4)         oversized FINAL

Zarr accepts a short final chunk and nothing else, so both are fatal.
`ObjectIDRegionWriter._initialise` writes the full-length template schema with `to_zarr`,
and under `compute_mode="streaming"` the run dies mid-merge with, respectively

    ValueError: Zarr requires uniform chunk sizes except for final chunk.
    ValueError: Final chunk of Zarr array must be the same size or smaller than the first.

The production crash was the ragged-interior form: job 27272708 at n_time=951, time
chunk 25, T_fill=4, reporting ``ID_field ... ((25 x 37, 24, 2), ...)``. The fixtures below
use the same shape at a length that runs in seconds. This is the tracker-side twin of the
shifted-slice trap guarded by tests/test_shifted_window_sum.py: realign to the input's own
boundaries rather than trusting what an overlap operation hands back.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.track.morphology import fill_time_gaps

TEST_DATA_DIR = Path(__file__).parent / "data"

# chunk 5 with T_fill=4 gives depth (4+1)//2 = 2, so a remainder of 1 triggers the borrow.
# n_time=21 -> natural tail (5 x 4, 1); pre-fix this came back as (5, 5, 5, 4, 2), the same
# ragged-interior shape as the production crash, which is why this is the trigger fixture.
TRIGGER_NT = 21
TRIGGER_CHUNK = 5
TRIGGER_T_FILL = 4

# (n_time, time_chunk, does the borrow fire?). The two triggering rows produce the two
# distinct illegal layouts; the two non-triggering rows guard against the realignment
# firing -- or costing -- where it is not needed.
LAYOUT_CASES = [
    pytest.param(21, 5, True, id="ragged-interior-trigger"),
    pytest.param(13, 3, True, id="oversized-final-trigger"),
    pytest.param(20, 5, False, id="no-remainder"),
    pytest.param(22, 5, False, id="remainder-at-depth"),
]


def _time_chunks(da, timedim="time"):
    return da.chunks[da.dims.index(timedim)]


def _zarr_rejects(chunks):
    """Why zarr would refuse this layout, or None if it would accept it.

    Zarr requires every chunk to be equal except the last, which may only be SHORTER.
    Both failure modes must be named: a varying interior and an oversized final chunk are
    rejected by different code paths with different messages, and a predicate that checks
    only the interior silently passes the (3, 3, 3, 4) case.
    """
    if len(chunks) < 2:
        return None
    if len(set(chunks[:-1])) > 1:
        return f"ragged interior: {chunks}"
    if chunks[-1] > chunks[0]:
        return f"oversized final chunk: {chunks}"
    return None


@pytest.fixture(scope="module")
def extremes():
    return xr.open_zarr(str(TEST_DATA_DIR / "extremes_gridded.zarr"), chunks={})


def _tile_to(da, nt, timedim="time"):
    """Repeat the record along time up to `nt` steps (values are irrelevant here)."""
    reps = int(np.ceil(nt / da.sizes[timedim]))
    out = xr.concat([da] * reps, dim=timedim).isel({timedim: slice(0, nt)})
    return out.assign_coords({timedim: np.arange(nt)})


def _fill(data_bin, mask, t_fill=TRIGGER_T_FILL):
    return fill_time_gaps(
        data_bin,
        T_fill=t_fill,
        R_fill=8,
        timedim="time",
        ydim="lat",
        unstructured_grid=False,
        dilate_sparse=None,
        xdim="lon",
        mask=mask,
        regional_mode=False,
    )


class TestZarrRejects:
    """The predicate the other tests lean on has to reject both illegal layouts."""

    @pytest.mark.parametrize(
        "chunks,rejected",
        [
            ((5, 5, 5, 4, 2), True),  # ragged interior -- the production crash
            ((3, 3, 3, 4), True),  # oversized final -- the small-chunk form
            ((5, 5, 5, 5, 1), False),  # short final -- legal
            ((5, 5, 5, 5), False),  # uniform -- legal
        ],
    )
    def test_predicate_names_both_illegal_layouts(self, chunks, rejected):
        assert (_zarr_rejects(chunks) is not None) is rejected


class TestFillTimeGapsRealignsTime:
    """The unit-level guard: the operation must hand back the chunking it was given."""

    @pytest.mark.parametrize("nt,chunk,triggers", LAYOUT_CASES)
    def test_input_time_chunking_is_preserved(self, extremes, nt, chunk, triggers):
        data_bin = _tile_to(extremes.extreme_events, nt).chunk({"time": chunk, "lat": -1, "lon": -1})
        expected = _time_chunks(data_bin)
        depth = (TRIGGER_T_FILL + 1) // 2
        assert (0 < nt % chunk < depth) is triggers, "fixture no longer matches its trigger label"

        got = _time_chunks(_fill(data_bin, extremes.mask))

        # `got == expected` is the load-bearing assertion: asserting only that the result is
        # zarr-legal passes on the pre-fix code at nt=13/chunk=3, whose (3, 3, 3, 4) has a
        # uniform interior.
        assert got == expected, f"nt={nt} chunk={chunk}: not realigned: {got} != {expected}"
        assert _zarr_rejects(got) is None, f"nt={nt} chunk={chunk}: {_zarr_rejects(got)}"

    def test_filled_result_actually_writes_to_zarr(self, extremes, tmp_path):
        """Assert against zarr itself, not only against our model of what zarr allows."""
        data_bin = _tile_to(extremes.extreme_events, TRIGGER_NT).chunk({"time": TRIGGER_CHUNK, "lat": -1, "lon": -1})
        filled = _fill(data_bin, extremes.mask)
        filled.rename("filled").to_dataset().to_zarr(str(tmp_path / "filled.zarr"), mode="w")

    def test_a_deeper_kernel_still_realigns(self, extremes):
        """T_fill=8 widens the depth to 4, so remainders 1..3 all trigger the borrow."""
        data_bin = _tile_to(extremes.extreme_events, 22).chunk({"time": 5, "lat": -1, "lon": -1})
        expected = _time_chunks(data_bin)
        got = _time_chunks(_fill(data_bin, extremes.mask, t_fill=8))
        assert got == expected, f"T_fill=8: not realigned: {got} != {expected}"


class TestStreamingTrackerSurvivesTheTriggerLength:
    """The end-to-end guard: the crash job 27272708 actually hit."""

    def test_streaming_run_completes_at_the_trigger_length(self, extremes, tmp_path, dask_client):
        data_bin = _tile_to(extremes.extreme_events, TRIGGER_NT).chunk({"time": TRIGGER_CHUNK, "lat": -1, "lon": -1})
        tracker = marEx.tracker(
            data_bin,
            extremes.mask,
            R_fill=8,
            area_filter_quartile=0.5,
            T_fill=TRIGGER_T_FILL,
            allow_merging=True,
            overlap_threshold=0.5,
            nn_partitioning=True,
            quiet=True,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
        )
        out = tracker.run()
        assert out.ID_field.sizes["time"] == TRIGGER_NT
        marEx.clear_staging(out)

    def test_realignment_is_value_neutral_at_the_trigger_length(self, extremes, tmp_path, dask_client):
        """The fix moves chunk boundaries relative to a MORPHOLOGICAL op, so prove it moves no value.

        No pre-existing test reaches a triggering length -- those runs crashed -- and the
        goldens all run at n_time % time_chunk == 0, where the realignment never fires, so
        nothing else in the suite exercises this code path at all. Labels are integers: zero
        tolerance, per the repo's equivalence policy.

        Scope, so this is not read as more than it is: it compares the two compute modes on
        the CURRENT tree. It cannot by itself distinguish a value-neutral fix from one that
        changed both modes identically; that was checked once out-of-band by tracking the
        same fixture on pre-fix and post-fix code under `persist` (which does not crash at a
        triggering length) and comparing the ID_field, and the two agreed.
        """
        data_bin = _tile_to(extremes.extreme_events, TRIGGER_NT).chunk({"time": TRIGGER_CHUNK, "lat": -1, "lon": -1})
        common = {
            "R_fill": 8,
            "area_filter_quartile": 0.5,
            "T_fill": TRIGGER_T_FILL,
            "allow_merging": True,
            "overlap_threshold": 0.5,
            "nn_partitioning": True,
            "quiet": True,
        }

        persist_out = marEx.tracker(data_bin, extremes.mask, **common).run()
        stream_out = marEx.tracker(
            data_bin,
            extremes.mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **common,
        ).run()

        try:
            assert stream_out.sizes["ID"] == persist_out.sizes["ID"]
            np.testing.assert_array_equal(
                stream_out.ID_field.values,
                persist_out.ID_field.values,
            )
        finally:
            marEx.clear_staging(stream_out)
