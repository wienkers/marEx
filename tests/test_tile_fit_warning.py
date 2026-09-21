"""The fit warning on the canonical rechunk.

The canonical layout holds time whole and caps the spatial tile so one task's
working set stays near an element budget. The cap is honoured by shrinking the
tile -- except in two cases, which are exactly what these tests pin:

* a ``window_spatial`` floor forces a horizontal chunk wider than the tile the
  budget allows (reachable today through the day-of-year histogram, whose tiling
  applies that floor);
* a single spatial cell already reads or writes more elements than the whole
  budget, so no tile is small enough.

In both the returned chunks are unchanged -- the check observes, it never nudges
a tile -- and the ordinary path stays silent.
"""

import numpy as np
import pytest
import xarray as xr

from marEx.core import dimensions as dims_mod
from marEx.core.dimensions import FIT_WARNING_MARKER, canonical_time_chunks, tile_spatial_chunks
from marEx.extremes import histogram as hist_mod

DIMENSIONS = {"time": "time", "y": "lat", "x": "lon"}


def _field(n_time=200, n_lat=20, n_lon=30):
    """Small gridded field, chunked the way a caller usually hands one over."""
    da = xr.DataArray(
        np.zeros((n_time, n_lat, n_lon), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(n_time),
            "lat": np.arange(n_lat),
            "lon": np.arange(n_lon),
        },
        name="sst",
    )
    return da.chunk({"time": 25, "lat": -1, "lon": -1})


def _fit_warnings(capture):
    return [m for m in capture.messages if FIT_WARNING_MARKER in m]


class TestSilentOnTheOrdinaryPath:
    """A warning that fires on the normal path is worse than no warning."""

    def test_canonical_rechunk_is_silent(self, marex_warnings):
        da = _field()
        with marex_warnings as cap:
            chunks = canonical_time_chunks(da, DIMENSIONS)
        assert _fit_warnings(cap) == []
        assert chunks["time"] == -1

    def test_forced_small_budget_is_silent(self, marex_warnings, monkeypatch):
        """The budget is honoured by shrinking the tile: that is not a misfit.

        This is the layout that tests/test_detect_chunk_invariance.py forces to
        exercise the multi-tile path (TASK_ELEMENTS 30_000).
        """
        monkeypatch.setattr(dims_mod, "TASK_ELEMENTS", 30_000)
        da = _field()
        with marex_warnings as cap:
            chunks = canonical_time_chunks(da, DIMENSIONS)
        assert _fit_warnings(cap) == []
        assert chunks["lat"] < 20 and chunks["lon"] < 30  # it really did tile

    def test_tile_side_rounding_alone_is_not_a_misfit(self, marex_warnings, monkeypatch):
        """A tile over budget from integer rounding ALONE is not reported.

        This has to be pinned on the day-of-year histogram path. That one takes a
        flat ``round(cells ** (1/rank))`` side, which can overshoot its own cell
        budget (measured: 1.33x at rank 2, up to 4.0x at rank 5); the greedy loop
        in ``tile_spatial_chunks`` cannot overshoot at all (measured 1.0x over
        40,920 cases), so the same test written against it is VACUOUS -- it never
        reaches the guard, and passes with the guard deleted.

        Budget 3,000,000 over a divisor of 1,000,000 gives 3 cells; the flat side
        is ``round(sqrt(3))`` = 2, so the tile is 4 cells = 4,000,000 elements,
        1.33x the budget with NO floor and NO single cell over budget. Silent.
        """
        monkeypatch.setattr(hist_mod, "_HISTOGRAM_TASK_ELEMENTS", 3_000_000)
        da = _field()
        with marex_warnings as cap:
            chunks = hist_mod._histogram_tile_chunks(da, DIMENSIONS, n_bins=1000, window_spatial=None, cycle_length=1000)
        # The case must actually BE over budget, or this test pins nothing. Asserted,
        # not assumed, so it can never go vacuous again the way its first draft did.
        estimate = chunks["lat"] * chunks["lon"] * 1_000_000
        assert estimate > 3_000_000, f"vacuous: {chunks} gives {estimate}, inside the budget"
        assert _fit_warnings(cap) == []


class TestSpatialWindowFloor:
    """A window wider than the budget's tile: the cap cannot be honoured."""

    def test_warns_and_names_the_window(self, marex_warnings):
        da = _field()
        with marex_warnings as cap:
            chunks = tile_spatial_chunks(
                da,
                ("lat", "lon"),
                input_elements_per_cell=1000,
                target_elements=10_000,
                floor_dims=("lat", "lon"),
                floor=15,
            )
        messages = _fit_warnings(cap)
        assert len(messages) == 1, messages
        message = messages[0]
        assert "15" in message  # the window that forced it
        assert "225,000" in message  # 15 x 15 cells x 1000 elements
        assert "10,000" in message  # the budget it broke
        # Observation only: the tile is exactly what it was before the check.
        assert chunks == {"lat": 15, "lon": 15}

    def test_floor_inside_the_budget_is_silent(self, marex_warnings):
        da = _field()
        with marex_warnings as cap:
            chunks = tile_spatial_chunks(
                da,
                ("lat", "lon"),
                input_elements_per_cell=1,
                target_elements=10_000,
                floor_dims=("lat", "lon"),
                floor=3,
            )
        assert _fit_warnings(cap) == []
        assert min(chunks.values()) >= 3


class TestSingleCellOverBudget:
    def test_warns_when_one_cell_exceeds_the_whole_budget(self, marex_warnings):
        da = _field()
        with marex_warnings as cap:
            chunks = tile_spatial_chunks(da, ("lat", "lon"), input_elements_per_cell=5_000, target_elements=100)
        messages = _fit_warnings(cap)
        assert len(messages) == 1, messages
        assert "single" in messages[0].lower()
        assert chunks == {"lat": 1, "lon": 1}

    def test_output_side_counts_too(self, marex_warnings):
        """The budget divisor is the larger of the two sides (e4fcc89)."""
        da = _field()
        with marex_warnings as cap:
            tile_spatial_chunks(
                da,
                ("lat", "lon"),
                input_elements_per_cell=1,
                output_elements_per_cell=5_000,
                target_elements=100,
            )
        assert len(_fit_warnings(cap)) == 1


class TestHistogramTiling:
    """The one production path that applies the window floor over the budget."""

    def test_sub_daily_cycle_with_a_wide_window_warns(self, marex_warnings, monkeypatch):
        monkeypatch.setattr(hist_mod, "_HISTOGRAM_TASK_ELEMENTS", 1_000_000)
        da = _field()
        with marex_warnings as cap:
            chunks = hist_mod._histogram_tile_chunks(
                da,
                DIMENSIONS,
                n_bins=1000,
                window_spatial=15,
                cycle_length=8784,  # hourly-of-year
            )
        messages = _fit_warnings(cap)
        assert len(messages) == 1, messages
        assert "15" in messages[0]
        # Unchanged: time whole, both horizontal dims at the window width.
        assert chunks == {"time": -1, "lat": 15, "lon": 15}

    def test_hourly_cadence_at_the_shipped_budget_warns(self, marex_warnings):
        """The reachability case: no monkeypatching, shipped defaults.

        A sub-daily cycle multiplies the per-cell output by the steps per day, so
        the cell budget collapses (50e6 / (8784 x 1000) = 5 cells) while
        ``window_spatial`` does not move. ``resolve_window_spatial`` substitutes 5
        for a structured grid on the approximate seasonal path, and 5x5 cells at
        8,784,000 elements each is 878 MB in one task -- 4.4x the budget, and
        silent until now.
        """
        da = _field(n_lat=60, n_lon=80)
        with marex_warnings as cap:
            chunks = hist_mod._histogram_tile_chunks(da, DIMENSIONS, n_bins=1000, window_spatial=5, cycle_length=8784)
        messages = _fit_warnings(cap)
        assert len(messages) == 1, messages
        assert "219,600,000" in messages[0]
        assert "50,000,000" in messages[0]
        assert chunks == {"time": -1, "lat": 5, "lon": 5}

    def test_the_same_window_on_a_daily_cycle_is_silent(self, marex_warnings):
        """The control for the test above: only the cadence changes."""
        da = _field(n_lat=60, n_lon=80)
        with marex_warnings as cap:
            chunks = hist_mod._histogram_tile_chunks(da, DIMENSIONS, n_bins=1000, window_spatial=5, cycle_length=366)
        assert _fit_warnings(cap) == []
        assert chunks == {"time": -1, "lat": 12, "lon": 12}

    def test_daily_default_is_silent(self, marex_warnings):
        da = _field()
        with marex_warnings as cap:
            hist_mod._histogram_tile_chunks(da, DIMENSIONS, n_bins=200, window_spatial=None, cycle_length=366)
        assert _fit_warnings(cap) == []

    def test_window_inside_the_budget_is_silent(self, marex_warnings):
        da = _field()
        with marex_warnings as cap:
            hist_mod._histogram_tile_chunks(da, DIMENSIONS, n_bins=200, window_spatial=5, cycle_length=366)
        assert _fit_warnings(cap) == []


class TestMessageIsActionable:
    def test_it_names_levers_the_user_actually_has(self, marex_warnings):
        da = _field()
        with marex_warnings as cap:
            tile_spatial_chunks(
                da,
                ("lat", "lon"),
                input_elements_per_cell=1000,
                target_elements=10_000,
                floor_dims=("lat", "lon"),
                floor=15,
            )
        message = _fit_warnings(cap)[0]
        assert "window_spatial" in message
        assert "streaming" in message
        # Never suggest chunking the input finer: the floor is applied before the
        # caller's own chunks are taken into account, so that would silently
        # narrow the tile below the rolling window.
        assert "chunk your input" not in message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
