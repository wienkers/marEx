"""Chunk-size invariance and chunking validation for the smoothed rolling climatology."""

import numpy as np
import xarray as xr

from marEx.anomaly.climatology import smoothed_rolling_climatology

SMOOTH_DAYS = 21


def _synthetic(nt=1100, ncells=60, seed=0):
    """Daily series with a seasonal cycle and permanently-NaN land cells."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((nt, ncells)).astype(np.float32) * 5 + 280
    values += 3 * np.sin(2 * np.pi * np.arange(nt)[:, None] / 365.25).astype(np.float32)
    values[:, :4] = np.nan
    return xr.DataArray(
        values,
        dims=("time", "ncells"),
        coords={
            "time": xr.date_range("2007-01-01", periods=nt, freq="D"),
            "lon": ("ncells", np.linspace(-180, 180, ncells, dtype=np.float32)),
            "lat": ("ncells", np.linspace(-80, 80, ncells, dtype=np.float32)),
        },
    )


class TestChunkSizeInvariance:
    """The climatology must not depend on how the input is chunked (D-091)."""

    def test_output_is_bit_identical_across_time_and_space_chunks(self):
        # bottleneck's move_mean restarts its running sum at each dask block boundary, and flox
        # accumulates its grouped mean block by block, so before D-091 the result moved by a
        # few float32 ULP with the time chunking -- enough to flip a 0.01 threshold bin. The
        # reduction now runs with time whole, so every layout gives the same bits.
        da = _synthetic()
        dims = {"time": "time", "x": "ncells"}
        coords = {"time": "time", "x": "lon", "y": "lat"}

        ref = smoothed_rolling_climatology(da.chunk({"time": -1, "ncells": -1}), 2, SMOOTH_DAYS, dims, coords).compute()
        for layout in ({"time": 2, "ncells": -1}, {"time": 21, "ncells": -1}, {"time": 40, "ncells": 7}):
            got = smoothed_rolling_climatology(da.chunk(layout), 2, SMOOTH_DAYS, dims, coords).compute()
            assert np.array_equal(got.values, ref.values, equal_nan=True), f"climatology moved under {layout}"

    def test_caller_layout_is_restored(self):
        da = _synthetic().chunk({"time": 40, "ncells": 7})
        result = smoothed_rolling_climatology(
            da, 2, SMOOTH_DAYS, {"time": "time", "x": "ncells"}, {"time": "time", "x": "lon", "y": "lat"}
        )
        assert dict(result.chunksizes) == dict(da.chunksizes)


class TestTimeChunkValidation:
    """Time chunks shorter than the smoothing window used to fail inside bottleneck."""

    def test_time_chunk_below_smoothing_window_is_accepted(self):
        # 20 % 2 == 0 formed a 20-element block against a 21-day window, which raised
        # (first from bottleneck, later as a ConfigurationError). The smoothing now sees the
        # time axis whole, so the chunking cannot produce a short block.
        da = _synthetic()
        dims = {"time": "time", "x": "ncells"}
        coords = {"time": "time", "x": "lon", "y": "lat"}

        got = smoothed_rolling_climatology(da.chunk({"time": 2, "ncells": -1}), 2, SMOOTH_DAYS, dims, coords).compute()
        ref = smoothed_rolling_climatology(da.chunk({"time": -1, "ncells": -1}), 2, SMOOTH_DAYS, dims, coords).compute()

        assert np.array_equal(got.values, ref.values, equal_nan=True)

    def test_chunk_not_dividing_the_pad_is_accepted(self):
        # chunk 4 with an 11-day window: 10 % 4 == 2, so no short block forms and this
        # has always worked. Rejecting it would break the detect golden tests, which is
        # exactly what a naive "chunk < window" rule did.
        #
        # The divisibility rule this pins is an emergent property of the
        # xarray -> dask.overlap -> bottleneck chain, verified against dask 2025.9.1 /
        # bottleneck 1.6.0. If an upgrade ever makes this configuration fail, that is
        # UPSTREAM BEHAVIOUR CHANGING, not a regression in marEx -- widen the guard in
        # smoothed_rolling_climatology towards "chunk >= smooth_days" rather
        # than relaxing this test.
        da = _synthetic().chunk({"time": 4, "ncells": -1})

        result = smoothed_rolling_climatology(
            da,
            2,
            11,
            {"time": "time", "x": "ncells"},
            {"time": "time", "x": "lon", "y": "lat"},
        ).compute()

        assert np.isfinite(result.values).any()

    def test_window_longer_than_series_is_accepted(self):
        # 10 days with a 21-day window computes fine and yields all-NaN, which is the
        # correct answer rather than an error. An earlier guard modelled the upstream
        # failure with a divisibility rule and rejected this, breaking a logging test
        # that had always passed. The guard now probes the real stack instead of
        # predicting it, so cases upstream accepts stay accepted.
        da = _synthetic(nt=10, ncells=6).chunk({"time": 5, "ncells": -1})

        result = smoothed_rolling_climatology(
            da,
            2,
            SMOOTH_DAYS,
            {"time": "time", "x": "ncells"},
            {"time": "time", "x": "lon", "y": "lat"},
        )

        assert result is not None
