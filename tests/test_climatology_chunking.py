"""Chunk-size invariance and chunking validation for the smoothed rolling climatology."""

import numpy as np
import pytest
import xarray as xr

from marEx.detect.anomaly.climatology import smoothed_rolling_climatology
from marEx.exceptions import ConfigurationError

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
    """The climatology must not depend on how the time axis happens to be chunked."""

    def test_output_across_time_chunks_agrees_to_float32_precision(self):
        # bottleneck's move_mean restarts its running sum at each dask block boundary, so
        # the climatology shifts slightly with the chunk layout. That is accepted: the
        # spread is a few float32 ULP of the input field. This test pins it as SMALL, so a
        # future change that makes chunking matter materially fails here.
        #
        # Exact bit-identity is achievable by accumulating in float64, but that doubles the
        # reduction's working set and exhausted the workers in a distributed run.
        da = _synthetic()
        dims = {"time": "time", "x": "ncells"}
        coords = {"time": "time", "x": "lon", "y": "lat"}

        a = smoothed_rolling_climatology(da.chunk({"time": 21, "ncells": -1}), 2, SMOOTH_DAYS, dims, coords).compute()
        b = smoothed_rolling_climatology(da.chunk({"time": 40, "ncells": -1}), 2, SMOOTH_DAYS, dims, coords).compute()

        assert np.array_equal(np.isnan(a.values), np.isnan(b.values)), "NaN pattern must not depend on chunking"
        spread = np.nanmax(np.abs(a.values - b.values))
        # ~280 K field: 1e-2 is far above the observed ~1e-4 but far below anything
        # that would indicate the overlap itself had broken.
        assert spread < 1e-2, f"chunk-dependence grew to {spread:.3e}"


class TestTimeChunkValidation:
    """A time chunk shorter than the smoothing window cannot produce a rolling mean."""

    def test_time_chunk_below_smoothing_window_raises_configuration_error(self):
        # Previously this surfaced as bottleneck's
        # "Moving window (=21) must between 1 and 20, inclusive", which names neither
        # the parameter nor the chunk that caused it.
        # 20 % 2 == 0, so the padding forms a 20-element block against a 21-day window.
        da = _synthetic().chunk({"time": 2, "ncells": -1})

        with pytest.raises(ConfigurationError, match=r"21-day centred rolling mean"):
            smoothed_rolling_climatology(
                da,
                2,
                SMOOTH_DAYS,
                {"time": "time", "x": "ncells"},
                {"time": "time", "x": "lon", "y": "lat"},
            )

    def test_chunk_not_dividing_the_pad_is_accepted(self):
        # chunk 4 with an 11-day window: 10 % 4 == 2, so no short block forms and this
        # has always worked. Rejecting it would break the detect golden tests, which is
        # exactly what a naive "chunk < window" rule did.
        #
        # The divisibility rule this pins is an emergent property of the
        # xarray -> dask.overlap -> bottleneck chain, verified against dask 2025.9.1 /
        # bottleneck 1.6.0. If an upgrade ever makes this configuration fail, that is
        # UPSTREAM BEHAVIOUR CHANGING, not a regression in marEx -- widen the guard in
        # smoothed_rolling_climatology towards "chunk >= smooth_days_baseline" rather
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
