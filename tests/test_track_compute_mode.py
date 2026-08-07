"""compute_mode on the tracker: validation, wiring, equivalence and laziness."""

from pathlib import Path

import pytest
import xarray as xr

import marEx
from marEx.detect.compute_mode import Materialiser
from marEx.exceptions import ConfigurationError

TEST_DATA_DIR = Path(__file__).parent / "data"
CHUNK_SIZE = {"time": 2, "lat": -1, "lon": -1}
TRACKER_KWARGS = {
    "R_fill": 8,
    "area_filter_quartile": 0.5,
    "T_fill": 2,
    "allow_merging": True,
    "overlap_threshold": 0.5,
    "nn_partitioning": True,
    "quiet": True,
}


@pytest.fixture(scope="module")
def extremes():
    return xr.open_zarr(str(TEST_DATA_DIR / "extremes_gridded.zarr"), chunks={})


class TestMaterialiserIsStreaming:
    def test_is_streaming_true_only_for_streaming(self, tmp_path):
        assert Materialiser("streaming", tmp_path).is_streaming is True
        assert Materialiser("persist").is_streaming is False
        assert Materialiser("lazy").is_streaming is False


class TestTrackerComputeModeValidation:
    def test_default_is_persist(self, extremes):
        tr = marEx.tracker(extremes.extreme_events.chunk(CHUNK_SIZE), extremes.mask, **TRACKER_KWARGS)
        assert tr.compute_mode == "persist"
        assert tr.materialiser.is_streaming is False

    def test_lazy_is_rejected(self, extremes):
        with pytest.raises(ConfigurationError, match="lazy"):
            marEx.tracker(
                extremes.extreme_events.chunk(CHUNK_SIZE),
                extremes.mask,
                compute_mode="lazy",
                **TRACKER_KWARGS,
            )

    def test_unknown_mode_is_rejected(self, extremes):
        with pytest.raises(ConfigurationError):
            marEx.tracker(
                extremes.extreme_events.chunk(CHUNK_SIZE),
                extremes.mask,
                compute_mode="nonsense",
                **TRACKER_KWARGS,
            )

    def test_streaming_requires_temp_dir(self, extremes):
        with pytest.raises(ConfigurationError, match="temp_dir"):
            marEx.tracker(
                extremes.extreme_events.chunk(CHUNK_SIZE),
                extremes.mask,
                compute_mode="streaming",
                **TRACKER_KWARGS,
            )

    def test_streaming_with_temp_dir_builds_a_staging_dir(self, extremes, tmp_path):
        tr = marEx.tracker(
            extremes.extreme_events.chunk(CHUNK_SIZE),
            extremes.mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **TRACKER_KWARGS,
        )
        assert tr.materialiser.is_streaming is True
        assert tr.staging_dir is not None
        assert Path(tr.staging_dir).exists()


class TestStagingLifetime:
    @pytest.mark.slow
    def test_streaming_output_advertises_its_staging_dir(self, extremes, tmp_path, dask_client):
        tr = marEx.tracker(
            extremes.extreme_events.chunk(CHUNK_SIZE),
            extremes.mask,
            compute_mode="streaming",
            temp_dir=str(tmp_path),
            **TRACKER_KWARGS,
        )
        events = tr.run()
        staged = events.attrs.get("marex_staging_dir")
        assert staged is not None, "streaming output must advertise its staging dir"
        assert Path(staged).exists(), "staging dir must OUTLIVE run(); the result reads from it"
        # The result must still be readable -- this is the trap clear_staging-on-return causes.
        assert int(events.ID_field.max().compute()) >= 0
        marEx.clear_staging(events)
        assert not Path(staged).exists()
