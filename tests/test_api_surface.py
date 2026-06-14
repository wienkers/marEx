"""
The public API surface after the anomaly/extremes split.

This file is what mechanically enforces two decisions that are otherwise only
prose. First, the clean break: the old ``marEx.detect`` paths and the loose
top-level re-exports are gone, and nothing should quietly reintroduce them.
Second, and more importantly, ``marEx.anomaly.compute`` must never grow a
threshold parameter -- the whole point of splitting the package was that a user
who wants a climatology never has to think about extreme detection.

Without these assertions the suite stays green if someone re-adds a shim.
"""

import importlib
import inspect

import pytest

import marEx


class TestPeerEntryPoints:
    """The two stages and the chainer over them."""

    def test_the_three_entry_points_exist(self):
        assert callable(marEx.anomaly.compute)
        assert callable(marEx.extremes.identify)
        assert callable(marEx.preprocess_data)

    def test_stages_are_exposed_as_modules(self):
        # They are peers, so they are exposed as namespaces rather than as loose
        # functions hanging off the package root.
        assert "anomaly" in marEx.__all__
        assert "extremes" in marEx.__all__
        assert inspect.ismodule(marEx.anomaly)
        assert inspect.ismodule(marEx.extremes)

    def test_climatology_helpers_live_under_anomaly(self):
        assert callable(marEx.anomaly.rolling_climatology)
        assert callable(marEx.anomaly.smoothed_rolling_climatology)
        assert callable(marEx.anomaly.compute_normalised_anomaly)
        assert callable(marEx.extremes.identify_extremes)


class TestAnomalyStageHasNoThresholdConcept:
    """The load-bearing guarantee of the whole reorganisation."""

    @pytest.mark.parametrize("forbidden", ["threshold", "percentile", "extreme"])
    def test_compute_has_no_detection_parameter(self, forbidden):
        params = inspect.signature(marEx.anomaly.compute).parameters
        offending = [name for name in params if forbidden in name.lower()]
        assert not offending, (
            f"marEx.anomaly.compute grew a '{forbidden}' parameter ({offending}). "
            "The anomaly stage must stay usable without any notion of detection."
        )

    def test_compute_still_offers_the_larger_than_memory_controls(self):
        # The corollary: the anomaly stage owns its own materialisation policy. If
        # these move out to the chainer, an anomaly-only user silently loses the
        # streaming path, which is the capability the split exists to expose.
        params = inspect.signature(marEx.anomaly.compute).parameters
        for required in ("compute_mode", "scratch_dir", "validate", "dask_chunks"):
            assert required in params, f"marEx.anomaly.compute lost '{required}'"


class TestCleanBreak:
    """No shims. Removed paths must fail loudly, not silently resolve."""

    def test_detect_module_is_gone(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("marEx.detect")

    @pytest.mark.parametrize(
        "removed",
        [
            "compute_normalised_anomaly",
            "identify_extremes",
            "rolling_climatology",
            "smoothed_rolling_climatology",
            "detect",
        ],
    )
    def test_top_level_re_exports_are_gone(self, removed):
        assert not hasattr(marEx, removed), (
            f"marEx.{removed} is back. The reorganisation was a clean break: "
            f"this name now lives under marEx.anomaly or marEx.extremes."
        )
        assert removed not in marEx.__all__


class TestRenamedVocabulary:
    """The generalised method names replaced the marine-heatwave ones outright."""

    def test_extreme_methods_are_domain_neutral(self):
        methods = marEx.extremes.api.METHODS
        assert set(methods) == {"seasonal_percentile", "global_percentile"}
        assert not any("hobday" in m for m in methods)

    def test_anomaly_methods_unchanged(self):
        # These were already domain-neutral and deliberately did not move.
        assert set(marEx.anomaly.api.METHODS) == {
            "shifting_baseline",
            "detrend_harmonic",
            "fixed_baseline",
            "detrend_fixed_baseline",
        }

    def test_chainer_uses_the_new_window_names(self):
        params = inspect.signature(marEx.preprocess_data).parameters
        for expected in ("window_days", "window_spatial", "window_years", "smooth_days", "standardise"):
            assert expected in params, f"preprocess_data is missing '{expected}'"
        for gone in ("window_days_hobday", "window_spatial_hobday", "window_year_baseline", "std_normalise"):
            assert gone not in params, f"preprocess_data still carries the old name '{gone}'"
