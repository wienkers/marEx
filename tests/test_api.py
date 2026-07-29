"""
The public API surface after the anomaly/extremes split.

This file is what mechanically enforces two decisions that are otherwise only
prose. First, the clean break: the old ``marEx.detect`` paths and the loose
top-level re-exports are gone, and nothing should quietly reintroduce them.
Second, and more importantly, ``marEx.anomaly.compute`` must never grow a
threshold parameter -- the whole point of splitting the package was that a user
who wants a climatology never has to think about extreme detection.

Without these assertions the suite stays green if someone re-adds a shim.

It also pins the optional-dependency probe (it must be able to say no).
"""

import importlib
import importlib.util
import inspect

import pytest

import marEx
from marEx._dependencies import DependencyTracker, has_dependency

from .conftest import assert_no_failures, check


def test_entry_points_are_peer_modules():
    """The two stages and the chainer over them."""
    # The three entry points exist.
    assert callable(marEx.anomaly.compute)
    assert callable(marEx.extremes.identify)
    assert callable(marEx.preprocess_data)

    # They are peers, so they are exposed as namespaces rather than as loose
    # functions hanging off the package root.
    assert "anomaly" in marEx.__all__
    assert "extremes" in marEx.__all__
    assert inspect.ismodule(marEx.anomaly)
    assert inspect.ismodule(marEx.extremes)

    # The climatology helpers live under the anomaly stage.
    assert callable(marEx.anomaly.rolling_climatology)
    assert callable(marEx.anomaly.smoothed_rolling_climatology)
    assert callable(marEx.anomaly.compute_normalised_anomaly)
    assert callable(marEx.extremes.identify_extremes)


def test_anomaly_stage_has_no_detection_parameter():
    """The load-bearing guarantee of the whole reorganisation."""
    params = inspect.signature(marEx.anomaly.compute).parameters

    def no_such_parameter(forbidden):
        offending = [name for name in params if forbidden in name.lower()]
        assert not offending, (
            f"marEx.anomaly.compute grew a '{forbidden}' parameter ({offending}). "
            "The anomaly stage must stay usable without any notion of detection."
        )

    failures = []
    for forbidden in ("threshold", "percentile", "extreme"):
        check(failures, f"forbidden={forbidden}", lambda forbidden=forbidden: no_such_parameter(forbidden))
    assert_no_failures(failures)

    # The corollary: the anomaly stage owns its own materialisation policy. If
    # these move out to the chainer, an anomaly-only user silently loses the
    # streaming path, which is the capability the split exists to expose.
    for required in ("compute_mode", "scratch_dir", "validate", "dask_chunks"):
        assert required in params, f"marEx.anomaly.compute lost '{required}'"


def test_removed_paths_fail_loudly():
    """No shims. Removed paths must fail loudly, not silently resolve."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("marEx.detect")

    def is_gone(removed):
        assert not hasattr(marEx, removed), (
            f"marEx.{removed} is back. The reorganisation was a clean break: "
            f"this name now lives under marEx.anomaly or marEx.extremes."
        )
        assert removed not in marEx.__all__

    failures = []
    for removed in (
        "compute_normalised_anomaly",
        "identify_extremes",
        "rolling_climatology",
        "smoothed_rolling_climatology",
        "detect",
    ):
        check(failures, f"removed={removed}", lambda removed=removed: is_gone(removed))
    assert_no_failures(failures)


def test_method_vocabulary():
    """The generalised method names replaced the marine-heatwave ones outright."""
    methods = marEx.extremes.api.METHODS
    assert set(methods) == {"seasonal_percentile", "global_percentile"}
    assert not any("hobday" in m for m in methods)

    # These were already domain-neutral and deliberately did not move.
    assert set(marEx.anomaly.api.METHODS) == {
        "shifting_baseline",
        "detrend_harmonic",
        "fixed_baseline",
        "detrend_fixed_baseline",
    }

    params = inspect.signature(marEx.preprocess_data).parameters
    for expected in ("window_days", "window_spatial", "window_years", "smooth_days", "standardise"):
        assert expected in params, f"preprocess_data is missing '{expected}'"
    for gone in ("window_days_hobday", "window_spatial_hobday", "window_year_baseline", "std_normalise"):
        assert gone not in params, f"preprocess_data still carries the old name '{gone}'"

    # smooth_max_missing was appended without moving any positional argument,
    # so an existing positional call binds as before.
    names = list(params)
    assert names[-1] == "smooth_max_missing", f"the last preprocess_data parameter is '{names[-1]}'"
    assert names[names.index("smooth_days") + 1] == "window_days", "a parameter was inserted after smooth_days"


def test_optional_dependency_probe_can_say_no():
    """The probe once reported every optional dependency as present; it must be able to return False."""
    # A module that cannot exist must probe as unavailable, and one that always exists as available.
    assert DependencyTracker._module_available("definitely_not_a_real_module_xyz") is False
    assert DependencyTracker._module_available("json") is True

    # Every tracked dependency answers as reality does: present exactly when its import name has a
    # spec. pillow is imported as ``PIL``, so probing the wrong name would report it missing. The
    # import names are written out here rather than read from the tracker, so a wrong probe name
    # in the tracker cannot make the expectation wrong in the same way.
    import_names = {
        "jax": "jax",
        "jaxlib": "jaxlib",
        "dask_jobqueue": "dask_jobqueue",
        "psutil": "psutil",
        "matplotlib": "matplotlib",
        "cartopy": "cartopy",
        "seaborn": "seaborn",
        "cmocean": "cmocean",
        "pillow": "PIL",
    }
    tracker = DependencyTracker()

    def agrees_with_reality(name, module):
        expected = importlib.util.find_spec(module) is not None
        assert has_dependency(name) is expected, (
            f"has_dependency({name!r}) is {has_dependency(name)}, import of {module!r} is {expected}"
        )
        assert tracker.has_dependency(name) is expected, (
            f"a fresh tracker reports {name!r} as {tracker.has_dependency(name)}, not {expected}"
        )

    failures = []
    for name, module in import_names.items():
        check(failures, f"dependency={name}", lambda name=name, module=module: agrees_with_reality(name, module))
    assert_no_failures(failures)
    assert has_dependency("a_name_that_is_not_tracked") is False

    # A missing dependency flows through the whole public API.
    tracker._dependencies["jax"] = False

    assert tracker.has_dependency("jax") is False
    assert "jax" in tracker.get_missing_dependencies()

    with pytest.raises(ImportError):
        tracker.require_dependencies(["jax"], "JAX acceleration")

    # With jax missing the profile can no longer be the all-inclusive 'full'.
    assert tracker.get_installation_profile() != "full"
