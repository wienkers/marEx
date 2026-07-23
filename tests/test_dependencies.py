"""Tests for the optional-dependency tracker (marEx._dependencies).

Regression coverage for §7.1 of the Fable review: the probes were stripped to
``try: pass``, so every optional dependency was unconditionally reported as
available. These tests pin that the probe reflects reality and, crucially, is
*capable of returning False*.
"""

from marEx._dependencies import DependencyTracker, has_dependency


def test_probe_reflects_reality_not_unconditional_true():
    """A module that cannot exist must probe as unavailable."""
    assert DependencyTracker._module_available("definitely_not_a_real_module_xyz") is False


def test_known_stdlib_module_available():
    """A module that always exists must probe as available."""
    assert DependencyTracker._module_available("json") is True


def test_pillow_probed_by_PIL_import_name():
    """pillow is imported as ``PIL`` — the tracker must probe that name."""
    tracker = DependencyTracker()
    # 'pillow' key exists in the tracked set regardless of install state.
    assert "pillow" in tracker._dependencies


def test_missing_dependency_flows_through_public_api():
    """has_dependency/require_dependencies/profile must react to a missing dep."""
    tracker = DependencyTracker()
    tracker._dependencies["jax"] = False

    assert tracker.has_dependency("jax") is False
    assert "jax" in tracker.get_missing_dependencies()

    import pytest

    with pytest.raises(ImportError):
        tracker.require_dependencies(["jax"], "JAX acceleration")

    # With jax missing the profile can no longer be the all-inclusive 'full'.
    assert tracker.get_installation_profile() != "full"


def test_module_level_has_dependency_callable():
    """The module-level convenience wrapper returns a bool."""
    assert isinstance(has_dependency("numpy"), bool)
