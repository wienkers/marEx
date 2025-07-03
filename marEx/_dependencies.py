"""Dependency management for marEx."""

import warnings
from typing import Dict, List


class DependencyTracker:
    """Tracks availability of optional dependencies"""

    def __init__(self):
        """Initialise the dependency tracker."""
        self._dependencies: Dict[str, bool] = {}
        self._check_all_dependencies()

    def _check_all_dependencies(self) -> None:
        """Check availability of all optional dependencies."""
        # Performance dependencies
        try:
            pass

            self._dependencies["jax"] = True
        except ImportError:
            self._dependencies["jax"] = False

        try:
            pass

            self._dependencies["jaxlib"] = True
        except ImportError:
            self._dependencies["jaxlib"] = False

        # HPC dependencies
        try:
            pass

            self._dependencies["dask_jobqueue"] = True
        except (ImportError, ValueError):
            # ValueError can occur when importing dask_jobqueue in worker threads
            # due to signal handling limitations
            self._dependencies["dask_jobqueue"] = False

        try:
            pass

            self._dependencies["psutil"] = True
        except ImportError:
            self._dependencies["psutil"] = False

        # Visualisation dependencies
        try:
            pass

            self._dependencies["matplotlib"] = True
        except ImportError:
            self._dependencies["matplotlib"] = False

        try:
            pass

            self._dependencies["cartopy"] = True
        except ImportError:
            self._dependencies["cartopy"] = False

        try:
            pass

            self._dependencies["seaborn"] = True
        except ImportError:
            self._dependencies["seaborn"] = False

        try:
            pass

            self._dependencies["cmocean"] = True
        except ImportError:
            self._dependencies["cmocean"] = False

        # PIL for image processing
        try:
            pass

            self._dependencies["pillow"] = True
        except ImportError:
            self._dependencies["pillow"] = False

    def has_dependency(self, dep_name: str) -> bool:
        """Check if a specific dependency is available."""
        return self._dependencies.get(dep_name, False)

    def require_dependencies(self, dependencies: List[str], feature: str = "This functionality") -> None:
        """
        Require specific dependencies for a feature.

        Parameters
        ----------
        dependencies : List[str]
            List of required dependency names
        feature : str
            Description of the feature requiring the dependencies

        Raises
        ------
        ImportError
            If any required dependencies are missing
        """
        missing = [dep for dep in dependencies if not self.has_dependency(dep)]

        if missing:
            if len(missing) == 1:
                dep_name = missing[0]
                install_cmd = self._get_install_command(dep_name)
                raise ImportError(f"{feature} requires {dep_name}. " f"Install with: {install_cmd}")
            else:
                dep_list = ", ".join(missing)
                install_cmd = "pip install marEx[full]"
                raise ImportError(f"{feature} requires the following dependencies: {dep_list}. " f"Install with: {install_cmd}")

    def warn_missing_dependency(self, dep_name: str, feature: str = "Some functionality") -> None:
        """
        Issue a warning for a missing optional dependency.

        Parameters
        ----------
        dep_name : str
            Name of the missing dependency
        feature : str
            Description of affected functionality
        """
        if not self.has_dependency(dep_name):
            install_cmd = self._get_install_command(dep_name)
            warnings.warn(
                f"{dep_name} not installed. {feature} will be slower or unavailable. "
                f"For best performance, install with: {install_cmd}",
                ImportWarning,
                stacklevel=3,
            )

    def _get_install_command(self, dep_name: str) -> str:
        """Get appropriate install command for a dependency."""
        specific_commands = {
            "jax": "pip install marEx[performance]",
            "jaxlib": "pip install marEx[performance]",
            "dask_jobqueue": "pip install marEx[hpc]",
            "pillow": "pip install pillow",
            "matplotlib": "pip install marEx[plotting]",
            "cartopy": "pip install marEx[plotting]",
            "seaborn": "pip install marEx[plotting]",
            "cmocean": "pip install marEx[plotting]",
        }

        return specific_commands.get(dep_name, f"pip install {dep_name}")

    def get_missing_dependencies(self) -> List[str]:
        """Get list of missing optional dependencies."""
        return [dep for dep, available in self._dependencies.items() if not available]

    def get_installation_profile(self) -> str:
        """Determine the current installation profile based on available dependencies."""
        profiles = {
            "minimal": [],
            "performance": ["jax", "jaxlib"],
            "hpc": ["dask_jobqueue", "psutil"],
            "plotting": ["seaborn", "cmocean"],
            "full": ["jax", "jaxlib", "dask_jobqueue", "seaborn", "cmocean"],
        }

        # Check which profiles are satisfied
        satisfied_profiles = []
        for profile, required_deps in profiles.items():
            if all(self.has_dependency(dep) for dep in required_deps):
                satisfied_profiles.append(profile)

        # Return the most comprehensive profile
        profile_hierarchy = ["full", "plotting", "hpc", "performance", "minimal"]
        for profile in profile_hierarchy:
            if profile in satisfied_profiles:
                return profile

        return "minimal"

    def print_dependency_status(self) -> None:
        """Print status of all tracked dependencies."""
        current_profile = self.get_installation_profile()
        print("marEx Dependency Status:")
        print("-" * 40)
        print(f"Current Profile: {current_profile}")

        # Group dependencies by category
        categories = {
            "Performance": ["jax", "jaxlib"],
            "HPC": ["dask_jobqueue", "psutil"],
            "Visualisation": ["matplotlib", "cartopy", "seaborn", "cmocean"],
            "Core": ["pillow"],
        }

        for category, deps in categories.items():
            print(f"\n{category}:")
            for dep in deps:
                if dep in self._dependencies:
                    status = "✓ Available" if self._dependencies[dep] else "✗ Missing"
                    print(f"  {dep:15} {status}")

        missing = self.get_missing_dependencies()
        if missing:
            print("\nInstallation suggestions:")
            print("  All features:     pip install marEx[full]")
            print("  Performance:      pip install marEx[performance]")
            print("  HPC:              pip install marEx[hpc]")
            print("  Visualisation:    pip install marEx[plotting]")


# Global dependency tracker instance
_dependency_tracker = DependencyTracker()


# Convenience functions
def has_dependency(dep_name: str) -> bool:
    """Check if a dependency is available."""
    return _dependency_tracker.has_dependency(dep_name)


def require_dependencies(dependencies: List[str], feature: str = "This functionality") -> None:
    """Require specific dependencies for a feature."""
    _dependency_tracker.require_dependencies(dependencies, feature)


def warn_missing_dependency(dep_name: str, feature: str = "Some functionality") -> None:
    """Issue a warning for a missing optional dependency."""
    _dependency_tracker.warn_missing_dependency(dep_name, feature)


def print_dependency_status() -> None:
    """Print status of all tracked dependencies."""
    _dependency_tracker.print_dependency_status()


def get_installation_profile() -> str:
    """Get the current installation profile."""
    return _dependency_tracker.get_installation_profile()


# Check and warn about missing optional dependencies on import
def _check_optional_dependencies_on_import():
    """Check optional dependencies and issue warnings on package import."""
    if not _dependency_tracker.has_dependency("jax"):
        _dependency_tracker.warn_missing_dependency("jax", "Some operations")


# Run the check when this module is imported
_check_optional_dependencies_on_import()
