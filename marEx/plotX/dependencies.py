"""Optional-dependency checks for the plotX visualisation system."""


def _check_plotting_dependencies() -> None:
    """Check if plotting dependencies are available and raise informative error if not."""
    from .._dependencies import require_dependencies

    require_dependencies(["matplotlib", "cartopy"], "Plotting functionality")
