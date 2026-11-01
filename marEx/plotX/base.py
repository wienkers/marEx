"""Backward-compatibility shim for the plotX base module.

The original ``base.py`` has been decomposed into focused modules:

- :mod:`marEx.plotX.config` — :class:`PlotConfig`
- :mod:`marEx.plotX.plotter_base` — :class:`PlotterBase`
- :mod:`marEx.plotX.animation` — :func:`make_frame` and the animation orchestrator
- :mod:`marEx.plotX.validation` — dimension/coordinate validation helpers
- :mod:`marEx.plotX.dependencies` — optional-dependency checks

This module re-exports the public names so existing imports such as
``from marEx.plotX.base import PlotterBase`` and ``from .base import PlotConfig``
continue to work unchanged.
"""

from .animation import make_frame
from .config import PlotConfig
from .dependencies import _check_plotting_dependencies
from .plotter_base import PlotterBase
from .validation import _validate_coordinates_exist, _validate_dimensions_exist

__all__ = [
    "PlotConfig",
    "PlotterBase",
    "make_frame",
    "_check_plotting_dependencies",
    "_validate_dimensions_exist",
    "_validate_coordinates_exist",
]
