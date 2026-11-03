"""
MarEx Track: 4-D (time, depth, lat, lon) event tracking.

:class:`VolumetricTracker` is what ``marEx.tracker(...)`` returns when ``dimensions``
declares a ``"z"`` entry; the other modules hold its building blocks.
"""

from .tracker4d import VolumetricTracker

__all__ = ["VolumetricTracker"]
