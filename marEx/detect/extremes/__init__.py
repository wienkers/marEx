"""
Extreme-identification subpackage for the marEx detection pipeline.

Re-exports the public extremes dispatcher for convenience. The authoritative
re-export contract for the detect package is in ``marEx.detect.__init__``.
"""

from .base import identify_extremes

__all__ = [
    "identify_extremes",
]
