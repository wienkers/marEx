"""
Anomaly-computation subpackage for the marEx detection pipeline.

Re-exports the public anomaly dispatcher and the rolling-climatology helpers for
convenience. The authoritative re-export contract for the detect package is in
``marEx.detect.__init__``.
"""

from .base import compute_normalised_anomaly
from .climatology import rolling_climatology, smoothed_rolling_climatology

__all__ = [
    "compute_normalised_anomaly",
    "rolling_climatology",
    "smoothed_rolling_climatology",
]
