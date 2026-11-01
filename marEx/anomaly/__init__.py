"""
Climatology, detrending, and anomaly computation.

This module computes smoothed climatologies, detrended anomalies, and
standardised anomalies on data too large to fit in memory, and can be used on
its own. Its four methods apply equally to sea surface temperature, 2 m air
temperature, precipitation, soil moisture, or a biogeochemical tracer, on
gridded or unstructured meshes.

    >>> import marEx
    >>> ds = marEx.anomaly.compute(t2m, method="shifting_baseline")
    >>> clim = marEx.anomaly.smoothed_rolling_climatology(t2m)
"""

from .api import compute
from .base import compute_normalised_anomaly
from .climatology import rolling_climatology, smoothed_rolling_climatology
from .fixed_baseline import _compute_anomaly_detrend_fixed_baseline, _compute_anomaly_fixed_baseline
from .harmonic import _compute_anomaly_detrended
from .shifting_baseline import _compute_anomaly_shifting_baseline

__all__ = [
    "compute",
    "compute_normalised_anomaly",
    "rolling_climatology",
    "smoothed_rolling_climatology",
]
