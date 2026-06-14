"""
Climatology, detrending, and anomaly computation.

A complete, standalone half of marEx. Use it when you want a smoothed daily
climatology, detrended anomalies, or standardised anomalies on data too large to
fit in memory -- and nothing else. There is no threshold parameter anywhere in
this package.

Nothing here is specific to any domain or variable. The same four methods apply
to sea surface temperature, 2 m air temperature, precipitation, soil moisture, or
a biogeochemical tracer, on gridded or unstructured meshes.

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
