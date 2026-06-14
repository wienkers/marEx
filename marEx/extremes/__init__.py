"""
Percentile thresholding and binary extreme-event identification.

Takes anomalies -- from :mod:`marEx.anomaly` or from anywhere else -- and returns
a boolean event field plus the thresholds that defined it. The two methods are a
constant-in-time percentile per cell, and a day-of-year resolved percentile
following Hobday et al. (2016).

Neither method is specific to any domain: the same definitions apply to marine
heatwaves, atmospheric heatwaves, and extremes in any other gridded or
unstructured field.

    >>> import marEx
    >>> events = marEx.extremes.identify(anomalies, threshold_percentile=90)
"""

from .api import identify
from .base import identify_extremes
from .global_percentile import _identify_extremes_constant
from .histogram import _compute_histogram_quantile_1d, _compute_histogram_quantile_2d, _rolling_histogram_quantile
from .seasonal_percentile import _identify_extremes_seasonal

__all__ = [
    "identify",
    "identify_extremes",
]
