"""
MarEx-Detect: Marine Extremes Detection Module

Preprocessing toolkit for marine extremes identification from scalar oceanographic data.
Converts raw time series into standardised anomalies and identifies extreme events
(e.g., Marine Heatwaves using Sea Surface Temperature).

Core capabilities:

* Two preprocessing methodologies: Detrended Baseline and Shifting Baseline
* Two definitions for extreme events: Global Extreme and Hobday Extreme
* Threshold-based extreme event identification
* Efficient processing of both structured (gridded) and unstructured data

Compatible data formats:

* Structured data: 3D arrays (time, lat, lon)
* Unstructured data: 2D arrays (time, cell)

This package preserves the historical ``marEx.detect.*`` import surface: in
addition to the public API, several private helpers are re-exported here because
the test suite reaches them directly via ``import marEx.detect as detect``.
"""

import logging

# Public anomaly API (dispatcher + climatology helpers)
from .anomaly.base import compute_normalised_anomaly
from .anomaly.climatology import rolling_climatology, smoothed_rolling_climatology

# Concrete anomaly methods (private, re-exported for completeness)
from .anomaly.fixed_baseline import _compute_anomaly_detrend_fixed_baseline, _compute_anomaly_fixed_baseline
from .anomaly.harmonic import _compute_anomaly_detrended
from .anomaly.shifting_baseline import _compute_anomaly_shifting_baseline

# Materialisation policy (compute_mode) for larger-than-memory runs
from .compute_mode import ComputeMode, Materialiser, clear_staging, create_staging_dir

# Public extremes API (dispatcher)
from .extremes.base import identify_extremes

# Concrete extremes methods and histogram kernels (private, re-exported for tests)
from .extremes.global_extreme import _identify_extremes_constant
from .extremes.histogram import _compute_histogram_quantile_1d, _compute_histogram_quantile_2d, _rolling_histogram_quantile
from .extremes.hobday import _identify_extremes_hobday

# Public pipeline API + private metadata helper (re-exported for tests)
from .pipeline import _get_preprocessing_steps, preprocess_data

# Pure helpers (add_decimal_year is public-by-test; re-exported for tests)
from .utils import add_decimal_year

# Validation helpers (private, re-exported for completeness)
from .validation import _infer_dims_coords, _validate_coordinates_exist, _validate_data_values, _validate_dimensions_exist

# Suppress noisy distributed logging (preserves the import-time side effect that
# previously lived in marEx/detect.py).
logging.getLogger("distributed.shuffle._scheduler_plugin").setLevel(logging.ERROR)

__all__ = [
    # Public API
    "preprocess_data",
    "compute_normalised_anomaly",
    "identify_extremes",
    "rolling_climatology",
    "smoothed_rolling_climatology",
    # Materialisation policy
    "ComputeMode",
    "Materialiser",
    "clear_staging",
    "create_staging_dir",
    # Public-by-test helper
    "add_decimal_year",
]
