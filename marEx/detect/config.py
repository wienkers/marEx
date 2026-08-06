"""
Configuration dataclass for the marEx preprocessing pipeline.

Provides :class:`PreprocessConfig`, a frozen dataclass that bundles the tuning
parameters of :func:`marEx.preprocess_data` into a single validated, immutable
container. The public ``preprocess_data`` keyword-argument signature is preserved
exactly; the entry point resolves its defaults and then constructs this config to
thread the configuration through the internal pipeline as one explicit argument.

This dataclass is a dumb container: it does not perform any data validation or
re-order the existing checks. It exists purely to give the internal interfaces a
clean, structured configuration object.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Immutable bundle of configuration options for :func:`preprocess_data`.

    All fields are required (no defaults): the public defaults live on the
    ``preprocess_data`` signature, which remains the single source of truth for
    the public API. The orchestrator resolves any ``None`` defaults (e.g.
    ``detrend_orders``, ``dask_chunks``) before constructing this config, so the
    values stored here are the fully-resolved ones used to drive the pipeline.

    Parameters
    ----------
    method_anomaly : str
        Anomaly computation method ('detrend_harmonic', 'shifting_baseline',
        'fixed_baseline', or 'detrend_fixed_baseline').
    method_extreme : str
        Extreme identification method ('global_extreme' or 'hobday_extreme').
    threshold_percentile : float
        Percentile threshold for extreme event detection.
    window_year_baseline : int
        Number of previous years for rolling climatology (shifting_baseline only).
    smooth_days_baseline : int
        Days for smoothing rolling climatology (shifting_baseline only).
    window_days_hobday : int
        Window size for day-of-year threshold calculation (hobday_extreme only).
    window_spatial_hobday : int, optional
        Spatial window size for the day-of-year threshold calculation
        (hobday_extreme only).
    std_normalise : bool
        Whether to standardise anomalies by rolling standard deviation
        (detrend_harmonic only).
    detrend_orders : list of int
        Polynomial orders for detrending (detrend_harmonic only).
    force_zero_mean : bool
        Whether to enforce zero mean in detrended anomalies (detrend_harmonic only).
    reference_period : tuple of (int, int), optional
        Year range (start_year, end_year) inclusive for computing the daily
        climatology (fixed_baseline and detrend_fixed_baseline only).
    method_percentile : str
        Method for percentile calculation ('exact' or 'approximate').
    precision : float
        Precision for histogram bins in approximate percentile method.
    max_anomaly : float
        Maximum anomaly value for histogram binning in the approximate method.
    dask_chunks : dict
        Chunking specification for distributed computation.
    compute_mode : str
        Materialisation policy ('persist', 'lazy', or 'streaming').
    scratch_dir : str, optional
        Directory for staged intermediates (required by 'streaming').
    validate : bool
        Whether to run the full-input finite-value validation pass.
    """

    method_anomaly: Literal["detrend_harmonic", "shifting_baseline", "fixed_baseline", "detrend_fixed_baseline"]
    method_extreme: Literal["global_extreme", "hobday_extreme"]
    threshold_percentile: float
    window_year_baseline: int
    smooth_days_baseline: int
    window_days_hobday: int
    window_spatial_hobday: Optional[int]
    std_normalise: bool
    detrend_orders: List[int]
    force_zero_mean: bool
    reference_period: Optional[Tuple[int, int]]
    method_percentile: Literal["exact", "approximate"]
    precision: float
    max_anomaly: float
    dask_chunks: Dict[str, int]
    compute_mode: Literal["persist", "lazy", "streaming"]
    scratch_dir: Optional[str]
    validate: bool
