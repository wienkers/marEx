"""
MarEx: Marine Extremes Detection and Tracking
==============================================

A Python package for efficient identification and tracking of marine extremes
such as Marine Heatwaves (MHWs).

Core Functionality
-----------------
- preprocess_data: Convert raw time series into standardised anomalies
- tracker: Identify and track extreme events through time

Example
-------
>>> import xarray as xr
>>> import marEx
>>> # Load SST data
>>> sst = xr.open_dataset('sst_data.nc').sst
>>> # Preprocess data to identify extreme events
>>> extreme_events_ds = marEx.preprocess_data(sst, threshold_percentile=95)
>>> # Track events through time
>>> tracker = marEx.tracker(extreme_events_ds.extreme_events, extreme_events_ds.mask, 
...                         R_fill=8, area_filter_quartile=0.5)
>>> events_ds = tracker.run()
"""

# Import core functionality
from .detect import (
    preprocess_data, 
    compute_normalised_anomaly,
    rechunk_for_cohorts,
    identify_extremes
)

from .track import tracker

# Import plotting utilities 
from .plotX import (
    specify_grid,
    PlotConfig
)

# Convenience variables
__all__ = [
    # Core data preprocessing
    'preprocess_data',
    'compute_normalised_anomaly',
    'rechunk_for_cohorts',
    'identify_extremes',
    
    # Tracking
    'tracker',
    
    # Visualization
    'specify_grid',
    'PlotConfig',
]

# Version information
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("marEx")
except PackageNotFoundError:
    # Package is not installed
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root="..", relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "unknown"