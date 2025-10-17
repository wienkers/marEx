"""
MarEx: Marine Extremes Detection and Tracking
==============================================

A Python package for efficient identification and tracking of marine extremes
such as Marine Heatwaves (MHWs).

Core Functionality
-----------------
- `detect`: Convert raw time series into standardised anomalies
- `track`: Identify and track extreme events through time

Example
-------
>>> import xarray as xr
>>> import marEx
>>> # Load SST data
>>> sst = xr.open_dataset('sst_data.nc').sst
>>> # Preprocess data to identify extreme events
>>> extreme_events_ds = marEx.preprocess_data(sst, threshold_percentile=95)
>>> # Track events through time
>>> events_ds = marEx.tracker(extreme_events_ds.extreme_events, extreme_events_ds.mask,
...                         R_fill=8, area_filter_quartile=0.5).run()
"""

# Initialize coverage for subprocesses if needed - MUST BE FIRST
try:
    from . import _coverage_init
except ImportError:
    pass

# Import dependency management
from ._dependencies import get_installation_profile, has_dependency, print_dependency_status

# Import core functionality
from .detect import (
    compute_normalised_anomaly,
    identify_extremes,
    preprocess_data,
    rolling_climatology,
    smoothed_rolling_climatology,
)

# Import exception hierarchy
from .exceptions import (  # Main exception hierarchy; Convenience constructors
    ConfigurationError,
    CoordinateError,
    DataValidationError,
    DependencyError,
    MarExError,
    ProcessingError,
    TrackingError,
    VisualisationError,
    create_coordinate_error,
    create_data_validation_error,
    create_processing_error,
    wrap_exception,
)

# Import HPC helper utilities
from .helper import configure_dask

# Import logging configuration functions
from .logging_config import (
    configure_logging,
    get_logger,
    get_verbosity_level,
    is_quiet_mode,
    is_verbose_mode,
    set_normal_logging,
    set_quiet_mode,
    set_verbose_mode,
)

# Import plotting utilities
from .plotX import PlotConfig, specify_grid
from .track import regional_tracker, tracker

# Coordinate validation utilities are now integrated into the main modules


# Convenience variables
__all__ = [
    # Core data preprocessing
    "preprocess_data",
    "compute_normalised_anomaly",
    "smoothed_rolling_climatology",
    "rolling_climatology",
    "identify_extremes",
    # Tracking
    "tracker",
    "regional_tracker",
    # Visualisation
    "specify_grid",
    "PlotConfig",
    # Exception hierarchy
    "MarExError",
    "DataValidationError",
    "CoordinateError",
    "ProcessingError",
    "ConfigurationError",
    "DependencyError",
    "TrackingError",
    "VisualisationError",
    "create_data_validation_error",
    "create_coordinate_error",
    "create_processing_error",
    "wrap_exception",
    # Dependency management
    "has_dependency",
    "print_dependency_status",
    "get_installation_profile",
    # Logging configuration
    "configure_logging",
    "set_verbose_mode",
    "set_quiet_mode",
    "set_normal_logging",
    "get_verbosity_level",
    "is_verbose_mode",
    "is_quiet_mode",
    "get_logger",
    # HPC helper utilities
    "configure_dask",
]

# Version information
from importlib.metadata import version

try:
    __version__ = version("marEx")
except ImportError:
    # Package is not installed
    __version__ = "unknown"
