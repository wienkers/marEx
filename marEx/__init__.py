"""
marEx: scalable detection and tracking of extremes in weather and climate data.

Three independent stages, each usable on its own:

- :mod:`marEx.anomaly` -- climatology, detrending, and anomalies. A complete
  product for anyone who wants a smoothed daily climatology on larger-than-memory
  data and nothing else. No threshold parameter appears here.
- :mod:`marEx.extremes` -- percentile thresholding and binary event
  identification, on anomalies from anywhere.
- :mod:`marEx.track` -- connected-component identification and tracking through
  time, with merge/split handling.

:func:`marEx.preprocess_data` chains the first two for convenience.

Nothing in marEx is specific to a domain or a variable: the same pipeline applies
to ocean, atmosphere, land surface, and biogeochemistry, on regular grids and
unstructured meshes alike.

Example
-------
>>> import xarray as xr
>>> import marEx
>>> t2m = xr.open_dataset("t2m.zarr", chunks={"time": 25}).t2m
>>> # Anomalies only -- no detection
>>> anomalies = marEx.anomaly.compute(t2m, method="shifting_baseline")
>>> # Or the full chain, then track the events through time
>>> ds = marEx.preprocess_data(t2m, threshold_percentile=95)
>>> events_ds = marEx.tracker(ds.extreme_events, ds.mask,
...                           R_fill=8, area_filter_quartile=0.5).run()
"""

# Import the analysis packages. `anomaly` and `extremes` are peers and are exposed
# as modules, not as loose functions, so that each reads as a self-contained stage.
from . import anomaly, extremes

# Import dependency management
from ._dependencies import get_installation_profile, has_dependency, print_dependency_status
from .core import ComputeMode, clear_staging

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
from .pipeline import preprocess_data

# Import plotting utilities
from .plotX import PlotConfig, specify_grid
from .track import regional_tracker, tracker

# Coordinate validation utilities are now integrated into the main modules


# Convenience variables
__all__ = [
    # Analysis stages
    "anomaly",
    "extremes",
    # Full chain over the two stages above
    "preprocess_data",
    # Materialisation policy (compute_mode)
    "ComputeMode",
    "clear_staging",
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
