"""
General-purpose helpers for the marEx detection pipeline.

Currently provides :func:`add_decimal_year`, a pure datetime helper used by the
harmonic detrending anomaly method. This module is a leaf in the detect package
dependency graph and imports only third-party libraries and the package logger.
"""

from typing import Optional

import pandas as pd
import xarray as xr

from ..logging_config import get_logger

# Get module logger
logger = get_logger(__name__)


def add_decimal_year(da: xr.DataArray, dim: str = "time", coord: Optional[str] = None) -> xr.DataArray:
    """
    Add decimal year coordinate to DataArray for trend analysis.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with datetime coordinate
    dim : str, optional
        Name of the time dimension
    coord : str, optional
        Name of the time coordinate (if different from dimension name)

    Returns
    -------
    xarray.DataArray
        Input data with added 'decimal_year' coordinate
    """
    # Use coordinate name if provided, otherwise use dimension name
    coord_name = coord if coord is not None else dim
    time = pd.to_datetime(da[coord_name])
    start_of_year = pd.to_datetime(time.year.astype(str) + "-01-01")
    start_of_next_year = pd.to_datetime((time.year + 1).astype(str) + "-01-01")
    year_elapsed = (time - start_of_year).days
    year_duration = (start_of_next_year - start_of_year).days

    decimal_year = time.year + year_elapsed / year_duration
    return da.assign_coords(decimal_year=(dim, decimal_year))
