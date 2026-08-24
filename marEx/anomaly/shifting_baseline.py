"""
Shifting-baseline anomaly method.

Provides :func:`_compute_anomaly_shifting_baseline`, which subtracts a smoothed
multi-year rolling day-of-year climatology from the input to form anomalies.
This is the implementation behind the ``shifting_baseline`` anomaly method.
"""

from typing import Dict, Optional

import numpy as np
import xarray as xr

from ..core.dimensions import spatial_dims
from ..core.validation import _infer_dims_coords
from ..logging_config import get_logger
from .climatology import smoothed_rolling_climatology

# Get module logger
logger = get_logger(__name__)


def _compute_anomaly_shifting_baseline(
    da: xr.DataArray,
    window_years: int = 15,
    smooth_days: int = 21,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
) -> xr.Dataset:
    """
    Compute anomalies using shifting baseline method with smoothed rolling climatology.

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    # Compute smoothed rolling climatology
    climatology_smoothed = smoothed_rolling_climatology(da, window_years, smooth_days, dimensions, coordinates)

    # Compute anomaly as difference from climatology
    anomalies = da - climatology_smoothed

    # Create valid-data mask from first time step.
    # Handle both spatial (2D/3D/4D) and time-series (1D) data, matching the two
    # sibling methods. Without the 1D branch a bare time series raises here.
    if spatial_dims(da, dimensions):
        mask = np.isfinite(da.isel({dimensions["time"]: 0})).drop_vars({coordinates["time"]})
    else:
        # 1D time series - scalar mask indicating whether any finite values exist
        mask = xr.DataArray(np.any(np.isfinite(da.values)), dims=[], attrs={"description": "Time series validity mask"})

    # Build output dataset
    return xr.Dataset({"dat_anomaly": anomalies, "mask": mask})
