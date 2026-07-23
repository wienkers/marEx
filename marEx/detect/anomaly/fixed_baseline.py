"""
Fixed-baseline anomaly methods.

Provides :func:`_compute_anomaly_fixed_baseline` (simple daily climatology) and
:func:`_compute_anomaly_detrend_fixed_baseline` (polynomial detrending followed by
a fixed daily climatology). These back the ``fixed_baseline`` and
``detrend_fixed_baseline`` anomaly methods respectively.
"""

from typing import Dict, List, Optional, Tuple

import flox.xarray
import numpy as np
import xarray as xr

from ...exceptions import ConfigurationError
from ...logging_config import get_logger
from ..validation import _infer_dims_coords
from .harmonic import _compute_anomaly_detrended

# Get module logger
logger = get_logger(__name__)


def _compute_anomaly_fixed_baseline(
    da: xr.DataArray,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    reference_period: Optional[Tuple[int, int]] = None,
) -> xr.Dataset:
    """
    Compute anomalies using fixed baseline method with full time series climatology.

    This method computes a daily climatology using all available years in the dataset
    (or a specified reference period), then subtracts this climatology from the
    original data to obtain anomalies.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    reference_period : tuple of (int, int), optional
        Year range (start_year, end_year) inclusive for computing the daily climatology.
        If None (default), uses all available years. Anomalies are still computed for
        the full time series.

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    # Select data for climatology computation (optionally restricted to reference period)
    if reference_period is not None:
        start_year, end_year = reference_period
        if start_year > end_year:
            raise ConfigurationError(
                f"Invalid reference_period: start year ({start_year}) must be <= end year ({end_year})",
                details="The reference_period tuple must be (start_year, end_year) with start_year <= end_year",
                suggestions=[f"Swap the order: use reference_period=({end_year}, {start_year})"],
            )
        years = da[coordinates["time"]].dt.year
        year_mask = (years >= start_year) & (years <= end_year)
        da_for_clim = da.isel({dimensions["time"]: year_mask})
        if da_for_clim.sizes[dimensions["time"]] == 0:
            data_min_year = int(years.min().values)
            data_max_year = int(years.max().values)
            raise ConfigurationError(
                f"No data found in reference_period ({start_year}, {end_year})",
                details=f"Dataset spans {data_min_year}-{data_max_year} but no timesteps fall within the specified period",
                suggestions=[
                    f"Adjust reference_period to overlap with data range ({data_min_year}-{data_max_year})",
                    "Set reference_period=None to use the full time series",
                ],
            )
        logger.debug(
            f"Using reference_period ({start_year}-{end_year}): "
            f"{da_for_clim.sizes[dimensions['time']]} of {da.sizes[dimensions['time']]} timesteps"
        )
    else:
        da_for_clim = da

    # Compute daily climatology using flox for efficiency
    logger.debug("Computing daily climatology across %s", "reference period" if reference_period else "all years")
    daily_climatology = flox.xarray.xarray_reduce(
        da_for_clim,
        da_for_clim[coordinates["time"]].dt.dayofyear,
        dim=dimensions["time"],
        func="nanmean",
        isbin=False,
        method="cohorts",
        dtype=np.float32,
    )

    # Ensure the climatology spans the full day-of-year range 1..366. If the reference
    # period contains no leap year it only has 365 groups, and subtracting it from a
    # full series that does include 29 Feb (day-of-year 366) would silently NaN every
    # such day. Reindex to 366 and forward-fill the missing tail group from day 365.
    # In the common (leap-containing) case both operations are no-ops. The dayofyear
    # dim is rechunked to a single chunk so the dask ffill is valid.
    daily_climatology = daily_climatology.reindex(dayofyear=np.arange(1, 367)).chunk({"dayofyear": -1}).ffill("dayofyear").persist()

    # Compute anomalies by subtracting daily climatology from original data
    logger.debug("Computing anomalies by subtracting daily climatology")
    da = da.assign_coords(dayofyear=da[coordinates["time"]].dt.dayofyear)
    anomalies = da.groupby(dayofyear=xr.groupers.UniqueGrouper(labels=np.arange(1, 367))) - daily_climatology
    anomalies = anomalies.astype(np.float32)

    # Drop dayofyear coordinate to avoid merge conflicts
    if "dayofyear" in anomalies.coords:
        anomalies = anomalies.drop_vars("dayofyear")

    # Create ocean/land mask from first time step
    # Handle both spatial (3D) and time-series (1D) data
    spatial_dims = [dim for dim in ["x", "y"] if dim in dimensions]
    if spatial_dims:
        # Spatial data - create 2D/3D mask.
        # `da` gained a per-timestep ``dayofyear`` coord above; dropping only the time
        # coord would leak a scalar ``dayofyear`` into the mask (and the output schema
        # under global_extreme). Drop both.
        chunk_dict_mask = {dimensions[dim]: -1 for dim in spatial_dims}
        coords_to_drop = [coordinates["time"]]
        if "dayofyear" in da.coords:
            coords_to_drop.append("dayofyear")
        mask = np.isfinite(da.isel({dimensions["time"]: 0})).drop_vars(coords_to_drop).chunk(chunk_dict_mask)
    else:
        # 1D time series - create scalar mask indicating if any finite values exist
        mask = xr.DataArray(np.any(np.isfinite(da.values)), dims=[], attrs={"description": "Time series validity mask"})

    # Build output dataset
    return xr.Dataset({"dat_anomaly": anomalies, "mask": mask})


def _compute_anomaly_detrend_fixed_baseline(
    da: xr.DataArray,
    detrend_orders: Optional[List[int]] = None,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    force_zero_mean: bool = True,
    reference_period: Optional[Tuple[int, int]] = None,
) -> xr.Dataset:
    """
    Compute anomalies using fixed detrended baseline method.

    This method first removes polynomial trends (without harmonics) from the data,
    then removes a daily climatology from the detrended signal. The trend removal
    always uses the full time series; only the climatology step respects reference_period.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    detrend_orders : list, optional
        Polynomial orders for trend removal (default: [1] for linear)
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    force_zero_mean : bool, default=True
        Whether to enforce zero mean in detrended data
    reference_period : tuple of (int, int), optional
        Year range (start_year, end_year) inclusive for computing the daily climatology.
        If None (default), uses all available years. Only affects the climatology step,
        not the polynomial detrending.

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    logger.debug(f"Removing polynomial trends of orders: {detrend_orders}")

    # Step 1: Remove polynomial trends (without harmonics) using _compute_anomaly_detrended
    detrended_result = _compute_anomaly_detrended(
        da=da,
        std_normalise=False,
        detrend_orders=detrend_orders,
        dimensions=dimensions,
        coordinates=coordinates,
        force_zero_mean=force_zero_mean,
        remove_harmonics=False,  # Only remove trends, not harmonics
    )["dat_anomaly"].persist()

    # Step 2: Compute daily climatology and anomalies using _compute_anomaly_fixed_baseline
    logger.debug("Computing daily climatology and anomalies from detrended data")
    final_result = _compute_anomaly_fixed_baseline(
        da=detrended_result,
        dimensions=dimensions,
        coordinates=coordinates,
        reference_period=reference_period,
    )

    return final_result
