"""
Rolling-climatology computation for the shifting-baseline anomaly method.

Provides :func:`rolling_climatology` and :func:`smoothed_rolling_climatology`,
both public entry points re-exported at the marEx top level. These use flox
cohorts to efficiently compute multi-year rolling day-of-year climatologies.
"""

from typing import Dict, Optional

import flox.xarray
import numpy as np
import pandas as pd
import xarray as xr
from dask import persist

from ...helper import checkpoint_to_zarr
from ...logging_config import get_logger
from ..validation import _infer_dims_coords

# Get module logger
logger = get_logger(__name__)


def rolling_climatology(
    da: xr.DataArray,
    window_year_baseline: int = 15,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    use_temp_checkpoints: bool = False,
) -> xr.DataArray:
    """
    Compute rolling climatology efficiently using flox cohorts.
    Uses the previous `window_year_baseline` years of data and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    window_year_baseline : int, default=15
        Number of years to include in each climatology window
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data

    Returns
    -------
    xarray.DataArray
        Rolling climatology with same shape as input data

    Examples
    --------
    Basic rolling climatology computation:

    >>> import xarray as xr
    >>> import marEx
    >>>
    >>> # Load 20 years of SST data
    >>> sst = xr.open_dataset('sst_data.nc', chunks={}).sst.chunk({'time': 30})
    >>>
    >>> # Compute 15-year rolling climatology
    >>> climatology = marEx.rolling_climatology(sst, window_year_baseline=15)
    >>> print(climatology.shape)
    (7305, 180, 360)  # Same as input
    >>>
    >>> # First 15 years will be NaN (insufficient history)
    >>> print(f"NaN values in first year: {climatology.isel(time=slice(0, 365)).isnull().all().compute()}")
    True

    Shorter window for datasets with limited time span:

    >>> # For datasets with only 10 years, use shorter window
    >>> short_climatology = marEx.rolling_climatology(
    ...     sst, window_year_baseline=5
    ... )
    >>> # First 5 years will be NaN instead of 15

    Processing unstructured data:

    >>> # ICON ocean model data
    >>> icon_sst = xr.open_dataset('icon_sst.nc', chunks={}).to.chunk({'time': 25})
    >>> icon_climatology = marEx.rolling_climatology(
    ...     icon_sst,
    ...     dimensions={"time": "time", "x": "ncells"}
    ...     coordinates={"time": "time", "x": "lon", "y": "lat"}
    ... )
    >>> print(icon_climatology.dims)
    Frozen({'time': 7305, 'ncells': 83886})

    Comparing with fixed climatology:

    >>> # Fixed climatology (traditional approach)
    >>> fixed_clim = sst.groupby(sst.time.dt.dayofyear).mean()
    >>>
    >>> # Rolling climatology (adaptive approach)
    >>> rolling_clim = marEx.rolling_climatology(sst)
    >>>
    >>> # Rolling climatology adapts to climate change
    >>> clim_2000 = rolling_clim.sel(time='2000').mean()
    >>> clim_2020 = rolling_clim.sel(time='2020').mean()
    >>> print(f"Climate change signal: {(clim_2020 - clim_2000).compute():.3f} °C")

    Memory considerations for large datasets:

    >>> # Ensure appropriate chunking for memory efficiency
    >>> large_sst = sst.chunk({'time': 30, 'lat': 45, 'lon': 90})
    >>> large_climatology = marEx.rolling_climatology(large_sst)
    >>> # Output maintains input chunking structure
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)
    timedim = dimensions["time"]
    time_coord = coordinates["time"]
    original_chunk_dict = dict(zip(da.dims, da.chunks))

    # Add temporal coordinates
    years = da[time_coord].dt.year
    doys = da[time_coord].dt.dayofyear
    da = da.assign_coords({"year": years, "dayofyear": doys})

    # Get temporal bounds
    years, doys = persist(years, doys)
    year_vals = years.values
    doy_vals = doys.values
    unique_years = np.unique(year_vals)
    min_year = int(unique_years.min().item())

    # Create long-form grouping variables
    # For each time point, determine which target years it contributes to
    contributing_time_indices = []
    contributing_target_years = []
    contributing_dayofyears = []

    for t_idx, (year_val, doy_val) in enumerate(zip(year_vals, doy_vals)):
        # Convert numpy scalars to Python ints to avoid dtype issues
        year_val = int(year_val)
        doy_val = int(doy_val)

        # Find target years this time point contributes to
        # A time point from year Y contributes to target years where:
        # target_year - window_year_baseline <= Y < target_year
        # Which means: Y < target_year <= Y + window_year_baseline
        candidate_targets = unique_years[(unique_years > year_val) & (unique_years <= year_val + window_year_baseline)]

        # Only include target years that have sufficient history
        valid_targets = candidate_targets[candidate_targets >= min_year + window_year_baseline]

        # Add entries for each valid target year
        n_targets = len(valid_targets)
        contributing_time_indices.extend([t_idx] * n_targets)
        contributing_target_years.extend(valid_targets.tolist())
        contributing_dayofyears.extend([doy_val] * n_targets)

    # Convert to numpy arrays with explicit dtypes
    time_indices = np.array(contributing_time_indices, dtype=np.int32)
    target_year_groups = np.array(contributing_target_years, dtype=np.int32)
    dayofyear_groups = np.array(contributing_dayofyears, dtype=np.int32)

    # Create long-form dataset by selecting the contributing time points
    long_form_data = da.isel({timedim: time_indices})

    # Create a new time dimension for the long-form data
    long_timedim = f"{timedim}_contrib"
    long_form_data = long_form_data.rename({timedim: long_timedim})

    # Convert grouping arrays to DataArrays with the correct dimension
    target_year_da = xr.DataArray(target_year_groups, dims=[long_timedim], name="target_year")
    dayofyear_da = xr.DataArray(dayofyear_groups, dims=[long_timedim], name="dayofyear")

    # Use flox with both grouping variables to compute climatologies
    climatologies = flox.xarray.xarray_reduce(
        long_form_data,
        target_year_da,
        dayofyear_da,
        dim=long_timedim,
        func="nanmean",
        expected_groups=(unique_years, np.arange(1, 367, dtype=np.int32)),
        isbin=(False, False),
        dtype=np.float32,
        fill_value=np.nan,
    ).chunk({"dayofyear": -1})

    if use_temp_checkpoints:
        logger.debug("Checkpointing climatologies to break graph dependencies")
        climatologies = checkpoint_to_zarr(climatologies, name="climatologies", timedim=timedim)

    # Create index arrays for final mapping
    year_to_idx = pd.Series(range(len(unique_years)), index=unique_years)
    year_indices = year_to_idx[year_vals].values

    # Select appropriate climatology for each time point
    result = climatologies.isel(
        target_year=xr.DataArray(year_indices, dims=[timedim]),
        dayofyear=xr.DataArray(doy_vals - 1, dims=[timedim]),
    )

    # Clean up dimensions and coordinates
    result = result.drop_vars(["target_year", "dayofyear"])

    return result.chunk(original_chunk_dict)


def smoothed_rolling_climatology(
    da: xr.DataArray,
    window_year_baseline: int = 15,
    smooth_days_baseline: int = 21,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    use_temp_checkpoints: bool = False,
) -> xr.DataArray:
    """
    Compute a smoothed rolling climatology using the previous `window_year_baseline` years of data
    and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    window_year_baseline : int, default=15
        Number of years to include in each climatology window
    smooth_days_baseline : int, default=21
        Number of days for temporal smoothing window
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data

    Returns
    -------
    xarray.DataArray
        Smoothed rolling climatology with same shape as input data

    Examples
    --------
    Basic smoothed rolling climatology:

    >>> import xarray as xr
    >>> import marEx
    >>>
    >>> # Load SST data
    >>> sst = xr.open_dataset('sst_data.nc', chunks={}).sst.chunk({'time': 30})
    >>>
    >>> # Compute smoothed rolling climatology
    >>> smooth_clim = marEx.smoothed_rolling_climatology(
    ...     sst,
    ...     window_year_baseline=15,
    ...     smooth_days_baseline=21
    ... )
    >>> print(smooth_clim.shape)
    (7305, 180, 360)

    Comparing different smoothing windows:

    >>> # Short smoothing - more day-to-day variability
    >>> clim_short = marEx.smoothed_rolling_climatology(
    ...     sst, smooth_days_baseline=7
    ... )
    >>>
    >>> # Long smoothing - smoother seasonal cycle
    >>> clim_long = marEx.smoothed_rolling_climatology(
    ...     sst, smooth_days_baseline=61
    ... )
    >>>
    >>> # Compare variability
    >>> var_short = clim_short.std(dim='time').mean().compute()
    >>> var_long = clim_long.std(dim='time').mean().compute()
    >>> print(f"Variability: short={var_short:.3f}, long={var_long:.3f}")

    Climatology for anomaly computation:

    >>> # Compute smoothed climatology then anomalies
    >>> climatology = marEx.smoothed_rolling_climatology(sst)
    >>> anomalies = sst - climatology
    >>>
    >>> # Check that anomalies have reasonable properties
    >>> print(f"Anomaly mean: {anomalies.mean().compute():.6f}")
    >>> print(f"Anomaly std: {anomalies.std().compute():.3f}")

    Unstructured data processing:

    >>> # ICON ocean data
    >>> icon_sst = xr.open_dataset('icon_sst.nc', chunks={}).to.chunk({'time': 25})
    >>> icon_smooth_clim = marEx.smoothed_rolling_climatology(
    ...     icon_sst,
    ...     dimensions={"time": "time", "x": "ncells"},
    ...     coordinates={"time": "time", "x": "lon", "y": "lat"},
    ...     window_year_baseline=10,
    ...     smooth_days_baseline=31
    ... )

    Effect of smoothing on seasonal cycle:

    >>> # Raw rolling climatology (no temporal smoothing)
    >>> raw_clim = marEx.rolling_climatology(sst, window_year_baseline=15)
    >>>
    >>> # Smoothed rolling climatology
    >>> smooth_clim = marEx.smoothed_rolling_climatology(
    ...     sst, window_year_baseline=15, smooth_days_baseline=21
    ... )
    >>>
    >>> # Compare seasonal cycle smoothness
    >>> # Extract annual cycle for a point
    >>> point_raw = raw_clim.isel(lat=90, lon=180).sel(time='2010')
    >>> point_smooth = smooth_clim.isel(lat=90, lon=180).sel(time='2010')
    >>>
    >>> print(f"Raw climatology range: {(point_raw.max() - point_raw.min()).compute():.3f}")
    >>> print(f"Smooth climatology range: {(point_smooth.max() - point_smooth.min()).compute():.3f}")

    Performance considerations:

    >>> # Efficient implementation smooths raw data first, then computes climatology
    >>> # This is more memory-efficient than smoothing the climatology
    >>> large_sst = sst.chunk({'time': 25, 'lat': 45, 'lon': 90})
    >>> efficient_clim = marEx.smoothed_rolling_climatology(large_sst)
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)
    timedim = dimensions["time"]

    # N.B.: It is more efficient (chunking-wise) to smooth the raw data rather than the climatology
    da_smoothed = (
        da.rolling({timedim: smooth_days_baseline}, center=True).mean().chunk(dict(zip(da.dims, da.chunks))).astype(np.float32)
    )

    clim = rolling_climatology(da_smoothed, window_year_baseline, dimensions, coordinates, use_temp_checkpoints)

    return clim
