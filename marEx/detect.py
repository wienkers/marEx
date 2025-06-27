"""
MarEx-Detect: Marine Extremes Detection Module

Preprocessing toolkit for marine extremes identification from scalar oceanographic data.
Converts raw time series into standardised anomalies and identifies extreme events
(e.g., Marine Heatwaves using Sea Surface Temperature).

Core capabilities:
- Two preprocessing methodologies: Detrended Baseline and Shifting Baseline
- Two definitions for extreme events: Global Extreme and Hobday Extreme
- Threshold-based extreme event identification 
- Efficient processing of both structured (gridded) and unstructured data

Compatible data formats:
- Structured data:   3D arrays (time, lat, lon)
- Unstructured data: 2D arrays (time, cell)
"""

import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask import delayed
from dask.base import is_dask_collection
import flox.xarray
from xhistogram.xarray import histogram
from numpy.lib.stride_tricks import sliding_window_view
import logging
from .helper import fix_dask_tuple_array
from typing import Dict, List, Tuple, Optional, Union, Literal, Any
from numpy.typing import NDArray

logging.getLogger('distributed.shuffle._scheduler_plugin').setLevel(logging.ERROR)


# ============================
# Methodology Selection
# ============================

def preprocess_data(
    da: xr.DataArray,
    method_anomaly: Literal['detrended_baseline', 'shifting_baseline'] = 'detrended_baseline',
    method_extreme: Literal['global_extreme', 'hobday_extreme'] = 'global_extreme',
    threshold_percentile: float = 95,
    window_year_baseline: int = 15,      # for shifting_baseline
    smooth_days_baseline: int = 21,      # "
    window_days_hobday: int = 11,        # for hobday_extreme
    std_normalise: bool = False,         # for detrended_baseline
    detrend_orders: List[int] = [1],     # "
    force_zero_mean: bool = True,        # "
    exact_percentile: bool = False,
    dask_chunks: Dict[str, int] = {'time': 25},
    dimensions: Dict[str, str] = {'time':'time', 'xdim':'lon'},
    neighbours: Optional[xr.DataArray] = None,
    cell_areas: Optional[xr.DataArray] = None
) -> xr.Dataset:
    """
    Complete preprocessing pipeline for marine extreme event identification.
    
    Supports separate methods for anomaly computation and extreme identification:
    
    Anomaly Methods:
    - 'detrended_baseline': Detrending with harmonics and polynomials -- more efficient, but biases statistics
    - 'shifting_baseline': Rolling climatology using previous window_year_baseline years -- more "correct", but shortens time series by window_year_baseline years
    
    Extreme Methods:
    - 'global_extreme': Global-in-time threshold value
    - 'hobday_extreme': Local day-of-year specific thresholds with windowing
    
    Parameters
    ----------
    da : xarray.DataArray
        Raw input data
    
    Method Selection:
    method_anomaly : str, optional
        Anomaly computation method ('detrended_baseline' or 'shifting_baseline')
    method_extreme : str, optional
        Extreme identification method ('global_extreme' or 'hobday_extreme')
    
    General Parameters:
    threshold_percentile : float, optional
        Percentile threshold for extreme event identification
    dask_chunks : dict, optional
        Chunking specification for distributed computation
    dimensions : dict, optional
        Mapping of dimension types to names in the data
    neighbours : xarray.DataArray, optional
        Neighbour connectivity for spatial clustering (optional)
    cell_areas : xarray.DataArray, optional
        Cell areas for weighted spatial statistics (optional)
    
    Shifting Baseline Method Parameters:
    window_year_baseline : int, optional
        Number of years for rolling climatology (shifting_baseline method only)
    smooth_days_baseline : int, optional
        Days for smoothing rolling climatology (shifting_baseline method only)
    
    Hobday Extreme Method Parameters:
    window_days_hobday : int, optional
        Window for day-of-year threshold calculation (hobday_extreme method only)
    
    Detrended Baseline Method Parameters:
    std_normalise : bool, optional
        Whether to standardise anomalies by rolling standard deviation (detrended_baseline only)
    detrend_orders : list, optional
        Polynomial orders for detrending (detrended_baseline method only)
    force_zero_mean : bool, optional
        Whether to enforce zero mean in detrended anomalies (detrended_baseline method only)
    
    Extreme Method Parameters:
    exact_percentile : bool, optional
        Whether to use exact or approximate percentile calculation (both global_extreme & hobday_extreme methods)
        N.B. Using exact percentile calculation requires both careful/thoughtful chunking & sufficient memory.
    
    Returns
    -------
    xarray.Dataset
        Processed dataset with anomalies and extreme event identification
    """
    # Check if input data is dask-backed
    if not is_dask_collection(da.data):
        raise ValueError('The input DataArray must be backed by a Dask array. Ensure the input data is chunked, e.g. with chunks={}')
    
    dask.config.set({'array.slicing.split_large_chunks': True})
    
    # Step 1: Compute anomalies
    ds = compute_normalised_anomaly(
        da.astype(np.float32), method_anomaly, dimensions, 
        window_year_baseline, smooth_days_baseline,
        std_normalise, detrend_orders, force_zero_mean
    )
    
    # For shifting baseline, remove first window_year_baseline years (insufficient climatology data)
    if method_anomaly == 'shifting_baseline':
        min_year = ds[dimensions['time']].dt.year.min().values
        max_year = ds[dimensions['time']].dt.year.max().values
        total_years = max_year - min_year + 1
        
        if total_years < window_year_baseline:
            raise ValueError(f"Insufficient data for shifting_baseline method. Dataset spans {total_years} years "
                           f"but window_year_baseline requires at least {window_year_baseline} years. "
                           f"Either use more data or reduce window_year_baseline parameter.")
        
        start_year = min_year + window_year_baseline
        ds['dat_anomaly'] = ds['dat_anomaly'].where(ds[dimensions['time']].dt.year >= start_year, drop=True)
    
    anomalies = ds.dat_anomaly
    
    # Step 2: Identify extreme events (both methods now return consistent tuple structures)
    extremes, thresholds = identify_extremes(
        anomalies, method_extreme, threshold_percentile, 
        dimensions, window_days_hobday, exact_percentile
    )
    
    
    # Add extreme events and thresholds to dataset
    ds['extreme_events'] = extremes
    ds['thresholds'] = thresholds
    
    # Handle standardised anomalies if requested (only for detrended_baseline)
    if std_normalise and method_anomaly == 'detrended_baseline':
        extremes_stn, thresholds_stn = identify_extremes(
            ds.dat_stn, method_extreme, threshold_percentile, 
            dimensions, window_days_hobday, exact_percentile
        )
        ds['extreme_events_stn'] = extremes_stn
        ds['thresholds_stn'] = thresholds_stn
    
    # Add optional spatial metadata
    if neighbours is not None:
        chunk_dict = {dim: -1 for dim in neighbours.dims}
        ds['neighbours'] = neighbours.astype(np.int32).chunk(chunk_dict)
        if 'nv' in neighbours.dims:
            ds = ds.assign_coords(nv=neighbours.nv)
    
    if cell_areas is not None:
        chunk_dict = {dim: -1 for dim in cell_areas.dims}
        ds['cell_areas'] = cell_areas.astype(np.float32).chunk(chunk_dict)
    
    # Add processing parameters to metadata
    ds.attrs.update({
        'method_anomaly': method_anomaly,
        'method_extreme': method_extreme,
        'threshold_percentile': threshold_percentile,
        'preprocessing_steps': _get_preprocessing_steps(method_anomaly, method_extreme, std_normalise, 
                                                       detrend_orders, window_year_baseline, smooth_days_baseline, window_days_hobday)
    })
    
    # Add method-specific parameters
    if method_anomaly == 'detrended_baseline':
        ds.attrs.update({
            'detrend_orders': detrend_orders,
            'force_zero_mean': force_zero_mean,
            'std_normalise': std_normalise
        })
    elif method_anomaly == 'shifting_baseline':
        ds.attrs.update({
            'window_year_baseline': window_year_baseline,
            'smooth_days_baseline': smooth_days_baseline
        })
    
    if method_extreme == 'hobday_extreme':
        ds.attrs.update({'window_days_hobday': window_days_hobday})
    elif method_extreme == 'global_extreme':
        ds.attrs.update({'exact_percentile': exact_percentile})
    
    # Final rechunking
    chunk_dict = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    chunk_dict[dimensions['time']] = dask_chunks['time']
    if method_extreme == 'hobday_extreme':
        chunk_dict['dayofyear'] = dask_chunks['time']
    ds = ds.chunk(chunk_dict)
    
    
    # Fix encoding issue with saving when calendar & units attribute is present
    if 'calendar' in ds[dimensions['time']].attrs:
        del ds[dimensions['time']].attrs['calendar']
    if 'units' in ds[dimensions['time']].attrs:
        del ds[dimensions['time']].attrs['units']
    
    
    ds = ds.persist(optimize_graph=True)
    ds['thresholds'] = ds.thresholds.compute()  # Patch for a dask-Zarr bug that has problems saving this data array...
    ds['dat_anomaly'] = fix_dask_tuple_array(ds.dat_anomaly)
    
    return ds


def _get_preprocessing_steps(
    method_anomaly: str,
    method_extreme: str,
    std_normalise: bool,
    detrend_orders: List[int],
    window_year_baseline: int,
    smooth_days_baseline: int,
    window_days_hobday: int
) -> List[str]:
    """
    Generate preprocessing steps description based on selected methods.
    """
    steps = []
    
    if method_anomaly == 'detrended_baseline':
        steps.append(f'Removed polynomial trend orders={detrend_orders} & seasonal cycle')
        if std_normalise:
            steps.append('Normalised by 30-day rolling STD')
    elif method_anomaly == 'shifting_baseline':
        steps.append(f'Rolling climatology using {window_year_baseline} years')
        steps.append(f'Smoothed with {smooth_days_baseline}-day window')
    
    # Extreme method steps
    if method_extreme == 'global_extreme':
        steps.append('Global percentile threshold applied to all days')
    elif method_extreme == 'hobday_extreme':
        steps.append(f'Day-of-year thresholds with {window_days_hobday} day window')
    
    return steps


def compute_normalised_anomaly(
    da: xr.DataArray,
    method_anomaly: Literal['detrended_baseline', 'shifting_baseline'] = 'detrended_baseline',
    dimensions: Dict[str, str] = {'time':'time', 'xdim':'lon', 'ydim':'lat'},
    window_year_baseline: int = 15,    # for shifting_baseline
    smooth_days_baseline: int = 21,    # "
    std_normalise: bool = False,       # for detrended_baseline
    detrend_orders: List[int] = [1],   # "
    force_zero_mean: bool = True       # "
) -> xr.Dataset:
    """
    Generate normalised anomalies using specified methodology.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data with dimensions matching the 'dimensions' parameter
    
    Method Selection:
    method_anomaly : str, optional
        Anomaly computation method ('detrended_baseline' or 'shifting_baseline')
    
    General Parameters:
    dimensions : dict, optional
        Mapping of conceptual dimensions to actual dimension names in the data
    
    Shifting Baseline Method Parameters:
    window_year_baseline : int, optional
        Number of years for rolling climatology (shifting_baseline only)
    smooth_days_baseline : int, optional
        Days for smoothing rolling climatology (shifting_baseline only)
    
    Detrended Baseline Method Parameters:
    std_normalise : bool, optional
        Whether to normalise by 30-day rolling standard deviation (detrended_baseline only)
    detrend_orders : list, optional
        Polynomial orders for trend removal (detrended_baseline only)
    force_zero_mean : bool, optional
        Explicitly enforce zero mean in final anomalies (detrended_baseline only)
        
    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies, mask, and metadata
    """
    if method_anomaly == 'detrended_baseline':
        return _compute_anomaly_detrended(da, std_normalise, detrend_orders, dimensions, force_zero_mean)
    elif method_anomaly == 'shifting_baseline':
        return _compute_anomaly_shifting_baseline(da, window_year_baseline, smooth_days_baseline, dimensions)
    else:
        raise ValueError(f"Unknown method_anomaly '{method_anomaly}'. Choose 'detrended_baseline' or 'shifting_baseline'")


def identify_extremes(
    da: xr.DataArray,
    method_extreme: Literal['global_extreme', 'hobday_extreme'] = 'global_extreme',
    threshold_percentile: float = 95,
    dimensions: Dict[str, str] = {'time':'time', 'xdim':'lon'},
    window_days_hobday: int = 11,    # for hobday_extreme
    exact_percentile: bool = False
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events exceeding a percentile threshold using specified method.
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing anomalies
    
    Method Selection:
    method_extreme : str, optional
        Method for threshold calculation ('global_extreme' or 'hobday_extreme')
    
    General Parameters:
    threshold_percentile : float, optional
        Percentile threshold (e.g., 95 for 95th percentile)
    dimensions : dict, optional
        Mapping of dimension types to names in the data
    
    Hobday Extreme Method Parameters:
    window_days_hobday : int, optional
        Window for day-of-year threshold (hobday_extreme only)
    
    Global Extreme Method Parameters:
    exact_percentile : bool, optional
        Whether to compute exact percentiles
        
    Returns
    -------
    tuple
        Tuple of (extremes, thresholds) where extremes is a boolean array 
        identifying extreme events and thresholds contains the threshold values used
    """
    if method_extreme == 'global_extreme':
        return _identify_extremes_constant(da, threshold_percentile, exact_percentile, dimensions)
    elif method_extreme == 'hobday_extreme':
        return _identify_extremes_hobday(da, threshold_percentile, window_days_hobday, exact_percentile, dimensions)
    else:
        raise ValueError(f"Unknown method_extreme '{method_extreme}'. Choose 'global_extreme' or 'hobday_extreme'")


# ===============================================
# Shifting Baseline Anomaly Method (New Method)
# ===============================================

def rolling_climatology(
    da: xr.DataArray,
    window_year_baseline: int = 15,
    dimensions: Dict[str, str] = {'time':'time', 'xdim':'lon', 'ydim':'lat'}
) -> xr.DataArray:
    """
    Compute rolling climatology efficiently using flox cohorts.
    Uses the previous `window_year_baseline` years of data and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.
    """
    
    time_dim = dimensions['time']
    original_chunk_dict = {dim: chunks for dim, chunks in zip(da.dims, da.chunks)}

    # Add temporal coordinates
    years = da[time_dim].dt.year
    doys = da[time_dim].dt.dayofyear
    da = da.assign_coords({'year': years, 'dayofyear': doys})

    # Get temporal bounds
    year_vals = years.compute().values
    doy_vals = doys.compute().values
    unique_years = np.unique(year_vals)
    min_year = unique_years.min()

    # Create long-form grouping variables
    # For each time point, determine which target years it contributes to
    contributing_time_indices = []
    contributing_target_years = []
    contributing_dayofyears = []

    for t_idx, (year_val, doy_val) in enumerate(zip(year_vals, doy_vals)):
        # Find target years this time point contributes to
        # A time point from year Y contributes to target years where:
        # target_year - window_year_baseline <= Y < target_year
        # Which means: Y < target_year <= Y + window_year_baseline
        candidate_targets = unique_years[(unique_years > year_val) & 
                                    (unique_years <= year_val + window_year_baseline)]
        
        # Only include target years that have sufficient history
        valid_targets = candidate_targets[candidate_targets >= min_year + window_year_baseline]
        
        # Add entries for each valid target year
        n_targets = len(valid_targets)
        contributing_time_indices.extend([t_idx] * n_targets)
        contributing_target_years.extend(valid_targets)
        contributing_dayofyears.extend([doy_val] * n_targets)

    # Convert to numpy arrays
    time_indices = np.array(contributing_time_indices)
    target_year_groups = np.array(contributing_target_years)
    dayofyear_groups = np.array(contributing_dayofyears)

    # Create long-form dataset by selecting the contributing time points
    long_form_data = da.isel({time_dim: time_indices})

    # Create a new time dimension for the long-form data
    long_time_dim = f"{time_dim}_contrib"
    long_form_data = long_form_data.rename({time_dim: long_time_dim})

    # Convert grouping arrays to DataArrays with the correct dimension
    target_year_da = xr.DataArray(target_year_groups, dims=[long_time_dim], name='target_year')
    dayofyear_da = xr.DataArray(dayofyear_groups, dims=[long_time_dim], name='dayofyear')

    # Use flox with both grouping variables to compute climatologies
    climatologies = flox.xarray.xarray_reduce(
        long_form_data,
        target_year_da,
        dayofyear_da,
        dim=long_time_dim,
        func='nanmean',
        expected_groups=(unique_years, np.arange(1, 367)),
        isbin=(False, False),
        dtype=np.float32,
        fill_value=np.nan
    )

    # Create index arrays for final mapping
    year_to_idx = pd.Series(range(len(unique_years)), index=unique_years)
    year_indices = year_to_idx[year_vals].values

    # Select appropriate climatology for each time point
    result = climatologies.isel(
        target_year=xr.DataArray(year_indices, dims=[time_dim]),
        dayofyear=xr.DataArray(doy_vals - 1, dims=[time_dim])
    )

    # Clean up dimensions and coordinates
    result = result.drop_vars(['target_year', 'dayofyear'])
    
    return result.chunk(original_chunk_dict)


def smoothed_rolling_climatology(
    da: xr.DataArray,
    window_year_baseline: int = 15,
    smooth_days_baseline: int = 21,
    dimensions: Dict[str, str] = {'time':'time', 'xdim':'lon', 'ydim':'lat'}
) -> xr.DataArray:
    """
    Compute a smoothed rolling climatology using the previous `window_year_baseline` years of data and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.
    """
    
    # N.B.: It is more efficient (chunking-wise) to smooth the raw data rather than the climatology
    da_smoothed = da.rolling({dimensions['time']: smooth_days_baseline}, center=True).mean().chunk({dim: chunks for dim, chunks in zip(da.dims, da.chunks)})

    clim = rolling_climatology(da_smoothed, window_year_baseline, dimensions)
    
    return clim


def _compute_anomaly_shifting_baseline(
    da: xr.DataArray,
    window_year_baseline: int = 15,
    smooth_days_baseline: int = 21,
    dimensions: Dict[str, str] = {'time':'time', 'xdim':'lon', 'ydim':'lat'}
) -> xr.Dataset:
    """
    Compute anomalies using shifting baseline method with smoothed rolling climatology.
    
    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # Compute smoothed rolling climatology
    climatology_smoothed = smoothed_rolling_climatology(da, window_year_baseline, smooth_days_baseline, dimensions)
    
    # Compute anomaly as difference from climatology
    anomalies = da - climatology_smoothed
    
    # Create ocean/land mask from first time step
    chunk_dict_mask = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    mask = np.isfinite(da.isel({dimensions['time']: 0})).drop_vars({'time'}).chunk(chunk_dict_mask)
    
    # Build output dataset
    return xr.Dataset({
        'dat_anomaly': anomalies,
        'mask': mask
    })


# ==========================
# Hobday Extreme Definition 
# ==========================

def _identify_extremes_hobday(
    da: xr.DataArray,
    threshold_percentile: float = 95,
    window_days_hobday: int = 11,
    exact_percentile: bool = False,
    dimensions: Dict[str, str] = {'time':'time', 'xdim':'lon', 'ydim':'lat'}
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events using day-of-year (i.e. climatological percentile threshold).
    
    For each spatial point and day-of-year, computes the p-th percentile of values within a window_days_hobday day window across all years. 
    This implements the standard methodology for marine heatwave detection threshold calculation.
        
    Parameters:
    -----------
    da : xarray.DataArray
        Anomaly data with dimensions (time, lat, lon)
        Must be chunked with time dimension unbounded (time: -1)
    threshold_percentile : float, default 95
        Percentile to compute (0-100)
    window_days_hobday : int, default 11
        Window in days
    exact_percentile : bool, optional
        Whether to compute exact percentiles
    
    Returns:
    --------
    tuple
        (extreme_bool, thresholds)
        extreme_bool : xarray.DataArray
            Boolean mask indicating extreme events (True for extreme days)
        thresholds : xarray.DataArray
            Threshold values with dimensions (dayofyear, lat, lon)
    """
    # Add day-of-year coordinate
    da = da.assign_coords(dayofyear=da[dimensions['time']].dt.dayofyear)
    
    # Group by day-of-year and compute percentile
    if exact_percentile:
        # Construct rolling window dimension
        da_windowed = da.rolling({dimensions['time']: window_days_hobday}, center=True).construct('window')
        
        thresholds = da_windowed.groupby('dayofyear').reduce(
                            np.nanpercentile,
                            q=threshold_percentile,
                            dim=('window', dimensions['time'])
                        )
    else:  # Optimised histogram approximation method
        thresholds = compute_histogram_quantile_2d(da, threshold_percentile/100.0, window_days_hobday=window_days_hobday, dimensions=dimensions)
    
    # Ensure spatial dimensions are fully loaded for efficient comparison
    spatial_chunks = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    thresholds = thresholds.chunk(spatial_chunks)
    
    # Compare anomalies to day-of-year specific thresholds
    extremes = da.groupby(da[dimensions['time']].dt.dayofyear) >= thresholds
    extremes = extremes.astype(bool).chunk(spatial_chunks)
    
    return extremes, thresholds


# ===============================================
# Detrended Baseline Anomaly Method (Old Method)
# ===============================================

def add_decimal_year(
    da: xr.DataArray,
    dim: str = 'time'
) -> xr.DataArray:
    """
    Add decimal year coordinate to DataArray for trend analysis.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data with datetime coordinate
    dim : str, optional
        Name of the time dimension
    
    Returns
    -------
    xarray.DataArray
        Input data with added 'decimal_year' coordinate
    """
    time = pd.to_datetime(da[dim])
    start_of_year = pd.to_datetime(time.year.astype(str) + '-01-01')
    start_of_next_year = pd.to_datetime((time.year + 1).astype(str) + '-01-01')
    year_elapsed = (time - start_of_year).days
    year_duration = (start_of_next_year - start_of_year).days
    
    decimal_year = time.year + year_elapsed / year_duration
    return da.assign_coords(decimal_year=(dim, decimal_year))


def _compute_anomaly_detrended(
    da: xr.DataArray,
    std_normalise: bool = False,
    detrend_orders: List[int] = [1],
    dimensions: Dict[str, str] = {'time':'time', 'xdim':'lon', 'ydim':'lat'},
    force_zero_mean: bool = True
) -> xr.Dataset:
    """
    Generate normalised anomalies by removing trends, seasonal cycles, and optionally
    standardising by local temporal variability using the detrended baseline method.
    """
    da = da.astype(np.float32)
    
    # Ensure time is the first dimension for efficient processing
    if da.dims[0] != dimensions['time']:
        da = da.transpose(dimensions['time'], ...)
    
    # Warn if using higher-order detrending without linear component
    if 1 not in detrend_orders and len(detrend_orders) > 1:
        print('Warning: Higher-order detrending without linear term may be unstable')
    
    # Add decimal year for trend modelling
    da = add_decimal_year(da)
    dy = da.decimal_year.compute()
    
    # Build model matrix with constant term, trends, and seasonal harmonics
    model_components = [np.ones(len(dy))]  # Constant term
    
    # Add polynomial trend terms
    centered_time = da.decimal_year - np.mean(dy)
    for order in detrend_orders:
        model_components.append(centered_time ** order)
    
    # Add annual and semi-annual cycles (harmonics)
    model_components.extend([
        np.sin(2 * np.pi * dy),     # Annual sine
        np.cos(2 * np.pi * dy),     # Annual cosine
        np.sin(4 * np.pi * dy),     # Semi-annual sine
        np.cos(4 * np.pi * dy)      # Semi-annual cosine
    ])
    
    # Convert to numpy array for matrix operations
    model = np.array(model_components)
    
    # Orthogonalise model components for numerical stability
    for i in range(1, model.shape[0]):
        model[i] = model[i] - np.mean(model[i]) * model[0]
    
    # Compute pseudo-inverse for model fitting
    pmodel = np.linalg.pinv(model)
    n_coeffs = len(model_components)
    
    # Convert model matrices to xarray
    model_da = xr.DataArray(
        model.T, 
        dims=[dimensions['time'],'coeff'], 
        coords={dimensions['time']: da[dimensions['time']].values, 
                'coeff': np.arange(1, n_coeffs+1)}
    ).chunk({dimensions['time']: da.chunks[0]})
    
    pmodel_da = xr.DataArray(
        pmodel.T,
        dims=['coeff', dimensions['time']],
        coords={'coeff': np.arange(1, n_coeffs+1), 
                dimensions['time']: da[dimensions['time']].values}
    ).chunk({dimensions['time']: da.chunks[0]})
    
    # Prepare dimensions for model coefficients based on data structure
    dims = ['coeff']
    coords = {'coeff': np.arange(1, n_coeffs + 1)}
    
    # Handle both 2D (unstructured) and 3D (gridded) data
    if 'ydim' in dimensions:  # 3D gridded case
        dims.extend([dimensions['ydim'], dimensions['xdim']])
        coords[dimensions['ydim']] = da[dimensions['ydim']].values
        coords[dimensions['xdim']] = da[dimensions['xdim']].values
    else:  # 2D unstructured case
        dims.append(dimensions['xdim'])
        coords.update(da[dimensions['xdim']].coords)

    # Fit model to data
    model_fit_da = xr.DataArray(
        pmodel_da.dot(da),
        dims=dims,
        coords=coords
    )
    
    # Remove trend and seasonal cycle
    da_detrend = (da.drop_vars({'decimal_year'}) - model_da.dot(model_fit_da).astype(np.float32))
    
    # Force zero mean if requested
    if force_zero_mean:
        da_detrend = da_detrend - da_detrend.mean(dim=dimensions['time'])
    
    # Create ocean/land mask from first time step
    chunk_dict_mask = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    mask = np.isfinite(da.isel({dimensions['time']: 0})).chunk(chunk_dict_mask).drop_vars({'decimal_year', 'time'})
    
    # Initialise output dataset
    data_vars = {
        'dat_anomaly': da_detrend,
        'mask': mask
    }    
    
    # Standardise anomalies by temporal variability if requested
    if std_normalise: 
        
        # Calculate day-of-year standard deviation using cohorts
        std_day = flox.xarray.xarray_reduce(
            da_detrend,
            da_detrend[dimensions['time']].dt.dayofyear,
            dim=dimensions['time'],
            func='std',
            isbin=False,
            method='cohorts'
        )
        
        # Calculate 30-day rolling standard deviation with annual wrapped padding
        std_day_wrap = std_day.pad(dayofyear=16, mode='wrap')
        std_rolling = np.sqrt(
            (std_day_wrap**2)
            .rolling(dayofyear=30, center=True)
            .mean()
        ).isel(dayofyear=slice(16, 366+16))
        
        # Divide anomalies by rolling standard deviation
        # Replace any zeros or extremely small values with NaN to avoid division warnings
        std_rolling_safe = std_rolling.where(std_rolling > 1e-10, np.nan)
        da_stn = da_detrend.groupby(da_detrend[dimensions['time']].dt.dayofyear) / std_rolling_safe
        
        # Rechunk data for efficient processing
        chunk_dict_std = chunk_dict_mask.copy()
        chunk_dict_std['dayofyear'] = -1
        
        da_stn = da_stn.chunk(chunk_dict_std)
        std_rolling = std_rolling.chunk(chunk_dict_std)
        
        # Add standardised data to output
        data_vars['dat_stn'] = da_stn.drop_vars({'dayofyear', 'decimal_year'})
        data_vars['STD'] = std_rolling
    
    # Build output dataset with metadata
    return xr.Dataset(data_vars=data_vars)


def _rolling_histogram_quantile(
    hist_chunk: NDArray[np.float64],
    window_days_hobday: int,
    q: float,
    bin_centers: NDArray[np.float64]
) -> NDArray[np.float32]:
    """
    Efficiently compute quantile thresholds from histogram data using vectorised numpy operations.
    
    Parameters
    ----------
    hist_chunk : numpy.ndarray
        Histogram data with shape (dayofyear, da_bin)
    window_days_hobday : int
        Rolling window size for day-of-year smoothing
    q : float
        Quantile to compute (0-1)
    bin_centers : numpy.ndarray
        Bin centre values for interpolation
        
    Returns
    -------
    numpy.ndarray
        Quantile thresholds with shape (dayofyear,)
    """
    n_doy, n_bins = hist_chunk.shape
    
    # Pad histogram with wrap mode for day-of-year cycling
    pad_size = window_days_hobday // 2
    hist_pad = np.concatenate([
        hist_chunk[-pad_size:],
        hist_chunk,
        hist_chunk[:pad_size]
    ], axis=0)
    
    # Apply rolling sum using stride tricks (FTW)
    windowed_view = sliding_window_view(hist_pad, window_days_hobday, axis=0)
    hist_windowed = np.sum(windowed_view, axis=-1)
    
    # Compute PDF and CDF in single pass
    hist_sum = np.sum(hist_windowed, axis=1, keepdims=True) + 1e-10
    pdf = hist_windowed / hist_sum
    cdf = np.cumsum(pdf, axis=1)
    
    # Find first bin exceeding quantile threshold
    mask = cdf >= q
    first_true = np.argmax(mask, axis=1)
    idx_prev = np.clip(first_true - 1, 0, n_bins - 1)
    
    # Extract CDF values for linear interpolation
    doy_indices = np.arange(n_doy)
    cdf_prev = cdf[doy_indices, idx_prev]
    cdf_next = cdf[doy_indices, first_true]
    
    bin_prev = bin_centers[idx_prev]
    bin_next = bin_centers[first_true]
    
    denom = cdf_next - cdf_prev
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    frac = (q - cdf_prev) / denom
    threshold = bin_prev + frac * (bin_next - bin_prev)
    
    return threshold.astype(np.float32)


def compute_histogram_quantile_2d(
    da: xr.DataArray,
    q: float,
    window_days_hobday: int = 11,
    bin_edges: Optional[NDArray[np.float64]] = None,
    dimensions: Dict[str, str] = {'time':'time', 'xdim':'lon', 'ydim':'lat'}
) -> xr.DataArray:
    """
    Efficiently compute quantiles using binned histograms optimised for extreme values.
    Uses fine-grained bins for positive anomalies and a single bin for negative values.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    q : float
        Quantile to compute (0-1)
    dim : str, optional
        Dimension along which to compute quantile
        
    Returns
    -------
    xarray.DataArray
        Computed quantile value for each spatial location
    """
    if bin_edges is None:
        # Default asymmetric bins
        precision = 0.01
        max_anomaly = 5.0
        bin_edges = np.concatenate([
            [-np.inf, 0.],
            np.arange(precision, max_anomaly + precision, precision)
        ])
    
    bin_centers_array = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_centers_array[0] = 0.
    
    bin_centers = xr.DataArray(
        bin_centers_array,
        dims=['da_bin'],
        coords={'da_bin': np.arange(len(bin_centers_array))},
        name='bin_centers'
    )
    
    chunk_dict = {dimensions['time']: -1}
    for d in ['xdim', 'ydim']:
        if d in dimensions:
            chunk_dict[dimensions[d]] = 10
    
    da_bin = xr.DataArray(
        np.digitize(da.data, bin_edges) - 1,  # -1 so first bin is 0
        dims=da.dims,
        coords=da.coords,
        name="da_bin"
    ).chunk(chunk_dict)
    
    # Construct 2D histogram using flox (in doy & anomaly)
    hist_raw = flox.xarray.xarray_reduce(
        da_bin,
        da_bin.dayofyear,
        da_bin,
        dim=[dimensions['time']],
        func="count",
        expected_groups=(np.arange(1, 367), np.arange(len(bin_edges)-1)),
        isbin=(False,False),
        dtype=np.int32,
        fill_value=0
    )
    hist_raw.name = None
    
    def _compute_quantile_with_params(hist_chunk, bin_centers_chunk):
        return _rolling_histogram_quantile(hist_chunk, window_days_hobday, q, bin_centers_chunk)
    
    # Apply the optimised computation using apply_ufunc
    threshold = xr.apply_ufunc(
        _compute_quantile_with_params,
        hist_raw,
        bin_centers,
        input_core_dims=[['dayofyear', 'da_bin'], ['da_bin']],
        output_core_dims=[['dayofyear']],
        dask='parallelized',
        vectorize=True,
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={'output_sizes': {'dayofyear': 366}},
        keep_attrs=True
    )
        
    return threshold
    
    
def compute_histogram_quantile_1d(
    da: xr.DataArray,
    q: float,
    dim: str = 'time',
    bin_edges: Optional[NDArray[np.float64]] = None
) -> xr.DataArray:
    """
    Efficiently compute quantiles using binned histograms optimised for extreme values.
    Uses fine-grained bins for positive anomalies and a single bin for negative values.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    q : float
        Quantile to compute (0-1)
    dim : str, optional
        Dimension along which to compute quantile
        
    Returns
    -------
    xarray.DataArray
        Computed quantile value for each spatial location
    """
    if bin_edges is None:
        # Default asymmetric bins
        precision = 0.01
        max_anomaly = 5.0
        bin_edges = np.concatenate([
            [-np.inf, 0.],
            np.arange(precision, max_anomaly + precision, precision)
        ])
    
    # Compute histogram
    hist = histogram(da, bins=[bin_edges], dim=[dim])
    
    # Convert to PDF and CDF
    hist_sum = hist.sum(dim=f'{da.name}_bin') + 1e-10
    pdf = hist / hist_sum
    cdf = pdf.cumsum(dim=f'{da.name}_bin')
    
    # Get bin centers
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_centers[0] = 0.  # Set negative bin centre to 0
    
    # Find first bin exceeding quantile
    mask = cdf >= q
    first_true = mask.argmax(dim=f'{da.name}_bin')
    
    # Linearly interpolate between the two points around the 0 crossing
    idx = first_true.compute()
    idx_prev = np.clip(idx - 1, 0, len(bin_centers) - 1)

    cdf_prev = cdf.isel({f'{da.name}_bin': xr.DataArray(idx_prev, dims=first_true.dims)}).data
    cdf_next = cdf.isel({f'{da.name}_bin': xr.DataArray(idx, dims=first_true.dims)}).data
    bin_prev = bin_centers[idx_prev]
    bin_next = bin_centers[idx]

    denom = cdf_next - cdf_prev
    frac = (q - cdf_prev) / denom
    result_data = bin_prev + frac * (bin_next - bin_prev)

    result = first_true.copy(data=result_data)
    
    return result


# ======================================
# Constant (in time) Extreme Definition 
# ======================================

def _identify_extremes_constant(
    da: xr.DataArray,
    threshold_percentile: float = 95,
    exact_percentile: bool = False,
    dimensions: Dict[str, str] = {'time':'time', 'xdim':'lon'}
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events exceeding a constant (in time) percentile threshold.
    i.e. There is 1 threshold for each spatial point, computed across all time.
    
    Returns both the extreme events boolean mask and the thresholds used.
    """
    if exact_percentile:  # Compute exact percentile (memory-intensive)
        # Determine appropriate chunk size based on data dimensions
        if 'ydim' in dimensions:
            rechunk_size = 'auto'
        else:
            rechunk_size = 100*int(np.sqrt(da.ncells.size)*1.5/100)
        # N.B.: If this rechunk_size is too small, then dask will be overwhelmed by the number of tasks
        chunk_dict = {dimensions[dim]: rechunk_size for dim in ['xdim', 'ydim'] if dim in dimensions}
        chunk_dict[dimensions['time']] = -1
        da_rechunk = da.chunk(chunk_dict)
    
        # Calculate threshold
        threshold = da_rechunk.quantile(threshold_percentile/100.0, dim=dimensions['time'])
    
    else:  # Use an efficient histogram-based method with specified accuracy
        threshold = compute_histogram_quantile_1d(da, threshold_percentile/100.0, dim=dimensions['time'])
    
    # Clean up coordinates if needed
    if 'quantile' in threshold.coords:
        threshold = threshold.drop_vars('quantile')
    
    # Ensure spatial dimensions are fully loaded for efficient comparison
    spatial_chunks = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    threshold = threshold.chunk(spatial_chunks)
    
    # Create boolean mask for values exceeding threshold
    extremes = da >= threshold
    
    # Clean up coordinates if needed
    if 'quantile' in extremes.coords:
        extremes = extremes.drop_vars('quantile')
    
    extremes = extremes.astype(bool).chunk({dim: chunks for dim, chunks in zip(da.dims, da.chunks)})
    
    return extremes, threshold


