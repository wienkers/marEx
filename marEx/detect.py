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
import logging

logging.getLogger('distributed.shuffle._scheduler_plugin').setLevel(logging.ERROR)


# ============================
# Methodology Selection
# ============================

def preprocess_data(da, method_anomaly='detrended_baseline', method_extreme='global_extreme', 
                   threshold_percentile=95, 
                   window_year_baseline=15, smooth_days_baseline=21,  # for shifting_baseline
                   window_days_hobday=11,                  # for hobday_extreme
                   std_normalise=False, detrend_orders=[1], force_zero_mean=True, # for detrended_baseline
                   exact_percentile=False,         # for global_extreme
                   dask_chunks={'time': 25}, dimensions={'time':'time', 'xdim':'lon'}, 
                   neighbours=None, cell_areas=None):
    """
    Complete preprocessing pipeline for marine extreme event identification.
    
    Supports separate methods for anomaly computation and extreme identification:
    
    Anomaly Methods:
    - 'detrended_baseline': Detrending with harmonics and polynomials -- more efficient, but biases statistics
    - 'shifting_baseline': Rolling climatology using previous window_year_baseline years -- more "correct", but shortens time series by window_year_baseline years
    
    Extreme Methods:
    - 'global_extreme': Global-in-time percentile threshold
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
    
    Global Extreme Method Parameters:
    exact_percentile : bool, optional
        Whether to use exact or approximate percentile calculation (global_extreme method only)
    
    Returns
    -------
    xarray.Dataset
        Processed dataset with anomalies and extreme event identification
    """
    # Check if input data is dask-backed
    if not is_dask_collection(da.data):
        raise ValueError('The input DataArray must be backed by a Dask array. Ensure the input data is chunked, e.g. with chunks={}')
    
    # Step 1: Compute anomalies
    ds = compute_normalised_anomaly(
        da, method_anomaly, dimensions, 
        window_year_baseline, smooth_days_baseline,
        std_normalise, detrend_orders, force_zero_mean
    ).persist()
    
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
        ds = ds.where(ds[dimensions['time']].dt.year >= start_year, drop=True)
    
    anomalies = ds.dat_detrend
    
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
    ds = ds.chunk(chunk_dict)
    
    # Fix encoding issue with saving when calendar & units attribute is present
    if 'calendar' in ds[dimensions['time']].attrs:
        del ds[dimensions['time']].attrs['calendar']
    if 'units' in ds[dimensions['time']].attrs:
        del ds[dimensions['time']].attrs['units']
    
    return ds


def _get_preprocessing_steps(method_anomaly, method_extreme, std_normalise, detrend_orders, 
                            window_year_baseline, smooth_days_baseline, window_days_hobday):
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


def compute_normalised_anomaly(da, method_anomaly='detrended_baseline',
                               dimensions={'time':'time', 'xdim':'lon', 'ydim':'lat'},
                               window_year_baseline=15, smooth_days_baseline=21,  # for shifting_baseline
                               std_normalise=False, detrend_orders=[1], force_zero_mean=True):  # for detrended_baseline
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
        return _compute_detrended_anomaly(da, std_normalise, detrend_orders, dimensions, force_zero_mean)
    elif method_anomaly == 'shifting_baseline':
        return _compute_shifting_baseline_anomaly(da, window_year_baseline, smooth_days_baseline, dimensions)
    else:
        raise ValueError(f"Unknown method_anomaly '{method_anomaly}'. Choose 'detrended_baseline' or 'shifting_baseline'")


def identify_extremes(da, method_extreme='global_extreme', 
                     threshold_percentile=95, dimensions={'time':'time', 'xdim':'lon'},
                     window_days_hobday=11,  # for hobday_extreme
                     exact_percentile=False):  # for global_extreme
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
        Whether to compute exact percentiles (global_extreme only)
        
    Returns
    -------
    tuple
        Tuple of (extremes, thresholds) where extremes is a boolean array 
        identifying extreme events and thresholds contains the threshold values used
    """
    if method_extreme == 'global_extreme':
        return _identify_extremes_detrended(da, threshold_percentile, exact_percentile, dimensions)
    elif method_extreme == 'hobday_extreme':
        return _identify_extremes_hobday(da, threshold_percentile, window_days_hobday, dimensions)
    else:
        raise ValueError(f"Unknown method_extreme '{method_extreme}'. Choose 'global_extreme' or 'hobday_extreme'")


# ===============================================
# Shifting Baseline Anomaly Method (New Method)
# ===============================================

def rolling_climatology(da, window_year_baseline=15, time_dim='time'):
    """
    Compute rolling climatology efficiently using flox cohorts.
    Uses the previous `window_year_baseline` years of data and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.
    """
    # Add year and day-of-year coordinates
    da = da.assign_coords({
        'year': (time_dim, da[time_dim].dt.year.data),
        'dayofyear': (time_dim, da[time_dim].dt.dayofyear.data)
    })
    
    # Get unique years
    years = da.year.values
    unique_years = np.unique(years)
    min_year = unique_years.min()
    
    # Pre-compute year mappings for all valid target years
    year_mappings = {}
    for target_year in unique_years:
        start_year = target_year - window_year_baseline
        if start_year >= min_year:
            year_mappings[target_year] = np.arange(start_year, target_year)
    
    # Initialise output with NaN
    result = xr.full_like(da, np.nan)
    
    # Process all years in parallel using flox
    for target_year, source_years in year_mappings.items():
        # Create mask for source years
        source_mask = da.year.isin(source_years)
        
        # Compute climatology for this window using flox
        window_clim = flox.xarray.xarray_reduce(
            da.where(source_mask),
            da.dayofyear.where(source_mask),
            func='nanmean',
            dim=time_dim,
            expected_groups=np.arange(1, 367),
            isbin=False,
            method='cohorts',
            engine='flox'
        )
        
        # Assign to target year using groupby for efficiency
        target_mask = da.year == target_year
        target_doys = da.dayofyear.where(target_mask)
        
        # Map climatology to target year positions
        result = xr.where(
            target_mask,
            window_clim.sel(dayofyear=da.dayofyear),
            result
        )
    
    # Clean up coordinates
    result = result.drop_vars(['year', 'dayofyear'])
    
    return result.chunk(da.chunks)


def smoothed_rolling_climatology(da, window_year_baseline=15, smooth_days_baseline=21, time_dim='time'):
    """
    Compute a smoothed rolling climatology using the previous `window_year_baseline` years of data and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.
    """
    # Rechunk for efficient time operations using flox
    da_rechunked = flox.xarray.rechunk_for_cohorts(
        da,
        dim=time_dim,
        labels=da[time_dim].dt.year,
        chunksize='auto',
        ignore_old_chunks=False,
        force_new_chunk_at=np.unique(da[time_dim].dt.year)
    )
    
    clim = rolling_climatology(da_rechunked, window_year_baseline, time_dim)
    clim_smoothed = clim.rolling({time_dim: smooth_days_baseline}, center=True).mean().chunk(da_rechunked.chunks)
    
    return clim_smoothed


def _compute_shifting_baseline_anomaly(da, window_year_baseline=15, smooth_days_baseline=21, dimensions={'time':'time', 'xdim':'lon', 'ydim':'lat'}):
    """
    Compute anomalies using shifting baseline method with rolling climatology.
    
    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # Compute smoothed rolling climatology
    climatology_smoothed = smoothed_rolling_climatology(da, window_year_baseline, smooth_days_baseline, time_dim=dimensions['time'])
    
    # Compute anomaly as difference from climatology
    anomalies = da - climatology_smoothed
    
    # Create ocean/land mask from first time step
    chunk_dict_mask = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    mask = np.isfinite(da.isel({dimensions['time']: 0})).chunk(chunk_dict_mask)
    
    # Build output dataset
    return xr.Dataset({
        'dat_detrend': anomalies,
        'mask': mask
    }, attrs={
        'description': 'Shifting Baseline Anomalies',
        'method': 'shifting_baseline',
        'window_year_baseline': window_year_baseline,
        'smooth_days_baseline': smooth_days_baseline
    })


# ==========================
# Hobday Extreme Definition 
# ==========================

def compute_percentile_threshold(sst_anom, threshold_percentile=95, window_days_hobday=11, time_dim='time'):
    """
    Compute climatological percentile threshold for SST anomaly data.
    
    For each spatial point and day-of-year, computes the p-th percentile of values within a ±window_days_hobday day window across all years. 
    This implements the standard methodology for marine heatwave detection threshold calculation.
    
    Parameters:
    -----------
    sst_anom : xarray.DataArray
        SST anomaly data with dimensions (time, lat, lon)
        Must be chunked with time dimension unbounded (time: -1)
    threshold_percentile : float, default 95
        Percentile to compute (0-100)
    window_days_hobday : int, default 11
        Window in days
    
    Returns:
    --------
    xarray.DataArray
        Threshold values with dimensions (dayofyear, lat, lon)
        Preserves spatial chunking from input array
    """
    # # Ensure proper chunking for dask efficiency
    # if not hasattr(sst_anom.data, 'chunks'):
    #     sst_anom = sst_anom.chunk({time_dim: -1})
    
    window_half_width = window_days_hobday // 2
    
    # day-of-year coordinate for groupby
    dayofyear = sst_anom[time_dim].dt.dayofyear
    sst_anom = sst_anom.assign_coords(dayofyear=(time_dim, dayofyear.data))
    
    def get_windowed_doys(target_doy, window_half_width, max_doy=366):
        """Generate day-of-year values for windowed selection with year wrapping."""
        window_doys = []
        for offset in range(-window_half_width, window_half_width + 1):
            doy = target_doy + offset
            # Year boundary wrapping
            if doy <= 0:
                doy += max_doy
            elif doy > max_doy:
                doy -= max_doy
            window_doys.append(doy)
        return window_doys
    
    unique_doys = sorted(np.unique(sst_anom.dayofyear.values))
    
    # Compute threshold for each day-of-year
    threshold_list = []
    
    for target_doy in unique_doys:
        # Get all day-of-year values in the window_half_width window
        window_doys = get_windowed_doys(target_doy, window_half_width)
        
        # Select all time points within the window across all years
        window_mask = sst_anom.dayofyear.isin(window_doys)
        window_data = sst_anom.where(window_mask, drop=True)
        
        threshold = window_data.quantile(
            threshold_percentile/100, 
            dim=time_dim, 
            skipna=True,
            keep_attrs=True
        )
        
        # Assign day-of-year coordinate to this threshold
        threshold = threshold.assign_coords(dayofyear=target_doy)
        threshold_list.append(threshold)
    
    result = xr.concat(threshold_list, dim='dayofyear')
    result = result.sortby('dayofyear')
    
    result.attrs.update({
        'long_name': f'{threshold_percentile}th percentile threshold',
        'description': f'Climatological {threshold_percentile}th percentile computed using {window_days_hobday}-day rolling window',
        'window_size': f'{window_days_hobday} days',
        'percentile': threshold_percentile,
        'method': 'day-of-year groupby with windowed percentile calculation'
    })
    
    return result


def _identify_extremes_hobday(da, threshold_percentile=95, window_days_hobday=11, 
                             dimensions={'time':'time', 'xdim':'lon'}):
    """
    Identify extreme events using day-of-year specific thresholds (Hobday method).
    
    Returns both the extreme events boolean mask and the thresholds used.
    """
    # Compute day-of-year specific thresholds
    thresholds = compute_percentile_threshold(da, threshold_percentile, window_days_hobday, time_dim=dimensions['time'])
    
    if 'quantile' in thresholds.coords:
        thresholds = thresholds.drop_vars('quantile')
    
    thresholds = thresholds.chunk({'dayofyear': -1})
    
    # Compare anomalies to day-of-year specific thresholds
    extreme_bool = da.groupby(da[dimensions['time']].dt.dayofyear) >= thresholds
    
    return extreme_bool, thresholds


# ===============================================
# Detrended Baseline Anomaly Method (Old Method)
# ===============================================

def add_decimal_year(da, dim='time'):
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


def rechunk_for_cohorts(da, chunksize=100, dim='time'):
    """
    Optimise chunking for climatology calculations using day-of-year cohorts.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    chunksize : int, optional
        Target chunk size
    dim : str, optional
        Homogeneous dimension along which to rechunk
        
    Returns
    -------
    xarray.DataArray
        Optimally chunked data for climatology calculations
    """
    return flox.xarray.rechunk_for_cohorts(da, 
                                          dim=dim, 
                                          labels=da[dim].dt.dayofyear, 
                                          force_new_chunk_at=1, 
                                          chunksize=chunksize, 
                                          ignore_old_chunks=True)


def _compute_detrended_anomaly(da, std_normalise=False, detrend_orders=[1], 
                               dimensions={'time':'time', 'xdim':'lon', 'ydim':'lat'},
                               force_zero_mean=True):
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
        'dat_detrend': da_detrend,
        'mask': mask
    }    
    
    # Standardise anomalies by temporal variability if requested
    if std_normalise: 
        print('Note: It is highly recommended to use rechunk_for_cohorts on input data before STD normalisation')
        print('    e.g. To compute optimal cohort chunks for your data on disk, and then load directly into the dask array:')
        print('           da_predictor = xr.open_dataset(\'path_to_data.nc\', chunks={}).var')
        print('           time_chunk = marex.rechunk_for_cohorts(da_predictor).chunks[0]')
        print('           var = xr.open_dataset(\'path_to_data.nc\', chunks={\'time\': time_chunk})')
        
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
    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            'description': 'Standardised & Detrended Data',
            'method': 'detrended_baseline',
            'preprocessing_steps': [
                f'Removed {"polynomial trend orders=" + str(detrend_orders)} & seasonal cycle',
                'Normalised by 30-day rolling STD' if std_normalise else 'No STD normalisation'
            ],
            'detrend_orders': detrend_orders,
            'force_zero_mean': force_zero_mean
        }
    )


def compute_histogram_quantile(da, q, dim='time'):
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
    # Configure histogram with asymmetric bins (higher resolution for positive values)
    precision   = 0.025  # Bin width for positive values
    max_anomaly = 10.0   # Maximum expected anomaly value
    
    # Create bin edges with special treatment for negative values
    bin_edges = np.concatenate([
        [-np.inf, 0.],  # Single bin for all negative values
        np.arange(precision, max_anomaly+precision, precision)  # Fine bins for positive values
    ])
    
    # Compute histogram along specified dimension
    hist = histogram(
        da,
        bins=[bin_edges],
        dim=[dim]
    )
    
    # Convert to PDF and CDF with handling for empty histograms
    hist_sum = hist.sum(dim='dat_detrend_bin') + 1e-10
    pdf = hist / hist_sum
    cdf = pdf.cumsum(dim='dat_detrend_bin')
    
    # Get bin centres for interpolation
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_centers[0] = 0.  # Set negative bin centre to 0
    
    # Find bins where CDF crosses desired quantile
    mask = cdf >= q
    
    # Get the first bin that exceeds the quantile
    first_true = mask.argmax(dim='dat_detrend_bin')
    
    # Convert bin indices to actual values
    result = first_true.copy(data=bin_centers[first_true])
    
    return result


# ==========================
# Global Extreme Definition 
# ==========================

def _identify_extremes_detrended(da, threshold_percentile=95, exact_percentile=False, 
                                dimensions={'time':'time', 'xdim':'lon'}):
    """
    Identify extreme events exceeding a global percentile threshold.
    
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
        threshold = compute_histogram_quantile(da, threshold_percentile/100.0, dim=dimensions['time'])
    
    # Clean up coordinates if needed
    if 'quantile' in threshold.coords:
        threshold = threshold.drop_vars('quantile')
    
    # Add attributes for documentation
    threshold.attrs.update({
        'long_name': f'{threshold_percentile}th percentile threshold',
        'description': f'Global {threshold_percentile}th percentile computed across all time',
        'percentile': threshold_percentile,
        'method': 'global percentile calculation'
    })
    
    # Create boolean mask for values exceeding threshold
    extremes = da >= threshold
    
    # Clean up coordinates if needed
    if 'quantile' in extremes.coords:
        extremes = extremes.drop_vars('quantile')
    
    return extremes, threshold


