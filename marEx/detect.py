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
                   exact_percentile=False,         # for both extremes algorithms
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
        return _compute_anomaly_detrended(da, std_normalise, detrend_orders, dimensions, force_zero_mean)
    elif method_anomaly == 'shifting_baseline':
        return _compute_anomaly_shifting_baseline(da, window_year_baseline, smooth_days_baseline, dimensions)
    else:
        raise ValueError(f"Unknown method_anomaly '{method_anomaly}'. Choose 'detrended_baseline' or 'shifting_baseline'")


def identify_extremes(da, method_extreme='global_extreme', 
                     threshold_percentile=95, dimensions={'time':'time', 'xdim':'lon'},
                     window_days_hobday=11,  # for hobday_extreme
                     exact_percentile=False):
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


def smoothed_rolling_climatology(da, window_year_baseline=15, smooth_days_baseline=21, time_dim='time', flox_chunksize=8):
    """
    Compute a smoothed rolling climatology using the previous `window_year_baseline` years of data and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.
    """
    
    # N.B.: It is more efficient (chunking-wise) to smooth the raw data rather than the climatology
    da_smoothed = da.rolling({time_dim: smooth_days_baseline}, center=True).mean()

    clim = rolling_climatology(da_smoothed, window_year_baseline, time_dim)
    
    return clim


def _compute_anomaly_shifting_baseline(da, window_year_baseline=15, smooth_days_baseline=21, dimensions={'time':'time', 'xdim':'lon', 'ydim':'lat'}, flox_chunksize=8):
    """
    Compute anomalies using shifting baseline method with smoothed rolling climatology.
    Returned anomaly is chunked for cohorts.
    
    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # Compute smoothed rolling climatology
    climatology_smoothed = smoothed_rolling_climatology(da, window_year_baseline, smooth_days_baseline, time_dim=dimensions['time'])
    
    # Rechunk for efficient time operations using flox
    da_rechunked = flox.xarray.rechunk_for_cohorts(
        da,
        dim=dimensions['time'],
        labels=da[dimensions['time']].dt.dayofyear,
        chunksize=flox_chunksize,
        ignore_old_chunks=True,
        force_new_chunk_at=1
    )
    
    # Compute anomaly as difference from climatology
    anomalies = da_rechunked - climatology_smoothed
    
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

def _identify_extremes_hobday(da, threshold_percentile=95, window_days_hobday=11, exact_percentile=False, 
                             dimensions={'time':'time', 'xdim':'lon', 'ydim':'lat'}):
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
    
    
    thresholds.attrs.update({
        'long_name': f'{threshold_percentile}th percentile threshold',
        'description': f'Climatological {threshold_percentile}th percentile computed using {window_days_hobday}-day rolling window',
        'window_size': f'{window_days_hobday} days',
        'percentile': threshold_percentile,
        'method': 'day-of-year histogram approximation' if not exact_percentile else 'exact percentile'
    })
    
    # Ensure spatial dimensions are fully loaded for efficient comparison
    spatial_chunks = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    thresholds = thresholds.chunk(spatial_chunks).persist()
    
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


def _compute_anomaly_detrended(da, std_normalise=False, detrend_orders=[1], 
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

def compute_histogram_quantile_2d(da, q, window_days_hobday=11, bin_edges=None, dimensions={'time':'time', 'xdim':'lon', 'ydim':'lat'}):
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
    
    chunk_dict = {dimensions['time']: -1}
    for d in ['xdim', 'ydim']:
        if d in dimensions:
            chunk_dict[dimensions[d]] = 10
    
    da_bin = xr.DataArray(
        np.digitize(da.data, bin_edges) - 1,  # -1 so first bin is 0
        dims=da.dims,
        coords=da.coords,
        name="da_bin"
    ).chunk(chunk_dict).persist()
    
    # Construct 2D histogram using flox (in doy & anomaly)
    hist_raw = flox.xarray.xarray_reduce(
        da_bin,
        da_bin.dayofyear,
        da_bin,
        dim=[dimensions['time']],
        func="count",
        expected_groups=(np.arange(1, 367), np.arange(len(bin_edges)-1)),
        isbin=(False,False),
        dtype='int32',
        fill_value=0
    )
    
    # Pad and then window histogram
    hist_pad = hist_raw.pad({'dayofyear':window_days_hobday//2}, mode='wrap' )
    hist = hist_pad.rolling(dayofyear=window_days_hobday, center=True).sum().isel(dayofyear=slice(window_days_hobday//2, -window_days_hobday//2+1)).chunk({'dayofyear': -1})

    #### N.B.: This *temporarily* requires up to ~20x the original dataset size as scratch space
    hist = hist.persist()    # hist dims: (dayofyear, da_bin, lat, lon)
    
    # Calculate PDF and CDF
    hist_sum = hist.sum(dim="da_bin") + 1e-10
    pdf = hist / hist_sum
    cdf = pdf.cumsum(dim="da_bin")

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_centers[0] = 0.
    
    # Determine the threshold
    mask = cdf >= q
    first_true = mask.argmax(dim="da_bin")
    idx = first_true.compute()
    idx_prev = np.clip(idx - 1, 0, len(bin_centers) - 1)
    
    # Interpolate to get better estimate of the thresholds
    cdf_prev = cdf.isel({"da_bin": xr.DataArray(idx_prev, dims=first_true.dims)}).data
    cdf_next = cdf.isel({"da_bin": xr.DataArray(idx, dims=first_true.dims)}).data
    bin_prev = bin_centers[idx_prev]
    bin_next = bin_centers[idx]
    
    denom = cdf_next - cdf_prev
    frac = (q - cdf_prev) / denom
    result_data = bin_prev + frac * (bin_next - bin_prev)

    threshold = first_true.copy(data=result_data)  # dims: (dayofyear, lat, lon)
    
    return threshold
    
    


def compute_histogram_quantile_1d(da, q, dim='time', bin_edges=None):
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

def _identify_extremes_constant(da, threshold_percentile=95, exact_percentile=False, 
                                dimensions={'time':'time', 'xdim':'lon'}):
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


