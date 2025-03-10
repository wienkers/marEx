"""
MarEx-Detect: Marine Extremes Detection Module

Preprocessing toolkit for marine extremes identification from scalar oceanographic data.
Converts raw time series into standardised anomalies and identifies extreme events
(e.g., Marine Heatwaves using Sea Surface Temperature).

Core capabilities:
- Detrending with polynomial fitting and seasonal cycle removal
- Optional standardisation using rolling statistics
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
from dask.base import is_dask_collection
import flox.xarray
from xhistogram.xarray import histogram
import logging

logging.getLogger('distributed.shuffle._scheduler_plugin').setLevel(logging.ERROR)


# ============================
# Data Preparation Functions
# ============================

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


# ============================
# Core Anomaly Processing
# ============================

def compute_normalised_anomaly(da, std_normalise=False, detrend_orders=[1], 
                                dimensions={'time':'time', 'xdim':'lon', 'ydim':'lat'},
                                force_zero_mean=True):
    """
    Generate normalised anomalies by removing trends, seasonal cycles, and optionally
    standardising by temporal variability.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data with dimensions matching the 'dimensions' parameter
    std_normalise : bool, optional
        Whether to normalise by 30-day rolling standard deviation
    detrend_orders : list, optional
        Polynomial orders for trend removal (e.g., [1] for linear, [1,2] for linear+quadratic, etc.)
    dimensions : dict, optional
        Mapping of conceptual dimensions to actual dimension names in the data
    force_zero_mean : bool, optional
        Explicitly enforce zero mean in final anomalies
        
    Returns
    -------
    xarray.Dataset
        Dataset containing detrended anomalies, optional standardised anomalies,
        ocean/land mask, and rolling standard deviation if requested
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
            'preprocessing_steps': [
                f'Removed {"polynomial trend orders=" + str(detrend_orders)} & seasonal cycle',
                'Normalised by 30-day rolling STD' if std_normalise else 'No STD normalisation'
            ],
            'detrend_orders': detrend_orders,
            'force_zero_mean': force_zero_mean
        }
    )


# ============================
# Extreme Event Identification
# ============================

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


def identify_extremes(da, threshold_percentile=95, exact_percentile=False, 
                     dimensions={'time':'time', 'xdim':'lon'}):
    """
    Identify extreme events exceeding a percentile threshold.
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing detrended anomalies
    threshold_percentile : float, optional
        Percentile threshold (e.g., 95 for 95th percentile)
    exact_percentile : bool, optional
        Whether to compute exact percentiles (slower) or use histogram approximation (faster)
    dimensions : dict, optional
        Mapping of dimension types to names in the data
        
    Returns
    -------
    xarray.DataArray
        Boolean array identifying extreme events
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
    
    # Create boolean mask for values exceeding threshold
    extremes = da >= threshold
    
    # Clean up coordinates if needed
    if 'quantile' in extremes.coords:
        extremes = extremes.drop_vars('quantile')
    
    return extremes


# ============================
# Main Processing Pipeline
# ============================

def preprocess_data(da, std_normalise=False, threshold_percentile=95, 
                   detrend_orders=[1], force_zero_mean=True,
                   exact_percentile=False, dask_chunks={'time': 25}, 
                   dimensions={'time':'time', 'xdim':'lon'}, 
                   neighbours=None, cell_areas=None):
    """
    Complete preprocessing pipeline for marine extreme event identification.
    
    Workflow:
    1. Compute detrended (and optionally standardised) anomalies
    2. Identify values exceeding the percentile threshold
    3. Attach optional spatial metadata (neighbour connectivity, cell areas)
    
    Parameters
    ----------
    da : xarray.DataArray
        Raw input data
    std_normalise : bool, optional
        Whether to standardise anomalies by rolling standard deviation
    threshold_percentile : float, optional
        Percentile threshold for extreme event identification
    detrend_orders : list, optional
        Polynomial orders for detrending
    force_zero_mean : bool, optional
        Whether to enforce zero mean in detrended anomalies
    exact_percentile : bool, optional
        Whether to use exact or approximate percentile calculation
    dask_chunks : dict, optional
        Chunking specification for distributed computation
    dimensions : dict, optional
        Mapping of dimension types to names in the data
    neighbours : xarray.DataArray, optional
        Neighbour connectivity for spatial clustering (optional)
    cell_areas : xarray.DataArray, optional
        Cell areas for weighted spatial statistics (optional)
    
    Returns
    -------
    xarray.Dataset
        Processed dataset with anomalies and extreme event identification
    """
    # Check if input data is dask-backed
    if not is_dask_collection(da.data):
        raise ValueError('The input DataArray must be backed by a Dask array. Ensure the input data is chunked, e.g. with chunks={}')
    
    # Step 1: Compute anomalies with detrending and optional standardisation
    ds = compute_normalised_anomaly(
        da, 
        std_normalise, 
        detrend_orders=detrend_orders,
        force_zero_mean=force_zero_mean,
        dimensions=dimensions
    )
    
    # Prevent dask from splitting chunks during slicing operations (otherwise dask task graph may be overwhelmed!)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        
        # Step 2: Identify extreme events in detrended anomalies
        extremes_detrend = identify_extremes(
            ds.dat_detrend,
            threshold_percentile=threshold_percentile,
            exact_percentile=exact_percentile,
            dimensions=dimensions
        )
        ds['extreme_events'] = extremes_detrend
        
        # Also process standardised anomalies if requested
        if std_normalise:
            extremes_stn = identify_extremes(
                ds.dat_stn,
                threshold_percentile=threshold_percentile,
                exact_percentile=exact_percentile,
                dimensions=dimensions
            )
            ds['extreme_events_stn'] = extremes_stn
        
        # Step 3: Add optional spatial metadata
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
        'threshold_percentile': threshold_percentile,
        'detrend_orders': detrend_orders,
        'force_zero_mean': force_zero_mean,
        'std_normalise': std_normalise
    })
    
    # Final rechunking for optimal performance
    chunk_dict = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    chunk_dict[dimensions['time']] = dask_chunks['time']
    ds = ds.chunk(chunk_dict)
    
    # Fix encoding issue with saving when calendar & units attribute is present
    if 'calendar' in ds[dimensions['time']].attrs:
        del ds[dimensions['time']].attrs['calendar']
    if 'units' in ds[dimensions['time']].attrs:
        del ds[dimensions['time']].attrs['units']
    
    return ds