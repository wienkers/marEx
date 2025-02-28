"""
Marine heatwave pre-processing module for converting sea surface temperature data
to standardised anomalies and identifying extreme (temperature) events.

This module implements preprocessing steps for marine heatwave detection including:
- Detrending and removing seasonal cycle
- Normalisation using rolling 30-day standard deviation
- Threshold-based extreme event identification
"""

import numpy as np
import pandas as pd
import xarray as xr
import dask
import flox.xarray


def add_decimal_year(da, dim='time'):
    """
    Calculate decimal year from datetime coordinate and add as a new coordinate.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data array with datetime coordinate named 'time'
    
    Returns
    -------
    da : xarray.DataArray
        Input data array with 'decimal_year' coordinate added
    """
    
    time = pd.to_datetime(da[dim])
    start_of_year = pd.to_datetime(time.year.astype(str) + '-01-01')
    start_of_next_year = pd.to_datetime((time.year + 1).astype(str) + '-01-01')
    year_elapsed = (time - start_of_year).days
    year_duration = (start_of_next_year - start_of_year).days
    
    decimal_year = time.year + year_elapsed / year_duration

    da = da.assign_coords(decimal_year=(dim, decimal_year))
    
    return da


def rechunk_for_cohorts(da, chunksize=100, dim='time'):
    """
    Rechunk data array for efficient climatology & day-of-year STD reductions.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    chunksize : int, optional
        Chunk size for rechunking
        
    Returns
    -------
    xarray.DataArray
        Rechunked data array
    """
    
    da = flox.xarray.rechunk_for_cohorts(da, 
                                         dim=dim, 
                                         labels=da[dim].dt.dayofyear, 
                                         force_new_chunk_at=1, 
                                         chunksize=chunksize, 
                                         ignore_old_chunks=True)
    
    return da


def compute_normalised_anomaly(da, std_normalise=False, detrend_orders=[1], 
                        dask_chunks={'time': 25}, 
                        dimensions={'time':'time', 'xdim':'lon', 'ydim':'lat'},
                        force_zero_mean=True):
    """
    Standardise data by:
    1. Removing trend and seasonal cycle using a model with:
       - Mean
       - Polynomial trends of specified orders
       - Annual & semi-annual harmonics
    2. Optionally dividing by 30-day rolling standard deviation
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data with dimensions (time, lat, lon)
    std_normalise : bool, optional
        Whether to normalize by standard deviation
    detrend_orders : list, optional
        List of polynomial orders to use for detrending.
        Example: [1] for linear only, [1,2] for linear+quadratic,
        [1,2,3] for linear+quadratic+cubic
    dask_chunks : dict, optional
        Chunking specification for dask arrays
    dimensions : dict, optional
        Dictionary mapping dimension names
    force_zero_mean : bool, optional
        Whether to explicitly force zero mean in detrended data
        
    Returns
    -------
    xarray.Dataset
        Dataset containing:
        - Raw (Detrended) and STD normalised anomalies
        - Rolling standard deviation
        - Ocean/land mask
    """
    
    # Ensure the time dimension is the first dimension
    if da.dims[0] != dimensions['time']:
        da = da.transpose(dimensions['time'], ...)
    
    # Add decimal year coordinate to data array
    da = add_decimal_year(da)
    
    # Start with constant term for the model
    model_components = [np.ones(len(da.decimal_year))]
    
    # Add polynomial trend terms based on detrend_orders
    centered_time = da.decimal_year - np.mean(da.decimal_year)
    for order in detrend_orders:
        model_components.append(centered_time ** order)
    
    # Add annual and semi-annual harmonics
    model_components.extend([
        np.sin(2 * np.pi * da.decimal_year),
        np.cos(2 * np.pi * da.decimal_year),
        np.sin(4 * np.pi * da.decimal_year),
        np.cos(4 * np.pi * da.decimal_year)
    ])
    
    # Convert to numpy array
    model = np.array(model_components)
    
    # Orthogonalise model components to ensure higher-order terms have 0-mean
    for i in range(1, model.shape[0]):
        # Remove projection onto constant term
        model[i] = model[i] - np.mean(model[i]) * model[0]
    
    # Take pseudo-inverse of model
    pmodel = np.linalg.pinv(model)
    
    # Number of coefficients
    n_coeffs = len(model_components)
    
    # Convert to xarray DataArrays
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
    )
    
    # Calculate model coefficients
    model_fit_da = xr.DataArray(
        pmodel_da.dot(da),
        dims=['coeff', dimensions['ydim'], dimensions['xdim']],
        coords={
            'coeff': np.arange(1, n_coeffs+1),
            dimensions['ydim']: da[dimensions['ydim']].values,
            dimensions['xdim']: da[dimensions['xdim']].values
        }
    )
    
    # Remove trend and seasonal cycle
    da_detrend = (da - model_da.dot(model_fit_da))
    
    # Force zero mean if requested
    if force_zero_mean:
        da_detrend = da_detrend - da_detrend.mean(dim=dimensions['time'])
    
    # Rechunk the data
    da_detrend = da_detrend.chunk({
        'time': dask_chunks['time'], 
        dimensions['ydim']: -1, 
        dimensions['xdim']: -1
    })
    
    # Create mask from first timestep
    mask = np.isfinite(da.isel({dimensions['time']: 0})).\
           chunk({dimensions['ydim']: -1, dimensions['xdim']: -1}).\
           drop_vars({'decimal_year', 'time'})
    
    data_vars = {
        'dat_detrend': da_detrend.drop_vars({'decimal_year'}),
        'mask': mask
    }
    
    ## Standardise Data Anomalies
    #  This step places equal variance on anomaly at all spatial points
    if std_normalise: 
        # Calculate daily standard deviation
        std_day = flox.xarray.xarray_reduce(
            da_detrend,
            da_detrend[dimensions['time']].dt.dayofyear,
            dim=dimensions['time'],
            func='std',
            isbin=False,
            method='cohorts'
        )
        
        # Calculate 30-day rolling standard deviation
        std_day_wrap = std_day.pad(dayofyear=16, mode='wrap')
        std_rolling = np.sqrt(
            (std_day_wrap**2)
            .rolling(dayofyear=30, center=True)
            .mean()
        ).isel(dayofyear=slice(16, 366+16))
        
        # STD Normalised anomalies
        da_stn = da_detrend.groupby(da_detrend[dimensions['time']].dt.dayofyear) / std_rolling
        
        # Rechunk the data
        da_stn = da_stn.chunk({
            'time': dask_chunks['time'], 
            dimensions['ydim']: -1, 
            dimensions['xdim']: -1
        })
        std_rolling = std_rolling.chunk({
            'dayofyear': -1, 
            dimensions['ydim']: -1, 
            dimensions['xdim']: -1
        })
        
        data_vars['dat_stn'] = da_stn.drop_vars({'dayofyear', 'decimal_year'})
        data_vars['STD'] = std_rolling
    
    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            'description': 'Standardised & Detrended Data',
            'preprocessing_steps': [
                f'Removed {"polynomial trend orders=" + str(detrend_orders)} & seasonal cycle',
                'Normalise by 30-day rolling STD' if std_normalise else 'No STD normalization'
            ],
            'detrend_orders': detrend_orders,
            'force_zero_mean': force_zero_mean
        }
    )



def identify_extremes(da, threshold_percentile=95, dask_chunks={'time': 25}, dimensions={'time':'time', 'xdim':'lon', 'ydim':'lat'}):
    """
    Identify extreme events above a percentile threshold.
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing detrended data
    threshold_percentile : float, optional
        Percentile threshold for extreme event identification
        
    Returns
    -------
    xarray.DataArray
        Boolean array marking extreme events
    """
    
    # Rechunk the dataset for quantile calculation
    da = da.chunk({dimensions['time']: -1, dimensions['ydim']: 'auto', dimensions['xdim']: 'auto'})
    
    # Calculate threshold
    threshold = da.quantile(threshold_percentile/100.0, dim=dimensions['time'])
    
    # Identify points above threshold
    extremes = da >= threshold
    
    extremes = extremes.chunk({dimensions['time']:dask_chunks['time'], dimensions['ydim']:-1, dimensions['xdim']:-1})
    extremes = extremes.drop_vars('quantile')
    
    return extremes



def preprocess_data(da, std_normalise=False, threshold_percentile=95, 
                   detrend_orders=[1], force_zero_mean=True,
                   dask_chunks={'time': 25}, 
                   dimensions={'time':'time', 'xdim':'lon', 'ydim':'lat'}):
    """
    Complete preprocessing pipeline from raw Data to extreme event identification.
    
    Parameters
    ----------
    da : xarray.DataArray
        Raw input data
    std_normalise : bool, optional
        Additionally compute the Normalised/Standardised (by STD) Anomalies
    threshold_percentile : float, optional
        Percentile threshold for extremes
    detrend_orders : list, optional
        List of polynomial orders to use for detrending
    force_zero_mean : bool, optional
        Whether to explicitly force zero mean in detrended data
    dask_chunks : dict, optional
        Chunking specification
    dimensions : dict, optional
        Dictionary mapping dimension names
        
    Returns
    -------
    xarray.Dataset
        Processed dataset containing normalised anomalies and extreme events
    """
    
    # Compute Anomalies and Normalise/Standardise
    ds = compute_normalised_anomaly(
        da, std_normalise, 
        detrend_orders=detrend_orders,
        force_zero_mean=force_zero_mean,
        dask_chunks=dask_chunks, 
        dimensions=dimensions
    )
    
    # Identify extreme events:  
    extremes_detrend = identify_extremes(
        ds.dat_detrend,
        threshold_percentile=threshold_percentile,
        dask_chunks=dask_chunks,
        dimensions=dimensions
    )
    ds['extreme_events'] = extremes_detrend
    
    if std_normalise:  # Also compute normalised/standardised anomalies
        extremes_stn = identify_extremes(
            ds.dat_stn,
            threshold_percentile=threshold_percentile,
            dask_chunks=dask_chunks,
            dimensions=dimensions
        )
        ds['extreme_events_stn'] = extremes_stn
    
    # Add processing parameters to attributes
    ds.attrs.update({
        'threshold_percentile': threshold_percentile,
        'detrend_orders': detrend_orders,
        'force_zero_mean': force_zero_mean,
        'std_normalise': std_normalise
    })
    
    return ds