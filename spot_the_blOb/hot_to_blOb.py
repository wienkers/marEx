"""
Marine heatwave pre-processing module for converting sea surface temperature data
to standardised anomalies and identifying extreme (temperature) events.

This module implements preprocessing steps for marine heatwave detection including:
- Detrending and removing seasonal cycle
- Normalisation using rolling 30-day standard deviation
- Threshold-based extreme event identification

Modified to handle both 2D (time, xdim), i.e. when unstructured grid, and 3D (time, ydim, xdim) data
"""

import numpy as np
import pandas as pd
import xarray as xr
import dask
import flox.xarray
from xhistogram.xarray import histogram


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


def compute_normalised_anomaly(da, std_normalise=False, dimensions={'time':'time', 'xdim':'lon'}):
    """
    Standardise data by:
    1. Removing trend and seasonal cycle using a 6-coefficient model (mean, trend, annual & semi-annual harmonics)
    2. Dividing by 30-day rolling standard deviation
    
    Parameters
    ----------
    sst : xarray.DataArray
        Input data with dimensions (time, xdim) or (time, ydim, xdim)
    dask_chunks : dict, optional
        Chunking specification for dask arrays
    dimensions : dict, optional
        Dictionary mapping dimension types to actual dimension names
        Must contain 'time' and 'xdim', optionally 'ydim'
        
    Returns
    -------
    xarray.Dataset
        Dataset containing:
        - Raw (Detrended) and STD normalised anomalies
        - Rolling standard deviation
        - Ocean/land mask
    """
    
    da = da.astype(np.float32)
    
    # Ensure the time dimension is the first dimension
    if da.dims[0] != dimensions['time']:
        da = da.transpose(dimensions['time'], ...)
    
    # Add decimal year coordinate to data array
    da = add_decimal_year(da)
    dy = da.decimal_year.compute()
    
    # Construct model for detrending
    # 6 coefficient model: mean, trend, annual & semi-annual harmonics
    model = np.array([
        np.ones(len(dy)),
        dy - np.mean(dy),
        np.sin(2 * np.pi * dy),
        np.cos(2 * np.pi * dy),
        np.sin(4 * np.pi * dy),
        np.cos(4 * np.pi * dy)
    ])
    
    # Take pseudo-inverse of model
    pmodel = np.linalg.pinv(model)
    
    # Convert to xarray DataArrays
    model_da = xr.DataArray(
        model.T, 
        dims=[dimensions['time'],'coeff'], 
        coords={dimensions['time']: da[dimensions['time']].values, 'coeff': np.arange(1,7,1)}
    ).chunk({dimensions['time']: da.chunks[0]})
    
    pmodel_da = xr.DataArray(
        pmodel.T,
        dims=['coeff',dimensions['time']],
        coords={'coeff': np.arange(1,7,1), dimensions['time']: da[dimensions['time']].values}
    ).chunk({dimensions['time']: da.chunks[0]})
    
    # Calculate model coefficients - handle both 2D and 3D cases
    if 'ydim' in dimensions:
        model_fit_da = xr.DataArray(
            pmodel_da.dot(da),
            dims=['coeff', dimensions['ydim'], dimensions['xdim']],
            coords={
                'coeff': np.arange(1,7,1),
                dimensions['ydim']: da[dimensions['ydim']].values,
                dimensions['xdim']: da[dimensions['xdim']].values
            }
        )
    else:
        model_fit_da = xr.DataArray(
            pmodel_da.dot(da),
            dims=['coeff', dimensions['xdim']],
            coords={
                'coeff': np.arange(1,7,1),
                **da[dimensions['xdim']].coords
            }
        )
    
    # Remove trend and seasonal cycle
    da_detrend = (da.drop_vars({'decimal_year'}) - model_da.dot(model_fit_da).astype(np.float32))
    
    # Create mask
    mask = np.isfinite(da.isel({dimensions['time']:0}))
    chunk_dict = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    mask = mask.chunk(chunk_dict)
    mask = mask.drop_vars({'decimal_year', 'time'})
    
    data_vars = {
        'dat_detrend': da_detrend,
        'mask': mask
    }    
    
    ## Standardise Data Anomalies
    #  This step places equal variance on anomaly at all spatial points
    if std_normalise: 
        print('N.B.: It is _highly_ recommended to use `rechunk_for_cohorts` on the input data before proceeding with STD normalisation.')
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
        ).isel(dayofyear=slice(16,366+16))
        
        # STD Normalised anomalies
        da_stn = da_detrend.groupby(da_detrend[dimensions['time']].dt.dayofyear) / std_rolling
        
        chunk_dict_std = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
        chunk_dict_std['dayofyear'] = -1
        std_rolling = std_rolling.chunk(chunk_dict_std)
        
        data_vars['dat_stn'] = da_stn.drop_vars({'dayofyear', 'decimal_year'})
        data_vars['STD'] = std_rolling
    
    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            'description': 'Standardised & Detrended Data',
            'preprocessing_steps': [
                'Removed trend & seasonal cycle'
            ]
        }
    )
    
def compute_histogram_quantile(da, q, dim='time'):
    """
    Compute quantiles efficiently by constructing a histogram with custom bins
    optimised for positive anomalies. Uses a single bin for negative values and
    bins of specified precision for positive values.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data array with dimensions including 'time' and spatial dimensions
    q : float
        Quantile to compute (between 0 and 1)
    dim : str, optional
        Dimension along which to compute quantile
        
    Returns
    -------
    xarray.DataArray
        Computed quantile values along the specified dimension
    """
    
    precision = 0.025
    max_anomaly = 10.0
    
    # Create custom bin edges
    bin_edges = np.concatenate([
        [-np.inf, 0.],  # Single bin for negative values
        np.arange(precision, max_anomaly+precision, precision)  # Bins of size "precision" from 0 to "max_anomaly"
    ])
    
    hist = histogram(
        da,
        bins=[bin_edges],
        dim=[dim]
    )
    
    pdf = hist / hist.sum(dim='dat_detrend_bin')
    cdf = pdf.cumsum(dim='dat_detrend_bin')
    
    # Get bin centers for interpolation -- special case for negative bin...
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_centers[0] = 0.  # Set negative bin center to 0
    
    # Find the bins where CDF crosses the desired quantile
    mask = cdf >= q
    
    # Get the first True along the histogram dimension
    first_true = mask.argmax(dim='dat_detrend_bin')
    
    # Convert bin indices to actual values using bin centers
    result = first_true.copy(data=bin_centers[first_true])
    
    return result


def identify_extremes(da, threshold_percentile=95, exact_percentile=False, dimensions={'time':'time', 'xdim':'lon'}):
    """
    Identify extreme events above a percentile threshold.
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing detrended data
    threshold_percentile : float, optional
        Percentile threshold for extreme event identification
    dimensions : dict, optional
        Dictionary mapping dimension types to actual dimension names
        Must contain 'time' and 'xdim', optionally 'ydim'
        
    Returns
    -------
    xarray.DataArray
        Boolean array marking extreme events
    """
    
    if exact_percentile:  # Compute the Percentile exactly (by rechunking contiguous in time)
        
        # Create dynamic chunking dictionary for quantile calculation
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
    
    # Identify points above threshold
    extremes = da >= threshold
    
    if 'quantile' in extremes.coords:
        extremes = extremes.drop_vars('quantile')
    
    return extremes


def preprocess_data(da, std_normalise=False, threshold_percentile=95, exact_percentile=False, dask_chunks={'time': 25}, neighbours = None, cell_areas = None, dimensions={'time':'time', 'xdim':'lon'}):
    """
    Complete preprocessing pipeline from raw Data to extreme event identification.
    
    Parameters
    ----------
    da : xarray.DataArray
        Raw input data with dimensions (time, xdim) or (time, ydim, xdim)
    std_normalise=True : bool, optional
        Additionally compute the Normalised/Standardised (by STD) Anomalies
    threshold_percentile : float, optional
        Percentile threshold for extremes
    exact_percentile : bool, optional
        Whether to exactly compute the percentile (rechunking in time), or assemble a histogram and estimate the quantile
    dask_chunks : dict, optional
        Chunking specification
    dimensions : dict, optional
        Dictionary mapping dimension types to actual dimension names
        Must contain 'time' and 'xdim', optionally 'ydim'
        
    Returns
    -------
    xarray.Dataset
        Processed dataset containing normalised anomalies and extreme events
    """
    
    # Compute Anomalies and Normalise/Standardise
    ds = compute_normalised_anomaly(
        da, 
        std_normalise, 
        dimensions=dimensions
    )#.persist()
    
    # Don't rechunk... otherwise there can be many many tasks in the dask graph...
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        
        # Identify extreme events
        extremes_detrend = identify_extremes(
            ds.dat_detrend,
            threshold_percentile=threshold_percentile,
            exact_percentile=exact_percentile,
            dimensions=dimensions
        )
        ds['extreme_events'] = extremes_detrend
        
        if std_normalise:  # Also compute normalised/standardised anomalies
            extremes_stn = identify_extremes(
                ds.dat_stn,
                threshold_percentile=threshold_percentile,
                exact_percentile=exact_percentile,
                dimensions=dimensions
            )
            ds['extreme_events_stn'] = extremes_stn
        
        
        if neighbours is not None:  # Add neighbours to dataset with appropriate chunking
            chunk_dict = {dim: -1 for dim in neighbours.dims}
            ds['neighbours'] = neighbours.astype(np.int32).chunk(chunk_dict)
            
            if 'nv' in neighbours.dims:
                ds = ds.assign_coords(nv=neighbours.nv)
        
        if cell_areas is not None: # Add cell areas to dataset with appropriate chunking
            chunk_dict = {dim: -1 for dim in cell_areas.dims}
            ds['cell_areas'] = cell_areas.astype(np.float32).chunk(chunk_dict)
        
        # Add processing parameters to attributes
        ds.attrs.update({
            'threshold_percentile': threshold_percentile
        })
    
    ## Rechunk the entire ds:
    chunk_dict = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    chunk_dict[dimensions['time']] = dask_chunks['time']
    ds = ds.chunk(chunk_dict)
    
    return ds