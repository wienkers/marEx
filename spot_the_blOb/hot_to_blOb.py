"""
Marine extremes pre-processing module for converting scalar data
into standardised anomalies and identifying extreme events.
(e.g. Marine Heatwave (MHW) ID using SST)

This module implements preprocessing steps (for e.g. MHW detection) including:
- Detrending and removing seasonal cycle
- Normalisation using rolling 30-day standard deviation
- Threshold-based extreme event identification

Works with both unstructured & structured data:
- Structured data:   3D (time, ydim, xdim) data
- Unstructured data: 2D (time, xdim) data
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


def compute_normalised_anomaly(da, std_normalise=False, detrend_orders=[1], 
                                dask_chunks={'time': 25}, 
                                dimensions={'time':'time', 'xdim':'lon', 'ydim':'lat'},
                                force_zero_mean=True):
    """
    Standardise data by:
    1. Removing trend and seasonal cycle using a model with:
       - Mean
       - Polynomial trends of arbitrary orders
       - Annual & semi-annual harmonics
    2. Optionally dividing by 30-day rolling standard deviation
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data with dimensions (time, lat, lon)
    std_normalise : bool, optional
        Whether to normalise by standard deviation
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
    
    da = da.astype(np.float32)
    
    # Ensure the time dimension is the first dimension
    if da.dims[0] != dimensions['time']:
        da = da.transpose(dimensions['time'], ...)
    
    # Post warning if using higher-order detrending _without_ linear
    if 1 not in detrend_orders and len(detrend_orders) > 1:
        print('Warning: Higher-order detrending _without_ linear term may not be well-posed!')
        
    
    # Add decimal year coordinate to data array
    da = add_decimal_year(da)
    dy = da.decimal_year.compute()
    
    # Start with constant term for the model
    model_components = [np.ones(len(dy))]
    
    # Add polynomial trend terms based on detrend_orders
    centered_time = da.decimal_year - np.mean(dy)
    for order in detrend_orders:
        model_components.append(centered_time ** order)
    
    # Add annual and semi-annual harmonics
    model_components.extend([
        np.sin(2 * np.pi * dy),
        np.cos(2 * np.pi * dy),
        np.sin(4 * np.pi * dy),
        np.cos(4 * np.pi * dy)
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
    ).chunk({dimensions['time']: da.chunks[0]})
    
    # Calculate model coefficients - handle both 2D and 3D cases
    dims = ['coeff']
    coords = {'coeff': np.arange(1, n_coeffs + 1)}
    if 'ydim' in dimensions:
        dims.extend([dimensions['ydim'], dimensions['xdim']])
        coords[dimensions['ydim']] = da[dimensions['ydim']].values
        coords[dimensions['xdim']] = da[dimensions['xdim']].values
    else: # Unstructured
        dims.append(dimensions['xdim'])
        coords.update(da[dimensions['xdim']].coords)

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
    
    # Create mask
    chunk_dict_mask = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    mask = np.isfinite(da.isel({dimensions['time']: 0})).chunk(chunk_dict_mask).drop_vars({'decimal_year', 'time'})
    
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
        ).isel(dayofyear=slice(16, 366+16))
        
        # STD Normalised anomalies
        da_stn = da_detrend.groupby(da_detrend[dimensions['time']].dt.dayofyear) / std_rolling
        
        # Rechunk the data
        chunk_dict_std = chunk_dict_mask
        chunk_dict_std['dayofyear'] = -1
        
        da_stn = da_stn.chunk(chunk_dict_std)
        std_rolling = std_rolling.chunk(chunk_dict_std)
        data_vars['dat_stn'] = da_stn.drop_vars({'dayofyear', 'decimal_year'})
        data_vars['STD'] = std_rolling
    
    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            'description': 'Standardised & Detrended Data',
            'preprocessing_steps': [
                f'Removed {"polynomial trend orders=" + str(detrend_orders)} & seasonal cycle',
                'Normalise by 30-day rolling STD' if std_normalise else 'No STD normalisation'
            ],
            'detrend_orders': detrend_orders,
            'force_zero_mean': force_zero_mean
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


def preprocess_data(da, std_normalise=False, threshold_percentile=95, 
                    detrend_orders=[1], force_zero_mean=True,
                    exact_percentile=False, dask_chunks={'time': 25}, 
                    dimensions={'time':'time', 'xdim':'lon'}, neighbours = None, cell_areas = None, ):
    """
    Complete preprocessing pipeline from raw Data to extreme event identification.
    
    Parameters
    ----------
    da : xarray.DataArray
        Raw input data with dimensions (time, xdim) or (time, ydim, xdim)
    std_normalise : bool, optional
        Additionally compute the Normalised/Standardised (by STD) Anomalies
    threshold_percentile : float, optional
        Percentile threshold for extremes
    detrend_orders : list, optional
        List of polynomial orders to use for detrending
    force_zero_mean : bool, optional
        Whether to explicitly force zero mean in detrended data
    exact_percentile : bool, optional
        Whether to exactly compute the percentile (rechunking in time), or assemble a histogram and estimate the quantile
    dask_chunks : dict, optional
        Chunking specification
    dimensions : dict, optional
        Dictionary mapping dimension types to actual dimension names
        Must contain 'time' and 'xdim', optionally 'ydim'
    neighbours & cell_areas : xarray.DataArray, optional
        Neighbours and cell areas for each grid cell
    
    Returns
    -------
    xarray.Dataset
        Processed dataset containing normalised anomalies and extreme events
    """
    
    # Compute Anomalies and Normalise/Standardise
    ds = compute_normalised_anomaly(
        da, 
        std_normalise, 
        detrend_orders=detrend_orders,
        force_zero_mean=force_zero_mean,
        dask_chunks=dask_chunks, 
        dimensions=dimensions
    )
    
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
        'threshold_percentile': threshold_percentile,
        'detrend_orders': detrend_orders,
        'force_zero_mean': force_zero_mean,
        'std_normalise': std_normalise
    })
    
    ## Rechunk the entire ds:
    chunk_dict = {dimensions[dim]: -1 for dim in ['xdim', 'ydim'] if dim in dimensions}
    chunk_dict[dimensions['time']] = dask_chunks['time']
    ds = ds.chunk(chunk_dict)
    
    return ds