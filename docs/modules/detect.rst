==============================
Detection Module (:mod:`marEx.detect`)
==============================

.. currentmodule:: marEx.detect

The :mod:`marEx.detect` module provides functions for data preprocessing, detrending,
anomaly detection, and extreme event identification. This module forms the core of the
marEx preprocessing pipeline, transforming raw oceanographic data into standardised
anomalies and binary extreme event masks.

Overview
========

The detection module implements a comprehensive workflow for converting raw oceanographic
time series data into standardised anomalies and binary extreme event masks. It supports
both structured (regular lat/lon grids) and unstructured (irregular mesh) data formats
with advanced statistical methods for robust extreme event detection.

**Key Features:**

* **Dual Anomaly Methods**: Detrended baseline vs. shifting baseline approaches
* **Flexible Extreme Detection**: Global percentile vs. day-of-year specific thresholds
* **Dask Integration**: Memory-efficient processing of large datasets with parallel computation
* **Grid Agnostic**: Works seamlessly with both structured and unstructured grids
* **Statistical Rigor**: Advanced statistical methods for robust anomaly calculation

Main Functions
==============

.. autosummary::
   :toctree: ../generated/

   preprocess_data
   compute_normalised_anomaly
   rolling_climatology
   smoothed_rolling_climatology
   identify_extremes

Detailed Documentation
======================

Main Preprocessing Function
---------------------------

.. autofunction:: preprocess_data

Core Processing Functions
-------------------------

.. autofunction:: compute_normalised_anomaly

.. autofunction:: rolling_climatology

.. autofunction:: smoothed_rolling_climatology

.. autofunction:: identify_extremes

Basic Usage Examples
====================

Simple Preprocessing
--------------------

.. code-block:: python

   import xarray as xr
   import marEx

   # Load sea surface temperature data
   sst = xr.open_dataset('sst_daily.nc', chunks={'time': 365}).sst

   # Basic preprocessing with default parameters
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95
   )

   # Result contains anomalies and extreme events
   print(extremes_ds)

Advanced Preprocessing
----------------------

.. code-block:: python

   # Advanced preprocessing with custom parameters
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=90,
       method_anomaly='shifting_baseline',      # Use rolling climatology
       method_extreme='hobday_extreme',         # Day-of-year specific thresholds
       window_year_baseline=20,                 # 20-year rolling baseline
       smooth_days_baseline=31,                 # 31-day smoothing
       window_days_hobday=11,                   # 11-day window for thresholds
       dask_chunks={'time': 25}
   )

Unstructured Grid Processing
----------------------------

.. code-block:: python

   # For unstructured grids, specify dimensions
   dimensions = {'time': 'time', 'x': 'ncells'}
   coordinates = {'time': 'time', 'x': 'lon', 'y': 'lat'}

   extremes_ds = marEx.preprocess_data(
       sst_unstructured,
       threshold_percentile=95,
       dimensions=dimensions,
       coordinates=coordinates
   )

Output Data Structure
=====================

The preprocessing function returns an xarray Dataset with the following structure:

.. code-block:: python

   # extremes_ds Dataset structure:
   xarray.Dataset
   Dimensions:     (lat, lon, time, dayofyear)
   Coordinates:
       lat         (lat)
       lon         (lon)
       time        (time)
       dayofyear   (dayofyear)    # Only for hobday_extreme method
   Data variables:
       dat_anomaly     (time, lat, lon)        float64     # Anomaly field
       mask            (lat, lon)              bool        # Land-sea mask
       extreme_events  (time, lat, lon)        bool        # Binary extreme events
       thresholds      (dayofyear, lat, lon)   float64     # Thresholds

**Key Variables:**

* **dat_anomaly**: Anomaly field (either detrended or from rolling climatology)
* **mask**: Deduced land-sea mask (True = ocean, False = land)
* **extreme_events**: Binary field locating extreme events (True = extreme)
* **thresholds**: Thresholds

Method Comparison
=================

Anomaly Detection Methods
-------------------------

**Detrended Baseline** (``method_anomaly='detrended_baseline'``):

.. code-block:: python

   # Fast polynomial detrending with harmonic components
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='detrended_baseline',
       threshold_percentile=95,
       # Additional parameters:
       std_normalise=False,          # Optional STD normalisation
       detrend_orders=[1],           # Linear detrending (default)
       force_zero_mean=True          # Enforce zero mean
   )

**Characteristics:**
  * Faster computation using polynomial fitting
  * Uses harmonic components (annual, semi-annual cycles)
  * May introduce biases in variability statistics
  * Best for: Quick analysis

**Shifting Baseline** (``method_anomaly='shifting_baseline'``):

.. code-block:: python

   # Rolling climatology from previous years
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='shifting_baseline',
       threshold_percentile=95,
       window_year_baseline=15,      # 15-year rolling baseline
       smooth_days_baseline=21       # 21-day smoothing window
   )

**Characteristics:**
  * More accurate climatology using rolling window
  * Shortens time series by baseline window length
  * Computationally intensive but scientifically rigorous
  * Best for: Research applications, intricate & accurate analysis

Extreme Event Detection Methods
-------------------------------

**Global Extreme** (``method_extreme='global_extreme'``):

.. code-block:: python

   # Single threshold across all time points
   extremes_ds = marEx.preprocess_data(
       sst,
       method_extreme='global_extreme',
       threshold_percentile=95,
       # Optional STD normalisation
       std_normalise=True
   )

**Characteristics:**
  * Uses percentiles from entire time series
  * Single threshold value for all time points
  * Simple interpretation and fast computation
  * Best for: Exploratory analysis

**Hobday Extreme** (``method_extreme='hobday_extreme'``):

.. code-block:: python

   # Day-of-year specific thresholds
   extremes_ds = marEx.preprocess_data(
       sst,
       method_extreme='hobday_extreme',
       threshold_percentile=95,
       window_days_hobday=11,        # 11-day window
       window_spatial_hobday=None    # No spatial clustering (default)
   )

**Characteristics:**
  * Day-of-year specific percentile thresholds
  * Accounts for seasonal variations
  * Follows Hobday et al. (2016) methodology

Parameter Reference
===================

Core Parameters
---------------

**threshold_percentile** : float, default=95
  Percentile threshold for extreme event identification (e.g., 95 for 95th percentile)

**method_anomaly** : {'detrended_baseline', 'shifting_baseline'}, default='detrended_baseline'
  Method for anomaly computation

**method_extreme** : {'global_extreme', 'hobday_extreme'}, default='global_extreme'
  Method for extreme identification

**dask_chunks** : dict, default={'time': 25}
  Chunk sizes for Dask arrays for memory management

Anomaly Method Parameters
-------------------------

**Detrended Baseline Parameters:**

**std_normalise** : bool, default=False
  Whether to normalise anomalies using 30-day rolling standard deviation

**detrend_orders** : list of int, default=[1]
  Polynomial orders for detrending (e.g., [1, 2] for linear + quadratic)

**force_zero_mean** : bool, default=True
  Whether to explicitly enforce zero mean in final anomalies

**Shifting Baseline Parameters:**

**window_year_baseline** : int, default=15
  Number of years for rolling climatology baseline

**smooth_days_baseline** : int, default=21
  Number of days for smoothing the rolling climatology baseline

Extreme Detection Parameters
----------------------------

**Hobday Extreme Parameters:**

**window_days_hobday** : int, default=11
  Window size for day-of-year threshold calculation

**window_spatial_hobday** : int, optional
  Spatial window size for clustering (None = no spatial clustering)

**method_percentile** : {'exact', 'approximate'}, default='approximate'
  Method for percentile calculation

**precision** : float, default=0.01
  Precision for histogram bins in approximate percentile calculation

**max_anomaly** : float, default=5.0
  Maximum anomaly value for approximate percentile calculation

Grid Configuration
------------------

**dimensions** : dict
  Dimension mapping for different grid types:

  * Structured: ``{'time': 'time', 'x': 'lon', 'y': 'lat'}``
  * Unstructured: ``{'time': 'time', 'x': 'ncells'}``

**coordinates** : dict
  Coordinate mapping for different grid types:

  * Structured: ``{'time': 'time', 'x': 'lon', 'y': 'lat'}``
  * Unstructured: ``{'time': 'time', 'x': 'lon', 'y': 'lat'}``

**neighbours** : xarray.DataArray, optional
  Neighbour connectivity array for spatial clustering (unstructured grids)

**cell_areas** : xarray.DataArray, optional
  Cell areas for weighted spatial statistics (unstructured grids)

Advanced Usage Examples
=======================

Method Combinations
-------------------

.. code-block:: python

   # Most rigorous combination (computationally intensive)
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='shifting_baseline',
       method_extreme='hobday_extreme',
       threshold_percentile=90,
       window_year_baseline=20,
       smooth_days_baseline=31,
       window_days_hobday=11
   )

   # Fastest combination (less rigorous)
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='detrended_baseline',
       method_extreme='global_extreme',
       threshold_percentile=95,
       std_normalise=False
   )

Performance Optimisations
-------------------------

.. code-block:: python

   # Optimised chunking for large datasets
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       dask_chunks={'time': 25, 'lat': 200, 'lon': 200},
       # Use approximate percentiles for speed
       method_percentile='approximate',
       precision=0.05,  # Coarser precision for speed
       max_anomaly=10.0
   )

Multi-Variable Processing
-------------------------

.. code-block:: python

   # Process multiple variables
   variables = ['sst', 'sss', 'chlorophyll']
   extreme_datasets = {}

   for var_name in variables:
       data = xr.open_dataset(f'{var_name}_daily.nc')[var_name]

       extreme_datasets[var_name] = marEx.preprocess_data(
           data,
           threshold_percentile=95,
           method_anomaly='shifting_baseline',
           method_extreme='hobday_extreme'
       )

Integration with Tracking
=========================

Complete Workflow
-----------------

.. code-block:: python

   import xarray as xr
   import marEx

   # Step 1: Load data
   sst = xr.open_dataset('sst_daily.nc', chunks={'time': 365}).sst

   # Step 2: Preprocess extremes
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='shifting_baseline',
       method_extreme='hobday_extreme',
       threshold_percentile=95,
       window_year_baseline=15,
       smooth_days_baseline=21,
       window_days_hobday=11,
       dask_chunks={'time': 25}
   )

   # Step 3: Track events
   event_tracker = marEx.tracker(
       extremes_ds.extreme_events,
       extremes_ds.mask,
       R_fill=8,
       area_filter_quartile=0.5
   )

   tracked_events = event_tracker.run()

   # Step 4: Visualise results
   config = marEx.PlotConfig(
       title='Marine Heatwave Events',
       plot_IDs=True
   )

   fig, ax, im = tracked_events.ID_field.isel(time=0).plotX.single_plot(config)

Quality Control
===============

Data Validation
---------------

.. code-block:: python

   # Check preprocessing results
   extremes_ds = marEx.preprocess_data(sst, threshold_percentile=95)

   # Validate anomaly statistics
   anomaly_stats = extremes_ds.dat_anomaly.std()
   print(f"Anomaly standard deviation: {anomaly_stats.values:.3f}")

   # Check extreme event frequency
   event_frequency = extremes_ds.extreme_events.mean() * 100
   print(f"Extreme event frequency: {event_frequency.values:.1f}%")

   # Validate mask coverage
   ocean_fraction = extremes_ds.mask.mean() * 100
   print(f"Ocean coverage: {ocean_fraction.values:.1f}%")

Threshold Validation
--------------------

.. code-block:: python

   # For hobday_extreme method, examine thresholds
   if 'thresholds' in extremes_ds:
       # Check seasonal threshold variation
       seasonal_range = (extremes_ds.thresholds.max(dim='dayofyear') -
                        extremes_ds.thresholds.min(dim='dayofyear'))
       print(f"Seasonal threshold range: {seasonal_range.mean().values:.3f}")

       # Plot threshold climatology
       threshold_clim = extremes_ds.thresholds.mean(dim=['lat', 'lon'])
       import matplotlib.pyplot as plt
       plt.plot(threshold_clim.dayofyear, threshold_clim.values)
       plt.xlabel('Day of Year')
       plt.ylabel('Threshold Value')
       plt.title('Seasonal Threshold Climatology')

Error Handling
==============

Common Issues and Solutions
---------------------------

**Memory Errors**:

.. code-block:: python

   # Solution: Optimise chunking and use approximate methods
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       dask_chunks={'time': 15},     # Smaller chunks
       method_percentile='approximate',
       precision=0.1                 # Coarser precision
   )

**Performance Issues**:

.. code-block:: python

   # Solution: Use faster methods for exploration
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       method_anomaly='detrended_baseline',
       method_extreme='global_extreme',
       std_normalise=False
   )

**Threshold Calculation Issues**:

.. code-block:: python

   # Solution: Adjust window sizes and use spatial clustering
   extremes_ds = marEx.preprocess_data(
       sst,
       method_extreme='hobday_extreme',
       window_days_hobday=21,        # Larger window
       window_spatial_hobday=5,      # Add spatial clustering
       precision=0.01                # Higher precision
   )

**Coordinate System Issues**:

.. code-block:: python

   # Solution: Specify custom dimensions and coordinates
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       dimensions={'time': 'time', 'x': 'longitude', 'y': 'latitude'},
       coordinates={'time': 'time', 'x': 'longitude', 'y': 'latitude'}
   )

Performance Benchmarks
======================

Method Performance Comparison
-----------------------------

.. code-block:: python

   # Relative performance for global 0.25° daily data:

   # Fastest: detrended_baseline + global_extreme
   # - Processing time: ~2 minutes per decade
   # - Memory usage: ~4 GB
   # - Accuracy: Good for trend analysis

   # Balanced: shifting_baseline + global_extreme
   # - Processing time: ~8 minutes per decade
   # - Memory usage: ~8 GB
   # - Accuracy: Better climatology

   # Most rigorous: shifting_baseline + hobday_extreme
   # - Processing time: ~25 minutes per decade
   # - Memory usage: ~15 GB
   # - Accuracy: Best for research applications

Scaling Characteristics
-----------------------

.. code-block:: python

   # Scaling with dask (64 cores, 25-day chunks):
   # - Linear scaling up to ~100 cores
   # - Memory usage: ~2-4 GB per core
   # - I/O becomes bottleneck beyond 200 cores
   # - Optimal chunk size depends on data resolution

Best Practices
==============

Method Selection Guidelines
---------------------------

1. **Exploratory Analysis**: Use ``detrended_baseline`` + ``global_extreme``
2. **Climate Change Research Studies**: Use ``shifting_baseline`` + ``hobday_extreme``

Chunking Guidelines
-------------------

.. code-block:: python

   # Optimal chunking strategies:

   # For global 0.25° data:
   optimal_chunks = {'time': 25, 'lat': 200, 'lon': 200}

   # For regional high-resolution data:
   optimal_chunks = {'time': 50, 'lat': 100, 'lon': 100}

   # For unstructured grids:
   optimal_chunks = {'time': 25, 'ncells': 50000}

See Also
========

* :mod:`marEx.track` - Event tracking module
* :mod:`marEx.plotX` - Visualisation module
* :mod:`marEx.helper` - HPC utilities
