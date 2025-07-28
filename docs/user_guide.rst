==========
User Guide
==========

This guide covers all aspects of using marEx for marine extreme detection and tracking, from basic concepts to advanced workflows.

.. contents:: Table of Contents
   :local:
   :depth: 3

Introduction
============

marEx is a Python package for efficient identification and tracking of marine extremes (e.g., Marine Heatwaves) in oceanographic data. It provides a complete pipeline from raw data preprocessing to tracked event visualisation.

Key Features
------------

* **Flexible Data Support**: Both structured (lat/lon grids) and unstructured (irregular meshes) data
* **Multiple Methods**: Various anomaly computation and threshold definition methods
* **Parallel Processing**: Built on Dask for efficient computation on large datasets
* **Event Tracking**: Sophisticated algorithms for tracking coherent events through time
* **Rich Visualisation**: Automatic plotting system with publication-ready figures
* **HPC Integration**: Optimised for supercomputing environments

Installation and Setup
======================

Basic Installation
------------------

Install marEx from PyPI::

   pip install marEx

For full functionality including GPU acceleration::

   pip install marEx[full]

For development work::

   pip install marEx[dev]

System Requirements
-------------------

* **Python**: 3.10 or higher
* **Memory**: Minimum 8GB RAM, 32GB+ recommended for large datasets
* **Storage**: Lustre system recommended for large Zarr datasets
* **Optional**: GPU for JAX acceleration

Quick Setup Check
-----------------

.. code-block:: python

   import marEx

   # Check installation
   print(f"marEx version: {marEx.__version__}")

   # Check dependencies
   marEx.print_dependency_status()

   # Start local cluster
   client = marEx.helper.start_local_cluster()

Data Preparation
================

Input Data Requirements
-----------------------

marEx works with xarray DataArrays containing oceanographic time series data:

**Required Dimensions:**

* **Time dimension**: Regular or irregular time steps (daily to monthly)
* **Spatial dimensions**: Either structured (lat, lon) or unstructured (cell/ncells)

**Supported Formats:**

* NetCDF (.nc)
* Zarr (.zarr)
* Any xarray-compatible format

**Data Structure Examples:**

For structured/gridded data::

   data.dims = ('time', 'lat', 'lon')
   # Coordinates: time, lat, lon as dimensions

For unstructured data::

   data.dims = ('time', 'ncells')
   # Coordinates: time as dimension, lat/lon as coordinates

Data Quality Requirements
-------------------------

* **Temporal coverage**: Minimum 10 years for robust climatology
* **Spatial coverage**: Consistent grid throughout time series

Data Loading Examples
---------------------

**Loading NetCDF files:**

.. code-block:: python

   import xarray as xr

   # Single file
   ds = xr.open_dataset('sst_data.nc', chunks={})
   sst = ds.sst

   # Multiple files
   ds = xr.open_mfdataset('sst_*.nc', parallel=True, chunks={})
   sst = ds.sst

**Loading Zarr datasets:**

.. code-block:: python

   # Local Zarr
   ds = xr.open_zarr('data.zarr', chunks={})

   # Cloud-optimised Zarr
   ds = xr.open_zarr('gs://bucket/data.zarr', chunks={})

**Optimising data loading:**

.. code-block:: python

   # Apply chunking for efficient processing
   sst = sst.chunk({'time': 365, 'lat': 50, 'lon': 100})

   # Ensure Dask backing
   if not marEx.is_dask_collection(sst.data):
       sst = sst.chunk()

Core Workflow
=============

The marEx workflow consists of three main steps:

1. **Preprocessing**: Convert raw data to anomalies and identify extremes
2. **Tracking**: Track extreme events through time
3. **Visualisation**: Analyse and visualise results

Step 1: Preprocessing
---------------------

The preprocessing step transforms raw oceanographic data into anomalies and detects extreme event locations.

**Basic preprocessing:**

.. code-block:: python

   import marEx

   # Basic preprocessing with default settings
   extremes = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       method_anomaly='detrended_baseline',
       method_extreme='global_extreme'
   )

**Advanced configuration:**

.. code-block:: python

   # Advanced preprocessing configuration
   extremes = marEx.preprocess_data(
       sst,
       # Anomaly computation method
       method_anomaly='shifting_baseline',  # or 'detrended_baseline'
       window_year_baseline=15,             # For shifting baseline
       smooth_days_baseline=21,             # Smoothing for climatology

       # Extreme identification method
       method_extreme='hobday_extreme',     # or 'global_extreme'
       threshold_percentile=95,             # 95th percentile
       window_days_hobday=11,               # Days around each day of year
       window_spatial_hobday=5,             # Spatial window for percentile calculation

       # Output options
       dask_chunks={'time': 25}
   )

The resulting xarray dataset ``extremes`` will have the following structure & entries:
```
xarray.Dataset
Dimensions:     (lat, lon, time)
Coordinates:
    lat         (lat)
    lon         (lon)
    time        (time)
Data variables:
    dat_anomaly     (time, lat, lon)        float64     dask.array
    mask            (lat, lon)              bool        dask.array
    extreme_events  (time, lat, lon)        bool        dask.array
    thresholds      (dayofyear, lat, lon)   float64     dask.array
```
where
* ``dat_anomaly`` (time, lat, lon): Anomaly data
* ``extreme_events`` (time, lat, lon): Binary field locating extreme events (1=event, 0=background)
* ``thresholds`` (dayofyear, lat, lon): Extreme event thresholds used to determine extreme events
* ``mask`` (lat, lon): Valid data mask

See, e.g. ``./examples/unstructured data/01_preprocess_extremes.ipynb`` for a detailed example of pre-processing on an *unstructured* grid.

Step 2: Event Tracking
----------------------

The tracking step identifies coherent extreme events and follows them through time.

**Basic tracking:**

.. code-block:: python

   tracked_events = marEx.tracker(
       extremes.extreme_events,
       extremes.mask,
       area_filter_quartile=0.5,  # Filter small events
       R_fill=8,                  # Radius for filling gaps (in grid cells)
   ).run()

**Advanced tracking options:**

.. code-block:: python

   # Initialise advanced tracker
   tracker = marEx.tracker(
       extremes.extreme_events,
       extremes.mask,

       # Temporal criteria
       T_fill=4,                    # Fill gaps up to 4 days (to maintain continuous events)

       # Spatial criteria
       R_fill=8,                    # Fill small holes with radius up to 8 grid cells
       area_filter_quartile=0.5,    # Size filtering

       # Merging criteria
       allow_merging=True,          # Allow merging of events (and keep track of merged IDs & events)
       overlap_threshold=0.5,       # 50% overlap for merging (otherwise events keep independent IDs)
       nn_partitioning=True,        # Use nearest-neighbour partitioning when splitting events
   )

   # Run tracking and return merging data
   tracked_events, merge_events = tracker.run(return_merges=True)

The resulting xarray dataset ``tracked_events`` will have the following structure & entries:
```
xarray.Dataset
Dimensions: (lat, lon, time, ID, component, sibling_ID)
Coordinates:
    lat         (lat)
    lon         (lon)
    time        (time)
    ID          (ID)
Data variables:
    ID_field              (time, lat, lon)        int32       dask.array
    global_ID             (time, ID)              int32       ndarray
    area                  (time, ID)              float32     ndarray
    centroid              (component, time, ID)   float32     ndarray
    presence              (time, ID)              bool        ndarray
    time_start            (ID)                    datetime64  ndarray
    time_end              (ID)                    datetime64  ndarray
    merge_ledger          (time, ID, sibling_ID)  int32       ndarray

```
where
* ``ID_field``: Field containing the IDs of tracked events (0=background)
* ``global_ID``: Unique global ID of each object; ``global_ID.sel(ID=10)`` maps event ID 10 to its original ID at each time
* ``area``: Area of each event as a function of time
* ``centroid``: (x, y) centroid coordinates of each event as a function of time
* ``presence``: Presence (boolean) of each event at each time (anywhere in space)
* ``time_start``: Start time of each event
* ``time_end``: End time of each event
* ``merge_ledger``: Sibling IDs for merging events (matching ``ID_field``); ``-1`` indicates no merging event occurred

When running with ``return_merges=True``, the resulting xarray dataset ``merge_events`` will have the following structure & entries:
```
xarray.Dataset
Dimensions: (merge_ID, parent_idx, child_idx)
Data variables:
    parent_IDs      (merge_ID, parent_idx)  int32       ndarray
    child_IDs       (merge_ID, child_idx)   int32       ndarray
    overlap_areas   (merge_ID, parent_idx)  int32       ndarray
    merge_time      (merge_ID)              datetime64  ndarray
    n_parents       (merge_ID)              int8        ndarray
    n_children      (merge_ID)              int8        ndarray
```
where
* ``parent_IDs``: Original parent IDs of each merging event
* ``child_IDs``: Original child IDs of each merging event
* ``overlap_areas``: Area of overlap between parent and child objects in each merging event
* ``merge_time``: Time of each merging event
* ``n_parents``: Number of parent objects in each merging event
* ``n_children``: Number of child objects in each merging event

See, e.g. ``./examples/unstructured data/02_id_track_events.ipynb`` for a detailed example of identification, tracking, & merging on an *unstructured* grid.

Step 3: Visualisation
---------------------

marEx includes a powerful visualisation system called ``plotX`` that automatically detects data types and creates appropriate plots.

**Basic visualisation:**

.. code-block:: python

   # Plot global extreme event frequency
   event_frequency = (tracked_events.ID_field > 0).mean("time")

   # Configure plot appearance
   config = marEx.PlotConfig(
       var_units="MHW Frequency",
       cmap="hot_r",
       cperc=[0, 96],
       grid_labels=True
   )

   # Create single plot
   fig, ax, im = event_frequency.plotX.single_plot(config)

**Advanced visualisation:**

.. code-block:: python

   # Multi-panel visualisation: seasonal extreme event frequency
   seasonal_frequency = (tracked_events.ID_field > 0).groupby("time.season").mean(dim="time")

   # Configure plot appearance
   config = marEx.PlotConfig(
       var_units="MHW Frequency",
       cmap="hot_r",
       cperc=[0, 96],
       grid_labels=True
   )

   # Create multi-panel plot
   fig, ax = seasonal_frequency.plotX.multi_plot(config, col="season", col_wrap=2)

   # Create animation of tracked events
   ID_field_subset = tracked_events.ID_field.sel(time=slice("2020-01-01", "2022-05-31"))
   config = marEx.PlotConfig(plot_IDs=True)
   ID_field_subset.plotX.animate(config, plot_dir=plot_dir, file_name="marine_heatwaves")

   # Plot consecutive time periods
   ID_field_subset = tracked_events.ID_field.sel(time=slice("2021-01-01", "2021-01-06"))
   config = marEx.PlotConfig(plot_IDs=True)
   fig, ax = ID_field_subset.plotX.multi_plot(config, col="time", col_wrap=3)

Method Selection Guide
======================

Anomaly Methods
---------------

**Detrended Baseline** (``detrended_baseline``):

* Detrends with an OLS 6+ coefficient model (mean, annual & semi-annual harmonics, and arbitrary polynomial trends)
* **Best for**: Datasets with linear trends, operational monitoring
* **Pros**: Fast & memory efficient
* **Cons**: Does not capture phenological shifts and non-harmonic seasonal variability. Strongly biases certain statistics.
* **Use when**: Real-time processing

**Shifting Baseline** (``shifting_baseline``):

* Removes seasonal & long-term trends using a smoothed rolling climatology
* **Best for**: Climate change studies, non-stationary data
* **Pros**: Captures non-linear trends, adapts to changing climate, and seasonal timing variability
* **Cons**: Computationally expensive, shortens effective time series
* **Use when**: Long-term climate studies, accurate & robust analysis

Extreme Identification Methods
------------------------------

**Global Extreme** (``global_extreme``):

* Applies a global-in-time (i.e. constant in time) threshold
   N.B.: This method is a hack designed for ``ocetrac``, which when paired with ``std_normalise=True`` can approximate ``hobday_extreme`` in very specific cases and with a number of caveats. (However, normalising by local STD is again memory-intensive, at which point there is little gained by this approximate method.)
* **Best for**: Simple threshold, constant across seasons
* **Pros**: Simple interpretation, fast computation (without ``std_normalise``)
* **Cons**: Seasonal bias, may miss winter extremes, variability distribution is skewed
* **Use when**: Initial analysis, 0th order comparisons

**Hobday Method** (``hobday_extreme``):

* Defines a local day-of-year specific threshold within a rolling window (equivalent to the Hobday et al. (2016) definition for simple time-series)
* **Best for**: Long-term scientific climate studies, seasonal studies, biological impacts
* **Pros**: Accounts for seasonal variability, literature standard
* **Cons**: Complex threshold interpretation, computationally intensive
* **Use when**: Ecological studies, seasonal analysis, literature comparison

Threshold Percentiles
---------------------

* **90th percentile**: More events, captures moderate extremes
* **95th percentile**: Standard for marine heatwaves, balanced approach
* **99th percentile**: Only most extreme events, rare events focus

Performance Optimisation
========================

Chunking Strategy
-----------------

Optimal chunking is crucial for performance (it is also an art):

.. code-block:: python

   # General guidelines for chunking
   optimal_chunks = {
       'time': 30,              # ~1 month of daily data
       'lat': -1,               # All latitude points (if memory permits)
       'lon': -1                # All longitude points (if memory permits)
   }

   # For unstructured data
   optimal_chunks_unstruct = {
       'time': 10,
       'ncells': -1             # If memory permits
   }


Parallel Processing
-------------------

.. code-block:: python

   # Start optimised cluster
   client = marEx.helper.start_local_cluster(
       n_workers=4,
       threads_per_worker=2,
       memory_limit='8GB'
   )

   # For HPC systems
   cluster = marEx.helper.start_distributed_cluster(
       cores=128,
       memory='256GB',
       queue='compute'
   )

Grid Types and Coordinate Systems
=================================

Structured/Gridded Data
-----------------------

Regular latitude-longitude grids are the most common data type:

.. code-block:: python

   # Typical structured data
   print(sst.dims)         # e.g. ('time', 'lat', 'lon')
   print(sst.coords)       # e.g. ('time', 'lat', 'lon')

   # marEx automatically detects structured grids
   processed = marEx.preprocess_data(sst)

Unstructured Data
-----------------

Irregular meshes common in coastal and global ocean models:

.. code-block:: python

   # Typical unstructured data (e.g., ICON model)
   print(sst.dims)         # e.g. ('time', 'ncells')
   print(sst.coords)       # e.g. ('time', 'lat', 'lon') as coordinates, not dims

   # Specify grid configuration
   marEx.specify_grid(
       grid_type='unstructured',
       fpath_tgrid='grid_info.nc',      # Grid topology file
       fpath_ckdtree='kdtree.pkl'       # Spatial index (optional)
   )

**Requirements:**

* Spatial dimension name (``ncells``, ``cell``, etc.)
* Latitude/longitude as coordinate arrays
* Optional: Grid topology information for advanced/unstructured features


Advanced Features
=================

Compound Events
---------------

Analyse events that exceed multiple thresholds or variables:

.. code-block:: python

   # Multiple variable analysis
   sst_extremes = marEx.preprocess_data(sst, threshold_percentile=95)
   salinity_extremes = marEx.preprocess_data(salinity, threshold_percentile=5)  # Low salinity

   # Compound events
   compound_events = sst_extremes.extreme_events & salinity_extremes.extreme_events


Best Practices and Guidelines
=============================

Research Workflow
-----------------

1. **Exploratory Analysis**: Start with basic preprocessing to understand data
2. **Method Comparison**: Test different methods on subset of data
3. **Quality Control**: Validate results thoroughly
4. **Full Processing**: Apply chosen method to complete dataset
5. **Validation**: Compare with known events and literature


Literature Compliance
--------------------

For marine heatwave studies following Hobday et al. (2016):

.. code-block:: python

   # Standard MHW definition
   mhw_config = {
       'method_anomaly': 'detrended_baseline',
       'method_extreme': 'hobday_extreme',
       'threshold_percentile': 90,
       'window_days_hobday': 11,
       'window_year_baseline': 30,
       'smooth_days_baseline': 11,
       'window_spatial_hobday': 1,  # Hobday et al. (2016) considers only single points
   }


Getting Help and Support
========================

Documentation Resources
-----------------------

* **Tutorials**: Step-by-step guides in ``docs/tutorials/``
* **API Reference**: Complete function documentation
* **Examples**: Real-world analysis examples
* **Performance Guide**: Optimisation tips and tricks

Community Support
-----------------

* **GitHub Issues**: Bug reports and feature requests
* **Discussions**: Community Q&A and examples
* **Documentation**: Contributions welcome

Citations and References
========================

When using marEx in publications, please cite:

* **marEx package**
* **Marine heatwave methods**: Hobday et al. (2016) for standard MHW definition

For more detailed examples and advanced usage, see the tutorial notebooks in :doc:`tutorials/index`.
