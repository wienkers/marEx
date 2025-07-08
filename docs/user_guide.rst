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

The preprocessing step transforms raw oceanographic data into anomalies and identifies extreme events.

**Basic preprocessing:**

.. code-block:: python

   import marEx

   # Basic preprocessing with default settings
   processed = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       method_anomaly='detrended_baseline',
       method_extreme='global_extreme'
   )

**Advanced configuration:**

.. code-block:: python

   # Advanced preprocessing configuration
   processed = marEx.preprocess_data(
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

**Output variables:**

* ``dat_anomaly``: Anomaly data
* ``extreme_events``: Binary extreme events
* ``thresholds``: Extreme event threshold
* ``mask``: Valid data mask

Step 2: Event Tracking
----------------------

The tracking step identifies coherent extreme events and follows them through time.

**Basic tracking:**

.. code-block:: python

   # Initialise tracker
   tracker = marEx.tracker(
       processed.extreme_events,
       processed.mask,
       area_filter_quartile=0.5,  # Filter small events
       R_fill=8,                  # Radius for filling gaps (in grid cells)
   )

   # Run tracking
   tracked_events = tracker.run()

**Advanced tracking options:**

.. code-block:: python

   # Advanced tracking configuration
   tracker = marEx.tracker(
       processed.extreme_events,
       processed.mask,

       # Temporal criteria
       T_fill=4,                    # Fill gaps up to 4 days (to keep continuous events)

       # Spatial criteria
       R_fill=8,                    # Fill small holes with radius up to 8 grid cells
       area_filter_quartile=0.5,    # Size filtering

       # Merging criteria
       allow_merging=True,          # Allow merging of events (and keep track of merged IDs & events)
       overlap_threshold=0.5,       # 50% overlap for merging
       nn_partitioning=True,        # Use nearest neighbour partitioning when splitting events
   )

**Output variables:**

* ``event_labels``: Event IDs for each grid point and time
* ``event_stats``: Statistical properties of each event
* ``event_tracks``: Temporal evolution of events
* ``ID_field``: Binary field indicating tracked events (1=event, 0=background)
* ``global_ID``: Unique global ID for each event; ``global_ID.sel(ID=10)`` maps event ID 10 to its original ID at each time
* ``area``: Area of each event as a function of time
* ``centroid``: (x, y) centroid coordinates of each event as a function of time
* ``presence``: Presence (True/False) of each event at each time (anywhere in space)
* ``time_start``: Start time of each event
* ``time_end``: End time of each event
* ``merge_ledger``: Sibling IDs for merging events (matching ``ID_field``); ``-1`` indicates no merging event occurred

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

* **Best for**: Datasets with linear trends, operational monitoring
* **Pros**: Fast & memory efficient
* **Cons**: Does not capture phenological shifts and non-harmonic seasonal variability
* **Use when**: Real-time processing

**Shifting Baseline** (``shifting_baseline``):

* **Best for**: Climate change studies, non-stationary data
* **Pros**: Captures non-linear trends, adapts to changing climate, and seasonal timing variability
* **Cons**: Computationally expensive, shortens effective time series
* **Use when**: Long-term climate studies, accurate & robust analysis

Extreme Identification Methods
------------------------------

**Global Extreme** (``global_extreme``):

* **Best for**: Simple threshold, consistent across seasons
* **Pros**: Simple interpretation, fast computation
* **Cons**: Seasonal bias, may miss winter extremes, variability distribution is skewed
* **Use when**: Initial analysis, simple comparisons

**Hobday Method** (``hobday_extreme``):

* **Best for**: Long-term climate studies, seasonal studies, biological impacts
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

Custom Event Criteria
---------------------

Define specialised event detection criteria:

.. code-block:: python

   # Custom intensity-duration criteria
   def custom_event_filter(binary_events, min_intensity=2.0, min_duration=7):
       """Apply custom filtering to extreme events"""

       # Implementation --
       return filtered_events


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
