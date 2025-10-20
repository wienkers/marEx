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

For complete installation instructions including system requirements, optional dependencies, and HPC environments, see :doc:`installation`.

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

   # Apply chunking for efficient processing (chunk sizes vary by operation)
   sst = sst.chunk({'time': 365, 'lat': 50, 'lon': 100})

   # Ensure Dask backing
   if not marEx.is_dask_collection(sst.data):
       sst = sst.chunk()

**Note**: Optimal chunk sizes depend on your dataset size and the operation (preprocessing vs tracking). See the Performance Optimisation section below for detailed chunking strategies.

Core Workflow
=============

The marEx workflow consists of three main steps. For a conceptual overview and foundations, see :doc:`concepts`.

.. code-block:: text

   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
   │  1. Detect      │  →   │  2. Track       │  →   │  3. Visualise   │
   │    Extremes     │      │    Events       │      │     & Analyse   │
   └─────────────────┘      └─────────────────┘      └─────────────────┘
           ↓                        ↓                        ↓
    preprocess_data()           tracker()                  plotX()
           ↓                        ↓                        ↓
    Binary extreme map        Tracked objects          Maps, animations,
                              with unique IDs           & statistics

**Summary:**

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
       method_anomaly='shifting_baseline',
       method_extreme='hobday_extreme'
   )

**Advanced configuration:**

.. code-block:: python

   # Advanced preprocessing configuration
   extremes = marEx.preprocess_data(
       sst,
       # Anomaly computation method
       method_anomaly='detrend_fixed_baseline',  # or 'detrend_harmonic', 'fixed_baseline', 'shifting_baseline'
       detrend_order=[1, 2],
       smooth_days_baseline=21,             # Smoothing for climatology

       # Extreme identification method
       method_extreme='hobday_extreme',     # or 'global_extreme'
       threshold_percentile=95,             # 95th percentile
       window_days_hobday=11,               # Days around each day of year
       window_spatial_hobday=5,             # Spatial window for percentile calculation

       # Output options
       dask_chunks={'time': 25}
   )

The resulting xarray dataset ``extremes`` will have the following structure & entries::

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
where:

* ``dat_anomaly`` (time, lat, lon): Anomaly data
* ``extreme_events`` (time, lat, lon): Binary field locating extreme events (1=event, 0=background)
* ``thresholds`` (dayofyear, lat, lon): Extreme event thresholds used to determine extreme events
* ``mask`` (lat, lon): Valid data mask

See ``./examples/unstructured data/01_preprocess_extremes.ipynb`` in :doc:`examples` for a detailed example of pre-processing on an *unstructured* grid.

Step 2: Event Tracking
----------------------

The tracking step identifies coherent extreme events and follows them through time.

**Basic tracking:**

.. code-block:: python

   tracked_events = marEx.tracker(
       extremes.extreme_events,
       extremes.mask,
       area_filter_absolute=100   # Remove objects smaller than 100 grid cells
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
       area_filter_quartile=0.5,    # Remove smallest 50% of events (alternative to area_filter_absolute)
       cell_areas=grid_areas,       # Optional: physical cell areas for structured grids (m²)

       # Merging criteria
       allow_merging=True,          # Allow merging of events (and keep track of merged IDs & events)
       overlap_threshold=0.5,       # 50% overlap for merging (otherwise events keep independent IDs)
       nn_partitioning=True,        # Use nearest-neighbour partitioning when splitting events
   )

   # Run tracking and return merging data
   tracked_events, merge_events = tracker.run(return_merges=True)

The resulting xarray dataset ``tracked_events`` will have the following structure & entries::

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
where:

* ``ID_field``: Field containing the IDs of tracked events (0=background)
* ``global_ID``: Unique global ID of each object; ``global_ID.sel(ID=10)`` maps event ID 10 to its original ID at each time
* ``area``: Area of each event as a function of time (in units of cell counts, or physical units if cell_areas/grid_resolution provided)
* ``centroid``: (x, y) centroid coordinates of each event as a function of time
* ``presence``: Presence (boolean) of each event at each time (anywhere in space)
* ``time_start``: Start time of each event
* ``time_end``: End time of each event
* ``merge_ledger``: Sibling IDs for merging events (matching ``ID_field``); ``-1`` indicates no merging event occurred

When running with ``return_merges=True``, the resulting xarray dataset ``merge_events`` will have the following structure & entries::

   xarray.Dataset
   Dimensions: (merge_ID, parent_idx, child_idx)
   Data variables:
       parent_IDs      (merge_ID, parent_idx)  int32       ndarray
       child_IDs       (merge_ID, child_idx)   int32       ndarray
       overlap_areas   (merge_ID, parent_idx)  int32       ndarray
       merge_time      (merge_ID)              datetime64  ndarray
       n_parents       (merge_ID)              int8        ndarray
       n_children      (merge_ID)              int8        ndarray
where:

* ``parent_IDs``: Original parent IDs of each merging event
* ``child_IDs``: Original child IDs of each merging event
* ``overlap_areas``: Area of overlap between parent and child objects in each merging event
* ``merge_time``: Time of each merging event
* ``n_parents``: Number of parent objects in each merging event
* ``n_children``: Number of child objects in each merging event

See ``./examples/unstructured data/02_id_track_events.ipynb`` in :doc:`examples` for a detailed example of identification, tracking, & merging on an *unstructured* grid.

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

This section provides practical guidance for selecting appropriate methods. For foundations and definitions, see :doc:`concepts`.

Choosing the Right Anomaly Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this decision tree to select the most appropriate anomaly calculation method for your research:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────┐
   │ Anomaly Method Selection Decision Tree                              │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                     │
   │ Need full time series? ──No──> SHIFTING BASELINE                    │
   │         │                       (most accurate, shortens data by    │
   │        Yes                       window_year_baseline years)        │
   │         │                                                           │
   │         ├─> Remove trends? ──No──> FIXED BASELINE                   │
   │         │         │                 (keeps trends in anomaly)       │
   │         │        Yes                                                │
   │         │         │                                                 │
   │         │         └──> Need efficiency? ──Yes──> DETREND HARMONIC   │
   │         │                      │                  (fast, biased)    │
   │         │                     No                                    │
   │         │                      │                                    │
   │         │                      └──> DETREND FIXED BASELINE          │
   │         │                           (accurate detrending)           │
   └─────────────────────────────────────────────────────────────────────┘


Anomaly Methods
---------------

**Harmonic Detrending** (``detrend_harmonic``):

* Detrends with an OLS 6+ coefficient model (mean, annual & semi-annual harmonics, and arbitrary polynomial trends)
* **Best for**: Datasets with linear trends, operational monitoring
* **Pros**: Fast & memory efficient
* **Cons**: Does not capture phenological shifts and non-harmonic seasonal variability. Strongly biases certain statistics.
* **Use when**: Real-time processing

**Fixed Baseline** (``fixed_baseline``):

* Daily climatology using full time series -- does not remove climate trends
* **Best for**: Simple anomaly calculation without detrending
* **Pros**: Straightforward interpretation, preserves long-term trends
* **Cons**: Does not account for climate change trends, seasonal timing shifts
* **Use when**: Baseline comparison studies, trend-inclusive analysis, public outreach

**Detrend Fixed Baseline** (``detrend_fixed_baseline``):

* Polynomial detrending followed by fixed daily climatology -- keeps full time-series of data, but does not account for trends in the timing of seasonal transitions
* **Best for**: Studies requiring detrending but maintaining full temporal data coverage
* **Pros**: Removes long-term trends while preserving seasonal cycles, maintains full time series
* **Cons**: Does not account for changes in seasonal timing or seasonality
* **Use when**: Climate variability studies with trend removal

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

Spatial Window for Hobday Extreme (``window_spatial_hobday``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**New in v3.0+**: The Hobday extreme method now supports optional spatial windowing to create more statistically robust thresholds.

.. code-block:: python

   # Traditional Hobday (temporal window only)
   extremes_traditional = marEx.preprocess_data(
       sst,
       method_extreme='hobday_extreme',
       window_days_hobday=11,          # 11-day temporal window
       window_spatial_hobday=None      # No spatial window
   )

   # With spatial windowing (recommended for most applications)
   extremes_window = marEx.preprocess_data(
       sst,
       method_extreme='hobday_extreme',
       window_days_hobday=11,          # 11-day temporal window
       window_spatial_hobday=5         # 5×5 spatial window (marEx default; use 1 for strict Hobday et al. 2016 definition for time-series)
   )

**How Spatial Windowing Works**::

   Traditional Hobday:                  With Spatial Window (5×5):
   Single cell samples                  25 cells × 11 days = 275 samples
   (lat, lon) ──> 11 days               ┌───┬───┬───┬───┬───┐
                                        │   │   │   │   │   │
   Only temporal pooling                ├───┼───┼───┼───┼───┤
                                        │   │   │   │   │   │
                                        ├───┼───┼─●─┼───┼───┤ Central cell
                                        │   │   │   │   │   │
                                        ├───┼───┼───┼───┼───┤
                                        │   │   │   │   │   │
                                        └───┴───┴───┴───┴───┘
                                        Spatial + temporal pooling

**Benefits**:

* **Spatially coherent thresholds**: Reduces noise from individual grid cells
* **More robust statistics**: Larger sample size for robust percentile calculation

**Limitations**:

* **Structured grids only**: Not supported for unstructured (irregular) grids
* **Requires approximate method**: Only works with ``method_percentile='approximate'``

**When to use**:

* Short time-series of data
* High percentile thresholds defining the extremes
* Noisy SST fields (e.g., satellite with gaps)
* Default (``window_spatial_hobday=5``) works well for most cases

**Example: Comparing threshold noise**:

.. code-block:: python

   # Compare threshold variability
   threshold_std_traditional = extremes_traditional.thresholds.std(dim='dayofyear').mean()
   threshold_std_smooth = extremes_smooth.thresholds.std(dim='dayofyear').mean()

   print(f"Traditional threshold noise: {threshold_std_traditional.compute():.4f}")
   print(f"Windowed threshold noise: {threshold_std_window.compute():.4f}")

Threshold Percentiles
---------------------

* **90th percentile**: More events, captures moderate extremes
* **95th percentile**: Standard for marine heatwaves, balanced approach
* **99th percentile**: Only most extreme events, rare events focus

Performance Optimisation
========================

This section provides guidance for optimising marEx performance across different computing environments and dataset sizes.

Memory Management and Chunking
-------------------------------

Chunking Strategy
~~~~~~~~~~~~~~~~~

Optimal chunking strategies vary by dataset size and operation type:

.. code-block:: python

   # Small datasets (< 10GB): Chunk time dimension
   sst = sst.chunk({'time': 365, 'lat': -1, 'lon': -1})

   # Large datasets (> 100GB): Chunk all dimensions
   sst = sst.chunk({'time': 365, 'lat': 180, 'lon': 360})

   # For tracking: Use smaller time chunks
   sst = sst.chunk({'time': 25, 'lat': -1, 'lon': -1})  # Recommended for tracker

Memory Requirements Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Operation
     - Memory (per core)
     - Optimal Time Chunk
     - Notes
   * - ``preprocess_data``
     - 2-4 GB
     - 365+ days
     - Depends on method_anomaly
   * - ``tracker``
     - 1-2 GB
     - 25-50 days
     - Keep spatial dims unbounded (-1)
   * - Exact percentiles
     - 8-16 GB
     - Full time series
     - Requires careful chunking
   * - Approximate percentiles
     - 1-2 GB
     - Any
     - Memory efficient

Exact vs Approximate Percentile Methods
----------------------------------------

When to Use Each Method
~~~~~~~~~~~~~~~~~~~~~~~

**APPROXIMATE (default)**: For large datasets

.. code-block:: python

   extremes = marEx.preprocess_data(
       sst,
       method_percentile='approximate',
       precision=0.01,        # ~0.01°C bins
       max_anomaly=5.0,       # Histogram range ±5°C
       dask_chunks={'time': 365}  # Any chunking works
   )
   # ✓ Memory efficient (1-2 GB/core)
   # ✓ Fast parallel computation
   # ✓ ~0.01°C precision (sufficient for most studies)

**EXACT**: For small datasets or high precision needs

.. code-block:: python

   extremes_exact = marEx.preprocess_data(
       sst,
       method_percentile='exact',
       dask_chunks={'time': -1, 'lat': 100, 'lon': 100}  # Careful chunking required!
   )
   # ⚠ High memory (8-16 GB/core), depending on time-series length
   # ⚠ Slower computation
   # ✓ Mathematically exact percentiles


JAX Acceleration
----------------

When JAX Helps Most
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import marEx

   # Check JAX availability
   if marEx.has_dependency('jax'):
       print("JAX acceleration active")
       # Best for:
       # - Harmonic detrending (10-50× speedup)
       # - Large spatial operations
       # - GPU/TPU available systems
   else:
       print("Install JAX: pip install marEx[full]")
       # Falls back to NumPy (still fast with Numba JIT)

Performance Gains
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Operation
     - NumPy+Numba
     - JAX (CPU)
     - JAX (GPU)
   * - Harmonic detrend
     - 1× (baseline)
     - 15× faster
     - 50× faster
   * - Percentile calc
     - 1×
     - 1× (similar)
     - 2-3× faster
   * - Tracking
     - 1×
     - 1× (similar)
     - N/A

HPC Cluster Setup
-----------------

SLURM Distributed Cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~

For supercomputers:

.. code-block:: python

   import marEx

   # Start distributed Dask cluster on SLURM
   client = marEx.helper.start_distributed_cluster(
       n_workers=64,           # Number of workers
       cores_per_worker=4,     # Cores per worker
       memory_per_worker='8GB' # Memory per worker
   )

   # Get dashboard URL for monitoring
   marEx.helper.get_cluster_info(client)

   # Run analysis
   extremes = marEx.preprocess_data(sst, ...)
   events = marEx.tracker(extremes.extreme_events, ...).run()

   # Clean up
   client.close()

Local Cluster
~~~~~~~~~~~~~

For workstations:

.. code-block:: python

   # Auto-detect system resources
   client = marEx.helper.start_local_cluster(
       n_workers=8,            # Leave some cores for system
       threads_per_worker=2,
       memory_limit='16GB'     # Per worker
   )

Example Batch Scripts
~~~~~~~~~~~~~~~~~~~~~

The ``examples/batch jobs/`` directory provides ready-to-use scripts designed to be run from a login node on HPC systems with SLURM:

**Workflow:**

1. **run_detect.py**: Launches a distributed Dask job to execute the preprocessing and extreme detection pipeline
2. **run_track.py**: Launches a distributed Dask job for event identification and tracking

Both scripts utilise ``marEx.helper.start_distributed_cluster()`` for cluster management and are configured via environment variables.

**Configuration Environment Variables:**

* ``DASK_N_WORKERS``: Number of Dask workers to launch (default varies by script: 128 for detect, 32 for track)
* ``DASK_WORKERS_PER_NODE``: Workers per SLURM node (default varies by script: 64 for detect, 32 for track)
* ``DASK_RUNTIME``: Maximum runtime in minutes (default: 39 for detect, 89 for track)
* ``SLURM_ACCOUNT``: SLURM account for billing (default: 'bk1377')
* **Additional for run_track.py**: ``RUN_BASIC_TRACKER``, ``GRID_RESOLUTION``, ``AREA_FILTER``, ``R_FILL``, ``T_FILL``, ``OVERLAP_THRESHOLD``

**Example Usage:**

.. code-block:: bash

   # Run preprocessing on SLURM cluster
   export DASK_N_WORKERS=256
   export DASK_RUNTIME=120
   export SLURM_ACCOUNT=my_project
   python examples/batch\ jobs/run_detect.py

   # Run tracking after preprocessing completes
   export DASK_N_WORKERS=64
   export DASK_RUNTIME=180
   python examples/batch\ jobs/run_track.py

**Customisation:**

These scripts are designed to be copied and modified for your specific:

* Data file paths and chunk sizes
* Preprocessing parameters (anomaly method, thresholds)
* Tracking parameters (spatial filters, merge/split settings)
* Cluster resources (workers, memory, runtime)

The scripts handle cluster setup, data processing, and saving results to zarr/netCDF files on the scratch filesystem.

Checkpointing for Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When processing very large datasets on HPC systems, Dask may recompute intermediate results multiple times, especially for complex operations like 2D histogram-based percentile calculations. This can significantly increase computation time and memory usage.

**The Problem:**

Dask builds a computation graph that tracks all operations. For large preprocessing pipelines, this graph can become deeply nested, causing Dask to recompute expensive intermediate steps (climatologies, anomalies, thresholds) whenever downstream operations need them.

**The Solution:**

The ``use_temp_checkpoints=True`` parameter breaks the Dask computation graph by saving intermediate results to temporary zarr stores and immediately reloading them. This prevents expensive recomputations at the cost of some disk I/O.

**How It Works:**

1. Intermediate arrays (anomalies, climatologies, thresholds, extremes) are saved to temporary zarr files
2. The saved data is immediately reloaded as a fresh Dask array
3. This breaks the dependency chain in the computation graph
4. Temporary files are automatically cleaned up after reloading

**When to Use:**

* Large datasets (>100 GB) where preprocessing takes hours
* HPC environments with fast scratch storage
* When 2D histogram percentile calculations are a bottleneck
* When you notice Dask recomputing the same operations multiple times

**Example Usage:**

.. code-block:: python

   import xarray as xr
   import marEx

   # Load large dataset
   sst = xr.open_zarr('large_dataset.zarr', chunks={'time': 30}).sst

   # Enable checkpointing to prevent expensive recomputations
   extremes = marEx.preprocess_data(
       sst,
       method_anomaly='shifting_baseline',
       method_extreme='hobday_extreme',
       threshold_percentile=95,
       use_temp_checkpoints=True,  # Enable checkpointing
       dask_chunks={'time': 25}
   )

**Performance Trade-offs:**

* **Benefit**: Prevents expensive recomputations, reduces memory pressure
* **Cost**: Requires fast disk I/O for temporary zarr stores
* **Recommendation**: Enable on HPC systems with high-speed scratch storage; test on your specific dataset to measure performance impact

**Note:** Temporary files are automatically removed after data is reloaded, so no manual cleanup is required.

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
       'method_anomaly': 'shifting_baseline',
       'method_extreme': 'hobday_extreme',
       'threshold_percentile': 90,
       'window_days_hobday': 11,
       'window_year_baseline': 30,
       'smooth_days_baseline': 11,
       'window_spatial_hobday': 1,  # Hobday et al. (2016) considers only single points
   }


Getting Help and Support
========================

For troubleshooting common issues, diagnostic checklists, and support resources, see :doc:`troubleshooting`.

Citations and References
========================

When using marEx in publications, please cite:

* **marEx package**
* **Marine heatwave methods**: Hobday et al. (2016) for standard MHW definition

For more detailed examples and advanced usage, see the :doc:`examples`.
