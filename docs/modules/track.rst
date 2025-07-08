============================
Tracking Module (:mod:`marEx.track`)
============================

.. currentmodule:: marEx.track

The :mod:`marEx.track` module provides sophisticated algorithms for identifying and tracking
extreme events through time. It handles complex scenarios including event merging, splitting,
and temporal evolution using advanced image processing and graph theory techniques.

Overview
========

The tracking module implements a comprehensive workflow for converting binary extreme event
masks into tracked event objects with unique identifiers, statistical properties, and
temporal evolution information. It includes advanced algorithms for handling event lifecycles,
including merging and splitting events whilst maintaining unique identities.

**Key Features:**

* **Binary Object Tracking**: Advanced connected component analysis with dask parallelisation
* **Merge/Split Handling**: Algorithms for event lifecycle management
* **Morphological Operations**: Image processing for event preprocessing and hole filling
* **Statistical Analysis**: Comprehensive event property calculation
* **Memory Efficient**: Optimised for large datasets with intelligent chunking
* **Grid Agnostic**: Works with both structured and unstructured grids

Main Classes
============

.. autosummary::
   :toctree: ../generated/

   tracker

Detailed Documentation
======================

Tracker Class
-------------

.. autoclass:: tracker
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage Examples
====================

Simple Event Tracking
----------------------

.. code-block:: python

   import xarray as xr
   import marEx

   # Load preprocessed extreme events data
   extremes_ds = xr.open_dataset('extreme_events.nc')

   # Basic tracking with default parameters
   event_tracker = marEx.tracker(
       extremes_ds.extreme_events,  # Binary extreme events field
       extremes_ds.mask,            # Land-sea mask
       R_fill=8,                    # Fill holes with radius < 8 cells
       area_filter_quartile=0.5     # Remove smallest 50% of events
   )

   # Run tracking algorithm
   tracked_events = event_tracker.run()

Advanced Tracking Configuration
-------------------------------

.. code-block:: python

   # Advanced tracking with merge/split handling
   event_tracker = marEx.tracker(
       extremes_ds.extreme_events,
       extremes_ds.mask,
       R_fill=8,                    # Fill holes with radius < 8 cells
       area_filter_quartile=0.5,    # Remove smallest 50% of events
       T_fill=2,                    # Allow 2-day gaps in tracking
       allow_merging=True,          # Enable merge/split tracking
       overlap_threshold=0.5,       # 50% overlap required for continuity
       nn_partitioning=True         # Use nearest-neighbor partitioning
   )

   # Run tracking and get merge information
   tracked_events, merges_ds = event_tracker.run(return_merges=True)

Unstructured Grid Tracking
---------------------------

.. code-block:: python

   # For unstructured grids, specify dimensions
   dimensions = {'time': 'time', 'x': 'ncells'}
   coordinates = {'time': 'time', 'x': 'lon', 'y': 'lat'}

   event_tracker = marEx.tracker(
       extremes_ds.extreme_events,
       extremes_ds.mask,
       R_fill=5,                    # Adjusted for unstructured grid
       area_filter_quartile=0.4,
       dimensions=dimensions,
       coordinates=coordinates
   )

   tracked_events = event_tracker.run()

Parameter Reference
===================

Required Parameters
-------------------

**data_bin** : xarray.DataArray
  Binary data to identify and track objects in (True = event, False = background).
  Must be Dask-backed and boolean type.

**mask** : xarray.DataArray
  Binary mask indicating valid regions (True = valid, False = invalid).
  Must be boolean type.

**R_fill** : int
  Radius for filling holes/gaps in spatial domain (in grid cells).
  Controls morphological operations to fill small gaps in binary objects.

**area_filter_quartile** : float
  Quantile (0-1) for filtering smallest objects.
  For example, 0.25 removes smallest 25% of objects, 0.5 removes smallest 50%.

Core Tracking Parameters
-------------------------

**T_fill** : int, default=2
  Number of timesteps for filling temporal gaps (must be even).
  Allows events to be tracked across short temporal interruptions.

**allow_merging** : bool, default=True
  Allow objects to split and merge across time.

  * ``True``: Apply splitting & merging criteria, track merge events, and maintain original identities
  * ``False``: Classical connected component labeling with simple time connectivity

**nn_partitioning** : bool, default=True
  Use nearest-neighbor partitioning for merging events.

  * ``True``: Partition merged child objects based on closest parent cell (recommended)
  * ``False``: Use parent centroids for partitioning (may have issues with small objects)

**overlap_threshold** : float, default=0.5
  Minimum fraction of overlap between objects to consider them the same event.
  Fraction of the smaller object's area that must overlap with the larger object's area.

Grid Configuration Parameters
-----------------------------

**dimensions** : dict, default={'time': 'time', 'x': 'lon', 'y': 'lat'}
  Mapping of conceptual dimensions to actual dimension names.
  For unstructured grids, use: ``{'time': 'time', 'x': 'ncells'}``

**coordinates** : dict, default={'time': 'time', 'x': 'lon', 'y': 'lat'}
  Mapping of conceptual coordinates to actual coordinate names.
  For unstructured grids, use: ``{'time': 'time', 'x': 'lon', 'y': 'lat'}``

Output Data Structure
=====================

The tracking algorithm returns an xarray Dataset with the following structure:

Main Tracking Dataset
---------------------

.. code-block:: python

   # tracked_events Dataset structure:
   xarray.Dataset
   Dimensions: (lat, lon, time, ID, component, sibling_ID)
   Coordinates:
       lat         (lat)
       lon         (lon)
       time        (time)
       ID          (ID)
   Data variables:
       ID_field              (time, lat, lon)        int32       # Event ID field
       global_ID             (time, ID)              int32       # Global ID mapping
       area                  (time, ID)              float32     # Event areas
       centroid              (component, time, ID)   float32     # Event centroids
       presence              (time, ID)              bool        # Event presence
       time_start            (ID)                    datetime64  # Start times
       time_end              (ID)                    datetime64  # End times
       merge_ledger          (time, ID, sibling_ID)  int32       # Merge information

**Key Variables:**

* **ID_field**: Binary field with tracked event IDs
* **global_ID**: Unique ID mapping for each event at each time
* **area**: Spatial area of each event through time
* **centroid**: (x,y) centroid coordinates of each event
* **presence**: Boolean indicating event presence at each time
* **time_start/time_end**: Temporal bounds of each event
* **merge_ledger**: Sibling IDs for merging events (-1 = no merge)

Merge Information Dataset
-------------------------

.. code-block:: python

   # merges_ds Dataset structure (when return_merges=True):
   xarray.Dataset
   Dimensions: (merge_ID, parent_idx, child_idx)
   Data variables:
       parent_IDs      (merge_ID, parent_idx)  int32       # Parent event IDs
       child_IDs       (merge_ID, child_idx)   int32       # Child event IDs
       overlap_areas   (merge_ID, parent_idx)  int32       # Overlap areas
       merge_time      (merge_ID)              datetime64  # Merge timestamps
       n_parents       (merge_ID)              int8        # Number of parents
       n_children      (merge_ID)              int8        # Number of children

**Key Variables:**

* **parent_IDs/child_IDs**: Original parent and child IDs in merge events
* **overlap_areas**: Spatial overlap between parent and child objects
* **merge_time**: Timestamp of each merge event
* **n_parents/n_children**: Number of objects involved in each merge

Advanced Usage Examples
========================

Analysing Event Properties
--------------------------

.. code-block:: python

   # Run tracking
   tracked_events, merges_ds = event_tracker.run(return_merges=True)

   # Analyse event durations
   event_durations = (tracked_events.time_end - tracked_events.time_start).dt.days
   long_events = tracked_events.ID.where(event_durations > 10, drop=True)

   # Find events present at specific time
   time_slice = tracked_events.sel(time='2020-07-15')
   active_events = time_slice.ID.where(time_slice.presence, drop=True)

   # Calculate maximum event areas
   max_areas = tracked_events.area.max(dim='time')
   large_events = tracked_events.ID.where(max_areas > threshold, drop=True)

Merge Event Analysis
---------------------

.. code-block:: python

   # Analyse merge events
   if merges_ds is not None:
       # Find complex merge events (multiple parents/children)
       complex_merges = merges_ds.where(
           (merges_ds.n_parents > 1) | (merges_ds.n_children > 1),
           drop=True
       )

       # Analyse merge timing
       merge_times = merges_ds.merge_time.values
       seasonal_merges = merges_ds.groupby(merges_ds.merge_time.dt.season).count()

       # Find parent-child relationships
       for merge_id in complex_merges.merge_ID:
           parents = merges_ds.parent_IDs.sel(merge_ID=merge_id).values
           children = merges_ds.child_IDs.sel(merge_ID=merge_id).values
           print(f"Merge {merge_id}: {parents} → {children}")

Visualisation Integration
-------------------------

.. code-block:: python

   # Plot tracked events
   config = marEx.PlotConfig(
       title='Tracked Marine Heatwave Events',
       plot_IDs=True,          # Special handling for event IDs
       cmap='tab20'            # Discrete colormap
   )

   # Plot single timestep
   fig, ax, im = tracked_events.ID_field.isel(time=0).plotX.single_plot(config)

   # Create animation
   movie_path = tracked_events.ID_field.plotX.animate(
       config,
       plot_dir='./animations',
       file_name='tracked_events'
   )

Performance Optimisation
========================

Chunking Strategy
-----------------

.. code-block:: python

   # Optimal chunking for tracking
   chunk_size = {'time': 25, 'lat': -1, 'lon': -1}  # Keep spatial dims together
   extremes_ds = xr.open_dataset('extremes.nc', chunks=chunk_size)

   # For very large datasets
   event_tracker = marEx.tracker(
       extremes_ds.extreme_events,
       extremes_ds.mask,
       R_fill=8,
       area_filter_quartile=0.5,
       # Performance optimizations
       dask_chunks=chunk_size
   )

Memory Management
-----------------

.. code-block:: python

   # For memory-constrained environments
   event_tracker = marEx.tracker(
       extremes_ds.extreme_events,
       extremes_ds.mask,
       R_fill=6,                    # Smaller fill radius
       area_filter_quartile=0.75,   # Remove more small events
       T_fill=1,                    # Reduce temporal gap filling
       allow_merging=False          # Disable merge tracking for speed
   )

Algorithm Details
=================

Tracking Workflow
-----------------

The tracking algorithm follows these key steps:

1. **Binary Object Identification**
   * Connected component labeling on each time slice
   * Morphological operations (opening/closing) for noise reduction
   * Size filtering based on area quartiles

2. **Temporal Matching**
   * Overlap-based matching between consecutive time steps
   * Handles one-to-one, one-to-many, and many-to-one relationships
   * Gap filling for short-duration interruptions

3. **Event Lifecycle Management**
   * Track initialisation, continuation, merging, and splitting
   * Unique ID assignment and genealogy tracking
   * Statistical property calculation

4. **Post-processing**
   * Merge event documentation
   * Output dataset construction

Merge/Split Algorithm
---------------------

The advanced merge/split tracking implements:

.. code-block:: python

   # Improved merge partitioning logic:
   # - Partition child objects based on nearest parent cell
   # - Maintain original event identities across merges
   # - Track genealogy of splitting and merging events
   # - Use overlap threshold to determine event continuity


Error Handling
==============

Common Issues and Solutions
---------------------------

**Memory Errors**:

.. code-block:: python

   # Solution: Reduce chunk sizes and increase filtering
   event_tracker = marEx.tracker(
       data_bin, mask,
       R_fill=6,                  # Smaller fill radius
       area_filter_quartile=0.8,  # Remove more small events
       T_fill=1,                  # Reduce temporal gap filling
       dask_chunks={'time': 15}   # Smaller time chunks
   )

**Performance Issues**:

.. code-block:: python

   # Solution: Optimise parameters for your data
   event_tracker = marEx.tracker(
       data_bin, mask,
       R_fill=4,                  # Reduce morphological operations
       area_filter_quartile=0.75, # Remove more small events
       allow_merging=False        # Disable merge tracking
   )

**Tracking Quality Issues**:

.. code-block:: python

   # Solution: Tune overlap and temporal parameters
   event_tracker = marEx.tracker(
       data_bin, mask,
       R_fill=8,
       area_filter_quartile=0.5,
       overlap_threshold=0.3,     # Lower overlap requirement
       T_fill=4,                  # Allow longer temporal gaps
       allow_merging=True         # Enable merge tracking
   )

Integration with Preprocessing
==============================

Complete Workflow Example
--------------------------

.. code-block:: python

   import xarray as xr
   import marEx

   # Step 1: Preprocess data
   sst = xr.open_dataset('sst_daily.nc', chunks={'time': 365}).sst

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

   # Step 2: Track events
   event_tracker = marEx.tracker(
       extremes_ds.extreme_events,
       extremes_ds.mask,
       R_fill=8,
       area_filter_quartile=0.5,
       T_fill=2,
       allow_merging=True,
       overlap_threshold=0.5,
       nn_partitioning=True
   )

   # Step 3: Run tracking
   tracked_events, merges_ds = event_tracker.run(return_merges=True)

   # Step 4: Save results
   tracked_events.to_netcdf('tracked_events.nc')
   if merges_ds is not None:
       merges_ds.to_netcdf('merge_events.nc')

See Also
========

* :mod:`marEx.detect` - Data preprocessing module
* :mod:`marEx.plotX` - Visualisation module
* :mod:`marEx.helper` - HPC utilities
