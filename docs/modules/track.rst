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
       area_filter_absolute=100     # Remove objects smaller than 100 grid cells
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
       area_filter_quartile=0.5,    # Remove smallest 50% of events (alternative to area_filter_absolute)
       T_fill=2,                    # Allow 2-day gaps in tracking
       allow_merging=True,          # Enable merge/split tracking
       overlap_threshold=0.5,       # 50% overlap required for continuity
       nn_partitioning=True,        # Use nearest-neighbor partitioning
       cell_areas=grid_areas        # Optional: physical cell areas (m²)
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

  **How R_fill Works**:

  The `R_fill` parameter controls a two-step morphological cleaning operation: **closing** followed by **opening**.
  This process makes identified events spatially coherent and removes small, isolated noise.

  **Physical Rationale**:

  1. **Closing (Fill Holes)**: Marine heatwaves are generally coherent spatial phenomena. Small "holes"
     within a larger event are often artefacts of data gridding or minor fluctuations below the threshold.
     Closing (dilation → erosion) fills these internal gaps, making the event representation physically realistic.

  2. **Opening (Remove Noise)**: Very small, isolated pixels flagged as events are often statistical noise
     or sensor errors rather than genuine marine heatwaves. Opening (erosion → dilation) removes these
     spurious objects, ensuring only spatially significant events are tracked.

  **Visual Process** (using ``R_fill=1`` as example)::

        Initial State                  After Closing                  After Opening
    ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
    │   █████              │       │   █████              │       │   █████              │
    │  ███████             │       │  ███████             │       │  ███████             │
    │ ███ █████            │       │ █████████            │       │ █████████            │
    │ ███  ████   █        │       │ █████████   █        │       │ █████████            │
    │  ████████            │       │  ████████            │       │  ████████            │
    │   ███████            │       │   ███████            │       │   ███████            │
    │    █████             │       │    █████             │       │    █████             │
    │           █          │       │           █          │       │                      │
    └──────────────────────┘       └──────────────────────┘       └──────────────────────┘
    │                              │                              │
    └─ Main event with             └─ Closing fills internal      └─ Opening removes
       2 small holes inside           holes, making event            isolated noise pixels,
       + 2 isolated noise pixels      more coherent                  leaving clean event

  **Structuring Element (Kernel)**:

  Operations use a disk-shaped kernel with diameter = ``2 * R_fill + 1``.
  For ``R_fill=2``, the kernel diameter is 5 pixels::

       ┌──────────────┐
       │      ██      │     Any hole or isolated object smaller
       │   ██ ██ ██   │     than this disk will be filled or removed
       │██ ██ ██ ██ ██│
       │   ██ ██ ██   │
       │      ██      │
       └──────────────┘

  **Example**: ``R_fill=8`` uses a (circular) disk kernel with diameter 17 pixels, filling holes and
  removing objects up to ~8 grid cells in radius.

  **When to Adjust**:

  * **Smaller R_fill** (e.g., 3-5): Preserves small-scale features; use for coarse data or fine structures
  * **Larger R_fill** (e.g., 10-15): Creates more coherent objects; use for noisy data or large-scale events

**area_filter_quartile** : float
  Quantile (0-1) for filtering smallest objects.
  For example, 0.25 removes smallest 25% of objects, 0.5 removes smallest 50%.
  Mutually exclusive with area_filter_absolute.

**area_filter_absolute** : int
  Minimum area (in grid cells) for an object to be retained.
  For example, 25 keeps only events with 25 or more grid cells.
  Mutually exclusive with area_filter_quartile.

Core Tracking Parameters
-------------------------

**T_fill** : int, default=2
  Number of timesteps for filling temporal gaps (must be even).
  Allows events to be tracked across short temporal interruptions.

  **How T_fill Works**::

     T_fill = Maximum temporal gap (in days) for event continuity

     Timeline with T_fill=2:

     Day 0: ████████  ← Event detected
     Day 1:            ← Gap (no detection)
     Day 2:            ← Gap (no detection)  } T_fill=2 days
     Day 3: ████████  ← Event detected again
            │        │
            └────────┴─ SAME event ID (gap ≤ T_fill)

     Timeline with T_fill=0 (no gap filling):

     Day 0: ████████  ← Event ID: 1
     Day 1:            ← Gap
     Day 2: ████████  ← NEW event ID: 2 (no gap tolerance)

  **Example**: ``T_fill=2`` allows 2-day gaps while maintaining the same event ID.

  **When to Adjust**:

  * **T_fill=0**: No gap filling, strict continuity requirement, or for weekly/monthly sampling
  * **T_fill=2-4**: Standard gap filling for daily data (recommended)
  * **T_fill=6-10**: More permissive for noisy data

**allow_merging** : bool, default=True
  Allow objects to split and merge across time.

  * ``True``: Apply splitting & merging criteria, track merge events, and maintain original identities
  * ``False``: Classical connected component labeling with simple time connectivity

**nn_partitioning** : bool, default=True
  Use nearest-neighbor cell-based partitioning for merging events (improved algorithm).

  * ``True``: Partition merged child objects based on closest parent **cell** (recommended)
  * ``False``: Use parent **centroids** for partitioning (legacy method with known issues)

  **Why This Matters**: When objects merge, their new area must be partitioned and assigned back to
  the original parent identities. The choice of algorithm for this partitioning is critical.

  The ``centroid`` method partitions based on proximity to parent centroids. This can be problematic
  for non-convex or L-shaped features, where the centroid may lie far from the feature's cells,
  leading to physically unrealistic partitions.

  In contrast, the ``nn`` (nearest-neighbor) method assigns each cell to the parent of the *closest*
  cell from the previous timestep. This results in a more intuitive and physically-based partition
  that respects the geometry of the parent features.

  **Visual Comparison**:

  **Scenario:** A large C-shaped feature (A) and a small nearby feature (B)
  merge into a single large object. We compare how centroid-based vs nearest-neighbor methods
  partition the merged child back to parent lineages.

  **1. Initial State (Time t)**

  Two distinct parent features with centroids marked by ●.

  ::

       ┌───┬───┬───┬───┬───┐
     5 │ A │ A │ A │ A │ A │
       ├───┼───┼───┼───┼───┤
     4 │ A │ A │   │   │   │
       ├───┼───┼───┼───┼───┤
     3 │ A │ ● │   │ B ● B │  ← Centroid A at (3, 2), Centroid B at (3, 4.5)
       ├───┼───┼───┼───┼───┤
     2 │ A │ A │   │   │   │
       ├───┼───┼───┼───┼───┤
     1 │ A │ A │ A │ A │ A │
       └───┴───┴───┴───┴───┘
         1   2   3   4   5

  **2. Merged State (Time t+1)**

  The features grow and merge into a single large object marked with '#'. 
  Question: how should these 20 cells be attributed to lineages A and B ?

  ::

       ┌───┬───┬───┬───┬───┐
     5 │   │ # │ # │ # │ # │
       ├───┼───┼───┼───┼───┤
     4 │ # │ # │ # │   │   │
       ├───┼───┼───┼───┼───┤
     3 │ # │ # │ # │ # │ # │
       ├───┼───┼───┼───┼───┤
     2 │ # │ # │ # │   │   │
       ├───┼───┼───┼───┼───┤
     1 │   │ # │ # │ # │ # │
       └───┴───┴───┴───┴───┘
         1   2   3   4   5

  **3. Centroid Partition (nn_partitioning=False) - ❌ PROBLEMATIC**

  Each cell assigned to nearest parent *centroid*. The partition line ║ creates a geometric
  boundary that ignores actual cell topology. Result: B is an odd disjoint object now !

  ::

       ┌───┬───┬──╦┬───┬───┐
     5 │   │ A │ A║│ B │ B │
       ├───┼───┼──╫┼───┼───┤
     4 │ A │ A │ A║│   │   │
       ├───┼───┼──╫┼───┼───┤
     3 │ A │ A │ A║│ B │ B │
       ├───┼───┼──╫┼───┼───┤
     2 │ A │ A │ A║│   │   │
       ├───┼───┼──╫┼───┼───┤
     1 │   │ A │ A║│ B │ B │
       └───┴───┴──╩┴───┴───┘
         1   2   3   4   5

  **Problem:** B is an odd disjoint object with portions unrelated to the original parent.
  The original C-shaped object means that the centroid misrepresents A's main body, thus
  creating unrealistic/nonrepresentative partitions.

  **4. Nearest-Neighbour Partition (nn_partitioning=True)

  Each cell assigned to parent of nearest parent *cell*. Natural boundaries respect actual
  spatial connectivity. Result: B remains contiguous (realistic growth pattern).

  ::

       ┌───┬───┬───┬───┬───┐
     5 │   │ A │ A │ A │ A │
       ├───┼───┼───┼───┼───┤
     4 │ A │ A │ A │   │   │
       ├───┼───┼───┼───┼───┤
     3 │ A │ A │ B │ B │ B │
       ├───┼───┼───┼───┼───┤
     2 │ A │ A │ A │   │   │
       ├───┼───┼───┼───┼───┤
     1 │   │ A │ A │ A │ A │
       └───┴───┴───┴───┴───┘
         1   2   3   4   5

  **Solution:** Growth of the resulting merged object is correctly attributed to A's
  lineage based on cell-level proximity, not abstract geometric centroids.

  **Example: Small Object Problem**:

  .. code-block:: python

     # Centroid method (old, has issues)
     tracker_centroid = marEx.tracker(
         extremes, mask,
         R_fill=8,
         allow_merging=True,
         nn_partitioning=False  # Small objects get unrealistic portions
     )

     # Nearest-neighbor method (new, recommended)
     tracker_nn = marEx.tracker(
         extremes, mask,
         R_fill=8,
         allow_merging=True,
         nn_partitioning=True   # Realistic partitioning
     )

  **When to Use**:

  * **Always use ``nn_partitioning=True``** (default) for accurate merging/splitting
  * Only use ``nn_partitioning=False`` for comparison with legacy results or Sun et al. (2023) methodology


**overlap_threshold** : float, default=0.5
  Minimum fraction of overlap between objects to consider them the same event.
  Fraction of the smaller object's area that must overlap with the larger object's area.

  **How overlap_threshold Works - Two-Stage Process**::

     ═══════════════════════════════════════════════════════════════════════════════
     STAGE 1: OVERLAP CHECKING (Determines IF objects are related)
     ═══════════════════════════════════════════════════════════════════════════════

     Formula: overlap_fraction = overlap_area / min(area_parent, area_child)
     Decision: IF overlap_fraction >= overlap_threshold → Objects are LINKED

     This stage applies to ALL transitions (1→1, 1→many, many→1, many→many)
     This stage is INDEPENDENT of nn_partitioning choice

     ───────────────────────────────────────────────────────────────────────────────
     Example 1: Threshold MET (overlap_threshold = 0.5)
     ───────────────────────────────────────────────────────────────────────────────

     Time t=0:          Time t=1:          Overlap Check:

     ┌──────────┐       ┌─────────┐        overlap_area = 60 cells
     │  Object  │       │ Object  │        min(100, 80) = 80
     │    A     │   →   │    C    │
     │   100    │       │   80    │        overlap_fraction = 60/80 = 0.75
     │  cells   │       │  cells  │
     └──────────┘       └─────────┘        0.75 >= 0.5 ✓ → LINKED (same event)

     ───────────────────────────────────────────────────────────────────────────────
     Example 2: Threshold NOT MET (overlap_threshold = 0.5)
     ───────────────────────────────────────────────────────────────────────────────

     Time t=0:          Time t=1:          Overlap Check:

     ┌──────────┐       ┌─────────┐        overlap_area = 30 cells
     │  Object  │       │ Object  │        min(100, 80) = 80
     │    A     │   ╳   │    C    │
     │   100    │       │   80    │        overlap_fraction = 30/80 = 0.375
     │  cells   │       │  cells  │
     └──────────┘       └─────────┘        0.375 < 0.5 ✗ → NOT LINKED

                                           Result: A terminates, C starts as NEW event

     ═══════════════════════════════════════════════════════════════════════════════
     STAGE 2: PARTITIONING (Determines HOW to divide merged areas)
     ═══════════════════════════════════════════════════════════════════════════════

     This stage ONLY applies when a MERGE is detected (many parents → one child),
     which also requires the overlap criterion (Stage 1) to be met for each parent.
     This stage does NOT apply to splits (one parent → many children)
     Choice of nn_partitioning (True/False) affects ONLY this stage

     ───────────────────────────────────────────────────────────────────────────────
     Example 3: MERGE Scenario - Partitioning Applies
     ───────────────────────────────────────────────────────────────────────────────

     Time t=0:                Time t=1:

     ┌─────────┐              ┌──────────────┐
     │ Object A│              │   Object C   │
     │   100   │──┐           │     120      │
     │  cells  │  │           │    cells     │
     └─────────┘  │           └──────────────┘
                  │  MERGE
     ┌─────────┐  │
     │ Object B│──┘           STAGE 1 - Overlap Checking:
     │   40    │              ────────────────────────────
     │  cells  │              A→C: 80/min(100,120) = 80/100 = 0.8 >= 0.5 ✓ LINKED
     └─────────┘              B→C: 35/min(40,120)  = 35/40  = 0.875 >= 0.5 ✓ LINKED

                              Both A and B linked to C → MERGE DETECTED!

                              STAGE 2 - Partitioning (triggered by merge):
                              ─────────────────────────────────────────────
                              Question: How to divide C's 120 cells between A and B lineages?

                              if nn_partitioning=True:  Use nearest parent CELL
                              if nn_partitioning=False: Use nearest parent CENTROID

                              (See nn_partitioning documentation for partition methods)

     ───────────────────────────────────────────────────────────────────────────────
     Example 4: SPLIT Scenario - No Partitioning Occurs
     ───────────────────────────────────────────────────────────────────────────────

     Time t=0:          Time t=1:

     ┌─────────┐        ┌─────────┐
     │ Object A│───┬───→│ Object B│
     │   100   │   │    │   60    │
     │  cells  │   │    │  cells  │
     └─────────┘   │    └─────────┘
                   │
                   │    ┌─────────┐
                   └───→│ Object C│
                        │   50    │
                        │  cells  │
                        └─────────┘

     STAGE 1 - Overlap Checking:
     ────────────────────────────
     A→B: 55/min(100,60) = 55/60 = 0.917 >= 0.5 ✓ LINKED
     A→C: 45/min(100,50) = 45/50 = 0.9   >= 0.5 ✓ LINKED

     One parent linked to two children → SPLIT DETECTED!

     STAGE 2 - Partitioning:
     ────────────────────────
     N.B.: NO PARTITIONING occurs for splits!
     All objects (A, B, C) grouped into same event with same ID
     Object B & C continue to evolve disjoint, yet will ultimately be labelled with the same event ID

  **Key Principles**:

  * **Overlap checking happens FIRST** for all object pairs between consecutive timesteps
  * **Partitioning happens SECOND** and ONLY when merges are detected (many→one)
  * **nn_partitioning choice does NOT affect** which objects are linked in Stage 1
  * **Splits (one→many) do NOT trigger** partitioning - all objects share the same event ID

  **When to Adjust**:

  * **Lower threshold** (e.g., 0.3): More continuous tracking, may link distant objects
  * **Higher threshold** (e.g., 0.7): More conservative, stricter continuity requirement
  * **Default** (0.5): Balanced approach for most applications

Grid Configuration Parameters
-----------------------------

**dimensions** : dict, default={'time': 'time', 'x': 'lon', 'y': 'lat'}
  Mapping of conceptual dimensions to actual dimension names.
  For unstructured grids, use: ``{'time': 'time', 'x': 'ncells'}``

**coordinates** : dict, default={'time': 'time', 'x': 'lon', 'y': 'lat'}
  Mapping of conceptual coordinates to actual coordinate names.
  For unstructured grids, use: ``{'time': 'time', 'x': 'lon', 'y': 'lat'}``

**grid_resolution** : float, optional
  Grid resolution in degrees for structured grids (ignored for unstructured grids).
  When provided, automatically calculates physical cell areas using spherical geometry.
  **Overrides** any provided ``cell_areas`` parameter for structured grids.

  .. code-block:: python

     # Automatic from grid resolution
     tracker = marEx.tracker(
         extremes, mask,
         R_fill=8,
         grid_resolution=0.25  # Automatically calculates spherical areas for 0.25° grid
     )

     # This is equivalent to manually provide 2D data array of cell areas
     tracker = marEx.tracker(
         extremes, mask,
         R_fill=8,
         cell_areas=my_calculated_areas  # Manual calculation required
     )

  The calculation accounts for latitude-dependent cell areas using spherical geometry,
  providing accurate physical areas (in km²) for object/event size calculations.

**cell_areas** : xr.DataArray, optional
  Physical cell areas for area calculations.

  * **For structured grids**: Optional. If not provided, defaults to 1.0 for each cell (i.e. cell counts).
    N.B.: If neither ``cell_areas`` nor ``grid_resolution`` is provided, areas are in units of cells/pixels.
    Note: Overridden by ``grid_resolution`` if provided.
  * **For unstructured grids**: Required for physical area calculations.

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
* **area**: Spatial area of each event through time (in units of cell counts, or physical units if cell_areas/grid_resolution provided)
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
