=======================
Tracking Extreme Events
=======================

The ``marEx.tracker`` module turns the boolean ``extreme_events`` field from the
``marEx.extremes`` module into **events**: labelled objects that keep one identity through time,
with their area, centroid, lifetime and merge history. It accepts any boolean field on a
latitude/longitude grid or an unstructured mesh, including masks from outside marEx. See
:doc:`../api/track` for the signatures of :class:`marEx.tracker` and
:func:`marEx.regional_tracker`.

.. contents::
   :local:
   :depth: 2

Why Overlap-Based Tracking
==========================

Labelling the field as a 3-D array (time as a third spatial axis) joins any two objects that
touch in space-time. Events that brush against each other for one day become permanently
linked, and a few weeks later a single "event" spans the basin, with a duration, area and
intensity that describe no physical phenomenon.

The marEx tracker labels each timestep separately and links objects at consecutive timesteps
only when they overlap by at least ``overlap_threshold`` of the smaller object's area. When
several objects merge, the merged child is partitioned back to its parents, and each merge is
recorded in ``merge_ledger``. The video compares both approaches on the same data.

.. video:: /_static/videos/tracking_comparison.mp4
   :width: 700
   :autoplay:
   :loop:

Left: chain-reaction merging, where A touches B, the result touches C, and all three become
one event. Right: overlap-thresholded merging with genealogy, where the three events keep
their identities.

Quick Start
===========

The tracker needs an active ``dask.distributed`` client. Pass ``memory_limit`` explicitly,
since Dask otherwise sizes its workers from the memory of the whole node and ignores the limit
of a batch allocation.

.. code-block:: python

   import xarray as xr
   import marEx

   # In a script, run this under `if __name__ == "__main__":` (its worker processes re-import the script).
   client = marEx.helper.start_local_cluster(
       n_workers=4, threads_per_worker=1, memory_limit="8GB"
   )

   # Output of marEx.preprocess_data (or any boolean field), chunked in time only
   ds = xr.open_zarr("extremes.zarr")
   extreme_events = ds.extreme_events.chunk({"time": 25, "lat": -1, "lon": -1})

   event_tracker = marEx.tracker(
       extreme_events,
       ds.mask,
       R_fill=8,                    # radius, in grid cells, of the spatial closing/opening
       area_filter_quartile=0.5,    # drop the smallest 50 % of objects
       T_fill=2,                    # close temporal gaps of up to 2 timesteps
       allow_merging=True,
       overlap_threshold=0.5,
       nn_partitioning=True,
       grid_resolution=0.25,        # areas in km^2 on a regular 0.25 degree grid
   )
   events = event_tracker.run()

   events.to_zarr("tracked_events.zarr", mode="w")

``R_fill`` is required and ``mask`` is the second positional parameter, so
``marEx.tracker(events, 8)`` binds ``8`` to ``mask``. Always pass ``R_fill`` by keyword.
``data_bin`` must be a Dask-backed boolean array. A tracker instance runs only once.

How the Tracker Works
=====================

``run()`` applies the following steps in order.

1. **Prefilter** (optional, ``prefilter_min_cells``). Connected components smaller than the
   given number of cells are dropped at each timestep, before any morphology.
2. **Spatial closing and opening** (``R_fill``). A closing (dilation then erosion) fills holes
   and bridges narrow gaps within an object. An opening (erosion then dilation) removes
   isolated specks. Both use a disk of radius ``R_fill``.
3. **Temporal gap filling** (``T_fill``). A closing along the time axis with a kernel of
   ``T_fill + 1`` timesteps joins an object to its later self across a short absence. Holes
   this opens up in space are refilled with a radius of ``R_fill // 2``.
4. **Area filter** (``area_filter_quartile`` or ``area_filter_absolute``). Each timestep is
   labelled into connected components, and objects below the area threshold are removed.
5. **Identification across time.** Objects at consecutive timesteps are linked when
   ``overlap_area / min(area_parent, area_child) >= overlap_threshold``.
6. **Merge and split handling** (``allow_merging=True``). A child with several accepted
   parents is a merge, a parent with several accepted children is a split. Events are the
   connected groups of linked objects, and the properties of each are recomputed.

The Morphological Step
----------------------

Extreme events are spatially coherent, so small holes inside one are usually artefacts of a
threshold crossing at neighbouring cells, and isolated flagged cells are usually noise. The
closing fills the holes and the opening removes the noise. With ``R_fill=1``::

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

The kernel is a disk of diameter ``2 * R_fill + 1`` cells. On a gridded field ``R_fill`` counts
grid cells. On an unstructured mesh it counts neighbour hops, so the same value spans a
different physical distance on each mesh and needs to be chosen per resolution.

The closing can also bridge two objects separated by less than the kernel, and a speck between
them acts as a stepping stone. The area filter runs after the closing, so the bridge stays.
Set ``prefilter_min_cells`` to remove the specks first.

Merges and Splits
-----------------

Linking depends on ``overlap_threshold`` alone. With the default of 0.5, a 100-cell object at
time *t* and an 80-cell object at *t+1* are linked when at least 40 cells overlap (the smaller
area is the denominator).

* **Split** (one parent, several linked children): the children continue under the parent's
  event ID. No partitioning is needed.
* **Merge** (several parents, one child): the cells of the child are partitioned back to the
  parents, so each lineage keeps its own identity and area history. The parents are recorded in
  ``merge_ledger``.

``nn_partitioning`` selects how the child is partitioned. The default ``False`` assigns each
cell to the nearest parent centroid. ``True`` assigns it to the parent that owns the nearest
parent cell. Centroid partitioning misassigns cells of non-convex parents. Take a C-shaped object A and a
small nearby object B that merge into one 19-cell child::

   Merged child (#)                  Centroid partition            Nearest-cell partition
   ┌───┬───┬───┬───┬───┐            ┌───┬───┬──╦┬───┬───┐         ┌───┬───┬───┬───┬───┐
 5 │   │ # │ # │ # │ # │          5 │   │ A │ A║│ B │ B │       5 │   │ A │ A │ A │ A │
   ├───┼───┼───┼───┼───┤            ├───┼───┼──╫┼───┼───┤         ├───┼───┼───┼───┼───┤
 4 │ # │ # │ # │   │   │          4 │ A │ A │ A║│   │   │       4 │ A │ A │ A │   │   │
   ├───┼───┼───┼───┼───┤            ├───┼───┼──╫┼───┼───┤         ├───┼───┼───┼───┼───┤
 3 │ # │ # │ # │ # │ # │          3 │ A │ A │ A║│ B │ B │       3 │ A │ A │ B │ B │ B │
   ├───┼───┼───┼───┼───┤            ├───┼───┼──╫┼───┼───┤         ├───┼───┼───┼───┼───┤
 2 │ # │ # │ # │   │   │          2 │ A │ A │ A║│   │   │       2 │ A │ A │ A │   │   │
   ├───┼───┼───┼───┼───┤            ├───┼───┼──╫┼───┼───┤         ├───┼───┼───┼───┼───┤
 1 │   │ # │ # │ # │ # │          1 │   │ A │ A║│ B │ B │       1 │   │ A │ A │ A │ A │
   └───┴───┴───┴───┴───┘            └───┴───┴──╩┴───┴───┘         └───┴───┴───┴───┴───┘
     1   2   3   4   5                1   2   3   4   5               1   2   3   4   5

The centroid of A lies in the hollow of the C, away from A's cells, so the dividing line cuts
the child into a B that is spatially disjoint. The nearest-cell partition keeps B contiguous.
Use ``nn_partitioning=True`` unless you need to reproduce a centroid-based method (Sun and
Zhang, 2023).

One child can have at most 64 parents, and one timestep at most 64 merges. Beyond either limit
the tracker raises ``TrackingError``, and raising ``overlap_threshold`` reduces the number of
accepted parents.

Parameters
==========

Required
--------

``data_bin`` : :class:`xarray.DataArray`
   Boolean, Dask-backed, with dimensions ``(time, lat, lon)`` or ``(time, cells)``. For a field
   with an extra dimension such as depth, select one level with ``isel`` first, or loop over
   levels.

``R_fill`` : int
   See `The Morphological Step`_. Small values (3 to 5) preserve narrow features and suit
   coarse grids, and larger values (10 to 15) give more coherent objects from noisy fields.
   Choose it from the physical scale you want to bridge, in cells.

Optional Inputs
---------------

``mask`` : :class:`xarray.DataArray`, optional
   Boolean validity mask (``True`` = valid), as returned by ``preprocess_data``. Omitting it is
   equivalent to an all-``True`` mask, which suits a field with no invalid region, such as an
   atmospheric variable.

``prefilter_min_cells`` : int, optional (keyword-only)
   Drop connected components smaller than this many cells at each timestep, before the
   closing. Default ``None`` (off). On a fine grid it stops specks from bridging objects. The
   unstructured example in the repository uses 23 cells, roughly one 0.25 degree cell on the
   ICON R02B09 mesh.

``area_filter_quartile`` : float in (0, 1)
   Fraction of the smallest objects to remove. Default 0.5 when neither area filter is given.
   The cut-off adapts to the object-size distribution of each dataset. On an unstructured mesh
   the quantile is taken over objects above 50 cells, so the same value removes a different
   fraction there than on a regular grid.

``area_filter_absolute`` : int
   Minimum object area (in cells) to keep, reproducible across datasets. Mutually exclusive
   with ``area_filter_quartile``. The applied cut-off is recorded in the attribute
   ``area_threshold (cells)``.

``T_fill`` : int, default 2
   Temporal closing. Must be even. The kernel is ``T_fill + 1`` timesteps, so on daily data
   ``T_fill=2`` bridges absences of up to two days, and on a weekly or monthly cadence the unit
   is weeks or months. ``T_fill=0`` skips the step and is the usual choice for coarse
   cadences. For daily data 2 to 4 is typical, and larger values link more intermittent
   events at the cost of joining distinct ones.

``allow_merging`` : bool, default ``True``
   ``False`` runs classical connected-component labelling with time connectivity, and the
   output contains ``ID_field`` only. The unstructured tracker always uses the merge path.

``overlap_threshold`` : float, default 0.5
   Fraction of the smaller object's area that must overlap for two objects to be linked.
   Lower values (0.3) give more continuous tracks and link more marginal pairs. Higher values
   (0.7) are stricter.

``nn_partitioning`` : bool, default ``False``
   See `Merges and Splits`_.

``max_iteration`` : int, default 40
   Iteration limit of the unstructured merge loop (regular grids ignore it). A change near the
   start of the record can need one pass per remaining time chunk to propagate, so when events
   span much of the record, set it to at least the number of time chunks.

``checkpoint`` : ``'save'`` | ``'load'`` | ``None``
   Writes or reads the preprocessed binary field and its statistics in ``temp_dir``, so a
   rerun with different linking parameters skips the morphology. Requires ``temp_dir``.

``debug`` : int (0 to 2), ``verbose``, ``quiet``
   Logging controls. ``run()`` also prints a "Tracking Statistics" block to standard output at
   every logging level.

Grid and Areas
--------------

``dimensions``, ``coordinates`` : dict
   Defaults ``{"time": "time", "x": "lon", "y": "lat"}``. Set them when your names differ. For
   an unstructured mesh ``dimensions`` contains only the cell dimension, under ``"x"``.

``grid_resolution`` : float, optional
   Grid spacing in degrees, for regular grids only. Computes spherical cell areas in
   km\ :sup:`2` (Earth radius 6378 km) and takes precedence over ``cell_areas``.

``cell_areas`` : :class:`xarray.DataArray`, optional
   Physical cell areas, which set the unit of ``area``. On a regular grid without
   ``grid_resolution`` or ``cell_areas``, every cell has unit area and ``area`` is a cell
   count. Required on unstructured meshes.

``unstructured_grid``, ``neighbours``, ``temp_dir`` : see `Unstructured Meshes`_.

``regional_mode``, ``coordinate_units`` : see `Regional Domains`_.

``compute_mode``, ``temp_dir`` : see `Compute Modes and Larger-Than-Memory Tracking`_.

Grid Types
==========

Regular Latitude/Longitude Grids
--------------------------------

Coordinates are converted to degrees internally. The tracker infers the units, and whether the
domain wraps in longitude, from a coordinate range of about 360 degrees (or 2π). For a field
that covers only part of the globe, use the regional tracker (see `Regional Domains`_).

Unstructured Meshes
-------------------

An unstructured mesh needs the connectivity of its cells, their areas, and a scratch
directory, all passed explicitly:

.. code-block:: python

   import marEx

   ds = xr.open_zarr("icon_extremes.zarr")  # from marEx.preprocess_data(..., neighbours=..., cell_areas=...)

   event_tracker = marEx.tracker(
       ds.extreme_events.chunk({"time": 5, "ncells": -1}),
       ds.mask,
       R_fill=80,                           # in neighbour hops
       area_filter_absolute=13549,
       T_fill=4,
       overlap_threshold=0.25,
       allow_merging=True,
       nn_partitioning=True,
       prefilter_min_cells=23,
       unstructured_grid=True,
       dimensions={"time": "time", "x": "ncells"},
       coordinates={"time": "time", "x": "lon", "y": "lat"},
       neighbours=ds.neighbours,            # shape (3, ncells), dims ("nv", "ncells")
       cell_areas=ds.cell_areas,
       temp_dir="/path/to/scratch/marex_tracking",
   )
   events = event_tracker.run()

These are the settings of the ICON R02B09 tracking video (14.9 million cells), and a starting
point for a comparable mesh. Retune them for another resolution.

Events on an unstructured mesh are identical across time chunkings and worker layouts, apart
from cells equidistant from two parents. The nearest-cell partition breaks such ties by cell
order, which can change with the chunking.

Regional Domains
----------------

For a limited-area field, use ``regional_tracker``. It takes the same arguments plus the
required ``coordinate_units``, and works on regular grids:

.. code-block:: python

   regional = marEx.regional_tracker(
       region_events,                  # boolean, dims (time, lat, lon)
       region_mask,
       coordinate_units="degrees",     # "degrees" or "radians"
       R_fill=8,
       area_filter_quartile=0.5,
   ).run()

Output
======

``run()`` returns a Dataset with dimensions ``time, lat, lon, ID, component, sibling_ID`` (or
``cells`` in place of ``lat, lon``):

.. list-table::
   :header-rows: 1
   :widths: 18 24 58

   * - Variable
     - Dimensions
     - Content
   * - ``ID_field``
     - ``(time, lat, lon)``, int32
     - Event ID at each cell. 0 is background. The values index the ``ID`` dimension.
   * - ``global_ID``
     - ``(time, ID)``, int32
     - The per-timestep object ID (before events are formed) that belongs to each event.
   * - ``area``
     - ``(time, ID)``, float32
     - Event area at each time, in the units of ``cell_areas`` (km\ :sup:`2` with
       ``grid_resolution``, otherwise cells).
   * - ``centroid``
     - ``(component, time, ID)``, float32
     - Latitude (``component=0``) and longitude (``component=1``) of the centroid, in the
       input's coordinate units.
   * - ``presence``
     - ``(time, ID)``, bool
     - True while the event exists.
   * - ``time_start``, ``time_end``
     - ``(ID)``, datetime64
     - First and last time of presence.
   * - ``merge_ledger``
     - ``(time, ID, sibling_ID)``, int32
     - Parent event IDs at merge times, ``-1`` where unused.

``ID`` takes the values 1 to ``N_events_final``. The dataset attributes record the
configuration and what the filters did: ``N_objects_prefiltered`` and ``N_objects_filtered``
(object counts before and after the area filter), ``N_events_final``, ``R_fill``, ``T_fill``,
``area_threshold (cells)``, ``accepted_area_fraction``, ``preprocessed_area_fraction``, and,
with merging, ``overlap_threshold``, ``nn_partitioning``, ``total_merges`` and
``multi_parent_merges``.

``preprocessed_area_fraction`` is the area of the input field divided by the area after the
closing, opening and filtering. It is measured against the input before any
``prefilter_min_cells`` step, so enabling the prefilter changes it. Compare it with 1 to see
how much area the morphology added or removed. ``accepted_area_fraction`` is the share of
object area that survives the area filter.

With ``run(return_merges=True)`` the tracker also returns a second Dataset with one entry per
merge: ``parent_IDs``, ``child_IDs``, ``overlap_areas``, ``merge_time``, ``n_parents`` and
``n_children``. Its IDs are the per-timestep object IDs from the identification step, whereas
``merge_ledger`` in the main dataset uses final event IDs and is the one to use for event-level
questions.

.. code-block:: python

   # Call this in place of run(): a tracker instance runs only once.
   events, merges = event_tracker.run(return_merges=True)

Event Statistics
================

All per-event variables other than ``ID_field`` are small, so compute them once and work in
memory. Keep ``ID_field`` lazy.

.. code-block:: python

   stats = events[["area", "centroid", "presence", "time_start", "time_end", "merge_ledger"]].compute()

   # Duration of every event, in days (timesteps are daily here)
   duration = (stats.time_end - stats.time_start).dt.days + 1

   # Peak and mean area over each event's lifetime
   area = stats.area.where(stats.presence)
   max_area = area.max("time")
   mean_area = area.mean("time")

   # Centroid track of event 17: latitude and longitude against time
   track = stats.centroid.sel(ID=17).where(stats.presence.sel(ID=17))
   lat_track, lon_track = track.sel(component=0), track.sel(component=1)

   # Number of recorded parents at each merge of every event
   n_parents = (stats.merge_ledger >= 0).sum("sibling_ID")

   # Footprint of event 17: every cell it ever covered (computes over time)
   footprint = (events.ID_field == 17).any("time")

   # Events longer than 10 days that reached 100,000 km^2 (with grid_resolution set)
   keep = (duration > 10) & (max_area > 1e5)
   long_large_events = stats.ID.where(keep, drop=True)

``duration`` counts calendar days only on a daily cadence. On other cadences count timesteps
with ``stats.presence.sum("time")``.

See :doc:`../applications/index` for event catalogues and other worked cases.

Visualising Tracks
==================

``ID_field`` plots directly with ``plot_IDs=True``:

.. code-block:: python

   config = marEx.PlotConfig(title="Tracked Events", plot_IDs=True)
   fig, ax, im = events.ID_field.isel(time=0).plotX.single_plot(config)

Movies can overlay event outlines and centroids through the ``object_ids`` and ``centroids``
arguments of ``animate`` (see :doc:`visualisation`).

.. _larger-than-memory-tracking:

Compute Modes and Larger-Than-Memory Tracking
=============================================

``compute_mode`` controls what happens to the whole-field intermediates (the filled and
filtered fields, the ID field and the merge ledger accumulator). The tracker has two modes, because its merge loop is sequential in time.

``'persist'`` (default)
   Pins them in cluster memory. The default, and the right choice whenever the run fits.

``'streaming'``
   Stages them to Zarr under ``temp_dir`` and reads them back, so the bytes pinned in worker
   memory are set by the cluster and the time chunk, independent of the length of the series.
   Requires ``temp_dir``. Supports both grid types.

.. code-block:: python

   event_tracker = marEx.tracker(
       extreme_events,
       ds.mask,
       R_fill=8,
       area_filter_quartile=0.5,
       compute_mode="streaming",
       temp_dir="/path/to/scratch/marex_staging",
   )
   events = event_tracker.run()

   # The result reads lazily from the staging store: write it, then release the store.
   events.to_zarr("tracked_events.zarr", mode="w")
   marEx.clear_staging(events)

With ``streaming``, the gridded tracker completed 15 years of 6-hourly 0.25 degree SST
extremes (a 91.0 GB ID field) in a 32 GiB job with 3 x 9 GB workers. Size the cluster from the
pinned bytes, with a margin. :doc:`performance` lists the measured runs for both grid types.

Staging, disk and chunking requirements
---------------------------------------

* **Disk.** About 14 bytes per cell-timestep before compression: roughly 55 GB for a 0.25
  degree global field over 3804 days, and about 230 GB for ICON R02B09 over 1096 steps. The ID
  fields are mostly zero, so the compressed size is smaller. Size ``temp_dir`` for it.
* **Cleanup.** The staging directory outlives ``run()``. Write your output, then call
  :func:`marEx.clear_staging`. The path is stored in ``events.encoding["marex_staging_dir"]``.
  ``xr.merge`` and ``xr.concat`` drop that encoding, after which ``clear_staging`` can only
  warn. Cleanup also runs at interpreter exit, but a ``SIGKILL`` (such as a wall-clock kill)
  skips it, so sweep ``temp_dir`` periodically.
* **Equivalence.** On gridded data ``ID_field`` and the event properties are bit-identical
  across the two modes and across time chunkings, and a full global track (3438 days, 3116
  events) matched in every compared output. The unstructured tracker matches apart from
  the equidistant tie-breaks (see `Unstructured Meshes`_).

Chunking
========

The tracker needs the spatial dimensions whole and chunks time. Connected-component labelling
and the dilation are global in space, so a spatially split field is rebuilt into one chunk per
timestep, and this transposition is expensive. ``preprocess_data`` on a large unstructured
mesh is the opposite case, and chunks the cell dimension.

.. code-block:: python

   # regular grid: space whole, time in blocks
   data_bin = data_bin.chunk({"time": 25, "lat": -1, "lon": -1})

   # unstructured: cells whole, small time blocks
   data_bin = data_bin.chunk({"time": 5, "ncells": -1})

* The gridded examples use a time chunk of 25 (about 26 MB per chunk for a 0.25 degree global
  boolean field). Keep the chunk at least ``T_fill + 1`` timesteps long, which avoids
  rebalancing chunk boundaries in the temporal closing.
* On the full ICON R02B09 mesh the time chunk is a few timesteps, because one chunk is already
  tens of megabytes.
* Time chunks must be uniform, apart from a shorter final chunk (a single step is accepted).
  ``.chunk({"time": k})`` gives this, and ragged input is re-chunked with a warning.
* ``neighbours`` and ``cell_areas`` can be chunked independently of the input.

Troubleshooting
===============

``No clients found``
   Start a ``dask.distributed`` client before ``run()``.

``DataValidationError``: not binary
   ``data_bin`` must be boolean. Compare to a threshold, or ``.astype(bool)``.

``TrackingError``: extra dimension
   The tracker is 2-D in space. Select a level (``isel(depth=0)``) or loop over levels.

``TrackingError``: the area filter removed every object
   ``area_filter_absolute`` is larger than the largest object. Lower it.

``TrackingError``: no objects larger than the cutoff on an unstructured grid
   Before the area filter, objects of 5 cells or fewer (50 under ``area_filter_quartile``) are
   dropped, and none was larger. Check that ``data_bin`` has events inside the mask, or relax
   the detection threshold so that events are larger.

``TrackingError``: the objects kept after filtering have zero total area
   ``cell_areas`` is zero over the kept objects. It must be positive wherever the mask is True.

``CoordinateError`` on a regional domain
   The field covers only part of the globe. Use ``regional_tracker`` with
   ``coordinate_units``.

Workers killed, or ``P2PConsistencyError``
   These are symptoms. The cause is usually a worker over its termination threshold (95 % of
   its memory limit), so look for a worker memory warning above the traceback. Try
   ``compute_mode="streaming"``, smaller time chunks, or more memory per worker. For ICON-scale
   unstructured tracking, give each worker at least 12 GB.

One enormous event
   ``overlap_threshold`` is too low, ``T_fill`` or ``R_fill`` too large, or the specks that
   chain objects together need ``prefilter_min_cells``.

Many tiny events
   Increase the area filter, or raise ``R_fill`` so the opening removes them.

``ValueError`` that the merge IDs or object count exceed the int32 range
   On an unstructured mesh the IDs are checked against int32 before merging. Track a shorter
   period or filter more small objects.

Complete Workflow
=================

.. code-block:: python

   import xarray as xr
   import marEx

   # In a script, run this under `if __name__ == "__main__":` (its worker processes re-import the script).
   client = marEx.helper.start_local_cluster(
       n_workers=4, threads_per_worker=1, memory_limit="8GB"
   )

   sst = xr.open_zarr("sst_daily.zarr").sst.chunk({"time": 365})

   extremes = marEx.preprocess_data(
       sst,
       method_anomaly="shifting_baseline",
       method_extreme="seasonal_percentile",
       threshold_percentile=95,
       window_years=15,
       smooth_days=21,
       window_days=11,
       dask_chunks={"time": 25},
   )

   events = marEx.tracker(
       extremes.extreme_events,
       extremes.mask,
       R_fill=8,
       area_filter_quartile=0.5,
       T_fill=2,
       allow_merging=True,
       overlap_threshold=0.5,
       nn_partitioning=True,
       grid_resolution=0.25,
   ).run()

   events.to_zarr("events.zarr", mode="w")
