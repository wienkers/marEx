=======================
Tracking Extreme Events
=======================

The tracker turns the boolean ``extreme_events`` field from the detect stage into **events**:
labelled objects that keep one identity through time, with their area, centroid, lifetime and
merge history. It accepts any boolean field on a latitude/longitude grid or an unstructured
mesh, so it also runs on masks that did not come from marEx. See :doc:`../api/track` for the
signatures of :class:`marEx.tracker` and :func:`marEx.regional_tracker`.

.. contents::
   :local:
   :depth: 2

Why Overlap-Based Tracking
==========================

Labelling the field as a 3-D array (time as a third spatial axis) joins any two objects that
touch in space-time. Events that brush against each other for one day become permanently
linked, and a few weeks later a single "event" spans the basin. Its duration, area and
intensity describe no physical phenomenon.

The tracker instead labels each timestep separately and links objects between consecutive
timesteps only when they overlap by at least ``overlap_threshold`` of the smaller object's
area. Where several objects merge, the merged child is partitioned back to its parents, and
each merge is written to ``merge_ledger``. The video shows both approaches on the same data.

.. video:: /_static/videos/tracking_comparison.mp4
   :width: 700
   :autoplay:
   :loop:

Left: chain-reaction merging, where A touches B, the result touches C, and all three become
one event. Right: overlap-thresholded merging with genealogy, where the three events keep
their identities.

Quick Start
===========

The tracker needs an active ``dask.distributed`` client. Without one, ``run()`` fails with
"No clients found". Pass ``memory_limit`` explicitly: Dask reads the memory of the whole node,
not the cgroup limit of a batch allocation, so without it a memory constraint never binds.

.. code-block:: python

   import xarray as xr
   import marEx

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

``R_fill`` has no default and ``mask`` is the second positional parameter, so
``marEx.tracker(events, 8)`` binds ``8`` to ``mask``. Always pass ``R_fill`` by keyword.
``data_bin`` must be Dask-backed and of boolean dtype. A tracker instance is single use: a
second ``run()`` raises ``TrackingError``.

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
closing handles the first, the opening the second. With ``R_fill=1``::

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

Two properties of the closing matter in practice. It can bridge two objects that are separated
by less than the kernel, and a speck between them acts as a stepping stone. The area filter
runs after the closing, so it cannot undo such a bridge. ``prefilter_min_cells`` removes the
specks first.

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
cell to the nearest parent **centroid**. ``True`` assigns it to the parent that owns the
nearest parent **cell**. Centroid partitioning goes wrong for non-convex parents. Take a
C-shaped object A and a small nearby object B that merge into one 20-cell child::

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
Zhang, 2023). The choice affects only how merged children are partitioned. It does not change
which objects are linked.

The number of parents of one child, and the number of merges in one timestep, are held in
fixed-width arrays of 64 entries. These are implementation widths, not physical limits, and
an event that exceeds them raises ``TrackingError``. Raising ``overlap_threshold`` reduces the
number of accepted parents.

Parameters
==========

Required
--------

``data_bin`` : :class:`xarray.DataArray`
   Boolean, Dask-backed, with dimensions ``(time, lat, lon)`` or ``(time, cells)``. Extra
   dimensions such as depth are rejected with ``TrackingError``: select one level with
   ``isel`` first, or loop over levels.

``R_fill`` : int
   See `The Morphological Step`_. It is required and has no default. Guidance: small values (3 to
   5) preserve narrow features and suit coarse grids, larger values (10 to 15) produce more
   coherent objects from noisy fields. Choose it against the physical scale you want to
   bridge, in cells.

Optional Inputs
---------------

``mask`` : :class:`xarray.DataArray`, optional
   Boolean validity mask (``True`` = valid), as returned by ``preprocess_data``. Omit it for
   a field with no invalid region, such as an atmospheric variable. Omitting it is equivalent
   to passing an all-``True`` mask. An all-``False`` mask raises an error.

``prefilter_min_cells`` : int, optional (keyword-only)
   Drop connected components smaller than this many cells at each timestep, before the
   closing. Default ``None`` (off). On a fine grid it stops specks from bridging objects. The
   unstructured example in the repository uses 23 cells, roughly one 0.25 degree cell on the
   ICON R02B09 mesh. The attribute ``prefilter_min_cells`` is written to the output only when
   the option is set.

``area_filter_quartile`` : float in (0, 1)
   Fraction of the smallest objects to remove. Default 0.5 when neither area filter is given.
   Adaptive: the cut-off adapts to the object-size distribution of each dataset.
   On an unstructured mesh the quantile is evaluated over objects above a small size cut-off
   (50 cells), not over every speck, so the same value removes a different fraction there
   than on a regular grid.

``area_filter_absolute`` : int
   Minimum object area (in cells) to keep. Reproducible across datasets. Mutually exclusive
   with ``area_filter_quartile``. The cut-off that was applied is recorded as the attribute
   ``area_threshold (cells)``.

``T_fill`` : int, default 2
   Temporal closing. Must be even. The kernel is ``T_fill + 1`` **timesteps**, so on daily data
   ``T_fill=2`` bridges absences of up to two days, and on a weekly or monthly cadence the unit
   is weeks or months. ``T_fill=0`` skips the step and is the usual choice for coarse
   cadences. For daily data 2 to 4 is typical, and larger values link more intermittent
   events at the cost of joining distinct ones.

``allow_merging`` : bool, default ``True``
   ``False`` runs classical connected-component labelling with time connectivity. The output
   then holds ``ID_field`` only. The unstructured tracker always uses the merge path.

``overlap_threshold`` : float, default 0.5
   Fraction of the smaller object's area that must overlap for two objects to be linked.
   Lower values (0.3) give more continuous tracks and link more marginal pairs. Higher values
   (0.7) are stricter.

``nn_partitioning`` : bool, default ``False``
   See `Merges and Splits`_.

``max_iteration`` : int, default 40
   Iteration limit of the unstructured merge loop. Unused on regular grids.

``checkpoint`` : ``'save'`` | ``'load'`` | ``None``
   Writes or reads the preprocessed binary field and its statistics in ``temp_dir``, so a
   rerun with different linking parameters skips the morphology. Requires ``temp_dir``.

``debug`` : int (0 to 2), ``verbose``, ``quiet``
   Logging controls. ``run()`` also prints a "Tracking Statistics" block to standard output,
   independent of the logging level.

Grid and Areas
--------------

``dimensions``, ``coordinates`` : dict
   Defaults ``{"time": "time", "x": "lon", "y": "lat"}``. Required for other names. For an
   unstructured mesh ``dimensions`` carries only the cell dimension under ``"x"``.

``grid_resolution`` : float, optional
   Degrees, for regular grids. Computes spherical cell areas in km\ :sup:`2` (Earth radius
   6378 km) and **overrides** ``cell_areas``. Not accepted for unstructured meshes.

``cell_areas`` : :class:`xarray.DataArray`, optional
   Physical cell areas, in whatever unit you supply. Without ``grid_resolution`` or
   ``cell_areas`` on a regular grid, every cell has unit area and ``area`` is a cell count.
   Required on unstructured meshes.

``unstructured_grid``, ``neighbours``, ``temp_dir`` : see `Unstructured Meshes`_.

``regional_mode``, ``coordinate_units`` : see `Regional Domains`_.

``compute_mode``, ``temp_dir`` : see `Compute Modes and Larger-Than-Memory Tracking`_.

Grid Types
==========

Regular Latitude/Longitude Grids
--------------------------------

Coordinates are converted to degrees internally. The tracker decides the units, and whether
the domain wraps in longitude, from a coordinate range of about 360 degrees (or 2π). A field
that does not span the globe raises ``CoordinateError``: use the regional tracker.

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
       R_fill=80,                           # neighbour hops, not cells of a regular grid
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

These settings are the ones used for the ICON R02B09 tracking video (14.9 million cells).
They are a starting point for a comparable mesh, not defaults to reuse at another
resolution.

The unstructured tracker's events are independent of the time chunking and of the worker
layout, except where a cell is equidistant from two parents: the nearest-cell partition then breaks the tie
by cell order, and a different chunking can break it differently. In the full-mesh
comparisons this affected a very small number of cells.

Regional Domains
----------------

A limited-area field has no 360 degree range to detect. Use ``regional_tracker``, which takes
the same arguments plus ``coordinate_units``:

.. code-block:: python

   regional = marEx.regional_tracker(
       region_events,                  # boolean, dims (time, lat, lon)
       region_mask,
       coordinate_units="degrees",     # "degrees" or "radians"
       R_fill=8,
       area_filter_quartile=0.5,
   ).run()

``regional_tracker`` without ``coordinate_units`` raises ``ConfigurationError``, and regional
mode is not available on unstructured meshes.

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

The ``ID`` coordinate runs from 1 to ``N_events_final``. The dataset attributes record how
the run was configured and what the filters did: ``N_objects_prefiltered`` and
``N_objects_filtered`` (object counts before and after the area filter),
``N_events_final``, ``R_fill``, ``T_fill``, ``area_threshold (cells)``,
``accepted_area_fraction``, ``preprocessed_area_fraction``, and, with merging,
``overlap_threshold``, ``nn_partitioning``, ``total_merges`` and ``multi_parent_merges``.

``preprocessed_area_fraction`` is the area of the input field divided by the area after the
closing, opening and filtering. It is measured against the input **before** any
``prefilter_min_cells`` step, so enabling the prefilter changes it. Compare it with 1 to see
how much area the morphology added or removed.
``accepted_area_fraction`` is the share of object area that survives the area filter.

With ``run(return_merges=True)`` the tracker also returns a second Dataset with one entry per
merge: ``parent_IDs``, ``child_IDs``, ``overlap_areas``, ``merge_time``, ``n_parents`` and
``n_children``. Its IDs are the object IDs of the identification stage. ``merge_ledger`` in the
main dataset is in final event IDs, and is the one to use for event-level questions.

.. code-block:: python

   events, merges = event_tracker.run(return_merges=True)

Event Statistics
================

All per-event variables other than ``ID_field`` are small, so compute them once and work in
memory. ``ID_field`` stays lazy.

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

These variables are the inputs to event catalogues: duration, area and the footprint give
the frequency and severity of events for exposure studies. See :doc:`../applications/index`
for worked cases.

Visualising Tracks
==================

``ID_field`` plots directly with ``plot_IDs=True`` (see :doc:`visualisation`). Movies can
overlay event outlines and centroids through the ``object_ids`` and ``centroids`` arguments
of ``animate``:

.. code-block:: python

   config = marEx.PlotConfig(title="Tracked Events", plot_IDs=True)
   fig, ax, im = events.ID_field.isel(time=0).plotX.single_plot(config)

.. _larger-than-memory-tracking:

Compute Modes and Larger-Than-Memory Tracking
=============================================

``compute_mode`` controls what happens to the whole-field intermediates (the filled and
filtered fields, the ID field and the merge ledger accumulator).

``'persist'`` (default)
   Pins them in cluster memory. Fastest, and the right choice whenever the run fits.

``'streaming'``
   Stages them to Zarr under ``temp_dir`` and reads them back, so the bytes held in worker
   memory depend on the cluster and the time chunk and not on the length of the series.
   Requires ``temp_dir``. Supports both grid types.

``'lazy'`` is rejected: the merge loop is sequential in time, so recomputing buys nothing.

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

What streaming changes
----------------------

Streaming cuts the **pinned** (resident) bytes. It does not by itself lower peak memory in
every configuration, and the measurements below say which cases move.

Measured on DKRZ Levante, single runs unless stated:

* Gridded tracker, 0.25 degree global field, 3804 days (a 15.8 GB int32 ID field), at a
  96 GB dask budget: pinned bytes fell from 337 GB (persist) to 0.34 GB (streaming), and peak
  cluster memory from 56.7 GB to 19.1 GB. Wall time was within 1 % (1104 s and 1100 s).
* The same tracker in a 24 GiB SLURM allocation with a 16 GB dask budget (4 workers x 4 GB):
  ``persist`` was OOM-killed in 5 of 5 runs, while ``streaming`` completed in 7 of 7 runs in
  the same allocation, with peak memory of 6.65 to 7.30 GB. This is the larger-than-memory
  result. The ID field alone is 15.8 GB against the 24 GiB allocation, so the margin is
  modest, and no allocation threshold for ``persist`` is claimed.
* ICON R02B09 (14.9 million cells), 1096 days, 16 x 12 GB: pinned bytes fell from 751 GB to
  148 GB, but peak memory fell only 6.5 %, and ``persist`` completes at this budget (about
  72 minutes). With 4 x 8 GB, ``persist`` did not finish within the 5 hour limit in two runs,
  and ``streaming`` completed in 3 h 50 min. That is a wall-clock statement, not a
  memory-failure one.

The gridded peak reduction therefore does not transfer to the unstructured tracker. Plan
around pinned bytes, and size the cluster with a margin: ``P2PConsistencyError``,
``KilledWorker`` and a timeout are symptoms, and the cause is usually a worker over 95 % of
its memory limit.

Staging, disk and chunking requirements
---------------------------------------

* **Disk.** About 14 bytes per cell-timestep before compression (roughly 55 GB for the
  gridded run above, less on disk since the ID fields are mostly zero). The ICON run needs
  about 230 GB at 1096 steps. Size ``temp_dir`` for it.
* **The staging directory outlives** ``run()`` by design. Write your output, then call
  :func:`marEx.clear_staging`. The path is on ``events.encoding["marex_staging_dir"]``, not
  in ``attrs``. ``xr.merge`` and ``xr.concat`` drop that encoding, after which
  ``clear_staging`` can only warn. Cleanup also runs at interpreter exit, but not after a
  ``SIGKILL`` such as a wall-clock kill, so sweep ``temp_dir`` periodically.
* **Uniform time chunks.** Every chunk except a shorter last one. ``.chunk({"time": k})``
  satisfies this. Ragged input is re-chunked with a warning.
* **Equivalence.** On gridded data ``ID_field`` and the event properties are bit-identical
  across the two modes and across time chunkings in the tests, and a full global tracking
  comparison (3438 days, 3116 events) matched in every compared output. The unstructured
  tracker matches except for the equidistant tie-breaks described above.

Chunking
========

The tracker needs the **spatial dimensions whole** and chunks **time**. Connected-component
labelling and the dilation are global in space, so a spatially split field is rebuilt into
one chunk per timestep, and the transposition is the expensive part. This is the opposite of
advice for ``preprocess_data`` on a large unstructured mesh, which chunks the cell dimension.

.. code-block:: python

   # regular grid: space whole, time in blocks
   data_bin = data_bin.chunk({"time": 25, "lat": -1, "lon": -1})

   # unstructured: cells whole, small time blocks
   data_bin = data_bin.chunk({"time": 5, "ncells": -1})

* A time chunk of 25 is what the gridded examples use (about 26 MB per chunk for a 0.25
  degree global boolean field). Keep the chunk at least ``T_fill + 1`` timesteps long, so the
  temporal closing does not have to rebalance chunk boundaries.
* On the full ICON R02B09 mesh the time chunk is a few timesteps, because a chunk is already
  tens of megabytes. Every chunk must be at least 2 timesteps, and the last one must not be a
  single step. A time chunk of 4 over 365 days leaves a one-step tail and fails, whereas 5
  works. After a ``.sel`` along time, re-chunk.
* ``neighbours`` and ``cell_areas`` do not constrain the chunking of the input.

Troubleshooting
===============

``No clients found``
   Start a ``dask.distributed`` client before ``run()``.

``DataValidationError``: not binary
   ``data_bin`` must be boolean. Compare to a threshold, or ``.astype(bool)``.

``TrackingError``: extra dimension
   The tracker is 2-D in space. Select a level (``isel(depth=0)``) or loop over levels.

``CoordinateError`` on a regional domain
   The longitude range is not about 360 degrees. Use ``regional_tracker`` with
   ``coordinate_units``.

Workers killed, or ``P2PConsistencyError``
   Look for a worker memory warning above the traceback. Try ``compute_mode="streaming"``,
   smaller time chunks, or more memory per worker. For ICON-scale unstructured tracking,
   12 GB per worker is a stable setting, whereas 9 GB per worker was bistable.

One enormous event
   ``overlap_threshold`` is too low, ``T_fill`` or ``R_fill`` too large, or the specks that
   chain objects together need ``prefilter_min_cells``.

Many tiny events
   Increase the area filter, or raise ``R_fill`` so the opening removes them.

Complete Workflow
=================

.. code-block:: python

   import xarray as xr
   import marEx

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
