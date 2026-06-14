==========
Quickstart
==========

This guide gets you detecting, tracking, and visualising extreme events in a
few minutes. For the full runnable versions, see the :doc:`../tutorials/index`.

Pre-process SST data
====================

*cf.* the gridded :doc:`Preprocess notebook <../tutorials/gridded/01_preprocess_extremes>`.

.. code-block:: python

   import xarray as xr
   import marEx

   # Load SST data & rechunk for optimal processing
   file_name = 'path/to/sst/data'
   sst = xr.open_dataset(file_name, chunks={'time': 500}).sst

   # Process data
   extremes = marEx.preprocess_data(
      sst,
      method_anomaly='shifting_baseline',      # Anomalies from a rolling climatology using previous window_year years
      method_extreme='seasonal_percentile',         # Local day-of-year specific thresholds with windowing
      threshold_percentile=95,                 # 95th percentile threshold for extremes
      window_years=15,                 # Rolling climatology window
      smooth_days=21,                 #    and smoothing window for determining the anomalies
      window_days=11,                   # Window size of compiled samples collected for the extremes detection
   )

   # Performance note: JAX acceleration automatically used if available
   if marEx.has_dependency('jax'):
      print("🚀 Using JAX-accelerated computations")

**Output variables:**

* ``dat_anomaly`` (time, lat, lon): Anomaly data
* ``extreme_events`` (time, lat, lon): Binary field locating extreme events (1=event, 0=background)
* ``thresholds`` (dayofyear, lat, lon): Extreme event thresholds used to determine extreme events
* ``mask`` (lat, lon): Valid data mask

Identify & Track Extreme Events
=================================

*cf.* the gridded :doc:`Tracking notebook <../tutorials/gridded/02_id_track_events>`.

.. code-block:: python

   import xarray as xr
   import marEx

   # Load pre-processed data
   file_name = 'path/to/binary/extreme/data'
   chunk_size = {'time': 25, 'lat': -1, 'lon': -1}
   extremes = xr.open_dataset(file_name, chunks=chunk_size)

   # ID, track, & merge
   tracker = marEx.tracker(
      extremes.extreme_events,
      extremes.mask,
      area_filter_quartile=0.5,      # Remove the smallest 50% of the identified coherent extreme areas
      R_fill=8,                      # Fill small holes with radius < 8 _cells_
      T_fill=2,                      # Allow gaps of 2 days and still continue the event tracking with the same ID
      allow_merging=True,            # Allow extreme events to split/merge. Keeps track of merge events & unique IDs.
      overlap_threshold=0.5,         # Overlap threshold for merging events. If overlap < threshold, events keep independent IDs.
      nn_partitioning=True,          # Use nearest-neighbor partitioning
   )
   tracked_events, merge_events = tracker.run(return_merges=True)

**Output variables:**

* ``ID_field`` (time, lat, lon): Field containing the IDs of tracked events (0=background)
* ``global_ID`` (time, ID): Unique global ID of each object; ``global_ID.sel(ID=10)`` maps event ID 10 to its original ID at each time
* ``area`` (time, ID): Area of each event as a function of time
* ``centroid`` (component, time, ID): (x, y) centroid coordinates of each event as a function of time
* ``presence`` (time): Presence (boolean) of each event at each time (anywhere in space)
* ``time_start`` (ID): Start time of each event
* ``time_end`` (ID): End time of each event
* ``merge_ledger`` (time, ID, sibling_ID): Sibling IDs for merging events (matching ``ID_field``); ``-1`` indicates no merging event occurred

* If ``return_merges=True``, the ``merge_events`` dataset will include:

  * ``parent_IDs`` (merge_ID, parent_idx): Original parent IDs of each merging event
  * ``child_IDs`` (merge_ID, child_idx): Original child IDs of each merging event
  * ``overlap_areas`` (merge_ID, parent_idx): Area of overlap between parent and child objects in each merging event
  * ``merge_time`` (merge_ID): Time of each merging event
  * ``n_parents`` (merge_ID): Number of parent objects in each merging event
  * ``n_children`` (merge_ID): Number of child objects in each merging event

Visualise results
=================

*cf.* the gridded :doc:`Visualisation notebook <../tutorials/gridded/03_visualise_events>`.

.. code-block:: python

   # Plot MHW frequency
   fig, ax, im = (tracked_events.ID_field > 0).mean("time").plotX.single_plot(
       marEx.PlotConfig(var_units="MHW Frequency", cmap="hot_r", cperc=[0, 96])
   )

   # Create an animated movie of tracked events
   tracked_events.ID_field.plotX.animate(
       marEx.PlotConfig(plot_IDs=True), plot_dir="./plots", file_name="mhw_animation"
   )

That's it! You've detected, tracked, and visualised extreme events in your data.

Next steps
==========

* Read the :doc:`../guide/index` for detailed workflows and method selection.
* Explore the :doc:`../api/index` for all available functions.
* Work through the :doc:`../tutorials/index` for complete notebooks.
