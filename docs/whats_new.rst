==============
What's New
==============

What's New in 5.0
=================

Version 5.0 splits the detect stage into three peer stages, each usable on its own: ``marEx.anomaly`` (climatologies and anomalies), ``marEx.extremes`` (thresholding) and the tracker. The full list of changes is in the `changelog <https://github.com/wienkers/marEx/blob/main/CHANGELOG.md>`_.

* **Any variable, any cadence.** Fields with an extra dimension such as depth or pressure level run through all four anomaly methods and both threshold methods, and a 3-D run equals the per-level 2-D runs. Monthly and sub-daily axes are supported; the within-year axis is inferred (``dayofyear``, ``month``, ``hourofyear``) or passed as ``cycle=marEx.SeasonalCycle(...)``.
* **Lower tail.** ``tail="lower"`` flags cold spells, droughts and other low-side extremes. ``threshold_percentile=5, tail="lower"`` is the coldest 5 %.
* **Data-derived histogram range.** For the approximate percentile path, ``precision`` alone sets the bin width and the binned range follows the data, so the same call works for temperature, wind or precipitation. If a threshold reaches the edge of the range, the range is regrown and the thresholds recomputed.
* **Streaming compute mode.** ``compute_mode="streaming"`` keeps intermediates on disk instead of pinning them in worker memory, for detect and for the tracker. It cuts pinned bytes. Peak memory falls only where pinned data dominated, as on the long gridded track. Measured on the gridded tracker (3804 days), ``persist`` was OOM-killed in a 24 GiB allocation while ``streaming`` completed in the same allocation. Write your output, then call ``marEx.clear_staging(ds)``.
* **Chunk-independent detect.** Detect output no longer depends on how the input is chunked in time or space. This is verified on the test fixtures for every anomaly and threshold method.
* **Tracker.** ``prefilter_min_cells`` drops small objects before the morphology step; ``mask`` is optional; merge and parent records are 64 wide (were 20 and 10); the unstructured tracker's events no longer depend on the time chunking, except for equidistant tie-breaks.
* **Zarr 3.** zarr-python 2 (``>=2.18``) and 3 are both supported, along with current xarray releases. Dask is un-pinned (``>=2025.9.0``).
* **Resource monitor.** ``marEx.helper.ResourceMonitor`` reports wall time and memory per pipeline stage.

Already in 4.1.x
----------------

Two fixes were released as 4.1.1 and 4.1.2 and are not new in 5.0: the ``fill_holes`` fix at the periodic-longitude seam in the tracker, and the ``to_netcdf`` fix for tuple-valued attributes.

Migrating from 4.x
==================

There are no compatibility shims. Old names raise rather than warn, except ``max_anomaly`` and ``n_bins``, which still work with a ``FutureWarning``.

Paths and Entry Points
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 4.x
     - 5.0
   * - ``marEx.detect`` (module)
     - Removed. ``import marEx.detect`` raises ``ModuleNotFoundError``.
   * - ``marEx.preprocess_data(da, ...)``
     - Same name, new parameters (below).
   * - ``marEx.compute_normalised_anomaly``
     - ``marEx.anomaly.compute(da, method=...)``, or ``marEx.anomaly.compute_normalised_anomaly`` for the legacy return shape.
   * - ``marEx.identify_extremes`` (returns a tuple)
     - ``marEx.extremes.identify(data, method=...)`` returns a Dataset. ``marEx.extremes.identify_extremes`` keeps the tuple return.
   * - ``marEx.rolling_climatology``, ``marEx.smoothed_rolling_climatology``
     - ``marEx.anomaly.rolling_climatology``, ``marEx.anomaly.smoothed_rolling_climatology``
   * - ``marEx.detect.add_decimal_year``
     - ``marEx.core.time_axis.add_decimal_year``
   * - ``marEx.helper.checkpoint_to_zarr``, ``marEx.helper.fix_dask_tuple_array``
     - Removed, with no replacement (checkpointing is no longer needed).
   * - ``marEx.helper`` (module)
     - A package. ``configure_dask``, ``start_local_cluster``, ``start_distributed_cluster`` and ``get_cluster_info`` keep their names; ``ResourceMonitor`` is new.
   * - ``marEx.track``, ``marEx.regional_tracker``, ``marEx.plotX``
     - Same top-level names. ``track`` is now a package.
   * - (none)
     - ``marEx.anomaly``, ``marEx.extremes``, ``marEx.ComputeMode``, ``marEx.SeasonalCycle``, ``marEx.infer_cycle``, ``marEx.clear_staging``

Methods and Parameters
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 4.x
     - 5.0
   * - ``method_extreme="hobday_extreme"``
     - ``"seasonal_percentile"``
   * - ``method_extreme="global_extreme"``
     - ``"global_percentile"``
   * - ``method_anomaly`` values
     - Unchanged. On ``anomaly.compute`` and ``extremes.identify`` the argument is ``method``.
   * - ``window_year_baseline``
     - ``window_years``
   * - ``smooth_days_baseline``
     - ``smooth_days``
   * - ``window_days_hobday``
     - ``window_days``
   * - ``window_spatial_hobday``
     - ``window_spatial``
   * - ``std_normalise``
     - ``standardise``
   * - ``use_temp_checkpoints``
     - Removed.
   * - ``precision=0.01``, ``max_anomaly=5.0`` (defaults)
     - ``precision=None``; the range is derived from the data. ``max_anomaly`` and ``n_bins`` are deprecated; use ``precision``.
   * - (none)
     - ``tail``, ``compute_mode``, ``scratch_dir``, ``validate``, ``cycle``
   * - tracker: ``mask`` required
     - ``mask=None`` is allowed. ``R_fill`` is still required.
   * - (none)
     - tracker ``compute_mode``, ``prefilter_min_cells``
   * - ``tracker(..., checkpoint=)``
     - Still present; ``checkpoint="save"`` or ``"load"`` now requires ``temp_dir``.

Output Attributes and Variables
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 4.x
     - 5.0
   * - attr ``method_extreme`` of ``"hobday_extreme"`` or ``"global_extreme"``
     - ``"seasonal_percentile"`` or ``"global_percentile"``
   * - attrs ``window_year_baseline``, ``smooth_days_baseline``, ``std_normalise``, ``window_days_hobday``
     - ``window_years``, ``smooth_days``, ``standardise``, ``window_days``
   * - attr ``window_spatial_hobday``
     - ``window_spatial`` (the effective value; ``None`` on the exact path)
   * - (none)
     - attr ``tail``. The streaming staging directory is on ``ds.encoding["marex_staging_dir"]``, not in attrs.
   * - boolean and ``None`` attrs
     - Stored as int8 and ``"None"``, so output is NetCDF-safe.
   * - attrs inherited from the input onto ``dat_anomaly`` and similar
     - Not inherited on any xarray release.
   * - ``dayofyear`` scalar coordinate leaking from ``fixed_baseline``
     - Dropped.
   * - Variables ``dat_anomaly``, ``mask``, ``extreme_events``, ``thresholds`` (and ``_stn``)
     - Same.
   * - Tracker variables ``ID_field``, ``global_ID``, ``area``, ``centroid``, ``presence``, ``time_start``, ``time_end``, ``merge_ledger``
     - Same names.

Changed Defaults
----------------

* ``fixed_baseline`` and ``detrend_fixed_baseline`` smooth their day-of-year climatology with a 21-day circular moving average. ``smooth_days=1`` restores the unsmoothed climatology.
* The approximate percentile range and bin width come from the data (about 3000 bins over the tail's range) instead of a fixed ``precision=0.01`` and ``max_anomaly=5.0``.
* Tracker merge and parent limits: 64 and 64 (were 20 and 10).
* ``dask`` is ``>=2025.9.0`` (was ``==2025.3.0``); ``zarr>=2.18`` is declared.
* ``window_spatial`` is recorded as ``None`` for ``method_percentile="exact"``, where it was never used.
* On the exact path, a threshold at or below zero is nudged to the smallest value above zero.

Now Raises
----------

* A threshold reaching the outermost bin of a range you pinned with ``max_anomaly``: ``ConfigurationError`` (was a warning). With a derived range the range is regrown, or a ``UserWarning`` is issued.
* A ``precision`` that gives more than 65000 bins (more than 10000 warns).
* ``smooth_days`` spanning the whole cycle.
* ``detrend_harmonic`` on sub-daily data, which previously left the diurnal cycle in the anomaly silently.
* A ``dimensions`` mapping that names ``y`` or any non-time dimension without ``x``.
* An empty anomaly series (a ``ZeroDivisionError`` on the exact path before).
* ``compute_mode="lazy"`` on the tracker.
* Calling ``tracker.run()`` twice, and ``checkpoint="save"`` without ``temp_dir``.
* A ``NaT`` in the time axis, in the decimal-year step.

Results That Change on Purpose
==============================

Some outputs differ from 4.x because the earlier behaviour was wrong, depended on chunking, or depended on a library release. Each is listed in the changelog. Expect small differences when you rerun an old analysis.

* **Chunk-independent detect.** Anomalies from ``shifting_baseline``, ``fixed_baseline`` and ``detrend_fixed_baseline`` used to move with the time chunking. They no longer do.
* **Rolling means on the NumPy path.** The optional ``bottleneck`` running sum drifts with series length and chunking, and current xarray no longer uses it by default. marEx now always uses the NumPy path.
* **Smoothed fixed baselines.** ``fixed_baseline`` and ``detrend_fixed_baseline`` anomalies change by default (see above).
* **Approximate global quantile.** The 1-D quantile interpolates within the containing bin, values beyond the range count in the outermost bin instead of being dropped (thresholds can only move up), and cells valid for part of the year keep a threshold.
* **Data-derived range.** Default bins are no longer 0.01 in the data's units, so thresholds shift by a fraction of a bin and a small fraction of events flip.
* **Exact percentiles on degenerate cells.** Cells with no variance, such as permanent sea ice, no longer flag every timestep.
* **Histogram counts** no longer wrap at 65535 samples per (cell, slot, bin).
* **Leap years.** ``fixed_baseline`` with a reference period containing no leap year no longer leaves day 366 NaN.
* **Tracker morphology.** Closing and opening use an exact Euclidean disk, and the periodic-longitude seam is padded to the full reach. This changes events slightly near the seam.
* **Tracker first object.** The first object in raster order at the first timestep was dropped on every gridded and regional track.
* **Tracker area-filter ties and edges.** Objects exactly at the area cutoff are kept on both grid types; latitude no longer wraps in the morphology padding; the antimeridian margin scales with the grid.
* **Unstructured tracker.** Events used to depend on the time chunking and worker layout. They no longer do, except for cells exactly equidistant between candidate parents.
* **Merge loop.** Events fused by an ID-range overrun now stay separate.

Shifting-baseline anomalies on sub-daily data still leave part of the diurnal cycle in the anomaly. This is unchanged from 4.x; prefer ``fixed_baseline`` or ``detrend_fixed_baseline`` for sub-daily input.
