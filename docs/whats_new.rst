==============
What's New
==============

What's New in 5.0
=================

The full list of changes is in the `changelog <https://github.com/wienkers/marEx/blob/main/CHANGELOG.md>`_.

* **Three modules.** The detect step is split into ``marEx.anomaly`` (climatologies and anomalies) and ``marEx.extremes`` (thresholds), alongside the tracker. Each runs on its own, and ``preprocess_data`` chains the first two.
* **Larger than memory.** ``compute_mode="streaming"`` keeps intermediates on disk for detection and tracking, so far less data is pinned in worker memory. Write your output, then call ``marEx.clear_staging(ds)``.
* **Any variable, any cadence.** Fields with an extra dimension such as depth or pressure level run through every anomaly and threshold method. Monthly and sub-daily axes are supported, with the cycle inferred (``dayofyear``, ``month``, ``hourofyear``) or passed as ``cycle=marEx.SeasonalCycle(...)``.
* **Lower tail.** ``tail="lower"`` flags cold spells, droughts and other low-side extremes. ``threshold_percentile=5, tail="lower"`` is the coldest 5 %.
* **Data-derived histogram range.** ``precision`` alone sets the bin width and the binned range follows the data, so the same call works for temperature, wind or precipitation.
* **Tracker.** ``prefilter_min_cells`` drops small objects before the morphology step, ``mask`` is optional, and merge and parent records are 64 wide (were 20 and 10).
* **Dependencies and monitoring.** zarr-python 2 (``>=2.18``) and 3 are both supported, dask is ``>=2025.9.0``, and ``marEx.helper.ResourceMonitor`` reports wall time and memory per stage.

Migrating from 4.x
==================

Old names raise an error, except ``max_anomaly`` and ``n_bins``, which still work with a ``FutureWarning``.

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
     - A package. ``configure_dask``, ``start_local_cluster``, ``start_distributed_cluster`` and ``get_cluster_info`` keep their names. ``ResourceMonitor`` is new.
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
     - ``precision=None``. The range is derived from the data. ``max_anomaly`` and ``n_bins`` are deprecated. Use ``precision``.
   * - (none)
     - ``tail``, ``compute_mode``, ``scratch_dir``, ``validate``, ``cycle``
   * - tracker: ``mask`` required
     - ``mask=None`` is allowed. ``R_fill`` is still required.
   * - (none)
     - tracker ``compute_mode``, ``prefilter_min_cells``
   * - ``tracker(..., checkpoint=)``
     - Still present. ``checkpoint="save"`` or ``"load"`` now requires ``temp_dir``.

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
     - ``window_spatial`` (the effective value, omitted on the exact path)
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

Rerunning a 4.x Analysis
========================

Some outputs differ from 4.x. Expect small differences when you rerun an old analysis, mainly from these changes:

* **Chunk-independent detection.** Anomalies from ``shifting_baseline``, ``fixed_baseline`` and ``detrend_fixed_baseline`` used to move with the time chunking. They now depend only on the data.
* **Smoothed fixed baselines.** ``fixed_baseline`` and ``detrend_fixed_baseline`` smooth their day-of-year climatology with a 21-day circular moving average. ``smooth_days=1`` restores the unsmoothed climatology.
* **Data-derived bins.** The approximate percentile range and bin width come from the data (about 3000 bins over the tail's range), so thresholds shift by a fraction of a bin and a small fraction of events flip.
* **Leap years.** The day-366 climatology pools every year's 31 December, so anomalies change on that day and thresholds move near the year end.
* **Sub-daily shifting baseline.** ``shifting_baseline`` smooths each time of day across days, which removes the diurnal cycle from the anomaly.
* **Tracker.** Closing and opening use an exact Euclidean disk, and the periodic-longitude seam is padded to the full reach. The first object in raster order at the first timestep, previously dropped on gridded and regional tracks, is kept. Unstructured events are independent of the time chunking up to exactly equidistant cells. Events change slightly.
* **New errors.** Several inputs now raise, among them ``detrend_harmonic`` on sub-daily data, ``compute_mode="lazy"`` on the tracker, a second ``tracker.run()`` and ``checkpoint="save"`` without ``temp_dir``.

The changelog lists every change, with the remaining edge cases.
