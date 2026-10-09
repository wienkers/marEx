===========
Why marEx?
===========

marEx is built around five design choices. This page states each one, what it
buys, and where its limits are. Numbers are single runs on DKRZ Levante and are
measured, not guaranteed.

1. Stages That Stand Alone
==========================

The pipeline is three functions with three outputs: anomalies
(:func:`marEx.anomaly.compute`), extremes (:func:`marEx.extremes.identify`) and
tracked events (:class:`marEx.tracker`). :func:`marEx.preprocess_data` chains
the first two.

* The anomaly stage has no threshold parameter anywhere. A smoothed climatology
  or a detrended anomaly of decades of global daily data is a complete
  result on its own.
* The extreme stage accepts a DataArray of anomalies from any source, or a
  Dataset carrying ``dat_anomaly``.
* The tracker takes any boolean field, not only marEx output.

Four anomaly methods (``shifting_baseline``, ``detrend_fixed_baseline``,
``fixed_baseline``, ``detrend_harmonic``) and two threshold methods
(``seasonal_percentile``, ``global_percentile``) cover the usual definitions,
including the Hobday et al. (2016) marine heatwave definition. The seasonal
method extends the temporal window with an optional spatial window
(``window_spatial``) that pools neighbouring cells, which stabilises thresholds
on short or noisy records. It applies to gridded data with approximate
percentiles only.

See :doc:`guide/anomalies` and :doc:`guide/extremes`.

2. Any Grid, Cadence and Tail
=============================

* **Grids**: the same call runs on lat/lon grids and on unstructured meshes
  (FESOM, ICON, MPAS). The tracker uses sparse-matrix morphology on meshes.
* **Dimensions**: a field may carry an extra dimension such as depth or
  pressure level. Detection treats it as a broadcast axis, and a 3-D run equals
  the per-level 2-D runs. The tracker and ``plotX`` are 2-D, so select one
  level first.
* **Cadence**: daily, monthly and sub-daily axes are inferred, or passed as a
  :class:`marEx.SeasonalCycle`. ``detrend_harmonic`` rejects sub-daily data
  with a clear error.
* **Tail**: ``tail="lower"`` flags the coldest or driest values, for cold
  spells, wind drought and precipitation drought.

See :doc:`guide/dimensions_and_time` and :doc:`applications/index`.

3. Larger Than Memory
=====================

``compute_mode`` takes ``persist`` (the default), ``lazy`` or ``streaming``. In
``streaming`` mode, intermediates that several later steps read are written to a
Zarr store and re-opened instead of being pinned in worker memory.

What this changes is the amount of data pinned in worker memory. It does not, in
general, lower peak memory. The measured evidence is the gridded tracker on a
0.25 degree global field of 3804 days:

* In a 24 GiB SLURM allocation, ``persist`` was OOM-killed in 5 of 5 runs, and
  ``streaming`` completed in 7 of 7 at a 4 x 4 GB dask budget.
* At a 96 GB budget, one run each: pinned bytes fell from 337 GB to 0.34 GB, the
  peak from 56.7 GB to 19.1 GB, and wall time stayed within 1 %.
* On the unstructured ICON R02B09 mesh (14.9 M cells, 1096 days) at 4 x 8 GB,
  ``persist`` had not finished after 5 h in two runs and ``streaming`` finished
  in 3 h 50 min. At 16 x 12 GB both finished in about 72 min, and pinned bytes
  fell from 751 GB to 148 GB.

Detection with ``streaming`` produced the same arrays as ``persist`` at full
scale (40 years of daily 0.25 degree data in about an hour on one node, 4 workers
x 22 GB), but did not show a peak-memory saving. These are single runs for
detection and few for tracking. See :doc:`guide/performance` for sizing guidance
and the full caveats.

4. Tracking That Does Not Fuse Unrelated Events
===============================================

**The mega-event problem.** Basic 3-D connected-component labelling treats time
as one more spatial dimension and joins any objects that touch anywhere in
space-time. Events that briefly touch become permanently linked, and the result
is a basin-spanning "mega-event" that combines many independent phenomena. The
statistics of such an object have no mechanistic meaning.

**What marEx does.** Objects are matched between timesteps by overlap. A merge
or split requires that the overlap exceed ``overlap_threshold`` (a fraction of
the smaller object's area), and contact alone does not qualify. Each merge
records its parents in ``merge_ledger``, so the history of a large event can be
reconstructed.

.. video:: /_static/videos/tracking_comparison.mp4
   :width: 700
   :autoplay:
   :loop:

**Left**: chain-reaction merging. Event A touches B and becomes AB, AB touches C,
and all three are one event. **Right**: overlap-thresholded merging with
genealogy, where the three events keep their identities.

Supporting controls:

* ``nn_partitioning=True`` allocates the area of a split event by the nearest
  parent cell instead of the parent centroid.
* ``R_fill`` applies morphological closing and opening in space before tracking,
  and ``T_fill`` closes short gaps in time.
* ``area_filter_quartile`` (adaptive) or ``area_filter_absolute`` (reproducible
  across datasets) removes small objects, and ``prefilter_min_cells`` drops
  specks before the morphology step.
* ``grid_resolution`` computes spherical cell areas on regular grids.
* ``regional_tracker`` handles bounded domains.

See :doc:`guide/tracking`.

5. Results Independent of Chunking
==================================

Dask results can depend on how the data is chunked, through reduction order or
window boundaries. marEx runs its reductions on an internal canonical layout and
restores the caller's chunking afterwards. Detection output does not depend on
the input chunking, verified on the test fixtures for every anomaly and
threshold method. The gridded tracker agreed across time chunks and between
``persist`` and ``streaming`` in the cases tested. The unstructured tracker is
chunk-independent except for the assignment of exactly equidistant cells.

Outputs are also held to reference-output tests at zero tolerance, apart from 2e-14 of round-off on one threshold field. The test
scope, and the cases it does not cover, are described in :doc:`guide/validation`.

Infrastructure
==============

* **Histogram percentiles**: the default ``method_percentile="approximate"``
  accumulates a histogram per cell and cycle slot, so the full time series never
  has to be resident. Its accuracy is the bin width, set with ``precision``, and
  the binned range is derived from the data. ``"exact"`` is available when the
  series fits.
* **Numba** is a core dependency for the tracking kernels. JAX is optional and
  used only in building the sparse dilation matrix for unstructured grids.
* **HPC**: :mod:`marEx.helper` starts local or SLURM clusters, and
  ``ResourceMonitor`` reports wall time, memory and spill per stage. Pass
  ``memory_limit`` explicitly, since Dask does not see a SLURM memory cap.
* **plotX**: an xarray accessor that detects the grid type and plots maps,
  panels and animations for gridded and unstructured data alike.
* **Coordinates**: degrees and radians are auto-detected for global grids, with
  an override for regional domains.

Next Steps
==========

* :doc:`installation`: install marEx
* :doc:`getting_started/quickstart`: a first run in five minutes
* :doc:`guide/index`: the user guide
* :doc:`applications/index`: worked cases by domain
* :doc:`whats_new`: changes in 5.0
