===========
Why marEx?
===========

marEx has three modules. :func:`marEx.anomaly.compute` returns anomalies, :func:`marEx.extremes.identify` thresholds anomalies from any source,
and :class:`marEx.tracker` tracks events in any boolean field.
:func:`marEx.preprocess_data` chains the first two. Four anomaly methods
(``shifting_baseline``, ``detrend_fixed_baseline``, ``fixed_baseline``,
``detrend_harmonic``) and two threshold methods (``seasonal_percentile``,
``global_percentile``) cover the common definitions, including the Hobday et al.
(2016) marine heatwave definition.

1. Larger Than Memory
=====================

For detection ``compute_mode`` takes ``persist`` (the default), ``lazy`` or ``streaming``, and for the tracker ``persist`` or ``streaming``. In
``streaming`` mode, intermediates that several later steps read are written to a
Zarr store and re-opened from disk, so far less data is pinned in worker memory.
On 20 years of 6-hourly 0.25 degree global SST, ``streaming`` tracked a 91 GB ID
field inside a 32 GiB job.

The default percentile method accumulates a histogram per cell and cycle slot
from one chunk at a time. ``precision`` sets its bin width, and ``method_percentile="exact"`` is available when the series fits in memory.

:mod:`marEx.helper` starts local or SLURM clusters, and ``ResourceMonitor``
reports wall time, memory and spill per stage. Pass ``memory_limit``
explicitly, since Dask sizes workers from the node's physical memory whatever
the job's allocation. See :doc:`guide/performance` for sizing.

2. Overlap-Based Tracking
=========================

Basic space-time connected-component labelling treats time as one more spatial
dimension, so any two objects that touch anywhere in space-time are joined.
Events that touch briefly become permanently linked, and the result is a
basin-spanning 'mega-event' made of many independent phenomena, whose
statistics have no mechanistic meaning.

marEx matches objects between timesteps by area overlap. A merge or split
requires an overlap above ``overlap_threshold``, a fraction of the smaller
object's area, and each merge records its parents in ``merge_ledger``, so the
history of a large event can be reconstructed.

.. video:: /_static/videos/tracking_comparison.mp4
   :width: 700
   :autoplay:
   :loop:

**Left**: chain-reaction merging. Event A touches B and becomes AB, AB touches C,
and all three are one event. **Right**: overlap-thresholded merging with
genealogy, where the three events keep their identities.

* ``R_fill`` closes and opens objects in space before tracking, and ``T_fill``
  fills short gaps in time.
* ``area_filter_quartile`` (adaptive) or ``area_filter_absolute`` (comparable
  across datasets) removes small objects, and ``prefilter_min_cells`` drops
  specks before the morphology step.
* ``nn_partitioning=True`` assigns each cell of a split event to its nearest
  parent cell, and the default assigns it by parent centroid.
* ``regional_tracker`` handles bounded domains, and ``grid_resolution`` gives
  spherical cell areas on regular grids.

See :doc:`guide/tracking`.

3. Any Grid, Cadence and Tail
=============================

* **Grids**: the same call runs on lat/lon grids and on unstructured meshes
  (FESOM, ICON, MPAS). On meshes the tracker uses sparse-matrix morphology, and
  the ``plotX`` accessor detects the grid type and draws maps, panels and
  animations for both.
* **Dimensions**: a field may have an extra dimension such as depth or
  pressure level, which detection treats as a broadcast axis, so a 4-D run
  equals the per-level 3-D runs. With depth declared, the tracker follows 4-D events as
  volumes (m\ :sup:`3`) through ``(time, depth, lat, lon)``, while ``plotX`` takes one level at a time.
* **Cadence**: daily, monthly and sub-daily time axes are inferred, or passed as
  a :class:`marEx.SeasonalCycle`.
* **Tail**: ``tail="lower"`` flags the coldest or driest values, for cold
  spells, wind drought and precipitation drought.

See :doc:`guide/dimensions_and_time` and :doc:`applications/index`.

Next Steps
==========

* :doc:`installation`: install marEx
* :doc:`getting_started/quickstart`: a first run
* :doc:`guide/index`: the user guide
* :doc:`applications/index`: worked cases by domain
* :doc:`whats_new`: changes in 5.0
