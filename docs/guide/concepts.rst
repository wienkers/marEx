=============
Core Concepts
=============

marEx turns a gridded or unstructured time series into a catalogue of tracked extreme
events. The work is split into three stages, each usable on its own, and every stage runs
on Dask so the data never has to fit in memory at once. This page defines the vocabulary
the rest of the guide uses and shows how the entry points fit together.

The Pipeline
============

.. code-block:: text

   field ──> anomaly ──> extremes ──> track ──> events
              stage       stage       stage

   marEx.anomaly.compute   marEx.extremes.identify   marEx.tracker
   └────────── marEx.preprocess_data ───────────┘

1. **Anomaly stage.** Subtract a climatology from the raw field. Output: ``dat_anomaly``
   and a ``mask``. See :doc:`anomalies`.
2. **Extremes stage.** Compare the anomaly with a percentile threshold. Output:
   ``thresholds`` and a boolean ``extreme_events`` field. See :doc:`extremes`.
3. **Tracking stage.** Label connected extreme regions, follow them through time, and
   handle merges and splits. Output: ``ID_field`` plus per-event statistics. See
   :doc:`tracking`.

:func:`marEx.preprocess_data` chains stages 1 and 2. Stage 3 takes the boolean
``extreme_events`` field (and, optionally, the ``mask``) from any source, so a threshold
you computed elsewhere can be tracked without running marEx's own stages 1 and 2.

Vocabulary
==========

Climatology
  The typical value of the field for a location and a position in the seasonal cycle,
  estimated from the data. A day-of-year mean of sea surface temperature is the usual
  example, but any variable with a seasonal cycle works.

Anomaly
  The field minus its climatology, ``anomaly = observed - climatology``. marEx has four
  ways to build the climatology, which differ in how they treat trends
  (:doc:`anomalies`).

Threshold
  The percentile of the anomaly distribution that separates ordinary from extreme values,
  one number per cell, and per position in the seasonal cycle for ``seasonal_percentile``.
  ``threshold_percentile=95`` with ``tail="upper"`` gives the value exceeded 5 % of the
  time (Hobday et al. 2016).

Extreme
  A cell and time at which the anomaly passes the threshold on the chosen side (``>=`` for
  ``tail="upper"``, ``<=`` for ``tail="lower"``). The result is a boolean field with the
  same shape as the anomaly.

Event
  A connected region of extremes followed through time. The tracker assigns each event an
  integer ID and records its area, centroid, lifetime and its merge and split history.

Mask
  A boolean field marking the valid cells, computed once as ``isfinite`` of the first
  timestep. Land, or any cell that is NaN at the first timestep, is excluded for the whole
  run. See :doc:`dimensions_and_time`.

Three Entry Points
==================

The anomaly and extremes stages are available as standalone functions as well as through
the chainer. The chainer is the shortest way to a result.

.. code-block:: python

   import xarray as xr
   import marEx

   sst = xr.open_zarr("sst_daily.zarr").sst      # Dask-backed

   ds = marEx.preprocess_data(
       sst,
       method_anomaly="shifting_baseline",
       method_extreme="seasonal_percentile",
       threshold_percentile=95,
   )

The same two stages, run separately:

.. code-block:: python

   anomalies = marEx.anomaly.compute(sst, method="shifting_baseline", window_years=15)
   ds = marEx.extremes.identify(
       anomalies, method="seasonal_percentile", threshold_percentile=95
   )

:func:`marEx.extremes.identify` accepts a DataArray of anomalies from any source, or a
Dataset carrying ``dat_anomaly``. :func:`marEx.anomaly.compute` has no threshold parameter
anywhere, so it can be used for climatologies and anomalies alone
(:doc:`../applications/climatologies_only`).

The choice of method is spelled differently at each entry point:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Meaning
     - ``preprocess_data``
     - ``anomaly.compute``
     - ``extremes.identify``
   * - Anomaly method
     - ``method_anomaly``
     - ``method``
     - not applicable
   * - Threshold method
     - ``method_extreme``
     - not applicable
     - ``method``
   * - Everything else (``window_years``, ``smooth_days``, ``window_days``, ``tail``, ...)
     - keyword
     - keyword-only
     - keyword-only

``method`` is the second positional argument of ``compute`` and ``identify``. Every other
parameter is keyword-only there. Parameters of the same name mean the same thing at every
entry point. Some combinations are rejected outright, for example ``standardise=True``
with any method but ``detrend_harmonic``, or ``reference_period`` with a method that has
no fixed reference. Each is listed on the page for its stage.

Outputs
=======

``preprocess_data`` returns one Dataset (daily gridded data, default methods):

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Variable
     - Dimensions
     - Meaning
   * - ``dat_anomaly``
     - ``(time, lat, lon)``
     - Anomaly field, float32
   * - ``mask``
     - ``(lat, lon)``
     - Valid cells
   * - ``extreme_events``
     - ``(time, lat, lon)``
     - Boolean extreme flag
   * - ``thresholds``
     - ``(lat, lon, dayofyear)``
     - Threshold per cell and cycle slot

With ``standardise=True`` the Dataset also carries ``dat_stn``, ``STD``,
``extreme_events_stn`` and ``thresholds_stn``. ``anomaly.compute`` returns the anomaly
variables and ``mask``. ``extremes.identify`` returns its input variables plus
``extreme_events`` and ``thresholds``. The attributes of the Dataset record the resolved
method names, window lengths, percentile, tail and the histogram bin width that was
actually used, so a saved file documents how it was made.

Any dimension besides time and the horizontal ones is carried through as an extra
dimension (depth, pressure level, ensemble member), and ``mask`` and ``thresholds`` gain it
too.

Grids
=====

A field has dimensions ``(time, *extra, *horizontal)``. A **gridded** field has two
horizontal dimensions, for example ``(time, lat, lon)``. An **unstructured** field has one,
for example ``(time, ncells)``, with latitude and longitude as coordinates. The same
functions handle both, with the dimension names passed through ``dimensions`` and
``coordinates``. Details, including what each grid type additionally requires for tracking,
are in :doc:`dimensions_and_time`.

Scale and Compute Modes
=======================

marEx requires Dask-backed input and builds lazy graphs, so the data is read and reduced
in pieces. ``compute_mode`` sets what happens to intermediates that more than one later step
needs:

* ``persist`` (default) pins them in cluster memory.
* ``streaming`` writes them to Zarr under ``scratch_dir`` and reads them back. It reduces the
  bytes pinned in memory, not the peak memory of a run.
* ``lazy`` keeps nothing and recomputes.

Two measurements, both single runs on one DKRZ Levante compute node:

* 40 years of daily 0.25° global SST (output 9282 × 720 × 1440 after the 15-year baseline
  is trimmed), anomalies and approximate percentiles, 4 workers × 22 GB with 16 threads
  each: 3954 s with ``persist`` and 3663 s with ``streaming``, with identical results.
* Gridded tracking of 3804 days of 0.25° data: ``persist`` was OOM-killed in a 24 GiB
  allocation while ``streaming`` completed in the same allocation.

Chunking, sizing and cluster set-up are in :doc:`performance`. How correctness is tested is
in :doc:`validation`.

Where to Go Next
================

* :doc:`../getting_started/quickstart` runs the full pipeline end to end.
* :doc:`anomalies` and :doc:`extremes` explain how to choose methods and settings.
* :doc:`dimensions_and_time` covers extra dimensions, grids and cadences.
* :doc:`tracking` and :doc:`visualisation` cover the later stages.
* :doc:`performance` covers chunking, memory sizing and clusters.

References
==========

* Hobday et al. (2016), "A hierarchical approach to defining marine heatwaves", *Progress
  in Oceanography* 141, 227-238.
  `doi:10.1016/j.pocean.2015.12.014 <https://doi.org/10.1016/j.pocean.2015.12.014>`_.
  Defines the day-of-year percentile threshold that ``seasonal_percentile`` generalises.
* Sun et al. (2023), "Marine heatwaves in the Arctic Region: Variation in Different Ice
  Covers", *Progress in Oceanography* 203, 102947.
  `doi:10.1016/j.pocean.2022.102947 <https://doi.org/10.1016/j.pocean.2022.102947>`_.
  The tracking approach that marEx extends with improved merge and split partitioning.
