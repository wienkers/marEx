========
Extremes
========

The ``marEx.extremes`` module turns anomalies into a threshold field and a boolean event
flag. Call it alone as :func:`marEx.extremes.identify`, or as the second half of
:func:`marEx.preprocess_data`. The function reference is in :doc:`../api/extremes`.

.. code-block:: python

   import xarray as xr
   import marEx

   # Anomalies from marEx, or from any other source
   anomalies = xr.open_zarr("anomalies.zarr").dat_anomaly      # Dask-backed

   ds = marEx.extremes.identify(
       anomalies,
       method="seasonal_percentile",
       threshold_percentile=95,
       tail="upper",
       window_days=11,
   )
   ds.extreme_events      # bool, same shape as the anomaly
   ds.thresholds          # one value per cell and cycle slot

``data`` may be a DataArray of anomalies or a Dataset containing ``dat_anomaly``, whose
other variables pass through to the output.

Threshold Methods
=================

``seasonal_percentile`` (default)
  One threshold per cell and per position in the seasonal cycle (``dayofyear``, ``month``
  or ``hourofyear``). The samples for each slot come from all years, from the
  ``window_days`` around the slot and, on a grid, from a ``window_spatial`` neighbourhood
  of cells. This is the day-of-year definition of Hobday et al. (2016). Use it when the
  spread of the anomaly changes through the year, which is the usual case.

``global_percentile``
  One threshold per cell from the whole series. Use it when the anomaly has no useful
  seasonal structure left (for instance after standardising), when the series is too
  short to populate a seasonal window, or for a fast first pass. It needs no seasonal
  cycle, so it works on any time axis, including irregular ones (the anomaly module may
  still need a ``cycle``).

``thresholds`` has a cycle dimension for ``seasonal_percentile`` only. Select from it by
dimension name, because the order of dimensions differs between the exact and approximate
paths.

.. _extremes-tail:

Which Tail
==========

``tail="upper"`` (default) flags ``anomaly >= threshold``, and ``tail="lower"`` flags
``anomaly <= threshold``. ``threshold_percentile`` is a percentile of the whole
distribution, so the coldest 5 % is ``threshold_percentile=5`` with ``tail="lower"``.

.. code-block:: python

   cold = marEx.extremes.identify(anomalies, threshold_percentile=5, tail="lower")

Both tails are resolved at the same bin width. Use the lower tail for cold spells, drought,
low wind and any other low-side extreme, and run the two tails separately to get both.

A cell whose anomaly is exactly zero throughout (permanent sea ice, a permanently masked
cell) stays unflagged in both tails.

Seasonal Window and Spatial Pooling
===================================

``window_days`` (default 11) is a duration in days, counted in cycle slots (:ref:`cadence-rules`),
so 11 days of 5-day data pools 11 day-of-year slots. On a day-of-year cycle (daily and
multi-day data) it must be odd, because the window is symmetric about the day. On other
cadences it is forced to an odd number of slots.

``window_spatial`` adds the samples of neighbouring cells to each cell's threshold to give
the percentile more samples. The field itself is left unsmoothed. Its value resolves as
follows:

* ``None`` (default) gives ``5``, a 5 × 5 neighbourhood, for ``seasonal_percentile`` with
  ``method_percentile="approximate"`` on a gridded field, and no pooling everywhere else.
* An odd integer sets the neighbourhood explicitly. ``window_spatial=1`` gives single-cell
  thresholds, as in Hobday et al. (2016). An explicit value is accepted only in the same
  configuration as the default 5 × 5.

When the x dimension has a one-dimensional numeric coordinate, the neighbourhood wraps in
longitude only if that coordinate spans the globe, so on a regional box the edge columns
pool only real neighbours, as the latitude edges always do. Without such a coordinate the
neighbourhood always wraps. The resolved value is recorded in the output attribute
``window_spatial``, which is present only when spatial pooling is used.

For daily data, the number of samples behind each threshold is about ``n_years × window_days × window_spatial²``,
and the number expected beyond the threshold is that times ``1 - q`` in the upper tail and
``q`` in the lower tail, with ``q = threshold_percentile / 100``. marEx logs a warning when
fewer than 10 samples are expected beyond the threshold.

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Configuration
     - Samples
     - Beyond the 95th
     - Beyond the 99th
   * - 20 years, ``window_spatial=1``
     - 220
     - 11
     - 2
   * - 20 years, ``window_spatial=5``
     - 5,500
     - 275
     - 55
   * - 30 years, ``window_spatial=1``
     - 330
     - 16
     - 3
   * - 30 years, ``window_spatial=5``
     - 8,250
     - 412
     - 82

Short records and high percentiles both call for pooling. For a 99th percentile on 30
years, a 7 × 7 window gives about 160 samples beyond the threshold. Pooling assumes
neighbouring cells share a distribution, the same assumption Hobday et al. make in time
with an 11-day window. Across a sharp coastline or front, where that assumption fails, use
a smaller window and a longer record.

Exact and Approximate Percentiles
=================================

``method_percentile="approximate"`` (default) counts samples into histogram bins and
interpolates inside the bin where the cumulative count crosses the percentile. It processes spatial tiles, so each task's memory is bounded, and it processes 40 years
of daily 0.25° global data (:doc:`performance`). ``"exact"`` sorts every sample a cell
contributes and applies ``np.nanpercentile`` with numpy's default linear rule. It needs each
tile's full series in memory at once, and on 20 years of daily 0.25° global data it peaked
at about 80 GB. Size an exact run from a short pilot.

.. _extremes-convention-gap:

The Convention Gap
------------------

The two methods can return different thresholds for the same data. The size of the gap
depends mainly on the number of samples per threshold, and much less on the bin width.

``global_percentile`` pools the whole series per cell, so samples near the threshold lie far
closer together than a bin, and on 20 years of daily data the approximate threshold is
within about one bin of the exact one. The gap widens as the tail thins, up to 7.8 bins at
the 99th percentile on 10 years (about 37 samples above the threshold per cell).

``seasonal_percentile`` pools only the days near each day of year, which is 220 samples for
20 years and an 11-day window. The two methods then take different order statistics of
those samples. The histogram returns, to within 1.5 bins, the sample of rank
``floor(q * n) + 1`` (for whole-number percentiles, usually numpy's ``higher`` rule in the
upper tail and ``lower`` in the lower tail). ``"exact"`` interpolates linearly between two
samples. In a tail of 220 samples those neighbours are several bins apart, so the two
threshold fields can differ by tens of bins in places, whatever the ``precision``. Both are
legitimate conventions for a percentile of a small sample.

The event flag changes much less than the thresholds, because both thresholds usually fall
in the same gap between samples. On 20 years of Gaussian anomalies at the 90th percentile,
``"exact"`` flagged 10.01 % of days and ``"approximate"`` with ``window_spatial=1`` flagged
10.16 %. The approximate excess grows with bin width (10.97 % on heavy-tailed data whose
derived bins were nine times wider), so pass a finer ``precision`` when the derived bin is
coarse for your variable.

.. note::

   On a grid the approximate seasonal path pools 5 × 5 by default and ``"exact"`` uses
   single cells, so by default they compute different statistics. Compare them with
   ``window_spatial=1``.

.. _extremes-bin-range:

Precision and the Data-Derived Range
------------------------------------

``precision`` is the histogram bin width, in the units of your data. Left unset, it gives about 3000 bins over a range that marEx derives
from the anomaly itself, so no hand-tuned range is needed.

1. **The tail's own extreme caps the range.** A threshold is a percentile of the samples, so
   it cannot pass the largest anomaly (``upper``) or the most negative one (``lower``). The
   opposite side may be clipped freely, because those samples still count towards the
   percentile.
2. **A per-cell estimate lowers it when that extreme is an outlier.** marEx takes each
   cell's mean and standard deviation, estimates its threshold as ``mean + z_p * std`` with
   ``z_p`` the normal quantile of the percentile, and sets the range to three times the
   largest estimate if that is below the cap. The factor covers variance concentrated in one
   season and heavy tails.
3. **A threshold that reaches the edge regrows the range.** If the estimate was too low
   somewhere, marEx widens the range to the cap at the same bin width, recomputes the
   thresholds and logs a warning. Thresholds that were inside the old range are unchanged.
   This happens at most once and costs a second threshold pass.

On global 0.25° OSTIA at the 95th percentile, the range is the warmest anomaly, ±21.0 K,
and the bins are 0.014 K wide. Bin width affects the event count more than the thresholds.
On OSTIA from 2003 to 2022, bins of 0.055 K moved the thresholds by at most 0.08 K relative
to 0.01 K bins, but flagged 10 to 13 % more extreme days.

A threshold in the outermost bin triggers a ``UserWarning``, which means the window
containing the most extreme sample has too few samples to resolve the percentile.

marEx warns above 10,000 bins, where thresholding becomes markedly slower, and above
65,000 bins (the limit of the bin index) it raises an error before any work is done. It also warns when the derived bin is
wider than 0.03 standard deviations of the anomaly. The resolved ``precision`` is logged
and recorded in the output attributes. ``precision`` applies to
``method_percentile="approximate"`` only. Deriving the range costs one pass over the
anomaly, which is cheap under ``persist`` and a full walk of the anomaly graph under
``compute_mode="lazy"``.

Choosing Settings for Other Variables
=====================================

The defaults assume a dense, roughly continuous anomaly with a seasonal cycle. For other
variables, check these settings:

1. **Tail.** Heat, heavy rain and high wind use ``upper``. Cold, drought and low wind use
   ``lower``.
2. **Percentile.** 95 and 90 are the usual choices. Rarer percentiles need more samples. Use
   the table above to see how many lie beyond the threshold.
3. **Samples.** If marEx warns that fewer than 10 lie beyond the threshold, raise
   ``window_spatial`` on a grid, widen ``window_days``, or move the percentile toward the
   median (lower it for ``tail="upper"``, raise it for ``tail="lower"``).
4. **Units.** ``precision`` is in the units of the anomaly. Leave it unset first and read
   the derived value from the log or the output attributes. Set it only if that bin width is
   coarse relative to the spread of the variable.
5. **Cadence.** Monthly and sub-daily data work, with window durations rounded to whole
   steps (:ref:`cadence-rules`).

For low wind speed, with daily anomalies on a lat/lon grid:

.. code-block:: python

   import xarray as xr
   import marEx

   wind = xr.open_zarr("wind100m_daily.zarr").speed

   ds = marEx.preprocess_data(
       wind,
       method_anomaly="fixed_baseline",
       reference_period=(1991, 2020),
       method_extreme="seasonal_percentile",
       threshold_percentile=10,
       tail="lower",
       window_days=15,
       window_spatial=5,
   )

Hobday-Style Marine Heatwaves
=============================

A configuration close to the original definition, with a 30-year climatology, an 11-day
threshold window, single-cell thresholds and the 90th percentile:

.. code-block:: python

   ds = marEx.preprocess_data(
       sst,
       method_anomaly="fixed_baseline",
       reference_period=(1983, 2012),
       smooth_days=31,
       method_extreme="seasonal_percentile",
       threshold_percentile=90,
       window_days=11,
       window_spatial=1,
   )

Hobday et al. also require events to last at least five days. Apply that criterion after
tracking, using each event's ``time_start`` and ``time_end``.

Combining Variables
===================

Run each variable separately and combine the boolean fields logically to get compound
events. The runs must share a grid and time axis, so with ``shifting_baseline`` both need
the same ``window_years``.

.. code-block:: python

   sst_ext = marEx.preprocess_data(sst, threshold_percentile=95)
   sal_ext = marEx.preprocess_data(salinity, threshold_percentile=5, tail="lower")

   compound = sst_ext.extreme_events & sal_ext.extreme_events   # warm and fresh

Checking the Output
===================

.. code-block:: python

   freq = ds.extreme_events.mean("time")                     # fraction flagged per cell
   print(float(freq.where(ds.mask).mean()))                  # close to 1 - q (q for the lower tail)?

   spread = ds.thresholds.max("dayofyear") - ds.thresholds.min("dayofyear")

The mean flagged fraction should be close to ``1 - threshold_percentile / 100`` for the
upper tail (``threshold_percentile / 100`` for the lower), with larger departures where
cells have few samples. The tracker takes ``extreme_events`` and ``mask`` as input
(:doc:`tracking`).
