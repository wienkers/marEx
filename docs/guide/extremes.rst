========
Extremes
========

The extremes stage turns anomalies into a threshold field and a boolean event flag. It is
available alone as :func:`marEx.extremes.identify` and as the second half of
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

``data`` may be a DataArray of anomalies or a Dataset carrying ``dat_anomaly``. A Dataset's
variables are carried through to the output. Nothing in the stage assumes the anomalies came
from marEx, or that the variable is a temperature.

Threshold Methods
=================

``seasonal_percentile`` (default)
  One threshold per cell and per position in the seasonal cycle (``dayofyear``, ``month``
  or ``hourofyear``). The samples for each slot are drawn from all years, from the
  ``window_days`` around the slot, and on a grid from a ``window_spatial`` neighbourhood of
  cells. This is the day-of-year definition of Hobday et al. (2016). Use it when the spread
  of the anomaly changes through the year, which is the usual case.

``global_percentile``
  One threshold per cell from the whole series. Use it when the anomaly has no useful
  seasonal structure left (for instance after standardising), when the series is too
  short to populate a seasonal window, or for a fast first pass. It needs no seasonal
  cycle of its own, so the stage works on any time axis, including irregular ones (the
  anomaly stage may still need a ``cycle``).

``thresholds`` has a cycle dimension for ``seasonal_percentile`` and none for
``global_percentile``. Select by dimension name, never by position, because the order of
dimensions differs between the exact and approximate paths.

.. _extremes-tail:

Which Tail
==========

``tail="upper"`` (default) flags ``anomaly >= threshold``. ``tail="lower"`` flags
``anomaly <= threshold``. ``threshold_percentile`` always names a percentile of the
distribution, never of the tail, so the coldest 5 % is ``threshold_percentile=5`` with
``tail="lower"``.

.. code-block:: python

   cold = marEx.extremes.identify(anomalies, threshold_percentile=5, tail="lower")

The lower tail is computed directly and not by negating the data, so both tails are resolved
at the same bin width. Use it for cold spells, drought, low wind and any other low-side
extreme. ``tail="both"`` is not supported. Run the two tails separately if you need both.

A cell whose anomaly is exactly zero throughout (permanent sea ice, a permanently masked
cell) is never flagged on either side.

Seasonal Window and Spatial Pooling
===================================

``window_days`` (default 11) is a duration in days. On a daily axis it must be odd, because
the window is symmetric about the day. On other cadences it is converted to whole timesteps
(:ref:`cadence-rules`).

``window_spatial`` pools the samples of neighbouring cells into each cell's threshold. It is
a way to get more samples for a percentile, and the field itself is not smoothed. The rule
that decides its value is:

* ``None`` (default) resolves to ``5``, a 5 × 5 neighbourhood, for
  ``seasonal_percentile`` with ``method_percentile="approximate"`` on a gridded field.
  Everywhere else it resolves to no pooling.
* An odd integer sets the neighbourhood explicitly. ``window_spatial=1`` gives single-cell
  thresholds, as in Hobday et al. (2016).
* Passing any value raises a ``ConfigurationError`` on unstructured meshes, with
  ``global_percentile`` and with ``method_percentile="exact"``.

The value that was used is recorded in the output attribute ``window_spatial``.

The number of samples behind each threshold is ``n_years × window_days × window_spatial²``,
and the number beyond the threshold is that times ``1 - q``. marEx logs a warning when fewer
than 50 samples lie beyond the threshold.

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

Short records and high percentiles both call for pooling. For a 99th percentile on 30 years,
a 7 × 7 window gives about 160 samples beyond the threshold. Pooling assumes neighbouring
cells share a distribution, the same assumption Hobday et al. make in time with an 11-day
window. Where that fails, such as across a sharp coastline or front, prefer a smaller window
and a longer record.

Exact and Approximate Percentiles
=================================

``method_percentile="approximate"`` (default) counts samples into histogram bins and
interpolates inside the bin where the cumulative count crosses the percentile. It processes
each cell's time series in pieces, so its memory use is bounded and it has run on 40 years of
daily 0.25° global data (a single run, :doc:`performance`). ``"exact"`` sorts every sample a
cell contributes and applies ``np.nanpercentile`` with numpy's default linear rule. It needs
each tile's full series in memory at once, cannot pool spatially, and was measured once on 20
years of daily 0.25° global data at a cgroup peak of about 80 GB across 16 workers of 6 GB.
Size an exact run from a short pilot.

.. _extremes-convention-gap:

The Convention Gap
------------------

The two methods can return different thresholds for the same data. The cause is the number of
samples per threshold, much more than the bin width.

``global_percentile`` pools the whole series per cell, so neighbouring samples near the
threshold lie far closer together than a bin and the approximate threshold tracks the exact
one to about one bin. On 20 years of daily data the largest gap across four distributions and
five percentiles was 1.09 bins. The agreement weakens as the tail thins. At the 99th
percentile on 10 years (about 37 samples above the threshold per cell) the gap reached up to
7.8 bins.

``seasonal_percentile`` pools only the days near each day of year, which is 220 samples for 20
years and an 11-day window. The two methods then take different order statistics of those
samples. The histogram returns, to within 1.5 bins, the sample of rank ``floor(q * n) + 1``
(for whole-number percentiles, usually numpy's ``higher`` rule in the upper tail and ``lower``
in the lower tail). ``"exact"`` interpolates linearly between two samples. In a tail of 220
samples those neighbours are several bins apart, so the two threshold fields can differ by
tens of bins in places, and a finer ``precision`` does not close the gap. Both are legitimate
conventions for a percentile of a small sample.

The event flag moves much less than the threshold values suggest, because both thresholds
usually fall in the same gap between samples. On 20 years of Gaussian anomalies at the 90th
percentile, ``"exact"`` flagged 10.01 % of days and ``"approximate"`` with
``window_spatial=1`` flagged 10.16 %. The approximate excess grows with bin width. On
heavy-tailed data the derived bins were nine times wider and the approximate method flagged
10.97 %, so pass a finer ``precision`` when the derived bin is coarse for your variable.

.. note::

   On a grid the approximate seasonal path pools 5 × 5 by default and ``"exact"`` cannot
   pool, so out of the box they compute different statistics. Compare them with
   ``window_spatial=1``.

.. _extremes-bin-range:

Precision and the Data-Derived Range
------------------------------------

``precision`` is the histogram bin width, in the units of your data. It is the only
histogram parameter. Omitted, it gives about 3000 bins over a range that marEx derives from
the anomaly itself, so no variable or unit needs a hand-tuned range.

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
   thresholds and logs a warning. Thresholds that were inside the old range come out
   identical. This happens at most once and costs a second threshold pass.

On global 0.25° OSTIA the range at the 95th percentile is the warmest anomaly, ±21.0 K, and
the bins are 0.014 K wide. Bin width matters more than its effect on thresholds suggests. On
OSTIA from 2003 to 2022, bins of 0.055 K (what a coarser derivation gave once one 27 K
anomaly set the range) moved the thresholds by at most 0.08 K relative to 0.01 K bins, yet
flagged 10 to 13 % more extreme days.

When the range is derived, a threshold in the outermost bin is a ``UserWarning``, meaning the
window holding the most extreme sample has too few samples to resolve the percentile. When a
caller pins the range, the same event raises a ``ConfigurationError``, because samples beyond
a pinned edge were clipped into the end bin.

Beyond 10,000 bins marEx warns that the threshold stage gets markedly slower, and beyond
65,000 (the limit of the bin index) it raises before any work is done. A warning is also
logged when the derived bin is wider than 0.03 standard deviations of the anomaly. The
resolved ``precision`` is logged and recorded in the output attributes. ``precision`` is
rejected with ``method_percentile="exact"``, which builds no histogram. The derivation costs
one pass over the anomaly, cheap under ``persist`` but a full walk of the anomaly graph
under ``compute_mode="lazy"``.

Choosing Settings for Other Variables
=====================================

Nothing in the stage depends on the variable, but the defaults assume a dense, roughly
continuous anomaly with a seasonal cycle. Work through these in order.

1. **Tail.** Heat, heavy rain and high wind use ``upper``. Cold, drought and low wind use
   ``lower``.
2. **Percentile.** 95 and 90 are the usual choices. Rarer percentiles need more samples. Use
   the table above to see how many lie beyond the threshold.
3. **Samples.** If fewer than about 50 lie beyond the threshold, raise ``window_spatial`` on
   a grid, widen ``window_days``, or lower the percentile.
4. **Units.** ``precision`` is in the units of the anomaly. Leave it unset first and read the
   derived value from the log or the output attributes. Set it only if that bin width is
   coarse relative to the spread of the variable.
5. **Cadence.** Monthly and sub-daily data work, with window durations rounded to whole
   steps (:ref:`cadence-rules`).

A low-wind-speed example, with daily anomalies on a lat/lon grid:

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

Hobday et al. also require events to last at least five days. That duration criterion is not
applied at this stage. Tracked events carry ``time_start`` and ``time_end``, so it can be
applied after tracking.

Combining Variables
===================

Each variable gets its own run, and compound events are logical combinations of the boolean
fields. The two runs must share the same grid and time axis, so a shifting baseline that
trims one variable trims the other identically only if both use the same ``window_years``.

.. code-block:: python

   sst_ext = marEx.preprocess_data(sst, threshold_percentile=95)
   sal_ext = marEx.preprocess_data(salinity, threshold_percentile=5, tail="lower")

   compound = sst_ext.extreme_events & sal_ext.extreme_events   # warm and fresh

Checks Worth Making
===================

.. code-block:: python

   freq = ds.extreme_events.mean("time")                     # fraction flagged per cell
   print(float(freq.where(ds.mask).mean()))                  # close to 1 - q / 100?

   spread = ds.thresholds.max("dayofyear") - ds.thresholds.min("dayofyear")

The mean flagged fraction should sit near ``1 - threshold_percentile / 100`` for the upper
tail (``threshold_percentile / 100`` for the lower), with larger departures where cells have few samples. The next stage, :doc:`tracking`,
takes ``extreme_events`` and ``mask``.
