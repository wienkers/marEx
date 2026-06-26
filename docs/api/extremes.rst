=====================================
Extremes (:mod:`marEx.extremes`)
=====================================

.. currentmodule:: marEx.extremes

Percentile thresholding and binary extreme-event identification. Takes anomalies
(from :mod:`marEx.anomaly` or from anywhere else) and returns a boolean event
field plus the thresholds that defined it.

``seasonal_percentile`` resolves its thresholds on a within-year cycle inferred from
the time coordinate (``dayofyear``, ``month`` or ``hourofyear``), overridable via
``cycle=``; ``global_percentile`` uses no cycle at all.

``tail='lower'`` flags the low side of the distribution (cold spells, drought) instead
of the high side, and the histogram bins are symmetric about zero so both tails resolve
at the same precision. The bin range is always derived from the data (the requested
tail's extreme, lowered by a per-cell estimate of the largest threshold and regrown if a
threshold reaches it), which is what lets the defaults work on a variable that is not an
SST anomaly in kelvin; ``precision`` alone sets the bin width.

For method-selection guidance, worked examples, and the tail, bin-geometry and
time-resolution tables, see :doc:`../guide/extremes`.

.. autosummary::
   :nosignatures:

   identify
   identify_extremes

Detailed reference
==================

.. autofunction:: identify

.. autofunction:: identify_extremes

Full chain
==========

.. currentmodule:: marEx

:func:`preprocess_data` runs both stages back to back.

.. autofunction:: preprocess_data

Staging cleanup
===============

With ``compute_mode="streaming"`` the returned dataset reads lazily from a staging
directory on disk. Write your output first, then call :func:`clear_staging`. The path
is on ``ds.encoding["marex_staging_dir"]``. See :doc:`../guide/performance`.

.. autofunction:: clear_staging
