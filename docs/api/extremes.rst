=====================================
Extremes (:mod:`marEx.extremes`)
=====================================

.. currentmodule:: marEx.extremes

Percentile thresholding and binary extreme-event identification. Takes anomalies
-- from :mod:`marEx.anomaly` or from anywhere else -- and returns a boolean event
field plus the thresholds that defined it.

For method-selection guidance and worked examples, see
:doc:`../guide/detection`.

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
