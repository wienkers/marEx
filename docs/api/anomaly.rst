================================
Anomalies (:mod:`marEx.anomaly`)
================================

.. currentmodule:: marEx.anomaly

Climatology, detrending, and anomaly computation. A complete, standalone stage:
nothing in this module takes a threshold, and its output is a finished product
for anyone who wants a climatology and anomalies and nothing further.

Works on any field with a time dimension -- ocean, atmosphere, land surface, or
biogeochemistry -- on regular grids and unstructured meshes alike.

For method-selection guidance and worked examples, see :doc:`../guide/detection`.

.. autosummary::
   :nosignatures:

   compute
   compute_normalised_anomaly
   rolling_climatology
   smoothed_rolling_climatology

Detailed reference
==================

.. autofunction:: compute

.. autofunction:: compute_normalised_anomaly

.. autofunction:: rolling_climatology

.. autofunction:: smoothed_rolling_climatology
