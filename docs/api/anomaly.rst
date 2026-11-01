================================
Anomalies (:mod:`marEx.anomaly`)
================================

.. currentmodule:: marEx.anomaly

Climatology, detrending and anomaly computation. The module runs on its own, and
its climatologies and anomalies are a complete result. It works on any field with a time dimension (ocean, atmosphere, land surface or
biogeochemistry) on regular grids and unstructured meshes alike.

The within-year axis the climatology is resolved on is inferred from the time
coordinate's cadence, and every entry point accepts a ``cycle=`` override. See
:doc:`../guide/anomalies` for method-selection guidance, worked examples, and the
time-resolution table.

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
