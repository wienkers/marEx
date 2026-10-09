============================
Helper (:mod:`marEx.helper`)
============================

.. currentmodule:: marEx.helper

The :mod:`marEx.helper` module provides utilities for HPC environments:
managing Dask clusters on SLURM systems and tuning performance for large
datasets. For deployment guidance and examples, see :doc:`../guide/performance`.

.. autosummary::
   :nosignatures:

   start_distributed_cluster
   start_local_cluster
   configure_dask
   get_cluster_info
   ResourceMonitor

Detailed reference
==================

.. autofunction:: start_distributed_cluster

.. autofunction:: start_local_cluster

.. autofunction:: configure_dask

.. autofunction:: get_cluster_info

.. autoclass:: ResourceMonitor
   :members: stage, summary

Helpers are reached as ``marEx.helper.<name>`` and are not exported at the top level
(only :func:`marEx.configure_dask` is).
