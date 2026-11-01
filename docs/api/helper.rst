============================
Helper (:mod:`marEx.helper`)
============================

.. currentmodule:: marEx.helper

The :mod:`marEx.helper` module starts local and SLURM Dask clusters, configures
Dask for large datasets and monitors resources. For deployment guidance and examples, see :doc:`../guide/performance`.

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

Helpers are reached as ``marEx.helper.<name>``. Only :func:`marEx.configure_dask`
is also exported at the top level.
