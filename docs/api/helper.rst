============================
Helper (:mod:`marEx.helper`)
============================

.. currentmodule:: marEx.helper

The :mod:`marEx.helper` module provides utilities for HPC environments —
managing Dask clusters on SLURM systems and tuning performance for large
datasets. For deployment guidance and examples, see :doc:`../guide/performance`.

.. autosummary::
   :nosignatures:

   start_distributed_cluster
   start_local_cluster
   configure_dask
   get_cluster_info
   fix_dask_tuple_array

Detailed reference
==================

.. autofunction:: start_distributed_cluster

.. autofunction:: start_local_cluster

.. autofunction:: configure_dask

.. autofunction:: get_cluster_info

.. autofunction:: fix_dask_tuple_array
