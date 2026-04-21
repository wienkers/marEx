"""
HPC Dask Helper: Utilities for High-Performance Computing with Dask
--------------------------------------------------------------------

This package provides utilities for setting up and managing Dask clusters
in HPC environments, with specific support for the DKRZ Levante Supercomputer.

The implementation is split across submodules:

- :mod:`marEx.helper.dask_config` — Dask configuration (:func:`configure_dask`)
- :mod:`marEx.helper.cluster` — cluster management
  (:func:`start_local_cluster`, :func:`start_distributed_cluster`,
  :func:`get_cluster_info`)
- :mod:`marEx.helper.checkpoint` — checkpointing and Dask array utilities
  (:func:`checkpoint_to_zarr`, :func:`fix_dask_tuple_array`)

All public functions are re-exported here so that ``marEx.helper`` continues
to behave exactly as the previous single-module implementation did.
"""

from .checkpoint import checkpoint_to_zarr, fix_dask_tuple_array
from .cluster import get_cluster_info, start_distributed_cluster, start_local_cluster
from .dask_config import configure_dask

__all__ = [
    "configure_dask",
    "start_local_cluster",
    "start_distributed_cluster",
    "get_cluster_info",
    "checkpoint_to_zarr",
    "fix_dask_tuple_array",
]
