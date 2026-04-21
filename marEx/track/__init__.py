"""
MarEx Track: Marine extreme event identification, tracking, and splitting/merging.

This package identifies and tracks extreme events in oceanographic data across
time, supporting both structured (regular grid) and unstructured datasets. It is
the modular successor to the single-file ``marEx.track`` module: the tracker
orchestrator class lives in :mod:`marEx.track.tracker`, the regional factory in
:mod:`marEx.track.regional`, and the JIT-compiled helper primitives in
:mod:`marEx.track.partitioning` and :mod:`marEx.track.overlap`.

Every public name previously importable as ``marEx.track.<name>`` is re-exported
here so existing import paths (and the test suite) keep working unchanged.
"""

from .overlap import sparse_bool_power
from .partitioning import (
    create_grid_index_arrays,
    partition_centroid_unstructured,
    partition_nn_grid,
    partition_nn_unstructured,
    partition_nn_unstructured_optimised,
    wrapped_euclidian_distance_mask_parallel,
    wrapped_euclidian_distance_points,
)
from .regional import regional_tracker
from .tracker import tracker

__all__ = [
    "tracker",
    "regional_tracker",
    "wrapped_euclidian_distance_mask_parallel",
    "wrapped_euclidian_distance_points",
    "create_grid_index_arrays",
    "sparse_bool_power",
    "partition_nn_grid",
    "partition_nn_unstructured",
    "partition_nn_unstructured_optimised",
    "partition_centroid_unstructured",
]
