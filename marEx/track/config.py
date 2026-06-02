"""
Configuration container for the MarEx tracker.

This module defines :class:`TrackerConfig`, a frozen dataclass that bundles the
validated, resolved scalar configuration of a :class:`marEx.track.tracker.tracker`
instance into a single immutable object. It is an internal threading vehicle that
groups the many small configuration values (fill radii, filtering thresholds,
merging options, dimension/coordinate names, grid flags, etc.) so they can be
passed and inspected together rather than as a dozen loose scalars.

The config is intentionally *scalar/name only*: heavyweight data and grid-state
arrays (``data_bin``, ``mask``, ``lat``, ``lon``, ``cell_area`` and the
unstructured-grid helpers) remain ordinary instance attributes on the tracker so
that their lifecycle (reassignment during validation, ``del`` to free memory) is
unchanged. ``TrackerConfig`` is not part of the public API and is not re-exported
from :mod:`marEx.track`.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class TrackerConfig:
    """Immutable snapshot of a tracker's resolved scalar configuration.

    All fields mirror the corresponding attributes set on the ``tracker`` instance
    after input validation and parameter resolution have run. Values are stored
    *after* any transformation performed in ``tracker.__init__`` (e.g. ``R_fill``
    coerced to ``int``, ``coordinate_units`` auto-detected, ``ydim`` set to ``None``
    for unstructured grids, area-filter parameters resolved).

    Parameters
    ----------
    R_fill : int
        Spatial hole-filling radius (in cells).
    T_fill : int
        Temporal gap-filling window (in time steps).
    area_filter_quartile : float or None
        Quartile threshold used for relative size filtering.
    area_filter_absolute : int or None
        Absolute area threshold used for absolute size filtering.
    use_absolute_filtering : bool
        Whether absolute (rather than quartile) area filtering is active.
    allow_merging : bool
        Whether merging/splitting of objects is tracked.
    nn_partitioning : bool
        Whether nearest-neighbour partitioning is used for merge resolution.
    overlap_threshold : float
        Minimum fractional overlap required to link objects across time.
    unstructured_grid : bool
        Whether the data lives on an unstructured grid.
    timedim, xdim, ydim : str or None
        Dimension names for time / x / y (``ydim`` is ``None`` for unstructured grids).
    timecoord, xcoord, ycoord : str
        Coordinate names for time / x / y.
    regional_mode : bool
        Whether regional (non-global, non-periodic) tracking is active.
    coordinate_units : {"degrees", "radians"} or None
        Units of the input coordinates (auto-detected when not supplied).
    max_iteration : int
        Maximum number of iterations for the parallel unstructured merge solver.
    checkpoint : {"save", "load", "None"} or None
        Checkpoint strategy for preprocessing.
    debug : int
        Debug/verbosity level controlling warning suppression.
    """

    R_fill: int
    T_fill: int
    area_filter_quartile: Optional[float]
    area_filter_absolute: Optional[int]
    use_absolute_filtering: bool
    allow_merging: bool
    nn_partitioning: bool
    overlap_threshold: float
    unstructured_grid: bool
    timedim: str
    xdim: str
    ydim: Optional[str]
    timecoord: str
    xcoord: str
    ycoord: str
    regional_mode: bool
    coordinate_units: Optional[Literal["degrees", "radians"]]
    max_iteration: int
    checkpoint: Optional[Literal["save", "load", "None"]]
    debug: int
