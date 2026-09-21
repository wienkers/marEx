"""
The dimension contract shared by every marEx analysis package.

A marEx field is ``(time, *extra, *horizontal)``. The *horizontal* dimensions are
the ones a map is drawn on -- ``(y, x)`` for a gridded field, a single cell
dimension for an unstructured mesh. Everything else that is not time is an
*extra* dimension: depth, level, ensemble member, band. The split is load-bearing
rather than cosmetic:

* **chunking and masking apply to all spatial dims** (horizontal plus extra), so
  a depth axis must be tiled and masked like any other spatial axis;
* **the ``window_spatial`` rolling window applies to horizontal dims only** -- a
  depth axis must never be smoothed over.

The helpers here derive both sets from the data itself, so a caller with a
``(time, depth, lat, lon)`` field changes nothing about how they call marEx. They
replace the ``[dim for dim in ["x", "y"] if dim in dimensions]`` comprehensions
that used to be written out at every chunking site, each of which silently
dropped any dimension the ``dimensions`` mapping did not name.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import xarray as xr

from ..exceptions import ConfigurationError
from ..logging_config import get_logger

logger = get_logger(__name__)

# The keys the public ``dimensions`` mapping may carry. ``z`` is optional and, when
# given, is checked against the extra dimensions derived from the data.
HORIZONTAL_KEYS: Tuple[str, ...] = ("y", "x")
COORDINATE_KEYS: Tuple[str, ...] = ("time",) + HORIZONTAL_KEYS

# Target number of array elements one task may touch. Shared by every spatial
# tiling decision so a single number governs the per-task working set.
TASK_ELEMENTS = 50_000_000

# Stable substring of the fit warning, so a caller (or a test) can recognise one
# without matching the whole sentence.
FIT_WARNING_MARKER = "exceeds the per-task element budget"


def horizontal_dims(dimensions: Dict[str, str]) -> Tuple[str, ...]:
    """Names of the horizontal dimensions, in ``(y, x)`` order.

    Gridded data yields two entries, an unstructured mesh one. Derived from the
    ``dimensions`` mapping alone, since that is what defines which axes are
    horizontal.
    """
    return tuple(dimensions[key] for key in HORIZONTAL_KEYS if key in dimensions)


def extra_dims(
    obj: Union[xr.DataArray, xr.Dataset],
    dimensions: Dict[str, str],
    exclude: Iterable[str] = (),
) -> Tuple[str, ...]:
    """Names of ``obj``'s spatial dimensions that the mapping does not name.

    These are the extra dims -- depth, level, member -- carried through every
    reduction as a broadcast axis. ``exclude`` names dimensions that are neither
    spatial nor time (a cycle index such as ``dayofyear``, a histogram bin axis)
    and must not be mistaken for one.
    """
    named = set(dimensions.values()) | set(exclude)
    return tuple(str(d) for d in obj.dims if str(d) not in named)


def spatial_dims(
    obj: Union[xr.DataArray, xr.Dataset],
    dimensions: Dict[str, str],
    exclude: Iterable[str] = (),
) -> Tuple[str, ...]:
    """Every spatial dimension of ``obj``: horizontal first, then extra.

    Only names actually present on ``obj`` are returned, so this is safe to call
    on a reduced array (a threshold field with no time dimension, say).
    """
    present = {str(d) for d in obj.dims}
    horizontal = tuple(d for d in horizontal_dims(dimensions) if d in present)
    return horizontal + extra_dims(obj, dimensions, exclude)


def spatial_chunks(
    obj: Union[xr.DataArray, xr.Dataset],
    dimensions: Dict[str, str],
    size: Union[int, str] = -1,
    exclude: Iterable[str] = (),
) -> Dict[str, Union[int, str]]:
    """Chunk dict setting every spatial dimension of ``obj`` to ``size``."""
    return {dim: size for dim in spatial_dims(obj, dimensions, exclude)}


def check_tile_fit(
    chunks: Dict[str, int],
    tiled_dims: Sequence[str],
    elements_per_cell: int,
    target_elements: int,
    *,
    floor_bound_dims: Sequence[str] = (),
    floor: int = 1,
    stage: str = "Canonical rechunk",
    itemsize: Optional[int] = None,
) -> Optional[str]:
    """Warn when a tile could not be brought under its per-task element budget.

    The design behind the canonical layout was "time-whole spatial
    tiles **plus a fit warning**": warn, with the remedy named, before a rechunk
    that cannot fit crashes the run. The tiles shipped; this is the warning.

    A tile is normally brought under budget by shrinking it, and that is not a
    misfit however small the budget -- the tiler did its job. Only two things make
    the budget unreachable, and only they fire here:

    * **a spatial-window floor.** ``window_spatial`` needs every horizontal chunk
      at least as wide as the window, so a window wider than the tile the budget
      allows overrides the budget. Reachable today through
      :func:`marEx.extremes.histogram._histogram_tile_chunks`, where a sub-daily
      cycle shrinks the cell budget into single figures while the window does not
      move.
    * **a single cell over budget.** When one spatial cell alone reads or writes
      more than ``target_elements``, no tile is small enough.

    Deliberately NOT a trigger: a tile a little above the cell budget because the
    tile side was rounded up. That is a property of the tiler, not of the caller's
    configuration, it is bounded by the rounding, and warning on it would put a
    warning on the ordinary path -- which is worse than no warning at all.

    This is **observation only**. It never changes a chunk: the tile a caller gets
    is the same whether or not this fires, so a warning can never move a result
    (the standing rule that bit-identity is blind to the graph cuts both ways).

    Parameters
    ----------
    chunks
        The tile that was decided, as ``{dim: size}``.
    tiled_dims
        The spatial dimensions of ``chunks`` that make up one task's tile. Any
        held-whole axis in ``chunks`` (time, typically) is excluded by leaving it
        out of this list; its length belongs in ``elements_per_cell``.
    elements_per_cell
        Elements one spatial cell costs -- the larger of the read and written
        sides, as the tiling itself budgets it.
    target_elements
        The per-task element budget the tile was sized against.
    floor_bound_dims
        Dimensions whose chunk was raised to a floor **above** what the budget
        allowed. Empty when no floor bound anything.
    floor
        The floor that bound them, reported so the user can recognise it as their
        own ``window_spatial``.
    stage
        Human-readable name of the rechunk, used to open the message.
    itemsize
        Bytes per element, when known, so the message can carry a size in bytes
        as well as in elements.

    Returns
    -------
    str or None
        The message that was logged, or ``None`` when the tile fits.
    """
    cells = 1
    for dim in tiled_dims:
        cells *= max(1, int(chunks[dim]))
    estimate = cells * max(1, int(elements_per_cell))
    if estimate <= int(target_elements):
        return None

    floor_bound = list(floor_bound_dims)
    single_cell_over = int(elements_per_cell) > int(target_elements)
    if not (floor_bound or single_cell_over):
        return None

    size = ""
    if itemsize:
        size = f", ~{estimate * int(itemsize) / 1e6:.0f} MB at {int(itemsize)} B/element"
    message = (
        f"{stage}: one task would touch {estimate:,} elements{size}, which {FIT_WARNING_MARKER} "
        f"of {int(target_elements):,} ({estimate / max(1, int(target_elements)):.1f}x)."
    )
    if floor_bound:
        widths = ", ".join(f"{d}={int(chunks[d])}" for d in floor_bound)
        message += (
            f" The {int(floor)}-cell spatial window holds {widths}, wider than the tile the budget allows"
            f" -- a rolling window may not cross a chunk boundary, so the floor wins over the budget."
        )
    if single_cell_over:
        message += (
            f" A single spatial cell alone touches {int(elements_per_cell):,} elements, more than the "
            f"whole budget, so no tile is small enough."
        )
    message += (
        " The tile is used as it stands -- this warning changes nothing. If the run then dies on memory,"
        " the levers are: a narrower window_spatial, a shorter window_years, or fewer histogram bins"
        " (each cuts the per-cell element count); fewer threads per worker, so one task gets more of the"
        " worker's memory; or compute_mode='streaming'."
    )
    logger.warning(message)
    return message


def tile_spatial_chunks(
    obj: Union[xr.DataArray, xr.Dataset],
    dims: Sequence[str],
    input_elements_per_cell: int,
    output_elements_per_cell: int = 1,
    target_elements: Optional[int] = None,
    floor_dims: Iterable[str] = (),
    floor: int = 1,
) -> Dict[str, int]:
    """Cap the spatial chunks of ``obj`` so one task's working set stays bounded.

    A reduction that holds one axis whole reads ``input_elements_per_cell`` and
    writes ``output_elements_per_cell`` elements for every spatial cell in the
    tile. Budgeting **both** sides is the point: sizing on the input alone leaves
    the output growing as the input axis shrinks, so a shorter run would silently
    allocate a larger task (the mistake fixed in ``e4fcc89``).

    The tile side is the ``len(dims)``-th root of the cell budget, which makes
    this rank-agnostic: a depth axis of 50 does not multiply the task, it shrinks
    each side until the product is back under budget.

    This is a **cap, never a target**. The returned size for a dimension is the
    smallest of the tile side, the dimension's current largest chunk, and its
    length -- so a caller who has already chunked more finely than the budget
    requires is left exactly as they were, and this can only ever reduce a task's
    working set. That is what makes it a pure rechunk with no memory surprise.

    Parameters
    ----------
    obj
        The array whose spatial dimensions are to be tiled.
    dims
        Spatial dimensions to tile. Dimensions absent from ``obj`` are ignored.
    input_elements_per_cell
        Elements read per spatial cell, e.g. the length of the held-whole axis.
    output_elements_per_cell
        Elements produced per spatial cell, e.g. ``n_years x cycle_length``.
    target_elements
        Per-task element budget. Defaults to :data:`TASK_ELEMENTS`, read at call
        time so the module-level value stays the single knob.
    floor_dims, floor
        Dimensions that must not be tiled below ``floor`` -- the horizontal dims
        under a rolling spatial window, which needs every chunk at least as wide
        as the window. Never applied to extra dims, which are not smoothed over.

    Returns
    -------
    dict
        Chunk sizes, suitable for ``obj.chunk(...)``. Empty when there is nothing
        to tile.
    """
    present = [d for d in dims if d in obj.sizes]
    if not present:
        return {}

    if target_elements is None:
        target_elements = TASK_ELEMENTS

    divisor = max(1, int(input_elements_per_cell), int(output_elements_per_cell))
    cells_per_tile = max(1, int(target_elements) // divisor)

    # Spend the cell budget greedily, shortest dimension first. A flat
    # `budget ** (1/rank)` side would waste most of the budget whenever one
    # dimension is shorter than that side -- a depth axis of 3 against a side of 34
    # leaves the tile 11x smaller than it may be, which is 11x the tasks for nothing.
    # Taking the short dimensions whole first hands their unused share to the long
    # ones, and reduces to exactly `budget ** (1/rank)` when all dims are long.
    # The array's own chunks, not `obj.chunksizes`: that unifies across dask-backed coordinates
    # too, and raises when a coordinate on the time axis is chunked differently from the data
    # (a centred rolling mean splits the data's last time block but not its coordinates).
    if isinstance(obj, xr.DataArray):
        current = dict(zip(obj.dims, obj.chunks)) if obj.chunks else {}
    else:
        current = {d: c for v in obj.data_vars.values() if v.chunks for d, c in zip(v.dims, v.chunks)}
    remaining_cells = cells_per_tile
    chunks: Dict[str, int] = {}
    floor_bound: list = []
    ordered = sorted(present, key=lambda d: int(obj.sizes[d]))
    for position, dim in enumerate(ordered):
        side = max(1, int(round(remaining_cells ** (1.0 / (len(ordered) - position)))))
        budget_side = side
        if dim in floor_dims:
            side = max(side, int(floor))
        existing = current.get(dim)
        existing_max = max(existing) if existing else int(obj.sizes[dim])
        chunks[dim] = max(1, min(side, int(existing_max), int(obj.sizes[dim])))
        # Record only a floor that actually widened this chunk past what the budget
        # allowed: a floor the tile was already above binds nothing.
        if dim in floor_dims and chunks[dim] > budget_side:
            floor_bound.append(dim)
        remaining_cells = max(1, remaining_cells // chunks[dim])

    logger.debug(
        f"Spatial tiling: {cells_per_tile} cells/task over {len(present)} dims "
        f"(input {input_elements_per_cell}, output {output_elements_per_cell} per cell) -> {chunks}"
    )
    check_tile_fit(
        chunks,
        ordered,
        divisor,
        target_elements,
        floor_bound_dims=floor_bound,
        floor=floor,
        stage="Canonical rechunk",
        itemsize=obj.dtype.itemsize if isinstance(obj, xr.DataArray) else None,
    )
    return chunks


def canonical_time_chunks(
    obj: Union[xr.DataArray, xr.Dataset],
    dimensions: Dict[str, str],
    input_elements_per_cell: Optional[int] = None,
    output_elements_per_cell: int = 1,
) -> Dict[str, int]:
    """Chunk dict holding the time axis whole inside bounded spatial tiles.

    Every reduction along time in the anomaly stage (a rolling mean, a flox
    group mean, a harmonic fit) accumulates block by block, so its floating-point
    result depends on where the time chunk boundaries fall: ~1e-4 K on SST, enough
    to move a threshold across a 0.01 bin (D-091). Holding time whole makes the
    answer a property of the data alone. It also keeps the graph small: flox's
    task count scales with tiles x time chunks, and a 27-year daily input at
    30-day chunks never finished building (D-090).

    The spatial side is capped with :func:`tile_spatial_chunks`, so one task's
    working set stays near :data:`TASK_ELEMENTS` whatever the caller passed in.
    Each cell is reduced independently, so the spatial layout never moves values.
    """
    timedim = dimensions["time"]
    if input_elements_per_cell is None:
        input_elements_per_cell = int(obj.sizes[timedim])
    chunks: Dict[str, int] = {timedim: -1}
    chunks.update(
        tile_spatial_chunks(
            obj,
            spatial_dims(obj, dimensions),
            input_elements_per_cell=input_elements_per_cell,
            output_elements_per_cell=output_elements_per_cell,
        )
    )
    return chunks


@dataclass(frozen=True)
class DimSpec:
    """Resolved dimension contract for one field.

    Attributes
    ----------
    time, time_coord
        Time dimension and the coordinate indexing it.
    horizontal
        Horizontal dimensions in ``(y, x)`` order; a single entry for an
        unstructured mesh.
    extra
        Every other non-time dimension, in the field's own order.
    is_gridded
        True when both a ``y`` and an ``x`` dimension are named.
    dimensions, coordinates
        The legacy name mappings, kept for ``track`` and ``plotX`` interop and
        for the functions that still take them directly.
    """

    time: str
    time_coord: str
    horizontal: Tuple[str, ...]
    extra: Tuple[str, ...]
    is_gridded: bool
    dimensions: Dict[str, str]
    coordinates: Dict[str, str]

    @property
    def spatial(self) -> Tuple[str, ...]:
        """Horizontal dimensions followed by extra dimensions."""
        return self.horizontal + self.extra

    def spatial_chunks(self, size: Union[int, str] = -1) -> Dict[str, Union[int, str]]:
        """Chunk dict setting every spatial dimension to ``size``."""
        return {dim: size for dim in self.spatial}

    def horizontal_chunks(self, size: Union[int, str] = -1) -> Dict[str, Union[int, str]]:
        """Chunk dict setting only the horizontal dimensions to ``size``."""
        return {dim: size for dim in self.horizontal}


def resolve_dims(
    da: Union[xr.DataArray, xr.Dataset],
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
) -> DimSpec:
    """Build the :class:`DimSpec` for ``da``.

    ``dimensions`` and ``coordinates`` must already have been through
    :func:`marEx.core.validation._infer_dims_coords`, which applies the defaults
    and checks the named dimensions exist.

    An explicit ``"z"`` entry in ``dimensions`` is optional. When present it is
    checked against the extra dimensions derived from the data, so a typo is
    reported here rather than silently ignored.
    """
    time = dimensions["time"]
    horizontal = tuple(d for d in horizontal_dims(dimensions) if d in da.dims)
    extra = extra_dims(da, dimensions)

    declared = dimensions.get("z")
    if declared is not None:
        declared_tuple = (declared,) if isinstance(declared, str) else tuple(declared)
        if set(declared_tuple) != set(extra):
            raise ConfigurationError(
                "Declared 'z' dimensions do not match the data",
                details=(
                    f"dimensions['z'] names {list(declared_tuple)}, but the extra "
                    f"(non-time, non-horizontal) dimensions of the data are {list(extra)}"
                ),
                suggestions=[
                    "Remove the 'z' entry -- extra dimensions are detected automatically",
                    f"Set dimensions['z'] to {list(extra)}",
                    "Check the horizontal dimension names in 'dimensions'",
                ],
                context={"declared_z": list(declared_tuple), "derived_extra": list(extra)},
            )

    return DimSpec(
        time=time,
        time_coord=coordinates.get("time", time),
        horizontal=horizontal,
        extra=extra,
        is_gridded=("y" in dimensions and "x" in dimensions),
        dimensions=dimensions,
        coordinates=coordinates,
    )
