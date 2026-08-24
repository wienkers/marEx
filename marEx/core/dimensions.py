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


def tile_spatial_chunks(
    obj: Union[xr.DataArray, xr.Dataset],
    dims: Sequence[str],
    input_elements_per_cell: int,
    output_elements_per_cell: int = 1,
    target_elements: int = TASK_ELEMENTS,
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
        Per-task element budget.
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

    divisor = max(1, int(input_elements_per_cell), int(output_elements_per_cell))
    cells_per_tile = max(1, int(target_elements) // divisor)
    side = max(1, int(round(cells_per_tile ** (1.0 / len(present)))))

    current = obj.chunksizes
    chunks: Dict[str, int] = {}
    for dim in present:
        limit = side
        if dim in floor_dims:
            limit = max(limit, int(floor))
        existing = current.get(dim)
        existing_max = max(existing) if existing else int(obj.sizes[dim])
        chunks[dim] = max(1, min(int(limit), int(existing_max), int(obj.sizes[dim])))

    logger.debug(
        f"Spatial tiling: {cells_per_tile} cells/task over {len(present)} dims "
        f"(input {input_elements_per_cell}, output {output_elements_per_cell} per cell) -> {chunks}"
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
