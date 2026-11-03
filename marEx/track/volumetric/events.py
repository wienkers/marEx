"""
MarEx Track: clustering objects into events for 4-D tracking.

The 3-D clusterer (:func:`marEx.track.merge_split.cluster_rename_objects_and_props`) already
handles a field with a single flat spatial dimension: the unstructured-grid path. A
``(time, z, y, x)`` ID field is reshaped to ``(time, z*y*x)`` (a view of the same chunks,
since depth and the horizontal are single chunks), clustered by that unchanged code, and
the relabelled ID field is reshaped back. No property is relabelled through the clusterer:
the 4-D diagnostics are computed afterwards from the final labels.
"""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ...exceptions import ConfigurationError
from ...logging_config import get_logger
from ..merge_split import cluster_rename_objects_and_props
from .config import VolumetricDims

logger = get_logger(__name__)

# Name of the flattened (z, y, x) dimension handed to the 3-D clusterer. Chosen so it cannot
# collide with a user dimension (the 3-D clusterer itself stacks into a dimension named "z"
# on its structured path, which this adapter never takes).
FLAT_DIM: str = "__marex_voxel__"

# The clusterer prepends a padding row built from ``object_props.isel(ID=0)``, so the
# property set needs one ID label. It carries no variables, so nothing is relabelled
# through it and its area/centroid recomputation is skipped; the label value is never read.
_PLACEHOLDER_IDS: NDArray[np.int32] = np.array([1], dtype=np.int32)


def _check_layout(id_field: xr.DataArray, dims: VolumetricDims) -> None:
    if FLAT_DIM in id_field.dims:
        raise ConfigurationError(f"The ID field already has a dimension named '{FLAT_DIM}'")
    if tuple(id_field.dims) != dims.internal_order():
        raise ConfigurationError(
            f"The ID field must have dimensions {dims.internal_order()}, got {tuple(id_field.dims)}",
            suggestions=["Transpose the field to (time, z, y, x) before clustering"],
        )
    if id_field.chunks is None:
        raise ConfigurationError(
            "The ID field must be dask-backed and chunked in time",
            suggestions=[f"Chunk the field, e.g. id_field.chunk({{'{dims.time}': 30}})"],
        )
    for axis, name in enumerate(dims.internal_order()[1:], start=1):
        if len(id_field.chunks[axis]) != 1:
            raise ConfigurationError(
                f"The ID field must be a single chunk along '{name}', got {len(id_field.chunks[axis])} chunks",
                details="Flattening (z, y, x) without a rechunk needs depth and the horizontal whole",
                suggestions=[f"Rechunk with {{'{name}': -1}}"],
            )


def _time_coords(id_field: xr.DataArray, timedim: str) -> dict:
    """Coordinates of ``id_field`` that depend on time only (incl. a separate time coordinate)."""
    return {name: coord for name, coord in id_field.coords.items() if coord.dims == (timedim,)}


def cluster_events(
    id_field: xr.DataArray,
    links: NDArray[np.int64],
    merges: xr.Dataset,
    dims: VolumetricDims,
    timecoord: str,
    timechunks: int,
    *,
    materialiser,
) -> xr.Dataset:
    """
    Cluster linked objects into events and relabel them ``1..N``.

    Parameters
    ----------
    id_field : xarray.DataArray
        int32 ``(time, z, y, x)`` object IDs, unique across time, dask-backed and chunked in
        time only (the merge loop's output).
    links : numpy.ndarray
        ``(n, 2)`` linked ``(id_t, id_t+1)`` pairs, as from :func:`final_links`.
    merges : xarray.Dataset
        The merge ledger (``MergeLedger.to_dataset()``): ``parent_IDs`` ``(merge_ID,
        parent_idx)`` and ``merge_time``.
    dims : VolumetricDims
        Dimension names.
    timecoord : str
        Name of the time coordinate (may differ from the time dimension).
    timechunks : int
        Time chunk of the ``merge_ledger`` output.
    materialiser : Materialiser or None
        Anchors the relabelled field and the ``(time, ID)`` mapping: staged to disk in
        streaming mode, persisted otherwise.

    Returns
    -------
    xarray.Dataset
        ``ID_field`` int32 ``(time, z, y, x)`` with the input's chunks (0 = background), and
        ``global_ID``, ``presence``, ``time_start``, ``time_end`` and ``merge_ledger`` with
        their 3-D meaning.
    """
    _check_layout(id_field, dims)
    timedim = dims.time
    nt, nz, ny, nx = (id_field.sizes[d] for d in dims.internal_order())

    # (time, z, y, x) -> (time, z*y*x): merges the single-chunk trailing axes, no rechunk.
    flat_data = id_field.data.reshape(nt, nz * ny * nx)
    flat = xr.DataArray(flat_data, dims=(timedim, FLAT_DIM), coords=_time_coords(id_field, timedim), name=id_field.name)

    logger.info("Clustering objects into events")
    events = cluster_rename_objects_and_props(
        flat,
        xr.Dataset(coords={"ID": _PLACEHOLDER_IDS}),
        np.asarray(links, dtype=np.int64).reshape(-1, 2),
        merges,
        True,  # one flat spatial dimension: the unstructured path
        timedim,
        timecoord,
        timechunks,
        None,
        FLAT_DIM,
        None,
        None,
        None,
        False,
        materialiser=materialiser,
    )

    # (time, z*y*x) -> (time, z, y, x): splits the single trailing chunk back, no rechunk.
    data = events["ID_field"].data.reshape(nt, nz, ny, nx)
    id_field_out = xr.DataArray(data, dims=dims.internal_order(), coords=id_field.coords, name="ID_field")

    events = events.drop_vars("ID_field").drop_dims(FLAT_DIM, errors="ignore")
    events["ID_field"] = id_field_out
    return events[["ID_field"] + [v for v in events.data_vars if v != "ID_field"]]
