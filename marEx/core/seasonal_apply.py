"""Apply a labelled seasonal field to time chunks with bounded spatial tiles."""

from contextlib import nullcontext
from math import prod
from typing import Literal

import dask.array as dsa
import numpy as np
import pandas as pd
import xarray as xr

from .dimensions import TASK_ELEMENTS, tile_spatial_chunks

_OPERATIONS = {"subtract": np.subtract, "divide": np.true_divide, "upper": np.greater_equal, "lower": np.less_equal}
_CF_CODING_ATTRS = ("_FillValue", "missing_value", "scale_factor", "add_offset")


def _apply_block(source, lookup, positions, operation, arith_dtype, result_dtype):
    valid = positions >= 0
    complete = valid.all()
    fill = False if result_dtype.kind == "b" else np.nan
    if not complete and not valid.any():
        return np.full(source.shape, fill, dtype=result_dtype)

    rows = source if complete else source[valid]
    selected = lookup[positions if complete else positions[valid]]
    mask = np.ma.getmask(rows) | np.ma.getmask(selected)
    has_mask = mask.any()
    apply = _OPERATIONS[operation]
    with np.errstate(divide="ignore", invalid="ignore") if operation == "divide" else nullcontext():
        result = apply(np.ma.getdata(rows), np.ma.getdata(selected), dtype=arith_dtype, where=~mask if has_mask else True)
    result = result.astype(result_dtype, copy=False)
    if has_mask:
        result[mask] = fill
    if complete:
        return result
    out = np.full(source.shape, fill, dtype=result_dtype)
    out[valid] = result
    return out


def _cap_chunks(chunks: tuple[int, ...], cap: int) -> tuple[int, ...]:
    """Split oversized chunks at the cap while retaining every existing boundary."""
    tiled = []
    for size in chunks:
        if size <= cap:
            tiled.append(size)
        else:
            full, remainder = divmod(size, cap)
            tiled.extend([cap] * full)
            if remainder:
                tiled.append(remainder)
    return tuple(tiled)


def apply_seasonal_field(
    da: xr.DataArray,
    lookup: xr.DataArray,
    cycle_coord_values: np.ndarray,
    *,
    timedim: str,
    cycledim: str,
    operation: Literal["subtract", "divide", "upper", "lower"],
    task_elements: int = TASK_ELEMENTS,
) -> xr.DataArray:
    """Subtract, divide by, or compare with a seasonal field on the source's spatial grid.

    The lookup replaces the time axis with positive cycle labels. Missing or invalid
    labels and masked values produce NaN anomalies or false flags in place.
    Each lazy task reads a whole cycle for one spatial tile and applies only the slots
    used by its time chunk. The result keeps the source's name, layout and coordinates
    except for the cycle coordinate. With ``keep_attrs=True``, subtraction and division
    retain source attributes except ``_FillValue``, ``missing_value``, ``scale_factor``
    and ``add_offset``. Otherwise the result has no attributes.
    """
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown seasonal operation {operation!r}")
    values = np.asarray(cycle_coord_values)
    if values.shape != (da.sizes[timedim],):
        raise ValueError(f"Expected one seasonal label per {timedim!r} position")
    if cycledim in lookup.coords and cycledim not in lookup.xindexes:
        labels = pd.Index(np.asarray(lookup.coords[cycledim].values))
    else:
        labels = lookup.get_index(cycledim)
    positions = labels.get_indexer(values)
    label_values = labels.to_numpy()
    present = ~pd.isna(label_values)
    valid_labels = np.zeros(len(labels), dtype=bool)
    valid_labels[present] = label_values[present] > 0
    positions[~np.r_[valid_labels, False][positions]] = -1  # The final slot handles unmatched positions (-1).

    arith_dtype = _OPERATIONS[operation](np.empty(0, dtype=da.dtype), np.empty(0, dtype=lookup.dtype)).dtype
    dtype = arith_dtype
    if operation == "subtract" and arith_dtype.kind in "bui":
        # Dask exposes the chunk type at graph construction, before any masks are read.
        masked = any(np.ma.isMaskedArray(getattr(data, "_meta", data)) for data in (da.data, lookup.data))
        if masked or (positions < 0).any():
            dtype = np.result_type(arith_dtype, float)

    spatial = tuple(dim for dim in da.dims if dim != timedim)
    source = da.transpose(timedim, *spatial, transpose_coords=False)
    table = lookup.transpose(cycledim, *spatial, transpose_coords=False)
    source, table = xr.align(source, table, join="exact", copy=False)
    if source.chunks is None and table.chunks is None:
        result = _apply_block(source.data, table.data, positions, operation, arith_dtype, dtype)
    else:
        original = source.chunks or tuple((size,) for size in source.shape)
        time_chunks = _cap_chunks(original[0], max(1, task_elements))
        n_cycle = table.sizes[cycledim]
        tiles = original[1:]
        if prod(map(max, tiles)) * max(1, n_cycle, max(time_chunks)) > task_elements:
            caps = tile_spatial_chunks(source, spatial, n_cycle, max(time_chunks), target_elements=task_elements)
            tiles = tuple(_cap_chunks(chunks, caps[dim]) for dim, chunks in zip(spatial, tiles))

        data = source.data
        input_chunks = (time_chunks, *tiles)
        data = data.rechunk(input_chunks) if source.chunks is not None else dsa.from_array(data, chunks=input_chunks)

        table_data = table.data
        if table.chunks is not None:
            # Split in space before assembling a whole cycle, even if the incoming
            # table has multiple cycle chunks and wider spatial blocks.
            table_data = table_data.rechunk((table.chunks[0], *tiles)).rechunk({0: -1})
        else:
            table_data = dsa.from_array(table_data, chunks=((n_cycle,), *tiles))

        axes = tuple(range(source.ndim))
        result = dsa.blockwise(
            _apply_block,
            axes,
            data,
            axes,
            table_data,
            (source.ndim, *axes[1:]),
            dsa.from_array(positions, chunks=(time_chunks,)),
            (0,),
            operation=operation,
            arith_dtype=arith_dtype,
            result_dtype=dtype,
            dtype=dtype,
            meta=np.empty((0,) * source.ndim, dtype=dtype),
            # The cycle is contracted as one whole chunk and spatial chunks are
            # already aligned. Pass the table as an array, not a list of blocks.
            align_arrays=False,
            concatenate=True,
        ).rechunk(original)

    coords = da.coords
    if cycledim in coords:
        coords = coords.to_dataset().drop_vars(cycledim).coords
    if xr.get_options()["keep_attrs"] is True and operation in ("subtract", "divide"):
        attrs = {k: v for k, v in da.attrs.items() if k not in _CF_CODING_ATTRS}
    else:
        attrs = {}
    out = xr.DataArray(result, dims=source.dims, coords=coords, attrs=attrs).transpose(*da.dims, transpose_coords=False)
    out.name = da.name
    return out
