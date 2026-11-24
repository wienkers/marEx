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
    apply = _OPERATIONS[operation]
    valid = positions >= 0
    with np.errstate(divide="ignore", invalid="ignore") if operation == "divide" else nullcontext():
        if valid.all():
            return apply(source, lookup[positions], dtype=arith_dtype).astype(result_dtype, copy=False)
        out = np.full(source.shape, np.nan if operation in ("subtract", "divide") else False, dtype=result_dtype)
        if valid.any():
            out[valid] = apply(source[valid], lookup[positions[valid]], dtype=arith_dtype)
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

    The lookup replaces the time axis with positive cycle labels. Absent and invalid
    labels keep their positions in time and produce NaN anomalies or false flags.
    Each lazy task reads a whole cycle for one spatial tile and applies only the slots
    used by its time chunk. The result keeps the source's layout and metadata.
    """
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown seasonal operation {operation!r}")
    values = np.asarray(cycle_coord_values)
    if values.shape != (da.sizes[timedim],):
        raise ValueError(f"Expected one seasonal label per {timedim!r} position")
    labels = lookup.get_index(cycledim)
    positions = labels.get_indexer(values)
    matched = positions >= 0
    # 0, negative and missing labels are invalid even if they occur in the lookup.
    matched_labels = labels.to_numpy()[positions[matched]]
    valid_labels = ~pd.isna(matched_labels)
    if valid_labels.any():
        valid_labels[valid_labels.copy()] = np.asarray(matched_labels[valid_labels] > 0, dtype=bool)
    positions[matched] = np.where(valid_labels, positions[matched], -1)

    arith_dtype = np.dtype(bool)
    if operation == "subtract":
        arith_dtype = np.result_type(da.dtype, lookup.dtype)
    elif operation == "divide":
        arith_dtype = np.true_divide(np.empty(0, dtype=da.dtype), np.empty(0, dtype=lookup.dtype)).dtype
    dtype = arith_dtype
    if operation == "subtract" and arith_dtype.kind in "bui" and (positions < 0).any():
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

    coords = {name: coord for name, coord in da.coords.items() if name != cycledim}
    if xr.get_options()["keep_attrs"] is True and operation in ("subtract", "divide"):
        attrs = {k: v for k, v in da.attrs.items() if k not in _CF_CODING_ATTRS}
    else:
        attrs = {}
    out = xr.DataArray(result, dims=source.dims, coords=coords, attrs=attrs).transpose(*da.dims, transpose_coords=False)
    out.name = da.name
    return out
