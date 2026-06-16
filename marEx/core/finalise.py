"""
Shared output finalisation.

Every marEx entry point that returns a Dataset ends the same way: rechunk to the
caller's requested layout, clear stale encoding, materialise according to the
compute mode, and coerce attributes into a form both Zarr and NetCDF accept.
That tail is identical whether the caller asked for anomalies alone, extremes
alone, or the full chain, so it lives here rather than in any one of them.
"""

import logging
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import dask
import xarray as xr

from ..logging_config import get_logger, log_dask_info, log_memory_usage, log_timing
from .attrs import make_netcdf_safe_attrs
from .dimensions import TASK_ELEMENTS, extra_dim_chunks, horizontal_dims

# Get module logger
logger = get_logger(__name__)

# Cycle-index dimensions produced by the day-of-year style reductions. These are
# rechunked alongside time because a threshold field indexed by them is consumed
# in the same access pattern as the data it thresholds.
CYCLE_DIMS = ("dayofyear", "month", "hourofyear")


@contextmanager
def split_large_chunks():
    """
    Enable Dask's large-chunk splitting for the duration of a pipeline stage.

    Captures the caller's value and restores it on exit rather than leaking
    ``split_large_chunks=True`` into their global Dask config. Nesting is safe:
    an inner use restores the value the outer use had already set.
    """
    previous = dask.config.get("array.slicing.split_large_chunks", None)
    dask.config.set({"array.slicing.split_large_chunks": True})
    try:
        yield
    finally:
        dask.config.set({"array.slicing.split_large_chunks": previous})


def finalise_dataset(
    ds: xr.Dataset,
    dimensions: Dict[str, str],
    coordinates: Dict[str, str],
    dask_chunks: Dict[str, int],
    materialiser,
    staging_dir: Optional[object] = None,
    extra_dims: Tuple[str, ...] = (),
) -> xr.Dataset:
    """
    Apply the common output tail to a finished dataset.

    Parameters
    ----------
    ds
        The dataset to finalise.
    dimensions, coordinates
        Resolved dimension and coordinate name mappings.
    dask_chunks
        Requested output chunking. Only the time entry is honoured; horizontal
        dimensions are always made whole, which is what the tracker requires, and
        extra dimensions are chunked per level (stacked only on a small grid).
        An integer time entry is a step count, and with one, extra dimensions on
        time-indexed variables are sized so a chunk stays within 50 million
        elements where one level allows it. ``"auto"`` hands the time chunk to
        dask, whose budget is in bytes (``array.chunk-size``, 128 MiB by
        default), not elements: a 1-byte boolean variable gets more elements per
        chunk than a float32 one, and can exceed 50 million. A seasonal
        threshold's cycle axis (``dayofyear``, ``month``, ``hourofyear``) is
        chunked from the same entry, capped at the cycle length, and is not held
        to the element budget.
    materialiser
        The materialisation policy. Only ``persist`` mode materialises here.
    staging_dir
        Staging directory to record on ``ds.encoding["marex_staging_dir"]`` so that
        :func:`marEx.clear_staging` can find it later. Kept off ``ds.attrs`` deliberately:
        attrs are copied verbatim into whatever the caller writes the dataset to, and the
        staging directory is deleted by ``clear_staging`` immediately after that write, so
        an attrs-recorded path would be a dead reference baked into the caller's output.
        ``encoding`` travels with the in-memory object but is never serialised by
        ``to_zarr``/``to_netcdf``.
    extra_dims
        The field's extra (non-time, non-horizontal) dimensions -- depth, level,
        member -- resolved from the *input* by :func:`marEx.core.resolve_dims`.
        They take the budget one whole-horizontal time block leaves (one level per
        chunk on a large grid; see :func:`~marEx.core.dimensions.extra_dim_chunks`).

        Passed in rather than derived from ``ds``, because ``ds.dims`` is the union
        over every variable: on the unstructured path it also carries ``neighbours``'
        own ``nv`` axis, which is not a spatial dimension of the field and must not
        be rechunked here. Empty for a 2-D field, which is what makes this site
        produce exactly the chunk dict it always did.

    Returns
    -------
    xr.Dataset
        The finalised dataset, saveable to both Zarr and NetCDF.
    """
    # Record the staging directory so `marEx.clear_staging(ds)` can find it. In streaming
    # mode the returned Dataset reads lazily from this directory, so it deliberately
    # outlives this call; the caller clears it after writing their output. Stashed in
    # `encoding`, not `attrs`: `attrs` is copied into the caller's written output, and by
    # the time that write happens the staging directory is about to be deleted.
    if staging_dir is not None:
        ds.encoding["marex_staging_dir"] = str(staging_dir)

    # Final rechunking. Fall back to the documented default time chunk (25), not 10,
    # so a partial dask_chunks dict does not silently get 10-step chunks.
    time_chunks = dask_chunks.get(dimensions["time"], dask_chunks.get("time", 25))
    logger.debug(f"Final rechunking with time chunks: {time_chunks}")
    # The horizontal dims are made whole (the tracker requires it). Extra dims (depth,
    # level) take whatever of the per-task budget one whole-horizontal time block leaves:
    # one level per chunk on a large grid, several on a small one (D-128). Holding depth
    # whole made one chunk 30 x depth x horizontal, 6.2 GB at depth 50 on 720x1440; one
    # level per chunk everywhere made a 1-cell mooring 25-element chunks. The tracker
    # rejects extra dims, so it never sees this layout; select a level first.
    horizontal = [dim for dim in horizontal_dims(dimensions) if dim in ds.dims]
    chunk_dict = {dim: -1 for dim in horizontal}
    chunk_dict[dimensions["time"]] = time_chunks
    if extra_dims:
        # Size the extra dims from the time chunk dask actually chose: `time_chunks` may be
        # "auto", None, -1 or a tuple, none of which is a step count until it is applied.
        timed = ds.chunk(chunk_dict)
        steps = [
            max(v.chunksizes[dimensions["time"]])
            for v in timed.data_vars.values()
            if dimensions["time"] in v.dims and v.chunks is not None
        ]
        horizontal_cells = 1
        for dim in horizontal:
            horizontal_cells *= int(ds.sizes[dim])
        chunk_dict.update(
            extra_dim_chunks(
                ds.sizes,
                extra_dims,
                horizontal_tile_cells=horizontal_cells,
                horizontal_cells=horizontal_cells,
                budget_cells=TASK_ELEMENTS // max([1, *steps]),
            )
        )
    # A cycle-index dimension is only present when a seasonal threshold was computed,
    # so testing for it is equivalent to testing the extreme method -- and it keeps
    # this function ignorant of which method ran.
    for cycle_dim in CYCLE_DIMS:
        if cycle_dim in ds.dims:
            chunk_dict[cycle_dim] = time_chunks
    ds = ds.chunk(chunk_dict)

    # Clear encoding metadata that may conflict with actual Dask chunks
    # (stale ``chunks`` encoding can otherwise trigger chunk-misalignment errors on save)
    logger.debug("Clearing encoding metadata for Dask-backed variables")
    for var in ds.data_vars:
        if hasattr(ds[var].data, "chunks"):  # Only for Dask-backed variables
            if hasattr(ds[var], "encoding") and "chunks" in ds[var].encoding:
                del ds[var].encoding["chunks"]

    # Fix encoding issue with saving when calendar & units attribute is present
    if "calendar" in ds[coordinates["time"]].attrs:  # pragma: no cover
        logger.debug("Removing calendar attribute for Zarr compatibility")
        del ds[coordinates["time"]].attrs["calendar"]
    if "units" in ds[coordinates["time"]].attrs:  # pragma: no cover
        logger.debug("Removing units attribute for Zarr compatibility")
        del ds[coordinates["time"]].attrs["units"]

    logger.info("Persisting final dataset and optimising task graph")
    with log_timing(
        logger,
        "Dataset persistence and optimisation",
        log_memory=True,
        show_progress=True,
    ):
        if materialiser.mode == "persist":
            ds = ds.persist(optimize_graph=True)
        else:
            logger.info(f"Skipping final dataset persistence (compute_mode='{materialiser.mode}')")

        log_memory_usage(logger, "After dataset persistence", logging.DEBUG)

    logger.debug(f"Final dataset shape: {ds.dims}")
    log_dask_info(logger, ds, "Final dataset")

    # Ensure the returned dataset is directly saveable to *both* Zarr and NetCDF.
    # Booleans/None in attrs round-trip through Zarr but break Dataset.to_netcdf.
    return make_netcdf_safe_attrs(ds)
