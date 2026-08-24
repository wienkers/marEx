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
from typing import Dict, Optional

import dask
import xarray as xr

from ..logging_config import get_logger, log_dask_info, log_memory_usage, log_timing
from .attrs import make_netcdf_safe_attrs
from .dimensions import spatial_chunks

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
        Requested output chunking. Only the time entry is honoured; spatial
        dimensions are always made whole, which is what the tracker requires.
    materialiser
        The materialisation policy. Only ``persist`` mode materialises here.
    staging_dir
        Staging directory to record on ``ds.attrs["marex_staging_dir"]`` so that
        :func:`marEx.clear_staging` can find it later.

    Returns
    -------
    xr.Dataset
        The finalised dataset, saveable to both Zarr and NetCDF.
    """
    # Record the staging directory so `marEx.clear_staging(ds)` can find it. In streaming
    # mode the returned Dataset reads lazily from this directory, so it deliberately
    # outlives this call; the caller clears it after writing their output.
    if staging_dir is not None:
        ds.attrs["marex_staging_dir"] = str(staging_dir)

    # Final rechunking. Fall back to the documented default time chunk (25), not 10,
    # so a partial dask_chunks dict does not silently get 10-step chunks.
    time_chunks = dask_chunks.get(dimensions["time"], dask_chunks.get("time", 25))
    logger.debug(f"Final rechunking with time chunks: {time_chunks}")
    # Every spatial dimension is made whole, extra dims (depth, level) included: the
    # tracker requires it, and a consumer of a 3D+time anomaly wants the same layout.
    # CYCLE_DIMS are excluded here because they are handled just below.
    chunk_dict = dict(spatial_chunks(ds, dimensions, -1, exclude=CYCLE_DIMS))
    chunk_dict[dimensions["time"]] = time_chunks
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
