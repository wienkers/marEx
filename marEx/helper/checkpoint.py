"""
Checkpointing and Dask array utilities.
--------------------------------------------------------------------

Provides :func:`checkpoint_to_zarr` for breaking Dask graph dependencies
by round-tripping data through a temporary Zarr store, and
:func:`fix_dask_tuple_array` for materialising Dask arrays that carry
tuple (task) references in their chunks before saving to Zarr.
"""

import logging
import shutil
import uuid
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Union

import dask
import dask.array as dask_array
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..logging_config import get_logger, log_timing

# Get module logger
logger = get_logger(__name__)


def checkpoint_to_zarr(
    data: Union[xr.DataArray, xr.Dataset],
    name: str = "checkpoint",
    cleanup: bool = False,
    timedim: str = "time",
) -> Union[xr.DataArray, xr.Dataset]:  # pragma: no cover
    """
    Save and reload a Dask-backed xarray object to break graph dependencies.

    This function materialises a Dask array/dataset to a temporary file
    and immediately reloads it, thereby breaking the computational graph.
    This prevents expensive recomputations when the same data is used multiple
    times downstream.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Dask-backed xarray object to checkpoint
    name : str, default='checkpoint'
        Name prefix for the temporary file (for logging/debugging)
    cleanup : bool, default=False
        Whether to delete the temporary file after reloading.
        By default (False), temp files are kept for the session and cleaned up
        by the OS temp directory manager. Set to True to immediately delete after reload.
    timedim : str, default='time'
        Name of the time dimension for chunking adjustments

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Reloaded data with broken graph dependencies

    Examples
    --------
    >>> import marEx
    >>> anomalies = marEx.compute_normalised_anomaly(sst)
    >>> anomalies_checkpointed = marEx.helper.checkpoint_to_zarr(
    ...     anomalies, name="anomalies"
    ... )
    """
    logger.debug(f"Checkpointing '{name}' to break graph dependencies")

    # Get dask temporary directory, fallback to system temp
    try:
        temp_base = dask.config.get("temporary-directory", None)
        if temp_base is None or not Path(temp_base).exists():
            temp_base = gettempdir()
    except Exception:
        temp_base = gettempdir()

    unique_id = uuid.uuid4().hex[:8]
    file_path = None

    try:
        try:
            zarr_path = Path(temp_base) / f"marEx_checkpoint_{name}_{unique_id}.zarr"
            file_path = zarr_path

            logger.debug(f"Attempting Zarr checkpoint: {zarr_path}")
            # Check if time dimension has irregular chunks that need fixing
            if timedim in data.dims and timedim in data.chunks:
                time_chunks = data.chunks[timedim]
                if len(time_chunks) > 1 and len(set(time_chunks)) > 1:
                    # Chunks are irregular - need to fix for Zarr
                    first_chunk = time_chunks[0]
                    total_size = data.sizes[timedim]

                    logger.debug(f"Irregular {timedim} chunks detected: {time_chunks}")
                    logger.debug(f"Total {timedim} dimension size: {total_size}")

                    # Calculate how many full chunks we can have
                    n_full_chunks = total_size // first_chunk
                    remainder = total_size % first_chunk

                    if remainder > 0:
                        # Need full chunks + one smaller final chunk
                        new_time_chunks = (first_chunk,) * n_full_chunks + (remainder,)
                    else:
                        # All chunks are equal size
                        new_time_chunks = (first_chunk,) * n_full_chunks

                    data = data.chunk({timedim: new_time_chunks})
                    logger.debug(f"Adjusted {timedim} chunks for Zarr: {new_time_chunks}")
                    logger.debug(f"Verification - sum of chunks: {sum(new_time_chunks)}, dimension size: {total_size}")

            with log_timing(logger, f"Saving '{name}' to Zarr", logging.DEBUG, log_memory=False):
                data.to_zarr(zarr_path, mode="w")

            logger.debug("Zarr save successful, reloading...")
            with log_timing(logger, f"Reloading '{name}' from Zarr", logging.DEBUG, log_memory=False):
                if isinstance(data, xr.Dataset):
                    reloaded = xr.open_zarr(zarr_path, chunks={})
                else:
                    ds_temp = xr.open_zarr(zarr_path, chunks={})
                    reloaded = ds_temp[list(ds_temp.data_vars)[0]]

            logger.info(f"Checkpoint '{name}' saved via Zarr: {zarr_path}")
            return reloaded

        except (ValueError, OSError) as e:
            if "incompatible" in str(e) or "chunk" in str(e).lower():
                logger.warning(f"Zarr failed due to irregular chunks: {str(e)[:200]}")
                # Clean up failed zarr attempt
                if zarr_path.exists():
                    shutil.rmtree(zarr_path)
            else:
                raise

    except Exception as e:
        logger.error(f"Failed to checkpoint '{name}' to disk: {e}")
        logger.warning(f"Falling back to in-memory persist() only for '{name}'")

        # Fallback to in-memory persist (no disk I/O)
        try:
            reloaded = data.persist()
            from distributed import wait

            wait(reloaded)
            logger.info(f"Checkpoint '{name}' persisted to distributed memory (no disk)")
            return reloaded
        except Exception as e2:
            logger.error(f"Even persist() failed for '{name}': {e2}")
            logger.warning("Returning original data without checkpointing")
            return data

    finally:
        # Cleanup if requested
        if cleanup and file_path and file_path.exists():
            try:
                if file_path.suffix == ".zarr":
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()
                logger.debug(f"Cleaned up checkpoint file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")


def fix_dask_tuple_array(da: xr.DataArray) -> xr.DataArray:
    """
    Fix a dask array that has tuple (i.e. task) references in its chunks.
    This addresses a longstanding issue/bug when dask arrays are saved to Zarr.
    Process chunk by chunk to maintain memory efficiency.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with Dask array backend that may have tuple chunk references

    Returns
    -------
    xarray.DataArray
        DataArray with materialised chunks that can be safely saved to Zarr

    """
    # N.B.: Analyse the outputs of:
    #   first_key = result.data.__dask_keys__()[0]
    #   first_chunk = dask.compute(first_key)[0]
    #   print(type(first_chunk), first_chunk)

    def materialise_chunk(block: NDArray[Any]) -> NDArray[Any]:  # pragma: no cover
        """Force materialisation of a single chunk."""
        # This ensures we return an actual numpy array, not a task reference
        return np.asarray(block)

    chunks = da.chunks

    # Use map_blocks to process each chunk
    clean_data = dask_array.map_blocks(
        materialise_chunk,
        da.data,
        dtype=da.dtype,
        chunks=chunks,
        drop_axis=[],  # Keep all axes
        meta=np.array([], dtype=da.dtype),
    )

    # Create new DataArray with clean dask array
    return xr.DataArray(clean_data, dims=da.dims, coords=da.coords, attrs=da.attrs, name=da.name)
