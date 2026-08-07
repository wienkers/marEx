"""Incremental zarr writer for the tracker's whole-field ID accumulator.

The gridded merge/split loop (:func:`marEx.track.merge_split.split_and_merge_objects`)
produces its output one time chunk at a time, as *concrete numpy*, from a Python loop.
That fits neither verb of :class:`marEx.detect.compute_mode.Materialiser`: ``pin`` takes
bounded intermediates and ``stage`` takes a finished lazy graph. Hence a third verb.

Why a plain region write is safe here
-------------------------------------
The store is **write-only for the duration of the loop** -- nothing the loop writes is
ever read back before :meth:`finalise`. Two properties of the loop establish this, and
both must be preserved by anyone editing it:

1. A chunk is fully finalised before it is handed over. The end-of-chunk consolidation
   mutates ``chunk_data[{timedim: -1}]`` (``merge_split.py:960``) *before* the append at
   ``:963``; the in-loop mutation at ``:745`` only ever touches the current chunk.
2. Cross-chunk continuity comes from memory, not from the field. At ``relative_t == 0``
   and ``== 1`` the loop reads ``t-1``/``t-2`` from ``updated_chunks[-1]``
   (``merge_split.py:705-724``), *not* from the accumulator. That is exactly why the
   flush writes ``updated_chunks[:-1]`` and retains ``[-1:]``.

So: only write chunks the loop has finished with. Do **not** "simplify" the flush to
write every chunk immediately -- the retained chunk is still needed as the next chunk's
``t-1``/``t-2`` source, and a zarr region write cannot be un-written.

Coordinates
-----------
:meth:`_initialise` writes every coordinate once, eagerly. Region writes then carry
**none** -- :meth:`write` drops them -- because zarr rejects a region whose coordinate
extents do not match the store's. A region therefore never re-writes even its own slice
of the time coordinate. This is deliberate, not an oversight.
"""

from pathlib import Path
from typing import Union

import xarray as xr

from ..logging_config import get_logger

logger = get_logger(__name__)


class ObjectIDRegionWriter:
    """Write disjoint time regions of an ID field to zarr, then re-open lazily.

    Parameters
    ----------
    template : xarray.DataArray
        The field whose schema (shape, dtype, dims, coords, chunking) the store takes.
        Its *values* are never written; only regions passed to :meth:`write` are.
    path : str or pathlib.Path
        Destination zarr store. Overwritten if it exists.
    timedim : str
        Name of the dimension the regions index.
    """

    def __init__(self, template: xr.DataArray, path: Union[str, Path], timedim: str) -> None:
        """Initialise the writer against ``template``'s schema; nothing is written yet."""
        self.path = str(path)
        self.timedim = timedim
        self.name = template.name or "ID_field"
        self._template = template
        self._initialised = False
        self._n_written = 0

    def _initialise(self) -> None:
        """Write schema and coordinates, but no field data."""
        ds = self._template.to_dataset(name=self.name)

        # Stale `chunks` encoding carried in from an upstream open_zarr conflicts with the
        # dask chunking now being written -- the same trap Materialiser._stage_to_zarr hits.
        for var in list(ds.variables):
            ds[var].encoding.pop("chunks", None)

        # compute=False writes metadata and eager (numpy) coordinates, leaving the dask-backed
        # field unwritten. Every element is then supplied by write().
        ds.to_zarr(self.path, mode="w", compute=False, consolidated=True)
        self._initialised = True
        logger.debug(f"Initialised ID-field region store at {self.path}")

    def write(self, start: int, end: int, data: xr.DataArray) -> None:
        """Write one finished time region.

        Parameters
        ----------
        start, end : int
            Half-open region bounds along ``timedim``.
        data : xarray.DataArray
            The finished chunk. Length along ``timedim`` must equal ``end - start``.
        """
        if not self._initialised:
            self._initialise()

        if data.sizes[self.timedim] != end - start:
            raise ValueError(
                f"region [{start}, {end}) expects {end - start} steps along {self.timedim}, " f"got {data.sizes[self.timedim]}"
            )

        ds = data.to_dataset(name=self.name)
        # Coordinates were written by _initialise; a region write must carry only the
        # data variable, or zarr rejects the mismatched coordinate extents.
        ds = ds.drop_vars(list(ds.coords))
        ds.to_zarr(self.path, region={self.timedim: slice(start, end)})
        self._n_written += 1

    def finalise(self) -> xr.DataArray:
        """Re-open the completed store lazily.

        Returns
        -------
        xarray.DataArray
            Dask-backed, reading from disk, with the on-disk chunking.
        """
        if not self._n_written:
            raise RuntimeError(f"finalise() called but no regions were written to {self.path}")

        reopened = xr.open_zarr(self.path, consolidated=True, chunks={})[self.name]

        # Drop round-tripped encoding so a later save by the caller does not conflict.
        for coord in reopened.coords:
            reopened[coord].encoding = {}
        reopened.encoding = {}

        logger.info(f"ID field staged to {self.path} in {self._n_written} regions")
        return reopened
