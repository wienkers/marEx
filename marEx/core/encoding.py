"""
Clearing store-specific ``encoding`` carried in from an input store.

A variable opened from a zarr store keeps that store's on-disk layout and codecs in
``.encoding``, and xarray re-applies them on the next ``to_zarr``. Two keys bite:

* ``chunks``: the input's on-disk chunking, which conflicts with the dask chunking being
  written and raises a chunk-misalignment error.
* the codec keys (``compressor``, ``filters``, ...): a store opened from zarr format 2
  carries ``numcodecs`` codecs that zarr-python 3 refuses to write into a format-3 store
  ("Expected a BytesBytesCodec"). Dropping them lets the writing store pick its own
  default compressor; values are unaffected.

CF keys (``units``, ``calendar``, ``_FillValue``, ``dtype``) are left alone.
"""

from typing import Union

import xarray as xr

STORE_ENCODING_KEYS = ("chunks", "compressor", "compressors", "filters", "serializer", "shards")


def clear_store_encoding(ds: Union[xr.Dataset, xr.DataArray]) -> None:
    """Drop the input store's chunk and codec encoding from every variable, in place."""
    variables = ds.variables.values() if isinstance(ds, xr.Dataset) else [ds.variable, *(c.variable for c in ds.coords.values())]
    for variable in variables:
        for key in STORE_ENCODING_KEYS:
            variable.encoding.pop(key, None)


def write_zarr(obj: Union[xr.Dataset, xr.DataArray], store, **kwargs):
    """``obj.to_zarr(store, **kwargs)`` without the store encoding ``obj`` was opened with.

    Every marEx write goes through here (a test greps for any that does not). The encoding
    is cleared on a shallow copy, whose variables get their own ``encoding`` dicts, so the
    caller's variables -- often the user's input coordinates -- keep theirs.
    """
    obj = obj.copy(deep=False)
    clear_store_encoding(obj)
    return obj.to_zarr(store, **kwargs)


def clear_inherited_attrs(ds: xr.Dataset, names) -> None:
    """Give each named variable empty attrs, in place, so outputs do not depend on xarray's ``keep_attrs``.

    xarray >= 2025.11 keeps attributes through arithmetic, comparisons and reductions by
    default, so a computed anomaly or event mask would otherwise carry the INPUT's
    ``standard_name``, ``units`` and ``valid_min``/``valid_max``, which do not describe it.
    These variables carried no attrs under the old default, and that is what they keep.
    A fresh dict is assigned rather than cleared: a shallow copy can share its attrs dict
    with the caller's input variable.
    """
    for name in names:
        if name in ds.variables:
            ds.variables[name].attrs = {}
