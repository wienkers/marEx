"""
NetCDF-safe attribute coercion.

Serialising an xarray object to NetCDF rejects attribute values that have no
NetCDF type (``None``, ``bool``, nested containers). These helpers coerce an
object's ``attrs`` into a writable form without touching any data.
"""

from typing import Any, Union

import numpy as np
import xarray as xr

from ..logging_config import get_logger

# Get module logger
logger = get_logger(__name__)


def _coerce_netcdf_safe(value: Any) -> Any:
    """Coerce a single attribute value to a NetCDF-serialisable type.

    NetCDF attributes support only strings and numeric scalars/arrays; Python/NumPy
    booleans and ``None`` are rejected by the netCDF4 backend. Booleans are stored as
    ``int8`` (0/1) -- the same convention the tracker already uses for its flag attrs --
    and ``None`` is stored as the string ``"None"``. All other values pass through
    unchanged.
    """
    if isinstance(value, (bool, np.bool_)):
        return np.int8(value)
    if value is None:
        return "None"
    return value


def make_netcdf_safe_attrs(ds: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """Coerce dataset/coord/variable attributes to NetCDF-serialisable types in place.

    Some preprocessing attributes are naturally Python booleans (e.g. ``force_zero_mean``,
    ``standardise``). These round-trip fine through Zarr but make ``Dataset.to_netcdf``
    raise ``TypeError: illegal data type for attribute ... got b1``. This sweep makes the
    returned dataset directly saveable to *both* Zarr and NetCDF without the caller having
    to sanitise anything -- booleans become ``int8`` and ``None`` becomes ``"None"``.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Object whose attributes (and those of its coords/variables) are sanitised.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The same object, with attributes coerced in place.
    """

    def _sweep(attrs: dict) -> None:
        for key, value in list(attrs.items()):
            attrs[key] = _coerce_netcdf_safe(value)

    _sweep(ds.attrs)
    for coord_name in ds.coords:
        _sweep(ds[coord_name].attrs)
    if isinstance(ds, xr.Dataset):
        for var_name in ds.data_vars:
            _sweep(ds[var_name].attrs)
    return ds
