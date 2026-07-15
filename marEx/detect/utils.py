"""
General-purpose helpers for the marEx detection pipeline.

Currently provides :func:`add_decimal_year`, a pure datetime helper used by the
harmonic detrending anomaly method. This module is a leaf in the detect package
dependency graph and imports only third-party libraries and the package logger.
"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
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
    ``std_normalise``). These round-trip fine through Zarr but make ``Dataset.to_netcdf``
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


def add_decimal_year(da: xr.DataArray, dim: str = "time", coord: Optional[str] = None) -> xr.DataArray:
    """
    Add decimal year coordinate to DataArray for trend analysis.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with datetime coordinate
    dim : str, optional
        Name of the time dimension
    coord : str, optional
        Name of the time coordinate (if different from dimension name)

    Returns
    -------
    xarray.DataArray
        Input data with added 'decimal_year' coordinate
    """
    # Use coordinate name if provided, otherwise use dimension name
    coord_name = coord if coord is not None else dim
    time = pd.to_datetime(da[coord_name])
    start_of_year = pd.to_datetime(time.year.astype(str) + "-01-01")
    start_of_next_year = pd.to_datetime((time.year + 1).astype(str) + "-01-01")
    year_elapsed = (time - start_of_year).days
    year_duration = (start_of_next_year - start_of_year).days

    decimal_year = time.year + year_elapsed / year_duration
    return da.assign_coords(decimal_year=(dim, decimal_year))
