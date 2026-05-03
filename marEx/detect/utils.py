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


def _is_cftime_coord(time_coord: xr.DataArray) -> bool:
    """Return True if ``time_coord`` holds cftime datetime objects.

    Only genuine cftime coordinates (object dtype containing ``cftime.datetime``
    instances) take the calendar-aware branch in :func:`add_decimal_year`; plain
    numeric or numpy-datetime64 coordinates fall through to the legacy path.
    """
    values = np.asarray(time_coord.values)
    if values.dtype != np.dtype("O") or values.size == 0:
        return False
    try:
        import cftime
    except ImportError:  # pragma: no cover - cftime ships with netcdf4
        return False
    return isinstance(values.reshape(-1)[0], cftime.datetime)


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
    time_coord = da[coord_name]

    if _is_cftime_coord(time_coord):
        # cftime / non-standard calendars (noleap, 360_day, ...): pd.to_datetime
        # cannot parse these. Derive the decimal year from day-of-year directly,
        # using a calendar-aware days-in-year so the fraction is correct. Matches the
        # legacy path's definition: year + (dayofyear - 1) / days_in_year.
        year = time_coord.dt.year
        dayofyear = time_coord.dt.dayofyear
        fixed_length = {"360_day": 360, "noleap": 365, "365_day": 365, "all_leap": 366, "366_day": 366}
        calendar = getattr(time_coord.dt, "calendar", "standard")
        if calendar in fixed_length:
            days_in_year = xr.full_like(year, fixed_length[calendar], dtype=float)
        else:
            # Calendars with real leap years (standard/gregorian/julian/proleptic_gregorian)
            days_in_year = 365.0 + time_coord.dt.is_leap_year.astype(float)
        decimal_year = np.asarray((year + (dayofyear - 1) / days_in_year).values)
    else:
        # Legacy path (unchanged; bit-identical for datetime64 and numeric coordinates).
        time = pd.to_datetime(time_coord)
        start_of_year = pd.to_datetime(time.year.astype(str) + "-01-01")
        start_of_next_year = pd.to_datetime((time.year + 1).astype(str) + "-01-01")
        year_elapsed = (time - start_of_year).days
        year_duration = (start_of_next_year - start_of_year).days
        decimal_year = time.year + year_elapsed / year_duration

    return da.assign_coords(decimal_year=(dim, decimal_year))
