"""
MarEx Track: Regional tracker factory.

Thin convenience factory that constructs a :class:`tracker` configured for
regional (non-global) data with ``regional_mode=True`` and explicit coordinate
units. Kept separate so the public ``regional_tracker`` entry point has a clear
home and the orchestrator class stays focused.
"""

from typing import Literal, Optional, Union

import xarray as xr

from .tracker import tracker


def regional_tracker(
    data_bin: xr.DataArray,
    mask: xr.DataArray,
    coordinate_units: Literal["degrees", "radians"],
    R_fill: Union[int, float],
    area_filter_quartile: Optional[float] = None,
    area_filter_absolute: Optional[int] = None,
    **kwargs,
) -> "tracker":
    """
    Create a tracker instance configured for regional (non-global) data.

    This is a convenience function that automatically sets regional_mode=True
    and requires explicit specification of coordinate units, since auto-detection
    may fail for regional coordinate ranges.

    Parameters
    ----------
    data_bin : xr.DataArray
        Binary data to identify and track objects in (True = object, False = background)
    mask : xr.DataArray
        Binary mask indicating valid regions (True = valid, False = invalid)
    coordinate_units : {'degrees', 'radians'}
        Units of the coordinate system. Must be specified for regional data.
    R_fill : int or float
        Radius for filling holes/gaps in spatial domain (in grid cells)
    area_filter_quartile : float, optional
        Quantile (0-1) for filtering smallest objects (e.g., 0.25 removes smallest 25%).
        Mutually exclusive with area_filter_absolute. Default is 0.5 if neither parameter is provided.
    area_filter_absolute : int, optional
        The minimum area (in grid cells) for an object to be retained. Mutually exclusive with area_filter_quartile.
    **kwargs
        Additional parameters passed to the tracker class

    Returns
    -------
    tracker
        Configured tracker instance with regional_mode=True

    Examples
    --------
    Track events in regional Mediterranean Sea data:

    >>> import marEx
    >>> # For regional data with degree coordinates
    >>> regional_tracker = marEx.regional_tracker(
    ...     extreme_events,
    ...     mask,
    ...     coordinate_units='degrees',
    ...     R_fill=5,
    ...     area_filter_quartile=0.3
    ... )
    >>> events = regional_tracker.run()

    Track events in regional data with radian coordinates:

    >>> # For model output with radian coordinates
    >>> regional_tracker = marEx.regional_tracker(
    ...     extreme_events,
    ...     mask,
    ...     coordinate_units='radians',
    ...     R_fill=8,
    ...     area_filter_quartile=0.5
    ... )
    >>> events = regional_tracker.run()

    Using absolute area filtering in regional mode:

    >>> # Keep only features larger than 15 grid cells
    >>> absolute_regional = marEx.regional_tracker(
    ...     extreme_events,
    ...     mask,
    ...     coordinate_units='degrees',
    ...     R_fill=5,
    ...     area_filter_absolute=15
    ... )
    >>> events = absolute_regional.run()
    """
    return tracker(
        data_bin=data_bin,
        mask=mask,
        R_fill=R_fill,
        area_filter_quartile=area_filter_quartile,
        area_filter_absolute=area_filter_absolute,
        regional_mode=True,
        coordinate_units=coordinate_units,
        **kwargs,
    )
