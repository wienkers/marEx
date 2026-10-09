"""
Fixed-baseline anomaly methods.

Provides :func:`_compute_anomaly_fixed_baseline` (simple daily climatology) and
:func:`_compute_anomaly_detrend_fixed_baseline` (polynomial detrending followed by
a fixed daily climatology). These back the ``fixed_baseline`` and
``detrend_fixed_baseline`` anomaly methods respectively.
"""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import flox.xarray
import numpy as np
import xarray as xr

from ..core.compute_mode import Materialiser
from ..core.dimensions import spatial_dims
from ..core.numerics import rolling_numerics
from ..core.time_axis import SeasonalCycle, resolve_cycle
from ..core.validation import _infer_dims_coords
from ..exceptions import ConfigurationError
from ..logging_config import get_logger
from .harmonic import _compute_anomaly_detrended

# Get module logger
logger = get_logger(__name__)


def _smooth_climatology_circular(clim: xr.DataArray, cycle: SeasonalCycle, smooth_days: float) -> xr.DataArray:
    """Smooth a per-cycle-slot climatology with a centred moving average that wraps the year.

    Hobday et al. (2016)'s smoothing step: the climatology is a closed cycle, so 31 December
    is averaged with 1 January rather than losing the window's half-width at either end.
    Even windows use the same ``center=True`` alignment as ``shifting_baseline``. NaN slots
    are skipped in the average and stay NaN, so each cell keeps the valid days it had.

    The average runs across DAYS at a fixed time of day. On sub-daily cycles the slots are
    viewed as ``(day, step-of-day)`` and only the day axis is smoothed: a moving average over
    consecutive hours would average the diurnal cycle out of the climatology and leave it in
    the anomaly. Daily and monthly cycles have one step per row, so this is the plain moving
    average along the cycle. ``smooth_days`` is converted with the row width (one day, or one
    month), not the data cadence.

    Operates on the single-chunk cycle axis, so it costs ``cycle.length x space`` and does
    not depend on how the input was chunked.
    """
    cycle_dim = cycle.index_name
    steps_per_row = cycle.steps_per_day if cycle.is_subdaily else 1
    n_rows = cycle.length // steps_per_row
    row_days = cycle.slot_days * steps_per_row
    window = replace(cycle, step_days=row_days).steps_for_days(smooth_days, name="smooth_days")
    if window >= n_rows:
        raise ConfigurationError(
            f"smooth_days={smooth_days} spans {window} of the {n_rows} rows of the {cycle_dim} cycle",
            details="Smoothing the fixed-baseline climatology over a whole cycle would remove the seasonal cycle itself",
            suggestions=["Use a smooth_days shorter than one year (the default is 21)", "Set smooth_days=1 to disable smoothing"],
            context={"smooth_days": smooth_days, "window": window, "cycle_rows": n_rows},
        )
    if window == 1:
        if smooth_days > 1:
            logger.info(
                "smooth_days=%s is under one %g-day %s row: the fixed-baseline climatology is not smoothed.",
                smooth_days,
                row_days,
                cycle_dim,
            )
        return clim

    # View the cycle as (row, step-of-row), pad the row axis by a whole window on each side
    # so every kept row sees a full window, cut the original span back out, then fold back.
    # The coordinate is dropped first: wrapping it would duplicate labels.
    labels = clim[cycle_dim].values
    dims = clim.dims
    rows = clim.drop_vars(cycle_dim).coarsen({cycle_dim: steps_per_row}).construct({cycle_dim: ("_row", "_step")})
    padded = rows.pad({"_row": (window, window)}, mode="wrap")
    # NaN-preserving: average the finite rows in each window, then re-mask rows that
    # were NaN before smoothing, so a seasonal-NaN cell (sea ice) keeps exactly its valid days
    # instead of losing half a window at each edge of its NaN season.
    with rolling_numerics():
        smoothed = (
            padded.rolling({"_row": window}, center=True, min_periods=1).mean().isel({"_row": slice(window, window + n_rows)})
        )
    smoothed = smoothed.where(rows.notnull())
    other = [d for d in dims if d != cycle_dim]
    smoothed = smoothed.transpose("_row", "_step", *other)
    folded = smoothed.data.reshape((cycle.length,) + smoothed.shape[2:])
    out = xr.DataArray(
        folded, dims=(cycle_dim, *other), coords={cycle_dim: labels, **{k: v for k, v in clim.coords.items() if k != cycle_dim}}
    )
    return out.transpose(*dims).astype(np.float32).chunk({cycle_dim: -1})


def _compute_anomaly_fixed_baseline(
    da: xr.DataArray,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    reference_period: Optional[Tuple[int, int]] = None,
    materialiser: Optional[Materialiser] = None,
    cycle: Optional[SeasonalCycle] = None,
    smooth_days: float = 21,
) -> xr.Dataset:
    """
    Compute anomalies using fixed baseline method with full time series climatology.

    This method computes a daily climatology using all available years in the dataset
    (or a specified reference period), smooths it with a ``smooth_days`` centred moving
    average that wraps the year (Hobday et al. 2016), then subtracts it from the
    original data to obtain anomalies.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    reference_period : tuple of (int, int), optional
        Year range (start_year, end_year) inclusive for computing the daily climatology.
        If None (default), uses all available years. Anomalies are still computed for
        the full time series.
    smooth_days : float, default=21
        Width of the circular moving average applied to the climatology, in days.
        ``smooth_days=1`` disables smoothing.

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # A None materialiser means "default to persist mode", which keeps every existing
    # caller, doctest and test working unchanged.
    if materialiser is None:
        materialiser = Materialiser("persist")

    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)
    cycle = resolve_cycle(da, coordinates["time"], cycle)
    cycle_dim = cycle.index_name

    # Select data for climatology computation (optionally restricted to reference period)
    if reference_period is not None:
        start_year, end_year = reference_period
        if start_year > end_year:
            raise ConfigurationError(
                f"Invalid reference_period: start year ({start_year}) must be <= end year ({end_year})",
                details="The reference_period tuple must be (start_year, end_year) with start_year <= end_year",
                suggestions=[f"Swap the order: use reference_period=({end_year}, {start_year})"],
            )
        years = da[coordinates["time"]].dt.year
        year_mask = (years >= start_year) & (years <= end_year)
        da_for_clim = da.isel({dimensions["time"]: year_mask})
        if da_for_clim.sizes[dimensions["time"]] == 0:
            data_min_year = int(years.min().values)
            data_max_year = int(years.max().values)
            raise ConfigurationError(
                f"No data found in reference_period ({start_year}, {end_year})",
                details=f"Dataset spans {data_min_year}-{data_max_year} but no timesteps fall within the specified period",
                suggestions=[
                    f"Adjust reference_period to overlap with data range ({data_min_year}-{data_max_year})",
                    "Set reference_period=None to use the full time series",
                ],
            )
        logger.debug(
            f"Using reference_period ({start_year}-{end_year}): "
            f"{da_for_clim.sizes[dimensions['time']]} of {da.sizes[dimensions['time']]} timesteps"
        )
    else:
        da_for_clim = da

    # Compute daily climatology using flox for efficiency
    logger.debug("Computing daily climatology across %s", "reference period" if reference_period else "all years")
    daily_climatology = flox.xarray.xarray_reduce(
        da_for_clim,
        cycle.index_of(da_for_clim[coordinates["time"]]),
        dim=dimensions["time"],
        func="nanmean",
        isbin=False,
        method="cohorts",
        dtype=np.float32,
    )

    # Ensure the climatology spans the full day-of-year range 1..366. If the reference
    # period contains no leap year it only has 365 groups, and subtracting it from a
    # full series that does include 29 Feb (day-of-year 366) would silently NaN every
    # such day. Reindex to 366 and forward-fill the missing tail group from day 365.
    # In the common (leap-containing) case both operations are no-ops. The dayofyear
    # dim is rechunked to a single chunk so the dask ffill is valid.
    daily_climatology = (
        daily_climatology.reindex({cycle_dim: np.arange(1, cycle.length + 1)}).chunk({cycle_dim: -1}).ffill(cycle_dim)
    )
    daily_climatology = _smooth_climatology_circular(daily_climatology, cycle, smooth_days)

    # Compute anomalies by subtracting daily climatology from original data
    logger.debug("Computing anomalies by subtracting daily climatology")
    da = da.assign_coords({cycle_dim: cycle.index_of(da[coordinates["time"]])})
    anomalies = da.groupby({cycle_dim: xr.groupers.UniqueGrouper(labels=np.arange(1, cycle.length + 1))}) - daily_climatology
    anomalies = anomalies.astype(np.float32)

    # Drop the cycle-index coordinate to avoid merge conflicts
    if cycle_dim in anomalies.coords:
        anomalies = anomalies.drop_vars(cycle_dim)

    # Create ocean/land mask from first time step
    # Handle both spatial (3D) and time-series (1D) data
    mask_dims = spatial_dims(da, dimensions)
    if mask_dims:
        # Spatial data - create 2D/3D mask.
        # `da` gained a per-timestep ``dayofyear`` coord above; dropping only the time
        # coord would leak a scalar ``dayofyear`` into the mask (and the output schema
        # under global_percentile). Drop both.
        # Extra (non-horizontal) spatial dims such as depth are made whole alongside
        # the horizontal ones, so the mask keeps the field's full spatial shape.
        chunk_dict_mask = {dim: -1 for dim in mask_dims}
        coords_to_drop = [coordinates["time"]]
        if cycle_dim in da.coords:
            coords_to_drop.append(cycle_dim)
        mask = np.isfinite(da.isel({dimensions["time"]: 0})).drop_vars(coords_to_drop).chunk(chunk_dict_mask)
    else:
        # 1D time series - create scalar mask indicating if any finite values exist
        mask = xr.DataArray(np.any(np.isfinite(da.values)), dims=[], attrs={"description": "Time series validity mask"})

    # Build output dataset
    return xr.Dataset({"dat_anomaly": anomalies, "mask": mask})


def _compute_anomaly_detrend_fixed_baseline(
    da: xr.DataArray,
    detrend_orders: Optional[List[int]] = None,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    force_zero_mean: bool = True,
    reference_period: Optional[Tuple[int, int]] = None,
    materialiser: Optional[Materialiser] = None,
    cycle: Optional[SeasonalCycle] = None,
    smooth_days: float = 21,
) -> xr.Dataset:
    """
    Compute anomalies using fixed detrended baseline method.

    This method first removes polynomial trends (without harmonics) from the data,
    then removes a daily climatology from the detrended signal. The trend removal
    always uses the full time series; only the climatology step respects reference_period.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    detrend_orders : list, optional
        Polynomial orders for trend removal (default: [1] for linear)
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    force_zero_mean : bool, default=True
        Whether to enforce zero mean in detrended data
    reference_period : tuple of (int, int), optional
        Year range (start_year, end_year) inclusive for computing the daily climatology.
        If None (default), uses all available years. Only affects the climatology step,
        not the polynomial detrending.
    smooth_days : float, default=21
        Width of the circular moving average applied to the climatology, in days.
        ``smooth_days=1`` disables smoothing.

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies and mask
    """
    # A None materialiser means "default to persist mode", which keeps every existing
    # caller, doctest and test working unchanged.
    if materialiser is None:
        materialiser = Materialiser("persist")

    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    logger.debug(f"Removing polynomial trends of orders: {detrend_orders}")

    # Step 1: Remove polynomial trends (without harmonics) using _compute_anomaly_detrended
    detrended_result = _compute_anomaly_detrended(
        da=da,
        standardise=False,
        detrend_orders=detrend_orders,
        dimensions=dimensions,
        coordinates=coordinates,
        force_zero_mean=force_zero_mean,
        remove_harmonics=False,  # Only remove trends, not harmonics
        cycle=cycle,
    )["dat_anomaly"]

    # Step 2: Compute daily climatology and anomalies using _compute_anomaly_fixed_baseline
    logger.debug("Computing daily climatology and anomalies from detrended data")
    final_result = _compute_anomaly_fixed_baseline(
        da=detrended_result,
        dimensions=dimensions,
        coordinates=coordinates,
        reference_period=reference_period,
        materialiser=materialiser,
        cycle=cycle,
        smooth_days=smooth_days,
    )

    return final_result
