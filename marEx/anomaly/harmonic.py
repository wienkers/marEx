"""
Harmonic-detrending anomaly method.

Provides :func:`_compute_anomaly_detrended`, which removes a polynomial trend and
optional seasonal harmonics from a time series to produce standardised anomalies.
This is the implementation behind the ``detrend_harmonic`` anomaly method.
"""

from typing import Dict, List, Optional

import flox.xarray
import numpy as np
import xarray as xr

from ..core.dimensions import TASK_ELEMENTS, extra_dim_chunks, horizontal_dims, spatial_dims
from ..core.numerics import rolling_numerics
from ..core.seasonal_apply import apply_seasonal_field
from ..core.time_axis import (
    _DAYS_PER_CALENDAR_YEAR,
    SeasonalCycle,
    _median_step_days,
    add_decimal_year,
    is_subdaily_axis,
    resolve_cycle,
)
from ..core.validation import _infer_dims_coords, unmask_dataarray
from ..exceptions import ConfigurationError
from ..logging_config import get_logger

# Get module logger
logger = get_logger(__name__)


def _compute_anomaly_detrended(
    da: xr.DataArray,
    standardise: bool = False,
    detrend_orders: Optional[List[int]] = None,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    force_zero_mean: bool = True,
    remove_harmonics: bool = True,
    cycle: Optional[SeasonalCycle] = None,
) -> xr.Dataset:
    """
    Generate normalised anomalies by removing trends, seasonal cycles, and optionally
    standardising by local temporal variability using the detrended baseline method.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    standardise : bool, default=False
        Whether to standardise anomalies by temporal variability
    detrend_orders : list, optional
        Polynomial orders for trend removal (default: [1] for linear)
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    force_zero_mean : bool, default=True
        Whether to enforce zero mean in detrended data
    remove_harmonics : bool, default=True
        Whether to remove seasonal harmonics (annual and semi-annual cycles)
    cycle : SeasonalCycle, optional
        Within-year axis the standardisation climatology is resolved on. Inferred
        from the time coordinate's cadence when omitted.

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies, mask, and optionally standardised data
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)
    da = unmask_dataarray(da)

    # Default detrend_orders to linear if not specified
    if detrend_orders is None:
        detrend_orders = [1]

    # Validate detrend_orders is not empty and contains valid values
    if not detrend_orders:
        raise ConfigurationError(
            "detrend_orders cannot be empty",
            details="At least one polynomial order must be specified for detrending",
            suggestions=[
                "Use detrend_orders=[1] for linear detrending",
                "Use detrend_orders=[1, 2] for linear + quadratic detrending",
                "Remove detrend_orders optional parameter to use default [1]",
            ],
        )

    # Validate all orders are positive integers
    if any(order < 1 for order in detrend_orders):
        invalid_orders = [order for order in detrend_orders if order < 1]
        raise ConfigurationError(
            f"Invalid polynomial orders: {invalid_orders}",
            details="Polynomial orders must be positive integers (≥ 1)",
            suggestions=[
                "Use only positive integers for polynomial orders",
                "Common values: [1] for linear, [1,2] for linear+quadratic",
                f"Remove invalid orders: {invalid_orders}",
            ],
        )

    da = da.astype(np.float32)

    # Ensure time is the first dimension for efficient processing
    if da.dims[0] != dimensions["time"]:
        da = da.transpose(dimensions["time"], ...)

    # Warn if using higher-order detrending without linear component
    if 1 not in detrend_orders and len(detrend_orders) > 1:
        logger.warning("Higher-order detrending without linear term may be unstable")

    # The harmonic basis below is annual + semi-annual only. On a sub-daily axis the
    # entire diurnal cycle therefore survives into `dat_anomaly` -- silently wrong
    # rather than slow, which is the worst failure mode available. Reject it and steer
    # to the group-based methods, whose climatology is resolved on the `hourofyear`
    # cycle and so removes the diurnal cycle for free.
    #
    # Only `remove_harmonics=True` is affected: `detrend_fixed_baseline` calls this
    # function with harmonics off and then subtracts a cycle-resolved climatology, which
    # handles a sub-daily cycle correctly.
    #
    # `is_subdaily_axis`, not `resolve_cycle`: this guard only needs to know whether the
    # axis is finer than daily, and `infer_cycle` raises on a mixed-cadence axis. Making
    # a harmonic run fail there would be the same "names the wrong problem" bug the
    # guard itself exists to prevent.
    if remove_harmonics and is_subdaily_axis(da, coordinates["time"], cycle):
        step_days = _median_step_days(da[coordinates["time"]]) if cycle is None else cycle.step_days
        raise ConfigurationError(
            "method_anomaly='detrend_harmonic' does not support sub-daily data",
            details=(
                f"The harmonic basis removes annual and semi-annual cycles only, so on a "
                f"{step_days * 24:g}-hour time axis the diurnal cycle would remain in "
                "the anomaly. That is a silently wrong result rather than a slow one, so it is "
                "rejected rather than computed."
            ),
            suggestions=[
                "Use method_anomaly='shifting_baseline' or 'fixed_baseline', whose climatology is "
                "resolved on the sub-daily cycle and removes the diurnal cycle",
                "Use method_anomaly='detrend_fixed_baseline' if a trend must also be removed",
                "Resample the data to daily before calling marEx",
            ],
            context={
                "method_anomaly": "detrend_harmonic",
                "step_days": step_days,
            },
        )

    # Add decimal year for trend modelling
    da = add_decimal_year(da, dim=dimensions["time"], coord=coordinates["time"])
    dy = da.decimal_year.compute()

    # Build model matrix with constant term, trends, and seasonal harmonics
    model_components = [np.ones(len(dy))]  # Constant term

    # Add polynomial trend terms
    centered_time = da.decimal_year - np.mean(dy)
    for order in detrend_orders:
        model_components.append(centered_time**order)

    # Add annual and semi-annual cycles (harmonics) if requested
    if remove_harmonics:
        model_components.extend(
            [
                np.sin(2 * np.pi * dy),  # Annual sine
                np.cos(2 * np.pi * dy),  # Annual cosine
                np.sin(4 * np.pi * dy),  # Semi-annual sine
                np.cos(4 * np.pi * dy),  # Semi-annual cosine
            ]
        )

    # Convert to numpy array for matrix operations
    model = np.array(model_components)

    # Orthogonalise model components for numerical stability
    for i in range(1, model.shape[0]):
        model[i] = model[i] - np.mean(model[i]) * model[0]

    # Compute pseudo-inverse for model fitting
    pmodel = np.linalg.pinv(model)
    n_coeffs = len(model_components)

    # Convert model matrices to xarray, keeping any existing index on the time dimension.
    time_dim = dimensions["time"]
    model_coords = {"coeff": np.arange(1, n_coeffs + 1)}
    if time_dim in da.indexes:
        model_coords[time_dim] = da.coords[time_dim]
    model_da = xr.DataArray(
        model.T,
        dims=[time_dim, "coeff"],
        coords=model_coords,
    ).chunk({time_dim: da.chunksizes[time_dim]})

    pmodel_da = xr.DataArray(
        pmodel.T,
        dims=["coeff", time_dim],
        coords=model_coords,
    ).chunk({time_dim: da.chunksizes[time_dim]})

    # Fit model to data - use the actual dimensions of the result
    dot_result = pmodel_da.dot(da)
    # For dot product result, dimensions match input data's spatial dimensions
    fit_dims = [dim for dim in da.dims if dim != dimensions["time"]]
    result_dims = ["coeff"] + fit_dims

    # Build coordinates for the result
    result_coords = {"coeff": np.arange(1, n_coeffs + 1)}
    for dim in fit_dims:
        if dim in da.coords:
            result_coords[dim] = da.coords[dim]

    # Persist the *fit* rather than the detrended series. The fit is a full reduction over
    # the input but is only (coeff, y, x) -- tens of MB -- whereas the detrended series is
    # the size of the input. With the coefficients materialised, the detrend below is a
    # cheap elementwise expression, so the pipeline-level persist of dat_anomaly is the
    # only full-size copy that ever exists.
    model_fit_da = xr.DataArray(dot_result, dims=result_dims, coords=result_coords).persist()

    # Remove trend and seasonal cycle
    da_detrend = da.drop_vars({"decimal_year"}) - model_da.dot(model_fit_da).astype(np.float32)

    # Force zero mean if requested
    if force_zero_mean:
        da_detrend = da_detrend - da_detrend.mean(dim=dimensions["time"])

    # Create ocean/land mask from first time step
    # Handle both spatial (3D) and time-series (1D) data
    # Every spatial dim is made whole, extra dims (depth, level) included.
    mask_dims = spatial_dims(da, dimensions)
    if mask_dims:
        # Spatial data - create 2D/3D mask
        chunk_dict_mask = {dim: -1 for dim in mask_dims}
        vars_to_drop = [c for c, v in da.coords.items() if dimensions["time"] in v.dims and c != dimensions["time"]]
        mask = (
            np.isfinite(da.isel({dimensions["time"]: 0}, drop=True)).drop_vars(vars_to_drop, errors="ignore").chunk(chunk_dict_mask)
        )
    else:
        # 1D time series - create scalar mask indicating if any finite values exist
        chunk_dict_mask = {}  # Empty for 1D case
        mask = xr.DataArray(np.any(np.isfinite(da.values)), dims=[], attrs={"description": "Time series validity mask"})

    # Initialise output dataset
    data_vars = {"dat_anomaly": da_detrend, "mask": mask}

    # Ensure all original coordinates are preserved in the dataset
    coords_to_preserve = {c: da.coords[c] for c in da.coords if c not in data_vars and c != "decimal_year"}

    # Standardise anomalies by temporal variability if requested
    if standardise:
        inferred_cycle = cycle is None
        cycle = resolve_cycle(da, coordinates["time"], cycle)
        cycle_dim = cycle.index_name
        time_coord = da_detrend[coordinates["time"]]
        cycle_length = cycle.length
        if inferred_cycle and cycle.is_daily:
            fixed_year_days = _DAYS_PER_CALENDAR_YEAR.get(time_coord.dt.calendar)
            year_days = fixed_year_days or (366 if bool(time_coord.dt.is_leap_year.any()) else 365)
            cycle_length = min(cycle_length, int(year_days))
        cycle_index = cycle.index_of(time_coord).compute()
        in_cycle = (cycle_index >= 1) & (cycle_index <= cycle_length)
        group_index = cycle_index
        if not bool(in_cycle.all()):
            group_index = cycle_index.where(in_cycle)
            da_detrend = da_detrend.where(in_cycle)
            data_vars["dat_anomaly"] = da_detrend

        if bool(in_cycle.any()):
            std_day = flox.xarray.xarray_reduce(
                da_detrend,
                group_index,
                dim=time_dim,
                func="std",
                ddof=0,
                isbin=False,
                method="cohorts",
                dtype=np.float32,
            )

            # Roll over physical calendar slots, including empty slots as gaps. Keep
            # the legacy even-window alignment (15 days before and 14 after on daily data).
            observed = std_day.get_index(cycle_dim).astype(cycle.labels.dtype)
            window = SeasonalCycle(cycle_dim, cycle_length, cycle.slot_days).steps_for_days(30, name="standardisation window")
            if window == 1:
                logger.warning(
                    "The 30-day standardisation window resolves to a single %s slot: "
                    "the per-slot standard deviation is used unsmoothed.",
                    cycle_dim,
                )
            window = min(window, cycle_length)
            pad = window // 2 + 1
            wrapped = (
                std_day.reindex({cycle_dim: cycle.labels[:cycle_length]}).chunk({cycle_dim: -1}).pad({cycle_dim: pad}, mode="wrap")
            )
            with rolling_numerics():
                std_rolling = np.sqrt((wrapped**2).rolling({cycle_dim: window}, center=True, min_periods=1).mean())
            std_rolling = std_rolling.isel({cycle_dim: slice(pad, pad + cycle_length)}).sel({cycle_dim: observed})
            std_rolling = std_rolling.assign_coords({cycle_dim: observed.to_numpy()})
        else:
            drop_time_coords = [c for c, v in da_detrend.coords.items() if time_dim in v.dims and c != time_dim]
            std_rolling = (
                da_detrend.drop_vars(drop_time_coords, errors="ignore")
                .isel({time_dim: 0}, drop=True)
                .expand_dims({cycle_dim: cycle.labels[:0]})
            )

        # Divide anomalies by rolling standard deviation
        # Replace any zeros or extremely small values with NaN to avoid division warnings
        std_rolling_safe = std_rolling.where(std_rolling > 1e-10, np.nan)
        da_stn = apply_seasonal_field(
            da_detrend,
            std_rolling_safe,
            cycle_index.values,
            timedim=dimensions["time"],
            cycledim=cycle_dim,
            operation="divide",
        )

        # Rechunk data for efficient processing: horizontal whole, extra dims (depth, level)
        # per level unless a small grid leaves budget to stack them -- the layout `finalise`
        # gives the output. Holding depth whole here made each chunk depth-times
        # its 2-D size before finalise split it again.
        horizontal = [dim for dim in chunk_dict_mask if dim in horizontal_dims(dimensions)]
        extra = [dim for dim in chunk_dict_mask if dim not in horizontal]
        horizontal_cells = int(np.prod([da_stn.sizes[dim] for dim in horizontal]))

        def _field_chunks(per_cell: int) -> Dict[str, int]:
            chunks = {dim: -1 for dim in horizontal}
            chunks.update(
                extra_dim_chunks(
                    da_stn.sizes,
                    extra,
                    horizontal_tile_cells=horizontal_cells,
                    horizontal_cells=horizontal_cells,
                    budget_cells=TASK_ELEMENTS // max(1, per_cell),
                )
            )
            return chunks

        chunk_dict_field = _field_chunks(max(da_stn.chunksizes[dimensions["time"]]))
        chunk_dict_std = _field_chunks(std_rolling.sizes[cycle_dim])
        chunk_dict_std[cycle_dim] = -1

        da_stn = da_stn.chunk(chunk_dict_field)
        std_rolling = std_rolling.chunk(chunk_dict_std)

        # Add standardised data to output
        data_vars["dat_stn"] = da_stn
        data_vars["STD"] = std_rolling

    # Build output dataset with metadata and restore custom auxiliary indexes
    ds_out = xr.Dataset(data_vars=data_vars, coords=coords_to_preserve).drop_vars("decimal_year", errors="ignore")
    for idx_name in da.xindexes:
        if idx_name not in ds_out.xindexes and idx_name in ds_out.coords:
            ds_out = ds_out.set_xindex(idx_name)
    return ds_out
