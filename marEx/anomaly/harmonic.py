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

from ..core.time_axis import add_decimal_year
from ..core.validation import _infer_dims_coords
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

    Returns
    -------
    xarray.Dataset
        Dataset containing anomalies, mask, and optionally standardised data
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

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
        print("Warning: Higher-order detrending without linear term may be unstable")

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

    # Convert model matrices to xarray
    model_da = xr.DataArray(
        model.T,
        dims=[dimensions["time"], "coeff"],
        coords={
            dimensions["time"]: da[coordinates["time"]].values,
            "coeff": np.arange(1, n_coeffs + 1),
        },
    ).chunk({dimensions["time"]: da.chunks[0]})

    pmodel_da = xr.DataArray(
        pmodel.T,
        dims=["coeff", dimensions["time"]],
        coords={
            "coeff": np.arange(1, n_coeffs + 1),
            dimensions["time"]: da[coordinates["time"]].values,
        },
    ).chunk({dimensions["time"]: da.chunks[0]})

    # Prepare dimensions for model coefficients based on data structure
    dims = ["coeff"]
    coords = {"coeff": np.arange(1, n_coeffs + 1)}

    # Handle 1D (time series), 2D (unstructured) and 3D (gridded) data
    if "y" in dimensions:  # 3D gridded case
        dims.extend([dimensions["y"], dimensions["x"]])
        coords[dimensions["y"]] = da[coordinates["y"]].values
        coords[dimensions["x"]] = da[coordinates["x"]].values
    elif "x" in dimensions:  # 2D unstructured case
        dims.append(dimensions["x"])
        coords.update(da[coordinates["x"]].coords)
    # else: 1D time series case - no spatial dimensions to add

    # Fit model to data - use the actual dimensions of the result
    dot_result = pmodel_da.dot(da)
    # For dot product result, dimensions match input data's spatial dimensions
    spatial_dims = [dim for dim in da.dims if dim != dimensions["time"]]
    result_dims = ["coeff"] + spatial_dims

    # Build coordinates for the result
    result_coords = {"coeff": np.arange(1, n_coeffs + 1)}
    for dim in spatial_dims:
        if dim in da.coords:
            result_coords[dim] = da.coords[dim]

    # Persist the *fit* rather than the detrended series. The fit is a full reduction over
    # the input but is only (coeff, y, x) -- tens of MB -- whereas the detrended series is
    # the size of the input. With the coefficients materialised, the detrend below is a
    # cheap elementwise expression, so the pipeline-level persist of dat_anomaly is the
    # only full-size copy that ever exists (review findings 2.5, 2.6).
    model_fit_da = xr.DataArray(dot_result, dims=result_dims, coords=result_coords).persist()

    # Remove trend and seasonal cycle
    da_detrend = da.drop_vars({"decimal_year"}) - model_da.dot(model_fit_da).astype(np.float32)

    # Force zero mean if requested
    if force_zero_mean:
        da_detrend = da_detrend - da_detrend.mean(dim=dimensions["time"])

    # Create ocean/land mask from first time step
    # Handle both spatial (3D) and time-series (1D) data
    spatial_dims = [dim for dim in ["x", "y"] if dim in dimensions]
    if spatial_dims:
        # Spatial data - create 2D/3D mask
        chunk_dict_mask = {dimensions[dim]: -1 for dim in spatial_dims}
        mask_temp = np.isfinite(da.isel({dimensions["time"]: 0})).chunk(chunk_dict_mask)
        # Drop time-related coordinates to create spatial mask
        vars_to_drop = []
        if "decimal_year" in mask_temp.coords:
            vars_to_drop.append("decimal_year")
        if dimensions["time"] in mask_temp.coords:
            vars_to_drop.append(dimensions["time"])
        if coordinates["time"] in mask_temp.coords:
            vars_to_drop.append(coordinates["time"])
        mask = mask_temp.drop_vars(vars_to_drop) if vars_to_drop else mask_temp
    else:
        # 1D time series - create scalar mask indicating if any finite values exist
        chunk_dict_mask = {}  # Empty for 1D case
        mask = xr.DataArray(np.any(np.isfinite(da.values)), dims=[], attrs={"description": "Time series validity mask"})

    # Initialise output dataset
    data_vars = {"dat_anomaly": da_detrend, "mask": mask}

    # Ensure all original coordinates are preserved in the dataset
    coords_to_preserve = {}
    for coord_name in da.coords:
        if coord_name not in data_vars:  # Don't override data variables
            coords_to_preserve[coord_name] = da.coords[coord_name]

    # Standardise anomalies by temporal variability if requested
    if standardise:

        # Calculate day-of-year standard deviation using cohorts
        std_day = flox.xarray.xarray_reduce(
            da_detrend,
            da_detrend[coordinates["time"]].dt.dayofyear,
            dim=dimensions["time"],
            func="std",
            isbin=False,
            method="cohorts",
            dtype=np.float32,
        )

        # Calculate 30-day rolling standard deviation with annual wrapped padding.
        # Slice/label by the actual number of day-of-year groups: a span with no
        # leap year yields 365 groups, and hardcoding 366 produced a duplicate wrap
        # label and an align error. The common (leap-containing) case is 366, so this
        # is behaviour-preserving there.
        n_doy = std_day.sizes["dayofyear"]
        std_day_wrap = std_day.pad(dayofyear=16, mode="wrap")
        std_rolling = np.sqrt((std_day_wrap**2).rolling(dayofyear=30, center=True).mean()).isel(dayofyear=slice(16, n_doy + 16))

        # Divide anomalies by rolling standard deviation
        # Replace any zeros or extremely small values with NaN to avoid division warnings
        std_rolling_safe = std_rolling.where(std_rolling > 1e-10, np.nan)
        da_detrend = da_detrend.assign_coords(dayofyear=da_detrend[coordinates["time"]].dt.dayofyear)
        da_stn = da_detrend.groupby(dayofyear=xr.groupers.UniqueGrouper(labels=np.arange(1, n_doy + 1))) / std_rolling_safe

        # Drop dayofyear coordinate to avoid merge conflicts
        if "dayofyear" in da_stn.coords:
            da_stn = da_stn.drop_vars("dayofyear")

        # Rechunk data for efficient processing
        chunk_dict_std = chunk_dict_mask.copy()
        chunk_dict_std["dayofyear"] = -1

        da_stn = da_stn.chunk(chunk_dict_mask)
        std_rolling = std_rolling.chunk(chunk_dict_std)

        # Add standardised data to output
        data_vars["dat_stn"] = da_stn
        data_vars["STD"] = std_rolling

    # Build output dataset with metadata
    return xr.Dataset(data_vars=data_vars, coords=coords_to_preserve).drop_vars("decimal_year")
