"""
Public entry point for anomaly computation.

:func:`compute` turns a raw time series into anomalies about a climatological
baseline. It is a peer of :func:`marEx.extremes.identify`, not a stage of it:
nothing here knows what a threshold is, and the returned dataset is a complete,
saveable product for callers who want a climatology and anomalies and nothing
else.

The materialisation policy (``compute_mode``) lives here rather than in the
caller. Anchoring the anomaly is what makes the larger-than-memory modes work,
and a consumer that anchored it from outside would leave every lazy expression
built on the original graph.
"""

import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import xarray as xr
from dask.base import is_dask_collection

from ..core.compute_mode import Materialiser, create_staging_dir
from ..core.dimensions import resolve_dims
from ..core.finalise import finalise_dataset, split_large_chunks
from ..core.validation import _infer_dims_coords, _validate_data_values
from ..exceptions import ConfigurationError, create_data_validation_error
from ..logging_config import configure_logging, get_logger, log_dask_info, log_memory_usage, log_timing
from .base import compute_normalised_anomaly

# Get module logger
logger = get_logger(__name__)

METHODS = ("shifting_baseline", "detrend_harmonic", "fixed_baseline", "detrend_fixed_baseline")


def _anomaly_steps(
    method: str,
    standardise: bool,
    detrend_orders: List[int],
    window_years: int,
    smooth_days: int,
    reference_period: Optional[Tuple[int, int]],
) -> List[str]:
    """Describe the anomaly stage for ``ds.attrs["preprocessing_steps"]``."""
    steps: List[str] = []

    if method == "detrend_harmonic":
        steps.append(f"Removed polynomial trend orders={detrend_orders} & seasonal cycle")
        if standardise:
            steps.append("Normalised by 30-day rolling STD")
    elif method == "shifting_baseline":
        steps.append(f"Rolling climatology using {window_years} years")
        steps.append(f"Smoothed with {smooth_days}-day window")
    elif method == "fixed_baseline":
        if reference_period is not None:
            steps.append(f"Daily climatology computed from {reference_period[0]}-{reference_period[1]}")
        else:
            steps.append("Daily climatology computed from full time series")
    elif method == "detrend_fixed_baseline":
        steps.append(f"Removed polynomial trend orders={detrend_orders}")
        if reference_period is not None:
            steps.append(f"Daily climatology computed from detrended data ({reference_period[0]}-{reference_period[1]})")
        else:
            steps.append("Daily climatology computed from detrended data")

    return steps


def _anomaly_attrs(
    method: str,
    standardise: bool,
    detrend_orders: List[int],
    force_zero_mean: bool,
    window_years: int,
    smooth_days: int,
    reference_period: Optional[Tuple[int, int]],
) -> Dict[str, object]:
    """Method-specific attributes recorded on the anomaly dataset."""
    if method == "detrend_harmonic":
        return {
            "detrend_orders": detrend_orders,
            "force_zero_mean": force_zero_mean,
            "standardise": standardise,
        }
    if method == "shifting_baseline":
        return {
            "window_years": window_years,
            "smooth_days": smooth_days,
        }
    if method == "fixed_baseline":
        return {"reference_period": list(reference_period)} if reference_period is not None else {}
    if method == "detrend_fixed_baseline":
        attrs: Dict[str, object] = {
            "detrend_orders": detrend_orders,
            "force_zero_mean": force_zero_mean,
        }
        if reference_period is not None:
            attrs["reference_period"] = list(reference_period)
        return attrs
    return {}


def _anomaly_core(
    da: xr.DataArray,
    method: str,
    window_years: int,
    smooth_days: int,
    detrend_orders: Optional[List[int]],
    force_zero_mean: bool,
    standardise: bool,
    reference_period: Optional[Tuple[int, int]],
    validate: bool,
    dimensions: Optional[Dict[str, str]],
    coordinates: Optional[Dict[str, str]],
    materialiser: Materialiser,
):
    """
    Compute anomalies, up to but not including output finalisation.

    Shared by :func:`compute` and by :func:`marEx.preprocess_data`, which needs
    the intermediate dataset before the extremes stage runs. Returns the dataset,
    the resolved dimension/coordinate mappings, and the resolved detrend orders.
    """
    if detrend_orders is None:
        detrend_orders = [1]

    if method not in METHODS:
        raise ConfigurationError(
            f"Unknown anomaly method '{method}'",
            details=f"Supported methods are: {', '.join(METHODS)}",
            suggestions=[f"Use method='{METHODS[0]}' (the default)"],
        )

    logger.info(f"Computing anomalies - method: {method}")
    logger.debug(
        f"Anomaly parameters: window_years={window_years}, smooth_days={smooth_days}, "
        + f"standardise={standardise}, detrend_orders={detrend_orders}, force_zero_mean={force_zero_mean}"
    )

    log_dask_info(logger, da, "Input data")
    log_memory_usage(logger, "Initial memory state")

    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)

    # Resolve the dimension contract: which axes are horizontal, and which are extra
    # (depth, level, member) and carried through as broadcast axes. Validates an
    # explicit dimensions["z"] against what the data actually has, so a typo is
    # reported here rather than ignored.
    dims = resolve_dims(da, dimensions, coordinates)
    if dims.extra:
        logger.info(f"Extra (non-horizontal) dimensions detected and carried through: {list(dims.extra)}")

    # Check if input data is dask-backed
    if not is_dask_collection(da.data):
        logger.error("Input DataArray is not Dask-backed - preprocessing requires chunked data")
        raise create_data_validation_error(
            "Input DataArray must be Dask-backed",
            details="Preprocessing requires chunked data for efficient computation",
            suggestions=[
                "Convert to Dask array: da = da.chunk({'time': 30})",
                "Load with chunking: xr.open_dataset('file.nc', chunks={'time': 30})",
            ],
            data_info={"data_type": type(da.data).__name__, "shape": da.shape},
        )

    # Validate reference_period before triggering any computation
    if reference_period is not None and method not in ("fixed_baseline", "detrend_fixed_baseline"):
        raise ConfigurationError(
            f"reference_period is not supported for method='{method}'",
            details="reference_period is only applicable to 'fixed_baseline' and 'detrend_fixed_baseline' methods",
            suggestions=[
                "Remove the reference_period parameter, or",
                "Use method='fixed_baseline' or 'detrend_fixed_baseline'",
            ],
        )

    # Standardisation is produced by the harmonic fit's rolling STD, so asking for it
    # with any other method used to be a silent no-op. Fail instead: a caller who asked
    # for `dat_stn` and got a dataset without it has been told nothing.
    if standardise and method != "detrend_harmonic":
        raise ConfigurationError(
            f"standardise=True is not supported for method='{method}'",
            details="Standardisation uses the rolling STD produced by the harmonic fit, "
            "so it is only available for method='detrend_harmonic'",
            suggestions=[
                "Use method='detrend_harmonic', or",
                "Remove standardise=True",
            ],
        )

    # Validate that all unmasked data is valid (finite values only)
    if validate:
        logger.debug("Validating data values for NaN/infinite values")
        _validate_data_values(da, dimensions)
    else:
        logger.debug("Skipping input finite-value validation (validate=False)")

    with log_timing(
        logger,
        f"Anomaly computation using {method} method",
        log_memory=True,
        show_progress=True,
    ):
        ds = compute_normalised_anomaly(
            da.astype(np.float32),
            method,
            dimensions,
            coordinates,
            window_years,
            smooth_days,
            standardise,
            detrend_orders,
            force_zero_mean,
            reference_period,
            materialiser=materialiser,
        )
        log_memory_usage(logger, "After anomaly computation", logging.DEBUG)

    # For shifting baseline, remove first window_years years (insufficient climatology data)
    if method == "shifting_baseline":
        min_year = int(ds[coordinates["time"]].dt.year.min().values.item())
        max_year = int(ds[coordinates["time"]].dt.year.max().values.item())
        total_years = max_year - min_year + 1

        logger.info(f"Shifting baseline data validation: {total_years} years available ({min_year}-{max_year})")

        if total_years < window_years:
            logger.error(f"Insufficient data: {total_years} years < {window_years} required")
            raise create_data_validation_error(
                "Insufficient data for shifting_baseline method",
                details=f"Dataset spans {total_years} years but requires at least {window_years} years",
                suggestions=[
                    "Use more years of data to meet minimum requirement",
                    f"Reduce window_years parameter (currently {window_years})",
                    "Consider using detrend_fixed_baseline or detrend_harmonic method instead",
                ],
                data_info={
                    "available_years": int(total_years),
                    "required_years": int(window_years),
                },
            )

        start_year = int(min_year + window_years)
        logger.info(f"Trimming data to start from {start_year} (removing first {window_years} years)")
        time_sel = (ds[coordinates["time"]].dt.year >= start_year).compute()
        ds = ds.isel({dimensions["time"]: time_sel})

    # Anchor the anomaly exactly once, here, before anything consumes it. It is read two
    # to three times downstream, so without an anchor the whole anomaly graph re-executes
    # each time -- for the default shifting_baseline that is the entire 15-year rolling
    # climatology. This anchor is what makes `compute_mode` mean anything, which is why
    # it belongs to this module rather than to whoever calls it.
    ds["dat_anomaly"] = materialiser.stage(ds.dat_anomaly, "dat_anomaly")

    ds.attrs.update({"method_anomaly": method})
    ds.attrs["preprocessing_steps"] = _anomaly_steps(
        method, standardise, detrend_orders, window_years, smooth_days, reference_period
    )
    ds.attrs.update(
        _anomaly_attrs(method, standardise, detrend_orders, force_zero_mean, window_years, smooth_days, reference_period)
    )

    return ds, dimensions, coordinates, detrend_orders


def compute(
    da: xr.DataArray,
    method: Literal["shifting_baseline", "detrend_harmonic", "fixed_baseline", "detrend_fixed_baseline"] = "shifting_baseline",
    *,
    window_years: int = 15,
    smooth_days: int = 21,
    detrend_orders: Optional[List[int]] = None,
    force_zero_mean: bool = True,
    standardise: bool = False,
    reference_period: Optional[Tuple[int, int]] = None,
    dask_chunks: Optional[Dict[str, int]] = None,
    compute_mode: Literal["persist", "lazy", "streaming"] = "persist",
    scratch_dir: Optional[str] = None,
    validate: bool = True,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> xr.Dataset:
    """
    Compute anomalies about a climatological baseline.

    This is a complete product in its own right. Use it when you want a smoothed
    climatology, detrended anomalies, or standardised anomalies, and have no
    interest in extreme-event detection -- there is no threshold parameter here.

    Works on any gridded or unstructured field with a time dimension: ocean,
    atmosphere, land surface, or biogeochemistry.

    Parameters
    ----------
    da
        Input time series. Must be Dask-backed.
    method
        Baseline method.

        * ``'shifting_baseline'`` (default) -- rolling climatology over the
          preceding ``window_years`` years. Most defensible, but shortens the
          series by ``window_years``.
        * ``'detrend_harmonic'`` -- polynomial trend plus harmonic seasonal
          cycle. Efficient, keeps the full series, biases the statistics.
        * ``'fixed_baseline'`` -- daily climatology over the whole series. Does
          *not* remove a climate trend.
        * ``'detrend_fixed_baseline'`` -- polynomial detrending then a fixed
          daily climatology. Keeps the full series but does not account for
          trends in the timing of seasonal transitions.
    window_years
        Number of preceding years in the rolling climatology
        (``shifting_baseline`` only).
    smooth_days
        Width of the centred smoothing window applied to the climatology
        (``shifting_baseline`` only).
    detrend_orders
        Polynomial orders to remove, e.g. ``[1]`` for a linear trend
        (harmonic methods only). Defaults to ``[1]``.
    force_zero_mean
        Force the detrended anomaly to have zero mean (harmonic methods only).
    standardise
        Also divide by a 30-day rolling standard deviation, adding ``dat_stn``
        and ``STD`` to the output. Requires ``method='detrend_harmonic'``.
    reference_period
        ``(start_year, end_year)`` restricting the climatology to a fixed
        period. Anomalies are still returned for the full series. Supported for
        the two fixed-baseline methods only.
    dask_chunks
        Output chunking. Only the time entry is honoured; spatial dimensions are
        made whole. Defaults to ``{"time": 25}``.
    compute_mode
        Materialisation policy. ``'persist'`` holds the anomaly in cluster
        memory (fastest, needs it to fit); ``'lazy'`` holds nothing and accepts
        recompute; ``'streaming'`` stages to Zarr under ``scratch_dir``, which is
        what makes genuinely larger-than-memory input work.
    scratch_dir
        Staging directory, required by ``compute_mode='streaming'``. The
        directory outlives this call, since the returned dataset reads from it
        lazily. Write your output, then call :func:`marEx.clear_staging`.
    validate
        Check that all unmasked values are finite before computing.
    dimensions, coordinates
        Name mappings, e.g. ``{"time": "time", "y": "lat", "x": "lon"}``.
        Inferred when omitted.
    verbose, quiet
        Logging verbosity overrides.

    Returns
    -------
    xr.Dataset
        ``dat_anomaly`` and ``mask``, plus ``dat_stn`` and ``STD`` when
        ``standardise=True``.

    Examples
    --------
    >>> import xarray as xr
    >>> import marEx
    >>> t2m = xr.open_dataset("t2m.zarr", chunks={"time": 25}).t2m
    >>> ds = marEx.anomaly.compute(t2m, method="shifting_baseline")
    >>> ds.dat_anomaly
    """
    if dask_chunks is None:
        dask_chunks = {"time": 25}

    if verbose is not None or quiet is not None:
        configure_logging(verbose=verbose, quiet=quiet)

    staging_dir = create_staging_dir(scratch_dir) if compute_mode == "streaming" and scratch_dir else None
    materialiser = Materialiser(compute_mode, staging_dir)

    with split_large_chunks():
        ds, dimensions, coordinates, _ = _anomaly_core(
            da,
            method,
            window_years,
            smooth_days,
            detrend_orders,
            force_zero_mean,
            standardise,
            reference_period,
            validate,
            dimensions,
            coordinates,
            materialiser,
        )
        ds = finalise_dataset(
            ds,
            dimensions,
            coordinates,
            dask_chunks,
            materialiser,
            staging_dir,
            extra_dims=resolve_dims(da, dimensions, coordinates).extra,
        )

    logger.info("Anomaly computation completed successfully")
    return ds
