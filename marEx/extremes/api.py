"""
Public entry point for extreme-event identification.

:func:`identify` turns anomalies into a boolean extreme-event field plus the
percentile thresholds that defined it. It is a peer of
:func:`marEx.anomaly.compute`, not a stage of it: it accepts anomalies from
anywhere, including a dataset this package never produced.

Nothing here knows how the anomalies were made, and nothing here mentions
standardisation -- a caller wanting thresholds on a standardised series simply
calls this function again with that series.
"""

import logging
from typing import Dict, List, Literal, Optional, Union

import xarray as xr

from ..core.compute_mode import Materialiser, create_staging_dir
from ..core.finalise import finalise_dataset, split_large_chunks
from ..core.validation import _infer_dims_coords
from ..exceptions import ConfigurationError, create_data_validation_error
from ..logging_config import configure_logging, get_logger, log_memory_usage, log_timing
from .base import identify_extremes

# Get module logger
logger = get_logger(__name__)

METHODS = ("seasonal_percentile", "global_percentile")


def _extreme_steps(
    method: str,
    window_days: int,
    window_spatial: Optional[int],
) -> List[str]:
    """Describe the extremes stage for ``ds.attrs["preprocessing_steps"]``."""
    steps: List[str] = []

    if method == "global_percentile":
        steps.append("Global percentile threshold applied to all days")
    elif method == "seasonal_percentile":
        if window_spatial is not None:
            steps.append(f"Day-of-year thresholds with {window_days} day window & {window_spatial} spatial neighbours")
        else:
            steps.append(f"Day-of-year thresholds with {window_days} day window")

    return steps


def _effective_window_spatial(
    method: str,
    window_spatial: Optional[int],
    dimensions: Dict[str, str],
    ds_or_da,
) -> Optional[int]:
    """
    Resolve the spatial window actually used.

    ``identify_extremes`` silently defaults a gridded seasonal run to a 5x5 window
    when the caller leaves it ``None``, so the recorded attributes and steps must
    reflect that rather than the raw ``None``. Mirrors the default in
    :mod:`marEx.extremes.base`.
    """
    if method == "seasonal_percentile" and window_spatial is None:
        if "y" in dimensions and dimensions["y"] in ds_or_da.dims:
            return 5
    return window_spatial


def _extremes_core(
    anomalies: xr.DataArray,
    method: str,
    threshold_percentile: float,
    window_days: int,
    window_spatial: Optional[int],
    method_percentile: str,
    precision: float,
    max_anomaly: float,
    dimensions: Dict[str, str],
    coordinates: Dict[str, str],
    materialiser: Materialiser,
    threshold_label: str = "thresholds",
):
    """
    Identify extremes, up to but not including output finalisation.

    Shared by :func:`identify` and by :func:`marEx.preprocess_data`.

    ``threshold_label`` names the materialiser staging slot. It is a parameter
    rather than a constant because a caller may run this stage twice on two
    different series (raw and standardised anomalies), and ``Materialiser``
    labels are single-owner: reusing one would raise.
    """
    if method not in METHODS:
        raise ConfigurationError(
            f"Unknown extreme method '{method}'",
            details=f"Supported methods are: {', '.join(METHODS)}",
            suggestions=[f"Use method='{METHODS[0]}' (the default)"],
        )

    with log_timing(
        logger,
        f"Extreme event identification using {method} method",
        log_memory=True,
        show_progress=True,
    ):
        logger.debug(
            f"Identifying extremes with parameters: method={method}, "
            f"percentile={threshold_percentile}%, method_percentile={method_percentile}"
        )
        extremes, thresholds = identify_extremes(
            anomalies,
            method,
            threshold_percentile,
            dimensions,
            coordinates,
            window_days,
            window_spatial,
            method_percentile,
            precision,
            max_anomaly,
            materialiser=materialiser,
            threshold_label=threshold_label,
        )
        log_memory_usage(logger, "After extreme identification", logging.DEBUG)

    # `thresholds` was already anchored inside the method module (before the comparison
    # that builds `extremes` was constructed on top of it), so this pin only covers
    # `extremes` itself in persist mode.
    return materialiser.pin(extremes, thresholds)


def _log_extreme_summary(ds: xr.Dataset, materialiser: Materialiser) -> None:
    """
    Report the event count, but only when it is free to do so.

    This sum is a pass over the whole field. Only pay for it when the INFO line
    will actually be emitted AND the field is already materialised -- in
    lazy/streaming mode it would execute the entire graph, silently defeating the
    point of the mode for any INFO-level user.
    """
    if logger.isEnabledFor(logging.INFO) and materialiser.mode == "persist":
        extreme_count = ds.extreme_events.sum()
        if hasattr(extreme_count, "compute"):
            extreme_count = extreme_count.compute()
        logger.info(f"Extreme identification completed - {extreme_count} extreme events identified")
    else:
        logger.info(f"Extreme identification graph constructed (compute_mode='{materialiser.mode}')")


def identify(
    data: Union[xr.DataArray, xr.Dataset],
    method: Literal["seasonal_percentile", "global_percentile"] = "seasonal_percentile",
    *,
    threshold_percentile: float = 95,
    window_days: int = 11,
    window_spatial: Optional[int] = None,
    method_percentile: Literal["exact", "approximate"] = "approximate",
    precision: float = 0.01,
    max_anomaly: float = 5.0,
    dask_chunks: Optional[Dict[str, int]] = None,
    compute_mode: Literal["persist", "lazy", "streaming"] = "persist",
    scratch_dir: Optional[str] = None,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> xr.Dataset:
    """
    Identify extreme events exceeding a percentile threshold.

    Parameters
    ----------
    data
        Anomalies to threshold. A ``DataArray``, or a ``Dataset`` carrying a
        ``dat_anomaly`` variable -- so that the output of
        :func:`marEx.anomaly.compute` composes directly.
    method
        Thresholding method.

        * ``'seasonal_percentile'`` (default) -- day-of-year resolved
          thresholds computed over a rolling day window, optionally pooled over
          spatial neighbours. Follows the Hobday et al. (2016) definition.
        * ``'global_percentile'`` -- one constant-in-time threshold per cell.
    threshold_percentile
        Percentile defining an extreme, e.g. ``95``.
    window_days
        Width of the rolling day-of-year window
        (``seasonal_percentile`` only).
    window_spatial
        Width of the spatial pooling window (``seasonal_percentile`` on gridded
        data only). Defaults to 5 on gridded input, and is unavailable on
        unstructured meshes.
    method_percentile
        ``'approximate'`` (default) uses a histogram-based quantile, which is
        what allows the reduction to stream. ``'exact'`` computes a true
        quantile and needs the full series resident per cell.
    precision, max_anomaly
        Histogram bin width and range for ``method_percentile='approximate'``.
        The defaults suit anomalies of order a few kelvin; a variable on a very
        different scale needs them set explicitly.
    dask_chunks
        Output chunking. Defaults to ``{"time": 25}``.
    compute_mode
        Materialisation policy: ``'persist'``, ``'lazy'`` or ``'streaming'``.
    scratch_dir
        Staging directory, required by ``compute_mode='streaming'``.
    dimensions, coordinates
        Name mappings. Inferred when omitted.
    verbose, quiet
        Logging verbosity overrides.

    Returns
    -------
    xr.Dataset
        ``extreme_events`` (boolean) and ``thresholds``. When given a Dataset,
        its existing variables are carried through.

    Examples
    --------
    >>> import marEx
    >>> anomalies = marEx.anomaly.compute(t2m)
    >>> events = marEx.extremes.identify(anomalies, threshold_percentile=90)
    """
    if dask_chunks is None:
        dask_chunks = {"time": 25}

    if verbose is not None or quiet is not None:
        configure_logging(verbose=verbose, quiet=quiet)

    if isinstance(data, xr.Dataset):
        if "dat_anomaly" not in data.data_vars:
            raise create_data_validation_error(
                "Dataset has no 'dat_anomaly' variable",
                details="identify() accepts a DataArray of anomalies, or a Dataset carrying 'dat_anomaly'",
                suggestions=[
                    "Pass the anomaly DataArray directly, or",
                    "Use the output of marEx.anomaly.compute()",
                ],
                data_info={"available_variables": list(data.data_vars)},
            )
        ds = data
        anomalies = data.dat_anomaly
    else:
        anomalies = data
        ds = xr.Dataset({"dat_anomaly": anomalies})

    logger.info(f"Identifying extremes - method: {method}, percentile: {threshold_percentile}%")

    dimensions, coordinates = _infer_dims_coords(anomalies, dimensions, coordinates)

    staging_dir = create_staging_dir(scratch_dir) if compute_mode == "streaming" and scratch_dir else None
    materialiser = Materialiser(compute_mode, staging_dir)

    with split_large_chunks():
        extremes, thresholds = _extremes_core(
            anomalies,
            method,
            threshold_percentile,
            window_days,
            window_spatial,
            method_percentile,
            precision,
            max_anomaly,
            dimensions,
            coordinates,
            materialiser,
        )
        ds = ds.copy()
        ds["extreme_events"] = extremes
        ds["thresholds"] = thresholds

        effective_window_spatial = _effective_window_spatial(method, window_spatial, dimensions, ds)
        ds.attrs.update({"method_extreme": method, "threshold_percentile": threshold_percentile})
        ds.attrs["preprocessing_steps"] = list(ds.attrs.get("preprocessing_steps", [])) + _extreme_steps(
            method, window_days, effective_window_spatial
        )
        if method == "seasonal_percentile":
            ds.attrs.update({"window_days": window_days})
            if effective_window_spatial is not None:
                ds.attrs.update({"window_spatial": effective_window_spatial})
        ds.attrs.update({"method_percentile": method_percentile, "precision": precision, "max_anomaly": max_anomaly})

        ds = finalise_dataset(ds, dimensions, coordinates, dask_chunks, materialiser, staging_dir)

    # After finalisation, so that in persist mode the count reads the materialised
    # field rather than re-walking the graph that built it.
    _log_extreme_summary(ds, materialiser)
    return ds
