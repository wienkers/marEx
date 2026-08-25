"""
The full preprocessing chain.

:func:`preprocess_data` runs :func:`marEx.anomaly.compute` and
:func:`marEx.extremes.identify` back to back and returns one dataset ready for
:class:`marEx.tracker`. It is a convenience over the two peers, not a third
implementation: everything it does beyond calling them in order is bookkeeping.

Reach for the peers directly when you want only one of the two stages. In
particular, a caller who wants a climatology and anomalies should call
``marEx.anomaly.compute`` and never meet a threshold parameter at all.
"""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import xarray as xr

from .anomaly.api import _anomaly_core
from .core.compute_mode import Materialiser, create_staging_dir
from .core.dimensions import resolve_dims
from .core.finalise import finalise_dataset, split_large_chunks
from .core.time_axis import SeasonalCycle
from .extremes.api import _effective_window_spatial, _extreme_steps, _extremes_core, _log_extreme_summary
from .logging_config import configure_logging, get_logger

# Get module logger
logger = get_logger(__name__)


def preprocess_data(
    da: xr.DataArray,
    method_anomaly: Literal[
        "detrend_harmonic", "shifting_baseline", "fixed_baseline", "detrend_fixed_baseline"
    ] = "shifting_baseline",
    method_extreme: Literal["global_percentile", "seasonal_percentile"] = "seasonal_percentile",
    threshold_percentile: float = 95,
    window_years: int = 15,
    smooth_days: int = 21,
    window_days: int = 11,
    window_spatial: Optional[int] = None,
    standardise: bool = False,
    detrend_orders: Optional[List[int]] = None,
    force_zero_mean: bool = True,
    reference_period: Optional[Tuple[int, int]] = None,
    method_percentile: Literal["exact", "approximate"] = "approximate",
    precision: float = 0.01,
    max_anomaly: float = 5.0,
    dask_chunks: Optional[Dict[str, int]] = None,
    compute_mode: Literal["persist", "lazy", "streaming"] = "persist",
    scratch_dir: Optional[str] = None,
    validate: bool = True,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    cycle: Optional[SeasonalCycle] = None,
    neighbours: Optional[xr.DataArray] = None,
    cell_areas: Optional[xr.DataArray] = None,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> xr.Dataset:
    """
    Compute anomalies and identify extreme events in one pass.

    Chains :func:`marEx.anomaly.compute` and :func:`marEx.extremes.identify`. See
    those two for the meaning of each parameter -- the names here match theirs,
    except that ``method`` is spelled ``method_anomaly`` and ``method_extreme``
    to distinguish the two stages.

    Parameters
    ----------
    da
        Input time series. Must be Dask-backed.
    method_anomaly
        Baseline method: ``'shifting_baseline'`` (default),
        ``'detrend_harmonic'``, ``'fixed_baseline'``, or
        ``'detrend_fixed_baseline'``.
    method_extreme
        Thresholding method: ``'seasonal_percentile'`` (default) or
        ``'global_percentile'``.
    threshold_percentile
        Percentile defining an extreme.
    window_years, smooth_days
        Rolling-climatology parameters (``shifting_baseline`` only).
    window_days, window_spatial
        Day-of-year and spatial pooling windows (``seasonal_percentile`` only).
    standardise
        Also threshold a standardised series, adding ``dat_stn``, ``STD``,
        ``extreme_events_stn`` and ``thresholds_stn``. Requires
        ``method_anomaly='detrend_harmonic'``.
    detrend_orders, force_zero_mean
        Polynomial detrending controls (harmonic methods only).
    reference_period
        ``(start_year, end_year)`` for the climatology (fixed-baseline methods
        only).
    method_percentile, precision, max_anomaly
        Percentile-estimation controls.
    dask_chunks
        Output chunking. Defaults to ``{"time": 25}``.
    compute_mode, scratch_dir
        Materialisation policy and its staging directory.
    validate
        Check that all unmasked values are finite before computing.
    dimensions, coordinates
        Name mappings. Inferred when omitted.
    cycle
        Within-year axis both stages are resolved on, as a
        :class:`~marEx.SeasonalCycle`. Inferred from the median spacing of the
        time coordinate when omitted: ``dayofyear`` for daily data, ``month``
        for monthly, ``hourofyear`` for sub-daily. Pass one explicitly to
        override the inference on an irregular time axis.
    neighbours, cell_areas
        Optional unstructured-mesh connectivity and cell areas, attached to the
        output for the tracker. They do not constrain the input chunking.
    verbose, quiet
        Logging verbosity overrides.

    Returns
    -------
    xr.Dataset
        ``dat_anomaly``, ``mask``, ``extreme_events`` and ``thresholds``, plus
        the standardised counterparts when ``standardise=True``.

    Examples
    --------
    >>> import xarray as xr
    >>> import marEx
    >>> t2m = xr.open_dataset("t2m.zarr", chunks={"time": 25}).t2m
    >>> ds = marEx.preprocess_data(t2m, threshold_percentile=90)
    >>> events = marEx.tracker(ds.extreme_events, ds.mask, R_fill=8,
    ...                        area_filter_quartile=0.5).run()
    """
    if detrend_orders is None:
        detrend_orders = [1]
    if dask_chunks is None:
        dask_chunks = {"time": 25}

    if verbose is not None or quiet is not None:
        configure_logging(verbose=verbose, quiet=quiet)

    logger.info(f"Starting data preprocessing - Method: {method_anomaly} -> {method_extreme}")
    logger.info(f"Parameters: percentile={threshold_percentile}%, method_percentile={method_percentile}")

    # Resolve the materialisation policy once, before any computation: every persist site
    # in both stages routes through this single object, so it is what distinguishes the
    # three compute modes. Constructing it here also means an invalid mode or a missing
    # scratch_dir fails immediately rather than after a full validation pass.
    staging_dir = create_staging_dir(scratch_dir) if compute_mode == "streaming" and scratch_dir else None
    materialiser = Materialiser(compute_mode, staging_dir)

    with split_large_chunks():
        # Stage 1: anomalies. This anchors dat_anomaly internally, which is what makes
        # the larger-than-memory modes work.
        # The anomaly stage passes the caller's cycle override straight back, so the
        # extremes stage below sees the same one. Neither stage resolves it eagerly:
        # each method resolves at the point it actually groups, so a run that needs no
        # within-year cycle never invokes `infer_cycle`.
        ds, dimensions, coordinates, detrend_orders, cycle = _anomaly_core(
            da,
            method_anomaly,
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
            cycle,
        )

        # Stage 2: extremes on the raw anomaly.
        extremes, thresholds = _extremes_core(
            ds.dat_anomaly,
            method_extreme,
            threshold_percentile,
            window_days,
            window_spatial,
            method_percentile,
            precision,
            max_anomaly,
            dimensions,
            coordinates,
            materialiser,
            cycle=cycle,
        )
        ds["extreme_events"] = extremes
        ds["thresholds"] = thresholds

        # Standardisation is the one option that crosses both stages: the anomaly stage
        # produces dat_stn, and the extremes stage must then run a second time on it.
        # That crossing lives here, in the chainer, so neither package has to know the
        # other exists. The second run needs its own staging label because Materialiser
        # labels are single-owner.
        if standardise:
            logger.info("Processing standardised anomalies for extreme identification")
            # Same anchor as dat_anomaly: it is consumed as many times, so leaving it
            # lazy re-runs the harmonic fit per consumer.
            ds["dat_stn"] = materialiser.stage(ds.dat_stn, "dat_stn")
            extremes_stn, thresholds_stn = _extremes_core(
                ds.dat_stn,
                method_extreme,
                threshold_percentile,
                window_days,
                window_spatial,
                method_percentile,
                precision,
                max_anomaly,
                dimensions,
                coordinates,
                materialiser,
                threshold_label="thresholds_stn",
                cycle=cycle,
            )
            ds["extreme_events_stn"] = extremes_stn
            ds["thresholds_stn"] = thresholds_stn

        # Optional spatial metadata for the tracker. These are attached to the output
        # and independently rechunked; they do not constrain the input chunking.
        if neighbours is not None:
            logger.debug("Adding neighbour connectivity data")
            chunk_dict = {dim: -1 for dim in neighbours.dims}
            ds["neighbours"] = neighbours.astype(np.int32).chunk(chunk_dict)
            if "nv" in neighbours.dims:
                ds = ds.assign_coords(nv=neighbours.nv)

        if cell_areas is not None:
            logger.debug("Adding cell area data")
            chunk_dict = {dim: -1 for dim in cell_areas.dims}
            ds["cell_areas"] = cell_areas.astype(np.float32).chunk(chunk_dict)

        # Merge the extremes stage's metadata onto the anomaly stage's. Each stage
        # appends its own steps, so the chained list reads in execution order.
        effective_window_spatial = _effective_window_spatial(method_extreme, window_spatial, dimensions, ds)
        ds.attrs.update({"method_extreme": method_extreme, "threshold_percentile": threshold_percentile})
        ds.attrs["preprocessing_steps"] = list(ds.attrs.get("preprocessing_steps", [])) + _extreme_steps(
            method_extreme, window_days, effective_window_spatial
        )
        if method_extreme == "seasonal_percentile":
            ds.attrs.update({"window_days": window_days})
            if effective_window_spatial is not None:
                ds.attrs.update({"window_spatial": effective_window_spatial})
        ds.attrs.update({"method_percentile": method_percentile, "precision": precision, "max_anomaly": max_anomaly})

        ds = finalise_dataset(
            ds,
            dimensions,
            coordinates,
            dask_chunks,
            materialiser,
            staging_dir,
            extra_dims=resolve_dims(da, dimensions, coordinates).extra,
        )

    _log_extreme_summary(ds, materialiser)
    return ds
