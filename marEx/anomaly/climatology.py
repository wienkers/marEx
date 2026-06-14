"""
Rolling-climatology computation for the shifting-baseline anomaly method.

Provides :func:`rolling_climatology` and :func:`smoothed_rolling_climatology`,
both public entry points re-exported at the marEx top level. These use flox
cohorts to efficiently compute multi-year rolling day-of-year climatologies.
"""

from typing import Dict, Optional

import flox.xarray
import numpy as np
import pandas as pd
import xarray as xr
from dask import persist

from ..core.dimensions import spatial_dims, tile_spatial_chunks
from ..core.time_axis import SeasonalCycle, resolve_cycle
from ..core.validation import _infer_dims_coords
from ..exceptions import ConfigurationError
from ..logging_config import get_logger

# Get module logger
logger = get_logger(__name__)


def rolling_climatology(
    da: xr.DataArray,
    window_years: int = 15,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    cycle: Optional[SeasonalCycle] = None,
) -> xr.DataArray:
    """
    Compute rolling climatology efficiently using flox cohorts.
    Uses the previous `window_years` years of data and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    window_years : int, default=15
        Number of years to include in each climatology window
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data
    cycle : SeasonalCycle, optional
        Within-year axis the climatology is resolved on. Inferred from the time
        coordinate's cadence when omitted (``dayofyear`` for daily data).

    Returns
    -------
    xarray.DataArray
        Rolling climatology with same shape as input data

    Examples
    --------
    Basic rolling climatology computation:

    >>> import xarray as xr
    >>> import marEx
    >>>
    >>> # Load 20 years of SST data
    >>> sst = xr.open_dataset('sst_data.nc', chunks={}).sst.chunk({'time': 30})
    >>>
    >>> # Compute 15-year rolling climatology
    >>> climatology = marEx.rolling_climatology(sst, window_years=15)
    >>> print(climatology.shape)
    (7305, 180, 360)  # Same as input
    >>>
    >>> # First 15 years will be NaN (insufficient history)
    >>> print(f"NaN values in first year: {climatology.isel(time=slice(0, 365)).isnull().all().compute()}")
    True

    Shorter window for datasets with limited time span:

    >>> # For datasets with only 10 years, use shorter window
    >>> short_climatology = marEx.rolling_climatology(
    ...     sst, window_years=5
    ... )
    >>> # First 5 years will be NaN instead of 15

    Processing unstructured data:

    >>> # ICON ocean model data
    >>> icon_sst = xr.open_dataset('icon_sst.nc', chunks={}).to.chunk({'time': 25})
    >>> icon_climatology = marEx.rolling_climatology(
    ...     icon_sst,
    ...     dimensions={"time": "time", "x": "ncells"}
    ...     coordinates={"time": "time", "x": "lon", "y": "lat"}
    ... )
    >>> print(icon_climatology.dims)
    Frozen({'time': 7305, 'ncells': 83886})

    Comparing with fixed climatology:

    >>> # Fixed climatology (traditional approach)
    >>> fixed_clim = sst.groupby(sst.time.dt.dayofyear).mean()
    >>>
    >>> # Rolling climatology (adaptive approach)
    >>> rolling_clim = marEx.rolling_climatology(sst)
    >>>
    >>> # Rolling climatology adapts to climate change
    >>> clim_2000 = rolling_clim.sel(time='2000').mean()
    >>> clim_2020 = rolling_clim.sel(time='2020').mean()
    >>> print(f"Climate change signal: {(clim_2020 - clim_2000).compute():.3f} °C")

    Memory considerations for large datasets:

    >>> # Ensure appropriate chunking for memory efficiency
    >>> large_sst = sst.chunk({'time': 30, 'lat': 45, 'lon': 90})
    >>> large_climatology = marEx.rolling_climatology(large_sst)
    >>> # Output maintains input chunking structure
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)
    timedim = dimensions["time"]
    time_coord = coordinates["time"]
    cycle = resolve_cycle(da, time_coord, cycle)
    cycle_dim = cycle.index_name
    original_chunk_dict = dict(zip(da.dims, da.chunks))

    # Add temporal coordinates
    years = da[time_coord].dt.year
    doys = cycle.index_of(da[time_coord])
    da = da.assign_coords({"year": years, cycle_dim: doys})

    # Get temporal bounds
    years, doys = persist(years, doys)
    year_vals = years.values
    doy_vals = doys.values
    unique_years = np.unique(year_vals)
    min_year = int(unique_years.min().item())

    # Create long-form grouping variables
    # For each time point, determine which target years it contributes to
    contributing_time_indices = []
    contributing_target_years = []
    contributing_cycle_indices = []

    for t_idx, (year_val, doy_val) in enumerate(zip(year_vals, doy_vals)):
        # Convert numpy scalars to Python ints to avoid dtype issues
        year_val = int(year_val)
        doy_val = int(doy_val)

        # Find target years this time point contributes to
        # A time point from year Y contributes to target years where:
        # target_year - window_years <= Y < target_year
        # Which means: Y < target_year <= Y + window_years
        candidate_targets = unique_years[(unique_years > year_val) & (unique_years <= year_val + window_years)]

        # Only include target years that have sufficient history
        valid_targets = candidate_targets[candidate_targets >= min_year + window_years]

        # Add entries for each valid target year
        n_targets = len(valid_targets)
        contributing_time_indices.extend([t_idx] * n_targets)
        contributing_target_years.extend(valid_targets.tolist())
        contributing_cycle_indices.extend([doy_val] * n_targets)

    # Convert to numpy arrays with explicit dtypes
    time_indices = np.array(contributing_time_indices, dtype=np.int32)
    target_year_groups = np.array(contributing_target_years, dtype=np.int32)
    cycle_groups = np.array(contributing_cycle_indices, dtype=np.int32)

    # Bound the long-form expansion before building it.
    #
    # The `isel` below materialises roughly `window_years` times the input along time,
    # and the flox reduction after it writes an `(n_years, cycle)` block per spatial
    # cell, which `.chunk({"dayofyear": -1})` then forces whole. Neither side had a
    # budget: on a field left spatially whole one task is the entire array (174 GB on
    # the ICON mesh, where it thrashed rather than failing), and an extra dimension
    # such as depth multiplies that directly.
    #
    # Tiling the spatial dims -- every one of them, so depth is tiled exactly like
    # latitude -- bounds both sides. It is a pure rechunk of SPATIAL dims only: the
    # reduction is independent per cell and each cell's reduced axis stays whole inside
    # its tile, so flox's ordering along that axis is unchanged and the result is
    # bit-identical (verified against both goldens and pinned by
    # tests/test_climatology_tiling.py).
    #
    # The time chunking is deliberately untouched. Changing it WOULD move values: the
    # smoothing in `smoothed_rolling_climatology` runs through bottleneck's move_mean,
    # whose running sum restarts at every dask block boundary. `original_chunk_dict` was
    # captured before this point, so the restore at the end of this function returns the
    # caller's own layout regardless.
    spatial_tile = tile_spatial_chunks(
        da,
        spatial_dims(da, dimensions),
        input_elements_per_cell=len(time_indices),
        output_elements_per_cell=len(unique_years) * cycle.length,
    )
    if spatial_tile:
        da = da.chunk(spatial_tile)

    # Create long-form dataset by selecting the contributing time points
    long_form_data = da.isel({timedim: time_indices})

    # Create a new time dimension for the long-form data
    long_timedim = f"{timedim}_contrib"
    long_form_data = long_form_data.rename({timedim: long_timedim})

    # Convert grouping arrays to DataArrays with the correct dimension
    target_year_da = xr.DataArray(target_year_groups, dims=[long_timedim], name="target_year")
    cycle_da = xr.DataArray(cycle_groups, dims=[long_timedim], name=cycle_dim)

    # Use flox with both grouping variables to compute climatologies
    climatologies = flox.xarray.xarray_reduce(
        long_form_data,
        target_year_da,
        cycle_da,
        dim=long_timedim,
        func="nanmean",
        expected_groups=(unique_years, cycle.labels),
        isbin=(False, False),
        dtype=np.float32,
        fill_value=np.nan,
    ).chunk({cycle_dim: -1})

    # Create index arrays for final mapping
    year_to_idx = pd.Series(range(len(unique_years)), index=unique_years)
    year_indices = year_to_idx[year_vals].values

    # Select appropriate climatology for each time point
    result = climatologies.isel(
        {
            "target_year": xr.DataArray(year_indices, dims=[timedim]),
            cycle_dim: xr.DataArray(doy_vals - 1, dims=[timedim]),
        }
    )

    # Clean up dimensions and coordinates
    result = result.drop_vars(["target_year", cycle_dim])

    return result.chunk(original_chunk_dict)


def smoothed_rolling_climatology(
    da: xr.DataArray,
    window_years: int = 15,
    smooth_days: int = 21,
    dimensions: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, str]] = None,
    cycle: Optional[SeasonalCycle] = None,
) -> xr.DataArray:
    """
    Compute a smoothed rolling climatology using the previous `window_years` years of data
    and reassemble it to match the original data structure.
    Years without enough previous data will be filled with NaN.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with time coordinate
    window_years : int, default=15
        Number of years to include in each climatology window
    smooth_days : int, default=21
        Number of days for temporal smoothing window
    dimensions : dict, optional
        Mapping of dimensions to names in the data
    coordinates : dict, optional
        Mapping of coordinates to names in the data

    Returns
    -------
    xarray.DataArray
        Smoothed rolling climatology with same shape as input data

    Examples
    --------
    Basic smoothed rolling climatology:

    >>> import xarray as xr
    >>> import marEx
    >>>
    >>> # Load SST data
    >>> sst = xr.open_dataset('sst_data.nc', chunks={}).sst.chunk({'time': 30})
    >>>
    >>> # Compute smoothed rolling climatology
    >>> smooth_clim = marEx.smoothed_rolling_climatology(
    ...     sst,
    ...     window_years=15,
    ...     smooth_days=21
    ... )
    >>> print(smooth_clim.shape)
    (7305, 180, 360)

    Comparing different smoothing windows:

    >>> # Short smoothing - more day-to-day variability
    >>> clim_short = marEx.smoothed_rolling_climatology(
    ...     sst, smooth_days=7
    ... )
    >>>
    >>> # Long smoothing - smoother seasonal cycle
    >>> clim_long = marEx.smoothed_rolling_climatology(
    ...     sst, smooth_days=61
    ... )
    >>>
    >>> # Compare variability
    >>> var_short = clim_short.std(dim='time').mean().compute()
    >>> var_long = clim_long.std(dim='time').mean().compute()
    >>> print(f"Variability: short={var_short:.3f}, long={var_long:.3f}")

    Climatology for anomaly computation:

    >>> # Compute smoothed climatology then anomalies
    >>> climatology = marEx.smoothed_rolling_climatology(sst)
    >>> anomalies = sst - climatology
    >>>
    >>> # Check that anomalies have reasonable properties
    >>> print(f"Anomaly mean: {anomalies.mean().compute():.6f}")
    >>> print(f"Anomaly std: {anomalies.std().compute():.3f}")

    Unstructured data processing:

    >>> # ICON ocean data
    >>> icon_sst = xr.open_dataset('icon_sst.nc', chunks={}).to.chunk({'time': 25})
    >>> icon_smooth_clim = marEx.smoothed_rolling_climatology(
    ...     icon_sst,
    ...     dimensions={"time": "time", "x": "ncells"},
    ...     coordinates={"time": "time", "x": "lon", "y": "lat"},
    ...     window_years=10,
    ...     smooth_days=31
    ... )

    Effect of smoothing on seasonal cycle:

    >>> # Raw rolling climatology (no temporal smoothing)
    >>> raw_clim = marEx.rolling_climatology(sst, window_years=15)
    >>>
    >>> # Smoothed rolling climatology
    >>> smooth_clim = marEx.smoothed_rolling_climatology(
    ...     sst, window_years=15, smooth_days=21
    ... )
    >>>
    >>> # Compare seasonal cycle smoothness
    >>> # Extract annual cycle for a point
    >>> point_raw = raw_clim.isel(lat=90, lon=180).sel(time='2010')
    >>> point_smooth = smooth_clim.isel(lat=90, lon=180).sel(time='2010')
    >>>
    >>> print(f"Raw climatology range: {(point_raw.max() - point_raw.min()).compute():.3f}")
    >>> print(f"Smooth climatology range: {(point_smooth.max() - point_smooth.min()).compute():.3f}")

    Performance considerations:

    >>> # Efficient implementation smooths raw data first, then computes climatology
    >>> # This is more memory-efficient than smoothing the climatology
    >>> large_sst = sst.chunk({'time': 25, 'lat': 45, 'lon': 90})
    >>> efficient_clim = marEx.smoothed_rolling_climatology(large_sst)
    """
    # Infer and validate dimensions and coordinates
    dimensions, coordinates = _infer_dims_coords(da, dimensions, coordinates)
    timedim = dimensions["time"]
    cycle = resolve_cycle(da, coordinates["time"], cycle)

    # `smooth_days` is a physical duration, so convert it to whole timesteps. On daily
    # data this is the identity (21 days -> 21 steps) and every existing result is
    # unchanged. It is deliberately NOT forced odd: an even smoothing window is legal
    # here and forcing it would move existing daily output. On a monthly axis it
    # collapses to 1 step, which `steps_for_days` warns about -- the smoothing then
    # degenerates to `rolling_climatology`, which is correct but worth saying out loud.
    smooth_steps = cycle.steps_for_days(smooth_days, name="smooth_days")
    if smooth_steps == 1:
        # Warning, not info. `steps_for_days` only warns when the realised duration is
        # more than half a step from the request, and a 21-day ask on a 31-day axis is
        # inside that -- 31 days genuinely is the closest representable window. But the
        # user asked for smoothing and is getting none, which is a different method from
        # the one they requested, so it is said out loud regardless.
        logger.warning(
            "smooth_days=%s resolves to a single timestep on this %g-day axis: no smoothing "
            "is applied and smoothed_rolling_climatology reduces to rolling_climatology.",
            smooth_days,
            cycle.step_days,
        )

    # Whether a given (length, chunking, window) combination can actually be reduced is a
    # property of the xarray -> dask.overlap -> bottleneck chain, and it is not a simple
    # one. It has at least three regimes: chunks that divide smooth_days - 1 leave
    # a block one element short of the window; arrays shorter than the overlap depth are
    # rejected outright; and a window longer than the whole series is fine and yields NaN.
    # Modelling that here would hard-code one version's behaviour and go stale silently.
    #
    # Instead, ask the real stack. The probe reproduces the exact time geometry on a 1-D
    # zero array -- a few KB and a few ms even for decades of daily data -- so whatever
    # upstream does, the user gets a clear error here instead of a cryptic one from
    # bottleneck after the pipeline has been running for half an hour.
    time_chunks = da.chunksizes.get(timedim, ())
    if time_chunks:
        probe = xr.DataArray(np.zeros(sum(time_chunks), dtype=np.float32), dims=[timedim]).chunk({timedim: time_chunks})
        try:
            probe.rolling({timedim: smooth_steps}, center=True).mean().compute()
        except ValueError as exc:
            raise ConfigurationError(
                f"Time chunking cannot support a {smooth_days}-day centred rolling mean",
                details=(
                    f"Reducing a {smooth_steps}-step ({smooth_days}-day) window over time chunks "
                    f"{sorted(set(time_chunks))} failed with: {exc}"
                ),
                suggestions=[
                    f"Rechunk the time dimension to at least the window length: " f"da.chunk({{'{timedim}': {smooth_steps}}})",
                    "Chunk the spatial dimension instead, to keep chunk sizes manageable",
                    "Reduce smooth_days",
                ],
                context={
                    "time_chunks": sorted(set(time_chunks)),
                    "smooth_days": smooth_days,
                    "smooth_steps": smooth_steps,
                    "upstream_error": str(exc),
                },
            ) from exc

    # N.B.: It is more efficient (chunking-wise) to smooth the raw data rather than the climatology
    #
    # Kept in float32 deliberately. bottleneck's move_mean carries a running sum that
    # restarts at each dask block boundary, so the result shifts slightly with the chunk
    # layout: ~1e-4 for a 21-day window on SST, i.e. a few float32 ULP at 280 K. Accumulating
    # in float64 removes that exactly, but doubles the working set of this reduction, which
    # was enough to exhaust the workers and take down a distributed run. Forcing xarray off
    # bottleneck also removes it, at 33x the cost. A spread at float32 precision is the
    # accepted trade -- see the tolerance in tests/test_climatology_chunking.py.
    da_smoothed = da.rolling({timedim: smooth_steps}, center=True).mean().chunk(dict(zip(da.dims, da.chunks))).astype(np.float32)

    clim = rolling_climatology(da_smoothed, window_years, dimensions, coordinates, cycle)

    return clim
