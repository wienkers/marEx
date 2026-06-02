"""
Histogram-based quantile estimation kernels for extreme detection.

Provides approximate quantile computation from precomputed histograms, used by
the ``approximate`` percentile branches of both the global and Hobday extreme
methods. Contains a pure NumPy stride-tricks kernel plus 1-D and 2-D
(day-of-year resolved) histogram-quantile drivers. This module is a leaf in the
detect package dependency graph (it imports only package-level helpers and
logging).
"""

import warnings
from typing import Dict, Optional

import flox.xarray
import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from xhistogram.xarray import histogram

from ...helper import checkpoint_to_zarr
from ...logging_config import get_logger

# Get module logger
logger = get_logger(__name__)


def _rolling_histogram_quantile(
    hist_chunk: NDArray[np.int32],
    window_days_hobday: int,
    q: float,
    bin_centers: NDArray[np.float64],
) -> NDArray[np.float32]:
    """
    Efficiently compute quantile thresholds from histogram data using vectorised numpy operations.
    Improved robust interpolation handles sparse histograms, especially in the tails.

    Parameters
    ----------
    hist_chunk : numpy.ndarray
        Histogram data with shape (dayofyear, da_bin)
    window_days_hobday : int
        Rolling window size for day-of-year smoothing
    q : float
        Quantile to compute (0-1)
    bin_centers : numpy.ndarray
        Bin centre values for interpolation

    Returns
    -------
    numpy.ndarray
        Quantile thresholds with shape (dayofyear,)
    """
    n_doy, n_bins = hist_chunk.shape
    eps = 1e-10

    # Pad histogram with wrap mode for day-of-year cycling
    pad_size = window_days_hobday // 2
    hist_pad = np.concatenate([hist_chunk[-pad_size:], hist_chunk, hist_chunk[:pad_size]], axis=0)

    # Apply rolling sum using stride tricks FTW
    windowed_view = sliding_window_view(hist_pad, window_days_hobday, axis=0)
    hist_windowed = np.sum(windowed_view, axis=-1)

    # Apply gaussian smoothing along bin dimension
    # sigma = 2
    # hist_smoothed = gaussian_filter1d(
    #     hist_windowed.astype(np.float32), sigma=sigma, axis=1, mode="constant", cval=0.0  # Along bin dimension
    # ).astype(np.float32)

    # Count-based interpolation (rather than interpolating CDF in probability space)
    # Calculate cumulative counts (not normalized CDF)
    cumsum = np.cumsum(hist_windowed, axis=1, dtype=np.int32)
    total_counts = cumsum[:, -1]  # Total count for each day

    # Calculate the exact position where the quantile should be
    # For n samples, the q-th quantile is at position q*(n-1)
    # It is q*n here since we're working with cumulative counts
    quantile_position = q * total_counts

    # Vectorised search for the bins containing the quantile position
    # searchsorted with side='right' gives the first bin where cumsum > quantile_position
    idx_upper = np.zeros(n_doy, dtype=np.int32)

    for i in range(n_doy):
        if total_counts[i] <= 0:  # No data
            idx_upper[i] = 0
        else:
            # Find first bin where cumulative count exceeds target position
            idx_upper[i] = np.searchsorted(cumsum[i], quantile_position[i], side="right")

    # Clip to valid range
    idx_upper = np.clip(idx_upper, 0, n_bins - 1)
    idx_lower = np.maximum(0, idx_upper - 1)

    # Extract values for vectorised interpolation
    doy_indices = np.arange(n_doy, dtype=np.int32)

    # Get cumulative counts at the boundaries
    count_lower = np.where(idx_lower >= 0, cumsum[doy_indices, idx_lower], 0)
    count_upper = cumsum[doy_indices, idx_upper]

    # Bin centers for interpolation
    bin_lower = bin_centers[idx_lower]
    bin_upper = bin_centers[idx_upper]

    # Compute interpolation fraction based on counts
    count_diff = count_upper - count_lower
    safe_diff = np.where(count_diff > eps, count_diff, 1.0)
    frac = np.where(count_diff > eps, (quantile_position - count_lower) / safe_diff, 0.5)  # If no difference, use midpoint

    # Linear interpolation between bin centers
    threshold = bin_lower + frac * (bin_upper - bin_lower)

    # Handle edge cases
    # If total_counts is 0, return NaN
    threshold = np.where(total_counts > 0, threshold, np.nan)

    # If at the first bin (all data is negative), use the first bin center
    threshold = np.where((idx_upper == 0) & (total_counts > 0), bin_centers[0], threshold)

    return threshold.astype(np.float32)


def _compute_histogram_quantile_2d(
    da: xr.DataArray,
    q: float,
    window_days_hobday: int = 11,
    window_spatial_hobday: Optional[int] = None,
    bin_edges: Optional[NDArray[np.float64]] = None,
    dimensions: Optional[Dict[str, str]] = None,
    precision: float = 0.01,
    max_anomaly: float = 5.0,
    use_temp_checkpoints: bool = False,
) -> xr.DataArray:
    """
    Efficiently compute quantiles using binned histograms optimised for extreme values.
    Uses fine-grained bins for positive anomalies and a single bin for negative values.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    q : float
        Quantile to compute (0-1)
    window_days_hobday : int, default=11
        Rolling window size for day-of-year quantiles
    window_spatial_hobday : int, default=None
        Spatial window size for day-of-year quantiles
    bin_edges : numpy.ndarray, optional
        Custom bin edges for histogram computation
    dimensions : dict, optional
        Dimension mapping dictionary
    precision : float, default=0.01
        Precision for positive anomaly bins
    max_anomaly : float, default=5.0
        Maximum anomaly value for binnin

    Returns
    -------
    xarray.DataArray
        Computed quantile value for each spatial location
    """
    if bin_edges is None:
        # Create optimised asymmetric bins
        bin_edges = np.concatenate(
            [[-np.inf], np.arange(-precision, max_anomaly + precision, precision, dtype=np.float32)], dtype=np.float32
        )

    bin_centers_array = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_centers_array[0] = 0.0

    bin_centers = xr.DataArray(
        bin_centers_array.astype(np.float32),
        dims=["da_bin"],
        coords={"da_bin": np.arange(len(bin_centers_array), dtype=np.uint16)},
        name="bin_centers",
    )

    chunk_dict = {dimensions["time"]: -1}
    chunk_dict[dimensions["x"]] = 16
    if "y" in dimensions:
        chunk_dict[dimensions["y"]] = 16

    da_bin = (
        xr.DataArray(
            np.digitize(da.data, bin_edges) - 1,  # -1 so first bin is 0
            dims=da.dims,
            coords=da.coords,
            name="da_bin",
        )
        .chunk(chunk_dict)
        .astype(np.uint16)
    )

    if use_temp_checkpoints:
        logger.debug("Checkpointing binned data to break graph dependencies")
        da_bin = checkpoint_to_zarr(da_bin, name="da_bin", timedim=dimensions["time"]).chunk(chunk_dict)

    # Construct 2D histogram using flox (in doy & anomaly)
    hist_raw = flox.xarray.xarray_reduce(
        da_bin,
        da_bin.dayofyear,
        da_bin,
        dim=[dimensions["time"]],
        func="count",
        expected_groups=(np.arange(1, 367, dtype=np.uint16), np.arange(len(bin_edges) - 1, dtype=np.uint16)),
        isbin=(False, False),
        dtype=np.uint16,
        fill_value=0,
    )
    hist_raw.name = None

    # Apply spatial-kernel smoothing to the histogram
    if window_spatial_hobday is not None and window_spatial_hobday > 1:
        pad_size = window_spatial_hobday // 2
        lon_dim, lat_dim = dimensions.get("x"), dimensions.get("y")

        hist_rolled = hist_raw

        # Periodic padding in longitude, rolling mean in both dimensions, then trim
        if lon_dim in hist_raw.dims:
            hist_rolled = hist_rolled.pad({lon_dim: pad_size}, mode="wrap")
            hist_rolled = hist_rolled.rolling({lon_dim: window_spatial_hobday}, center=True, min_periods=1).sum()
            hist_rolled = hist_rolled.isel({lon_dim: slice(pad_size, pad_size + hist_raw.sizes[lon_dim])})

        # Standard rolling in latitude
        if lat_dim in hist_raw.dims:
            hist_rolled = hist_rolled.rolling({lat_dim: window_spatial_hobday}, center=True, min_periods=1).sum()

        hist_raw = hist_rolled

    def _compute_quantile_with_params(hist_chunk, bin_centers_chunk):
        return _rolling_histogram_quantile(hist_chunk, window_days_hobday, q, bin_centers_chunk)

    # Rechunk histogram so core dimensions are unchunked for apply_ufunc
    # Create chunk dict for hist_raw that preserves spatial chunks but drops time
    hist_chunk_dict = {dimensions["x"]: chunk_dict.get(dimensions["x"], 16), "dayofyear": -1, "da_bin": -1}
    if "y" in dimensions:
        hist_chunk_dict[dimensions["y"]] = chunk_dict.get(dimensions["y"], 16)

    hist_raw = hist_raw.chunk(hist_chunk_dict)

    # Apply the optimised computation using apply_ufunc
    threshold = xr.apply_ufunc(
        _compute_quantile_with_params,
        hist_raw,
        bin_centers,
        input_core_dims=[["dayofyear", "da_bin"], ["da_bin"]],
        output_core_dims=[["dayofyear"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"dayofyear": 366}},
        keep_attrs=True,
    )

    if use_temp_checkpoints:
        logger.debug("Checkpointing threshold to break graph dependencies")
        threshold = checkpoint_to_zarr(threshold, name="threshold")

    # Drop time coordinate to avoid conflicts when comparing with data grouped by dayofyear
    if dimensions["time"] in threshold.coords:
        threshold = threshold.drop_vars(dimensions["time"])

    # Set threshold to NaN for spatial points that contain NaN values
    nan_mask = da.isel({dimensions["time"]: 0}).isnull().compute()
    threshold = threshold.where(~nan_mask).persist()

    # Validate threshold values against bounds
    upper_bound = bin_edges[-2]
    lower_bound = bin_edges[3]  # We want this to be positive so that constant=0 anomalies will not be "extreme"

    # Check if any values are too high (ignore NaN values)
    too_high = threshold > upper_bound
    if too_high.any():
        warnings.warn(
            f"Quantile values exceed expected range: max={threshold.max().compute():.4f} > {upper_bound:.4f}. "
            f"Consider increasing max_anomaly parameter (currently {max_anomaly:.2f}) or using a lower percentile threshold.",
            UserWarning,
            stacklevel=2,
        )

    # Check if any values are too low (ignore NaN values)
    too_low = threshold < lower_bound
    if too_low.any():
        warnings.warn(
            f"Quantile values below expected range in some locations: min={threshold.min().compute():.4f} < {lower_bound:.4f}. "
            "This is likely due to a constant anomaly in certain (e.g. due to sea ice). "
            "Double check the computed threshold values are correct.",
            UserWarning,
            stacklevel=2,
        )
        # Set too low values to lower bound -- This is to ensure that constant=0 anomalies will not be "extreme"
        threshold = threshold.where(~too_low, lower_bound)

    return threshold


def _compute_histogram_quantile_1d(
    da: xr.DataArray,
    q: float,
    dim: str = "time",
    bin_edges: Optional[NDArray[np.float64]] = None,
    precision: float = 0.01,
    max_anomaly: float = 5.0,
) -> xr.DataArray:
    """
    Efficiently compute quantiles using binned histograms optimised for extreme values.
    Uses fine-grained bins for positive anomalies and a single bin for negative values.
    Improved robust interpolation handles empty bins in the tails.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    q : float
        Quantile to compute (0-1)
    dim : str, optional
        Dimension along which to compute quantile
    bin_edges : numpy.ndarray, optional
        Custom bin edges for histogram computation
    precision : float, default=0.01
        Precision for positive anomaly bins
    max_anomaly : float, default=5.0
        Maximum anomaly value for binning

    Returns
    -------
    xarray.DataArray
        Computed quantile value for each spatial location
    """
    if bin_edges is None:
        # Create optimised asymmetric bins
        bin_edges = np.concatenate([[-np.inf], np.arange(-precision, max_anomaly + precision, precision)])

    # Compute histogram
    hist = histogram(da, bins=[bin_edges], dim=[dim]).persist()

    # Convert to PDF and CDF
    hist_sum = hist.sum(dim=f"{da.name}_bin") + 1e-10
    pdf = hist / hist_sum
    cdf = pdf.cumsum(dim=f"{da.name}_bin").persist()

    # Get bin centers
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_centers[0] = 0.0  # Set negative bin centre to 0
    eps = 1e-10

    # Find bins for interpolation
    # Find first bin where CDF >= (q - eps) - this becomes upper bound
    cdf_above_q = cdf >= (q - eps)
    idx_upper = cdf_above_q.argmax(dim=f"{da.name}_bin")

    # Get CDF value one point to the left of idx_upper
    idx_before_upper = xr.where(idx_upper - 1 > 0, idx_upper - 1, 0)

    # Extract the target CDF value (avoiding negative indexing issues)
    idx_before_upper_computed = idx_before_upper.compute()
    cdf_target = cdf.isel({f"{da.name}_bin": idx_before_upper_computed})

    # Find idx_lower: first bin where CDF > cdf_target
    cdf_above_target = cdf > cdf_target
    idx_lower = cdf_above_target.argmax(dim=f"{da.name}_bin")

    # Ensure bounds are valid
    idx_lower = xr.where(idx_lower < 0, 0, xr.where(idx_lower > len(bin_centers) - 2, len(bin_centers) - 2, idx_lower))
    idx_upper = xr.where(idx_upper < 1, 1, xr.where(idx_upper > len(bin_centers) - 1, len(bin_centers) - 1, idx_upper))

    # Extract CDF and bin values for interpolation
    idx_lower_computed = idx_lower.compute()
    idx_upper_computed = idx_upper.compute()

    cdf_lower = cdf.isel({f"{da.name}_bin": idx_lower_computed})
    cdf_upper = cdf.isel({f"{da.name}_bin": idx_upper_computed})
    bin_lower = bin_centers[idx_lower_computed]
    bin_upper = bin_centers[idx_upper_computed]

    # Robust interpolation with proper handling of degenerate cases
    denom = cdf_upper - cdf_lower

    # Handle exact matches and zero denominators
    exact_match = (xr.ufuncs.fabs(cdf_lower - q) < eps).persist()
    zero_denom = (xr.ufuncs.fabs(denom) <= eps).persist()

    # Standard interpolation
    frac = (q - cdf_lower) / xr.where(xr.ufuncs.fabs(denom) > eps, denom, 1.0)
    threshold = bin_lower + frac * (bin_upper - bin_lower)

    # For exact matches, use the lower bin center
    threshold = xr.where(exact_match, bin_lower, threshold)

    # For zero denominator without exact match, use bin midpoint
    no_exact_match = zero_denom & ~exact_match
    threshold = xr.where(no_exact_match, (bin_lower + bin_upper) / 2, threshold)

    # Set threshold to NaN for spatial points that contain NaN values
    nan_mask = da.isnull().any(dim=dim)
    threshold = threshold.where(~nan_mask).drop_vars(f"{da.name}_bin").persist()

    # Validate threshold against bounds
    upper_bound = bin_edges[-2]
    lower_bound = bin_edges[3]  # We want this to be positive so that constant=0 anomalies will not be "extreme"

    # Check if any values are too high (ignore NaN values)
    too_high = (threshold > upper_bound) & threshold.notnull()
    if too_high.any():
        warnings.warn(
            f"Quantile values exceed expected range: max={threshold.max().compute():.4f} > {upper_bound:.4f}. "
            f"Consider increasing max_anomaly parameter (currently {max_anomaly:.2f}) or using a lower percentile threshold.",
            UserWarning,
            stacklevel=2,
        )

    # Check if any values are too low (ignore NaN values)
    too_low = (threshold < lower_bound) & threshold.notnull()
    if too_low.any():
        warnings.warn(
            f"Quantile values below expected range in some locations: min={threshold.min().compute():.4f} < {lower_bound:.4f}. "
            "This is likely due to a constant anomaly in certain (e.g. due to sea ice). "
            "Double check the computed threshold values are correct.",
            UserWarning,
            stacklevel=2,
        )
        # Set too low values to lower bound -- This is to ensure that constant=0 anomalies will not be "extreme"
        threshold = threshold.where(~too_low, lower_bound).persist()

    return threshold
