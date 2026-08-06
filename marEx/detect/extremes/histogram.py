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

import dask
import flox.xarray
import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from xhistogram.xarray import histogram

from ...logging_config import get_logger
from ..compute_mode import Materialiser

# Get module logger
logger = get_logger(__name__)

# Target number of array elements per histogram task. Tiling the spatial dimensions
# to roughly this many elements keeps each task's working set bounded (~200 MB for
# float32) regardless of field resolution or input chunking, which is what allows the
# histogram-quantile path to run at full resolution without exhausting worker memory.
_HISTOGRAM_TASK_ELEMENTS = 50_000_000


def _chunk_spatial_for_histogram(da: xr.DataArray, dim: str, target_elements: int = _HISTOGRAM_TASK_ELEMENTS) -> xr.DataArray:
    """Tile the non-reduced dimensions of ``da`` for memory-safe histogram reduction.

    The histogram-quantile kernels reduce over ``dim`` (typically time), which must stay
    in a single chunk per task. The remaining (spatial) dimensions are tiled so that the
    per-task element count stays near ``target_elements`` -- independent of the caller's
    chunking or the field resolution. This is a pure rechunk: every spatial cell's reduced
    axis lies wholly within one tile, so it changes only task granularity, never values.

    Parameters
    ----------
    da : xarray.DataArray
        Input array to be reduced along ``dim``.
    dim : str
        Name of the dimension that is reduced (kept unchunked).
    target_elements : int, optional
        Approximate number of array elements per spatial tile.

    Returns
    -------
    xarray.DataArray
        ``da`` rechunked with ``dim`` unchunked and the spatial dimensions tiled.
    """
    spatial_dims = [d for d in da.dims if d != dim]
    if not spatial_dims:
        return da.chunk({dim: -1})

    ntime = max(int(da.sizes[dim]), 1)
    tile_area = max(1, target_elements // ntime)
    side = max(1, int(round(tile_area ** (1.0 / len(spatial_dims)))))

    chunks = {d: min(int(da.sizes[d]), side) for d in spatial_dims}
    chunks[dim] = -1
    return da.chunk(chunks)


def _shifted_window_sum(da: xr.DataArray, dim: str, window: int, periodic: bool) -> xr.DataArray:
    """Centred window sum along ``dim`` that preserves the input's integer dtype.

    Equivalent to ``da.rolling({dim: window}, center=True, min_periods=1).sum()`` -- with a
    wrap-pad first when ``periodic`` -- for odd ``window``, but built from shifted views so
    the counts never leave their integer dtype. bottleneck's rolling sum promotes to float64
    and allocates a halo, which on the bin-resolved histogram is the dominant memory spike.

    Zero padding is what makes the non-periodic case exact: summing a full window over a
    zero-padded array is the same number as summing the partial window ``min_periods=1``
    admits at the edges.
    """
    left, right = (window - 1) // 2, window // 2

    # Each shifted slice starts at a different offset, so its chunk boundaries are offset
    # too. Adding them straight up makes dask unify_chunks to the common refinement of all
    # `window` boundary sets, which shreds the tiling into width-1 slivers (measured: at
    # window=5 over two spatial dims, 2654 tasks and 1783 rechunk keys where the tiled
    # input had 6 chunks, output chunks (1,1,16,1,1,...)). That all-to-all rechunk is what
    # OOM-killed the full-scale gridded hobday run. Putting every slice back on the input's
    # own boundaries first keeps each add chunk-aligned: the shift becomes a local overlap
    # (each output chunk draws on at most two input chunks) and the tiling survives
    # -- 274 tasks / 104 rechunk keys for the same window=5 case, values bit-identical.
    target_chunks = None
    if da.chunks is not None:
        target_chunks = da.chunks[da.dims.index(dim)]

    if periodic:
        padded = da.pad({dim: (left, right)}, mode="wrap")
    else:
        padded = da.pad({dim: (left, right)}, mode="constant", constant_values=0)

    # Drop the padded dimension coordinate so the shifted slices add positionally rather
    # than aligning on (now meaningless) padded labels; restore the original afterwards.
    coord = da.coords[dim] if dim in da.coords else None
    padded = padded.drop_vars(dim, errors="ignore")

    n = da.sizes[dim]

    def _slice(offset: int) -> xr.DataArray:
        window_slice = padded.isel({dim: slice(offset, offset + n)})
        return window_slice if target_chunks is None else window_slice.chunk({dim: target_chunks})

    total = _slice(0)
    for offset in range(1, window):
        total = total + _slice(offset)

    if coord is not None:
        total = total.assign_coords({dim: coord})
    return total


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

    # Vectorised search for the bins containing the quantile position.
    # ``searchsorted(row, v, side="right")`` on a non-decreasing row is exactly the count
    # of entries <= v, so the whole 366-iteration Python loop of searchsorted calls (run
    # once per cell inside an apply_ufunc(vectorize=True)) collapses to one comparison
    # against the broadcast quantile positions (review finding 3.13).
    idx_upper = (cumsum <= quantile_position[:, None]).sum(axis=1).astype(np.int32)
    # Days with no data keep index 0 rather than the all-zero row's full-width count.
    idx_upper[total_counts <= 0] = 0

    # Clip to valid range
    idx_upper = np.clip(idx_upper, 0, n_bins - 1)
    idx_lower = np.maximum(0, idx_upper - 1)

    # Extract values for vectorised interpolation
    doy_indices = np.arange(n_doy, dtype=np.int32)

    # Get cumulative counts at the boundaries
    count_lower = np.where(idx_lower >= 0, cumsum[doy_indices, idx_lower], 0)
    count_upper = cumsum[doy_indices, idx_upper]

    # Bin CENTRES for interpolation -- deliberately not the bin edges used by the 1D path
    # (_compute_histogram_quantile_1d). Cumulative counts formally correspond to bin upper
    # edges, so edge-based interpolation looks like the "correct" inverse CDF, and for the 1D
    # path it is: that path pools the whole time series into these same ~500 bins, so the bins
    # around the quantile hold many samples, the interpolation fraction is meaningful, and
    # edge interpolation tracks np.percentile to ~1/5 of a bin.
    #
    # This 2D path pools only a per-day-of-year window (tens to a few hundred samples) over the
    # same bins, so those bins are empty or hold a single sample. The fraction then degenerates
    # to 0 or 1 and edge interpolation snaps to a bin boundary, adding a systematic half-bin
    # bias. Switching this path to edges was measured to shift every threshold up by exactly
    # +precision/2 (uniformly across 11/21/41-day windows) and moved the 41-day result from
    # 0.0006 to 0.0044 away from the analytic 90th percentile. Keep centres here.
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
    materialiser: Optional[Materialiser] = None,
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
    # A None materialiser means "default to persist mode", which keeps every existing
    # caller, doctest and test working unchanged.
    if materialiser is None:
        materialiser = Materialiser("persist")

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

    # Size the spatial tiles from the histogram OUTPUT budget rather than a hardcoded
    # 16 cells. Each spatial cell yields a (366, n_bins) histogram, so cap the cells
    # per tile to keep the per-task output near the element budget. On a gridded grid
    # this reproduces the previous ~16x16 tiling; on an unstructured (x-only) grid it
    # uses ~256 cells/chunk instead of 16, avoiding a task-graph explosion (the
    # "hobday scheduler OOM on unstructured" failure).
    n_bins = len(bin_centers_array)
    spatial_dims_present = [dimensions[d] for d in ("y", "x") if d in dimensions]
    cells_per_tile = max(1, _HISTOGRAM_TASK_ELEMENTS // (366 * max(1, n_bins)))
    tile_side = max(1, int(round(cells_per_tile ** (1.0 / max(1, len(spatial_dims_present))))))
    # The spatial smoothing below rolls a window of `window_spatial_hobday` over each
    # spatial dim, which requires every chunk to be at least that wide; never tile below it.
    if window_spatial_hobday is not None and window_spatial_hobday > 1:
        tile_side = max(tile_side, int(window_spatial_hobday))
    chunk_dict = {dimensions["time"]: -1}
    for d in spatial_dims_present:
        chunk_dict[d] = min(int(da.sizes[d]), tile_side)

    da_bin = (
        xr.DataArray(
            # Clip finite data into the last bin (its centre) before digitizing so
            # out-of-range-high values are counted in the top bin rather than silently
            # dropped by the flox expected_groups (which biased every approximate
            # threshold low). Clipping the data (not the index) preserves NaN, which
            # still digitizes out of range and is correctly dropped.
            np.digitize(np.clip(da.data, None, bin_centers_array[-1]), bin_edges) - 1,
            dims=da.dims,
            coords=da.coords,
            name="da_bin",
        )
        # Cast BEFORE the rechunk. np.digitize returns int64, and the rechunk below is the
        # all-to-all shuffle of the hobday path, so casting afterwards moved 4x the bytes
        # it needed to (~77 GB vs ~19 GB at 9282x720x1440). Values are unchanged: the bin
        # indices are small non-negative integers (review finding 3.5).
        .astype(np.uint16).chunk(chunk_dict)
    )

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

        # Integer-preserving window sums. xarray's .rolling().sum() goes through
        # bottleneck, which promotes these uint16 chunks to float64 and carries a halo
        # overlap -- ~0.4-0.8 GB transient per task over the (y, x, 366, ~502) histogram,
        # the dominant memory spike of the default gridded hobday path (review finding
        # 3.6). Summing explicit shifted views keeps the counts in an integer dtype and
        # is exactly equal to the rolling sum for odd windows: a zero-padded full window
        # equals a min_periods=1 partial window, and a wrap-padded one equals the periodic
        # case. Even windows keep the old path, whose centre alignment they depend on.
        use_integer_window = window_spatial_hobday % 2 == 1
        hist_rolled = hist_raw.astype(np.uint32) if use_integer_window else hist_raw

        # Periodic padding in longitude, rolling mean in both dimensions, then trim
        if lon_dim in hist_raw.dims:
            if use_integer_window:
                hist_rolled = _shifted_window_sum(hist_rolled, lon_dim, window_spatial_hobday, periodic=True)
            else:
                hist_rolled = hist_rolled.pad({lon_dim: pad_size}, mode="wrap")
                hist_rolled = hist_rolled.rolling({lon_dim: window_spatial_hobday}, center=True, min_periods=1).sum()
                hist_rolled = hist_rolled.isel({lon_dim: slice(pad_size, pad_size + hist_raw.sizes[lon_dim])})

        # Standard rolling in latitude
        if lat_dim in hist_raw.dims:
            if use_integer_window:
                hist_rolled = _shifted_window_sum(hist_rolled, lat_dim, window_spatial_hobday, periodic=False)
            else:
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

    # Drop time coordinate to avoid conflicts when comparing with data grouped by dayofyear
    if dimensions["time"] in threshold.coords:
        threshold = threshold.drop_vars(dimensions["time"])

    # Set threshold to NaN only for spatial points that are NaN at *every* timestep.
    # Masking on NaN at t=0 alone would give seasonal cells (e.g. sea ice, valid part
    # of the year) a permanent NaN threshold. (Consistent with the 1D path.)
    nan_mask = da.isnull().all(dim=dimensions["time"]).compute()
    threshold = materialiser.pin_one(threshold.where(~nan_mask))

    # Validate threshold values against bounds
    upper_bound = bin_edges[-2]
    lower_bound = bin_edges[3]  # We want this to be positive so that constant=0 anomalies will not be "extreme"

    # One scheduler round-trip for the whole bounds check. Each predicate used to be its
    # own ``if ...any():`` -- DataArray.__bool__ computes -- plus a ``.max()``/``.min()``
    # inside the warning body, so up to four traversals of a graph that re-executes the
    # entire upstream anomaly whenever it is not pinned. The four reductions share their
    # upstream, so computing them together costs a single pass. ``too_low`` stays lazy as
    # an array: only its ``.any()`` becomes a scalar, because the clamp below still needs
    # the elementwise mask.
    #
    # The predicates here deliberately lack the ``& notnull()`` guard the 1D path carries.
    # NaN comparisons are False either way; adding it would alter the clamp mask.
    too_high = threshold > upper_bound
    too_low = threshold < lower_bound
    any_high, any_low, thr_max, thr_min = dask.compute(too_high.any(), too_low.any(), threshold.max(), threshold.min())

    if bool(any_high):
        warnings.warn(
            f"Quantile values exceed expected range: max={float(thr_max):.4f} > {upper_bound:.4f}. "
            f"Consider increasing max_anomaly parameter (currently {max_anomaly:.2f}) or using a lower percentile threshold.",
            UserWarning,
            stacklevel=2,
        )

    if bool(any_low):
        warnings.warn(
            f"Quantile values below expected range in some locations: min={float(thr_min):.4f} < {lower_bound:.4f}. "
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
    materialiser: Optional[Materialiser] = None,
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
    # A None materialiser means "default to persist mode", which keeps every existing
    # caller, doctest and test working unchanged.
    if materialiser is None:
        materialiser = Materialiser("persist")

    if bin_edges is None:
        # Create optimised asymmetric bins
        bin_edges = np.concatenate([[-np.inf], np.arange(-precision, max_anomaly + precision, precision)])

    # Tile the non-reduced (spatial) dimensions so each histogram task processes a
    # bounded working set, independent of how the caller chunked the input. Without
    # this, a spatially-unchunked full-resolution field (e.g. a single 720x1440 chunk)
    # forces one task to hold the entire (time x space) array plus the per-bin working
    # memory, which exhausts worker memory at scale. Because every spatial cell's
    # time series lies wholly within its own tile (time stays unchunked), the per-cell
    # histogram counts -- and therefore the resulting quantiles -- are independent of
    # this tiling: the operation is bit-for-bit identical to the unchunked computation.
    da = _chunk_spatial_for_histogram(da, dim)

    # Clip finite data into the last bin (its centre) so out-of-range-high values are
    # counted in the top bin instead of being dropped by xhistogram, which renormalised
    # the CDF over a truncated total and biased the threshold low. NaN is preserved by
    # clip and is still dropped by the histogram, so the mask below is unaffected.
    top_clip = float((bin_edges[-2] + bin_edges[-1]) / 2)

    # Compute histogram
    hist = histogram(da.clip(max=top_clip), bins=[bin_edges], dim=[dim])

    # Convert to PDF and CDF. Only the CDF (and the tiny per-cell totals) are persisted:
    # `hist` is an intermediate of the same graph and is never read once `cdf` exists, so
    # persisting it too doubled the bin-resolved footprint (~4 GB each at 720x1440)
    # (review finding 3.7).
    total_counts = hist.sum(dim=f"{da.name}_bin")
    pdf = hist / (total_counts + 1e-10)
    cdf, total_counts = materialiser.pin(pdf.cumsum(dim=f"{da.name}_bin"), total_counts)

    eps = 1e-10

    # Locate the bin containing the q-th quantile: the first bin whose cumulative CDF
    # reaches q. ``cdf[i]`` is the fraction of samples <= the UPPER edge of bin i, so the
    # quantile is interpolated WITHIN that bin between its histogram edges. Interpolating
    # on edges (not bin centres) removes a systematic half-bin-width low bias: the earlier
    # code interpolated between bin centres, which was ~precision/2 less accurate than the
    # true percentile (and the previous strict-`>` variant degenerated to a bin centre).
    n_bins = len(bin_edges) - 1
    idx_upper = (cdf >= (q - eps)).argmax(dim=f"{da.name}_bin")
    # Clamp so idx_upper-1 >= 0 and idx_upper+1 indexes a valid (finite) upper edge.
    idx_upper = xr.where(idx_upper < 1, 1, xr.where(idx_upper > n_bins - 1, n_bins - 1, idx_upper)).compute()
    idx_lower = idx_upper - 1

    cdf_lower = cdf.isel({f"{da.name}_bin": idx_lower})  # CDF at the bin's lower edge
    cdf_upper = cdf.isel({f"{da.name}_bin": idx_upper})  # CDF at the bin's upper edge
    edge_lower = xr.DataArray(bin_edges[idx_upper.values], dims=idx_upper.dims, coords=idx_upper.coords)
    edge_upper = xr.DataArray(bin_edges[idx_upper.values + 1], dims=idx_upper.dims, coords=idx_upper.coords)

    # Within-bin linear interpolation of the inverse CDF. idx_upper is the FIRST bin to
    # reach q, so cdf_upper >= q > cdf_lower and the denominator is strictly positive
    # (guarded for degenerate all-NaN/constant cells, which are masked below anyway).
    denom = cdf_upper - cdf_lower
    frac = (q - cdf_lower) / xr.where(denom > eps, denom, 1.0)
    threshold = edge_lower + frac * (edge_upper - edge_lower)

    # Set threshold to NaN only for spatial points that are NaN at *every* timestep,
    # so a cell that is valid for part of the year (e.g. seasonal sea ice) still gets a
    # real threshold rather than a permanent NaN. (Consistent with the 2D path.)
    # The mask comes free from the histogram totals: the histogram drops NaN and counts
    # every finite sample (values above the range are clipped in, above), so a zero total
    # is exactly "NaN at every timestep". Recomputing it from `da` re-ran the whole
    # upstream anomaly graph plus this function's spatial re-tiling (review finding 3.8).
    nan_mask = total_counts == 0
    threshold = materialiser.pin_one(threshold.where(~nan_mask).drop_vars(f"{da.name}_bin"))

    # Validate threshold against bounds
    upper_bound = bin_edges[-2]
    lower_bound = bin_edges[3]  # We want this to be positive so that constant=0 anomalies will not be "extreme"

    # One scheduler round-trip for the whole bounds check -- see the matching comment in
    # the 2D path. The four reductions share their upstream, so computing them together
    # costs a single pass instead of one traversal per predicate.
    too_high = (threshold > upper_bound) & threshold.notnull()
    too_low = (threshold < lower_bound) & threshold.notnull()
    any_high, any_low, thr_max, thr_min = dask.compute(too_high.any(), too_low.any(), threshold.max(), threshold.min())

    if bool(any_high):
        warnings.warn(
            f"Quantile values exceed expected range: max={float(thr_max):.4f} > {upper_bound:.4f}. "
            f"Consider increasing max_anomaly parameter (currently {max_anomaly:.2f}) or using a lower percentile threshold.",
            UserWarning,
            stacklevel=2,
        )

    if bool(any_low):
        warnings.warn(
            f"Quantile values below expected range in some locations: min={float(thr_min):.4f} < {lower_bound:.4f}. "
            "This is likely due to a constant anomaly in certain (e.g. due to sea ice). "
            "Double check the computed threshold values are correct.",
            UserWarning,
            stacklevel=2,
        )
        # Set too low values to lower bound -- This is to ensure that constant=0 anomalies will not be "extreme"
        threshold = materialiser.pin_one(threshold.where(~too_low, lower_bound))

    return threshold
