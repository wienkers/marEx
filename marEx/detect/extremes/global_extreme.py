"""
Global (constant-in-time) extreme identification.

Provides :func:`_identify_extremes_constant`, which applies a single
time-invariant percentile threshold to identify extreme events. This backs the
``global_extreme`` extreme-detection method.
"""

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import xarray as xr

from ...logging_config import get_logger
from .histogram import _compute_histogram_quantile_1d

# Get module logger
logger = get_logger(__name__)


def _identify_extremes_constant(
    da: xr.DataArray,
    threshold_percentile: float = 95,
    method_percentile: Literal["exact", "approximate"] = "approximate",
    dimensions: Optional[Dict[str, str]] = None,
    precision: float = 0.01,
    max_anomaly: float = 5.0,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify extreme events exceeding a constant (in time) percentile threshold.
    i.e. There is 1 threshold for each spatial point, computed across all time.

    Returns both the extreme events boolean mask and the thresholds used.
    """
    if method_percentile == "exact":  # Compute exact percentile (memory-intensive)
        # Determine appropriate chunk size based on data dimensions
        if "y" in dimensions:
            rechunk_size = "auto"
        else:
            # For small unstructured grids (< ~4445 cells) the rounded expression is 0, an
            # invalid zero-size chunk. Clamping it to 1 is also wrong: a one-cell chunk means
            # one task per cell, which is precisely the task explosion warned about below.
            # Floor at 100 cells instead (or the whole grid, if it is smaller than that).
            # The exact quantile is a per-cell reduction over time, so chunking along cells
            # cannot change the result -- only the task count.
            n_cells = da[dimensions["x"]].size
            rechunk_size = max(min(n_cells, 100), 100 * int(np.sqrt(n_cells) * 1.5 / 100))
        # N.B.: If this rechunk_size is too small, then dask will be overwhelmed by the number of tasks
        chunk_dict = {dimensions[dim]: rechunk_size for dim in ["x", "y"] if dim in dimensions}
        chunk_dict[dimensions["time"]] = -1
        da_rechunk = da.chunk(chunk_dict)

        # Calculate threshold
        threshold = da_rechunk.quantile(threshold_percentile / 100.0, dim=dimensions["time"])

    else:  # Use an efficient histogram-based method with specified accuracy
        threshold = _compute_histogram_quantile_1d(
            da, threshold_percentile / 100.0, dim=dimensions["time"], precision=precision, max_anomaly=max_anomaly
        )

    # Clean up coordinates if needed
    if "quantile" in threshold.coords:
        threshold = threshold.drop_vars("quantile")

    # Ensure spatial dimensions are fully loaded for efficient comparison
    spatial_chunks = {dimensions[dim]: -1 for dim in ["x", "y"] if dim in dimensions}
    threshold = threshold.chunk(spatial_chunks).persist()

    # Create boolean mask for values exceeding threshold
    extremes = da >= threshold

    # Clean up coordinates if needed
    if "quantile" in extremes.coords:
        extremes = extremes.drop_vars("quantile")

    extremes = extremes.astype(bool).chunk(dict(zip(da.dims, da.chunks))).persist()

    return extremes, threshold
