"""The unstructured hole filling passes the sparse dilation matrix to apply_ufunc as core dimensions.

Left to dask's automatic chunking, a large mesh splits those arrays (``indices`` is ~0.8 GB at
ICON R02B09), and apply_gufunc has to merge them back while re-splitting the loop dimension,
raising "Increasing number of chunks". A small ``array.chunk-size`` reproduces that on a test mesh.
"""

import warnings

import dask
import numpy as np
import xarray as xr
from dask.array.core import PerformanceWarning

from marEx.track.grid import build_sparse_dilation_matrix
from marEx.track.morphology import fill_holes

from .mre_mesh import build_mesh


def test_sparse_matrix_core_dims_do_not_multiply_chunks():
    nb, _, _, _ = build_mesh(20, 70)
    neighbours = (xr.DataArray(nb, dims=("nv", "ncells")).astype(np.int32) - 1).chunk(-1)  # 0-based, as the tracker holds it
    ncells = nb.shape[1]
    rng = np.random.default_rng(0)
    data = xr.DataArray(rng.random((10, ncells)) > 0.6, dims=("time", "ncells")).chunk({"time": 5, "ncells": -1})
    mask = xr.DataArray(np.ones(ncells, bool), dims="ncells").chunk(-1)
    with dask.config.set({"array.chunk-size": "2KiB"}), warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        filled = fill_holes(data, 1, True, build_sparse_dilation_matrix(neighbours), "ncells", mask, False, None)
    assert filled.chunks == data.chunks
    with dask.config.set(scheduler="synchronous"):
        reference = fill_holes(data, 1, True, build_sparse_dilation_matrix(neighbours), "ncells", mask, False, None)
        np.testing.assert_array_equal(filled.values, reference.values)
