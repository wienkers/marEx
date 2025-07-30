"""Configuration for coverage-aware testing.

This module provides utilities to ensure proper coverage measurement
when working with Dask, xarray.apply_ufunc, and Numba-compiled functions.
"""

import os


def disable_numba_jit():
    """Disable Numba JIT compilation for coverage measurement."""
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    os.environ["NUMBA_CACHE_DIR"] = "/tmp"  # Avoid cache issues

    try:
        import numba

        numba.config.DISABLE_JIT = True
        numba.config.NUMBA_DISABLE_JIT = 1
    except ImportError:
        pass

    # Also patch numbagg which is commonly used by xarray
    try:
        import numbagg
        import numpy as np

        # Store original functions before patching
        if not hasattr(numbagg, "_coverage_original_nanquantile"):
            numbagg._coverage_original_nanquantile = numbagg.nanquantile

            def pure_numpy_nanquantile(a, quantiles, axis=None, **kwargs):
                """Pure numpy replacement for numbagg.nanquantile."""
                # numbagg uses 'quantiles' parameter, numpy uses 'q'
                return np.nanquantile(a, quantiles, axis=axis)

            numbagg.nanquantile = pure_numpy_nanquantile

            # Also patch other numbagg functions that might be used
            if hasattr(numbagg, "nanpercentile"):
                numbagg._coverage_original_nanpercentile = numbagg.nanpercentile

                def pure_numpy_nanpercentile(a, percentiles, axis=None, **kwargs):
                    """Pure numpy replacement for numbagg.nanpercentile."""
                    return np.nanpercentile(a, percentiles, axis=axis)

                numbagg.nanpercentile = pure_numpy_nanpercentile
    except ImportError:
        pass


def setup_dask_for_coverage():
    """Configure Dask for coverage measurement using threads scheduler."""
    if os.environ.get("COVERAGE_PROCESS_START") or os.environ.get("PYTEST_COVERAGE"):
        try:
            import dask

            # Use threads scheduler for coverage - this ensures all code
            # executes in the main process where coverage is active, but still
            # supports distributed operations like wait()
            dask.config.set(
                {
                    "scheduler": "threads",  # Use threads instead of synchronous
                    "distributed.worker.daemon": False,
                    "distributed.admin.tick.limit": "300s",
                    "array.slicing.split_large_chunks": False,  # Important for apply_ufunc
                    "num_workers": 1,  # Single worker for coverage
                }
            )

            # Also set the scheduler globally to override any test-specific settings
            os.environ["DASK_SCHEDULER"] = "threads"

        except ImportError:
            pass


def setup_coverage_environment():
    """Complete setup for coverage measurement."""
    if os.environ.get("COVERAGE_PROCESS_START") or os.environ.get("PYTEST_COVERAGE"):
        disable_numba_jit()
        setup_dask_for_coverage()


def is_coverage_mode():
    """Check if we're running in coverage mode."""
    return bool(os.environ.get("COVERAGE_PROCESS_START") or os.environ.get("PYTEST_COVERAGE"))
