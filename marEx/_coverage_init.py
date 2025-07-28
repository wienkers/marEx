"""Coverage initialization for subprocess/multiprocessing support.

This module ensures coverage measurement works correctly in Dask workers
and other subprocesses created during testing.
"""

import os


def initialize_coverage():
    """Initialize coverage in subprocesses if COVERAGE_PROCESS_START is set."""
    # Only try to initialize coverage in worker processes, not the main process
    if (os.environ.get("COVERAGE_PROCESS_START") or os.environ.get("PYTEST_COVERAGE")) and os.environ.get("DASK_WORKER_PROCESS"):
        try:
            import coverage

            coverage.process_startup()
        except ImportError:
            # Coverage not available, silently continue
            pass
        except Exception:
            # Any other coverage-related error, silently continue
            pass


def disable_numba_if_coverage():
    """Disable Numba JIT compilation if running under coverage."""
    if os.environ.get("COVERAGE_PROCESS_START") or os.environ.get("PYTEST_COVERAGE"):
        # Disable Numba JIT compilation for coverage measurement
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        os.environ["NUMBA_CACHE_DIR"] = "/tmp"  # Avoid cache issues

        try:
            import numba

            numba.config.DISABLE_JIT = True
            # Also disable numba globally
            numba.config.NUMBA_DISABLE_JIT = 1
        except ImportError:
            pass

        # Disable numbagg which is used by xarray for quantile operations
        try:
            import numbagg
            import numpy as np

            # Monkey-patch numbagg to use pure numpy versions
            if not hasattr(numbagg, "_coverage_original_nanquantile"):
                numbagg._coverage_original_nanquantile = numbagg.nanquantile

                def pure_numpy_nanquantile(a, quantiles, axis=None, **kwargs):
                    """Pure numpy replacement for numbagg.nanquantile."""
                    return np.nanquantile(a, quantiles, axis=axis)

                numbagg.nanquantile = pure_numpy_nanquantile

                # Also patch other numbagg functions
                if hasattr(numbagg, "nanpercentile"):
                    numbagg._coverage_original_nanpercentile = numbagg.nanpercentile

                    def pure_numpy_nanpercentile(a, percentiles, axis=None, **kwargs):
                        """Pure numpy replacement for numbagg.nanpercentile."""
                        return np.nanpercentile(a, percentiles, axis=axis)

                    numbagg.nanpercentile = pure_numpy_nanpercentile
        except ImportError:
            pass


# Only disable Numba - don't initialize coverage in main process
disable_numba_if_coverage()
