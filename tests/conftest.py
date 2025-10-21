"""Test configuration and fixtures for marEx package."""

# Configure coverage environment BEFORE any other imports
from .coverage_config import setup_coverage_environment

setup_coverage_environment()

import gc
import logging
import time

import dask
import numpy as np
import pytest
from dask.distributed import Client, LocalCluster


# Configure logging to suppress noisy distributed messages
def pytest_configure(config):
    """Configure pytest with custom logging settings."""
    # Configure matplotlib to use non-interactive backend for CI/testing
    # This must be done before importing pyplot to avoid TkAgg errors on Windows
    import matplotlib

    matplotlib.use("Agg")

    # Suppress all distributed logging to ERROR level or higher
    # This removes all the noisy INFO messages from distributed components
    distributed_loggers = [
        "distributed",  # Root distributed logger
        "distributed.protocol.core",
        "distributed.comm.tcp",
        "distributed.worker",
        "distributed.scheduler",
        "distributed.nanny",
        "distributed.core",
        "distributed.batched",
        "distributed.http.proxy",
        "distributed.client",
        "distributed.utils",
        "distributed.comm.core",
        "distributed.shuffle",
        "distributed.stealing",
        "distributed.metrics",
    ]

    for logger_name in distributed_loggers:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    # Also suppress tornado logging which can be noisy
    logging.getLogger("tornado").setLevel(logging.ERROR)

    # Create a custom filter to catch any remaining protocol messages
    class DistributedFilter(logging.Filter):
        def filter(self, record):
            # Filter out specific message patterns
            message = record.getMessage()
            if "Failed to serialize" in message:
                return False
            if "Starting worker" in message and record.levelname == "INFO":
                return False
            if "Registered to:" in message and record.levelname == "INFO":
                return False
            if "dashboard at:" in message and record.levelname == "INFO":
                return False
            return True

    # Add the filter to the root logger to catch everything
    logging.getLogger().addFilter(DistributedFilter())


@pytest.fixture(scope="session")
def dask_client():
    """Create a Dask LocalCluster for tests."""
    import os

    # Detect if running under coverage
    is_coverage = os.environ.get("COVERAGE_PROCESS_START") or os.environ.get("PYTEST_COVERAGE")

    if is_coverage:
        # For coverage: use threads scheduler with a single-threaded LocalCluster
        # This ensures all code executes in the main process where coverage is active
        # but still provides a distributed client for operations like wait()
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,  # Use threads, not processes
            memory_limit="8GB",
            dashboard_address=None,
            silence_logs=True,
        )
        client = Client(cluster)
        yield client

        # Cleanup
        client.close()
        cluster.close()
    else:
        # Normal testing: use distributed client
        dask.config.set(scheduler="distributed")

        # Configure Dask for testing
        dask.config.set(
            {
                "distributed.worker.daemon": False,
                "distributed.admin.log-format": "%(name)s - %(levelname)s - %(message)s",
                "distributed.worker.memory.target": 0.8,
                "distributed.worker.memory.spill": 0.9,
                "distributed.worker.memory.pause": 0.95,
                "distributed.worker.memory.terminate": 0.98,
                # Add more aggressive memory management
                "distributed.worker.memory.recent-to-old-time": "3s",
                "distributed.worker.memory.rebalance.measure": "managed_in_memory",
                # Optimise task scheduling for memory-intensive workflows
                "distributed.scheduler.allowed-failures": 20,  # Increased from 10
                "distributed.comm.timeouts.connect": "120s",  # Increased from 60s
                "distributed.comm.timeouts.tcp": "120s",  # Increased from 60s
                # Add additional robustness for coverage runs
                "distributed.worker.multiprocessing.initializer": None,
                "distributed.worker.multiprocessing.initialize": None,
                "distributed.comm.retry.count": 5,
                "distributed.comm.retry.delay.min": "1s",
                "distributed.comm.retry.delay.max": "10s",
                "distributed.scheduler.work-stealing": False,  # Disable work stealing for stability
            }
        )

        cluster = LocalCluster(
            n_workers=2,  # Reduced from 4 for better stability
            threads_per_worker=1,
            memory_limit="6GB",  # Increased from 3GB for better stability
            dashboard_address=None,  # Disable dashboard in CI
            silence_logs=True,
            # Add explicit process handling for coverage compatibility
            processes=True,
            protocol="tcp",
        )

        client = Client(cluster)

        yield client

        # Cleanup
        client.close()
        cluster.close()


@pytest.fixture(scope="session", autouse=True)
def configure_dask():
    """Configure Dask for testing environment."""
    # Use synchronous scheduler for small computations
    dask.config.set(
        {
            "array.chunk-size": "32MB",
            "array.slicing.split_large_chunks": True,
            "distributed.worker.memory.recent-to-old-time": "10s",
            # Add optimisation settings for complex computations
            "optimization.fuse.ave-width": 2,
            "optimization.fuse.max-width": 4,
            "optimization.fuse.max-depth": 4,
            # Optimise for memory-intensive reductions
            "array.chunk-options.split-every": {"reduction-dimension": 4},
        }
    )


# Statistical Test Helper Functions
def assert_percentile_frequency(
    frequency,
    expected_percentile,
    tolerance_std=2.0,
    sample_size=None,
    description=None,
):
    """
    Test that observed frequency matches expected percentile with statistical
    significance.

    For a percentile threshold (e.g., 95th percentile), we expect the frequency
    to be close to (100 - percentile)/100. This uses a binomial distribution to
    test if the observed frequency is statistically reasonable.

    Parameters:
    -----------
    frequency : float
        Observed frequency of extreme events (between 0 and 1)
    expected_percentile : float
        The percentile threshold used (e.g., 95 for 95th percentile)
    tolerance_std : float, default=2.0
        Number of standard deviations to allow as tolerance
    sample_size : int, optional
        Sample size for more precise statistical test. If None, uses approximate test.

    Raises:
    -------
    AssertionError : If frequency is outside statistically expected range
    """
    expected_frequency = (100 - expected_percentile) / 100.0

    if sample_size is not None:
        # Use exact binomial test with known sample size
        # Calculate confidence interval for binomial distribution
        std_error = np.sqrt(expected_frequency * (1 - expected_frequency) / sample_size)
        lower_bound = expected_frequency - tolerance_std * std_error
        upper_bound = expected_frequency + tolerance_std * std_error

        desc_suffix = f" ({description})" if description else ""
        assert lower_bound <= frequency <= upper_bound, (
            f"Extreme frequency {frequency:.4f} is outside statistically "
            f"expected range [{lower_bound:.4f}, {upper_bound:.4f}] for "
            f"{expected_percentile}th percentile (expected: "
            f"{expected_frequency:.4f}, sample_size: {sample_size})"
            f"{desc_suffix}"
        )
    else:
        # Use approximate test with reasonable tolerance for unknown sample size
        # For typical dataset sizes, allow broader tolerance
        relative_tolerance = 0.20  # 20% relative tolerance
        absolute_tolerance = max(0.005, expected_frequency * relative_tolerance)

        lower_bound = expected_frequency - absolute_tolerance
        upper_bound = expected_frequency + absolute_tolerance

        desc_suffix = f" ({description})" if description else ""
        assert lower_bound <= frequency <= upper_bound, (
            f"Extreme frequency {frequency:.4f} is outside expected range "
            f"[{lower_bound:.4f}, {upper_bound:.4f}] for "
            f"{expected_percentile}th percentile (expected: "
            f"{expected_frequency:.4f} ± {absolute_tolerance:.4f})"
            f"{desc_suffix}"
        )


def assert_reasonable_bounds(
    value,
    expected_value,
    tolerance_relative=0.1,
    tolerance_absolute=None,
    description="value",
):
    """
    Test that a value is within reasonable bounds of an expected value.

    Uses relative tolerance by default, with optional absolute tolerance override.

    Parameters:
    -----------
    value : float
        Observed value to test
    expected_value : float
        Expected/reference value
    tolerance_relative : float, default=0.1
        Relative tolerance as fraction (e.g., 0.1 = 10%)
    tolerance_absolute : float, optional
        If provided, uses absolute tolerance instead of relative
    description : str
        Description of what is being tested for error messages
    """
    if tolerance_absolute is not None:
        lower_bound = expected_value - tolerance_absolute
        upper_bound = expected_value + tolerance_absolute
        tolerance_desc = f"±{tolerance_absolute}"
    else:
        tolerance_abs = abs(expected_value * tolerance_relative)
        lower_bound = expected_value - tolerance_abs
        upper_bound = expected_value + tolerance_abs
        tolerance_desc = f"±{tolerance_relative*100:.1f}%"

    assert lower_bound <= value <= upper_bound, (
        f"{description} {value} is outside reasonable bounds "
        f"[{lower_bound:.4f}, {upper_bound:.4f}] "
        f"(expected: {expected_value} {tolerance_desc})"
    )


def assert_count_in_reasonable_range(count, expected_count, tolerance=2):
    """
    Test that an integer count is within a reasonable range of expected count.

    For counting discrete objects (events, objects, etc.) where small variations
    are expected due to algorithmic differences or test data variations.

    Parameters:
    -----------
    count : int
        Observed count
    expected_count : int
        Expected count
    tolerance : int, default=2
        Absolute tolerance for count differences
    """
    lower_bound = expected_count - tolerance
    upper_bound = expected_count + tolerance

    assert lower_bound <= count <= upper_bound, (
        f"Count {count} is outside reasonable range [{lower_bound}, {upper_bound}] " f"(expected: {expected_count} ± {tolerance})"
    )


def assert_statistical_consistency(
    data_array,
    expected_property,
    test_type="mean",
    tolerance_std=2.0,
    description="data",
):
    """
    Test statistical properties of data arrays for consistency.

    Parameters:
    -----------
    data_array : array-like
        Data to test
    expected_property : float
        Expected value of the statistical property
    test_type : str
        Type of test: 'mean', 'std', 'median', 'percentile_95', etc.
    tolerance_std : float
        Number of standard deviations for tolerance
    description : str
        Description for error messages
    """
    data = np.array(data_array).flatten()
    data = data[~np.isnan(data)]  # Remove NaN values

    if test_type == "mean":
        observed = np.mean(data)
        # Use t-test for mean comparison
        std_error = np.std(data) / np.sqrt(len(data))
        tolerance = tolerance_std * std_error
    elif test_type == "std":
        observed = np.std(data)
        tolerance = expected_property * 0.2  # 20% tolerance for std
    elif test_type == "median":
        observed = np.median(data)
        tolerance = np.std(data) / np.sqrt(len(data)) * tolerance_std
    else:
        raise ValueError(f"Unsupported test_type: {test_type}")

    lower_bound = expected_property - tolerance
    upper_bound = expected_property + tolerance

    assert lower_bound <= observed <= upper_bound, (
        f"{description} {test_type} {observed:.4f} is outside expected range "
        f"[{lower_bound:.4f}, {upper_bound:.4f}] (expected: {expected_property:.4f})"
    )


@pytest.fixture(scope="module")
def dask_client_per_module():
    """Dask client scoped per test module for isolation."""
    import os

    # Detect if running under coverage
    is_coverage = os.environ.get("COVERAGE_PROCESS_START") or os.environ.get("PYTEST_COVERAGE")

    if is_coverage:
        # For coverage: use threads scheduler with a single-threaded LocalCluster
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,  # Use threads, not processes
            memory_limit="8GB",
            dashboard_address=None,
            silence_logs=True,
        )
        client = Client(cluster)
        yield client

        # Cleanup
        client.close()
        cluster.close()
    else:
        # Normal testing
        client = None
        try:
            client = Client(
                n_workers=2,
                threads_per_worker=1,
                memory_limit="7GB",
                silence_logs=logging.WARNING,
                dashboard_address=None,
                # Add more conservative settings for parallel execution
                timeout="300s",
                heartbeat_interval="30s",
            )
            # Let client fully initialise
            client.wait_for_workers(2, timeout=60)
            yield client
        finally:
            if client:
                try:
                    # Ensure all computations are finished before closing
                    client.cancel(client.list_datasets(), force=True)
                    client.restart()  # Clean restart before close
                    time.sleep(1)  # Brief pause
                    client.close(timeout=10)
                    client.cluster.close(timeout=10)
                except Exception as e:
                    print(f"Warning: Error during Dask client cleanup: {e}")
            gc.collect()
            time.sleep(0.5)


# Create module-specific fixtures
@pytest.fixture(scope="module")
def dask_client_unstructured(dask_client_per_module):
    """Dedicated Dask client for unstructured tests."""
    return dask_client_per_module


@pytest.fixture(scope="module")
def dask_client_integration(dask_client_per_module):
    """Dedicated Dask client for integration tests."""
    return dask_client_per_module


@pytest.fixture(scope="module")
def dask_client_gridded(dask_client_per_module):
    """Dedicated Dask client for gridded tests."""
    return dask_client_per_module
