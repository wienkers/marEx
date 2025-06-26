import pytest
import dask
from dask.distributed import Client, LocalCluster


@pytest.fixture(scope="session")
def dask_client():
    """Create a Dask LocalCluster for tests."""
    # Configure Dask for testing
    dask.config.set({
        'distributed.worker.daemon': False,
        'distributed.admin.log-format': '%(name)s - %(levelname)s - %(message)s',
        'distributed.worker.memory.target': 0.8,
        'distributed.worker.memory.spill': 0.9,
        'distributed.worker.memory.pause': 0.95,
        'distributed.worker.memory.terminate': 0.98,
    })
    
    # Create a LocalCluster with limited resources for CI
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        memory_limit='1GB',
        dashboard_address=None,  # Disable dashboard in CI
        silence_logs=True
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
    dask.config.set({
        'array.chunk-size': '32MB',
        'array.slicing.split_large_chunks': True,
        'distributed.worker.memory.recent-to-old-time': '10s',
    })