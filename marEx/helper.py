"""
HPC Dask Helper: Utilities for High-Performance Computing with Dask
--------------------------------------------------------------------

This module provides utilities for setting up and managing Dask clusters
in HPC environments, with specific support for the DKRZ Levante Supercomputer.
"""

import logging
import re
import shutil
import subprocess
import uuid
from getpass import getuser
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any, Dict, Optional, Union

import dask
import dask.array as dask_array
import numpy as np
import psutil
import xarray as xr
from dask.distributed import Client, LocalCluster
from numpy.typing import NDArray

from .exceptions import ConfigurationError
from .logging_config import configure_logging, get_logger, is_verbose_mode, log_memory_usage, log_timing

# Get module logger
logger = get_logger(__name__)

try:
    from dask_jobqueue import SLURMCluster
except (ImportError, ValueError):
    # ValueError can occur when importing dask_jobqueue in worker threads
    # due to signal handling limitations
    SLURMCluster = None


# Default configuration values optimised for HPC environments
# These settings are tuned for large-scale climate data processing
# and have been tested on DKRZ Levante with marEx workloads
DEFAULT_DASK_CONFIG = {
    # Array processing
    "array.slicing.split_large_chunks": False,
    "array.chunk-size": "24MiB",  # Optimal chunk size for oceanographic data
    # Worker memory management
    "distributed.worker.memory.target": 0.4,  # Target memory threshold for spilling to disk
    "distributed.worker.memory.spill": 0.5,  # Spill to disk threshold
    "distributed.worker.memory.pause": 0.6,  # Pause worker threshold
    "distributed.worker.memory.terminate": 0.8,  # Terminate worker threshold
    "distributed.worker.memory.recent-to-old-time": "10s",  # Time to consider data old
    "distributed.worker.daemon": False,  # Workers are not daemons
    # Scheduler stability settings
    "distributed.scheduler.allowed-failures": 50,  # Allow many retries (common on HPC)
    "distributed.scheduler.work-stealing": False,  # Disable for deterministic execution
    "distributed.scheduler.worker-ttl": "600s",  # Keep workers alive for 10 minutes
    # Communication timeouts - increased for HPC network latency
    "distributed.comm.timeouts.connect": "300s",  # Connection timeout
    "distributed.comm.timeouts.tcp": "300s",  # TCP timeout
    "distributed.comm.retry.count": 15,  # More retries before giving up
    "distributed.comm.retry.delay.min": "3s",  # Min delay between retries
    "distributed.comm.retry.delay.max": "30s",  # Max delay between retries
    # Admin and logging
    "distributed.admin.log-format": "%(name)s - %(levelname)s - %(message)s",  # Log format
}

# DKRZ-specific paths and configuration
DKRZ_SCRATCH_PATH = Path("/scratch") / getuser()[0] / getuser() / "clients"
DKRZ_LOG_PATH = Path("/home/b") / getuser() / ".log_trash"
DKRZ_ACCOUNT = "bk1377"

# Memory configuration for different node types
MEMORY_CONFIGS = {
    256: {"client_memory": "250GB", "constraint": "256", "job_extra": ["--mem=0"]},
    512: {
        "client_memory": "500GB",
        "constraint": "512",
        "job_extra": ["--constraint=512G --mem=0"],
    },
    1024: {
        "client_memory": "1000GB",
        "constraint": "1024",
        "job_extra": ["--constraint=1024G --mem=0"],
    },
}


def configure_dask(
    scratch_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> TemporaryDirectory:  # pragma: no cover
    """
    Configure Dask with appropriate settings for HPC environments.

    Parameters
    ----------
    scratch_dir : str or Path, optional
        Directory to use for temporary files.
    config : dict, optional
        Additional Dask configuration settings to apply.

    Returns
    -------
    TemporaryDirectory
        Temporary directory object that should be kept alive while Dask is in use.
    """
    logger.info("Configuring Dask for HPC environment")

    # Use provided scratch directory or default to DKRZ scratch
    scratch_path = Path(scratch_dir) if scratch_dir else DKRZ_SCRATCH_PATH
    logger.debug(f"Using scratch directory: {scratch_path}")

    # Create temporary directory
    if not scratch_path.exists():
        logger.debug(f"Creating scratch directory: {scratch_path}")
        scratch_path.mkdir(parents=True, exist_ok=True)

    temp_dir = TemporaryDirectory(dir=scratch_path)
    logger.info(f"Dask temporary directory: {temp_dir.name}")

    # Apply default configuration
    dask.config.set(temporary_directory=temp_dir.name)

    # Apply default settings
    logger.debug("Applying default Dask configuration")
    for key, value in DEFAULT_DASK_CONFIG.items():
        dask.config.set({key: value})
        logger.debug(f"Set Dask config: {key} = {value}")

    # Apply any additional configuration
    if config:
        logger.debug(f"Applying additional Dask configuration: {config}")
        dask.config.set(config)

    logger.info("Dask configuration completed")
    return temp_dir


def get_cluster_info(client: Client) -> Dict[str, str]:  # pragma: no cover
    """
    Get and print cluster connection information.

    Parameters
    ----------
    client : Client
        Dask client connected to a cluster.

    Returns
    -------
    dict
        Dictionary containing connection information.

    Examples
    --------
    Basic cluster info retrieval:

    >>> import marEx
    >>>
    >>> # Start local cluster
    >>> client = marEx.start_local_cluster(n_workers=2)
    >>>
    >>> # Get connection information
    >>> info = marEx.get_cluster_info(client)
    Hostname: login01
    Forward Port: login01:8787
    Dashboard Link: localhost:8787/status
    >>>
    >>> print(f"Connect via: {info['dashboard_link']}")
    Connect via: localhost:8787/status
    >>> client.close()

    SSH tunneling for remote access:

    >>> # Start cluster on HPC system
    >>> client = marEx.start_distributed_cluster(
    ...     n_workers=8, workers_per_node=4, dashboard_address=8889
    ... )
    >>>
    >>> # Get tunneling information
    >>> info = marEx.get_cluster_info(client)
    Hostname: levante-login01
    Forward Port: levante-login01:8889
    Dashboard Link: localhost:8889/status
    >>>
    >>> # Use this info to set up SSH tunnel:
    >>> # ssh -L 8889:localhost:8889 levante-login01.dkrz.de
    >>> # Then access dashboard at localhost:8889/status
    >>>
    >>> client.close()

    Monitoring cluster status:

    >>> client = marEx.start_local_cluster(n_workers=4)
    >>> info = marEx.get_cluster_info(client)
    >>>
    >>> # Access cluster details
    >>> print(f"Dashboard URL: {client.dashboard_link}")
    >>> print(f"Cluster address: {client.cluster.scheduler_address}")
    >>> print(f"Total threads: {client.nthreads()}")
    >>>
    >>> # Use port for programmatic access
    >>> import requests
    >>> try:
    ...     response = requests.get(f"http://localhost:{info['port']}/info")
    ...     print(f"Scheduler info: {response.status_code}")
    ... except:
    ...     print("Dashboard not accessible")
    >>>
    >>> client.close()
    """
    # Get hostname and dashboard port
    remote_node = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip().split(".")[0]
    port_match = re.search(r":(\d+)/", client.dashboard_link)
    if port_match is None:
        raise ValueError(f"Could not extract port from dashboard link: {client.dashboard_link}")
    port = port_match.group(1)

    # Print connection information
    print(f"Hostname: {remote_node}")
    print(f"Forward Port: {remote_node}:{port}")
    print(f"Dashboard Link: localhost:{port}/status")

    return {
        "hostname": remote_node,
        "port": port,
        "dashboard_link": f"localhost:{port}/status",
    }


def start_local_cluster(
    n_workers: int = 4,
    threads_per_worker: int = 1,
    scratch_dir: Optional[Union[str, Path]] = None,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
    **kwargs,
) -> Client:  # pragma: no cover
    """
    Start a local Dask cluster.

    Parameters
    ----------
    n_workers : int, default=4
        Number of worker processes to start.
    threads_per_worker : int, default=1
        Number of threads per worker.
    scratch_dir : str or Path, optional
        Directory to use for temporary files.
    verbose : bool, optional
        Enable verbose logging with detailed cluster startup information.
        If None, uses current global logging configuration.
    quiet : bool, optional
        Enable quiet logging with minimal output (warnings and errors only).
        If None, uses current global logging configuration.
        Note: quiet takes precedence over verbose if both are True.
    **kwargs
        Additional keyword arguments to pass to LocalCluster.

    Returns
    -------
    Client
        Dask client connected to the local cluster.

    Examples
    --------
    Basic local cluster for development:

    >>> import marEx
    >>>
    >>> # Start simple local cluster
    >>> client = marEx.start_local_cluster(n_workers=2, threads_per_worker=1)
    >>> print(client)
    <Client: 'tcp://127.0.0.1:xxxxx' processes=2 threads=2, memory=15.7 GB>
    >>>
    >>> # Check cluster status
    >>> print(f"Dashboard: {client.dashboard_link}")
    Dashboard: http://127.0.0.1:8787/status
    >>>
    >>> client.close()

    Optimised cluster for CPU-intensive work:

    >>> # Use one worker per physical core
    >>> client = marEx.start_local_cluster(
    ...     n_workers=8,           # Number of physical cores
    ...     threads_per_worker=1   # Avoid hyperthreading for compute
    ... )
    >>>
    >>> # Process data with the cluster
    >>> import xarray as xr
    >>> data = xr.open_dataset('large_data.nc').chunk({'time': 25})
    >>> result = data.mean().compute()
    >>> client.close()

    Memory-optimised cluster:

    >>> # Configure for large datasets
    >>> client = marEx.start_local_cluster(
    ...     n_workers=4,
    ...     threads_per_worker=2,
    ...     memory_limit='8GB',      # Limit memory per worker
    ...     scratch_dir='/tmp/dask'  # Fast (e.g. Lustre) local storage
    ... )

    Integration with marEx preprocessing:

    >>> # Start cluster then process data
    >>> client = marEx.start_local_cluster(n_workers=16)
    >>>
    >>> # Load and preprocess SST data
    >>> sst = xr.open_dataset('sst_data.nc').sst.chunk({'time': 30})
    >>> processed = marEx.preprocess_data(sst, threshold_percentile=95)
    >>>
    >>> # Track events using the cluster
    >>> tracker = marEx.tracker(
    ...     processed.extreme_events,
    ...     processed.mask,
    ...     R_fill=8,
    ...     area_filter_quartile=0.5
    ... )
    >>> events = tracker.run()
    >>>
    >>> print(f"Processed {len(sst.time)} time steps with {client.nthreads()} threads")
    >>> client.close()

    Custom worker configuration:

    >>> # Advanced configuration for specific workloads
    >>> client = marEx.start_local_cluster(
    ...     n_workers=4,
    ...     threads_per_worker=2,
    ...     processes=True,         # Use separate processes (default)
    ...     silence_logs=False,     # Keep logs for debugging
    ...     dashboard_address=':8787'  # Specific dashboard port
    ... )
    """
    # Configure logging if verbose/quiet parameters are provided
    if verbose is not None or quiet is not None:
        configure_logging(verbose=verbose, quiet=quiet)

    logger.info(f"Starting local Dask cluster with {n_workers} workers, {threads_per_worker} threads each")

    # Enhanced logging in verbose mode
    if is_verbose_mode():
        logger.debug(f"System resources - CPUs: {psutil.cpu_count()}, Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        logger.debug(f"Cluster configuration: n_workers={n_workers}, threads_per_worker={threads_per_worker}")

    # Configure Dask
    temp_dir = configure_dask(scratch_dir)

    # Check system resources
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory()

    logger.info(
        f"System resources: {physical_cores} physical cores, {logical_cores} logical cores, "
        f"{memory.total / 1024**3:.1f}GB total memory"
    )

    # Warn if requested resources exceed available
    total_threads = n_workers * threads_per_worker
    if total_threads > physical_cores:
        logger.warning(
            f"Requested {n_workers} workers with {threads_per_worker} threads each, "
            f"but only {physical_cores} physical cores available"
        )
        logger.warning("Hyper-threading can reduce performance for compute-intensive tasks!")
    elif total_threads > logical_cores:
        logger.warning(
            f"Requested {n_workers} workers with {threads_per_worker} threads each, "
            f"but only {logical_cores} logical cores available"
        )
        n_workers = logical_cores // threads_per_worker
        logger.info(f"Reducing to {n_workers} workers")

    memory_per_worker = memory.total / n_workers / (1024**3)
    logger.info(f"Memory per worker: {memory_per_worker:.2f} GB")

    # Create cluster and client
    logger.debug("Creating local cluster and client")

    # Configure Bokeh session token expiration via dask config
    # Set to 24 hours to prevent Bokeh 3.8+ token expiration during long notebooks
    dask.config.set({"distributed.scheduler.dashboard.bokeh-application.session_token_expiration": 86400000})
    logger.debug("Set Bokeh session token expiration to 24 hours (86400000ms)")

    with log_timing(logger, "Local cluster startup", log_memory=True, show_progress=True):
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, **kwargs)
        client = Client(cluster)

    # Store temporary directory (so it isn't garbage collected)
    client._temp_dir = temp_dir

    # Get and display connection information
    get_cluster_info(client)

    # Enhanced verbose mode reporting
    if is_verbose_mode():
        if hasattr(client, "cluster") and client.cluster is not None:
            logger.debug(f"Cluster address: {client.cluster.scheduler_address}")
        logger.debug(f"Dashboard URL: {client.dashboard_link}")
        logger.debug(f"Workers: {list(client.nthreads().keys())}")
        logger.debug(f"Total threads: {sum(client.nthreads().values())}")

    logger.info(f"Local cluster started successfully - Dashboard: {client.dashboard_link}")
    log_memory_usage(logger, "After cluster startup", level=20)  # DEBUG level

    return client


def start_distributed_cluster(
    n_workers: int,
    workers_per_node: int,
    runtime: int = 9,
    node_memory: int = 256,
    dashboard_address: int = 8889,
    queue: str = "compute",
    scratch_dir: Optional[Union[str, Path]] = None,
    account: Optional[str] = None,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
    **kwargs,
) -> Client:  # pragma: no cover
    """
    Start a distributed Dask cluster on a SLURM-based supercomputer.

    Parameters
    ----------
    n_workers : int
        Total number of workers to request.
    workers_per_node : int
        Number of workers per node.
    runtime : int, default=9
        Maximum runtime in minutes.
    node_memory : int, default=256
        Memory per node in GB (256, 512, or 1024).
    dashboard_address : int, default=8889
        Port for the Dask dashboard.
    queue : str, default='compute'
        SLURM queue to submit jobs to.
    scratch_dir : str or Path, optional
        Directory to use for temporary files.
    account : str, optional
        SLURM account to charge. Defaults to DKRZ_ACCOUNT.
    **kwargs
        Additional keyword arguments to pass to SLURMCluster.

    Returns
    -------
    Client
        Dask client connected to the distributed cluster.

    Examples
    --------
    Basic SLURM cluster for medium-scale processing:

    >>> import marEx
    >>>
    >>> # Start cluster with 16 workers on 4 nodes (4 workers per node)
    >>> client = marEx.start_distributed_cluster(
    ...     n_workers=16,
    ...     workers_per_node=4,
    ...     runtime=60,      # 1 hour
    ...     node_memory=128   # 128GB nodes
    ... )
    >>> print(f"Cluster: {client}")
    >>> # Access dashboard via SSH tunnel
    >>> cluster_info = marEx.get_cluster_info(client)
    >>> client.close()

    Processing large marEx workflow on HPC:

    >>> # Start cluster for full marEx pipeline
    >>> client = marEx.start_distributed_cluster(
    ...     n_workers=32,
    ...     workers_per_node=8,
    ...     runtime=60,  # 1 hour
    ...     node_memory=256   # 256GB nodes
    ... )
    >>>
    >>> # Load very large SST dataset
    >>> import xarray as xr
    >>> sst = xr.open_zarr('/work/data/large_sst.zarr').chunk({'time': 25})
    >>>
    >>> # Preprocess with distributed computing
    >>> processed = marEx.preprocess_data(
    ...     sst,
    ...     method_anomaly="shifting_baseline",
    ...     threshold_percentile=95
    ... )
    >>>
    >>> # Track events using full cluster
    >>> tracker = marEx.tracker(
    ...     processed.extreme_events,
    ...     processed.mask,
    ...     R_fill=12,
    ...     area_filter_quartile=0.3
    ... )
    >>> events = tracker.run()
    >>>
    >>> # Save results
    >>> events.to_zarr('/work/results/tracked_events.zarr')
    >>> client.close()

    Custom SLURM configuration:

    >>> # Advanced SLURM configuration
    >>> client = marEx.start_distributed_cluster(
    ...     n_workers=64,
    ...     workers_per_node=32,
    ...     runtime=20,           # 20 minutes
    ...     node_memory=256,
    ...     account='myproject',   # Custom account
    ...     queue='debug',           # debug queue
    ... )

    Dashboard access and monitoring:

    >>> # Start cluster and set up monitoring
    >>> client = marEx.start_distributed_cluster(
    ...     n_workers=16,
    ...     workers_per_node=4,
    ...     dashboard_address=8890  # Custom dashboard port
    ... )
    >>>
    >>> # Get connection info for SSH tunneling
    >>> info = marEx.get_cluster_info(client)
    >>> print(f"SSH tunnel: ssh -L {info['port']}:localhost:{info['port']} {info['hostname']}")
    >>> print(f"Dashboard: {info['dashboard_link']}")
    >>>
    >>> # Monitor cluster performance
    >>> print(f"Workers: {len(client.scheduler_info()['workers'])}")
    >>> print(f"Total memory: {sum(w['memory_limit'] for w in client.scheduler_info()['workers'].values()) / 1e9:.1f} GB")

    Memory optimisation strategies:

    >>> # Optimise for different workload types
    >>>
    >>> # Memory-intensive: fewer workers per node
    >>> memory_cluster = marEx.start_distributed_cluster(
    ...     n_workers=8, workers_per_node=2, node_memory=512
    ... )
    >>>
    >>> # CPU-intensive: more workers per node
    >>> cpu_cluster = marEx.start_distributed_cluster(
    ...     n_workers=64, workers_per_node=16, node_memory=256
    ... )
    """
    logger.info(f"Starting distributed SLURM cluster - {n_workers} workers on {n_workers//workers_per_node} nodes")
    logger.info(f"Configuration: {workers_per_node} workers/node, {node_memory}GB memory/node, {runtime}h runtime")

    if SLURMCluster is None:
        from ._dependencies import require_dependencies

        logger.error("dask_jobqueue not available - cannot create SLURM cluster")
        require_dependencies(["dask_jobqueue"], "SLURM cluster functionality")

    # Configure Dask
    temp_dir = configure_dask(scratch_dir)

    # Use default account if none specified
    if account is None:
        account = DKRZ_ACCOUNT

    # Validate node_memory
    if node_memory not in MEMORY_CONFIGS:
        logger.error(f"Unsupported node_memory value: {node_memory}")
        raise ConfigurationError(
            "Unsupported node_memory configuration",
            details=f"Value '{node_memory}' is not supported",
            suggestions=[
                f"Use one of the supported configurations: {list(MEMORY_CONFIGS.keys())}",
                "Check SLURM system documentation for available memory sizes",
            ],
            context={
                "provided_value": node_memory,
                "supported_values": list(MEMORY_CONFIGS.keys()),
            },
        )

    config = MEMORY_CONFIGS[node_memory]
    logger.debug(f"Using memory configuration: {config}")

    # Calculate runtime in hours and minutes
    runtime_hrs = runtime // 60
    runtime_mins = runtime % 60
    logger.debug(f"SLURM walltime: {runtime_hrs:02d}:{runtime_mins:02d}:00")

    # Create SLURM cluster
    logger.info("Creating SLURM cluster")

    # Configure Bokeh session token expiration via dask config
    # Set to 24 hours to prevent Bokeh 3.8+ token expiration during long notebooks
    dask.config.set({"distributed.scheduler.dashboard.bokeh-application.session_token_expiration": 86400000})
    logger.debug("Set Bokeh session token expiration to 24 hours (86400000ms)")

    # Merge user-provided scheduler_options from kwargs if present
    user_scheduler_options = kwargs.pop("scheduler_options", {})
    scheduler_options = {"dashboard_address": f":{dashboard_address}", **user_scheduler_options}

    logger.debug(f"Scheduler options: {scheduler_options}")

    with log_timing(logger, "SLURM cluster creation"):
        cluster = SLURMCluster(
            name="dask-cluster",
            cores=workers_per_node,
            memory=config["client_memory"],
            processes=workers_per_node,  # One process per core
            interface="ib0",
            queue=queue,
            account=account,
            walltime=f"{runtime_hrs:02d}:{runtime_mins:02d}:00",
            asynchronous=0,
            job_extra_directives=config["job_extra"],
            log_directory=DKRZ_LOG_PATH,
            local_directory=temp_dir.name,
            scheduler_options=scheduler_options,
            **kwargs,
        )

    memory_per_worker = node_memory / workers_per_node
    logger.info(f"Memory per worker: {memory_per_worker:.2f} GB")

    # Scale the cluster
    logger.info(f"Scaling cluster to {n_workers} workers")
    cluster.scale(n_workers)
    client = Client(cluster)

    # Store temporary directory (so it isn't garbage collected)
    client._temp_dir = temp_dir

    # Print connection information
    get_cluster_info(client)
    logger.info("Distributed cluster started successfully")

    return client


def checkpoint_to_zarr(
    data: Union[xr.DataArray, xr.Dataset],
    name: str = "checkpoint",
    cleanup: bool = False,
    timedim: str = "time",
) -> Union[xr.DataArray, xr.Dataset]:  # pragma: no cover
    """
    Save and reload a Dask-backed xarray object to break graph dependencies.

    This function materialises a Dask array/dataset to a temporary file
    and immediately reloads it, thereby breaking the computational graph.
    This prevents expensive recomputations when the same data is used multiple
    times downstream.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Dask-backed xarray object to checkpoint
    name : str, default='checkpoint'
        Name prefix for the temporary file (for logging/debugging)
    cleanup : bool, default=False
        Whether to delete the temporary file after reloading.
        By default (False), temp files are kept for the session and cleaned up
        by the OS temp directory manager. Set to True to immediately delete after reload.
    timedim : str, default='time'
        Name of the time dimension for chunking adjustments

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Reloaded data with broken graph dependencies

    Examples
    --------
    >>> import marEx
    >>> anomalies = marEx.compute_normalised_anomaly(sst)
    >>> anomalies_checkpointed = marEx.helper.checkpoint_to_zarr(
    ...     anomalies, name="anomalies"
    ... )
    """
    logger.debug(f"Checkpointing '{name}' to break graph dependencies")

    # Get dask temporary directory, fallback to system temp
    try:
        temp_base = dask.config.get("temporary-directory", None)
        if temp_base is None or not Path(temp_base).exists():
            temp_base = gettempdir()
    except Exception:
        temp_base = gettempdir()

    unique_id = uuid.uuid4().hex[:8]
    file_path = None

    try:
        try:
            zarr_path = Path(temp_base) / f"marEx_checkpoint_{name}_{unique_id}.zarr"
            file_path = zarr_path

            logger.debug(f"Attempting Zarr checkpoint: {zarr_path}")
            # Check if time dimension has irregular chunks that need fixing
            if timedim in data.dims and timedim in data.chunks:
                time_chunks = data.chunks[timedim]
                if len(time_chunks) > 1 and len(set(time_chunks)) > 1:
                    # Chunks are irregular - need to fix for Zarr
                    first_chunk = time_chunks[0]
                    total_size = data.sizes[timedim]

                    logger.debug(f"Irregular {timedim} chunks detected: {time_chunks}")
                    logger.debug(f"Total {timedim} dimension size: {total_size}")

                    # Calculate how many full chunks we can have
                    n_full_chunks = total_size // first_chunk
                    remainder = total_size % first_chunk

                    if remainder > 0:
                        # Need full chunks + one smaller final chunk
                        new_time_chunks = (first_chunk,) * n_full_chunks + (remainder,)
                    else:
                        # All chunks are equal size
                        new_time_chunks = (first_chunk,) * n_full_chunks

                    data = data.chunk({timedim: new_time_chunks})
                    logger.debug(f"Adjusted {timedim} chunks for Zarr: {new_time_chunks}")
                    logger.debug(f"Verification - sum of chunks: {sum(new_time_chunks)}, dimension size: {total_size}")

            with log_timing(logger, f"Saving '{name}' to Zarr", logging.DEBUG, log_memory=False):
                data.to_zarr(zarr_path, mode="w")

            logger.debug("Zarr save successful, reloading...")
            with log_timing(logger, f"Reloading '{name}' from Zarr", logging.DEBUG, log_memory=False):
                if isinstance(data, xr.Dataset):
                    reloaded = xr.open_zarr(zarr_path, chunks={})
                else:
                    ds_temp = xr.open_zarr(zarr_path, chunks={})
                    reloaded = ds_temp[list(ds_temp.data_vars)[0]]

            logger.info(f"Checkpoint '{name}' saved via Zarr: {zarr_path}")
            return reloaded

        except (ValueError, OSError) as e:
            if "incompatible" in str(e) or "chunk" in str(e).lower():
                logger.warning(f"Zarr failed due to irregular chunks: {str(e)[:200]}")
                # Clean up failed zarr attempt
                if zarr_path.exists():
                    shutil.rmtree(zarr_path)
            else:
                raise

    except Exception as e:
        logger.error(f"Failed to checkpoint '{name}' to disk: {e}")
        logger.warning(f"Falling back to in-memory persist() only for '{name}'")

        # Fallback to in-memory persist (no disk I/O)
        try:
            reloaded = data.persist()
            from distributed import wait

            wait(reloaded)
            logger.info(f"Checkpoint '{name}' persisted to distributed memory (no disk)")
            return reloaded
        except Exception as e2:
            logger.error(f"Even persist() failed for '{name}': {e2}")
            logger.warning("Returning original data without checkpointing")
            return data

    finally:
        # Cleanup if requested
        if cleanup and file_path and file_path.exists():
            try:
                if file_path.suffix == ".zarr":
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()
                logger.debug(f"Cleaned up checkpoint file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")


def fix_dask_tuple_array(da: xr.DataArray) -> xr.DataArray:
    """
    Fix a dask array that has tuple (i.e. task) references in its chunks.
    This addresses a longstanding issue/bug when dask arrays are saved to Zarr.
    Process chunk by chunk to maintain memory efficiency.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with Dask array backend that may have tuple chunk references

    Returns
    -------
    xarray.DataArray
        DataArray with materialised chunks that can be safely saved to Zarr

    """
    # N.B.: Analyse the outputs of:
    #   first_key = result.data.__dask_keys__()[0]
    #   first_chunk = dask.compute(first_key)[0]
    #   print(type(first_chunk), first_chunk)

    def materialise_chunk(block: NDArray[Any]) -> NDArray[Any]:  # pragma: no cover
        """Force materialisation of a single chunk."""
        # This ensures we return an actual numpy array, not a task reference
        return np.asarray(block)

    chunks = da.chunks

    # Use map_blocks to process each chunk
    clean_data = dask_array.map_blocks(
        materialise_chunk,
        da.data,
        dtype=da.dtype,
        chunks=chunks,
        drop_axis=[],  # Keep all axes
        meta=np.array([], dtype=da.dtype),
    )

    # Create new DataArray with clean dask array
    return xr.DataArray(clean_data, dims=da.dims, coords=da.coords, attrs=da.attrs, name=da.name)
