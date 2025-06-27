"""
HPC Dask Helper: Utilities for High-Performance Computing with Dask
--------------------------------------------------------------------

This module provides utilities for setting up and managing Dask clusters
in HPC environments, with specific support for the DKRZ Levante Supercomputer.
"""

import os
import re
import subprocess
import psutil
from tempfile import TemporaryDirectory
from getpass import getuser
from pathlib import Path
from typing import Dict, Optional, Union, Any

import dask
import dask.array as dask_array
import xarray as xr
import numpy as np
from dask.distributed import Client, LocalCluster

try:
    from dask_jobqueue import SLURMCluster
except ImportError:
    SLURMCluster = None


# Default configuration values
DEFAULT_DASK_CONFIG = {
    'array.slicing.split_large_chunks': False,
    'distributed.comm.timeouts.connect': '120s',  # Increased from default
    'distributed.comm.timeouts.tcp': '240s',      # Double the connection timeout
    'distributed.comm.retry.count': 10,           # More retries before giving up
}

# DKRZ-specific paths and configuration
DKRZ_SCRATCH_PATH = Path('/scratch') / getuser()[0] / getuser() / 'clients'
DKRZ_LOG_PATH = Path('/home/b') / getuser() / '.log_trash'
DKRZ_ACCOUNT = 'bk1377'

# Memory configuration for different node types
MEMORY_CONFIGS = {
    256: {'client_memory': '250GB', 'constraint': '256', 'job_extra': ['--mem=0']},
    512: {'client_memory': '500GB', 'constraint': '512', 'job_extra': ['--constraint=512G --mem=0']},
    1024: {'client_memory': '1000GB', 'constraint': '1024', 'job_extra': ['--constraint=1024G --mem=0']}
}


def configure_dask(scratch_dir: Optional[Union[str, Path]] = None, 
                  config: Optional[Dict[str, Any]] = None) -> TemporaryDirectory:
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
    # Use provided scratch directory or default to DKRZ scratch
    scratch_path = Path(scratch_dir) if scratch_dir else DKRZ_SCRATCH_PATH
    
    # Create temporary directory
    if not scratch_path.exists():
        scratch_path.mkdir(parents=True, exist_ok=True)
    
    temp_dir = TemporaryDirectory(dir=scratch_path)
    
    # Apply default configuration
    dask.config.set(temporary_directory=temp_dir.name)
    print(f"Dask Scratch: {repr(temp_dir.name)}")
    
    # Apply default settings
    for key, value in DEFAULT_DASK_CONFIG.items():
        dask.config.set({key: value})
    
    # Apply any additional configuration
    if config:
        dask.config.set(config)
    
    return temp_dir


def get_cluster_info(client: Client) -> Dict[str, str]:
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
    """
    # Get hostname and dashboard port
    remote_node = subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip().split('.')[0]
    port = re.search(r':(\d+)/', client.dashboard_link).group(1)
    
    # Print connection information
    print(f"Hostname: {remote_node}")
    print(f"Forward Port: {remote_node}:{port}")
    print(f"Dashboard Link: localhost:{port}/status")
    
    return {
        'hostname': remote_node,
        'port': port,
        'dashboard_link': f"localhost:{port}/status"
    }


def start_local_cluster(n_workers: int = 4, threads_per_worker: int = 1, 
                       scratch_dir: Optional[Union[str, Path]] = None,
                       **kwargs) -> Client:
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
    **kwargs
        Additional keyword arguments to pass to LocalCluster.
    
    Returns
    -------
    Client
        Dask client connected to the local cluster.
    """
    # Configure Dask
    temp_dir = configure_dask(scratch_dir)
    
    # Check system resources
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory()
    
    # Warn if requested resources exceed available
    total_threads = n_workers * threads_per_worker
    if total_threads > physical_cores:
        print(f"Warning: Requested {n_workers} workers with {threads_per_worker} threads each, but only {physical_cores} physical cores available.")
        print("Hyper-threading can reduce performance for compute-intensive tasks!")
    elif total_threads > logical_cores:
        print(f"Warning: Requested {n_workers} workers with {threads_per_worker} threads each, but only {logical_cores} logical cores available.")
        print(f"Reducing to {logical_cores // threads_per_worker} workers.")
        n_workers = logical_cores // threads_per_worker
    
    print(f"Memory per Worker: {memory.total / n_workers / (1024**3):.2f} GB")

    # Create cluster and client
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, **kwargs)
    client = Client(cluster)
    
    # Store temporary directory (so it isn't garbage collected)
    client._temp_dir = temp_dir

    # Print connection information
    get_cluster_info(client)
    
    return client


def start_distributed_cluster(n_workers: int, workers_per_node: int, 
                             runtime: int = 9, node_memory: int = 256, 
                             dashboard_address: int = 8889, queue: str = 'compute', 
                             scratch_dir: Optional[Union[str, Path]] = None, 
                             account: Optional[str] = None, **kwargs) -> Client:
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
    """
    if SLURMCluster is None:
        raise ImportError("dask_jobqueue is required for SLURM cluster functionality. Install with: pip install dask_jobqueue")
    
    # Configure Dask
    temp_dir = configure_dask(scratch_dir)
    
    # Use default account if none specified
    if account is None:
        account = DKRZ_ACCOUNT
    
    # Validate node_memory
    if node_memory not in MEMORY_CONFIGS:
        raise ValueError(f"Unsupported node_memory value: {node_memory}. Must be one of {list(MEMORY_CONFIGS.keys())}.")
    
    config = MEMORY_CONFIGS[node_memory]
    
    # Calculate runtime in hours and minutes
    runtime_hrs = runtime // 60
    runtime_mins = runtime % 60
    
    # Create SLURM cluster
    cluster = SLURMCluster(
        name='dask-cluster',
        cores=workers_per_node,
        memory=config['client_memory'],
        processes=workers_per_node,  # One process per core
        interface='ib0',
        queue=queue,
        account=account,
        walltime=f'{runtime_hrs:02d}:{runtime_mins:02d}:00',
        asynchronous=0,
        job_extra_directives=config['job_extra'],
        log_directory=DKRZ_LOG_PATH,
        local_directory=temp_dir.name,
        scheduler_options={'dashboard_address': f':{dashboard_address}'},
        **kwargs
    )

    print(f"Memory per Worker: {node_memory / workers_per_node:.2f} GB")
    
    # Scale the cluster
    cluster.scale(n_workers)
    client = Client(cluster)
    
    # Store temporary directory (so it isn't garbage collected)
    client._temp_dir = temp_dir
    
    # Print connection information
    get_cluster_info(client)
    
    return client


def fix_dask_tuple_array(da):
    """
    Fix a dask array that has tuple (i.e. task) references in its chunks.
    This addresses a longstanding issue/bug when dask arrays are saved to Zarr.
    Process chunk by chunk to maintain memory efficiency.
    """
    
    # N.B.: Analyse the outputs of:
    #   first_key = result.data.__dask_keys__()[0]
    #   first_chunk = dask.compute(first_key)[0]
    #   print(type(first_chunk), first_chunk)
    
    def materialise_chunk(block):
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
        meta=np.array([], dtype=da.dtype)
    )
    
    # Create new DataArray with clean dask array
    return xr.DataArray(
        clean_data,
        dims=da.dims,
        coords=da.coords,
        attrs=da.attrs,
        name=da.name
    )