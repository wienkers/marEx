from tempfile import TemporaryDirectory
from getpass import getuser
from pathlib import Path
import os

import subprocess
import re
import psutil

import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster, wait
from dask import persist

# Dask Cluster Wrappers

# Dask Config
cluster_scratch = Path('/scratch') / getuser()[0] / getuser() / 'clients'
dask_tmp_dir = TemporaryDirectory(dir=cluster_scratch)
dask.config.set(temporary_directory=dask_tmp_dir.name)
dask.config.set({'array.slicing.split_large_chunks': False})

# Make LocalCluster
def StartLocalCluster(n_workers=4, n_threads=1):
    
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory()
    
    
    if n_workers * n_threads > physical_cores:
        print(f"Warning: Requested {n_workers} workers with {n_threads} threads each, but there are only {physical_cores} physical cores.")
        print("Only hyper-thread if you know what you're doing!")
    elif n_workers * n_threads > logical_cores:
        print(f"Warning: Requested {n_workers} workers with {n_threads} threads each, but only {logical_cores} logical cores available.")
        print(f"Setting n_workers = {physical_cores // n_threads}")
        n_workers = physical_cores // n_threads
    
    print(f"Memory per Worker: {memory.total / n_workers / (1024**3):.2f} GB")

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=n_threads)
    client = Client(cluster)

    remote_node = subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip().split('.')[0]
    port = re.search(r':(\d+)/', client.dashboard_link).group(1)
    print('Hostname is ', remote_node)
    print(f"Forward Port = {remote_node}:{port}")
    print(f"Dashboard Link: localhost:{port}/status")
    
    return client
