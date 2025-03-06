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
dask_scratch = Path('/scratch') / getuser()[0] / getuser() / 'clients' ## N.B.: This is only for DKRZ Machine
dask_tmp_dir = TemporaryDirectory(dir=dask_scratch)
dask.config.set(temporary_directory=dask_tmp_dir.name)
dask.config.set({'array.slicing.split_large_chunks': False})
dask.config.set({
    'distributed.comm.timeouts.connect': '120s',  # Increase from default
    'distributed.comm.timeouts.tcp': '240s',      # Double the connection timeout
    'distributed.comm.retry.count': 10,           # More retries before giving up
})

# Make LocalCluster
def StartLocalCluster(n_workers=4, n_threads=1, dask_scratch=None):
    
    if not dask_scratch:
        dask_tmp_dir = TemporaryDirectory(dir=dask_scratch)
        dask.config.set(temporary_directory=dask_tmp_dir.name)
    
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


# Make Distributed Cluster
def StartDistributedCluster(n_workers, workers_per_node, runtime=9, node_memory=256, dashboard_address=8889, queue='compute', cluster_scratch=None):
    
    if not cluster_scratch:
        dask_tmp_dir = TemporaryDirectory(dir=cluster_scratch)
        dask.config.set(temporary_directory=dask_tmp_dir.name)

    if node_memory == 256:
        client_memory = '250GB'
        constraint_memory = '256'
    elif node_memory == 512:
        client_memory = '500GB'
        constraint_memory = '512'
    elif node_memory == 1024:
        client_memory = '1000GB'
        constraint_memory = '1024'
    else:
        print("Memory not defined")  
    
    runtime_hrs = runtime // 60
    runtime_mins = runtime % 60

    ## Distributed Cluster (without GPU)
    clusterDistributed = SLURMCluster(name='dask-cluster',
                                        cores=workers_per_node,
                                        memory=client_memory,
                                        processes=workers_per_node,  # Only 1 thread
                                        interface='ib0',
                                        queue=queue,
                                        account='bk1377',
                                        walltime=f'{runtime_hrs:02d}:{runtime_mins:02d}:00',
                                        asynchronous=0,
                                        job_extra_directives = [f'--constraint={constraint_memory}G --mem=0'] if node_memory != 256 else [f' --mem=0'],
                                        log_directory=f'/home/b/{getuser()}/.log_trash',
                                        local_directory=dask_tmp_dir.name,
                                        scheduler_options={'dashboard_address': ':{0}'.format(dashboard_address)},)

    print(f"Memory per Worker: {node_memory / workers_per_node:.2f} GB")
    
    clusterDistributed.scale(n_workers)
    clientDistributed = Client(clusterDistributed)
    
    remote_node = subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip().split('.')[0]
    port = re.search(r':(\d+)/', clientDistributed.dashboard_link).group(1)
    print('Hostname is ', remote_node)
    print(f"Forward Port = {remote_node}:{port}")
    print(f"Dashboard Link: localhost:{port}/status")
    
    return clientDistributed
