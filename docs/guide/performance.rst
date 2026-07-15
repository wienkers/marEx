=================
Performance & HPC
=================

See :doc:`../api/helper` for the full cluster-management reference.

Performance Optimisation
========================

This section provides guidance for optimising marEx performance across different computing environments and dataset sizes.

Memory Management and Chunking
-------------------------------

Chunking Strategy
~~~~~~~~~~~~~~~~~

Optimal chunking strategies vary by dataset size and operation type:

.. code-block:: python

   # Small datasets (< 10GB): Chunk time dimension
   sst = sst.chunk({'time': 365, 'lat': -1, 'lon': -1})

   # Large datasets (> 100GB): Chunk all dimensions
   sst = sst.chunk({'time': 365, 'lat': 180, 'lon': 360})

   # For tracking: Use smaller time chunks
   sst = sst.chunk({'time': 25, 'lat': -1, 'lon': -1})  # Recommended for tracker

Memory Requirements Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Operation
     - Memory (per core)
     - Optimal Time Chunk
     - Notes
   * - ``preprocess_data``
     - 2-4 GB
     - 365+ days
     - Depends on method_anomaly
   * - ``tracker``
     - 1-2 GB
     - 25-50 days
     - Keep spatial dims unbounded (-1)
   * - Exact percentiles
     - 8-16 GB
     - Full time series
     - Requires careful chunking
   * - Approximate percentiles
     - 1-2 GB
     - Any
     - Memory efficient

Exact vs Approximate Percentile Methods
----------------------------------------

When to Use Each Method
~~~~~~~~~~~~~~~~~~~~~~~

**APPROXIMATE (default)**: For large datasets

.. code-block:: python

   extremes = marEx.preprocess_data(
       sst,
       method_percentile='approximate',
       precision=0.01,        # ~0.01°C bins
       max_anomaly=5.0,       # Histogram range ±5°C
       dask_chunks={'time': 365}  # Any chunking works
   )
   # ✓ Memory efficient (1-2 GB/core)
   # ✓ Fast parallel computation
   # ✓ ~0.01°C precision (sufficient for most studies)

**EXACT**: For small datasets or high precision needs

.. code-block:: python

   extremes_exact = marEx.preprocess_data(
       sst,
       method_percentile='exact',
       dask_chunks={'time': -1, 'lat': 100, 'lon': 100}  # Careful chunking required!
   )
   # ⚠ High memory (8-16 GB/core), depending on time-series length
   # ⚠ Slower computation
   # ✓ Mathematically exact percentiles


JAX Acceleration
----------------

When JAX Helps Most
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import marEx

   # Check JAX availability
   if marEx.has_dependency('jax'):
       print("JAX acceleration active")
       # Best for:
       # - Harmonic detrending (10-50× speedup)
       # - Large spatial operations
       # - GPU/TPU available systems
   else:
       print("Install JAX: pip install marEx[full]")
       # Falls back to NumPy (still fast with Numba JIT)

Performance Gains
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Operation
     - NumPy+Numba
     - JAX (CPU)
     - JAX (GPU)
   * - Harmonic detrend
     - 1× (baseline)
     - 15× faster
     - 50× faster
   * - Percentile calc
     - 1×
     - 1× (similar)
     - 2-3× faster
   * - Tracking
     - 1×
     - 1× (similar)
     - N/A

HPC Cluster Setup
-----------------

SLURM Distributed Cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~

For supercomputers:

.. code-block:: python

   import marEx

   # Start distributed Dask cluster on SLURM
   client = marEx.helper.start_distributed_cluster(
       n_workers=64,           # Number of workers
       cores_per_worker=4,     # Cores per worker
       memory_per_worker='8GB' # Memory per worker
   )

   # Get dashboard URL for monitoring
   marEx.helper.get_cluster_info(client)

   # Run analysis
   extremes = marEx.preprocess_data(sst, ...)
   events = marEx.tracker(extremes.extreme_events, ...).run()

   # Clean up
   client.close()

Local Cluster
~~~~~~~~~~~~~

For workstations:

.. code-block:: python

   # Auto-detect system resources
   client = marEx.helper.start_local_cluster(
       n_workers=8,            # Leave some cores for system
       threads_per_worker=2,
       memory_limit='16GB'     # Per worker
   )

Example Batch Scripts
~~~~~~~~~~~~~~~~~~~~~

The ``examples/batch jobs/`` directory provides ready-to-use scripts designed to be run from a login node on HPC systems with SLURM:

**Workflow:**

1. **run_detect.py**: Launches a distributed Dask job to execute the preprocessing and extreme detection pipeline
2. **run_track.py**: Launches a distributed Dask job for event identification and tracking

Both scripts utilise ``marEx.helper.start_distributed_cluster()`` for cluster management and are configured via environment variables.

**Configuration Environment Variables:**

* ``DASK_N_WORKERS``: Number of Dask workers to launch (default varies by script: 128 for detect, 32 for track)
* ``DASK_WORKERS_PER_NODE``: Workers per SLURM node (default varies by script: 64 for detect, 32 for track)
* ``DASK_RUNTIME``: Maximum runtime in minutes (default: 39 for detect, 89 for track)
* ``SLURM_ACCOUNT``: SLURM account for billing (default: 'bk1377')
* **Additional for run_track.py**: ``RUN_BASIC_TRACKER``, ``GRID_RESOLUTION``, ``AREA_FILTER``, ``R_FILL``, ``T_FILL``, ``OVERLAP_THRESHOLD``

**Example Usage:**

.. code-block:: bash

   # Run preprocessing on SLURM cluster
   export DASK_N_WORKERS=256
   export DASK_RUNTIME=120
   export SLURM_ACCOUNT=my_project
   python examples/batch\ jobs/run_detect.py

   # Run tracking after preprocessing completes
   export DASK_N_WORKERS=64
   export DASK_RUNTIME=180
   python examples/batch\ jobs/run_track.py

**Customisation:**

These scripts are designed to be copied and modified for your specific:

* Data file paths and chunk sizes
* Preprocessing parameters (anomaly method, thresholds)
* Tracking parameters (spatial filters, merge/split settings)
* Cluster resources (workers, memory, runtime)

The scripts handle cluster setup, data processing, and saving results to zarr/netCDF files on the scratch filesystem.

Memory-Safe Processing of Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Large datasets (e.g. full-resolution global OSTIA or regridded ICON output) are
processed without any special configuration: ``marEx.preprocess_data`` and the
tracker are designed to keep each Dask task's working set bounded so the pipeline
runs at full resolution within reasonable per-worker memory.

The histogram-based percentile kernels (the ``approximate`` method used by both
``global_extreme`` and ``hobday_extreme``) are the most memory-intensive step. They
internally tile the spatial dimensions before building the per-cell histograms, so a
single task never has to hold the entire ``(time × space)`` field at once -- regardless
of how the input was chunked. As a result, no manual graph-breaking or checkpointing is
required, and the returned dataset can be written straight to Zarr or NetCDF.

.. note::

   Earlier versions exposed a ``use_temp_checkpoints`` parameter that round-tripped
   intermediate arrays through temporary Zarr stores to work around memory blow-ups in
   the histogram step. That workaround has been removed: the underlying memory issue is
   fixed at the algorithm level, so the parameter is no longer needed (or accepted).

**Saving results:**

.. code-block:: python

   import xarray as xr
   import marEx

   # Load a large dataset (chunk only the time dimension; spatial dims stay contiguous)
   sst = xr.open_zarr('large_dataset.zarr', chunks={'time': 30}).sst

   extremes = marEx.preprocess_data(
       sst,
       method_anomaly='shifting_baseline',
       method_extreme='hobday_extreme',
       threshold_percentile=95,
       dask_chunks={'time': 25},
   )

   # The returned dataset writes directly to either format:
   extremes.to_zarr('extremes.zarr', mode='w')
   extremes.to_netcdf('extremes.nc')

The HPC helper module
=====================

The helper module is designed to simplify the deployment and management of marEx workflows
on supercomputing systems. It provides tools for cluster management, memory optimisation,
and performance tuning specifically tailored for DKRZ Levante and similar HPC environments.

**Key Features:**

* **SLURM Integration**: Seamless integration with SLURM job schedulers
* **Automated Cluster Management**: Start, configure, and monitor Dask clusters
* **Memory Optimisation**: Intelligent memory management for large datasets
* **Performance Tuning**: Optimised configurations for HPC environments
* **Dashboard Integration**: Jupyter dashboard tunneling and monitoring
* **Resource Detection**: Automatic system resource detection and optimisation

Basic Usage Examples
====================

SLURM Cluster Setup
-------------------

.. code-block:: python

   import marEx
   from marEx.helper import start_distributed_cluster

   # Start a SLURM cluster optimised for marEx workflows
   cluster, client = start_distributed_cluster(
       n_workers=20,
       cores_per_worker=4,
       memory_per_worker='16GB',
       walltime='02:00:00',
       queue='compute',
       project='your_project_id'
   )

   # Now run marEx workflows with the cluster
   extremes_ds = marEx.preprocess_data(
       sst_data,
       threshold_percentile=95,
       dask_chunks={'time': 365, 'lat': 200, 'lon': 200}
   )

   # Close cluster when done
   cluster.close()
   client.close()

Local Cluster Setup
-------------------

.. code-block:: python

   from marEx.helper import start_local_cluster

   # Start a local cluster with automatic resource detection
   cluster, client = start_local_cluster(
       n_workers=8,
       threads_per_worker=2,
       memory_limit='32GB',
       dashboard_port=8787
   )

   # Get cluster information
   cluster_info = marEx.helper.get_cluster_info(client)
   print(f"Dashboard: {cluster_info['dashboard_url']}")
   print(f"Workers: {cluster_info['n_workers']}")
   print(f"Total Memory: {cluster_info['total_memory']}")

Dask Configuration
------------------

.. code-block:: python

   from marEx.helper import configure_dask

   # Configure Dask for HPC environment
   configure_dask(
       temporary_directory='/scratch/tmp',
       memory_limit='64GB',
       threads_per_worker=4,
       dashboard_port=8787,
       silence_logs=True
   )

Advanced HPC Configurations
============================

DKRZ Levante Optimisation
-------------------------

.. code-block:: python

   # Optimised configuration for DKRZ Levante supercomputer
   cluster, client = start_distributed_cluster(
       n_workers=40,
       cores_per_worker=8,
       memory_per_worker='32GB',
       walltime='04:00:00',
       queue='compute',
       project='your_project',
       # Levante-specific optimisations
       extra_directives=[
           '#SBATCH --exclusive',
           '#SBATCH --partition=compute'
       ],
       worker_extra_args=[
           '--local-directory=/scratch/tmp',
           '--memory-limit=30GB'  # Leave memory for system overhead
       ]
   )

Multi-Node Scaling
------------------

.. code-block:: python

   # Large-scale configuration for century-long datasets
   cluster, client = start_distributed_cluster(
       n_workers=100,               # Scale to many workers
       cores_per_worker=4,
       memory_per_worker='16GB',
       walltime='08:00:00',
       queue='compute',
       project='projectID',
       # Advanced scaling options
       adaptive_scaling=True,       # Enable adaptive scaling
       minimum_workers=20,          # Minimum number of workers
       maximum_workers=200,         # Maximum number of workers
       scale_factor=2.0            # Scaling responsiveness
   )

Performance Tuning
==================

Chunk Size Optimisation
-----------------------

.. code-block:: python

   def calculate_optimal_chunks(data_shape, n_workers, memory_per_worker_gb):
       """Calculate optimal chunk sizes for marEx workflows."""
       # Time chunks should be manageable for individual workers
       time_chunks = min(365, data_shape[0] // n_workers)

       # Spatial chunks based on available memory
       # Assuming 8 bytes per float64 value
       max_spatial_elements = (memory_per_worker_gb * 1e9 / 8) / time_chunks
       spatial_chunks = int(np.sqrt(max_spatial_elements))

       return {
           'time': time_chunks,
           'lat': min(spatial_chunks, data_shape[1]),
           'lon': min(spatial_chunks, data_shape[2])
       }

   # Use with marEx preprocessing
   chunks = calculate_optimal_chunks(
       sst.shape,
       n_workers=20,
       memory_per_worker_gb=16
   )

   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       dask_chunks=chunks
   )

Memory Management
-----------------

.. code-block:: python

   # Configure Dask for memory-constrained environments
   configure_dask(
       # Memory limits
       memory_limit='32GB',
       memory_target_fraction=0.8,     # Use 80% of available memory
       memory_spill_fraction=0.9,      # Spill at 90% usage
       memory_pause_fraction=0.95,     # Pause at 95% usage

       # Temporary storage
       temporary_directory='/scratch/dask-tmp',

       # Garbage collection
       garbage_collection_interval='10s',

       # Network optimisation
       tcp_timeout='300s',
       heartbeat_interval='5s'
   )

Dashboard and Monitoring
========================

Cluster Monitoring
------------------

.. code-block:: python

   # Monitor cluster status and performance
   cluster_info = marEx.helper.get_cluster_info(client)

   print("=== Cluster Status ===")
   print(f"Status: {cluster_info['status']}")
   print(f"Active Workers: {cluster_info['n_workers']}")
   print(f"Total Cores: {cluster_info['total_cores']}")
   print(f"Total Memory: {cluster_info['total_memory']}")
   print(f"Dashboard URL: {cluster_info['dashboard_url']}")

   # Check if SSH tunnel is needed
   if cluster_info.get('needs_tunnel'):
       print(f"SSH tunnel command: {cluster_info['tunnel_command']}")

Performance Monitoring
----------------------

.. code-block:: python

   # Monitor task performance during computation
   def monitor_computation(client, computation_future):
       """Monitor and report computation progress."""
       import time

       start_time = time.time()
       while not computation_future.done():
           # Get current status
           status = client.scheduler_info()

           # Report progress
           print(f"Tasks: {status['tasks']}")
           print(f"Workers: {len(status['workers'])}")
           print(f"Memory: {status['memory']} / {status['memory_limit']}")

           time.sleep(30)  # Update every 30 seconds

       end_time = time.time()
       print(f"Computation completed in {end_time - start_time:.1f} seconds")

   # Use during marEx operations
   future = client.compute(extremes_ds, sync=False)
   monitor_computation(client, future)
   result = future.result()

Data Management
===============

Temporary Directory Setup
-------------------------

.. code-block:: python

   import tempfile
   import os
   from pathlib import Path

   # Set up optimised temporary directory structure
   def setup_temp_directories(base_path='/scratch'):
       """Set up temporary directories for optimal I/O performance."""
       user = os.environ.get('USER', 'unknown')

       # Create user-specific temp directory
       temp_base = Path(base_path) / user / 'marEx_tmp'
       temp_base.mkdir(parents=True, exist_ok=True)

       # Set up subdirectories
       dask_temp = temp_base / 'dask'
       marEx_temp = temp_base / 'marEx'

       dask_temp.mkdir(exist_ok=True)
       marEx_temp.mkdir(exist_ok=True)

       return {
           'dask_temp': str(dask_temp),
           'marEx_temp': str(marEx_temp),
           'base_temp': str(temp_base)
       }

   # Configure for HPC environment
   temp_dirs = setup_temp_directories('/scratch')

   configure_dask(
       temporary_directory=temp_dirs['dask_temp']
   )

Zarr Storage Optimisation
-------------------------

.. code-block:: python

   # Optimise Zarr storage for intermediate results
   def setup_zarr_storage(output_path, chunk_config):
       """Configure optimised Zarr storage for marEx results."""
       import zarr

       # Configure Zarr compressor for climate data
       compressor = zarr.Blosc(
           cname='lz4',        # Fast compression
           clevel=3,           # Moderate compression level
           shuffle=1           # Byte shuffle for better compression
       )

       # Set up encoding for xarray to_zarr
       encoding = {}
       for var in ['dat_anomaly', 'extreme_events', 'thresholds']:
           encoding[var] = {
               'compressor': compressor,
               'chunks': tuple(chunk_config.values())
           }

       return encoding

   # Use with marEx workflows
   encoding = setup_zarr_storage('./results.zarr', chunks)
   extremes_ds.to_zarr('./extremes.zarr', encoding=encoding)

Integration with Job Schedulers
===============================

SLURM Integration
-----------------

.. code-block:: python

   # Advanced SLURM configuration
   cluster, client = start_distributed_cluster(
       # Basic configuration
       n_workers=50,
       cores_per_worker=8,
       memory_per_worker='32GB',
       walltime='06:00:00',

       # SLURM-specific options
       queue='compute',
       project='climate_research',
       job_name='marEx_analysis',

       # Advanced SLURM directives
       extra_directives=[
           '#SBATCH --exclusive',
           '#SBATCH --constraint=haswell',
           '#SBATCH --mail-type=END,FAIL',
           '#SBATCH --mail-user=user@institution.edu'
       ],

       # Worker optimisation
       worker_extra_args=[
           '--local-directory=/scratch/dask-temp',
           '--memory-limit=30GB',
           '--nthreads=8',
           '--death-timeout=300'
       ]
   )

PBS/Torque Integration
----------------------

.. code-block:: python

   # Example for PBS/Torque systems (implementation may vary)
   from dask_jobqueue import PBSCluster

   def start_pbs_cluster(n_workers=20, cores_per_worker=24, memory_per_worker='64GB'):
       """Start PBS cluster for marEx workflows."""
       cluster = PBSCluster(
           cores=cores_per_worker,
           memory=memory_per_worker,
           queue='normal',
           walltime='04:00:00',
           job_extra_directives=[
               '-l nodes=1:ppn=24',
               '-A climate_project'
           ],
           local_directory='/scratch/dask-temp'
       )

       cluster.scale(n_workers)
       client = Client(cluster)

       return cluster, client


Performance Optimisation
------------------------

.. code-block:: python

   # Performance optimisation checklist:
   # 1. Use local SSD storage for temporary files
   # 2. Optimise network settings for multi-node clusters
   # 3. Use appropriate compression for I/O
   # 4. Monitor task scheduling efficiency
   # 5. Balance compute vs I/O intensive operations

   configure_dask(
       # Network optimisation
       tcp_timeout='300s',
       heartbeat_interval='5s',

       # I/O optimisation
       temporary_directory='/local_ssd/dask-tmp',

       # Task optimisation
       distributed_scheduler_allowed_failures=3,
       distributed_worker_daemon=False
   )

HPC troubleshooting
===================

Common Issues and Solutions
---------------------------

**Cluster Startup Issues**:

.. code-block:: python

   # Issue: Workers not starting
   # Solution: Check SLURM queue status and resource availability

   # Debug cluster startup
   cluster, client = start_distributed_cluster(
       n_workers=10,
       cores_per_worker=4,
       memory_per_worker='16GB',
       walltime='02:00:00',
       queue='debug',  # Use debug queue for troubleshooting
       log_directory='./logs'  # Enable logging
   )

**Memory Errors**:

.. code-block:: python

   # Issue: Out of memory errors
   # Solution: Reduce chunk sizes and worker memory usage

   # Conservative memory configuration
   configure_dask(
       memory_limit='8GB',
       memory_target_fraction=0.7,
       memory_spill_fraction=0.8,
       temporary_directory='/scratch/spill'
   )

**Network Timeouts**:

.. code-block:: python

   # Issue: Network timeouts in large clusters
   # Solution: Increase timeout values

   configure_dask(
       tcp_timeout='600s',
       heartbeat_interval='10s',
       comm_retry_delay_min='1s',
       comm_retry_delay_max='20s'
   )

**Dashboard Access Issues**:

.. code-block:: python

   # Issue: Cannot access dashboard from login node
   # Solution: Set up SSH tunnel

   cluster_info = get_cluster_info(client)
   if cluster_info.get('needs_tunnel'):
       tunnel_cmd = cluster_info['tunnel_command']
       print(f"Run this command on your local machine:")
       print(tunnel_cmd)
       print(f"Then access dashboard at: http://localhost:8787")
