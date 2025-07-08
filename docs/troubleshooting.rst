===============
Troubleshooting
===============

This guide helps you diagnose and resolve common issues when using marEx for marine extreme event analysis.

.. contents:: Table of Contents
   :local:
   :depth: 3

Quick Diagnostic Checklist
===========================

Before diving into specific issues, run through this quick checklist:

**Environment Check:**

.. code-block:: python

   import marEx

   # Check version
   print(f"marEx version: {getattr(marEx, '__version__', 'development')}")

   # Check dependencies
   marEx.print_dependency_status()

   # Check Dask
   import dask
   print(f"Dask version: {dask.__version__}")

**Data Validation:**

.. code-block:: python

   # Check data structure
   print(f"Data dimensions: {your_data.dims}")
   print(f"Data shape: {your_data.shape}")
   print(f"Data coordinates: {list(your_data.coords)}")
   print(f"Is Dask array: {marEx.is_dask_collection(your_data.data)}")

Installation Issues
===================

Package Not Found
------------------

**Problem**: ``ModuleNotFoundError: No module named 'marEx'``

**Solutions**:

1. **Install marEx**::

    pip install marEx

2. **Check Python environment**::

    which python
    pip list | grep marEx

3. **Virtual environment issues**::

    # Activate correct environment
    conda activate your-env
    # or
    source your-venv/bin/activate

4. **Development installation**::

    # If working with source code
    pip install -e ./marEx

Dependency Conflicts
--------------------

**Problem**: Version conflicts with scientific packages

**Solutions**:

1. **Create clean environment**::

    conda create -n marex-env python=3.10
    conda activate marex-env
    pip install marEx[full]

2. **Update dependencies**::

    pip install --upgrade marEx
    pip install --upgrade dask xarray

3. **Pin problematic versions**::

    pip install "dask>=2024.7.0,<2026.0.0"

Missing Optional Dependencies
-----------------------------

**Problem**: Features not working due to missing optional packages

**Solutions**:

1. **Install full package**::

    pip install marEx[full]

2. **Install specific features**::

    pip install marEx[dev]     # Development tools
    pip install jax jaxlib     # GPU acceleration
    pip install dask-jobqueue  # HPC support

3. **Check what's missing**::

    python -c "import marEx; marEx.print_dependency_status()"


Coordinate Issues
-----------------

**Problem**: ``KeyError: 'lat'`` or coordinate not found

**Solutions**:

1. **Check coordinate names**::

    print(data.coords)
    print(data.dims)

2. **For unstructured data**::

    # Ensure lat/lon are coordinates, not dimensions
    print(f"Spatial dimensions: {[d for d in data.dims if d not in ['time']]}")

Chunking Problems
-----------------

**Problem**: ``ValueError: Can't rechunk from chunks`` or memory issues

**Solutions**:

1. **Check current chunks**::

    print(f"Current chunks: {data.chunks}")

2. **Rechunk appropriately**::

    # For preprocessing
    data = data.chunk({'time': 30, 'lat': -1, 'lon': -1})

3. **Avoid very small chunks**::

    # Bad: too many small chunks
    data = data.chunk({'time': 1, 'lat': 1, 'lon': 1})

    # Good: balanced chunks
    data = data.chunk({'time': 'auto', 'lat': -1, 'lon': -1})

Processing Issues
=================

Memory Errors
-------------

**Problem**: ``MemoryError`` or ``KilledWorker`` during processing

**Solutions**:

1. **Reduce chunk sizes**::

    # Smaller chunks use less memory
    data = data.chunk({'time': 30, 'lat': 100, 'lon': 100})

2. **Reduce worker memory**::

    client = marEx.helper.start_local_cluster(
        n_workers=2,
        memory_limit='4GB'  # Reduce from default
    )

3. **Use spill-to-disk**::

    import dask
    dask.config.set({'distributed.worker.memory.spill': 0.8})

Slow Performance
----------------

**Problem**: Processing takes much longer than expected

**Solutions**:

1. **Analyse the Dask Dashboard**::

2. **Optimise chunks for operation**::

    # For preprocessing (time series operations)
    data = data.chunk({'time': 1000, 'lat': 'auto', 'lon': 'auto'})

    # For spatial operations
    data = data.chunk({'time': 30, 'lat': -1, 'lon': -1})

3. **Use more workers**::

    client = marEx.helper.start_local_cluster(
        n_workers=min(32, os.cpu_count()),
        threads_per_worker=1
    )

4. **Profile performance**::

    from dask.distributed import performance_report

    with performance_report(filename="marex-profile.html"):
        result = marEx.preprocess_data(data)


Wrong Results
-------------

**Problem**: Unexpected values or patterns in results

**Solutions**:

1. **Check input data quality**::

    print(f"Data range: {data.min().values} to {data.max().values}")
    print(f"Missing values: {data.isnull().sum().values}")

2. **Validate preprocessing parameters**::

    # Check baseline period
    baseline_data = data.sel(time=slice('1990', '2020'))
    if len(baseline_data.time) < 365 * 10:
        print("Warning: Baseline period too short")

3. **Check anomaly mean**::

    # Anomalies should have near-zero mean
    anomaly_mean = processed['dat_anomaly'].mean().values
    if abs(anomaly_mean) > 0.1:
        print(f"Warning: Anomaly mean not zero: {anomaly_mean}")

4. **Verify extreme frequency**::

    # Should be close to threshold percentile
    extreme_freq = processed['extreme_events'].mean().values * 100
    expected_freq = 100 - threshold_percentile
    if abs(extreme_freq - expected_freq) > 2:
        print(f"Warning: Extreme frequency {extreme_freq:.1f}% != expected {expected_freq}%")


Worker Failures
---------------

**Problem**: Workers dying or becoming unresponsive

**Solutions**:

1. **Check system resources**::

    import psutil
    print(f"CPU usage: {psutil.cpu_percent()}%")
    print(f"Memory usage: {psutil.virtual_memory().percent}%")
    print(f"Available memory: {psutil.virtual_memory().available / 1e9:.1f} GB")

2. **Reduce worker load**::

    client = marEx.helper.start_local_cluster(
        n_workers=8,           # Fewer workers
        threads_per_worker=1,  # Fewer threads
        memory_limit='4GB'     # Less memory per worker
    )

3. **Configure worker limits**::

    dask.config.set({
        'distributed.worker.memory.target': 0.8,
        'distributed.worker.memory.spill': 0.9,
        'distributed.worker.memory.pause': 0.95,
        'distributed.worker.memory.terminate': 0.98
    })


HPC-Specific Issues
===================

SLURM Job Failures
-------------------

**Problem**: Jobs killed or failing on HPC systems

**Solutions**:

1. **Check resource limits**::

    scontrol show job $SLURM_JOB_ID

2. **Increase walltime**::

    #SBATCH --time=12:00:00

3. **Request more memory**::

    #SBATCH --mem=128G

4. **Use exclusive nodes**::

    #SBATCH --exclusive


Performance Issues
==================

General Performance Tips
-------------------------

1. **Tune your cluster**::

    # Don't use too many small workers
    # Better: fewer workers with more resources
    client = marEx.helper.start_local_cluster(
        n_workers=16,
        threads_per_worker=1,
        memory_limit='16GB'
    )

2. **Optimise chunk sizes**::

    # Target 100-400 MB chunks
    # Check with: data.nbytes / 1e6

3. **Monitor progress**::

    from dask.distributed import progress
    progress(result)

Getting Help
============

Community Resources
-------------------

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Documentation**: Check latest documentation online
- **Examples**: Browse example notebooks and scripts

Reporting Issues
----------------

When reporting issues, include:

1. **marEx version**: ``marEx.__version__``
2. **Python version**: ``python --version``
3. **Operating system**: OS and version
4. **Data description**: Size, format, structure
5. **Full error message**: Complete traceback
6. **Minimal example**: Simplified code that reproduces the issue
