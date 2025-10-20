=================
Examples Gallery
=================

This section provides comprehensive examples of using marEx for various scenarios.

Jupyter Notebooks
=================

The Jupyter notebooks in the ``examples/`` directory demonstrate complete workflows:

Gridded Data Examples
---------------------

* `01_preprocess_extremes.ipynb <https://github.com/wienkers/marEx/blob/main/examples/gridded%20data/01_preprocess_extremes.ipynb>`_ - Data preprocessing for regular grids
* `02_id_track_events.ipynb <https://github.com/wienkers/marEx/blob/main/examples/gridded%20data/02_id_track_events.ipynb>`_ - Event identification and tracking
* `03_visualise_events.ipynb <https://github.com/wienkers/marEx/blob/main/examples/gridded%20data/03_visualise_events.ipynb>`_ - Visualisation and analysis

Regional Data Examples
----------------------

* `01_preprocess_extremes.ipynb <https://github.com/wienkers/marEx/blob/main/examples/regional%20data/01_preprocess_extremes.ipynb>`_ - Data preprocessing for regional analysis
* `02_id_track_events.ipynb <https://github.com/wienkers/marEx/blob/main/examples/regional%20data/02_id_track_events.ipynb>`_ - Regional event identification and tracking
* `03_visualise_events.ipynb <https://github.com/wienkers/marEx/blob/main/examples/regional%20data/03_visualise_events.ipynb>`_ - Regional data visualisation

Unstructured Data Examples
---------------------------

* `01_preprocess_extremes.ipynb <https://github.com/wienkers/marEx/blob/main/examples/unstructured%20data/01_preprocess_extremes.ipynb>`_ - Preprocessing for irregular meshes
* `02_id_track_events.ipynb <https://github.com/wienkers/marEx/blob/main/examples/unstructured%20data/02_id_track_events.ipynb>`_ - Tracking on unstructured grids
* `03_visualise_events.ipynb <https://github.com/wienkers/marEx/blob/main/examples/unstructured%20data/03_visualise_events.ipynb>`_ - Unstructured data visualisation

Batch Job Examples for HPC
===========================

marEx provides example scripts for running detection and tracking workflows on HPC systems using SLURM job schedulers. These scripts are designed to be run from a login node and will automatically launch distributed Dask clusters to process large datasets efficiently.

**Available Scripts:**

* `run_detect.py <https://github.com/wienkers/marEx/blob/main/examples/batch%20jobs/run_detect.py>`_ - Launches SLURM jobs for preprocessing and detecting marine extremes using ``marEx.preprocess_data()``
* `run_track.py <https://github.com/wienkers/marEx/blob/main/examples/batch%20jobs/run_track.py>`_ - Launches SLURM jobs for identifying and tracking extreme events using ``marEx.tracker()``

**Key Features:**

* Automatic SLURM cluster setup via ``marEx.helper.start_distributed_cluster()``
* Configuration through environment variables (workers, runtime, SLURM account)
* Complete workflow from data loading to saving results
* Designed for large-scale oceanographic datasets on HPC systems

**Configuration:**

Both scripts accept configuration via environment variables:

* ``DASK_N_WORKERS``: Number of Dask workers to request
* ``DASK_WORKERS_PER_NODE``: Workers per compute node
* ``DASK_RUNTIME``: Job wall-clock time in minutes
* ``SLURM_ACCOUNT``: SLURM account/project for billing

**Usage:**

See the :ref:`HPC Cluster Setup` section in the :doc:`user_guide` for detailed configuration instructions and usage examples.

**Customisation:**

These scripts are intended as templates. Copy and modify them to fit your specific:

* Data sources and file paths
* Preprocessing methods and parameters
* Tracking configuration
* HPC cluster specifications
