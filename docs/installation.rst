============
Installation
============

marEx needs Python 3.10 to 3.13 and runs on Linux, macOS and Windows. Install the core package from PyPI:

.. code-block:: bash

   pip install marEx

The core install covers all three modules (anomalies, extremes, tracking) and the plotting accessor. The dependencies include ``dask[complete]>=2025.9.0``, ``zarr>=2.18``, ``flox>=0.10.1``, ``numba`` and ``xarray``.

Optional Extras
===============

.. code-block:: bash

   pip install "marEx[plotting]"      # seaborn, cmocean, ffmpeg (animations)
   pip install "marEx[hpc]"           # dask_jobqueue, psutil (SLURM clusters)
   pip install "marEx[performance]"   # jax, jaxlib
   pip install "marEx[full]"          # all of the above
   pip install "marEx[dev]"           # tests, linting, docs build

Animations need the ``ffmpeg`` binary on your path (``apt-get install ffmpeg``, ``brew install ffmpeg`` or ``choco install ffmpeg``).

The ``hpc`` extra provides ``marEx.helper.start_distributed_cluster``, which builds a SLURM cluster through ``dask_jobqueue``. Set ``queue``, ``account`` and ``node_memory`` for your system, or build your own ``dask.distributed`` cluster and pass the client to marEx.

JAX
---

JAX is optional and builds the sparse dilation matrix for the unstructured-grid tracker. Without JAX, ``import marEx`` emits an ``ImportWarning`` and the routine falls back to NumPy.

Zarr 2 and Zarr 3
=================

Both major versions of zarr-python are supported and tested. Zarr 3 requires Python 3.12 or newer, so on Python 3.10 and 3.11 pip resolves zarr 2. Outputs written to Zarr use the writing store's default compressor.

Development Install
===================

.. code-block:: bash

   git clone https://github.com/wienkers/marEx.git
   cd marEx
   pip install -e ".[dev]"
   pre-commit install

Checking the Installation
=========================

.. code-block:: python

   import marEx

   print(marEx.__version__)
   marEx.print_dependency_status()      # which optional packages were found
   print(marEx.has_dependency("jax"))

Upgrading
=========

.. code-block:: bash

   pip install --upgrade "marEx[full]"

Coming from 4.x? Several names changed in 5.0. See :doc:`whats_new` for the migration table.
