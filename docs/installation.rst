============
Installation
============

Installation Profiles
=====================

Full Installation (Complete Feature Set)
----------------------------------------

For research environments with comprehensive analysis workflows and maximum functionality:

.. code-block:: bash

   # Complete installation with all optional dependencies
   pip install marEx[full]


HPC Installation (Supercomputing Environments)
----------------------------------------------

For deployment on SLURM clusters, cloud computing, or multi-node processing scaling to 1000+ cores with optimised memory management:

.. code-block:: bash

   # Complete HPC setup with all features
   pip install marEx[full,hpc]


Development Installation (Contributing to MarEx)
------------------------------------------------

For contributing to MarEx, customizing algorithms, or debugging:

.. code-block:: bash

   # Clone and install for development
   git clone https://github.com/wienkers/marEx.git
   cd marEx
   pip install -e .[dev]

   # Install pre-commit hooks
   pre-commit install


Basic Installation
------------------

For basic functionality with minimal dependencies:

.. code-block:: bash

   # Basic installation
   pip install marEx

This installs the core package with required dependencies only.

Optional Dependencies
=====================

You can install specific optional dependency groups:

.. code-block:: bash

   # Performance enhancements
   pip install marEx[performance]    # JAX acceleration

   # Enhanced plotting
   pip install marEx[plotting]       # Seaborn, cmocean colormaps

   # HPC cluster support
   pip install marEx[hpc]            # SLURM integration, psutil

   # Development tools
   pip install marEx[dev]            # Testing, linting, pre-commit hooks

Checking Your Installation
==========================

After installation, verify that MarEx is working correctly and check dependency status:

.. code-block:: python

   import marEx

   # Check version
   print(f"MarEx version: {marEx.__version__}")

   # Check dependency status
   marEx.print_dependency_status()

   # Check if specific optional dependencies are available
   print(f"JAX acceleration available: {marEx.has_dependency('jax')}")

   # Configure logging (optional)
   marEx.set_verbose_mode()  # For detailed logging during development

System Requirements
===================

**Python Version**: 3.10 or higher

**Operating Systems**: Linux, macOS, Windows

**Storage**: SSD or Lustre system recommended for optimal I/O performance

Performance Dependencies
========================

For optimal performance, install these optional dependencies:

JAX Acceleration
----------------

.. code-block:: bash

   # Install JAX for GPU/TPU acceleration
   pip install marEx[performance]

JAX provides significant speedups for numerical computations and enables GPU/TPU acceleration when available.

FFmpeg (for animations)
-----------------------

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # macOS with Homebrew
   brew install ffmpeg

   # Windows with Chocolatey
   choco install ffmpeg

FFmpeg is required for creating animated visualizations.


Upgrading
=========

To upgrade MarEx to the latest version:

.. code-block:: bash

   pip install --upgrade marEx[full]

To upgrade to a specific version:

.. code-block:: bash

   pip install marEx[full]==3.0.0
