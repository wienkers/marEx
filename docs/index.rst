==================================
marEx: Marine Extremes Detection and Tracking
==================================

.. image:: https://img.shields.io/pypi/v/marEx.svg
   :target: https://pypi.org/project/marEx/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/marEx.svg
   :target: https://pypi.org/project/marEx/
   :alt: Python Versions

.. image:: https://readthedocs.org/projects/marex/badge/?version=latest
   :target: https://marex.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/github/license/wienkers/marEx.svg
   :target: https://github.com/wienkers/marEx/blob/main/LICENSE
   :alt: License

**marEx** is a Python package for efficient identification and tracking of marine extremes
(e.g., Marine Heatwaves) in oceanographic data. It provides a complete pipeline from raw data
preprocessing to tracked event visualisation, supporting both structured and unstructured grids.

Key Features
============

* **Comprehensive Marine Extreme Detection**: Advanced algorithms for identifying marine heatwaves, cold spells, and other extreme events
* **Flexible Grid Support**: Works with both structured (regular lat/lon) and unstructured (irregular mesh) grids
* **Parallel Processing**: Built on Dask for efficient memory management and parallel computation
* **Advanced Tracking**: Sophisticated event tracking through time with merge/split handling
* **Rich Visualisation**: Comprehensive plotting system with automatic grid detection
* **HPC Ready**: Optimised for supercomputing environments with SLURM integration

Quick Start
===========

Installation
------------

Install marEx using pip:

.. code-block:: bash

   pip install marEx

For the full installation with optional dependencies (JAX acceleration, HPC support):

.. code-block:: bash

   pip install marEx[full]

For development:

.. code-block:: bash

   pip install marEx[dev]

Basic Usage
-----------

Here's a simple example of detecting and tracking marine heatwaves:

.. code-block:: python

   import xarray as xr
   import marEx

   # Load your sea surface temperature data
   sst = xr.open_dataset('sst_data.nc').sst

   # Preprocess data to identify extreme events
   extreme_events_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       method_anomaly='detrended_baseline',
       method_extreme='global_extreme'
   )

   # Track events through time
   tracker = marEx.tracker(
       extreme_events_ds.extreme_events,
       extreme_events_ds.mask,
       R_fill=8,
       area_filter_quartile=0.5
   )

   # Run tracking algorithm
   events_ds = tracker.run()

   # Visualise results
   fig, ax, im = (tracked_events.ID_field > 0).mean("time").plotX.single_plot(marEx.PlotConfig(var_units="MHW Frequency", cmap="hot_r", cperc=[0, 96]))

Core Architecture
=================

marEx follows a modular architecture with three main components:

1. **Data Preprocessing** (:mod:`marEx.detect`): Raw data preprocessing, detrending, anomaly detection, and extreme event identification
2. **Event Tracking** (:mod:`marEx.track`): Coherent binary object identification, labelling, and tracking through time with merge/split handling
3. **Visualisation** (:mod:`marEx.plotX`): Polymorphic plotting system supporting both gridded and unstructured data

Data Flow Pipeline
------------------

.. code-block:: text

   Raw Data → detect.py (preprocessing) → track.py (tracking) → plotX (visualisation)

The package processes chunked xarray DataArrays through:

1. Detrending and anomaly computation
2. Percentile-based extreme event identification
3. Binary object labeling and temporal tracking
4. Statistical analysis and visualisation

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   user_guide
   examples
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
   modules/detect
   modules/track
   modules/plotx
   modules/helper

Grid Types Supported
====================

Structured Grids
-----------------
Regular rectangular grids:

* **Standard climate model output** (CMIP6, reanalysis data)
* **Satellite data** on regular grids
* **Observational products** on structured grids

Unstructured Grids
------------------
Irregular meshes with dimensions (e.g. time & cell) and connectivity information:

* **Ocean model output** (FESOM, ICON-O, MPAS-Ocean)
* **Atmospheric model output** on irregular grids
* **Finite element model** output

Performance Features
====================

* **Dask-First Architecture**: All processing uses Dask for parallel computation and memory management
* **Memory Optimisation**: Efficient chunking strategies and memory-aware algorithms
* **HPC Integration**: SLURM cluster support for supercomputing environments
* **JAX Acceleration**: Optional GPU/TPU acceleration for performance-critical operations
* **Numba JIT**: Just-in-time compilation for CPU-bound operations

Scientific Methods
==================

Anomaly Detection Methods
-------------------------

* **Detrended Baseline** (``detrended_baseline``): Efficient method using long-term polynomial trends & harmonics
* **Shifting Baseline** (``shifting_baseline``): More accurate method using rolling climatologies

Extreme Event Identification
----------------------------

* **Global Extreme** (``global_extreme``): Uses global percentile thresholds
* **Hobday Extreme** (``hobday_extreme``): Uses day-of-year specific percentile thresholds

Tracking Algorithms
-------------------

* **Binary Object Tracking**: Advanced algorithms for tracking connected components through time
* **Merge/Split Handling**: New logic for handling event merging and splitting
* **Morphological Operations**: Image processing techniques for binary event preprocessing

Getting Help
============

* **Documentation**: Complete API reference and user guide
* **Examples**: Jupyter notebooks demonstrating workflows
* **Issues**: `GitHub Issues <https://github.com/wienkers/marEx/issues>`_ for bug reports and feature requests

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
