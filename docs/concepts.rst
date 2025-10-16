=============
Core Concepts
=============

This guide provides a foundational understanding of marEx's design philosophy and core concepts.

.. contents:: Table of Contents
   :local:
   :depth: 2

Why marEx? The Challenge of Tracking Ocean Extremes
====================================================

Modern oceanography generates massive datasets from satellites, ocean models, and observational networks. Within these datasets, extreme events like marine heatwaves represent critical phenomena affecting marine ecosystems, fisheries, and climate systems. However, identifying and tracking these events presents significant challenges:

* **Scale**: Datasets often exceed hundreds of gigabytes or terabytes
* **Complexity**: Events move, grow, shrink, merge, and split over time
* **Variability**: Different scientific or industrial questions require different detection methods
* **Diversity**: Data comes in varied formats (regular grids, irregular meshes, different resolutions)

The Goal of marEx
-----------------

**marEx** (**Mar**\ ine **Ex**\ tremes) provides a scalable, flexible, and scientifically rigorous toolkit to automate the detection and tracking of marine extreme events. It handles the computational complexity so researchers can focus on scientific questions rather than implementation details.

What is a Marine Extreme Event?
================================

Understanding marine extremes requires four foundational concepts:

Climatology
-----------

The **climatology** represents the long-term "normal" state of the ocean for a given location and time of year. For example, the average sea surface temperature in the North Atlantic during July, based on 30 years of data.

Think of climatology as the baseline we use to define what "typical" conditions look like.

Anomaly
-------

An **anomaly** is the deviation from the climatological baseline:

.. code-block:: text

   Anomaly = Observed Value - Climatology

For example, if the climatology for a location in July is 20°C, and the observed temperature is 23°C, the anomaly is +3°C.

marEx provides multiple methods for calculating anomalies, each with different assumptions about trends and climate change (see :doc:`user_guide` for details).

Extreme Event
-------------

An **extreme event** is an anomaly that exceeds a statistical threshold, typically defined as a percentile of the anomaly distribution. For marine heatwaves, the standard definition uses the 95th percentile (Hobday et al. 2016):

.. code-block:: text

   Extreme Event = Anomaly > 95th percentile threshold

This creates a binary classification: at each location and time, conditions are either "extreme" or "not extreme."

Tracked Object
--------------

A **tracked object** is a coherent extreme event that has been identified as a spatially connected region and followed through time. Tracking assigns each object a unique ID and records its evolution: position, area, lifetime, and relationships with other events (merges/splits).

The marEx Workflow: A Three-Step Process
=========================================

marEx follows a clear three-stage pipeline that maps directly to its code architecture:

.. code-block:: text

   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
   │  1. Detect      │  →   │  2. Track       │  →   │  3. Visualise   │
   │    Extremes     │      │    Events       │      │     & Analyse   │
   └─────────────────┘      └─────────────────┘      └─────────────────┘
           ↓                        ↓                        ↓
    preprocess_data()           tracker()                  plotX()
           ↓                        ↓                        ↓
    Binary extreme map        Tracked objects          Maps, animations,
                              with unique IDs           & statistics

Step 1: Detect Extremes
-----------------------

**Function**: ``marEx.preprocess_data()``

**Purpose**: Transform raw oceanographic data (e.g., sea surface temperature) into a binary map showing where and when extreme conditions exist.

**Process**:

1. Calculate anomalies relative to a baseline climatology
2. Apply percentile thresholds to identify extreme values
3. Generate a binary field: True where extremes occur, False elsewhere

**Output**: An xarray Dataset containing anomalies, binary extreme events, thresholds, and a data mask.

**See**: :doc:`modules/detect` for detailed algorithm descriptions and :doc:`user_guide` for method selection guidance.

Step 2: Track Events
--------------------

**Function**: ``marEx.tracker()``

**Purpose**: Group spatially connected extreme points into objects and follow these objects through time, handling merges and splits.

**Process**:

1. Identify spatially connected regions (objects) in each time step
2. Track objects across time by computing overlap between consecutive frames
3. Handle merging (two events become one) and splitting (one event becomes two)
4. Assign unique IDs to each tracked event
5. Record event characteristics (area, centroid, lifetime)

**Output**: An xarray Dataset containing tracked event IDs, object statistics, and merge history.

**See**: :doc:`modules/track` for tracking algorithms and :doc:`user_guide` for parameter tuning.

Step 3: Analyse & Visualise
---------------------------

**Function**: ``.plotX`` accessor

**Purpose**: Create publication-quality visualisations and perform statistical analysis of tracked events.

**Capabilities**:

* Single-panel maps with customisable projections
* Multi-panel comparisons (seasonal, regional)
* Animated time series showing event evolution
* Automatic grid detection (works with both structured and unstructured grids)
* Statistical summaries (event frequency, duration, intensity)

**Output**: Matplotlib figures, saved images, MP4 animations.

**See**: :doc:`modules/plotx` for visualisation options and :doc:`examples` for real-world applications.

Key Feature: Handling All Ocean Data
=====================================

A major strength of marEx is its ability to work seamlessly with different types of ocean data grids.

Structured Grids
----------------

**Description**: Regular rectangular grids with dimensions ``(time, lat, lon)``

**Examples**:

* Satellite-derived sea surface temperature (e.g., NOAA OISST)
* Climate model output (e.g., CMIP6 models)
* Reanalysis products (e.g., ERA5 ocean component)

**Characteristics**: Familiar latitude/longitude coordinates on a regular grid. Data at each grid point represents a rectangular area.

.. code-block:: python

   # Structured grid example
   sst.dims        # ('time', 'lat', 'lon')
   sst.coords      # time, lat, lon as coordinate arrays

Unstructured Grids
------------------

**Description**: Irregular meshes with dimensions ``(time, ncells)`` and separate coordinate arrays for lat/lon

**Examples**:

* FESOM (Finite Element Sea ice-Ocean Model)
* ICON-O (Icosahedral Nonhydrostatic Ocean model)
* MPAS-Ocean (Model for Prediction Across Scales)

**Characteristics**: Irregular polygonal cells that allow variable resolution (e.g., higher resolution near coastlines). Requires connectivity information for spatial operations.

.. code-block:: python

   # Unstructured grid example
   sst.dims        # ('time', 'ncells')
   sst.coords      # time, lat, lon (lat/lon are coordinate arrays, not dimensions)

Transparent Grid Handling
-------------------------

marEx automatically detects the grid type based on the coordinate structure and applies appropriate algorithms. **You write the same code for both grid types**:

.. code-block:: python

   # Works identically for structured and unstructured grids
   extremes = marEx.preprocess_data(sst, threshold_percentile=95)
   tracked = marEx.tracker(extremes.extreme_events, extremes.mask).run()
   fig, ax, im = tracked.ID_field.isel(time=0).plotX.single_plot(config)

For unstructured grids, specify grid metadata using ``marEx.specify_grid()`` to enable advanced features like spatial windowing (see :doc:`user_guide`).

Key Feature: Built for Scale with Dask
=======================================

The Challenge of Big Data
--------------------------

Modern ocean datasets routinely exceed available computer memory:

* Global 0.25° daily SST for 30 years: ~100 GB
* High-resolution regional models: 200+ GB
* Coupled climate model ensembles: 10+ TB

Traditional analysis tools that load entire datasets into memory fail with these data sizes.

The Dask Solution
-----------------

marEx uses **Dask** as its computational backend. Dask is a parallel computing library that:

1. **Breaks data into chunks**: Divides large arrays into manageable pieces
2. **Processes chunks in parallel**: Utilises multiple CPU cores simultaneously
3. **Manages memory automatically**: Only loads necessary chunks, discarding when done
4. **Scales from laptops to supercomputers**: Same code works on 4-core laptop or 1000-core HPC cluster

How Dask Integration Works
---------------------------

**For users**: marEx requires input data to be Dask-backed xarray objects (use ``chunks={}`` when loading):

.. code-block:: python

   import xarray as xr

   # Load data with Dask chunking
   sst = xr.open_dataset('sst_data.nc', chunks={'time': 365}).sst

   # marEx handles all Dask operations internally
   extremes = marEx.preprocess_data(sst, threshold_percentile=95)

   # Computation happens when you request results
   result = extremes.extreme_events.compute()  # Triggers parallel computation

**Performance benefits**:

* Process datasets 100-1000× larger than available RAM
* Utilise all CPU cores for faster computation
* Seamless scaling to HPC clusters with SLURM integration (see ``marEx.helper``)

**See**: :doc:`user_guide` for chunking strategies and performance optimisation, :doc:`modules/helper` for HPC cluster setup.

Where to Go Next
=================

Now that you understand marEx's foundational concepts, here's how to proceed:

For Hands-On Learning
---------------------

* :doc:`quickstart` - Get started with a working example
* :doc:`examples` - Explore Jupyter notebooks showing complete workflows for gridded, regional, and unstructured data

For Detailed Guidance
---------------------

* :doc:`user_guide` - Complete guide covering:

  * Method selection (which anomaly/extreme detection method to use)
  * Scientific trade-offs between methods
  * Parameter tuning for different research questions
  * Performance optimisation strategies
  * Best practices for marine heatwave detection

For Technical Reference
-----------------------

* :doc:`api` - Complete API documentation for all functions
* :doc:`modules/detect` - Detection algorithms and implementation details
* :doc:`modules/track` - Tracking algorithms and merge/split handling
* :doc:`modules/plotx` - Visualisation system and customisation options
* :doc:`modules/helper` - HPC utilities and cluster management

For Troubleshooting
-------------------

* :doc:`troubleshooting` - Common issues and solutions for installation, performance, and data problems

Key References
==============

The scientific methods in marEx are based on established literature:

* **Hobday et al. (2016)**: "A hierarchical approach to defining marine heatwaves" *Progress in Oceanography* 141, 227-238. `doi:10.1016/j.pocean.2015.12.014 <https://doi.org/10.1016/j.pocean.2015.12.014>`_

  * Defines the standard marine heatwave detection methodology using day-of-year specific percentile thresholds

* **Sun et al. (2023)**: "Marine heatwaves in the Arctic Region: Variation in Different Ice Covers" *Progress in Oceanography* 203, 102947. `doi:10.1016/j.pocean.2022.102947 <https://doi.org/10.1016/j.pocean.2022.102947>`_

  * Provides tracking methodology that marEx extends with improved merge/split partitioning
