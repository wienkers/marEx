==============================
Plotting Module (:mod:`marEx.plotX`)
==============================

.. currentmodule:: marEx.plotX

The :mod:`marEx.plotX` module provides visualisation functions for marine extreme
events through an xarray accessor pattern. It automatically detects grid types and
applies appropriate visualisation methods for both structured and unstructured grids.

Overview
========

The plotting module implements an xarray accessor that seamlessly integrates with
xarray DataArrays to provide specialised plotting capabilities. The system automatically
detects whether data is on a structured (regular lat/lon) or unstructured (irregular mesh)
grid and applies the appropriate plotting method.

**Key Features:**

* **Automatic Grid Detection**: Detects structured vs. unstructured grids automatically
* **xarray Integration**: Access via `.plotX` accessor on DataArrays
* **Flexible Configuration**: Comprehensive plotting options via `PlotConfig`
* **Animation Support**: Built-in animation capabilities for time series data
* **Memory Efficient**: Global caching for triangulation and spatial indexing

Main Components
===============

.. autosummary::
   :toctree: ../generated/

   PlotConfig
   specify_grid
   PlotXAccessor

Grid Detection and Configuration
===============================

Grid Type Detection
-------------------

The plotting system automatically detects grid types based on coordinate structure:

* **Structured Grids**: Have separate latitude and longitude dimensions (e.g., `lat`, `lon`)
* **Unstructured Grids**: Have a single spatial dimension (e.g., `ncells`) with lat/lon as coordinates

.. autofunction:: specify_grid

Configuration Class
-------------------

.. autoclass:: PlotConfig
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
===========

Simple Plotting
---------------

.. code-block:: python

   import xarray as xr
   import marEx

   # Load data
   data = xr.open_dataset('example.nc').temperature

   # Basic plot - automatic grid detection
   fig, ax, im = data.plotX.single_plot(marEx.PlotConfig())

Advanced Configuration
----------------------

.. code-block:: python

   # Custom plot configuration
   config = marEx.PlotConfig(
       title='Sea Surface Temperature',
       var_units='°C',
       cmap='RdBu_r',
       issym=True,
       show_colorbar=True,
       grid_lines=True,
       grid_labels=True
   )

   # Create plot
   fig, ax, im = data.plotX.single_plot(config)

Multi-Panel Plotting
--------------------

.. code-block:: python

   # Plot multiple time steps
   config = marEx.PlotConfig(
       title='Temperature Evolution',
       var_units='°C',
       cmap='viridis'
   )

   # Create wrapped subplots
   fig, axes = data.plotX.multi_plot(config, col='time', col_wrap=3)

Animation
---------

.. code-block:: python

   # Create animation
   config = marEx.PlotConfig(
       title='Temperature Animation',
       var_units='°C',
       cmap='RdBu_r'
   )

   # Generate animation
   movie_path = data.plotX.animate(
       config,
       plot_dir='./animations',
       file_name='temperature_evolution'
   )

Structured Grid Usage
=====================

Regular Lat/Lon Grids
---------------------

For structured grids (typical climate model output):

.. code-block:: python

   # Load gridded data
   sst = xr.open_dataset('sst_regular.nc').sst

   # Configure for geographic plotting
   config = marEx.PlotConfig(
       title='Global Sea Surface Temperature',
       var_units='°C',
       cmap='coolwarm',
       show_colorbar=True,
       grid_lines=True,
       grid_labels=True
   )

   # Plot will automatically use GriddedPlotter
   fig, ax, im = sst.plotX.single_plot(config)

Custom Dimension Names
----------------------

.. code-block:: python

   # For data with non-standard coordinate names
   config = marEx.PlotConfig(
       title='Temperature',
       var_units='°C',
       # Specify custom dimension mapping
       dimensions={'time': 'time', 'y': 'latitude', 'x': 'longitude'},
       coordinates={'time': 'time', 'y': 'latitude', 'x': 'longitude'}
   )

   fig, ax, im = data.plotX.single_plot(config)

Unstructured Grid Usage
=======================

Ocean Model Grids
------------------

For unstructured grids (e.g., FESOM, ICON-O):

.. code-block:: python

   # First specify grid information globally
   marEx.specify_grid(
       grid_type='unstructured',
       fpath_tgrid='grid_info.nc',
       fpath_ckdtree='./ckdtree_indices/'
   )

   # Load unstructured data
   sst = xr.open_dataset('sst_unstructured.nc').sst

   # Configure plot
   config = marEx.PlotConfig(
       title='Ocean Model SST',
       var_units='°C',
       cmap='thermal',
       show_colorbar=True
   )

   # Plot will automatically use UnstructuredPlotter
   fig, ax, im = sst.plotX.single_plot(config)

Triangulation-Based Plotting
-----------------------------

.. code-block:: python

   # Use triangulation file for native mesh plotting
   marEx.specify_grid(
       grid_type='unstructured',
       fpath_tgrid='triangulation.nc'
   )

   config = marEx.PlotConfig(
       title='Native Mesh Visualization',
       var_units='Temperature (°C)',
       cmap='plasma'
   )

   fig, ax, im = data.plotX.single_plot(config)

Event ID Plotting
==================

Special Configuration for Event IDs
------------------------------------

.. code-block:: python

   # For plotting tracked event IDs
   config = marEx.PlotConfig(
       title='Marine Heatwave Events',
       plot_IDs=True,  # Special handling for event IDs
       cmap='tab20'    # Discrete colormap for IDs
   )

   fig, ax, im = event_ids.plotX.single_plot(config)

Color Scaling Options
=====================

Percentile-Based Scaling
------------------------

.. code-block:: python

   config = marEx.PlotConfig(
       title='Temperature Anomalies',
       var_units='°C',
       cmap='RdBu_r',
       cperc=[5, 95],  # Use 5th and 95th percentiles
       extend='both'
   )

Symmetric Scaling
-----------------

.. code-block:: python

   config = marEx.PlotConfig(
       title='Temperature Anomalies',
       var_units='°C',
       cmap='RdBu_r',
       issym=True,     # Symmetric around zero
       extend='both'
   )

Manual Color Limits
-------------------

.. code-block:: python

   config = marEx.PlotConfig(
       title='Temperature',
       var_units='°C',
       cmap='viridis',
       clim=(-2, 5),   # Manual color limits
       extend='both'
   )

Error Handling
==============

The plotting system provides comprehensive error handling:

.. code-block:: python

   try:
       fig, ax, im = data.plotX.single_plot(config)
   except marEx.VisualisationError as e:
       print(f"Plotting error: {e}")
       print(f"Suggestions: {e.suggestions}")
   except marEx.DependencyError as e:
       print(f"Missing dependency: {e}")


Integration with Matplotlib
===========================

Direct Matplotlib Integration
-----------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   # Create custom figure
   fig, ax = plt.subplots(figsize=(12, 8))

   # Use existing axes
   config = marEx.PlotConfig(title='Custom Plot')
   fig, ax, im = data.plotX.single_plot(config, ax=ax)

   # Add custom elements
   ax.set_title('Custom Title', fontsize=14)
   plt.tight_layout()


See Also
========

* :mod:`marEx.plotX.base` - Base plotting functionality
* :mod:`marEx.plotX.gridded` - Structured grid plotting
* :mod:`marEx.plotX.unstructured` - Unstructured grid plotting
* :mod:`marEx.detect` - Data preprocessing
* :mod:`marEx.track` - Event tracking
