==========================================
Unstructured Plotting (:mod:`marEx.plotX.unstructured`)
==========================================

.. currentmodule:: marEx.plotX.unstructured

The :mod:`marEx.plotX.unstructured` module provides specialised plotting functionality for
unstructured grids, such as those used in finite element ocean models (FESOM, ICON-O, MPAS),
atmospheric models on icosahedral grids, and other irregular mesh datasets.

Overview
========

This module implements the `UnstructuredPlotter` class, which handles irregular spatial grids
where data points are not arranged in a regular rectangular pattern. It provides triangulation
and interpolation capabilities for complex mesh structures with global caching for
performance optimisation.

**Key Features:**

* **Triangulation Support**: Native triangular mesh visualisation using matplotlib
* **KDTree Interpolation**: Fast interpolation to regular grids using pre-computed indices
* **Global Caching**: Persistent caching of expensive triangulation and spatial index operations
* **Flexible Input**: Supports both triangulation files and KDTree index directories
* **Memory Efficient**: Optimised memory usage for large unstructured datasets

Classes and Functions
=====================

.. autosummary::
   :toctree: ../../generated/

   UnstructuredPlotter
   clear_cache

Utility Functions
=================

.. autosummary::
   :toctree: ../../generated/

   _load_triangulation
   _load_ckdtree

UnstructuredPlotter Class
=========================

.. autoclass:: UnstructuredPlotter
   :members:
   :undoc-members:
   :show-inheritance:

The UnstructuredPlotter class handles irregular mesh grids:

.. code-block:: python

   # UnstructuredPlotter is typically accessed via the plotX accessor
   # Automatic selection based on grid type detection

   import xarray as xr
   import marEx

   # Set up grid information first
   marEx.specify_grid(
       grid_type='unstructured',
       fpath_tgrid='grid_triangulation.nc',
       fpath_ckdtree='./ckdtree_indices/'
   )

   # Load unstructured data
   data = xr.open_dataset('unstructured_data.nc').temperature

   # Plotting automatically uses UnstructuredPlotter
   config = marEx.PlotConfig(title='Ocean Model Temperature', var_units='°C')
   fig, ax, im = data.plotX.single_plot(config)

Methods
=======

Grid Specification
------------------

.. automethod:: UnstructuredPlotter.specify_grid

Set grid file paths for unstructured plotting:

.. code-block:: python

   # Method 1: Global specification (recommended)
   marEx.specify_grid(
       grid_type='unstructured',
       fpath_tgrid='triangulation.nc',
       fpath_ckdtree='./ckdtree_data/'
   )

   # Method 2: Per-plotter specification
   plotter = UnstructuredPlotter(data)
   plotter.specify_grid(
       fpath_tgrid='triangulation.nc',
       fpath_ckdtree='./ckdtree_data/'
   )

Plot Method
-----------

.. automethod:: UnstructuredPlotter.plot

The core plotting method supports two rendering modes:

1. **KDTree Interpolation** (if `fpath_ckdtree` provided): Fast interpolation to regular grid
2. **Triangulation** (if `fpath_tgrid` provided): Native triangular mesh rendering


Helper Functions
================

Triangulation Loading
---------------------

.. autofunction:: _load_triangulation

Loads and caches triangulation data:

.. code-block:: python

   # Triangulation files must contain:
   # - 'vertex_of_cell': connectivity array (1-based indexing)
   # - 'clon': cell longitude coordinates
   # - 'clat': cell latitude coordinates

   # File format example:
   # vertex_of_cell(ncells, nvertices_per_cell) = [[1, 2, 3], [2, 3, 4], ...]
   # clon(ncells) = [longitude values]
   # clat(ncells) = [latitude values]

KDTree Loading
--------------

.. autofunction:: _load_ckdtree

Loads and caches KDTree interpolation data:

.. code-block:: python

   # KDTree directory structure:
   # ckdtree_path/
   #   ├── res0.10.nc
   #   ├── res0.25.nc
   #   ├── res0.50.nc
   #   └── res1.00.nc

   # Each resolution file contains:
   # - 'ickdtree_c': indices for interpolation
   # - 'lon': regular grid longitude coordinates
   # - 'lat': regular grid latitude coordinates

Basic Usage Examples
====================

Setup and Simple Plot
----------------------

.. code-block:: python

   import xarray as xr
   import marEx

   # Set up grid information globally
   marEx.specify_grid(
       grid_type='unstructured',
       fpath_tgrid='ocean_grid.nc',
       fpath_ckdtree='./spatial_indices/'
   )

   # Load unstructured ocean model data
   sst = xr.open_dataset('fesom_sst.nc').sst

   # Basic plot
   config = marEx.PlotConfig(
       title='Ocean Model SST',
       var_units='°C',
       cmap='thermal',
       show_colorbar=True
   )

   fig, ax, im = sst.plotX.single_plot(config)

Triangulation-Only Plotting
----------------------------

.. code-block:: python

   # Use only triangulation (no interpolation)
   marEx.specify_grid(
       grid_type='unstructured',
       fpath_tgrid='triangulation.nc'
       # No ckdtree path - will use native triangulation
   )

   config = marEx.PlotConfig(
       title='Native Mesh Visualization',
       var_units='Temperature (°C)',
       cmap='plasma',
       show_colorbar=True
   )

   fig, ax, im = data.plotX.single_plot(config)

KDTree-Only Plotting
---------------------

.. code-block:: python

   # Use only KDTree interpolation
   marEx.specify_grid(
       grid_type='unstructured',
       fpath_ckdtree='./interpolation_indices/'
       # No triangulation path - will use interpolated regular grid
   )

   config = marEx.PlotConfig(
       title='Interpolated Visualization',
       var_units='°C',
       cmap='coolwarm',
       show_colorbar=True
   )

   fig, ax, im = data.plotX.single_plot(config)

Time Series Visualization
=========================

Multi-Panel Plots
------------------

.. code-block:: python

   # Plot multiple time steps
   config = marEx.PlotConfig(
       var_units='°C',
       cmap='RdBu_r',
       issym=True,
       show_colorbar=True
   )

   # Create wrapped subplots
   fig, axes = sst.plotX.multi_plot(config, col='time', col_wrap=3)

Animation
---------

.. code-block:: python

   # Create animation
   config = marEx.PlotConfig(
       title='Ocean Model Evolution',
       var_units='°C',
       cmap='thermal',
       show_colorbar=True
   )

   # Generate animation (requires ffmpeg)
   movie_path = sst.plotX.animate(
       config,
       plot_dir='./animations',
       file_name='ocean_evolution'
   )

Advanced Configuration
======================

Custom Dimension Names
----------------------

.. code-block:: python

   # For unstructured data with custom dimension names
   config = marEx.PlotConfig(
       title='Custom Dimensions',
       var_units='°C',
       cmap='viridis',
       # Custom dimension mapping for unstructured data
       dimensions={'time': 'time', 'x': 'cell_index'},
       coordinates={'time': 'time', 'x': 'cell_lon', 'y': 'cell_lat'}
   )

   fig, ax, im = data.plotX.single_plot(config)


Grid Type Detection
===================

The UnstructuredPlotter is automatically selected when:

* Data has a single spatial dimension (e.g., `ncells`)
* Latitude and longitude are **coordinates** rather than **dimensions**
* Typical structure: `(time, ncells)` with `lat(ncells)` and `lon(ncells)` coordinates

.. code-block:: python

   # Automatic detection based on:
   # - Single spatial dimension in data.dims
   # - lat/lon as coordinates (not dimensions)
   # - Irregular spatial arrangement

   # Example unstructured data structure:
   # Dimensions: (time: 365, ncells: 830305)
   # Coordinates: lat(ncells), lon(ncells), time(time)

   # Override detection if needed:
   marEx.specify_grid(grid_type='unstructured')

File Format Requirements
========================

Triangulation Files
-------------------

Triangulation files must contain specific variables:

.. code-block:: python

   # Required variables in triangulation NetCDF file:
   # - vertex_of_cell(ncells, nvertices_per_cell): connectivity array
   # - clon(ncells): cell longitude coordinates
   # - clat(ncells): cell latitude coordinates

   # Example file creation:
   import xarray as xr
   import numpy as np

   # Create triangulation file
   triangulation_ds = xr.Dataset({
       'vertex_of_cell': (['ncells', 'nvertices'], connectivity_array),
       'clon': (['ncells'], lon_coords),
       'clat': (['ncells'], lat_coords)
   })
   triangulation_ds.to_netcdf('triangulation.nc')

KDTree Index Files
------------------

KDTree directories contain resolution-specific files:

.. code-block:: python

   # Directory structure:
   # ckdtree_indices/
   #   ├── res0.10.nc  # High resolution
   #   ├── res0.25.nc  # Medium resolution
   #   ├── res0.50.nc  # Low resolution
   #   └── res1.00.nc  # Very low resolution

   # Each file contains:
   # - ickdtree_c(nlat, nlon): indices for interpolation
   # - lon(nlon): regular grid longitude coordinates
   # - lat(nlat): regular grid latitude coordinates

   # Example file creation:
   ckdtree_ds = xr.Dataset({
       'ickdtree_c': (['nlat', 'nlon'], index_array),
       'lon': (['nlon'], regular_lon),
       'lat': (['nlat'], regular_lat)
   })
   ckdtree_ds.to_netcdf('res0.25.nc')


Integration Examples
====================

Matplotlib Integration
----------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   # Create custom figure
   fig, ax = plt.subplots(figsize=(12, 8))

   # Use existing axes
   config = marEx.PlotConfig(title='Custom Layout', var_units='°C')
   fig, ax, im = data.plotX.single_plot(config, ax=ax)

   # Add custom annotations
   ax.set_title('Ocean Model Results', fontsize=16)
   plt.tight_layout()

Comparison Plots
----------------

.. code-block:: python

   # Compare model vs observations
   fig, axes = plt.subplots(1, 2, figsize=(16, 6))

   # Model data (unstructured)
   marEx.specify_grid(grid_type='unstructured', fpath_ckdtree='./model_indices/')
   config1 = marEx.PlotConfig(title='Model', var_units='°C', show_colorbar=False)
   fig, axes[0], im1 = model_data.plotX.single_plot(config1, ax=axes[0])

   # Observations (structured)
   marEx.specify_grid(grid_type='gridded')
   config2 = marEx.PlotConfig(title='Observations', var_units='°C', show_colorbar=False)
   fig, axes[1], im2 = obs_data.plotX.single_plot(config2, ax=axes[1])

   # Single colorbar
   fig.subplots_adjust(right=0.9)
   cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
   fig.colorbar(im1, cax=cbar_ax, extend='both')


See Also
========

* :mod:`marEx.plotX.base` - Base plotting functionality
* :mod:`marEx.plotX.gridded` - Structured grid plotting
* :mod:`marEx.plotX` - Main plotting module
* `matplotlib.tri documentation <https://matplotlib.org/stable/api/tri_api.html>`_
* `scipy.spatial.cKDTree documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html>`_
