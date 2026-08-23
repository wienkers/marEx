=====================
Visualisation (plotX)
=====================

See :doc:`../api/plotx` for the full :class:`marEx.PlotConfig` and accessor reference.

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

Grid detection
==============

The plotting system automatically detects the grid type from the coordinate
structure:

* **Structured grids**: have separate latitude and longitude dimensions
  (e.g. ``lat``, ``lon``).
* **Unstructured grids**: have a single spatial dimension (e.g. ``ncells``) with
  latitude/longitude provided as coordinates.

For unstructured grids, supply the grid topology once via
:func:`marEx.specify_grid` (see :doc:`../api/plotx`); the ``.plotX`` accessor
then selects the appropriate backend automatically.

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
       title='Extreme Events',
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


.. _plotx-grid-details:


Structured grids: projections & longitude wrapping
==================================================


Geographic Projections
----------------------

The GriddedPlotter uses Cartopy for geographic projections. The default projection is Robinson, but data is always transformed from PlateCarree:

.. code-block:: python

   # Default projection handling:
   # - Data coordinates assumed to be in PlateCarree (regular lat/lon)
   # - Display projection defaults to Robinson
   # - Coordinate transformation handled automatically

   # The plot method internally uses:
   plot_kwargs = {
       'transform': ccrs.PlateCarree(),  # Input data coordinate system
       'cmap': cmap,
       'shading': 'auto'
   }

   # And the axes are created with:
   ax = plt.axes(projection=ccrs.Robinson())  # Display projection

Global vs Regional Data
-----------------------

Longitude Wrapping Logic
-------------------------

The `wrap_lon` method automatically detects if longitude wrapping is needed:

.. code-block:: python

   # Wrapping is applied when:
   # - Data spans approximately 360 degrees
   # - abs(360 - (lon.max() - lon.min())) < 2 * lon_spacing

   # Example: longitude from 0 to 359.5 with 0.5 degree spacing
   # - Total span: 359.5 degrees
   # - Spacing: 0.5 degrees
   # - 360 - 359.5 = 0.5 < 2 * 0.5 = 1.0 → wrapping applied

   # No wrapping for regional data:
   # - longitude from -180 to -120 (60 degree span)
   # - longitude from 0 to 90 (90 degree span)


Unstructured grids: triangulation, KDTree & file formats
========================================================


Methods
-------

Grid Specification
------------------

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

The core plotting method supports two rendering modes:

1. **KDTree Interpolation** (if `fpath_ckdtree` provided): Fast interpolation to regular grid
2. **Triangulation** (if `fpath_tgrid` provided): Native triangular mesh rendering

Helper Functions
----------------

Triangulation Loading
---------------------

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

File Format Requirements
------------------------

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


Advanced customisation
======================


Customisation Examples
----------------------

Custom Colormap
---------------

.. code-block:: python

   import matplotlib.pyplot as plt
   from matplotlib.colors import ListedColormap

   # Create custom colormap
   custom_cmap = ListedColormap(['blue', 'white', 'red'])

   config = PlotConfig(
       title='Custom Colors',
       cmap=custom_cmap,
       issym=True
   )

Custom Normalisation
--------------------

.. code-block:: python

   from matplotlib.colors import BoundaryNorm

   # Create custom normalisation
   levels = [-2, -1, 0, 1, 2]
   norm = BoundaryNorm(levels, ncolors=256)

   config = PlotConfig(
       title='Custom Normalisation',
       norm=norm,
       extend='both'
   )

Implementation Details
----------------------

Color Scaling
-------------

The base class provides robust color scaling methods:

.. code-block:: python

   # Automatic percentile-based scaling
   config = PlotConfig(cperc=[10, 90])

   # Symmetric scaling around zero
   config = PlotConfig(issym=True)

   # Manual color limits
   config = PlotConfig(clim=(-2, 5))

Map Features
------------

Common map features are automatically added:

.. code-block:: python

   # Default map features include:
   # - Land areas (dark grey)
   # - Coastlines (black, 0.5 linewidth)
   # - Grid lines (optional, grey dashed)
   # - Grid labels (optional)
