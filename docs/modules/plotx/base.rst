===============================
Base Plotting (:mod:`marEx.plotX.base`)
===============================

.. currentmodule:: marEx.plotX.base

The :mod:`marEx.plotX.base` module provides the foundational classes and utilities for
marEx plotting. It defines the base plotting interface, configuration management, and
common functionality shared across different grid types.

Overview
========

This module implements the core plotting architecture using object-oriented design patterns.
It provides the base plotter class and configuration dataclass that define the plotting
interface and common functionality inherited by specialised plotters for different grid types.

**Key Components:**

* **PlotConfig**: Dataclass for comprehensive plot configuration
* **PlotterBase**: Abstract base class defining the plotting interface
* **Animation Support**: Built-in animation capabilities with ffmpeg integration
* **Error Handling**: Comprehensive error handling and validation

Classes and Functions
=====================

.. autosummary::
   :toctree: ../../generated/

   PlotConfig
   PlotterBase
   make_frame

Configuration Management
========================

PlotConfig Class
----------------

.. autoclass:: PlotConfig
   :members:
   :undoc-members:
   :show-inheritance:

The PlotConfig dataclass provides comprehensive configuration options for all plotting operations:

.. code-block:: python

   from marEx.plotX import PlotConfig

   # Basic configuration
   config = PlotConfig(
       title='Sea Surface Temperature',
       var_units='°C',
       cmap='RdBu_r',
       show_colorbar=True
   )

   # Advanced configuration
   config = PlotConfig(
       title='Temperature Anomalies',
       var_units='°C',
       issym=True,              # Symmetric color scaling
       cmap='RdBu_r',
       cperc=[5, 95],          # Percentile limits
       show_colorbar=True,
       grid_lines=True,
       grid_labels=True,
       extend='both',
       # Custom dimension mapping
       dimensions={'time': 'time', 'y': 'latitude', 'x': 'longitude'},
       coordinates={'time': 'time', 'y': 'latitude', 'x': 'longitude'}
   )

Configuration Options
--------------------

**Visual Appearance:**

* **title**: Plot title (optional)
* **var_units**: Units for colorbar label
* **cmap**: Colormap name or ListedColormap object
* **show_colorbar**: Whether to display colorbar (default: True)

**Color Scaling:**

* **issym**: Symmetric color scaling around zero (default: False)
* **cperc**: Percentile limits for color scaling (default: [4, 96])
* **clim**: Manual color limits as tuple (vmin, vmax)
* **norm**: Custom matplotlib normalisation object
* **extend**: Colorbar extension ('both', 'min', 'max', 'neither')

**Grid and Labels:**

* **grid_lines**: Show grid lines (default: True)
* **grid_labels**: Show grid labels (default: False)

**Special Modes:**

* **plot_IDs**: Special handling for event ID plotting (default: False)
* **dimensions**: Custom dimension name mapping
* **coordinates**: Custom coordinate name mapping

**Logging:**

* **verbose**: Enable verbose logging (optional)
* **quiet**: Enable quiet logging (optional)

Base Plotter Class
==================

PlotterBase Class
-----------------

.. autoclass:: PlotterBase
   :members:
   :undoc-members:
   :show-inheritance:

The PlotterBase class provides the core plotting infrastructure:

.. code-block:: python

   # PlotterBase is typically not used directly
   # It's inherited by GriddedPlotter and UnstructuredPlotter

   # Example of common workflow (implemented by subclasses):
   plotter = GriddedPlotter(data_array)
   fig, ax, im = plotter.single_plot(config)

Common Methods
--------------

**single_plot(config, ax=None)**
   Create a single plot with the given configuration.

   Args:
       config (PlotConfig): Plot configuration
       ax (matplotlib.axes.Axes, optional): Existing axes to use

   Returns:
       tuple: (figure, axes, image) objects

**multi_plot(config, col='time', col_wrap=3)**
   Create multiple subplots wrapped in a grid.

   Args:
       config (PlotConfig): Plot configuration
       col (str): Dimension to plot across panels
       col_wrap (int): Number of columns before wrapping

   Returns:
       tuple: (figure, axes_array) objects

**animate(config, plot_dir='./', file_name=None)**
   Create an animation from time series data.

   Args:
       config (PlotConfig): Plot configuration
       plot_dir (str or Path): Directory for output files
       file_name (str, optional): Output filename (without extension)

   Returns:
       str: Path to created animation file

Implementation Details
======================

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


Animation
=========

Animation Creation
------------------

The animation system uses dask for parallel frame generation:

.. code-block:: python

   # Create animation
   config = PlotConfig(
       title='Temperature Evolution',
       var_units='°C',
       cmap='thermal'
   )

   # Generate animation (requires ffmpeg)
   movie_path = data.plotX.animate(
       config,
       plot_dir='./animations',
       file_name='temperature_movie'
   )

Frame Generation
----------------

.. autofunction:: make_frame

The make_frame function is decorated with @dask.delayed for parallel processing:

.. code-block:: python

   # Internal animation workflow:
   # 1. Create delayed tasks for each frame
   # 2. Parallel frame generation using dask
   # 3. Combine frames with ffmpeg

   delayed_tasks = []
   for time_ind in range(len(data.time)):
       data_slice = data.isel(time=time_ind)
       delayed_tasks.append(make_frame(data_slice, time_ind, temp_dir, plot_params))

   # Execute in parallel
   filenames = dask.compute(*delayed_tasks)

Requirements
------------

Animation requires additional dependencies:

* **ffmpeg**: For video encoding (must be in system PATH)
* **PIL/Pillow**: For image processing
* **dask**: For parallel frame generation

Utilities
=========

Common Utilities
----------------

The base class provides several utility methods:

.. code-block:: python

   # Robust color limit calculation
   def clim_robust(data, issym=False, percentiles=[2, 98]):
       # Calculate color limits from data percentiles
       pass

   # ID plotting setup
   def setup_id_plot_params(cmap=None):
       # Special configuration for event ID plotting
       pass

   # Common parameter setup
   def setup_plot_params():
       # Configure matplotlib parameters
       pass


Integration Points
==================

Matplotlib Integration
----------------------

Full compatibility with matplotlib:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Use existing figure/axes
   fig, ax = plt.subplots(figsize=(12, 8))
   config = PlotConfig(title='Custom Plot')
   fig, ax, im = data.plotX.single_plot(config, ax=ax)

   # Add custom elements
   ax.set_title('Modified Title')
   plt.tight_layout()

Cartopy Integration
-------------------

Automatic geographic projection handling:

.. code-block:: python

   # Cartopy projections handled automatically
   # - PlateCarree for data transformation
   # - Robinson for default display
   # - Automatic feature addition

Customisation Examples
======================

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

See Also
========

* :mod:`marEx.plotX.gridded` - Structured grid plotting
* :mod:`marEx.plotX.unstructured` - Unstructured grid plotting
* :mod:`marEx.plotX` - Main plotting module
* `matplotlib documentation <https://matplotlib.org/>`_
* `cartopy documentation <https://scitools.org.uk/cartopy/docs/latest/>`_
