=======================================
Gridded Plotting (:mod:`marEx.plotX.gridded`)
=======================================

.. currentmodule:: marEx.plotX.gridded

The :mod:`marEx.plotX.gridded` module provides specialised plotting functionality for
structured (regular) grids, such as latitude-longitude gridded datasets commonly found
in climate models, reanalysis products, and satellite observations.

Overview
========

This module implements the `GriddedPlotter` class, which is optimised for handling regular
rectangular grids with coordinates organised as (time, lat, lon) or similar structures.
It provides advanced geographic visualisation capabilities with full cartographic projection
support through Cartopy.

**Key Features:**

* **Geographic Projections**: Full support for cartographic projections via Cartopy
* **Longitude Wrapping**: Automatic handling of periodic longitude boundaries
* **High-Performance Rendering**: Optimised for large gridded datasets
* **Seamless Integration**: Works seamlessly with xarray DataArrays via `.plotX` accessor

Classes
=======

.. autosummary::
   :toctree: ../../generated/

   GriddedPlotter

GriddedPlotter Class
====================

.. autoclass:: GriddedPlotter
   :members:
   :undoc-members:
   :show-inheritance:

The GriddedPlotter class handles structured grids with regular lat/lon coordinates:

.. code-block:: python

   # GriddedPlotter is typically accessed via the plotX accessor
   # Automatic selection based on grid type detection

   import xarray as xr
   import marEx

   # Load regular gridded data
   data = xr.open_dataset('regular_grid.nc').temperature

   # Plotting automatically uses GriddedPlotter
   config = marEx.PlotConfig(title='Temperature', var_units='°C')
   fig, ax, im = data.plotX.single_plot(config)

Methods
=======

Longitude Wrapping
------------------

.. automethod:: GriddedPlotter.wrap_lon

The `wrap_lon` method handles periodic longitude boundaries for global datasets:

.. code-block:: python

   # Longitude wrapping is automatically applied when needed
   # For global data spanning 360 degrees, adds duplicate column at lon=360

   # Example: data with longitude from 0 to 359.5
   # wrap_lon will add a column at lon=360 equal to lon=0
   # This prevents gaps in global plots

Plot Method
-----------

.. automethod:: GriddedPlotter.plot

The core plotting method for structured grids:

.. code-block:: python

   # The plot method is called internally by single_plot, multi_plot, etc.
   # It handles:
   # - Longitude wrapping for global data
   # - Cartopy coordinate transformations
   # - Optimal rendering with pcolormesh

Basic Usage Examples
====================

Simple Gridded Plot
--------------------

.. code-block:: python

   import xarray as xr
   import marEx

   # Load gridded SST data
   sst = xr.open_dataset('sst_gridded.nc').sst

   # Basic plot with automatic grid detection
   config = marEx.PlotConfig(
       title='Sea Surface Temperature',
       var_units='°C',
       cmap='thermal'
   )

   fig, ax, im = sst.plotX.single_plot(config)

Global Dataset Visualization
----------------------------

.. code-block:: python

   # Global dataset with longitude wrapping
   config = marEx.PlotConfig(
       title='Global Temperature',
       var_units='°C',
       cmap='RdBu_r',
       issym=True,
       show_colorbar=True,
       grid_lines=True,
       grid_labels=True
   )

   # Longitude wrapping handled automatically
   fig, ax, im = global_temp.plotX.single_plot(config)

Regional Subset
---------------

.. code-block:: python

   # Regional subset (no longitude wrapping needed)
   regional_data = sst.sel(lat=slice(30, 60), lon=slice(-180, -120))

   config = marEx.PlotConfig(
       title='North Pacific SST',
       var_units='°C',
       cmap='coolwarm',
       show_colorbar=True,
       grid_lines=True,
       grid_labels=True
   )

   fig, ax, im = regional_data.plotX.single_plot(config)

Time Series Visualization
=========================

Multi-Panel Time Series
------------------------

.. code-block:: python

   # Plot multiple time steps
   config = marEx.PlotConfig(
       var_units='°C',
       cmap='RdBu_r',
       issym=True,
       show_colorbar=True
   )

   # Create wrapped subplots
   fig, axes = sst.plotX.multi_plot(config, col='time', col_wrap=4)

Animation
---------

.. code-block:: python

   # Create time series animation
   config = marEx.PlotConfig(
       title='SST Evolution',
       var_units='°C',
       cmap='thermal',
       show_colorbar=True
   )

   # Generate animation (requires ffmpeg)
   movie_path = sst.plotX.animate(
       config,
       plot_dir='./animations',
       file_name='sst_evolution'
   )

Advanced Configuration
======================

Custom Dimension Names
----------------------

.. code-block:: python

   # For data with non-standard coordinate names
   config = marEx.PlotConfig(
       title='Temperature',
       var_units='°C',
       cmap='viridis',
       # Custom dimension mapping
       dimensions={'time': 'time', 'y': 'latitude', 'x': 'longitude'},
       coordinates={'time': 'time', 'y': 'latitude', 'x': 'longitude'}
   )

   fig, ax, im = data.plotX.single_plot(config)

Color Scaling Options
---------------------

.. code-block:: python

   # Percentile-based scaling
   config = marEx.PlotConfig(
       title='Temperature Anomalies',
       var_units='°C',
       cmap='RdBu_r',
       cperc=[5, 95],  # Use 5th and 95th percentiles
       extend='both'
   )

   # Symmetric scaling around zero
   config = marEx.PlotConfig(
       title='Temperature Anomalies',
       var_units='°C',
       cmap='RdBu_r',
       issym=True,     # Symmetric around zero
       extend='both'
   )

   # Manual color limits
   config = marEx.PlotConfig(
       title='Temperature',
       var_units='°C',
       cmap='plasma',
       clim=(-2, 5),   # Manual color limits
       extend='both'
   )

Integration with Matplotlib
===========================

Custom Figure Setup
--------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import cartopy.crs as ccrs

   # Create custom figure with specific projection
   fig = plt.figure(figsize=(12, 8))
   ax = plt.axes(projection=ccrs.PlateCarree())

   # Use existing axes
   config = marEx.PlotConfig(
       title='Custom Projection',
       var_units='°C',
       cmap='viridis'
   )

   fig, ax, im = data.plotX.single_plot(config, ax=ax)

   # Add custom elements
   ax.set_title('Modified Title', fontsize=16)
   plt.tight_layout()

Subplot Integration
-------------------

.. code-block:: python

   # Multi-panel comparison with custom layout
   fig, axes = plt.subplots(2, 2, figsize=(16, 12))

   datasets = [sst_obs, sst_model1, sst_model2, sst_diff]
   titles = ['Observations', 'Model 1', 'Model 2', 'Difference']

   for ax, dataset, title in zip(axes.flat, datasets, titles):
       config = marEx.PlotConfig(
           title=title,
           var_units='°C',
           cmap='RdBu_r',
           show_colorbar=False  # Add single colorbar later
       )

       fig, ax, im = dataset.plotX.single_plot(config, ax=ax)

   # Add single colorbar for all subplots
   fig.subplots_adjust(right=0.9)
   cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
   fig.colorbar(im, cax=cbar_ax, extend='both')


Geographic Projections
======================

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
=======================

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


Grid Detection
==============

The GriddedPlotter is automatically selected when:

* Data has separate latitude and longitude **dimensions**
* Coordinates follow structured grid patterns
* Typical dimension structure: `(time, lat, lon)` or `(lat, lon)`

.. code-block:: python

   # Automatic detection based on:
   # - 'lat' and 'lon' in data.dims
   # - Regular spacing in lat/lon coordinates
   # - Rectangular grid structure

   # Override detection if needed:
   marEx.specify_grid(grid_type='gridded')


See Also
========

* :mod:`marEx.plotX.base` - Base plotting functionality
* :mod:`marEx.plotX.unstructured` - Unstructured grid plotting
* :mod:`marEx.plotX` - Main plotting module
* `Cartopy documentation <https://scitools.org.uk/cartopy/docs/latest/>`_
* `xarray plotting guide <https://docs.xarray.dev/en/stable/user-guide/plotting.html>`_
