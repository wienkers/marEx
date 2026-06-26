=====================
Visualisation (plotX)
=====================

``plotX`` draws maps of marEx output (anomalies, extreme-event masks, tracked event IDs) from an
xarray accessor, on regular latitude/longitude grids and on unstructured meshes. See
:doc:`../api/plotx` for the :class:`marEx.PlotConfig` and :func:`marEx.specify_grid` reference.

.. contents::
   :local:
   :depth: 2

Overview
========

Importing ``marEx`` registers a ``.plotX`` accessor on every :class:`xarray.DataArray`. Calling
one of its methods builds a plotter for the data, and the plotter draws a Cartopy map. Three
methods cover the use cases:

``single_plot(config, ax=None)``
   One map for one timestep. Returns ``(fig, ax, im)``.

``multi_plot(config, col="time", col_wrap=3)``
   A wrapped grid of maps, one per entry along ``col``, with a shared colour scale and one
   colourbar. Returns ``(fig, axes)``.

``animate(config, plot_dir="./", file_name=None, centroids=None, object_ids=None)``
   An MP4 movie with one frame per timestep. Returns the file path, or ``None`` if ``ffmpeg``
   cannot be found.

All appearance options live in a :class:`marEx.PlotConfig`. Plotting is **two-dimensional**: a
field has one horizontal slice per timestep, and a field with a further dimension (depth,
level, member) is rejected with ``VisualisationError``. Select one level first, for example
``temperature.isel(depth=0)``.

Plotting needs ``matplotlib`` and ``cartopy``, which are installed with marEx. Movies also need
``pillow`` and an ``ffmpeg`` executable on the ``PATH`` (the ``plotting`` extra provides them,
or ``conda install -c conda-forge ffmpeg``).

Quick Start
===========

.. code-block:: python

   import numpy as np
   import xarray as xr
   import marEx

   lat = np.arange(-89.5, 90, 1.0)
   lon = np.arange(0.5, 360, 1.0)
   snapshot = xr.DataArray(
       np.random.default_rng(0).normal(size=(lat.size, lon.size)),
       dims=("lat", "lon"),
       coords={"lat": lat, "lon": lon},
       name="anomaly",
   )

   config = marEx.PlotConfig(title="Anomaly", var_units="K", issym=True)
   fig, ax, im = snapshot.plotX.single_plot(config)
   fig.savefig("anomaly.png", dpi=150)

``single_plot`` needs a single timestep. Pass ``data.isel(time=0)``, or a field whose time
dimension has length 1.

Grid Detection
==============

The accessor decides between the two backends from the dimensions of the data. A field is
**gridded** if the dimension named by ``dimensions["y"]`` (default ``lat``) is present, and
**unstructured** otherwise. For an unstructured field the cell dimension is the one that the
``x`` coordinate (default ``lon``) is defined on. A field with several non-time dimensions
where that cannot be decided (for example ``(time, depth, ncells)`` without a ``lon``
coordinate on ``ncells``) raises ``VisualisationError`` and asks for the dimensions
explicitly.

:func:`marEx.specify_grid` with ``grid_type="gridded"`` or ``"unstructured"`` overrides the
detection and logs a warning when it disagrees with the data.

Custom Dimension and Coordinate Names
-------------------------------------

Names other than ``time``, ``lat`` and ``lon`` are passed to the accessor call, not to
``PlotConfig``:

.. code-block:: python

   plotter = temperature.plotX(
       dimensions={"time": "time", "y": "latitude", "x": "longitude"},
       coordinates={"time": "time", "y": "latitude", "x": "longitude"},
   )
   fig, ax, im = plotter.single_plot(marEx.PlotConfig(title="Temperature"))

The accessor call returns a ``GriddedPlotter`` or an ``UnstructuredPlotter``, and all three
methods are available on it. The ``dimensions`` and ``coordinates`` fields of ``PlotConfig``
only select the title text in ``multi_plot``. The shortcuts ``da.plotX.single_plot(config)``
use the default names.

The PlotConfig
==============

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Field
     - Default
     - Meaning
   * - ``title``
     - ``None``
     - Axes title (``single_plot`` and ``animate`` frames).
   * - ``var_units``
     - ``""``
     - Colourbar label.
   * - ``cmap``
     - ``None``
     - Colormap name or object. ``None`` gives ``viridis``, or ``RdBu_r`` with ``issym=True``.
   * - ``issym``
     - ``False``
     - Make the colour limits symmetric about zero.
   * - ``cperc``
     - ``[4, 96]``
     - Percentiles that set the colour limits when ``clim`` and ``norm`` are not given.
   * - ``clim``
     - ``None``
     - Explicit ``(vmin, vmax)``. Overrides ``cperc``.
   * - ``norm``
     - ``None``
     - A matplotlib ``Normalize`` or ``BoundaryNorm``. Overrides both.
   * - ``extend``
     - ``"both"``
     - Colourbar extension: ``"neither"``, ``"both"``, ``"min"`` or ``"max"``.
   * - ``show_colorbar``
     - ``True``
     - Draw the colourbar.
   * - ``grid_lines``, ``grid_labels``
     - ``True``, ``False``
     - Dashed graticule, and whether it has labels.
   * - ``plot_IDs``
     - ``False``
     - Plot integer event IDs (see `Plotting Event IDs`_). Forces ``show_colorbar=False``.
   * - ``projection``
     - Robinson
     - Any Cartopy projection. Data are always interpreted as PlateCarree (lat/lon).
   * - ``framerate``
     - ``10``
     - Frames per second for ``animate``.
   * - ``ckdtree_res``
     - ``0.3``
     - Resolution (degrees) of the interpolation file used for unstructured movies.
   * - ``dimensions``, ``coordinates``
     - ``time``/``lat``/``lon``
     - Used for titles in ``multi_plot`` and as the animation's name mapping.
   * - ``verbose``, ``quiet``
     - ``None``
     - Logging controls.

Colour Scaling
--------------

Unless ``clim`` or ``norm`` is given, the limits are the ``cperc`` percentiles of the data. For
speed, every tenth timestep is sampled, and a very large field is also strided in space, so
the limits are approximate for a long record. Set ``clim`` for figures that must be
comparable between plots.

.. code-block:: python

   marEx.PlotConfig(cmap="RdBu_r", issym=True, cperc=[2, 98])     # symmetric about zero
   marEx.PlotConfig(cmap="viridis", clim=(-2, 5), extend="both")  # fixed limits

   from matplotlib.colors import BoundaryNorm
   norm = BoundaryNorm([-2, -1, 0, 1, 2], ncolors=256)
   marEx.PlotConfig(norm=norm, extend="both")                     # discrete levels

Single Plots and Existing Axes
==============================

``single_plot`` accepts an existing Cartopy axes, so a map can sit inside your own figure
layout. The axes need a projection:

.. code-block:: python

   import cartopy.crs as ccrs
   import matplotlib.pyplot as plt

   # `anomaly` is a (time, lat, lon) DataArray, for example from marEx.anomaly.compute
   fig = plt.figure(figsize=(12, 5))
   ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
   ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())

   config = marEx.PlotConfig(title="Day 1", clim=(-3, 3), issym=True)
   anomaly.isel(time=0).plotX.single_plot(config, ax=ax1)
   anomaly.isel(time=1).plotX.single_plot(config, ax=ax2)

Multi-Panel Plots
=================

``multi_plot`` places one map per entry along ``col`` (default ``"time"``), wrapped into
``col_wrap`` columns, with a single shared colourbar. The panel titles are the dates for a time
column, and ``name=value`` for any other dimension.

.. code-block:: python

   config = marEx.PlotConfig(title="Anomaly", var_units="K", issym=True)
   fig, axes = anomaly.isel(time=slice(0, 6)).plotX.multi_plot(config, col="time", col_wrap=3)

Select the panels before calling it: one map is rendered per entry, so slicing a decade of
daily data draws thousands.

Plotting Event IDs
==================

``plot_IDs=True`` is for the ``ID_field`` of the tracker (see :doc:`tracking`). IDs of 0 (the
background) are masked, every ID gets its own colour from a fixed random seed, and the
colourbar is switched off. Pass a ``cmap`` to use your own colours.

.. code-block:: python

   events = xr.open_zarr("tracked_events.zarr")
   config = marEx.PlotConfig(title="Tracked Events", plot_IDs=True)
   fig, ax, im = events.ID_field.isel(time=100).plotX.single_plot(config)

Animations
==========

``animate`` renders each timestep as a frame on the Dask scheduler and encodes the frames
with ``ffmpeg`` (H.264, ``yuv420p``, constant frame size). Frames are
rendered in batches, so a long record is not held in memory at once. If a Dask client is
running, the frames use its workers.

.. code-block:: python

   config = marEx.PlotConfig(
       title="Sea Surface Temperature Anomaly",
       var_units="K",
       issym=True,
       framerate=12,
   )
   movie_path = anomaly.plotX.animate(
       config, plot_dir="./animations", file_name="sst_anomaly"
   )

The file name gets ``.mp4`` appended when it is missing, and defaults to
``movie_<variable name>.mp4``. Colour limits are set once, from a sample of the record or from
``clim``, so the scale does not change between frames.

Overlaying Tracked Events
-------------------------

A movie of the extreme-event mask or the anomaly can show the tracker output as outlines and
centroid markers. ``object_ids`` takes an ID field (cells with ID > 0 are contoured as one
outline) and ``centroids`` the ``centroid`` array of the tracker, with dimensions
``(component, time, ID)`` in which ``component=0`` is latitude and ``1`` longitude:

.. code-block:: python

   movie_path = anomaly.plotX.animate(
       marEx.PlotConfig(title="Anomaly and Tracked Events", issym=True),
       plot_dir="./animations",
       file_name="events_on_anomaly",
       centroids=events.centroid,
       object_ids=events.ID_field,
   )

Unstructured Meshes
===================

An unstructured field has one cell dimension, with ``lat`` and ``lon`` as coordinates on it.
The accessor needs the mesh to draw it, supplied once with :func:`marEx.specify_grid`. There
are two rendering routes:

**Interpolation (``fpath_ckdtree``).** Cell values are mapped onto a regular lat/lon grid with
precomputed nearest-cell indices, and drawn with ``pcolormesh``. This is the fast route for
global maps and movies. If both paths are given, this route is used.

**Triangulation (``fpath_tgrid``).** The native triangles are drawn with ``tripcolor``. This
shows the true cell geometry and is slower on a large mesh.

.. code-block:: python

   marEx.specify_grid(
       grid_type="unstructured",
       fpath_tgrid="icon_grid.nc",          # triangulation
       fpath_ckdtree="./ckdtree_indices/",  # interpolation indices
   )

   sst = xr.open_zarr("icon_sst.zarr").sst   # dims (time, ncells), coords lon/lat on ncells
   fig, ax, im = sst.isel(time=0).plotX.single_plot(marEx.PlotConfig(title="ICON SST"))

Without either path, plotting raises ``VisualisationError`` that names the missing option.
``specify_grid`` is global to the process. To set the paths on one plotter only, use
``sst.plotX().specify_grid(fpath_tgrid=..., fpath_ckdtree=...)``.

The mesh files are read once and cached (``marEx.plotX.unstructured.clear_cache()`` empties the
cache).

File Formats
------------

Triangulation file (NetCDF)
   Variables ``vertex_of_cell`` (``(nvertices, ncells)`` as in ICON grid files, 1-based
   vertex indices), and ``clon`` and ``clat`` (longitude and latitude, in degrees). They are used as stored, with no unit conversion.

Interpolation directory
   One file per resolution named ``res<value>.nc`` (for example ``res0.30.nc``, with two
   decimals), each holding ``ickdtree_c`` (``(nlat, nlon)`` nearest-cell indices), ``lon`` and
   ``lat`` of the target regular grid. The resolution that is read is ``ckdtree_res``. For a
   movie it is ``PlotConfig.ckdtree_res``. For ``single_plot`` and ``multi_plot`` it is the
   ``ckdtree_res`` attribute of the plotter, which defaults to 0.3:

   .. code-block:: python

      plotter = sst.isel(time=0).plotX()
      plotter.ckdtree_res = 0.1            # reads ckdtree_indices/res0.10.nc
      fig, ax, im = plotter.single_plot(marEx.PlotConfig())

Regular Grids: Projections and Longitude Wrapping
=================================================

.. _plotx-grid-details:

The default display projection is Robinson, and any Cartopy projection can be passed as
``PlotConfig(projection=...)``. The data are always interpreted as PlateCarree
(regular latitude and longitude), so a field that is regular in lat/lon draws correctly on any
projection.

Global fields that span the full 360 degrees with the seam between the last and first column
get one extra column at ``lon + 360``, so the map has no gap at the date line. This applies when
``abs(360 - (lon.max() - lon.min())) < 2 * lon_spacing``. Regional fields, for example
``-180`` to ``-120`` or ``0`` to ``90``, are not wrapped.

Errors
======

Failures raise ``marEx.VisualisationError`` (or ``marEx.DependencyError`` when a plotting
dependency is missing). Both hold the cause and suggested fixes:

.. code-block:: python

   try:
       fig, ax, im = field.plotX.single_plot(config)
   except marEx.VisualisationError as e:
       print(e)
       print(e.suggestions)
   except marEx.DependencyError as e:
       print(f"Missing dependency: {e}")

Common causes: an extra dimension on the field (select a level), dimension or coordinate
names that differ from the defaults (pass ``dimensions`` and ``coordinates`` to the accessor
call), and an unstructured field without a mesh path (call ``specify_grid``).
