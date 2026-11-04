==================================
Visualisation (:mod:`marEx.plotX`)
==================================

.. currentmodule:: marEx

The :mod:`marEx.plotX` module provides visualisation through an xarray accessor
(``.plotX``) that detects whether the grid is structured or unstructured. For usage
patterns and examples, see :doc:`../guide/visualisation`.

.. autosummary::
   :nosignatures:

   PlotConfig
   specify_grid

Detailed reference
==================

.. autoclass:: PlotConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: specify_grid

4-D events
==========

Four functions draw the output of a 4-D tracker, a ``(time, depth, lat, lon)`` events
dataset (see :ref:`tracking-in-4d`). Each selects what it needs from the lazy ``ID_field``
before loading it. The two map functions take an ``extent`` in degrees to frame the events,
and ``plot_event_depth_time`` can draw depth on a square-root scale.

.. autofunction:: marEx.plotX.plot_event_levels

.. autofunction:: marEx.plotX.plot_event_depth_time

.. autofunction:: marEx.plotX.plot_event_summary

.. autofunction:: marEx.plotX.animate_event_levels

Plotter backends
================

The accessor dispatches to one of two backends depending on grid type.

.. autoclass:: marEx.plotX.gridded.GriddedPlotter
   :members:
   :show-inheritance:

.. autoclass:: marEx.plotX.unstructured.UnstructuredPlotter
   :members:
   :show-inheritance:
