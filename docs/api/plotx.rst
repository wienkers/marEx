==================================
Visualisation (:mod:`marEx.plotX`)
==================================

.. currentmodule:: marEx

The :mod:`marEx.plotX` module provides visualisation through an xarray accessor
(``.plotX``) that auto-detects structured vs. unstructured grids. For usage
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

Plotter backends
================

The accessor dispatches to one of two backends depending on grid type.

.. autoclass:: marEx.plotX.gridded.GriddedPlotter
   :members:
   :show-inheritance:

.. autoclass:: marEx.plotX.unstructured.UnstructuredPlotter
   :members:
   :show-inheritance:
