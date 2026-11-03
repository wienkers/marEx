==============================
Tracking (:mod:`marEx.track`)
==============================

.. currentmodule:: marEx

The :mod:`marEx.track` module identifies and tracks coherent extreme events
through time, handling merging and splitting. For parameter explanations
(``R_fill``, ``T_fill``, ``overlap_threshold``, ``nn_partitioning``) and worked
examples, see :doc:`../guide/tracking`.

.. autosummary::
   :nosignatures:

   tracker
   regional_tracker

Detailed reference
==================

.. autoclass:: tracker
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: regional_tracker

4-D tracking
============

Declaring ``dimensions["z"]`` makes :class:`marEx.tracker` return a 4-D tracker. It takes
the parameters of :class:`marEx.tracker` and the keyword-only parameters below. The guide
section :ref:`tracking-in-4d` explains them. ``nn_partitioning`` defaults to ``True`` (``False`` raises) and ``allow_merging`` must stay ``True``.
The ``area_filter_*`` and ``prefilter_min_cells`` parameters of the 3-D tracker raise.

``layer_bounds`` : :class:`xarray.DataArray` or ``"from_centres"``
   Layer interfaces in metres, ``(depth, 2)`` or ``(depth, lat, lon, 2)``. ``"from_centres"``
   reconstructs them from the level centres (approximate).

``layer_thickness`` : :class:`xarray.DataArray`, optional
   Thickness in metres, ``(depth, lat, lon)``, for partial cells. Requires ``layer_bounds``.

``cell_volumes`` : :class:`xarray.DataArray`, optional
   Volumes in m\ :sup:`3`, ``(depth, lat, lon)``, replacing area times thickness.

``max_vertical_gap_m`` : float, default 0.0
   Largest total thickness in metres of unoccupied wet cells bridged within a water column.
   0 joins consecutive stored levels only.

``volume_filter_quartile`` : float in [0, 1], optional
   Fraction of the smallest objects to remove, pooled over time. Default 0.5 when neither
   volume filter is given.

``volume_filter_absolute`` : float, optional
   Minimum object volume in m\ :sup:`3`. Mutually exclusive with ``volume_filter_quartile``.

``partition_scales`` : tuple of float, optional
   ``(L_H, L_Z)`` in metres, both positive. ``None`` derives them from the grid.

.. autoclass:: marEx.track.volumetric.tracker4d.VolumetricTracker
   :show-inheritance:
