=============
API Reference
=============

Complete reference for the public functions, classes, and modules in ``marEx``.
For narrative explanations and worked examples, see the :doc:`../guide/index`.

.. currentmodule:: marEx

Core entry points
=================

The main objects you will use directly (full documentation on the module pages
linked below):

.. autosummary::
   :nosignatures:

   anomaly.compute
   extremes.identify
   preprocess_data
   tracker
   regional_tracker
   specify_grid
   PlotConfig
   SeasonalCycle
   infer_cycle

Module reference
================

.. toctree::
   :maxdepth: 2

   anomaly
   extremes
   track
   plotx
   helper

Time resolution
===============

The within-year cycle a climatology or seasonal threshold is resolved on. Inferred
from the median spacing of the time coordinate -- ``dayofyear`` for daily data,
``month`` for monthly, ``hourofyear`` for sub-daily -- and overridable via the
``cycle=`` parameter on :func:`preprocess_data`, :func:`marEx.anomaly.compute` and
:func:`marEx.extremes.identify`. See :doc:`../guide/detection` for the durations table
and the sub-daily caveats.

.. autoclass:: SeasonalCycle
   :members:

.. autofunction:: infer_cycle

Exception hierarchy
===================

marEx provides a structured exception hierarchy for precise error handling.

.. autosummary::
   :toctree: ../_autosummary

   MarExError
   DataValidationError
   CoordinateError
   ProcessingError
   ConfigurationError
   DependencyError
   TrackingError
   VisualisationError
   create_data_validation_error
   create_coordinate_error
   create_processing_error
   wrap_exception

Logging system
==============

.. autosummary::
   :toctree: ../_autosummary

   configure_logging
   set_verbose_mode
   set_quiet_mode
   set_normal_logging
   get_verbosity_level
   is_verbose_mode
   is_quiet_mode
   get_logger

Dependency management
=====================

.. autosummary::
   :toctree: ../_autosummary

   has_dependency
   print_dependency_status
   get_installation_profile
