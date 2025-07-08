=============
API Reference
=============

This page provides a comprehensive reference for all public functions, classes, and modules in marEx.

Main Package
============

.. automodule:: marEx
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
==============

The main entry points for using marEx are:

.. autosummary::
   :toctree: _autosummary

   marEx.preprocess_data
   marEx.tracker
   marEx.regional_tracker
   marEx.specify_grid

Data Preprocessing
==================

Functions for data preprocessing, anomaly detection, and extreme event identification:

.. autosummary::
   :toctree: _autosummary

   marEx.compute_normalised_anomaly
   marEx.smoothed_rolling_climatology
   marEx.rolling_climatology
   marEx.identify_extremes

Visualisation
=============

Plotting configuration and utilities:

.. autosummary::
   :toctree: _autosummary

   marEx.PlotConfig
   marEx.specify_grid

Exception Hierarchy
===================

MarEx provides a structured exception hierarchy for precise error handling:

.. autosummary::
   :toctree: _autosummary

   marEx.MarExError
   marEx.DataValidationError
   marEx.CoordinateError
   marEx.ProcessingError
   marEx.ConfigurationError
   marEx.DependencyError
   marEx.TrackingError
   marEx.VisualisationError
   marEx.create_data_validation_error
   marEx.create_coordinate_error
   marEx.create_processing_error
   marEx.wrap_exception

Logging System
==============

Configurable logging system for development and debugging:

.. autosummary::
   :toctree: _autosummary

   marEx.configure_logging
   marEx.set_verbose_mode
   marEx.set_quiet_mode
   marEx.set_normal_logging
   marEx.get_verbosity_level
   marEx.is_verbose_mode
   marEx.is_quiet_mode
   marEx.get_logger

Dependency Management
=====================

Utilities for managing optional dependencies:

.. autosummary::
   :toctree: _autosummary

   marEx.has_dependency
   marEx.print_dependency_status
   marEx.get_installation_profile

Modules
=======

.. toctree::
   :maxdepth: 2

   modules/detect
   modules/track
   modules/plotx
   modules/helper

Detection Module (:mod:`marEx.detect`)
=======================================

.. automodule:: marEx.detect
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autosummary::
   :toctree: _autosummary

   marEx.detect.preprocess_data
   marEx.detect.compute_normalised_anomaly
   marEx.detect.smoothed_rolling_climatology
   marEx.detect.rolling_climatology
   marEx.detect.identify_extremes

Tracking Module (:mod:`marEx.track`)
=====================================

.. automodule:: marEx.track
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autosummary::
   :toctree: _autosummary

   marEx.track.tracker

Functions
---------

.. autosummary::
   :toctree: _autosummary

   marEx.track.regional_tracker

Plotting Module (:mod:`marEx.plotX`)
=====================================

.. automodule:: marEx.plotX
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Classes
---------------------

.. autosummary::
   :toctree: _autosummary

   marEx.plotX.PlotConfig

Functions
---------

.. autosummary::
   :toctree: _autosummary

   marEx.plotX.specify_grid

Submodules
----------

.. toctree::
   :maxdepth: 1

   modules/plotx/base
   modules/plotx/gridded
   modules/plotx/unstructured

Helper Module (:mod:`marEx.helper`)
====================================

.. automodule:: marEx.helper
   :members:
   :undoc-members:
   :show-inheritance:

HPC Cluster Management
----------------------

.. autosummary::
   :toctree: _autosummary

   marEx.helper.start_distributed_cluster
   marEx.helper.start_local_cluster
   marEx.helper.configure_dask
   marEx.helper.get_cluster_info
   marEx.helper.fix_dask_tuple_array

Detailed API Documentation
===========================

Detection and Preprocessing
----------------------------

.. autofunction:: marEx.preprocess_data

.. autofunction:: marEx.compute_normalised_anomaly

.. autofunction:: marEx.smoothed_rolling_climatology

.. autofunction:: marEx.rolling_climatology

.. autofunction:: marEx.identify_extremes

Event Tracking
---------------

.. autoclass:: marEx.tracker
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: marEx.regional_tracker

Visualisation
-------------

.. autoclass:: marEx.PlotConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: marEx.specify_grid

Exception Hierarchy
-------------------

.. autoexception:: marEx.MarExError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoexception:: marEx.DataValidationError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoexception:: marEx.CoordinateError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoexception:: marEx.ProcessingError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoexception:: marEx.ConfigurationError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoexception:: marEx.DependencyError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoexception:: marEx.TrackingError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoexception:: marEx.VisualisationError
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: marEx.create_data_validation_error

.. autofunction:: marEx.create_coordinate_error

.. autofunction:: marEx.create_processing_error

.. autofunction:: marEx.wrap_exception

Logging System
--------------

.. autofunction:: marEx.configure_logging

.. autofunction:: marEx.set_verbose_mode

.. autofunction:: marEx.set_quiet_mode

.. autofunction:: marEx.set_normal_logging

.. autofunction:: marEx.get_verbosity_level

.. autofunction:: marEx.is_verbose_mode

.. autofunction:: marEx.is_quiet_mode

.. autofunction:: marEx.get_logger

Dependency Management
---------------------

.. autofunction:: marEx.has_dependency

.. autofunction:: marEx.print_dependency_status

.. autofunction:: marEx.get_installation_profile
