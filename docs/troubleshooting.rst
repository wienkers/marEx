.. _troubleshooting-guide:

===============
Troubleshooting
===============

Most failures in a large marEx run are memory failures that surface under another name. Start with
:ref:`troubleshooting-reading-failures`, then use the sections below for configuration errors and
installation problems.

Quick Diagnostic Checklist
==========================

.. code-block:: python

   import dask
   import marEx

   print("marEx", getattr(marEx, "__version__", "development"))
   print("dask", dask.__version__)
   marEx.print_dependency_status()

   print(your_data.dims, your_data.shape)
   print(your_data.chunks)            # None means the array is not dask-backed

Every stage requires dask-backed input. ``DataValidationError`` at the first call usually means
the array was opened without ``chunks``.

.. _troubleshooting-reading-failures:

Reading a Failure
=================

``KilledWorker``, ``P2PConsistencyError`` and a job that hits its wall-clock limit are
**symptoms**. They are almost never the cause, and a run that dies with any of them should be
treated as a memory problem until the logs show otherwise.

The usual chain is an oversized task, a worker that crosses 95 % of its memory limit, a restart by the
nanny, and then an error somewhere else. A shuffle that loses its state when its worker restarts
raises ``P2PConsistencyError: No active shuffle with id=... found``. A task that has failed too often
raises ``KilledWorker``. A run that restarts workers repeatedly makes no progress and is eventually
stopped by the scheduler's wall time, which looks like a run that was too slow.

Look for the cause above the final traceback:

.. code-block:: text

   distributed.nanny.memory - WARNING - Worker ... exceeded 95% memory budget. Restarting...

.. _troubleshooting-worker-warnings:

The Absence of Worker Warnings Proves Nothing
---------------------------------------------

marEx sets the ``distributed.worker``, ``distributed.scheduler``, ``distributed.comm`` and
``distributed.core`` loggers to ``ERROR`` so that routine output stays readable. That hides the
worker-side memory warnings, such as the "unmanaged memory is high" and "worker is at 80 % memory"
messages. If the lines above are missing from your log, check the dashboard or the batch system's
accounting record before concluding that memory was not the problem. To see the worker warnings,
restore the level for the run:

.. code-block:: python

   import logging

   for name in ("distributed.worker", "distributed.scheduler"):
       logging.getLogger(name).setLevel(logging.WARNING)

What to Try, in Order
---------------------

1. **Find the real error.** Search the log for ``exceeded 95%`` and read what came before it.
2. **Compute the per-task working set** and compare it with one worker's ``memory_limit`` divided
   among its threads. This is the most common miss. For detect on a large mesh the working set is
   ``n_time × cells in one input spatial chunk × 4 B`` (see :ref:`performance-chunking`).
3. **Check the allocation.** Compare the batch system's maximum resident memory with the memory you
   requested, to rule out the client, the scheduler or the cgroup rather than the workers.
4. **Check ``memory_limit``.** If you did not pass one, workers may be sized from the whole machine and
   not from your allocation (see :ref:`performance-memory-limit`).
5. **Look at the store's chunks** and how far your requested chunking is from them.
6. **Reduce threads per worker before reducing workers.** Threads multiply concurrent task memory
   within one limit, and workers do not.
7. **Only then change ``compute_mode``.** It addresses pinned data, which is a different problem from
   an oversized task.

Change one thing at a time. Shortening a failing run does not reliably make it cheaper per task:
the internal tiling divides a fixed element budget by the length of the reduced axis, so a shorter
series can produce a larger spatial tile. Reduce the spatial extent or the tile size instead.

Do not turn off dask's spilling to make a memory test cleaner. Workers that cannot spill pause for
good, the cluster deadlocks, and both compute modes fail.

Streaming Staging Left Behind
=============================

A ``compute_mode="streaming"`` run stages intermediates in a directory that must outlive the call.
It is removed by ``marEx.clear_staging(ds)`` after you have written your output. A cleanup hook also
runs when the interpreter exits normally, but it cannot run when the process is killed. A job stopped by
``SIGKILL``, an out-of-memory kill or a wall-clock limit can leave the directory behind, and the
tracker's ``temp_dir`` accumulates stores the same way.

* Detect: directories named ``marex_stage_<pid>_<hash>`` under the ``scratch_dir`` you passed.
* Tracker: stores named ``marEx_temp_*.zarr`` under ``temp_dir``. Stores older than 24 hours are
  pruned when a new tracker is constructed.

Remove the directories of runs that are no longer alive. The path of a live result is
``ds.encoding["marex_staging_dir"]``. If you merged or concatenated the dataset, the encoding is gone
and ``clear_staging`` only warns, so pass the path you recorded earlier.

.. _troubleshooting-errors:

Configuration Errors
====================

Errors are raised before any compute starts wherever the problem can be seen up front. The messages
below are the common ones.

.. list-table::
   :header-rows: 1
   :widths: 34 36 30

   * - Message or condition
     - Cause
     - Fix
   * - ``Unknown extreme method`` / ``Unknown anomaly method``
     - A name that is not one of the current options.
     - Use ``"global_percentile"`` or ``"seasonal_percentile"`` for ``method_extreme``, and see
       :doc:`guide/anomalies` for ``method_anomaly``. See :doc:`whats_new` for the renamed options.
   * - ``window_days must be an odd number``
     - Even ``window_days`` on a daily time axis.
     - Use an odd number of days.
   * - ``window_spatial`` rejected
     - ``window_spatial`` is only valid for a gridded input with ``seasonal_percentile`` and
       ``method_percentile="approximate"``. It must be odd.
     - Leave it unset, or change the combination.
   * - ``precision`` rejected
     - ``precision`` was given with ``method_percentile="exact"``, or it implies more than 65,000 bins
       (a warning appears above 10,000).
     - Drop ``precision`` for the exact path, or use a coarser value.
   * - ``detrend_harmonic`` does not support sub-daily data
     - The harmonic method works on daily data.
     - Use ``shifting_baseline``, ``fixed_baseline`` or ``detrend_fixed_baseline``, or resample to
       daily.
   * - ``standardise`` rejected
     - ``standardise=True`` needs ``detrend_harmonic``.
     - Change the method or set ``standardise=False``.
   * - ``reference_period`` rejected
     - It is only used by ``fixed_baseline`` and ``detrend_fixed_baseline``.
     - Remove it for the other methods.
   * - ``smooth_days`` spans the whole cycle
     - The smoothing window is as long as one seasonal cycle.
     - Use a shorter ``smooth_days``.
   * - Time axis "too irregular" or "too finely resolved"
     - The cadence could not be inferred from the median step.
     - Pass ``cycle=marEx.SeasonalCycle(...)``, or use ``global_percentile``, which does not need
       a cycle.
   * - ``Cannot identify extremes: the anomaly series is empty``
     - ``shifting_baseline`` trims the first ``window_years`` years, and nothing was left.
     - Use a longer record or a smaller ``window_years``. A series that does not span
       ``window_years`` raises ``DataValidationError``.
   * - ``DataValidationError`` on ``dimensions``
     - A mapping names a non-time dimension without an ``"x"`` entry.
     - Give ``"x"`` (and ``"y"`` for a lat/lon grid). Unstructured data also needs
       ``coordinates``.
   * - ``DataValidationError`` on NaN or infinite values
     - The land mask is taken from the first timestep, and a valid cell contains NaN later.
     - Make the invalid cells NaN at the first step, or pass ``validate=False`` if you accept that.
   * - ``R_fill is required``
     - The tracker has no default radius.
     - Pass ``R_fill``. Pass the other arguments by keyword, since a positional second argument
       binds to ``mask``.
   * - ``T_fill`` rejected
     - ``T_fill`` must be even.
     - Use 0 (skip), 2, 4 and so on.
   * - Both ``area_filter_quartile`` and ``area_filter_absolute``
     - They are mutually exclusive.
     - Pass one.
   * - ``compute_mode="lazy"`` on the tracker
     - The tracker has no lazy mode.
     - Use ``"persist"`` or ``"streaming"``.
   * - Streaming without ``temp_dir``
     - The tracker stages to ``temp_dir``.
     - Pass a directory with enough free space. Budget about 14 bytes per cell-timestep.
   * - ``TrackingError`` about extra dimensions
     - The tracker and the plotting accessor are 2-D only.
     - Select one level, for example ``da.isel(depth=0)``, or loop over levels.
   * - ``TrackingError`` on a second ``run()``
     - A tracker instance is single use.
     - Construct a new tracker.
   * - ``CoordinateError`` on longitude range
     - The tracker detects coordinate units from a roughly 360 degree range.
     - For regional data use ``regional_tracker`` with ``coordinate_units``.
   * - Unstructured tracker rejects the input
     - It needs ``neighbours``, ``cell_areas`` and ``temp_dir``, and every time chunk must hold at
       least two steps.
     - Supply them, and re-chunk after any ``.sel`` so that no final chunk has a single step.
   * - Time chunks rejected in streaming
     - The tracker needs uniformly chunked time.
     - It re-chunks with a warning. Chunk time uniformly yourself to avoid the cost.

Day 366 is NaN
--------------

With ``shifting_baseline``, a trailing window that spans no leap year leaves day-of-year 366
undefined, which is exactly one timestep. This is expected.

.. _troubleshooting-tile-warning:

Warning About the Per-Task Element Budget
=========================================

On sub-daily data you may see a warning that one task "exceeds the per-task element budget". The
per-cell output of the day-of-year histogram grows with the number of steps per day, while the
spatial window set by ``window_spatial`` does not shrink, so at the default ``window_spatial=5`` the
tile the budget allows is smaller than the window needs. marEx uses the larger tile and warns, and the
warning never alters the result. It names the estimate, the budget and the window that forced it.

If the run then dies on memory, apply one of these:

* a narrower ``window_spatial``,
* a shorter ``window_years``,
* a coarser ``precision`` (fewer bins),
* fewer threads per worker at the same ``memory_limit``,
* ``compute_mode="streaming"``.

Memory and Speed
================

Memory Errors
-------------

``MemoryError``, ``KilledWorker`` and restarting workers are covered in
:ref:`troubleshooting-reading-failures`. The settings that most often help:

.. code-block:: python

   import marEx

   client = marEx.helper.start_local_cluster(
       n_workers=2,
       threads_per_worker=1,
       memory_limit="8GB",       # always pass this
   )

Aggregate memory does not rescue an oversized task. Only a larger per-worker limit or a smaller task
does.

Slow Runs
---------

* Open the dask dashboard (``marEx.helper.get_cluster_info(client)`` prints the link and the port
  forwarding command) and look for idle workers, which usually mean a chunking mismatch.
* Time the stages with :class:`marEx.helper.ResourceMonitor` (see :ref:`performance-resource-monitor`).
* Do not read a long wall time as evidence of a slow algorithm until you have checked for worker
  restarts.
* A chunk size of 25 steps is a sound start for the tracker and a modest time chunk for detect. The
  shipped examples run with chunks of 1.9 MB (lat/lon detect), 25.9 MB (lat/lon tracker) and
  59.5 MB (unstructured tracker). No single target size suits every stage.

Unexpected Results
------------------

* Check the input for the units you expect and for a land mask that is NaN at the first step.
* ``dat_anomaly`` should have a mean close to zero.
* The fraction of ``extreme_events`` should be close to ``100 - threshold_percentile`` per cell. With
  few years of data the fraction can be off, and marEx logs a warning when fewer than 50 samples lie
  beyond the threshold.
* Approximate thresholds differ from exact ones by up to a few histogram bins. See
  :doc:`guide/validation`.

Installation
============

``ModuleNotFoundError: No module named 'marEx'``
   Install into the environment you are running, then check with ``python -m pip show marEx``. For
   a source checkout use ``pip install -e .``.

Dependency conflicts
   Use a clean environment. marEx needs Python 3.10 or later and ``dask[complete]`` 2025.9 or
   later. Both Zarr 2.18 and Zarr 3 are supported.

Missing optional features
   ``pip install marEx[full]`` installs the optional extras. ``marEx.print_dependency_status()``
   lists what is missing. SLURM clusters need ``dask-jobqueue``.

``ImportWarning`` about JAX at import
   JAX is optional. Without it marEx uses NumPy, and one routine in the unstructured tracker builds
   its dilation matrix more slowly. The warning is hidden by default Python filters.

Coordinates
===========

``KeyError: 'lat'`` or a coordinate that cannot be found
   Print ``data.coords`` and ``data.dims``, and pass ``dimensions=`` and ``coordinates=`` that name
   what your data calls them. Unstructured data always needs ``coordinates``.

Getting Help
============

Report problems on the GitHub issue tracker. Include the marEx version, the Python version, the data
description (size, grid type and chunks), the full traceback with the log lines above it, and a minimal
example on synthetic data if you can make one.
