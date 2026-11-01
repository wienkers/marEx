=====================
Dimensions and Time
=====================

How marEx interprets the shape of your data: which dimensions are horizontal, which are
extra, and how the time axis sets the seasonal cycle. The anomaly and extremes modules
share these rules.

Input Requirements
==================

* An ``xarray.DataArray`` with a time dimension, backed by Dask.
* Land, or any other invalid cell, must be NaN at the **first timestep**. The ``mask`` is
  ``isfinite`` of that timestep and applies to the whole run. ``validate=True`` (default)
  checks that every unmasked cell is finite at every time, and ``validate=False`` skips the
  check.
* Every time value must be valid (no ``NaT``). Non-standard (cftime) calendars are
  handled.

.. code-block:: python

   import xarray as xr

   sst = xr.open_zarr("sst_daily.zarr").sst                       # Zarr
   sst = xr.open_mfdataset("sst_*.nc", chunks={}, parallel=True).sst   # many NetCDF files

``chunks={}`` keeps the on-disk chunking. Detection results are independent of the input
chunking, and on a lat/lon grid space can stay whole. :doc:`performance` covers chunking
for large runs, and :doc:`tracking` the chunking the tracker needs.

Gridded and Unstructured Inputs
===============================

A field is ``(time, *extra, *horizontal)``. marEx reads the horizontal dimensions from the
``dimensions`` mapping, and the latitude and longitude coordinate variables from
``coordinates``.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - Gridded
     - Unstructured
   * - Dimensions
     - ``(time, lat, lon)``
     - ``(time, ncells)``
   * - ``dimensions``
     - default ``{"time": "time", "x": "lon", "y": "lat"}``
     - ``{"time": "time", "x": "ncells"}`` (no ``"y"``)
   * - ``coordinates``
     - copied from ``dimensions``
     - **required**, for example ``{"time": "time", "x": "lon", "y": "lat"}``

A field is gridded when ``dimensions`` contains both ``"x"`` and ``"y"``, and unstructured
when it contains ``"x"`` alone. Any mapping that names a non-time dimension must include
``"x"``.

.. code-block:: python

   import marEx

   ds = marEx.preprocess_data(
       sst_cells,                                     # (time, ncells), Dask-backed
       dimensions={"time": "time", "x": "ncells"},
       coordinates={"time": "time", "x": "lon", "y": "lat"},
       threshold_percentile=95,
   )

Non-default names on a grid work the same way:
``dimensions={"time": "time", "x": "longitude", "y": "latitude"}``.

``preprocess_data`` also takes ``neighbours`` and ``cell_areas``. They are attached to the
output for the tracker (:doc:`tracking`), and the input can be chunked independently of
them.

Spatial pooling with ``window_spatial`` applies to gridded fields only. On a large
unstructured mesh, chunk the cell dimension of the input (for example
``{"time": 21, "ncells": 25000}``), because a spatially whole input turns the internal
regrouping into an all-to-all transpose. Tracking is the opposite and needs space whole.
See :doc:`performance`. Regional data spanning less than about 360° in longitude is tracked
with ``regional_tracker`` (:doc:`tracking`).

.. _extra-dimensions:

Extra Dimensions
================

Any dimension besides time and the horizontal ones is an extra dimension: depth, pressure
level, ensemble member. marEx detects it automatically and treats each index along it
independently, with no mapping entry needed.

.. code-block:: python

   # (time, depth, lat, lon)
   temp = xr.open_zarr("ocean_temperature.zarr").thetao

   ds = marEx.preprocess_data(
       temp,
       method_anomaly="fixed_baseline",
       method_extreme="seasonal_percentile",
       threshold_percentile=95,
   )
   ds.dat_anomaly        # (time, depth, lat, lon)
   ds.mask               # (depth, lat, lon)
   ds.extreme_events     # (time, depth, lat, lon)
   ds.thresholds         # (depth, lat, lon, dayofyear)

* A 3-D run gives the same result as separate 2-D runs on each level.
* ``mask`` has the extra dimension, so the seabed is masked by NaN at the first timestep of
  each level.
* ``window_spatial`` pools horizontal neighbours within each level.
* On a large grid each output chunk contains one level, and on a small one several levels
  share a chunk.
* The tracker and ``plotX`` take a 2-D field per timestep, so select a level with ``isel``
  (or loop over levels) before tracking or plotting. ``multi_plot(config, col="lev")`` draws
  one panel per level.

The :doc:`../tutorials/3d` tutorial runs this on a century of daily ocean temperature on seven
levels.

``dask_chunks`` (default ``{"time": 25}``) sets the time chunk of the output, and only its
time entry is used. An integer is a number of timesteps, and ``"auto"`` uses Dask's byte
budget (the ``array.chunk-size`` setting).

.. _cadence-rules:

Time Cadences
=============

marEx infers the **within-year cycle**, on which climatologies and seasonal thresholds are
resolved, from the median spacing of the time coordinate:

.. list-table::
   :header-rows: 1
   :widths: 24 22 18 36

   * - Median spacing
     - Cycle dimension
     - Slots
     - Typical data
   * - 28 days or more
     - ``month``
     - 12
     - Monthly means
   * - 1 day or more
     - ``dayofyear``
     - 366
     - Daily data (the default)
   * - Under 1 day
     - ``hourofyear``
     - ``366 × steps_per_day``
     - Hourly, 6-hourly

``marEx.infer_cycle(data.time)`` shows the inferred cycle. A 6-hourly axis gives
``SeasonalCycle(index_name="hourofyear", length=1464, step_days=0.25)``.

Window Durations
----------------

``window_days`` and ``smooth_days`` are always **durations in days**, whatever the cadence.
``smooth_days`` is converted to whole timesteps (to whole days at one time of day for
``shifting_baseline`` on sub-daily data). ``window_days`` pools neighbouring slots of the
cycle, so it is converted to cycle slots: one per day of year on daily and multi-day data,
one per step on sub-daily data, and one per month (the calendar's mean month) on monthly
data.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Cadence
     - ``window_days=11`` becomes
     - Effect
   * - Daily
     - 11 slots
     - Unchanged
   * - 5-day
     - 11 slots
     - 11 days, as on daily data
   * - 6-hourly
     - 45 slots
     - 11.25 days, with a warning about the rounding
   * - Monthly
     - 1 slot
     - 30.44 days (30 on a ``360_day`` calendar), with a warning

On a monthly axis an 11-day window becomes a single month, and the warning names the
requested and the realised duration. A window wider than the cycle pools every slot once.
On a day-of-year cycle (daily and multi-day data) ``window_days`` must be odd, and on other
cadences an even request is forced to an odd number of slots.

Methods by Cadence
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Method
     - Daily
     - Monthly
     - Sub-daily
   * - ``shifting_baseline``
     - yes
     - yes
     - yes
   * - ``fixed_baseline``
     - yes
     - yes
     - yes
   * - ``detrend_fixed_baseline``
     - yes
     - not tested
     - yes
   * - ``detrend_harmonic``
     - yes
     - not tested
     - **error**
   * - ``seasonal_percentile``
     - yes
     - yes
     - yes
   * - ``global_percentile``
     - yes
     - yes
     - yes

``detrend_harmonic`` rejects sub-daily data, because its basis has only annual and
semi-annual cycles and the diurnal cycle would stay in the anomaly. ``shifting_baseline``,
``fixed_baseline`` and ``detrend_fixed_baseline`` resolve their climatologies on the
sub-daily cycle and remove the diurnal cycle. ``shifting_baseline`` smooths each time of
day across days, which needs the timestamps on a regular sub-daily grid (missing steps may
be left out). Timestamps slightly off that grid, for example decoded from float offsets,
raise a ``ConfigurationError`` whose message gives the rounding that fixes them, such as
``da["time"] = da["time"].dt.round("3600s")`` for hourly data.

Sub-daily runs are expensive because each cell has more timesteps: 6-hourly data has four
times the steps of daily data over the same record, and hourly data twenty-four times. The seasonal threshold processes each cell's whole series in one task, so a task's memory depends on the record length.

Irregular axes and overrides
----------------------------

A time axis with no single characteristic spacing, for example a daily series appended to a
monthly one, or a cadence finer than about 8 minutes, needs an explicit ``cycle``:

.. code-block:: python

   ds = marEx.preprocess_data(
       data,
       method_anomaly="fixed_baseline",
       method_extreme="seasonal_percentile",
       cycle=marEx.SeasonalCycle("month", 12, 30.44),
   )

``preprocess_data``, ``anomaly.compute`` and ``extremes.identify`` all accept ``cycle=``,
and the chainer passes it to both modules. ``global_percentile`` needs no cycle.

Leap days
---------

The daily cycle has 366 slots. The day-366 climatology pools every year's 31 December, so a
leap year's last day has a climatology even when the window contains no other leap year
(:doc:`anomalies`).
