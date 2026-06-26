=====================
Dimensions and Time
=====================

How marEx reads the shape of your data: which dimensions are space, which are extra, how
the time axis sets the seasonal cycle, and what is not supported. The anomaly and extremes
stages share these rules.

Input Requirements
==================

* An ``xarray.DataArray`` with a time dimension, backed by Dask. A NumPy-backed array raises
  a ``DataValidationError``.
* Land, or any other invalid cell, must be NaN at the **first timestep**. The ``mask`` is
  ``isfinite`` of that timestep and holds for the whole run. With ``validate=True`` (default)
  any non-finite value at an unmasked cell, at any time, raises a ``DataValidationError``.
  ``validate=False`` skips the check.
* The data is cast to float32. Non-standard (cftime) calendars are handled. A
  ``NaT`` in the time axis raises a ``ConfigurationError``.

.. code-block:: python

   import xarray as xr

   sst = xr.open_zarr("sst_daily.zarr").sst                       # Zarr
   sst = xr.open_mfdataset("sst_*.nc", chunks={}, parallel=True).sst   # many NetCDF files

``chunks={}`` keeps the on-disk chunking. Detection does not depend on how the input is
chunked, and on a lat/lon grid keeping space whole is fine. Chunking advice for large runs is
in :doc:`performance`. Tracking has different chunking requirements and is covered in
:doc:`tracking`.

Gridded and Unstructured Inputs
===============================

A field is ``(time, *extra, *horizontal)``. marEx reads the horizontal dimensions from the
``dimensions`` mapping, and the coordinate variables that hold latitude and longitude from
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

A field is treated as gridded when both ``"x"`` and ``"y"`` appear in ``dimensions``. A
mapping that names any non-time dimension must therefore carry ``"x"``. A mapping with
``"y"`` and no ``"x"`` raises a ``DataValidationError`` instead of silently treating the
longitude axis as an extra dimension. Unstructured data without ``coordinates`` raises the
same error.

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
output and used by the tracker, and place no constraint on how the input is chunked. Their
use in tracking is described in :doc:`tracking`.

Spatial pooling with ``window_spatial`` is gridded-only. On a large unstructured mesh, chunk
the cell dimension of the input (for example ``{"time": 21, "ncells": 25000}``), because
leaving space whole turns the internal regrouping into an all-to-all transpose. Unstructured
detection on a large mesh left spatially whole is not recommended. Tracking is the opposite and
needs space whole. See :doc:`performance`.

.. _extra-dimensions:

Extra Dimensions
================

Any dimension besides time and the horizontal ones is an extra dimension: depth, pressure
level, ensemble member. marEx detects it automatically and treats each index along it
independently. No mapping entry is needed.

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

* A 3-D run equals the per-level 2-D runs, which the slice-equivalence tests in the suite check.
* ``mask`` has the extra dimension, so seabed depth is handled by NaN at the first timestep
  at each level.
* ``window_spatial`` pools horizontal cells only, never across depth.
* On a large grid each chunk of the output holds one level. On a small one several levels
  share a chunk. ``dask_chunks`` sets the time chunk only.
* Both the tracker and ``plotX`` accept a 2-D field per timestep. Select a level with
  ``isel`` (or loop over levels) before tracking or plotting. They raise an error that says
  so otherwise.

``dask_chunks`` defaults to ``{"time": 25}`` and only its time entry is used. An integer is a
number of timesteps. ``"auto"`` is Dask's byte budget (the ``array.chunk-size`` setting).

.. _cadence-rules:

Time Cadences
=============

marEx infers the **within-year cycle** that climatologies and seasonal thresholds are resolved
on from the median spacing of the time coordinate:

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

``marEx.infer_cycle(data.time)`` shows what would be inferred. A 6-hourly axis gives
``SeasonalCycle(index_name="hourofyear", length=1464, step_days=0.25)``.

Durations are in days
---------------------

``window_days`` and ``smooth_days`` are always **durations in days**, whatever the cadence,
and are converted to whole timesteps.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Cadence
     - ``window_days=11`` becomes
     - Effect
   * - Daily
     - 11 steps
     - Unchanged
   * - 6-hourly
     - 45 steps
     - 11.25 days, with a warning about the rounding
   * - Monthly
     - 1 step
     - 31 days, with a warning

An 11-day window cannot be represented on a monthly axis, so marEx uses a single month and
warns, naming the requested and the realised duration. On a daily axis ``window_days`` must be
odd. On other cadences the window is forced to an odd number of steps and an even request
does not raise. ``smooth_days`` is not forced odd, and on a monthly axis the default 21 days
rounds below one step, so no smoothing is applied and a warning says so.

Which methods run where
-----------------------

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
     - not recommended
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

``detrend_harmonic`` raises a ``ConfigurationError`` on sub-daily data, because its basis
removes annual and semi-annual cycles only and the diurnal cycle would remain in the anomaly.
For sub-daily data ``shifting_baseline`` is likewise not recommended. A test on a 6-hourly
fixture left a diurnal cycle larger than the signal in the anomaly. Use ``fixed_baseline``
or ``detrend_fixed_baseline``, whose climatologies are resolved on the sub-daily cycle and
remove the diurnal cycle.

Sub-daily runs are expensive. The threshold histogram is ``cycle_length × number of bins`` per cell,
so 6-hourly data is four times the daily size and hourly data twenty-four times. The internal
tiling shrinks each task in proportion, which keeps the working set bounded and multiplies the
task count. For long sub-daily series, prefer ``global_percentile`` or coarsen to daily first.
A tile-fit warning at the shipped chunk defaults on sub-daily data names a rechunk that
avoids it.

Irregular axes and overrides
----------------------------

A time axis with no single characteristic spacing, for example a daily series appended to a
monthly one, raises a ``ConfigurationError`` instead of being guessed. So does a cadence finer
than about 8 minutes. Pass the cycle yourself:

.. code-block:: python

   ds = marEx.preprocess_data(
       data,
       method_anomaly="fixed_baseline",
       method_extreme="seasonal_percentile",
       cycle=marEx.SeasonalCycle("month", 12, 30.44),
   )

``cycle=`` is accepted by ``preprocess_data``, ``anomaly.compute`` and ``extremes.identify``,
and the chainer passes it to both stages. ``global_percentile`` never needs a cycle.

Leap days
---------

The daily cycle has 366 slots. With ``shifting_baseline``, a window that contains no leap
year leaves day 366 without a climatology, so that one timestep has a NaN anomaly. This is
expected and is described in :doc:`anomalies`.

What Is Not Supported
=====================

* **Tracking and plotting of fields with extra dimensions.** Select one level first.
* **A time axis with mixed cadences**, without an explicit ``cycle`` (``global_percentile`` is
  exempt).
* **Sub-daily data with** ``detrend_harmonic``. ``shifting_baseline`` is not recommended
  there.
* **tail="both".** Run the lower and upper tails separately.
* **Spatial pooling on unstructured meshes**, with ``global_percentile``, or with exact
  percentiles.
* **Unstructured detection on a large mesh left spatially whole.** It thrashes. Chunk the cell dimension.
* **Regional tracking** on data that does not span about 360° in longitude needs
  ``regional_tracker``. See :doc:`tracking`.
