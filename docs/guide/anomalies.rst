=========
Anomalies
=========

The ``marEx.anomaly`` module subtracts a climatology from the raw field. Call it alone as
:func:`marEx.anomaly.compute`, or as the first half of :func:`marEx.preprocess_data`. The
function reference is in :doc:`../api/anomaly`.

.. code-block:: python

   import xarray as xr
   import marEx

   sst = xr.open_zarr("sst_daily.zarr").sst      # Dask-backed

   anomalies = marEx.anomaly.compute(
       sst,
       method="shifting_baseline",
       window_years=15,
       smooth_days=21,
   )
   anomalies.dat_anomaly      # (time, lat, lon), float32
   anomalies.mask             # (lat, lon), True where valid

The output Dataset contains ``dat_anomaly`` and ``mask``, plus ``dat_stn`` and ``STD`` when
``standardise=True``.

.. _anomaly-methods:

Choosing a Method
=================

The four methods differ in how the climatology is built and what happens to long-term
trends.

.. list-table::
   :header-rows: 1
   :widths: 22 36 20 22

   * - Method
     - Climatology
     - Trend in anomaly
     - Series length
   * - ``shifting_baseline``
     - Rolling, from the ``window_years`` preceding years
     - Followed by the baseline
     - Shortened by ``window_years`` years
   * - ``fixed_baseline``
     - Day-of-year mean over the whole series or a ``reference_period``
     - Kept
     - Unchanged
   * - ``detrend_fixed_baseline``
     - Polynomial trend removed, then a fixed day-of-year mean
     - Removed
     - Unchanged
   * - ``detrend_harmonic``
     - Polynomial trend plus annual and semi-annual harmonics, fitted by regression
     - Removed
     - Unchanged

**Use** ``shifting_baseline`` **when the question is about extremes relative to a recent
normal.** It is the default. Each year is compared with the years just before it, so the
baseline follows slow warming. It is the most expensive method, and it needs a series
longer than ``window_years``, whose first ``window_years`` years are dropped from the
output.

**Use** ``fixed_baseline`` **when the baseline itself is part of the question**, as in a
heatwave definition tied to a reference period such as ``reference_period=(1991, 2020)``.
The trend stays in the anomaly, so later years exceed a fixed threshold more often.

**Use** ``detrend_fixed_baseline`` **for a fixed climatology without the trend**, for
example to study variability without the warming signal, or when the record is too short
to spend ``window_years`` of it on a baseline.

**Use** ``detrend_harmonic`` **for fast exploratory runs, or with** ``standardise=True``.
Fitting a handful of harmonics is cheaper than building a day-of-year climatology.
The harmonics approximate the seasonal cycle, so any structure they cannot represent stays
in the anomaly.

For sub-daily data use one of the other three methods. They resolve their climatologies on
the sub-daily cycle, so the diurnal cycle is removed from the anomaly. See
:doc:`dimensions_and_time`.

Parameters by Method
====================

.. list-table::
   :header-rows: 1
   :widths: 26 14 14 14 14 18

   * - Parameter
     - shifting
     - fixed
     - detrend fixed
     - harmonic
     - Default
   * - ``window_years``
     - yes
     - no
     - no
     - no
     - 15
   * - ``smooth_days``
     - yes
     - yes
     - yes
     - ignored
     - 21
   * - ``smooth_max_missing``
     - yes
     - no
     - no
     - no
     - 0.1
   * - ``reference_period``
     - error
     - yes
     - yes
     - error
     - ``None``
   * - ``detrend_orders``
     - no
     - no
     - yes
     - yes
     - ``[1]``
   * - ``force_zero_mean``
     - no
     - no
     - yes
     - yes
     - ``True``
   * - ``standardise``
     - error
     - error
     - error
     - yes
     - ``False``

"Error" means the call raises a ``ConfigurationError``.

Smoothing
=========

``smooth_days`` is a duration in days, converted to whole timesteps for the cadence of the
data. ``smooth_days=1`` turns smoothing off.

* ``shifting_baseline`` smooths the raw field with a centred rolling mean of ``smooth_days``
  before taking the rolling climatology. On sub-daily data each time of day is averaged
  across days. A smoothed value is NaN once more than ``smooth_max_missing`` (default 0.1)
  of its window is missing (NaN, or a step absent from a sub-daily axis), and windows that
  reach past either end of the record are NaN. ``smooth_max_missing=0`` requires complete
  windows.
* ``fixed_baseline`` and ``detrend_fixed_baseline`` smooth the day-of-year climatology with
  a circular moving average, so 31 December is averaged with 1 January, as in Hobday et al.
  (2016). The window must be shorter than the whole cycle. On sub-daily cycles the average
  is taken across days at a fixed time of day.
* ``detrend_harmonic`` ignores ``smooth_days``.

On a monthly axis the default 21 days is less than one step, so no smoothing is applied and
a warning says so.

Detrending
==========

``detrend_orders`` lists the polynomial orders to remove, each 1 or higher. The default
``[1]`` removes a linear trend, and ``[1, 2]`` adds a quadratic. With
``force_zero_mean=True`` the detrended field is shifted to zero mean.
``detrend_fixed_baseline`` then subtracts its reference climatology, so its final anomaly
can have a non-zero mean.

.. code-block:: python

   anomalies = marEx.anomaly.compute(
       sst,
       method="detrend_fixed_baseline",
       detrend_orders=[1, 2],
       reference_period=(1991, 2020),
   )

Here ``reference_period`` sets only the climatology. The trend is fitted to, and the
anomaly returned for, the whole series.

Standardising
=============

``standardise=True`` (``detrend_harmonic`` only) divides the anomaly by a local standard
deviation (the square root of a circular 30-day rolling mean of the per-day-of-year
variance) and returns it as ``dat_stn``, with the divisor as ``STD``. Through
``preprocess_data`` the extremes module then runs a second time on ``dat_stn`` and adds
``extreme_events_stn`` and ``thresholds_stn``. Use it when variability differs strongly
between regions or seasons, to define events in units of local variability.

Trimming and Leap Days
======================

``shifting_baseline`` removes the first ``window_years`` years from the output, so a
20-year input with ``window_years=15`` gives 5 years of anomalies.

In all three baseline methods the day-of-year 366 climatology pools every year's 31
December (day 365 in a non-leap year). A leap year's 31 December therefore has a
climatology even when the window or reference period contains no other leap year.

Cadence
=======

Climatologies are resolved on a within-year cycle inferred from the time axis:
``dayofyear`` for daily data, ``month`` for monthly data and ``hourofyear`` for sub-daily
data. Override it with ``cycle=``. :doc:`dimensions_and_time` covers cadences in full,
including which methods run at each.

Checking the Output
===================

The mask comes from the first timestep only, so land must be NaN there
(:doc:`dimensions_and_time`). Check the result:

.. code-block:: python

   print(float(anomalies.dat_anomaly.std()))          # plausible spread?
   print(float(anomalies.mask.mean()) * 100)          # valid fraction, per cent
   anomalies.dat_anomaly.mean(["lat", "lon"]).plot()  # trend left over?

Next, :doc:`extremes` turns anomalies into thresholds and event flags.
