=========
Anomalies
=========

The anomaly stage subtracts a climatology from the raw field. It is available alone as
:func:`marEx.anomaly.compute`, which has no threshold parameter, and as the first half of
:func:`marEx.preprocess_data`. The function reference is in :doc:`../api/anomaly`.

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

The output Dataset holds ``dat_anomaly`` and ``mask``, plus ``dat_stn`` and ``STD`` when
``standardise=True``. Input is cast to float32.

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
normal.** It is the default. Each year is compared with the years just before it, so slow
warming does not turn the end of a record into one long extreme. It costs the most
computation, loses the first ``window_years`` years of output, and raises a
``DataValidationError`` if the series is not longer than ``window_years``.

**Use** ``fixed_baseline`` **when the baseline itself is part of the question.** A
heatwave definition tied to a fixed reference period (``reference_period=(1991, 2020)``,
say) is the typical case. Because the trend stays in the anomaly, later years exceed a
fixed threshold more often.

**Use** ``detrend_fixed_baseline`` **when you want a fixed climatology but not the trend**,
for example to study variability without the warming signal, or when the record is too short
to spend ``window_years`` of it on a baseline.

**Use** ``detrend_harmonic`` **for fast exploratory runs, or with** ``standardise=True``.
Fitting a handful of harmonics is much cheaper than building a day-of-year climatology.
It approximates the seasonal cycle, so structure those harmonics cannot represent stays in
the anomaly. It rejects sub-daily data with a ``ConfigurationError``. Monthly data is not blocked, but
the method has not been exercised there.

Sub-daily data needs ``fixed_baseline`` or ``detrend_fixed_baseline``, whose climatologies
are resolved on the sub-daily cycle. A test of ``shifting_baseline`` on a 6-hourly fixture
left a diurnal cycle larger than the signal in the anomaly, so it is not recommended there.
See :doc:`dimensions_and_time`.

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

``smooth_days`` is a duration in days and is converted to whole timesteps for the cadence of
your data. ``smooth_days=1`` turns smoothing off.

* ``shifting_baseline`` smooths the raw field with a centred rolling mean of ``smooth_days``
  steps before taking the rolling climatology.
* ``fixed_baseline`` and ``detrend_fixed_baseline`` smooth the day-of-year climatology with
  a circular moving average, so 31 December is averaged with 1 January, as in Hobday et al.
  (2016). A smoothing window as long as the whole cycle raises a ``ConfigurationError``.
  On sub-daily cycles the average runs across days at a fixed time of day.
* ``detrend_harmonic`` ignores ``smooth_days``.

On a monthly axis 21 days is less than one step, so no smoothing is applied and a warning
says so.

Detrending
==========

``detrend_orders`` lists the polynomial orders to remove. The default ``[1]`` removes a
linear trend, and ``[1, 2]`` adds a quadratic. It must be non-empty and contain orders of at
least 1. With ``force_zero_mean=True`` the final anomaly is shifted to zero mean.

.. code-block:: python

   anomalies = marEx.anomaly.compute(
       sst,
       method="detrend_fixed_baseline",
       detrend_orders=[1, 2],
       reference_period=(1991, 2020),
   )

Only the climatology step uses ``reference_period``. The trend is fitted to all data. The
anomaly is returned for the whole series regardless.

Standardising
=============

``standardise=True`` (``detrend_harmonic`` only) divides the anomaly by a local standard
deviation, a 30-day rolling mean of the per-day-of-year standard deviation, and returns it as
``dat_stn`` with the divisor as ``STD``. Through ``preprocess_data`` the extremes stage then
runs a second time on ``dat_stn`` and adds ``extreme_events_stn`` and ``thresholds_stn``.
Use it when variability differs strongly between regions or seasons and you want events
defined in units of local variability rather than in the units of the field.

Trimming and Leap Days
======================

``shifting_baseline`` removes the first ``window_years`` years from the output, so a
20-year input with ``window_years=15`` gives 5 years of anomalies. If the series is
too short, the stage raises a ``DataValidationError`` before computing anything.

Day-of-year 366 needs a leap year in the data that builds the climatology. When a
``shifting_baseline`` window contains no leap year (a small ``window_years`` can do this),
that one timestep has no climatology and its anomaly is NaN. This is expected. A
``fixed_baseline`` reference period without a leap year fills day 366 from day 365 instead.

Cadence
=======

Climatologies are resolved on a within-year cycle that is inferred from the time axis:
``dayofyear`` for daily data, ``month`` for monthly data and ``hourofyear`` for
sub-daily data. The cycle can be overridden with ``cycle=``. Everything about cadences,
including which methods run at which cadence, is on :doc:`dimensions_and_time`.

Checks Worth Making
===================

``validate=True`` (default) raises a ``DataValidationError`` if any unmasked cell is
non-finite at any time. The mask comes from the first timestep only, so land must be NaN
there. A quick look at the result catches most set-up mistakes:

.. code-block:: python

   print(float(anomalies.dat_anomaly.std()))          # plausible spread?
   print(float(anomalies.mask.mean()) * 100)          # valid fraction, per cent
   anomalies.dat_anomaly.mean(["lat", "lon"]).plot()  # trend left over?

Next, :doc:`extremes` turns anomalies into thresholds and event flags.
