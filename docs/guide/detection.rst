=====================
Detection & Anomalies
=====================

See :doc:`../api/anomaly` and :doc:`../api/extremes` for the full function reference.

Overview
========

The detection module implements a comprehensive workflow for converting raw oceanographic
time series data into standardised anomalies and binary extreme event masks. It supports
both structured (regular lat/lon grids) and unstructured (irregular mesh) data formats
with advanced statistical methods for robust extreme event detection.

**Key Features:**

* **Dual Anomaly Methods**: Detrended baseline vs. shifting baseline approaches
* **Flexible Extreme Detection**: Global percentile vs. day-of-year specific thresholds
* **Dask Integration**: Memory-efficient processing of large datasets with parallel computation
* **Grid Agnostic**: Works seamlessly with both structured and unstructured grids
* **Statistical Rigor**: Advanced statistical methods for robust anomaly calculation

Basic Usage Examples
====================

Simple Preprocessing
--------------------

.. code-block:: python

   import xarray as xr
   import marEx

   # Load sea surface temperature data
   sst = xr.open_dataset('sst_daily.nc', chunks={'time': 365}).sst

   # Basic preprocessing with default parameters
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95
   )

   # Result contains anomalies and extreme events
   print(extremes_ds)

Advanced Preprocessing
----------------------

.. code-block:: python

   # Advanced preprocessing with custom parameters
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=90,
       method_anomaly='shifting_baseline',      # Use rolling climatology
       method_extreme='seasonal_percentile',         # Day-of-year specific thresholds
       window_years=20,                 # 20-year rolling baseline
       smooth_days=31,                 # 31-day smoothing
       window_days=11,                   # 11-day window for thresholds
       window_spatial=5,                 # 5-cell spatial clustering
       dask_chunks={'time': 25}
   )

Unstructured Grid Processing
----------------------------

.. code-block:: python

   # For unstructured grids, specify dimensions
   dimensions = {'time': 'time', 'x': 'ncells'}
   coordinates = {'time': 'time', 'x': 'lon', 'y': 'lat'}

   extremes_ds = marEx.preprocess_data(
       sst_unstructured,
       threshold_percentile=95,
       dimensions=dimensions,
       coordinates=coordinates
   )

Output Data Structure
=====================

The preprocessing function returns an xarray Dataset with the following structure:

.. code-block:: python

   # extremes_ds Dataset structure:
   xarray.Dataset
   Dimensions:     (lat, lon, time, dayofyear)
   Coordinates:
       lat         (lat)
       lon         (lon)
       time        (time)
       dayofyear   (dayofyear)    # Only for seasonal_percentile method
   Data variables:
       dat_anomaly     (time, lat, lon)        float64     # Anomaly field
       mask            (lat, lon)              bool        # Land-sea mask
       extreme_events  (time, lat, lon)        bool        # Binary extreme events
       thresholds      (dayofyear, lat, lon)   float64     # Thresholds

**Key Variables:**

* **dat_anomaly**: Anomaly field (either detrended or from rolling climatology)
* **mask**: Deduced land-sea mask (True = ocean, False = land)
* **extreme_events**: Binary field locating extreme events (True = extreme)
* **thresholds**: Thresholds

Choosing an Anomaly Method
==================================

Use this decision tree to select the most appropriate anomaly calculation method for your research:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────┐
   │ Anomaly Method Selection Decision Tree                              │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                     │
   │ Need full time series? ──No──> SHIFTING BASELINE                    │
   │         │                       (most accurate, shortens data by    │
   │        Yes                       window_years years)        │
   │         │                                                           │
   │         ├─> Remove trends? ──No──> FIXED BASELINE                   │
   │         │         │                 (keeps trends in anomaly)       │
   │         │        Yes                                                │
   │         │         │                                                 │
   │         │         └──> Need efficiency? ──Yes──> DETREND HARMONIC   │
   │         │                      │                  (fast, biased)    │
   │         │                     No                                    │
   │         │                      │                                    │
   │         │                      └──> DETREND FIXED BASELINE          │
   │         │                           (accurate detrending)           │
   └─────────────────────────────────────────────────────────────────────┘

Method Comparison
=================

Anomaly Detection Methods
-------------------------

**Harmonic Detrending** (``method_anomaly='detrend_harmonic'``):

.. code-block:: python

   # Fast polynomial detrending with harmonic components
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='detrend_harmonic',
       threshold_percentile=95,
       # Additional parameters:
       standardise=False,          # Optional STD normalisation
       detrend_orders=[1],           # Linear detrending (default)
       force_zero_mean=True          # Enforce zero mean
   )

**Characteristics:**
  * Faster computation using polynomial fitting
  * Uses harmonic components (annual, semi-annual cycles)
  * May introduce biases in variability statistics
  * Best for: Quick analysis

**Fixed Baseline** (``method_anomaly='fixed_baseline'``):

.. code-block:: python

   # Daily climatology (calculated using the full time series) without detrending
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='fixed_baseline',
       threshold_percentile=95,
       # Additional parameters:
       smooth_days=11       # Smoothing window for climatology
   )

   # Or where the climatology is calculated for a specific reference period (e.g., 1990-2020)
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='fixed_baseline',
       threshold_percentile=95,
       reference_period=(1990, 2020)  # Climatology from 1990-2020 only
   )

**Characteristics:**
  * Anomaly relative to the daily climatology using full time series (or a specified ``reference_period``)
  * Preserves long-term / climate trends
  * Simple interpretation and fast computation
  * Best for: Baseline comparison studies, trend-inclusive analysis, public outreach

**Detrend Fixed Baseline** (``method_anomaly='detrend_fixed_baseline'``):

.. code-block:: python

   # Polynomial detrending followed by fixed daily climatology
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='detrend_fixed_baseline',
       threshold_percentile=95,
       # Additional parameters:
       detrend_orders=[1],           # Linear detrending (default)
       smooth_days=11,      # Smoothing window for climatology
       force_zero_mean=True,         # Enforce zero mean
       reference_period=(1990, 2020) # Optional: restrict climatology to a specific reference period
   )

**Characteristics:**
  * Polynomial detrending followed by removing the fixed daily climatology
  * Preserves the full time-series of data, but does not account for trends in the timing of seasonal transitions
  * Removes long-term trends
  * Best for: Climate variability studies with trend removal

**Shifting Baseline** (``method_anomaly='shifting_baseline'``):

.. code-block:: python

   # Rolling climatology from previous years
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='shifting_baseline',
       threshold_percentile=95,
       window_years=15,      # 15-year rolling baseline
       smooth_days=21       # 21-day smoothing window
   )

**Characteristics:**
  * More accurate climatology using rolling window
  * Shortens time series by baseline window length
  * Computationally intensive but scientifically rigorous
  * Best for: Research applications, intricate & accurate analysis

Extreme Event Detection Methods
-------------------------------

**Global Extreme** (``method_extreme='global_percentile'``):

.. code-block:: python

   # Single threshold across all time points
   extremes_ds = marEx.preprocess_data(
       sst,
       method_extreme='global_percentile',
       threshold_percentile=95,
       # Optional STD normalisation
       standardise=True
   )

**Characteristics:**
  * Uses percentiles from entire time series
  * Single threshold value for all time points
  * Simple interpretation and fast computation
  * Best for: Exploratory analysis

**Seasonal Percentile** (``method_extreme='seasonal_percentile'``):

.. code-block:: python

   # Day-of-year specific thresholds
   extremes_ds = marEx.preprocess_data(
       sst,
       method_extreme='seasonal_percentile',
       threshold_percentile=95,
       window_days=11,        # 11-day window
       window_spatial=None    # No spatial clustering (default)
   )

**Characteristics:**
  * Day-of-year specific percentile thresholds
  * Accounts for seasonal variations
  * Follows Hobday et al. (2016) methodology

Spatial Window Enhancement (``window_spatial``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**New in v3.0+**: The Hobday extreme method supports optional spatial pooling window for more robust, spatially coherent thresholds.

**Algorithm Details**:

For each grid cell ``(i, j)`` and each day-of-year ``d``:

1. **Temporal Sampling**: Collect anomalies from all years within ±``window_days`` days around day ``d``

   * Traditional: Samples from single cell ``(i, j)``
   * With spatial window: Samples from all cells in neighbourhood

2. **Spatial Pooling** (if ``window_spatial`` specified):

   * Define spatial window centered at ``(i, j)`` with radius ``r = (window_spatial - 1) / 2``
   * Pool samples from cells ``(i-r:i+r+1, j-r:j+r+1)``
   * Edge handling: Smaller windows near boundaries

3. **Percentile Calculation**:

   * Use histogram approximation (``method_percentile='approximate'``)
   * Bins are symmetric about zero, so both tails are resolved at the same precision
   * Bin width controlled by ``precision``; derived from the data when not supplied
   * Calculate threshold at ``threshold_percentile`` (e.g., 95th)

**Sample Size Comparison**::

   Configuration                         Sample Count
   ──────────────────────────────────   ─────────────
   Traditional (no spatial window)       N_years × window_days
   Example: 30 years × 11 days          = 330 samples

   With 5×5 spatial window               N_years × window_days × 25 cells
   Example: 30 years × 11 days × 25     = 8,250 samples

   With 9×9 spatial window               N_years × window_days × 81 cells
   Example: 30 years × 11 days × 81     = 26,730 samples

**Example: Enabling Spatial Window**:

.. code-block:: python

   # Very long time-series (50+ years): no spatial pooling needed
   extremes_highres = marEx.preprocess_data(
       sst_0.1deg,
       method_extreme='seasonal_percentile',
       window_days=11,
       window_spatial=None
   )

   # Short time-series: use spatial pooling to increase robustness of threshold calculation
   extremes_coarse = marEx.preprocess_data(
       sst_2deg,
       method_extreme='seasonal_percentile',
       window_days=11,
       window_spatial=5         # Increase samples in anomaly distribution using a 5×5 window
   )

   # High threshold percentile (>95%): use spatial pooling to robustly sample distribution tails
   extremes_99th = marEx.preprocess_data(
       sst,
       method_extreme='seasonal_percentile',
       threshold_percentile=99,        # Extreme percentiles require more samples
       window_days=11,
       window_spatial=7         # Larger window to sample tails (7×7 = 49 cells)
   )


**Performance Implications**:

* **Memory**: Approximate method remains memory-efficient regardless of window size
* **Computation**: Larger windows → more samples → slightly slower but still fast
* **Typical**: 5×5 window adds ~10-15% to computation time vs. no spatial window

**When to Use Spatial Windowing**:

Spatial windowing increases the sample size for percentile calculation, improving statistical robustness.
Note that this is not a spatial smoothing of the data itself, but rather a pooling of samples from
neighbouring grid cells to better estimate the percentile thresholds. This is motivated from a spatial decorrelation
length-scale argument, in the same way Hobday has argued for decorrelation time-scale for the 11-day time window.
This is critical in several scenarios:

**1. Short Time Series** (insufficient samples):

When time series length is limited, sample size for each day-of-year may be inadequate for robust percentile estimation:

.. code-block:: text

   Time Series Length    Samples (no spatial)    Extreme samples    Recommendation
   ──────────────────    ───────────────────     ───────────────    ──────────────
   10 years              110 samples             5 samples          ✓ Use spatial
   20 years              220 samples             11 samples         ✓ Use spatial
   30 years              330 samples             16 samples         ○ Optional
   50+ years             550+ samples            27+ samples        ✗ Not needed

*Guideline*: Use spatial windowing if time series < 30 years for robust 95th percentile estimation.

**2. Extreme Percentiles** (>95%, sampling distribution tails):

Higher percentiles require more samples to characterise the tail of the distribution accurately.
Without sufficient samples, extreme percentile thresholds become unreliable (Smith et al. 2025,
https://doi.org/10.1016/j.pocean.2024.103404).

.. code-block:: text

   Percentile    Min Samples Needed    30-years (no spatial)    30-year with 5×5    Recommendation
   ──────────    ──────────────────    ─────────────────────    ────────────────    ──────────────
   90th          100-200               330 ✓                    8,250 ✓             Either works
   95th          200-400               330 ○                    8,250 ✓             Spatial helps
   97.5th        400-800               330 ✗                    8,250 ✓             ✓ Use spatial
   99th          1000+                 330 ✗                    8,250 ✓             ✓ Use spatial
   99.9th        10000+                330 ✗                    8,250 ✗             ✓ Need 9×9+

*Guideline*: For percentiles >95th, spatial windowing is strongly recommended. For >97.5th, it is essential.

Parameter Reference
===================

Core Parameters
---------------

**threshold_percentile** : float, default=95
  Percentile threshold for extreme event identification (e.g., 95 for 95th percentile).
  Combined with ``tail``, this is the percentile of the *distribution*, not of the tail:
  the coldest 5 % is ``threshold_percentile=5, tail='lower'``.

**tail** : {'upper', 'lower'}, default='upper'
  Which side of the distribution counts as extreme. ``'upper'`` flags
  ``anomaly >= threshold``; ``'lower'`` flags ``anomaly <= threshold``.

**method_anomaly** : {'detrend_harmonic', 'fixed_baseline', 'detrend_fixed_baseline', 'shifting_baseline'}, default='shifting_baseline'
  Method for anomaly computation

**method_extreme** : {'global_percentile', 'seasonal_percentile'}, default='seasonal_percentile'
  Method for extreme identification

**dask_chunks** : dict, default={'time': 25}
  Chunk sizes for Dask arrays for memory management

Anomaly Method Parameters
-------------------------

**Harmonic Detrending Parameters:**

**standardise** : bool, default=False
  Whether to normalise anomalies using 30-day rolling standard deviation

**detrend_orders** : list of int, default=[1]
  Polynomial orders for detrending (e.g., [1, 2] for linear + quadratic)

**force_zero_mean** : bool, default=True
  Whether to explicitly enforce zero mean in final anomalies

**Fixed Baseline Parameters:**

**smooth_days** : int, default=11
  Number of days for smoothing the daily climatology

**reference_period** : tuple of (int, int), optional
  Year range ``(start_year, end_year)`` inclusive for computing the daily climatology.
  If ``None`` (default), uses all available years. Anomalies are computed for the full
  time series regardless. Example: ``reference_period=(1990, 2020)``

**Detrend Fixed Baseline Parameters:**

**detrend_orders** : list of int, default=[1]
  Polynomial orders for detrending (e.g., [1, 2] for linear + quadratic)

**smooth_days** : int, default=11
  Number of days for smoothing the daily climatology after detrending

**force_zero_mean** : bool, default=True
  Whether to explicitly enforce zero mean in final anomalies

**reference_period** : tuple of (int, int), optional
  Year range ``(start_year, end_year)`` inclusive for computing the daily climatology
  (only affects the climatology step; detrending still uses all data).
  If ``None`` (default), uses all available years.

**Shifting Baseline Parameters:**

**window_years** : int, default=15
  Number of years for rolling climatology baseline

**smooth_days** : int, default=21
  Number of **days** for smoothing the rolling climatology baseline. Converted to
  timesteps for non-daily data; on a monthly axis it clamps to a single step and the
  smoothing becomes a no-op, which is logged.

Extreme Detection Parameters
----------------------------

**Seasonal Percentile Parameters:**

**window_days** : int, default=11
  Window size, **in days**, for the seasonal threshold calculation. Converted to
  timesteps for non-daily data — see `Time Resolution`_ above. Must be odd on a daily
  axis (the window is symmetric about its centre step).

**window_spatial** : int, optional
  Spatial window size for clustering (None = no spatial clustering)

**method_percentile** : {'exact', 'approximate'}, default='approximate'
  Method for percentile calculation

**precision** : float, optional
  Histogram bin width for the approximate percentile calculation. Derived from
  ``max_anomaly`` and ``n_bins`` when omitted.

**max_anomaly** : float, optional
  Half-width of the binned range, in the units of your data. Derived from the data
  itself when omitted.

**n_bins** : int, default=1000
  Number of histogram bins spanning ``[-max_anomaly, +max_anomaly]``. Used to derive
  whichever of ``precision`` and ``max_anomaly`` was not supplied.

Which Tail
----------

By default marEx looks for extremes **above** a high percentile: marine heatwaves,
atmospheric heatwaves, extreme rainfall. ``tail='lower'`` flips the comparison for
low-side extremes -- cold spells, drought, hypoxia.

.. code-block:: python

   # The coldest 5 % of days, day-of-year resolved
   cold = marEx.preprocess_data(
       sst,
       threshold_percentile=5,
       tail="lower",
   )

``threshold_percentile`` always names the percentile of the distribution, never of the
tail, so ``5`` with ``tail='lower'`` and ``95`` with ``tail='upper'`` are the two ends
of the same distribution. The threshold field means the same thing in both cases; only
the comparison changes, and the tail chosen is recorded in ``ds.attrs['tail']``.

Two consequences worth knowing:

* The histogram bins are **symmetric about zero**. Before this they were
  ``[-inf, -precision, 0, precision, ...]`` -- one bin for every negative value -- which
  made a low percentile unresolvable and is why percentiles below 60 used to be
  rejected outright. That restriction is gone.
* The guard rail that keeps a constant-zero anomaly (sea ice, a permanently masked
  cell) from being flagged as extreme is applied on **both** sides. A flat-zero cell is
  never a cold extreme either.

``tail='both'`` is not supported: it would need a second threshold array and an extra
output dimension. Run the two tails separately if you need both.

Bin Geometry and Non-SST Variables
----------------------------------

``precision=0.01`` and ``max_anomaly=5.0`` are calibrated for **SST anomalies in
kelvin**. On precipitation in mm/day, with anomalies of tens, that range clips almost
everything into the end bins; on pressure in Pa it is off by three orders of magnitude.

So when neither is supplied, marEx derives the range from your data -- one fused
min/max pass, then ``max(|min|, |max|)`` -- and sets ``precision = 2 * max_anomaly /
n_bins``. Both resolved values are logged at INFO and recorded in the output attributes.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - You supply
     - What marEx uses
   * - nothing
     - ``max_anomaly`` from the data; ``precision = 2 * max_anomaly / n_bins``
   * - ``precision`` only
     - ``max_anomaly = precision * n_bins / 2`` (so ``precision=0.01`` still spans ±5.0)
   * - ``max_anomaly`` only
     - ``precision = 2 * max_anomaly / n_bins``
   * - both
     - exactly what you gave; ``n_bins`` is ignored

.. note::

   The derivation is skipped entirely for ``method_percentile='exact'``, which builds
   no histogram. It costs one pass over the anomaly, which is cheap in the default
   ``persist`` mode (the anomaly is already staged) but walks the whole anomaly graph
   under ``compute_mode='lazy'``. Pin ``max_anomaly`` there if that matters.

Time Resolution
---------------

marEx infers the **within-year cycle** its climatologies and seasonal thresholds are
resolved on from the median spacing of your time coordinate. Daily data is the common
case and behaves exactly as it always has; monthly and sub-daily series are supported
on the same code paths.

.. list-table::
   :header-rows: 1
   :widths: 22 22 18 38

   * - Median spacing
     - Cycle dimension
     - Slots
     - Notes
   * - ≥ 28 days
     - ``month``
     - 12
     - Monthly means, model output on a monthly axis
   * - ≥ 1 day
     - ``dayofyear``
     - 366
     - The default; unchanged from earlier releases
   * - < 1 day
     - ``hourofyear``
     - ``366 × steps_per_day``
     - Hourly, 6-hourly, sub-daily reanalysis

The cycle only matters for ``method_extreme='seasonal_percentile'`` and for the
climatology-based anomaly methods. ``global_percentile`` has no within-year cycle at
all and works on any time axis.

**Durations are physical, not step counts.** ``window_days`` and ``smooth_days`` are
always expressed in *days*, whatever the cadence, and are converted to timesteps
internally:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Cadence
     - ``window_days=11`` becomes
     - Effect
   * - Daily
     - 11 steps
     - Unchanged
   * - 6-hourly
     - 45 steps
     - The same 11 days of data
   * - Monthly
     - 1 step (clamped)
     - **Warns**: this month only

That last row is worth reading twice. An 11-day window cannot be represented on a
monthly axis, so marEx uses the closest thing it can — a single month — and emits a
warning naming both the requested and the realised duration. A user who asked for an
11-day window and silently received "this month only" has been given a different
method than the one they requested.

.. note::

   ``method_anomaly='detrend_harmonic'`` **rejects sub-daily input.** Its basis
   removes annual and semi-annual cycles only, so on an hourly axis the entire diurnal
   cycle would survive into the anomaly — a silently wrong result rather than a slow
   one. Use ``shifting_baseline``, ``fixed_baseline``, or ``detrend_fixed_baseline``
   instead: their climatologies are resolved on the sub-daily cycle and remove the
   diurnal cycle as a matter of course.

**Overriding the inference.** An irregular time axis — one whose spacings have no
single characteristic cadence, such as a daily series concatenated onto a monthly one
— raises a :class:`~marEx.ConfigurationError` rather than guessing. Pass an explicit
cycle to override:

.. code-block:: python

   import marEx

   ds = marEx.preprocess_data(
       data,
       method_anomaly='fixed_baseline',
       method_extreme='seasonal_percentile',
       cycle=marEx.SeasonalCycle('month', 12, 30.44),
   )

``cycle=`` is accepted by :func:`marEx.preprocess_data`,
:func:`marEx.anomaly.compute` and :func:`marEx.extremes.identify`; the chainer passes
it to both stages so they are always resolved on the same axis. To see what marEx
would infer without running anything:

.. code-block:: python

   >>> marEx.infer_cycle(data.time)
   SeasonalCycle(index_name='hourofyear', length=1464, step_days=0.25)

.. warning::

   Sub-daily runs are **supported but expensive**. The threshold histogram is
   ``cycle_length × n_bins`` per spatial cell, so 6-hourly data makes it four times
   the daily size and hourly data twenty-four times. The internal spatial tiling
   shrinks each task in proportion, which keeps the working set bounded but multiplies
   the task count. For long sub-daily series, prefer ``global_percentile``, or coarsen
   to daily first.

Grid Configuration
------------------

**dimensions** : dict
  Dimension mapping for different grid types:

  * Structured: ``{'time': 'time', 'x': 'lon', 'y': 'lat'}``
  * Unstructured: ``{'time': 'time', 'x': 'ncells'}``

**coordinates** : dict
  Coordinate mapping for different grid types:

  * Structured: ``{'time': 'time', 'x': 'lon', 'y': 'lat'}``
  * Unstructured: ``{'time': 'time', 'x': 'lon', 'y': 'lat'}``

**neighbours** : xarray.DataArray, optional
  Neighbour connectivity array for spatial clustering (unstructured grids)

**cell_areas** : xarray.DataArray, optional
  Cell areas for weighted spatial statistics (unstructured grids)

Advanced Usage Examples
=======================

Method Combinations
-------------------

.. code-block:: python

   # Most rigorous combination (computationally intensive)
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='shifting_baseline',
       method_extreme='seasonal_percentile',
       threshold_percentile=90,
       window_years=20,
       smooth_days=31,
       window_days=11
   )

   # Fastest combination (less rigorous)
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='detrend_harmonic',
       method_extreme='global_percentile',
       threshold_percentile=95,
       standardise=False
   )

Performance Optimisations
-------------------------

.. code-block:: python

   # Optimised chunking for large datasets
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       dask_chunks={'time': 25, 'lat': 200, 'lon': 200},
       # Use approximate percentiles for speed
       method_percentile='approximate',
       precision=0.05,  # Coarser precision for speed
       max_anomaly=10.0
   )

Multi-Variable Processing
-------------------------

.. code-block:: python

   # Process multiple variables
   variables = ['sst', 'sss', 'chlorophyll']
   extreme_datasets = {}

   for var_name in variables:
       data = xr.open_dataset(f'{var_name}_daily.nc')[var_name]

       extreme_datasets[var_name] = marEx.preprocess_data(
           data,
           threshold_percentile=95,
           method_anomaly='shifting_baseline',
           method_extreme='seasonal_percentile'
       )

Integration with Tracking
=========================

Complete Workflow
-----------------

.. code-block:: python

   import xarray as xr
   import marEx

   # Step 1: Load data
   sst = xr.open_dataset('sst_daily.nc', chunks={'time': 365}).sst

   # Step 2: Preprocess extremes
   extremes_ds = marEx.preprocess_data(
       sst,
       method_anomaly='shifting_baseline',
       method_extreme='seasonal_percentile',
       threshold_percentile=95,
       window_years=15,
       smooth_days=21,
       window_days=11,
       dask_chunks={'time': 25}
   )

   # Step 3: Track events
   event_tracker = marEx.tracker(
       extremes_ds.extreme_events,
       extremes_ds.mask,
       R_fill=8,
       area_filter_quartile=0.5
   )

   tracked_events = event_tracker.run()

   # Step 4: Visualise results
   config = marEx.PlotConfig(
       title='Extreme Events',
       plot_IDs=True
   )

   fig, ax, im = tracked_events.ID_field.isel(time=0).plotX.single_plot(config)

Quality Control
===============

Data Validation
---------------

.. code-block:: python

   # Check preprocessing results
   extremes_ds = marEx.preprocess_data(sst, threshold_percentile=95)

   # Validate anomaly statistics
   anomaly_stats = extremes_ds.dat_anomaly.std()
   print(f"Anomaly standard deviation: {anomaly_stats.values:.3f}")

   # Check extreme event frequency
   event_frequency = extremes_ds.extreme_events.mean() * 100
   print(f"Extreme event frequency: {event_frequency.values:.1f}%")

   # Validate mask coverage
   ocean_fraction = extremes_ds.mask.mean() * 100
   print(f"Ocean coverage: {ocean_fraction.values:.1f}%")

Threshold Validation
--------------------

.. code-block:: python

   # For seasonal_percentile method, examine thresholds
   if 'thresholds' in extremes_ds:
       # Check seasonal threshold variation
       seasonal_range = (extremes_ds.thresholds.max(dim='dayofyear') -
                        extremes_ds.thresholds.min(dim='dayofyear'))
       print(f"Seasonal threshold range: {seasonal_range.mean().values:.3f}")

       # Plot threshold climatology
       threshold_clim = extremes_ds.thresholds.mean(dim=['lat', 'lon'])
       import matplotlib.pyplot as plt
       plt.plot(threshold_clim.dayofyear, threshold_clim.values)
       plt.xlabel('Day of Year')
       plt.ylabel('Threshold Value')
       plt.title('Seasonal Threshold Climatology')

Error Handling
==============

Common Issues and Solutions
---------------------------

**Memory Errors**:

.. code-block:: python

   # Solution: Optimise chunking and use approximate methods
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       dask_chunks={'time': 15},     # Smaller chunks
       method_percentile='approximate',
       precision=0.1                 # Coarser precision
   )

**Performance Issues**:

.. code-block:: python

   # Solution: Use faster methods for exploration
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       method_anomaly='detrend_harmonic',
       method_extreme='global_percentile',
       standardise=False
   )

**Threshold Calculation Issues**:

.. code-block:: python

   # Solution: Adjust window sizes and use spatial clustering
   extremes_ds = marEx.preprocess_data(
       sst,
       method_extreme='seasonal_percentile',
       window_days=21,        # Larger window
       window_spatial=5,      # Add spatial clustering
       precision=0.01                # Higher precision
   )

**Coordinate System Issues**:

.. code-block:: python

   # Solution: Specify custom dimensions and coordinates
   extremes_ds = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       dimensions={'time': 'time', 'x': 'longitude', 'y': 'latitude'},
       coordinates={'time': 'time', 'x': 'longitude', 'y': 'latitude'}
   )

Performance Benchmarks
======================

Method Performance Comparison
-----------------------------

.. code-block:: python

   # Relative performance for global 0.25° daily data:

   # Fastest: detrend_harmonic + global_percentile
   # - Processing time: ~0.5 wall-minutes per decade (2 CPU-hours)
   # - Memory usage: ~1 GB
   # - Accuracy: Good for first analysis

   # Balanced: detrend_fixed_baseline + seasonal_percentile
   # - Processing time: ~5 wall-minutes per decade (21 CPU-hours)
   # - Memory usage: ~8 GB
   # - Accuracy: Better climatology

   # Most rigorous: shifting_baseline + seasonal_percentile
   # - Processing time: ~8 wall-minutes per decade (34 CPU-hours)
   # - Memory usage: ~12 GB
   # - Accuracy: Best for research applications

Scaling Characteristics
-----------------------

.. code-block:: python

   # Scaling with dask (64 cores, 25-day chunks):
   # - Linear scaling up to ~100 cores
   # - Memory usage: ~2-4 GB per core
   # - I/O becomes bottleneck beyond 200 cores
   # - Optimal chunk size depends on data resolution

Best Practices
==============

Method Selection Guidelines
---------------------------

1. **Quick Exploratory Analysis**: Use ``detrend_harmonic`` + ``global_percentile``
2. **Climate Change Research Studies**: Use ``shifting_baseline`` + ``seasonal_percentile``
3. **Limited Timeseries**: Use ``detrend_fixed_baseline`` + ``seasonal_percentile``

Chunking Guidelines
-------------------

.. code-block:: python

   # Optimal chunking strategies:

   # For global 0.25° data:
   optimal_chunks = {'time': 25, 'lat': 200, 'lon': 200}

   # For regional high-resolution data:
   optimal_chunks = {'time': 50, 'lat': 100, 'lon': 100}

   # For unstructured grids:
   optimal_chunks = {'time': 25, 'ncells': 50000}

Spatial windowing — visual guide
================================

**How Spatial Windowing Works**::

   Traditional Hobday:                  With Spatial Window (5×5):
   Single cell samples                  25 cells × 11 days = 275 samples
   (lat, lon) ──> 11 days               ┌───┬───┬───┬───┬───┐
                                        │   │   │   │   │   │
   Only temporal pooling                ├───┼───┼───┼───┼───┤
                                        │   │   │   │   │   │
                                        ├───┼───┼─●─┼───┼───┤ Central cell
                                        │   │   │   │   │   │
                                        ├───┼───┼───┼───┼───┤
                                        │   │   │   │   │   │
                                        └───┴───┴───┴───┴───┘
                                        Spatial + temporal pooling

**Benefits**:

* **Spatially coherent thresholds**: Reduces noise from individual grid cells
* **More robust statistics**: Larger sample size for robust percentile calculation

**Limitations**:

* **Structured grids only**: Not supported for unstructured (irregular) grids
* **Requires approximate method**: Only works with ``method_percentile='approximate'``

Threshold percentiles
=====================

* **90th percentile**: More events, captures moderate extremes
* **95th percentile**: A common choice for temperature extremes, balanced approach
* **99th percentile**: Only most extreme events, rare events focus

Compound Events
===============

Compound Events
---------------

Analyse events that exceed multiple thresholds or variables:

.. code-block:: python

   # Multiple variable analysis
   sst_extremes = marEx.preprocess_data(sst, threshold_percentile=95)
   salinity_extremes = marEx.preprocess_data(salinity, threshold_percentile=5)  # Low salinity

   # Compound events
   compound_events = sst_extremes.extreme_events & salinity_extremes.extreme_events

Research Workflow & Literature Compliance
=========================================

Research Workflow
-----------------

1. **Exploratory Analysis**: Start with basic preprocessing to understand data
2. **Method Comparison**: Test different methods on subset of data
3. **Quality Control**: Validate results thoroughly
4. **Full Processing**: Apply chosen method to complete dataset
5. **Validation**: Compare with known events and literature


Literature Compliance
---------------------

Following the Hobday et al. (2016) day-of-year definition:

.. code-block:: python

   # Standard MHW definition
   mhw_config = {
       'method_anomaly': 'shifting_baseline',
       'method_extreme': 'seasonal_percentile',
       'threshold_percentile': 90,
       'window_days': 11,
       'window_years': 30,
       'smooth_days': 11,
       'window_spatial': 1,  # Hobday et al. (2016) considers only single points
   }
