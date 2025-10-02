==========
User Guide
==========

This guide covers all aspects of using marEx for marine extreme detection and tracking, from basic concepts to advanced workflows.

.. contents:: Table of Contents
   :local:
   :depth: 3

Introduction
============

marEx is a Python package for efficient identification and tracking of marine extremes (e.g., Marine Heatwaves) in oceanographic data. It provides a complete pipeline from raw data preprocessing to tracked event visualisation.

Key Features
------------

* **Flexible Data Support**: Both structured (lat/lon grids) and unstructured (irregular meshes) data
* **Multiple Methods**: Various anomaly computation and threshold definition methods
* **Parallel Processing**: Built on Dask for efficient computation on large datasets
* **Event Tracking**: Sophisticated algorithms for tracking coherent events through time
* **Rich Visualisation**: Automatic plotting system with publication-ready figures
* **HPC Integration**: Optimised for supercomputing environments

Installation and Setup
======================

Basic Installation
------------------

Install marEx from PyPI::

   pip install marEx

For full functionality including GPU acceleration::

   pip install marEx[full]

For development work::

   pip install marEx[dev]

System Requirements
-------------------

* **Python**: 3.10 or higher
* **Memory**: Minimum 8GB RAM, 32GB+ recommended for large datasets
* **Storage**: Lustre system recommended for large Zarr datasets
* **Optional**: GPU for JAX acceleration

Quick Setup Check
-----------------

.. code-block:: python

   import marEx

   # Check installation
   print(f"marEx version: {marEx.__version__}")

   # Check dependencies
   marEx.print_dependency_status()

   # Start local cluster
   client = marEx.helper.start_local_cluster()

Data Preparation
================

Input Data Requirements
-----------------------

marEx works with xarray DataArrays containing oceanographic time series data:

**Required Dimensions:**

* **Time dimension**: Regular or irregular time steps (daily to monthly)
* **Spatial dimensions**: Either structured (lat, lon) or unstructured (cell/ncells)

**Supported Formats:**

* NetCDF (.nc)
* Zarr (.zarr)
* Any xarray-compatible format

**Data Structure Examples:**

For structured/gridded data::

   data.dims = ('time', 'lat', 'lon')
   # Coordinates: time, lat, lon as dimensions

For unstructured data::

   data.dims = ('time', 'ncells')
   # Coordinates: time as dimension, lat/lon as coordinates

Data Quality Requirements
-------------------------

* **Temporal coverage**: Minimum 10 years for robust climatology
* **Spatial coverage**: Consistent grid throughout time series

Data Loading Examples
---------------------

**Loading NetCDF files:**

.. code-block:: python

   import xarray as xr

   # Single file
   ds = xr.open_dataset('sst_data.nc', chunks={})
   sst = ds.sst

   # Multiple files
   ds = xr.open_mfdataset('sst_*.nc', parallel=True, chunks={})
   sst = ds.sst

**Loading Zarr datasets:**

.. code-block:: python

   # Local Zarr
   ds = xr.open_zarr('data.zarr', chunks={})

   # Cloud-optimised Zarr
   ds = xr.open_zarr('gs://bucket/data.zarr', chunks={})

**Optimising data loading:**

.. code-block:: python

   # Apply chunking for efficient processing
   sst = sst.chunk({'time': 365, 'lat': 50, 'lon': 100})

   # Ensure Dask backing
   if not marEx.is_dask_collection(sst.data):
       sst = sst.chunk()

Core Workflow
=============

The marEx workflow consists of three main steps:

1. **Preprocessing**: Convert raw data to anomalies and identify extremes
2. **Tracking**: Track extreme events through time
3. **Visualisation**: Analyse and visualise results

Step 1: Preprocessing
---------------------

The preprocessing step transforms raw oceanographic data into anomalies and detects extreme event locations.

**Basic preprocessing:**

.. code-block:: python

   import marEx

   # Basic preprocessing with default settings
   extremes = marEx.preprocess_data(
       sst,
       threshold_percentile=95,
       method_anomaly='shifting_baseline',
       method_extreme='hobday_extreme'
   )

**Advanced configuration:**

.. code-block:: python

   # Advanced preprocessing configuration
   extremes = marEx.preprocess_data(
       sst,
       # Anomaly computation method
       method_anomaly='detrend_fixed_baseline',  # or 'detrend_harmonic', 'fixed_baseline', 'shifting_baseline'
       detrend_order=[1 2],
       smooth_days_baseline=21,             # Smoothing for climatology

       # Extreme identification method
       method_extreme='hobday_extreme',     # or 'global_extreme'
       threshold_percentile=95,             # 95th percentile
       window_days_hobday=11,               # Days around each day of year
       window_spatial_hobday=5,             # Spatial window for percentile calculation

       # Output options
       dask_chunks={'time': 25}
   )

The resulting xarray dataset ``extremes`` will have the following structure & entries::

   xarray.Dataset
   Dimensions:     (lat, lon, time)
   Coordinates:
       lat         (lat)
       lon         (lon)
       time        (time)
   Data variables:
       dat_anomaly     (time, lat, lon)        float64     dask.array
       mask            (lat, lon)              bool        dask.array
       extreme_events  (time, lat, lon)        bool        dask.array
       thresholds      (dayofyear, lat, lon)   float64     dask.array
where:

* ``dat_anomaly`` (time, lat, lon): Anomaly data
* ``extreme_events`` (time, lat, lon): Binary field locating extreme events (1=event, 0=background)
* ``thresholds`` (dayofyear, lat, lon): Extreme event thresholds used to determine extreme events
* ``mask`` (lat, lon): Valid data mask

See, e.g. ``./examples/unstructured data/01_preprocess_extremes.ipynb`` for a detailed example of pre-processing on an *unstructured* grid.

Step 2: Event Tracking
----------------------

The tracking step identifies coherent extreme events and follows them through time.

**Basic tracking:**

.. code-block:: python

   tracked_events = marEx.tracker(
       extremes.extreme_events,
       extremes.mask,
       area_filter_absolute=100   # Remove objects smaller than 100 grid cells
       R_fill=8,                  # Radius for filling gaps (in grid cells)
   ).run()

**Advanced tracking options:**

.. code-block:: python

   # Initialise advanced tracker
   tracker = marEx.tracker(
       extremes.extreme_events,
       extremes.mask,

       # Temporal criteria
       T_fill=4,                    # Fill gaps up to 4 days (to maintain continuous events)

       # Spatial criteria
       R_fill=8,                    # Fill small holes with radius up to 8 grid cells
       area_filter_quartile=0.5,    # Remove smallest 50% of events (alternative to area_filter_absolute)
       cell_areas=grid_areas,       # Optional: physical cell areas for structured grids (m²)

       # Merging criteria
       allow_merging=True,          # Allow merging of events (and keep track of merged IDs & events)
       overlap_threshold=0.5,       # 50% overlap for merging (otherwise events keep independent IDs)
       nn_partitioning=True,        # Use nearest-neighbour partitioning when splitting events
   )

   # Run tracking and return merging data
   tracked_events, merge_events = tracker.run(return_merges=True)

The resulting xarray dataset ``tracked_events`` will have the following structure & entries::

   xarray.Dataset
   Dimensions: (lat, lon, time, ID, component, sibling_ID)
   Coordinates:
       lat         (lat)
       lon         (lon)
       time        (time)
       ID          (ID)
   Data variables:
       ID_field              (time, lat, lon)        int32       dask.array
       global_ID             (time, ID)              int32       ndarray
       area                  (time, ID)              float32     ndarray
       centroid              (component, time, ID)   float32     ndarray
       presence              (time, ID)              bool        ndarray
       time_start            (ID)                    datetime64  ndarray
       time_end              (ID)                    datetime64  ndarray
       merge_ledger          (time, ID, sibling_ID)  int32       ndarray
where:

* ``ID_field``: Field containing the IDs of tracked events (0=background)
* ``global_ID``: Unique global ID of each object; ``global_ID.sel(ID=10)`` maps event ID 10 to its original ID at each time
* ``area``: Area of each event as a function of time (in units of cell counts, or physical units if cell_areas/grid_resolution provided)
* ``centroid``: (x, y) centroid coordinates of each event as a function of time
* ``presence``: Presence (boolean) of each event at each time (anywhere in space)
* ``time_start``: Start time of each event
* ``time_end``: End time of each event
* ``merge_ledger``: Sibling IDs for merging events (matching ``ID_field``); ``-1`` indicates no merging event occurred

When running with ``return_merges=True``, the resulting xarray dataset ``merge_events`` will have the following structure & entries::

   xarray.Dataset
   Dimensions: (merge_ID, parent_idx, child_idx)
   Data variables:
       parent_IDs      (merge_ID, parent_idx)  int32       ndarray
       child_IDs       (merge_ID, child_idx)   int32       ndarray
       overlap_areas   (merge_ID, parent_idx)  int32       ndarray
       merge_time      (merge_ID)              datetime64  ndarray
       n_parents       (merge_ID)              int8        ndarray
       n_children      (merge_ID)              int8        ndarray
where:

* ``parent_IDs``: Original parent IDs of each merging event
* ``child_IDs``: Original child IDs of each merging event
* ``overlap_areas``: Area of overlap between parent and child objects in each merging event
* ``merge_time``: Time of each merging event
* ``n_parents``: Number of parent objects in each merging event
* ``n_children``: Number of child objects in each merging event

See, e.g. ``./examples/unstructured data/02_id_track_events.ipynb`` for a detailed example of identification, tracking, & merging on an *unstructured* grid.

Step 3: Visualisation
---------------------

marEx includes a powerful visualisation system called ``plotX`` that automatically detects data types and creates appropriate plots.

**Basic visualisation:**

.. code-block:: python

   # Plot global extreme event frequency
   event_frequency = (tracked_events.ID_field > 0).mean("time")

   # Configure plot appearance
   config = marEx.PlotConfig(
       var_units="MHW Frequency",
       cmap="hot_r",
       cperc=[0, 96],
       grid_labels=True
   )

   # Create single plot
   fig, ax, im = event_frequency.plotX.single_plot(config)

**Advanced visualisation:**

.. code-block:: python

   # Multi-panel visualisation: seasonal extreme event frequency
   seasonal_frequency = (tracked_events.ID_field > 0).groupby("time.season").mean(dim="time")

   # Configure plot appearance
   config = marEx.PlotConfig(
       var_units="MHW Frequency",
       cmap="hot_r",
       cperc=[0, 96],
       grid_labels=True
   )

   # Create multi-panel plot
   fig, ax = seasonal_frequency.plotX.multi_plot(config, col="season", col_wrap=2)

   # Create animation of tracked events
   ID_field_subset = tracked_events.ID_field.sel(time=slice("2020-01-01", "2022-05-31"))
   config = marEx.PlotConfig(plot_IDs=True)
   ID_field_subset.plotX.animate(config, plot_dir=plot_dir, file_name="marine_heatwaves")

   # Plot consecutive time periods
   ID_field_subset = tracked_events.ID_field.sel(time=slice("2021-01-01", "2021-01-06"))
   config = marEx.PlotConfig(plot_IDs=True)
   fig, ax = ID_field_subset.plotX.multi_plot(config, col="time", col_wrap=3)

Method Selection Guide
======================

Choosing the Right Anomaly Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this decision tree to select the most appropriate anomaly calculation method for your research:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────┐
   │ Anomaly Method Selection Decision Tree                              │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                     │
   │ Need full time series? ──No──> SHIFTING BASELINE                    │
   │         │                       (most accurate, shortens data by    │
   │        Yes                       window_year_baseline years)        │
   │         │                                                           │
   │         ├─> Remove trends? ──No──> FIXED BASELINE                   │
   │         │         │                 (keeps trends in anomaly)       │
   │         │        Yes                                                │
   │         │         │                                                 │
   │         │         └──> Need efficiency? ──Yes──> DETREND HARMONIC   │
   │         │                      │                  (fast, may bias)  │
   │         │                     No                                    │
   │         │                      │                                    │
   │         │                      └──> DETREND FIXED BASELINE          │
   │         │                           (accurate detrending)           │
   └─────────────────────────────────────────────────────────────────────┘


Anomaly Methods
---------------

**Harmonic Detrending** (``detrend_harmonic``):

* Detrends with an OLS 6+ coefficient model (mean, annual & semi-annual harmonics, and arbitrary polynomial trends)
* **Best for**: Datasets with linear trends, operational monitoring
* **Pros**: Fast & memory efficient
* **Cons**: Does not capture phenological shifts and non-harmonic seasonal variability. Strongly biases certain statistics.
* **Use when**: Real-time processing

**Fixed Baseline** (``fixed_baseline``):

* Daily climatology using full time series -- does not remove climate trends
* **Best for**: Simple anomaly calculation without detrending
* **Pros**: Straightforward interpretation, preserves long-term trends
* **Cons**: Does not account for climate change trends, seasonal timing shifts
* **Use when**: Baseline comparison studies, trend-inclusive analysis, public outreach

**Detrend Fixed Baseline** (``detrend_fixed_baseline``):

* Polynomial detrending followed by fixed daily climatology -- keeps full time-series of data, but does not account for trends in the timing of seasonal transitions
* **Best for**: Studies requiring detrending but maintaining full temporal data coverage
* **Pros**: Removes long-term trends while preserving seasonal cycles, maintains full time series
* **Cons**: Does not account for changes in seasonal timing or seasonality
* **Use when**: Climate variability studies with trend removal

**Shifting Baseline** (``shifting_baseline``):

* Removes seasonal & long-term trends using a smoothed rolling climatology
* **Best for**: Climate change studies, non-stationary data
* **Pros**: Captures non-linear trends, adapts to changing climate, and seasonal timing variability
* **Cons**: Computationally expensive, shortens effective time series
* **Use when**: Long-term climate studies, accurate & robust analysis

Extreme Identification Methods
------------------------------

**Global Extreme** (``global_extreme``):

* Applies a global-in-time (i.e. constant in time) threshold
   N.B.: This method is a hack designed for ``ocetrac``, which when paired with ``std_normalise=True`` can approximate ``hobday_extreme`` in very specific cases and with a number of caveats. (However, normalising by local STD is again memory-intensive, at which point there is little gained by this approximate method.)
* **Best for**: Simple threshold, constant across seasons
* **Pros**: Simple interpretation, fast computation (without ``std_normalise``)
* **Cons**: Seasonal bias, may miss winter extremes, variability distribution is skewed
* **Use when**: Initial analysis, 0th order comparisons

**Hobday Method** (``hobday_extreme``):

* Defines a local day-of-year specific threshold within a rolling window (equivalent to the Hobday et al. (2016) definition for simple time-series)
* **Best for**: Long-term scientific climate studies, seasonal studies, biological impacts
* **Pros**: Accounts for seasonal variability, literature standard
* **Cons**: Complex threshold interpretation, computationally intensive
* **Use when**: Ecological studies, seasonal analysis, literature comparison

Spatial Window for Hobday Extreme (``window_spatial_hobday``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**New in v3.0+**: The Hobday extreme method now supports optional spatial windowing to create more robust thresholds.

.. code-block:: python

   # Traditional Hobday (temporal window only)
   extremes_traditional = marEx.preprocess_data(
       sst,
       method_extreme='hobday_extreme',
       window_days_hobday=11,          # 11-day temporal window
       window_spatial_hobday=None      # No spatial window
   )

   # With spatial windowing (recommended for short time-series)
   extremes_window = marEx.preprocess_data(
       sst,
       method_extreme='hobday_extreme',
       window_days_hobday=11,          # 11-day temporal window
       window_spatial_hobday=5         # 5×5 spatial window (default)
   )

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
* **More robust statistics**: Larger sample size for percentile calculation

**Limitations**:

* **Structured grids only**: Not supported for unstructured (irregular) grids
* **Requires approximate method**: Only works with ``method_percentile='approximate'``

**When to use**:

* Short time-series of data
* High percentile thresholds defining the extremes
* Noisy SST fields (e.g., satellite with gaps)
* Default (``window_spatial_hobday=5``) works well for most cases

**Example: Comparing threshold noise**:

.. code-block:: python

   # Compare threshold variability
   threshold_std_traditional = extremes_traditional.thresholds.std(dim='dayofyear').mean()
   threshold_std_smooth = extremes_smooth.thresholds.std(dim='dayofyear').mean()

   print(f"Traditional threshold noise: {threshold_std_traditional.compute():.4f}")
   print(f"Windowed threshold noise: {threshold_std_window.compute():.4f}")

Threshold Percentiles
---------------------

* **90th percentile**: More events, captures moderate extremes
* **95th percentile**: Standard for marine heatwaves, balanced approach
* **99th percentile**: Only most extreme events, rare events focus

Scientific Methods: Detailed Concepts and Trade-offs
=====================================================

This section provides an in-depth understanding of the scientific methods and algorithms underlying marEx. Use this to make informed decisions about which approaches best suit your research questions.

Anomaly Detection: Detailed Concepts
-------------------------------------

What are Anomalies?
~~~~~~~~~~~~~~~~~~~

Anomalies represent deviations from "normal" conditions. For marine heatwaves, we calculate temperature anomalies: how much warmer is the ocean compared to its typical state?

.. code-block:: text

   Anomaly = Observed Temperature - Climatological Baseline

The key question: **What baseline should we use?**

Baseline Approaches: Detailed Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MarEx provides four baseline approaches, each with different scientific assumptions:

**1. Shifting Baseline** (``shifting_baseline``)

* **Concept**: Use a rolling window of previous years to define "normal"
* **Baseline**: Smoothed climatology from previous N years (e.g., 15 years)
* **Assumption**: Climate is non-stationary; "normal" changes over time

**Advantages**:

* Most scientifically rigorous for climate change research
* Captures non-linear trends and seasonal timing shifts
* Adapts to changing climate baseline

**Disadvantages**:

* Shortens time series by ``window_year_baseline`` years at the start
* Computationally intensive
* More complex baseline interpretation

**When to use**: Long-term climate studies, publications, ecological research

**2. Fixed Baseline** (``fixed_baseline``)

* **Concept**: Calculate day-of-year climatology from entire time series
* **Baseline**: Average conditions for each calendar day across all years
* **Assumption**: Long-term trends are part of the signal, not noise

**Advantages**:

* Simple interpretation: anomaly relative to long-term average
* Fast computation
* Preserves full time series length
* Highlights long-term warming trends

**Disadvantages**:

* Includes climate change trends in anomalies
* May not represent "extremes" in warming world
* Doesn't account for phenological shifts

**When to use**: Trend-inclusive analysis, public outreach, baseline comparisons

**3. Detrend Fixed Baseline** (``detrend_fixed_baseline``)

* **Concept**: Remove polynomial trends, then apply fixed baseline
* **Baseline**: Daily climatology after detrending
* **Assumption**: Linear/polynomial trends exist but seasonal timing is stationary

**Advantages**:

* Removes long-term trends while preserving full time series
* Maintains seasonal cycles accurately
* Good balance of rigor and simplicity

**Disadvantages**:

* Assumes trends are polynomial (may not capture complex climate change)
* Doesn't account for changing seasonal timing
* Medium computational cost

**When to use**: Climate variability studies, when full time series needed with detrending

**4. Harmonic Detrending** (``detrend_harmonic``)

* **Concept**: Model trends and seasons with harmonics, subtract fitted model
* **Baseline**: Harmonic model (mean + annual/semi-annual + polynomial trends)
* **Assumption**: Seasonality is purely harmonic (sinusoidal)

**Advantages**:

* Very fast computation (OLS regression)
* Full time series preserved
* Memory efficient

**Disadvantages**:

* Harmonic assumption may bias statistics
* Doesn't capture non-sinusoidal seasonal patterns
* May misrepresent phenological complexity

**When to use**: Large-scale screening, operational monitoring, efficiency priority

Decision Framework for Anomaly Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Question 1: Do you need the full time series?
   ├─ NO  → Use shifting_baseline (most accurate)
   └─ YES → Continue to Question 2

   Question 2: Do you need to remove climate trends?
   ├─ NO  → Use fixed_baseline (simplest)
   └─ YES → Continue to Question 3

   Question 3: Is computational efficiency critical?
   ├─ YES → Use detrend_harmonic (fastest)
   └─ NO  → Use detrend_fixed_baseline (balanced)

Extreme Identification: Detailed Concepts
------------------------------------------

Percentile Thresholds
~~~~~~~~~~~~~~~~~~~~~

Once anomalies are calculated, we identify "extreme" values using percentile thresholds:

.. code-block:: text

   Extreme Event = Anomaly > Threshold

   Where: Threshold = Pth percentile of anomaly distribution

Common thresholds:

* **90th percentile**: Moderate extremes, more frequent events
* **95th percentile**: Standard for marine heatwaves (Hobday et al. 2016)
* **99th percentile**: Rare, severe extremes

Global vs Local Thresholds: Detailed Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Global Extreme** (``global_extreme``)

* **Concept**: Single threshold value across all days/seasons
* **Calculation**: Percentile of entire anomaly time series
* **Assumption**: Extremes defined consistently year-round

**Advantages**:

* Simple interpretation
* Fast computation
* Consistent definition across seasons

**Disadvantages**:

* Seasonal bias (summer dominates in many regions)
* May miss winter extremes
* Not aligned with ecological/biological impacts

**When to use**: Exploratory analysis, first-order comparisons, non-seasonal systems

**Hobday Extreme** (``hobday_extreme``)

* **Concept**: Day-of-year specific thresholds accounting for seasonality
* **Calculation**: Percentile within ±N day window around each calendar day
* **Assumption**: Extremes should be defined relative to seasonal expectations

**Advantages**:

* Accounts for seasonal variability
* Literature standard (Hobday et al. 2016)
* Better ecological relevance
* Avoids seasonal bias

**Disadvantages**:

* Complex threshold interpretation (365 different values)
* Computationally intensive
* Requires sufficient data for each day-of-year

**When to use**: Research applications, ecological studies, publication-quality analysis

Spatial Smoothing Concept
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Traditional Hobday**: Calculates thresholds independently for each grid cell

**Spatial Window Enhancement** (``window_spatial_hobday``):

* Pool samples from neighboring cells (e.g., 5×5 window)
* Creates spatially coherent thresholds
* Reduces noise from individual cell variability

.. code-block:: text

   Single Cell:              With 5×5 Spatial Window:

   ┌───┐                     ┌───┬───┬───┬───┬───┐
   │ ● │ 330 samples         │   │   │ X │   │   │
   └───┘                     ├───┼───┼───┼───┼───┤
                             │   │   │ X │   │   │
                             ├───┼───┼─●─┼───┼───┤ 8,250 samples
                             │   │   │ X │   │   │
                             ├───┼───┼───┼───┼───┤
                             │   │   │ X │   │   │
                             └───┴───┴───┴───┴───┘

**When to use spatial smoothing**:

* Coarse resolution grids (> 1°)
* Noisy satellite data with gaps
* When spatial coherence is scientifically important

**When to skip**:

* High resolution grids (< 0.5°)
* Unstructured grids (not supported)
* When local variability is scientifically important

Percentile Calculation: Exact vs Approximate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Exact Method**

* Loads all data into memory
* Calculates mathematically exact percentiles
* **Memory**: High (8-16 GB per core)
* **Speed**: Slower
* **Precision**: Perfect

**Approximate Method**

* Uses histogram binning approximation
* **Memory**: Low (1-2 GB per core)
* **Speed**: Fast
* **Precision**: ~0.01°C (configurable)

**Trade-off Decision**:

.. code-block:: text

   Dataset Size × Memory Available → Choice

   Small dataset + Abundant memory   → Exact (if precision matters)
   Large dataset + Limited memory    → Approximate (sufficient for most research)

   Typical research: Approximate is sufficient (difference ~0.005°C)

Event Tracking: Detailed Concepts
----------------------------------

Connected Component Labeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binary extreme events (True/False) are grouped into spatially connected regions (objects):

.. code-block:: text

   Binary Field:         Labeled Objects:

   1 1 0 0 1 1           A A 0 0 B B
   1 1 0 1 1 1    →      A A 0 B B B
   0 0 0 0 1 1           0 0 0 0 B B

Each connected region gets a unique ID.

Temporal Tracking
~~~~~~~~~~~~~~~~~

Objects are tracked through time by computing overlap between consecutive timesteps:

.. code-block:: text

   Time t:          Time t+1:         Decision:

   Object A         Object A'         If overlap > threshold:
   ███████          ██████               → Same event (keep ID)
                                     Else:
                                        → New event (new ID)

**Overlap Fraction**: ``overlap = intersection / min(area_t, area_t+1)``

Morphological Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

**Opening + Closing** (controlled by ``R_fill``):

* **Opening**: Removes small isolated objects (noise)
* **Closing**: Fills small holes within objects

.. code-block:: text

   Raw Binary → Opening → Closing → Cleaned Objects

   Effect: Smooths boundaries, fills gaps up to R_fill radius

**Temporal Gap Filling** (controlled by ``T_fill``):

Allows tracking to continue across short temporal gaps:

.. code-block:: text

   Day 0: Object detected (ID=5)
   Day 1: Gap (no detection) ← T_fill allows continuity
   Day 2: Gap (no detection)
   Day 3: Object detected → Still ID=5 (if T_fill ≥ 2)

Merge and Split Handling
~~~~~~~~~~~~~~~~~~~~~~~~~

**The Challenge**: Objects can merge and split over time

.. code-block:: text

   Time t:                Time t+1:

   Object A  Object B     Object C (merged)
   ●─────┐   ●─────┐     ●──────────────┐
         │         │     │              │
         └─────────┴─────┴──────────────┘

   Question: How to partition C back to A and B identities?

**Centroid Method** (``nn_partitioning=False``):

* Assign each cell of C to nearest parent centroid
* **Problem**: Small objects can get unrealistically large portions

**Nearest-Neighbor Method** (``nn_partitioning=True``, recommended):

* Assign each cell of C to parent of nearest parent cell
* **Advantage**: Realistic partitioning based on actual proximity

This improves upon Sun et al. (2023) methodology.

Area Filtering Concepts
~~~~~~~~~~~~~~~~~~~~~~~~

Remove small, likely spurious objects:

**Quartile Filtering** (``area_filter_quartile``):

* Remove smallest fraction of objects
* Example: 0.5 → remove smallest 50%
* **Advantage**: Adaptive to dataset

**Absolute Filtering** (``area_filter_absolute``):

* Remove objects smaller than N grid cells
* Example: 100 → keep only objects ≥ 100 cells
* **Advantage**: Consistent physical size threshold

Scientific Considerations
-------------------------

Climatology Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum Data**: 10+ years for robust climatology
**Recommended**: 20-30 years for stable percentiles

**Why**: Percentiles are statistical measures requiring sufficient samples

For Hobday method with window_days_hobday=11:

.. code-block:: text

   Samples per day-of-year = N_years × 11 days

   10 years:  110 samples (minimum)
   30 years:  330 samples (robust)
   50 years:  550 samples (excellent)

Definition Alignment with Literature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hobday et al. (2016)** definition:

* Use 95th percentile threshold
* Day-of-year specific (11-day window)
* Relative to 30-year baseline climatology

**MarEx default alignment**:

* ``threshold_percentile=95``
* ``method_extreme='hobday_extreme'``
* ``window_days_hobday=11``
* ``method_anomaly='shifting_baseline'`` with ``window_year_baseline=15``

For strict Hobday et al. (2016) compliance, use 30-year baseline if available.

Trends vs Variability
~~~~~~~~~~~~~~~~~~~~~~

**Trend-inclusive** (``fixed_baseline``):

* Marine heatwaves defined relative to long-term average
* Increasing frequency reflects both natural variability AND climate change trend
* Interpretation: "How unusual compared to historical average?"

**Trend-removed** (``shifting_baseline``, ``detrend_fixed_baseline``):

* Marine heatwaves defined relative to recent conditions
* Frequency reflects natural variability in current climate state
* Interpretation: "How unusual compared to recent normal?"

**Choice depends on research question**:

* Impacts research: Often trend-inclusive (organisms experience absolute conditions)
* Attribution research: Often trend-removed (isolate natural variability)
* Climate change research: Compare both approaches

Summary Decision Matrix for Methods
------------------------------------

Quick Reference Table
~~~~~~~~~~~~~~~~~~~~~

+---------------------------+-------------------------+---------------------------+
| Research Goal             | Recommended Anomaly     | Recommended Extreme       |
+===========================+=========================+===========================+
| Publication-quality MHW   | shifting_baseline       | hobday_extreme            |
| detection                 |                         | window_spatial_hobday=5   |
+---------------------------+-------------------------+---------------------------+
| Climate change trends     | fixed_baseline          | hobday_extreme            |
| (trend-inclusive)         |                         |                           |
+---------------------------+-------------------------+---------------------------+
| Ecological impacts        | shifting_baseline       | hobday_extreme            |
|                           |                         | window_spatial_hobday=5   |
+---------------------------+-------------------------+---------------------------+
| Large-scale screening     | detrend_harmonic        | global_extreme            |
| (efficiency priority)     |                         |                           |
+---------------------------+-------------------------+---------------------------+
| Climate variability       | detrend_fixed_baseline  | hobday_extreme            |
| (full time series needed) |                         |                           |
+---------------------------+-------------------------+---------------------------+
| Operational monitoring    | detrend_harmonic        | hobday_extreme            |
|                           |                         | window_spatial_hobday=3   |
+---------------------------+-------------------------+---------------------------+

Performance Optimisation
========================

Chunking Strategy
-----------------

Optimal chunking is crucial for performance (it is also an art):

.. code-block:: python

   # General guidelines for chunking
   optimal_chunks = {
       'time': 30,              # ~1 month of daily data
       'lat': -1,               # All latitude points (if memory permits)
       'lon': -1                # All longitude points (if memory permits)
   }

   # For unstructured data
   optimal_chunks_unstruct = {
       'time': 10,
       'ncells': -1             # If memory permits
   }


Parallel Processing
-------------------

.. code-block:: python

   # Start optimised cluster
   client = marEx.helper.start_local_cluster(
       n_workers=4,
       threads_per_worker=2,
       memory_limit='8GB'
   )

   # For HPC systems
   cluster = marEx.helper.start_distributed_cluster(
       cores=128,
       memory='256GB',
       queue='compute'
   )

Grid Types and Coordinate Systems
=================================

Structured/Gridded Data
-----------------------

Regular latitude-longitude grids are the most common data type:

.. code-block:: python

   # Typical structured data
   print(sst.dims)         # e.g. ('time', 'lat', 'lon')
   print(sst.coords)       # e.g. ('time', 'lat', 'lon')

   # marEx automatically detects structured grids
   processed = marEx.preprocess_data(sst)

Unstructured Data
-----------------

Irregular meshes common in coastal and global ocean models:

.. code-block:: python

   # Typical unstructured data (e.g., ICON model)
   print(sst.dims)         # e.g. ('time', 'ncells')
   print(sst.coords)       # e.g. ('time', 'lat', 'lon') as coordinates, not dims

   # Specify grid configuration
   marEx.specify_grid(
       grid_type='unstructured',
       fpath_tgrid='grid_info.nc',      # Grid topology file
       fpath_ckdtree='kdtree.pkl'       # Spatial index (optional)
   )

**Requirements:**

* Spatial dimension name (``ncells``, ``cell``, etc.)
* Latitude/longitude as coordinate arrays
* Optional: Grid topology information for advanced/unstructured features


Advanced Features
=================

Compound Events
---------------

Analyse events that exceed multiple thresholds or variables:

.. code-block:: python

   # Multiple variable analysis
   sst_extremes = marEx.preprocess_data(sst, threshold_percentile=95)
   salinity_extremes = marEx.preprocess_data(salinity, threshold_percentile=5)  # Low salinity

   # Compound events
   compound_events = sst_extremes.extreme_events & salinity_extremes.extreme_events


Best Practices and Guidelines
=============================

Research Workflow
-----------------

1. **Exploratory Analysis**: Start with basic preprocessing to understand data
2. **Method Comparison**: Test different methods on subset of data
3. **Quality Control**: Validate results thoroughly
4. **Full Processing**: Apply chosen method to complete dataset
5. **Validation**: Compare with known events and literature


Literature Compliance
--------------------

For marine heatwave studies following Hobday et al. (2016):

.. code-block:: python

   # Standard MHW definition
   mhw_config = {
       'method_anomaly': 'shifting_baseline',
       'method_extreme': 'hobday_extreme',
       'threshold_percentile': 90,
       'window_days_hobday': 11,
       'window_year_baseline': 30,
       'smooth_days_baseline': 11,
       'window_spatial_hobday': 1,  # Hobday et al. (2016) considers only single points
   }


Getting Help and Support
========================

Documentation Resources
-----------------------

* **Tutorials**: Step-by-step guides in ``docs/tutorials/``
* **API Reference**: Complete function documentation
* **Examples**: Real-world analysis examples
* **Performance Guide**: Optimisation tips and tricks

Community Support
-----------------

* **GitHub Issues**: Bug reports and feature requests
* **Discussions**: Community Q&A and examples
* **Documentation**: Contributions welcome

Citations and References
========================

When using marEx in publications, please cite:

* **marEx package**
* **Marine heatwave methods**: Hobday et al. (2016) for standard MHW definition

For more detailed examples and advanced usage, see the :doc:`examples/index`.
