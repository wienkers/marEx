===========
Why marEx?
===========

marEx provides unique capabilities not available in alternative tools (e.g., ocetrac), making it the choice for extreme event analysis. This page highlights the distinguishing features that set marEx apart.

.. contents:: Table of Contents
   :local:
   :depth: 2

Advanced Tracking & Identification
===================================

Independent Event Tracking with Merge/Split Genealogy
------------------------------------------------------

**The Mega-Event Problem:** Basic tracking methods (e.g., 3D connected component labeling with `ocetrac`) treat time as just another spatial dimension, permanently merging ANY objects that touch at any point in space-time. This creates a chain reaction where distinct events that briefly touch become irrevocably linked, producing unrealistic basin-spanning "mega-events" that combine dozens of independent phenomena. These algorithmic artefacts have no coherent physical origin and produce meaningless statistics, making them unsuitable for mechanistic studies of extreme event dynamics.

**marEx Solution:** marEx tracks events through advection, spatio-temporal morphing, splitting, and merging with complete parent/child relationships recorded in ``merge_ledger``. This genealogical reconstruction prevents mega-events by requiring significant overlap (not just touching) and maintaining individual event identities. This approach is critical for mechanistic studies of ocean extreme event evolution and enables accurate, unbiased Lagrangian statistics on physically realistic events.

**Tracking Algorithm Comparison:**

The video below demonstrates why marEx's selective merging with overlap thresholds and genealogy tracking are crucial for scientifically meaningful event analysis:

.. video:: /_static/videos/tracking_comparison.mp4
   :width: 700
   :autoplay:
   :loop:

**Left (Basic Method — Chain-Reaction Merging):** 3D connected components permanently merge any touching objects. Event A touches B → merged as AB. AB touches C → all become one "mega-event". Over time, independent physical phenomena are spuriously linked into single, basin-spanning artefacts with no mechanistic relevance. Statistics derived from these mega-events are scientifically meaningless.

**Right (marEx Advanced Method — Genealogy Tracking):** Prevents mega-events using ``overlap_threshold`` to distinguish transient contact from true merging. Maintains individual event identities and records complete parent/child genealogy in ``merge_ledger``. Enables reconstruction of physically realistic event evolution where tracked objects correspond to actual coherent phenomena, providing mechanistically relevant statistics for scientific analysis.

**Key benefits:**

* Full parent/child ID tracking in ``merge_ledger`` dataset
* Overlap requirement for each merge/split event
* Enables reconstruction of complex event dynamics (e.g., fusion of two smaller marine heatwaves into one larger event)

**Code reference:** :mod:`marEx.track.tracker` with ``allow_merging=True`` parameter

Nearest-Neighbor Partitioning (nn_partitioning)
------------------------------------------------

marEx implements advanced partitioning for split events based on the nearest-neighbor parent cell, which is more robust than simpler centroid-based methods.

**Key benefits:**

* Prevents artefacts where small fragments "inherit" disproportionately large portions of new objects
* Provides physically realistic allocation of area and identity when events split
* Most accurate tracking method available

**Code reference:** :mod:`marEx.track.tracker` with ``nn_partitioning=True`` parameter

Temporal Gap Filling (T_fill)
------------------------------

marEx uses morphological closing along the time axis to maintain event continuity across short interruptions. This prevents a single, persistent event from being incorrectly split into multiple shorter events if it temporarily weakens below the detection threshold for a few time steps.

**Key benefits:**

* Maintains continuity of persistent events across brief interruptions
* More robust than simple rule-based gap filling
* Configurable gap length (e.g., T_fill=4 fills gaps up to 4 days)

**Code reference:** :mod:`marEx.track.tracker` with ``T_fill`` parameter

Morphological Preprocessing (R_fill)
-------------------------------------

marEx applies morphological closing and opening operations to spatially fill gaps and remove noise *before* tracking begins. For unstructured grids, this is implemented with a highly efficient sparse matrix approach. This creates more coherent and less noisy binary event fields, leading to more stable and meaningful object tracking.

**Key benefits:**

* Fills small holes within events and smooths boundaries before identification
* Prevents spurious small objects and artificially fragmented events
* Dual implementation: Dask-powered for structured grids, scipy sparse matrices for unstructured data

**Code reference:** :mod:`marEx.track.tracker` with ``R_fill`` parameter

Dual Area Filtering Strategies
-------------------------------

marEx allows both a percentile-based (``area_filter_quartile``, adaptive to dataset) OR absolute (``area_filter_absolute``, reproducible across datasets) thresholds for object filtering.

**Key benefits:**

* Quartile: Remove smallest X% of events (adaptive, useful for exploratory analysis)
* Absolute: Fixed minimum area threshold (reproducible, useful for cross-dataset comparison)

**Code reference:** :mod:`marEx.track.tracker` with ``area_filter_quartile`` or ``area_filter_absolute``

Flexible & Rigorous Anomaly Detection
======================================

Four Distinct Anomaly Methods
------------------------------

marEx provides **four** scientifically rigorous anomaly calculation methods with documented trade-offs.

**Available methods:**

* **shifting_baseline**: Rolling climatology that adapts to changing climate (most accurate, default in v3.0+)
* **detrend_fixed_baseline**: Polynomial detrending followed by fixed daily climatology (preserves full time-series length, removes long-term trends)
* **fixed_baseline**: Simple daily climatology (keeps trends in anomaly, straightforward interpretation)
* **detrend_harmonic**: Fast harmonic + polynomial model (efficient but may bias certain statistics)

**Key benefits:**

* Choose between accuracy, computational efficiency, time-series preservation, and trend handling
* Accommodate analyses that need to include/exclude long-term trends
* Account for shifting seasonal cycles or maintain stationary baselines

**Code reference:** :mod:`marEx.detect.preprocess_data` with ``method_anomaly`` parameter

Hobday Spatial Window Extension
--------------------------------

marEx extends/generalises the standard Hobday et al. (2016) temporal window by adding a spatial dimension (``window_spatial_hobday``). This creates a spatio-temporal cube of data points for calculating percentile thresholds (e.g., 5×5 spatial × 11 days = 275 samples per year), resulting in more robust and spatially coherent statistics. This is a major methodological advancement over the original Hobday definition.

**Key benefits:**

* Produces spatially coherent thresholds by pooling neighbouring grid cells (motivated by spatio-temporal correlation lengthscale)
* Reduces noise and statistical uncertainty in anomaly threshold calculations
* Especially valuable for short time series, high percentile thresholds, or noisy data (e.g., satellite SST with gaps)

**Limitations:**

* Structured grids only (not supported for unstructured/irregular grids)
* Requires ``method_percentile='approximate'``

**Code reference:** :mod:`marEx.detect.preprocess_data` with ``window_spatial_hobday`` parameter (default=5)

Histogram-Based Approximate Percentiles
----------------------------------------

marEx implements a clever 2D histogram approach for percentile calculation that is highly memory-efficient and parallelisable with Dask. This method is uses **100× less memory** than exact computation while maintaining ~0.01°C precision—sufficient for most studies. This method makes long time-series terabyte-scale percentile calculations feasible that were previously unachievable with daily data.

**Key benefits:**

* Enables global-in-time calculations on massive datasets
* ~0.01°C precision adequate for marine heatwave studies
* Overcomes the memory bottleneck of exact percentiles (which require loading entire time series)

**Code reference:** :mod:`marEx.detect.preprocess_data` with ``method_percentile='approximate'`` (default)

Extreme Scale & Performance
============================

Terabyte-Scale Processing
-------------------------

marEx features a "Dask-first" architecture with mandatory Dask validation that processes datasets **100-1000× larger than available RAM**. The package is designed from the ground up for exascale data, enabling baseline computations on 100+ years of daily global data.

**Key benefits:**

* Process massive climate datasets efficiently with intelligent chunking
* Explicit chunking control via ``dask_chunks`` parameter throughout pipeline

**Code reference:** All functions in :mod:`marEx.detect` and :mod:`marEx.track` require Dask-backed arrays

HPC/SLURM Integration
---------------------

marEx provides wrappers for easy deployment on supercomputers via the ``marEx.helper`` module with automatic cluster configuration, memory optimisation (256GB/512GB/1024GB nodes), and dashboard tunneling. Designed specifically for DKRZ Levante and adaptable to other HPC systems, it simplifies the process of scaling an analysis from a laptop to a supercomputer.

**Key benefits:**

* Abstracts away the complexity of configuring Dask for specific HPC environments
* Pre-configured memory settings for common node types
* Dashboard tunneling for remote monitoring
* System resource checking for local clusters

**Code reference:** :mod:`marEx.helper.start_distributed_cluster` for SLURM systems

JAX Acceleration
----------------

marEx can leverage JAX for significant performance gains (**10-50× speedup** reported) on critical-path calculations. The integration includes graceful fallbacks to NumPy+Numba if JAX is not installed, so users get acceleration if available, but code still works without it.

**Key benefits:**

* Dramatically reduces computation time for large datasets on GPU/TPU systems
* Moving from hours to minutes for key preprocessing steps
* Automatic backend selection

**Code reference:** Install with ``pip install marEx[full]`` for JAX support

Numba JIT Compilation
---------------------

marEx uses Numba's just-in-time (JIT) compilation as a **core dependency** for CPU-bound operations, providing performance acceleration without requiring any user configuration. Numba compiles Python functions to optimised machine code at runtime, delivering near-C performance for numerical computations.

**Key benefits:**

* Provides baseline acceleration, even without JAX/GPU
* Transparent performance gains on CPU-intensive tracking and grid operations

**Code reference:** Numba is a required dependency installed automatically with marEx


Grid-Agnostic & Universal
==========================

Grid-Agnostic Processing
------------------------

marEx provides the same API for structured (lat/lon), unstructured (FESOM/ICON/MPAS), regridded, coarse resolution, and regional domains. Specialised algorithms (e.g., sparse-matrix morphological operations for unstructured grids) adapt automatically based on grid type detection. Users can apply the exact same analysis workflow to data from traditional climate models, satellite products, and modern variable-resolution ocean models.

**Key benefits:**

* Transparent grid handling—write code once, use everywhere
* Automatic algorithm selection based on grid structure
* Supports regular rectangular grids and irregular meshes with connectivity

**Supported grid types:**

* Structured: Standard climate models (CMIP6), reanalysis, satellite data
* Unstructured: Ocean models (FESOM, ICON-O, MPAS-Ocean), finite element output

**Code reference:** :mod:`marEx.track.tracker` with ``unstructured_grid`` parameter (auto-detected)

Polymorphic Visualisation (plotX)
---------------------------------

marEx provides a visualisation system via an xarray accessor (``.plotX``) that automatically detects the grid type (structured vs. unstructured) and uses the appropriate plotting backend (``GriddedPlotter`` vs ``UnstructuredPlotter``). Same code produces single-panel plots, multi-panel comparisons, and MP4 animations for all grid types with automatic projection handling.

**Key benefits:**

* Simplifies creation of publication-quality maps and animations
* No need to write custom plotting logic for each grid type
* Global caches for triangulation and KDTree data (unstructured grid performance)

**Code reference:** :mod:`marEx.plotX` via ``.plotX`` accessor with :class:`marEx.PlotConfig`

Regional Tracker
----------------

marEx provides a convenience function ``regional_tracker()`` for spatially bounded analysis with coordinate unit specification (degrees/radians) for non-global domains. This handles for example high-resolution regional studies (e.g., 0.05° European domain).

**Key benefits:**

* Dedicated support for regional/nested domains
* Manual override for coordinate system when auto-detection insufficient
* Same robust tracking algorithms applied to bounded regions

**Code reference:** :func:`marEx.regional_tracker` convenience function

Automatic Grid Cell Area Calculation
-------------------------------------

marEx provides transparent conversion from cell counts to physical areas (km²) using spherical geometry for regular lat/lon grids. The ``grid_resolution`` parameter calculates Area = R² × |sin(lat + dlat/2) - sin(lat - dlat/2)| × dlon without requiring manual pre-computation of cell areas.

**Key benefits:**

* No need to provide pre-computed cell areas for regular grids
* Automatic spherical geometry calculations
* Transparent physical area reporting in tracking outputs

**Code reference:** :mod:`marEx.track.tracker` with ``grid_resolution`` parameter

Production-Ready Infrastructure
================================

Configurable Logging System
----------------------------

marEx provides three logging modes (verbose/normal/quiet) with performance monitoring, timing decorators, and memory usage tracking.

**Code reference:** :mod:`marEx.logging_config`


Coordinate Auto-Detection & Unification
----------------------------------------

marEx automatically detects degrees vs radians (checks if longitude range is ~360° or ~2π). This provides transparent handling of different coordinate conventions. Manual override is available via ``regional_mode=True`` and ``coordinate_units`` for regional domains where auto-detection may fail.

**Key benefits:**

* No need to manually convert coordinate systems
* Works with different dataset conventions out of the box
* Validation with informative errors when detection is ambiguous

**Code reference:** :mod:`marEx.track.tracker` coordinate detection logic


Summary
=======

These capabilities position marEx as a high-performance, scalable, and scientifically rigorous tool for extreme event analysis.

**Next Steps:**

* :doc:`installation` - Get marEx installed
* :doc:`quickstart` - Start analysing extremes in 5 minutes
* :doc:`user_guide` - Usage guide with method selection
* :doc:`examples` - Complete workflow demonstrations
