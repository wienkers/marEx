<img width="4913" height="1784" alt="logo" src="https://github.com/user-attachments/assets/17868fac-d112-4ada-805b-d2bd079c60f0" />

[![Continuous Integration](https://github.com/wienkers/marEx/actions/workflows/ci.yml/badge.svg)](https://github.com/wienkers/marEx/actions/workflows/ci.yml)
[![Tests](https://github.com/wienkers/marEx/workflows/Tests/badge.svg)](https://github.com/wienkers/marEx/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/wienkers/marEx/branch/main/graph/badge.svg)](https://codecov.io/gh/wienkers/marEx)
[![PyPI version](https://badge.fury.io/py/marEx.svg)](https://badge.fury.io/py/marEx)
[![Documentation Status](https://readthedocs.org/projects/marex/badge/?version=latest)](https://marex.readthedocs.io/en/latest/)
[![PyPI Downloads](https://static.pepy.tech/badge/marex)](https://pepy.tech/projects/marex)

Marine Extremes Python Package
==============================

**Efficient & scalable Marine Extremes detection, identification, & tracking for Exascale Climate Data.**

`MarEx` is a high-performance Python framework for identifying and tracking extreme oceanographic events (such as Marine Heatwaves or Acidity Extremes) in massive climate datasets. Built on advanced statistical methods and distributed computing, it processes decades of daily-resolution global ocean data with unprecedented efficiency and scalability.

## Key Capabilities

- **⚡ Extreme Performance**: Process 100+ years of high-resolution daily global data in minutes
- **🔬 Advanced Analytics**: Multiple statistical methodologies for robust extreme event detection
- **📈 Complex Event Tracking**: Seamlessly handles coherent object splitting, merging, and evolution
- **🌐 Universal Grid Support**: Native support for both regular (lat/lon) grids and unstructured ocean models
- **☁️ Cloud-Native Scaling**: Identical codebase scales from laptop to a supercomputer using up to 1024+ cores
- **🧠 Memory Efficient**: Intelligent chunking and lazy evaluation for datasets larger than memory

---
<details>
<summary>View 20 Years of <strong>marEx</strong> Tracking (Click to expand)</summary>

https://github.com/user-attachments/assets/36ee3150-c869-4cba-be68-628dc37e4775

</details>

![marEx_front](https://github.com/user-attachments/assets/939fceee-8990-46fb-b3f8-30e803b6c802)

---

## Features

### Data Pre-processing Pipeline

MarEx implements a highly-optimised preprocessing pipeline powered by `dask` for efficient parallel computation and scaling to very large spatio-temporal datasets. Included are two complementary methods for calculating anomalies and detecting extremes:

**Anomaly Calculation**:
  1. *Shifting Baseline* — Scientifically-rigorous definition of anomalies relative to a backwards-looking rolling smoothed climatology.
  2. *Detrended Baseline* — Efficiently removes trend & season cycle using a 6+ coefficient model (mean, annual & semi-annual harmonics, and arbitrary polynomial trends). (Highly efficient, but this approximation may lead to biases in certain statistics.)

**Extreme Detection**:
  1. *Hobday Extreme* — Implements a similar methodology to Hobday et al. (2016) with local day-of-year specific thresholds determined based on the quantile within a rolling window.
  2. *Global Extreme* — Applies a global-in-time percentile threshold at each point across the entire dataset. Optionally renormalises anomalies using a 30-day rolling standard deviation. (Highly efficient, but may misrepresent seasonal variability and differs from common definitions in literature.)


### Object Detection & Tracking
**Object Detection**:
  - Implements efficient algorithms for object detection in 2D geographical data.
  - Fully-parallelised workflow built on `dask` for extremely fast & larger-than-memory computation.
  - Uses morphological opening & closing to fill small holes and gaps in binary features.
  - Filters out small objects based on area thresholds.
  - Identifies and labels connected regions in binary data representing arbitrary events (e.g. SST or SSS extrema, tracer presence, eddies, etc...).
  - Performance/Scaling Test:  100 years of daily 0.25° resolution binary data with 64 cores...
    - Takes ~5 wall-minutes per _century_
    - Requires only 1 Gb memory per core (with `dask` chunks of 25 days)

**Object Tracking**:
  - Implements strict event tracking conditions to avoid very few, very large objects.
  - Permits temporal gaps (of `T_fill` days) between objects, to allow more continuous event tracking.
  - Requires objects to overlap by at least `overlap_threshold` fraction of the smaller objects's area to be considered the same event and continue tracking with the same ID.
  - Accounts for & keeps a history of object splitting & merging events, ensuring objects are more coherent and retain their previous identities & histories.
  - Improves upon the splitting & merging logic of [Sun et al. (2023)](https://doi.org/10.1038/s41561-023-01325-w):
    - _In this New Version_: Partition the child object based on the parent of the _nearest-neighbour_ cell (_not_ the nearest parent centroid).
  - Provides much more accessible and usable tracking outputs:
    - Tracked object properties (such as area, centroid, and any other user-defined properties) are mapped into `ID`-`time` space
    - Details & Properties of all Merging/Splitting events are recorded.
    - Provides other useful information that may be difficult to extract from the large `object ID field`, such as:
      - Event presence in time
      - Event start/end times and duration
      - etc...
  - Performance/Scaling Test:  100 years of daily 0.25° resolution binary data with 64 cores...
    - Takes ~8 wall-minutes per decade (cf. _Old_ Method, i.e. _without_ merge-split-tracking, time-gap filling, overlap-thresholding, et al., but here updated to leverage `dask`, now takes 1 wall-minute per decade!)
    - Requires only ~2 Gb memory per core (with `dask` chunks of 25 days)

### Visualisation
**Plotting**:
  - Provides a few helper functions to create pretty plots, wrapped subplots, and animations (e.g. below).

cf. Old (Basic) ID Method vs. New Tracking & Merging Algorithm:

https://github.com/user-attachments/assets/5acf48eb-56bf-43e5-bfc4-4ef1a7a90eff



## Technical Architecture

**Distributed Computing Stack:**
- **Framework**: `Dask` for distributed computation with asyncronous task scheduling
- **Parallelism**: Multi-level spatio-temporal parallelisation
- **Memory Management**: Lazy evaluation with automatic spilling and graph optimisation
- **I/O Optimisation**: Zarr-based intermediate storage with compression

**Performance Optimisations:**
- **JIT Compilation**: Numba-accelerated critical paths for numerical kernels
- **GPU Acceleration**: Optional JAX backend for tensor operations
- **Sparse Operations**: Custom sparse matrix algorithms for unstructured grids
- **Cache-Aware**: Memory access patterns optimised for modern CPU architectures


## Computational Workflow

1. **Preprocess**: Remove trends & seasonal cycles and identify anomalous extremes
2. **Detect**: Filter & label connected regions using morphological operations
3. **Track**: Follow objects through time, handling complex evolution patterns
4. **Analyse**: Extract event statistics, duration, and spatial properties

## Quick Start Example

```python
import xarray as xr
import marEx

# Load sea surface temperature data
sst = xr.open_dataset('sst_data.nc', chunks={}).sst

# Pre-process SST Data to identify extremes: cf. `01_preprocess_extremes.ipynb`
extreme_events_ds = marEx.preprocess_data(
    sst,
    threshold_percentile=95,
    method_anomaly='shifting_baseline',
    method_extreme='hobday_extreme'
)

# Identify & Track Marine Heatwaves through time: cf. `02_id_track_events.ipynb`
events_ds = marEx.tracker(
    extreme_events_ds.extreme_events,
    extreme_events_ds.mask,
    R_fill=8,
    area_filter_quartile=0.5,
    allow_merging=True
).run()

# Visualise results: cf. `03_visualise_events.ipynb`
fig, ax, im = (events_ds.ID_field > 0).mean("time").plotX.single_plot(marEx.PlotConfig(var_units="MHW Frequency", cmap="hot_r", cperc=[0, 96]))
```


## Installation & Setup

### Full Installation
```bash
# Complete HPC installation with all optional dependencies
pip install marEx[full,hpc]
```

### Development Installation
```bash
# Clone and install for development
git clone https://github.com/wienkers/marEx.git
cd marEx
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

## Getting Help

If you encounter installation issues:

1. **Documentation**: Check the [full documentation](https://marex.readthedocs.io/) for detailed guides and API reference
2. **Check Dependencies**: Run `marEx.print_dependency_status()` to identify missing components
3. **Search Issues**: Check the [GitHub Issues](https://github.com/wienkers/marEx/issues) for similar problems
4. **System Information**: Include your OS, Python version, and error messages when reporting issues
5. **Support**: Reach out to [Aaron Wienkers](mailto:aaron.wienkers@gmail.com)


## Funding

This project has received funding through:

* The [EERIE](https://eerie-project.eu) (European Eddy-Rich ESMs) Project
* The European Union's Horizon Europe research and innovation programme under Grant Agreement No. 101081383
* The Swiss State Secretariat for Education, Research and Innovation (SERI) under contract #22.00366

---
Please contact [Aaron Wienkers](mailto:aaron.wienkers@gmail.com) with any questions, comments, issues, or bugs.
