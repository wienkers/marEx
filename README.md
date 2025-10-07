![logo](https://github.com/user-attachments/assets/df99a84b-c88f-447d-a362-ac538d9b924b)

[![CI](https://github.com/wienkers/marEx/actions/workflows/ci.yml/badge.svg)](https://github.com/wienkers/marEx/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/wienkers/marEx/branch/main/graph/badge.svg)](https://codecov.io/gh/wienkers/marEx)
[![PyPI version](https://badge.fury.io/py/marEx.svg)](https://badge.fury.io/py/marEx)
[![Documentation Status](https://readthedocs.org/projects/marex/badge/?version=latest)](https://marex.readthedocs.io/en/latest/)
[![PyPI Downloads](https://static.pepy.tech/badge/marex)](https://pepy.tech/projects/marex)
[![DOI](https://zenodo.org/badge/945834123.svg)](https://doi.org/10.5281/zenodo.16922881)

# Marine Extremes Detection and Tracking

**Efficient & scalable marine extremes detection, identification, & tracking for exascale climate data.**

marEx is a high-performance Python framework for identifying and tracking extreme oceanographic events (such as Marine Heatwaves or Acidity Extremes) in massive climate datasets. Built on advanced statistical methods and distributed computing, it processes decades of daily-resolution global ocean data with unprecedented efficiency and scalability.

---

## Key Features

- **⚡ Extreme Performance**: Process 100+ years of high-resolution daily global data in minutes
- **🌐 Universal Grid Support**: Native support for both regular (lat/lon) grids and unstructured ocean models
- **📈 Advanced Event Tracking**: Handles coherent object splitting, merging, and evolution
- **📊 Multiple Detection Methods**: Scientifically rigorous algorithms for robust extreme event identification
- **☁️ Cloud-Native Scaling**: Identical codebase scales from laptop to supercomputer using up to 1024+ cores
- **🧠 Memory Efficient**: Intelligent chunking and lazy evaluation for datasets larger than memory


---

https://github.com/user-attachments/assets/501537ff-5adb-4e13-ba08-6a333bac2a02

![marEx_front](https://github.com/user-attachments/assets/939fceee-8990-46fb-b3f8-30e803b6c802)

---

## Quick Start

```python
import xarray as xr
import marEx

# Load sea surface temperature data
sst = xr.open_dataset('sst_data.nc', chunks={'time': 30}).sst

# Identify extreme events
extreme_events_ds = marEx.preprocess_data(
    sst,
    threshold_percentile=95,
    method_anomaly='shifting_baseline',
    method_extreme='hobday_extreme'
)

# Track events through time
events_ds = marEx.tracker(
    extreme_events_ds.extreme_events,
    extreme_events_ds.mask,
    R_fill=8,
    area_filter_absolute=100,
    allow_merging=True
).run()

# Visualise results
fig, ax, im = (events_ds.ID_field > 0).mean("time").plotX.single_plot(
    marEx.PlotConfig(var_units="MHW Frequency", cmap="hot_r", cperc=[0, 96])
)
```

---

## Installation

```bash
pip install marEx[full,hpc]
```

For detailed installation instructions, including HPC environments and optional dependencies, see the **[Installation Guide](https://marex.readthedocs.io/en/latest/installation.html)**.

---

## Core Workflow

marEx follows a three-stage pipeline:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  1. Detect      │  →   │  2. Track       │  →   │  3. Visualise   │
│    Extremes     │      │    Events       │      │     & Analyse   │
└─────────────────┘      └─────────────────┘      └─────────────────┘
        ↓                        ↓                        ↓
preprocess_data()           tracker()                  plotX()
        ↓                        ↓                        ↓
Binary extreme map        Tracked objects          Maps, animations,
                            with unique IDs           & statistics
```

**Learn more**: [Core Concepts](https://marex.readthedocs.io/en/latest/concepts.html) | [User Guide](https://marex.readthedocs.io/en/latest/user_guide.html)

---

## Documentation

For detailed guides, tutorials, and API reference, visit the full documentation:

### 📚 **[marEx Documentation on ReadTheDocs](https://marex.readthedocs.io/)**

**Quick Links:**
- **[Quickstart Guide](https://marex.readthedocs.io/en/latest/quickstart.html)** - Get started in 5 minutes
- **[Why marEx?](https://marex.readthedocs.io/en/latest/why_marex.html)** - Learn about the unique capabilities
- **[Core Concepts](https://marex.readthedocs.io/en/latest/concepts.html)** - Understanding marEx's design and workflow
- **[User Guide](https://marex.readthedocs.io/en/latest/user_guide.html)** - Comprehensive usage guide with method selection, parameter tuning, and performance optimisation
- **[API Reference](https://marex.readthedocs.io/en/latest/api.html)** - Complete function documentation
- **[Examples](https://marex.readthedocs.io/en/latest/examples.html)** - Jupyter notebooks for gridded, regional, and unstructured data
- **[Troubleshooting](https://marex.readthedocs.io/en/latest/troubleshooting.html)** - Common issues and solutions

---

## Examples

Explore complete workflows in the [example notebooks](https://github.com/wienkers/marEx/tree/main/examples):

- **Gridded Data**: Standard analysis for regular lat/lon grids (satellite data, CMIP6 models)
- **Regional Data**: Regional analysis with boundary handling (EURO-CORDEX)
- **Unstructured Data**: Analysis for irregular meshes (FESOM, ICON-O, MPAS-Ocean)

Each example demonstrates the full pipeline from preprocessing to visualisation.

---

## Key Capabilities

### Detection Methods

marEx provides multiple scientifically rigorous methods for anomaly calculation and extreme identification:

**Anomaly Detection:**
- **Shifting Baseline**: Rolling climatology (most accurate, research standard)
- **Detrend Fixed Baseline**: Polynomial detrending + fixed climatology (preserves full time series)
- **Fixed Baseline**: Simple daily climatology (trend-inclusive)
- **Harmonic Detrending**: Fast polynomial + harmonic model (efficient screening)

**Extreme Identification:**
- **Hobday Method**: Day-of-year specific thresholds with spatial window extension (literature standard, Hobday et al. 2016)
- **Global Method**: Single threshold across time (fast, exploratory analysis)

**[→ Learn more about method selection](https://marex.readthedocs.io/en/latest/user_guide.html#method-selection-guide)**

### Advanced Tracking

- **Morphological Operations**: Fill spatial gaps and smooth event boundaries
- **Temporal Gap Filling**: Maintain event continuity across short interruptions
- **Merge/Split Handling**: Track event genealogy with improved nearest-neighbor partitioning
- **Area Filtering**: Remove spurious small events with percentile or absolute thresholds

**[→ Explore tracking algorithms](https://marex.readthedocs.io/en/latest/modules/track.html)**

### Performance & Scalability

- **Dask-First Architecture**: Parallel computation with automatic memory management
- **JAX Acceleration**: Optional GPU/TPU support for 10-50× speedup
- **HPC Integration**: SLURM cluster support for supercomputing environments
- **Memory Optimisation**: Process datasets 100-1000× larger than available RAM

**[→ Performance tuning guide](https://marex.readthedocs.io/en/latest/user_guide.html#performance-optimisation)**

---

## Getting Help

### Support Channels

- **[Documentation](https://marex.readthedocs.io/)** - Detailed guides and API reference
- **[GitHub Issues](https://github.com/wienkers/marEx/issues)** - Bug reports and feature requests
- **[GitHub Discussions](https://github.com/wienkers/marEx/discussions)** - Questions, ideas, and community support
- **[Example Notebooks](https://github.com/wienkers/marEx/tree/main/examples)** - Complete workflow demonstrations

### Reporting Issues

When reporting issues, please include:
- marEx version (`marEx.__version__`)
- Python version and operating system
- Dependency status (`marEx.print_dependency_status()`)
- Minimal reproducible example
- Full error traceback

---

## Citation

When using marEx in publications, please cite:

- **marEx package**: DOI [10.5281/zenodo.16922881](https://doi.org/10.5281/zenodo.16922881)
- **Hobday et al. (2016)**: "A hierarchical approach to defining marine heatwaves." *Progress in Oceanography* 141, 227-238. DOI [10.1016/j.pocean.2015.12.014](https://doi.org/10.1016/j.pocean.2015.12.014)

---

## Funding

This project has received funding through:

* The [EERIE](https://eerie-project.eu) (European Eddy-Rich ESMs) Project
* The European Union's Horizon Europe research and innovation programme under Grant Agreement No. 101081383
* The Swiss State Secretariat for Education, Research and Innovation (SERI) under contract #22.00366

---

## Contact

For questions, comments, or collaboration opportunities, please contact [Aaron Wienkers](mailto:aaron.wienkers@gmail.com).
