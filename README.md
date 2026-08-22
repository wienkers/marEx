<img src="media/logo.png" alt="marEx logo" width="100%">

[![CI](https://github.com/wienkers/marEx/actions/workflows/ci.yml/badge.svg)](https://github.com/wienkers/marEx/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/wienkers/marEx/branch/main/graph/badge.svg)](https://codecov.io/gh/wienkers/marEx)
[![PyPI version](https://badge.fury.io/py/marEx.svg)](https://badge.fury.io/py/marEx)
[![Documentation Status](https://readthedocs.org/projects/marex/badge/?version=latest)](https://marex.readthedocs.io/en/latest/)
[![PyPI Downloads](https://static.pepy.tech/badge/marex)](https://pepy.tech/projects/marex)
[![DOI](https://zenodo.org/badge/945834123.svg)](https://doi.org/10.5281/zenodo.16922881)

# Marine Extremes Detection and Tracking

**Efficient & scalable marine extremes detection, identification, & tracking for exascale climate data.**

marEx is a high-performance Python framework for identifying and tracking extreme oceanographic events (such as Marine Heatwaves or Acidity Extremes) in massive climate datasets. Built on advanced statistical methods and distributed computing, it processes decades of daily-resolution global ocean data with unprecedented efficiency and scalability.

📚 **[Full documentation on ReadTheDocs →](https://marex.readthedocs.io/)**

---

## Key Features

- **⚡ Extreme Performance**: Process 100+ years of high-resolution daily global data in minutes
- **🌐 Universal Grid Support**: Native support for both regular (lat/lon) grids and unstructured ocean models (FESOM, ICON-O, MPAS-Ocean)
- **📈 Advanced Event Tracking**: Overlap-thresholded merge/split handling with genealogical record — avoids the spurious "mega-events" of naive 3D connected-component methods
- **📊 Multiple Detection Methods**: Four anomaly methods and a generalised Hobday extreme definition with spatial pooling
- **☁️ Cloud-Native Scaling**: Identical codebase scales from laptop to supercomputer using up to 1024+ cores
- **🧠 Memory Efficient**: Intelligent chunking and lazy evaluation for datasets larger than memory

---

https://github.com/user-attachments/assets/501537ff-5adb-4e13-ba08-6a333bac2a02

![marEx_front](https://github.com/user-attachments/assets/939fceee-8990-46fb-b3f8-30e803b6c802)

---

## Installation

```bash
pip install marEx[full,hpc]
```

For detailed instructions, including HPC environments and optional dependencies, see the **[Installation Guide](https://marex.readthedocs.io/en/latest/installation.html)**.

---

## Quick Start

```python
import xarray as xr
import marEx

# Load sea surface temperature data
sst = xr.open_dataset('sst_data.nc', chunks={'time': 30}).sst

# 1. Detect extreme events
extreme_events_ds = marEx.preprocess_data(
    sst,
    threshold_percentile=95,
    method_anomaly='shifting_baseline',
    method_extreme='hobday_extreme',
)

# 2. Track events through time
events_ds = marEx.tracker(
    extreme_events_ds.extreme_events,
    extreme_events_ds.mask,
    R_fill=8,
    area_filter_absolute=100,
    allow_merging=True,
).run()

# 3. Visualise results
fig, ax, im = (events_ds.ID_field > 0).mean("time").plotX.single_plot(
    marEx.PlotConfig(var_units="MHW Frequency", cmap="hot_r", cperc=[0, 96])
)
```

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

➡️ **[Five-minute Quickstart](https://marex.readthedocs.io/en/latest/getting_started/quickstart.html)** · **[Core Concepts](https://marex.readthedocs.io/en/latest/guide/concepts.html)**

---

## Documentation

The documentation is organised so you can find what you need quickly:

| Section | What's there |
| --- | --- |
| **[Getting Started](https://marex.readthedocs.io/en/latest/getting_started/index.html)** | Installation and a five-minute quickstart |
| **[Tutorials](https://marex.readthedocs.io/en/latest/tutorials/index.html)** | End-to-end notebooks for gridded, regional, and unstructured data |
| **[User Guide](https://marex.readthedocs.io/en/latest/guide/index.html)** | Concepts, method selection, parameter tuning, and performance |
| **[API Reference](https://marex.readthedocs.io/en/latest/api/index.html)** | Every public function and class |
| **[Why marEx?](https://marex.readthedocs.io/en/latest/why_marex.html)** | What sets marEx apart (with a tracking-comparison video) |
| **[Troubleshooting](https://marex.readthedocs.io/en/latest/troubleshooting.html)** | Common issues and solutions |

### Tutorials

Complete, runnable workflows (preprocess → track → visualise) are rendered directly in the docs:

- **[Gridded data](https://marex.readthedocs.io/en/latest/tutorials/gridded.html)** — regular lat/lon grids (satellite data, CMIP6 models)
- **[Regional data](https://marex.readthedocs.io/en/latest/tutorials/regional.html)** — spatially bounded, higher-resolution domains
- **[Unstructured data](https://marex.readthedocs.io/en/latest/tutorials/unstructured.html)** — irregular meshes (FESOM, ICON-O, MPAS-Ocean)

The source notebooks live in the [`examples/`](https://github.com/wienkers/marEx/tree/main/examples) directory.

---

## Capabilities at a Glance

**Detection** — see the **[Detection guide](https://marex.readthedocs.io/en/latest/guide/detection.html)**:

- Four anomaly methods: *shifting baseline* (rolling climatology, research standard), *detrend fixed baseline* (detrending + fixed climatology), *fixed baseline* (trend-inclusive), and *harmonic detrending* (fast screening)
- Two extreme definitions: the *Hobday* day-of-year method with a spatial-window extension (Hobday et al. 2016), and a fast *global* threshold
- Memory-efficient histogram-based approximate percentiles for terabyte-scale data

**Tracking** — see the **[Tracking guide](https://marex.readthedocs.io/en/latest/guide/tracking.html)**:

- Morphological gap-filling (`R_fill`) and temporal gap-filling (`T_fill`)
- Overlap-thresholded merge/split handling with nearest-neighbour partitioning
- Percentile or absolute area filtering; automatic spherical cell-area calculation

**Performance & scale** — see the **[Performance guide](https://marex.readthedocs.io/en/latest/guide/performance.html)**:

- Dask-first architecture for datasets 100–1000× larger than memory
- Optional JAX acceleration (10–50× speedup) with graceful NumPy/Numba fallback
- SLURM/HPC cluster integration via `marEx.helper`

---

## Getting Help

- **[Documentation](https://marex.readthedocs.io/)** — guides, tutorials, and API reference
- **[GitHub Issues](https://github.com/wienkers/marEx/issues)** — bug reports and feature requests
- **[GitHub Discussions](https://github.com/wienkers/marEx/discussions)** — questions, ideas, and community support

When reporting issues, please include: marEx version (`marEx.__version__`), Python version and OS, dependency status (`marEx.print_dependency_status()`), a minimal reproducible example, and the full error traceback.

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
