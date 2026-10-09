<img src="media/logo.png" alt="marEx logo" width="100%">

[![CI](https://github.com/wienkers/marEx/actions/workflows/ci.yml/badge.svg)](https://github.com/wienkers/marEx/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/wienkers/marEx/branch/main/graph/badge.svg)](https://codecov.io/gh/wienkers/marEx)
[![PyPI version](https://badge.fury.io/py/marEx.svg)](https://badge.fury.io/py/marEx)
[![Documentation Status](https://readthedocs.org/projects/marex/badge/?version=latest)](https://marex.readthedocs.io/en/latest/)
[![PyPI Downloads](https://static.pepy.tech/badge/marex)](https://pepy.tech/projects/marex)
[![DOI](https://zenodo.org/badge/945834123.svg)](https://doi.org/10.5281/zenodo.16922881)

# Weather & Climate Extremes Detection and Tracking

**Efficient & scalable climatologies, anomalies, extreme detection, & event tracking for exascale climate data.**

marEx is a Python framework of three stages, each usable on its own: smoothed climatologies and anomalies, extreme identification against a percentile threshold, and tracking of the resulting events through time. The same code runs on sea surface temperature, 2 m air temperature, precipitation, wind or a biogeochemical tracer, on regular grids and unstructured meshes, with a high or a low tail.

**[Full documentation on ReadTheDocs](https://marex.readthedocs.io/)**

---

https://github.com/user-attachments/assets/501537ff-5adb-4e13-ba08-6a333bac2a02

![marEx_front](https://github.com/user-attachments/assets/939fceee-8990-46fb-b3f8-30e803b6c802)

---

## The Three Stages

```
 marEx.anomaly.compute()      marEx.extremes.identify()         marEx.tracker()
┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
│ 1. Anomalies           │ → │ 2. Extremes            │ → │ 3. Tracking            │
│ climatology, detrend,  │   │ percentile thresholds, │   │ events with IDs, areas,│
│ standardise            │   │ upper or lower tail    │   │ merges and splits      │
└────────────────────────┘   └────────────────────────┘   └────────────────────────┘
          └───────────── marEx.preprocess_data() chains stages 1 and 2 ─────────────┘
```

Stage 1 never asks for a threshold, and stage 2 accepts anomalies from marEx or from anywhere else. If all you need is a smoothed daily climatology of a dataset that does not fit in memory:

```python
import xarray as xr
import marEx

sst = xr.open_dataset("sst_data.nc", chunks={"time": 25}).sst
anomalies = marEx.anomaly.compute(sst, method="shifting_baseline")
anomalies.dat_anomaly.to_zarr("sst_anomaly.zarr")
```

The full pipeline, from raw field to tracked events:

```python
import xarray as xr
import marEx

client = marEx.helper.start_local_cluster(n_workers=4, memory_limit="8GB")

sst = xr.open_dataset("sst_data.nc", chunks={"time": 25}).sst

extremes = marEx.preprocess_data(
    sst,
    method_anomaly="shifting_baseline",
    method_extreme="seasonal_percentile",
    threshold_percentile=95,
)

events = marEx.tracker(
    extremes.extreme_events,
    extremes.mask,
    R_fill=8,
    area_filter_absolute=100,
    allow_merging=True,
).run()

fig, ax, im = (events.ID_field > 0).mean("time").plotX.single_plot(
    marEx.PlotConfig(var_units="Event Frequency", cmap="hot_r", cperc=[0, 96])
)
```

`events` includes `ID_field`, `global_ID`, `area`, `centroid`, `presence`, `time_start`, `time_end` and `merge_ledger`.

---

## Key Features

- **Stages that stand alone**: anomalies without events, events from anomalies computed elsewhere, or the whole chain. No stage needs the next one's parameters.
- **Any grid, any cadence, either tail**: lat/lon grids and unstructured meshes (FESOM, ICON, MPAS) share one API. Fields with an extra dimension such as depth run through detection (a 3-D run equals the per-level 2-D runs), and monthly or sub-daily time axes are supported. `tail="lower"` flags cold spells and droughts.
- **Larger than memory**: `compute_mode="streaming"` keeps intermediates on disk instead of pinning them in worker memory. It cuts pinned bytes, not peak memory, and in the table below it completes a track in an allocation where `persist` is killed.
- **Advanced Event Tracking**: merges and splits require overlap rather than contact, and every parent and child relationship is written to `merge_ledger`. Naive 3-D connected-component labelling chains anything that touches into one basin-spanning event.
- **Results independent of how you chunk**: detection output does not change with the input chunking, verified on the test fixtures for every anomaly and threshold method, and the gridded tracker agreed across time chunks and compute modes in the cases tested. The unstructured tracker is chunk-independent except for equidistant tie-breaks.

---

## Measured at Scale

Single runs on one DKRZ Levante node, with outputs checked for agreement between modes.

| Stage and data | Configuration | Result |
| --- | --- | --- |
| Detect, 40 years of daily 0.25° global SST (9282 × 720 × 1440 output days) | 4 workers × 22 GB, 64 threads | 3954 s with `persist`, 3663 s with `streaming`, identical arrays |
| Track, 0.25° global, 3804 days | 24 GiB allocation, 4 × 4 GB dask budget | `persist` OOM-killed in 5 of 5 runs, `streaming` completed in 7 of 7 |
| Track, same field, 96 GB budget | single run each | pinned 337 GB → 0.34 GB; peak 56.7 → 19.1 GB; wall time within 1 % |
| Track, ICON R02B09 (14.9 M cells), 1096 days | 16 workers × 12 GB | `persist` 4297 s, `streaming` 4219 s; pinned 751 → 148 GB |

The squeeze result rests on the gridded tracker. Detection did not show a peak-memory saving from streaming. Details, sizing guidance and caveats are in the [performance guide](https://marex.readthedocs.io/en/latest/guide/performance.html).

---

## Applications

The [application gallery](https://marex.readthedocs.io/en/latest/applications/index.html) has a configuration and its caveats for each case.

- **Marine heatwaves**: the original use, on satellite and model SST.
- **Atmospheric heatwaves**: heat-driven electricity demand and heat stress.
- **Wind drought**: low-tail wind speed events for energy supply.
- **Precipitation drought**: monthly, lower-tail events for hydro and agriculture.
- **Subsurface ocean**: 3-D temperature extremes relevant to aquaculture and fisheries.
- **Event catalogues**: tracked footprints with duration, area and intensity, the input to frequency and severity analysis.

---

## Installation

```bash
pip install marEx[full,hpc]
```

For HPC environments and optional dependencies, see the **[Installation Guide](https://marex.readthedocs.io/en/latest/installation.html)**.

---

## Documentation

| Section | What's there |
| --- | --- |
| **[Getting Started](https://marex.readthedocs.io/en/latest/getting_started/index.html)** | Installation and a five-minute quickstart |
| **[Tutorials](https://marex.readthedocs.io/en/latest/tutorials/index.html)** | End-to-end notebooks for gridded, regional and unstructured data |
| **[Applications](https://marex.readthedocs.io/en/latest/applications/index.html)** | Worked cases by domain |
| **[User Guide](https://marex.readthedocs.io/en/latest/guide/index.html)** | [Anomalies](https://marex.readthedocs.io/en/latest/guide/anomalies.html), [extremes](https://marex.readthedocs.io/en/latest/guide/extremes.html), [dimensions and time](https://marex.readthedocs.io/en/latest/guide/dimensions_and_time.html), tracking, performance, [validation](https://marex.readthedocs.io/en/latest/guide/validation.html) |
| **[What's New](https://marex.readthedocs.io/en/latest/whats_new.html)** | Changes in 5.0 and the migration table |
| **[Why marEx?](https://marex.readthedocs.io/en/latest/why_marex.html)** | The design choices, with a tracking-comparison video |
| **[API Reference](https://marex.readthedocs.io/en/latest/api/index.html)** | Every public function and class |
| **[Troubleshooting](https://marex.readthedocs.io/en/latest/troubleshooting.html)** | Common issues and solutions |

---

## What's New in 5.0

- A standalone `marEx.anomaly.compute` and `marEx.extremes.identify`, a lower tail, fields with an extra dimension, and monthly or sub-daily time axes.
- `compute_mode` (`persist`, `lazy`, `streaming`) on detection and tracking, plus a per-stage `ResourceMonitor` and a small-object prefilter for the tracker.
- Renamed methods and parameters (`seasonal_percentile`, `window_years`, `standardise`) and a removed `marEx.detect` module. The [migration table](https://marex.readthedocs.io/en/latest/whats_new.html) maps every old name to its replacement.

---

## Development and Validation

I developed the scientific methodology and the implementation of marEx by hand up to and including v4.1 (April 2026). From v4.1 onward I have used Claude Code to help optimise, generalise and test it. Since mid-2026 every change has been held to reference-output (golden) tests at zero tolerance (one threshold field allows 2e-14 of round-off), plus tests that the results do not depend on chunking or compute mode. Deliberate corrections are listed in the changelog. How correctness is tested, and where the tests stop, is described on the [validation page](https://marex.readthedocs.io/en/latest/guide/validation.html), and the changes are listed in [What's New](https://marex.readthedocs.io/en/latest/whats_new.html).

---

## Getting Help

- **[Documentation](https://marex.readthedocs.io/)**: guides, tutorials and API reference
- **[GitHub Issues](https://github.com/wienkers/marEx/issues)**: bug reports and feature requests
- **[GitHub Discussions](https://github.com/wienkers/marEx/discussions)**: questions, ideas and community support

When reporting issues, please include the marEx version (`marEx.__version__`), Python version and OS, dependency status (`marEx.print_dependency_status()`), a minimal reproducible example and the full traceback.

---

## Citation

When using marEx in publications, please cite:

- **marEx package**: DOI [10.5281/zenodo.16922881](https://doi.org/10.5281/zenodo.16922881)
- **Hobday et al. (2016)**: "A hierarchical approach to defining marine heatwaves." *Progress in Oceanography* 141, 227-238. DOI [10.1016/j.pocean.2015.12.014](https://doi.org/10.1016/j.pocean.2015.12.014)

---

## Funding

* The [EERIE](https://eerie-project.eu) (European Eddy-Rich ESMs) Project
* The European Union's Horizon Europe research and innovation programme under Grant Agreement No. 101081383
* The Swiss State Secretariat for Education, Research and Innovation (SERI) under contract #22.00366

---

## Contact

For questions, comments, or collaboration opportunities, please contact [Aaron Wienkers](mailto:aaron.wienkers@gmail.com).
