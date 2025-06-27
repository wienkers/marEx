
![Logo](media/logo.png)


Marine Extremes Python Package
==============================

**Efficient & scalable Marine Extremes detection, identification, & tracking for Exascale Climate Data.**

`MarEx` is a high-performance Python framework for identifying and tracking extreme oceanographic events (such as Marine Heatwaves or Acidity Extremes) in massive climate datasets. Built on advanced statistical methods and distributed computing, it processes decades of daily-resolution global ocean data with unprecedented efficiency and scalability.

## Key Capabilities

- **⚡ Extreme Performance**: Process 100+ years of high-resolution daily global data in minutes
- **🔬 Advanced Analytics**: Multiple statistical methodologies for robust extreme event detection  
- **📈 Complex Event Tracking**: Seamlessly handles coherent object splitting, merging, and evolution
- **🌐 Universal Grid Support**: Native support for both regular (lat/lon) grids and unstructured ocean models
- **☁️ Cloud-Native Scaling**: Identical codebase scales from laptop to a supercomputer using to 1024+ cores
- **🧠 Memory Efficient**: Intelligent chunking and lazy evaluation for datasets larger than memory

---

## Features (v3.0)

### Data Pre-processing Pipeline

MarEx implements a highly-optimised preprocessing pipeline powered by `dask` for efficient parallel computation and scaling to very large spatio-temporal datasets. Included are two complementary methods for calculating anomalies and detecting extremes:

**Anomaly Calculation**:
  1. Shifting Baseline — Scientifically-rigorous definition of anomalies relative to a backwards-looking rolling smoothed climatology.
  2. Detrended Baseline — Efficiently removes trend & season cycle using a 6+ coefficient model (mean, annual & semi-annual harmonics, and arbitrary polynomial trends). (Highly efficient, but this approximation may lead to biases in certain statistics.)

**Extreme Detection**:
  1. Hobday Extreme — Implements a similar methodology to Hobday et al. (2016) with local day-of-year specific thresholds determined based on the quantile within a rolling window.
  2. Global Extreme — Applies a global-in-time percentile threshold at each point across the entire dataset. Optionally renormalises anomalies using a 30-day rolling standard deviation. (Highly efficient, but may misrepresent seasonal variability and differs from common definitions in literature.)


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

## Example Usage

### 1. Pre-process SST Data: cf. `01_preprocess_extremes.ipynb`
```python
import xarray as xr
import marEx

# Load SST data & rechunk for optimal processing
file_name = 'path/to/sst/data'
sst = xr.open_dataset(file_name, chunks={'time':500}).sst

# Process Data
extremes_ds = marEx.preprocess_data(
    sst, 
    method_anomaly='shifting_baseline',      # Anomalies from a rolling climatology using previous window_year years
    method_extreme='hobday_extreme',         # Local day-of-year specific thresholds with windowing
    threshold_percentile=95,                 # 95th percentile threshold for extremes
    window_year_baseline=15,                 # Rolling climatology window
    smooth_days_baseline=21,                 #    and smoothing window for determining the anomalies
    window_days_hobday=11                    # Window size of compiled samples collected for the extremes detection
)
```

The resulting xarray dataset `extremes_ds` will have the following structure & entries:
```
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
```
where
- `dat_anomaly` is the anomaly (either detrended or from a rolling climatology)
- `mask` is the deduced land-sea mask
- `extreme_events` is the binary field locating extreme events
- `thresholds` is the day-of-year specific thresholds used to determine extreme events (if `method_extreme='hobday_extreme'` is set).
- `extreme_events_stn` is the STD-renormalised anomalies (if `normalise=True`)

Optional arguments for `marEx.preprocess_data()` include:
- `method_anomaly`: Method for anomaly detection. 
  - `'detrended_baseline'`: (Default) Uses a 6+ coefficient model (mean, annual & semi-annual harmonics, and arbitrary polynomial trends). This method is much faster to preprocess, but care must be taken because the harmonic detrending strongly biases certain statistics. Requires the additional arguments:
    - `std_normalise`: Whether to normalise the anomalies using a 30-day rolling standard deviation. Default is `False`.
    - `detrend_orders`: List of polynomial orders for detrending. Default is `[1]`, i.e. 1st order (linear) detrend. `[1,2]` e.g. would use a linear+quadratic detrending.
  - `'shifting_baseline'`: Uses a smoothed rolling climatology of the previous `window_year_baseline` years with `smooth_days_baseline` days of smoothing. This method is more "correct", but shortens the time series by `window_year_baseline` years. Requires the additional arguments: 
    - `window_year_baseline`: The number of years to use for the rolling climatology baseline. Default is `15` previous years.
    - `smooth_days_baseline`: The number of days to use for smoothing the rolling climatology baseline. Default is `21` days.
- `method_extreme`: Method for identifying extreme events. 
  - `'global_extreme'`: (Default) Applies a global-in-time (i.e. constant in time) threshold. This method is a hack introduced by Hillary Scannell in `ocetrac`, which when paired with `std_normalise=True` can approximate `hobday_extreme` in very specific cases and with some caveats. N.B.: Normalising by local STD is again memory-intensive, at which point there is little gained by this approximate method.
  - `'hobday_extreme'`: Defines a local day-of-year specific threshold within a window of `window_days_hobday` (closer to the Hobday et al. (2016) definition for simple time-series). This method is more "correct", but is very computationally- & memory-demanding (requires ~5x additional scratch space). Requires the additional argument:
    - `window_days_hobday`: The window size to include in the day-of-year threshold calculation. Default is `11` days.
- `threshold_percentile`: The percentile threshold for extreme event detection. Default is `95`.
- `exact_percentile`: Whether to use exact or an approximate (PDF-based) percentile calculation. Default is `True`. N.B. Using the exact percentile calculation requires both careful/thoughtful chunking & sufficient memory, in consideration of the limitations inherent to distributed parallel I/O & processing.
- `dimensions`: The names of the time, latitude, and longitude dimensions in the data array. Default is `('time', 'lat', 'lon')`.
- `dask_chunks`: The chunk size for the output dataset. Default is `{'time': 25}`.


See, e.g. `./examples/unstructured data/01_preprocess_extremes.ipynb` for a detailed example of pre-processing on an _unstructured_ grid.

---
### 2. Identify & Track Marine Heatwaves: cf. `02_id_track_events.ipynb`
```python
import xarray as xr
import marEx


# Load Pre-processed Data
file_name = 'path/to/binary/extreme/data'
chunk_size = {'time': 25, 'lat': -1, 'lon': -1}
extremes_ds = xr.open_dataset(file_name, chunks=chunk_size)

# ID, Track, & Merge
tracker = marEx.tracker(
    extremes_ds.extreme_events, 
    extremes_ds.mask,
    area_filter_quartile=0.5,      # Remove the smallest 50% of the identified coherent extreme areas
    R_fill=8,                      # Fill small holes with radius < 8 _cells_
    T_fill=2,                      # Allow gaps of 2 days and still continue the event tracking with the same ID
    allow_merging=True,            # Allow extreme events to split/merge. Keeps track of merge events & unique IDs.
    overlap_threshold=0.5,         # Overlap threshold for merging events. If overlap < threshold, events keep independent IDs.
    nn_partitioning=True           # Use nearest-neighbor partitioning
)
extreme_events_ds, merges_ds = tracker.run(return_merges=True)
```

The resulting xarray dataset `extreme_events_ds` will have the following structure & entries:
```
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

```
where 
- `ID_field` is the binary field of tracked events,
- `global_ID` is the unique ID of each object. `global_ID.sel(ID=10)` tells you the corresponding mapped `original_id` of event ID 10 at every time,
- `area` is the area of each event as a function of time,
- `centroid` is the (x,y) centroid of each event as a function of time,
- `presence` indicates the presence of each event at each time (anywhere in space),
- `time_start` and `time_end` are the start and end times of each event,
- `merge_ledger` gives the sibling IDs (matching `ID_field`) of each merging event. Values of `-1` indicate no merging event occurred.

Additionally, if running with `return_merges=True`, the resulting xarray dataset `merges_ds` will have the following structure & entries:
```
xarray.Dataset 
Dimensions: (merge_ID, parent_idx, child_idx) 
Data variables:
    parent_IDs      (merge_ID, parent_idx)  int32       ndarray
    child_IDs       (merge_ID, child_idx)   int32       ndarray
    overlap_areas   (merge_ID, parent_idx)  int32       ndarray
    merge_time      (merge_ID)              datetime64  ndarray
    n_parents       (merge_ID)              int8        ndarray
    n_children      (merge_ID)              int8        ndarray
```
where
- `parent_IDs` and `child_IDs` are the _original_ parent and child IDs of each merging event,
- `overlap_areas` is the area of overlap between the parent and child objects in each merging event,
- `merge_time` is the time of each merging event,
- `n_parents` and `n_children` are the number of parent and child objects participating in each merging event.

Arguments for `marEx.tracker()` include: 
- `data_bin`: The binary field of events to group & label. _Must represent an underlying `dask` array_.
- `mask`: The land-sea mask to apply to the binary field, indicating points to keep.
- `area_filter_quartile`: The fraction of the smallest objects to discard, i.e. the quantile defining the smallest area object retained.
- `R_fill`: The size of the structuring element used in morphological opening & closing, relating to the largest hole that can be filled. In units of pixels.
- `T_fill`: The permissible temporal gap between objects for tracking continuity to be maintained. Default is `2` days.
- `allow_merging`:
  - `True`: (Default) Apply splitting & merging criteria, track merge events, and maintain original identities of merged objects across time.
  - `False`: Classical `ndmeasure.label` with simple time connectivity, i.e. Scannell et al. 
- `nn_partitioning`: 
  - `True`: (Default) Implement a better partitioning of merged child objects _based on closest parent cell_.
  - `False`: Use the _parent centroids_ to determine partitioning between new child objects, i.e. Di Sun & Bohai Zhang 2023. N.B.: This has major problems with small merging objects suddenly obtaining unrealistically-large (and often disjoint) fractions of the larger object.
- `overlap_threshold`: The fraction of the smaller object's area that must overlap with the larger object's area to be considered the same event and continue tracking with the same ID. Default is `0.5`.
- `timedim`, `xdim`, `ydim`: The names of the time, latitude, and longitude dimensions in the data array. Default is `('time', 'lat', 'lon')`.

See, e.g. `./examples/unstructured data/02_id_track_events.ipynb` for a detailed example of identification, tracking, & merging on an _unstructured_ grid.


## Installation & Setup

### Standard Installation
```bash
# Complete installation with performance optimisations
pip install marEx[full]

# Minimal installation (fallback if JAX unavailable)  
pip install marEx
```

### High-Performance Computing Integration

MarEx includes HPC utility functions for deployment on cloud/supercomputing environments:

```python
import marEx.helper as helper

# Automated SLURM cluster management
client = helper.start_distributed_cluster(
    n_workers=512,           # Scale to 512 workers
    workers_per_node=64,
    node_memory=512,
    runtime=120              # in minutes
)

# Automatic dashboard tunneling and monitoring
helper.get_cluster_info(client)
```

---
Please contact [Aaron Wienkers](mailto:aaron.wienkers@gmail.com) with any questions, comments, issues, or bugs.
