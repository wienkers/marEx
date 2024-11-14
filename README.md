spot_the_blOb
==============================

Efficient and scalable Marine Heatwave detection and tracking.

## Features

### Data Pre-processing
**Detrending & Anomaly Detection**:
  - Removes trend and seasonal cycle using a 6-coefficient model (mean, trend, annual & semi-annual harmonics).
  - (Optional) Normalises anomalies using a 30-day rolling standard deviation.
  - Identifies extreme events based on a percentile threshold.
  - Utilises `dask` for efficient parallel computation and scaling to very large datasets.

### Blob Detection & Tracking
**Blob Detection**:
  - Implements efficient algorithms for blob detection in 2D geographical data.
  - Fully-parallelised workflow built on `dask` for fast & larger-than-memory computation.
  - Uses morphological opening & closing to fill small holes and gaps in binary features.
  - Filters out small objects based on area thresholds.
  - Identifies and labels connected regions in binary data.

**Blob Tracking**:
  - Implements strict event tracking conditions to avoid very few very large blobs.
  - Requires blobs to overlap by at least a certain fraction of the smaller blob's area to be considered the same event and continue tracking.
  - Accounts for & keeps a history of blob splitting & merging, ensuring blobs are more coherent and retain their previous identities.
  - Implements the splitting and merging logic of [Sun et al. (2023)](https://doi.org/10.1038/s41561-023-01325-w):
    - The merged blob is split by locality to the respective parent centroids.

### Visualisation
**Plotting**:
  - Provides a few helper functions to create pretty plots and wrapped subplots.


## Usage

### Pre-process SST Data: cf. `01_preprocess_extremes.ipynb`
```python
import xarray as xr
import dask
import spot_the_blOb.hot_to_blOb as hot

# Load SST data & rechunk for optimal processing
file_name = 'path/to/sst/data'
sst = xr.open_dataset(file_name).sst
sst_rechunk = hot.rechunk_for_cohorts(sst)

# Process Data
extreme_events_ds = hot.preprocess_data(sst_rechunk, threshold_percentile=95)
```

### Identify & Track Marine Heatwaves: cf. `02_id_track_events.ipynb`
```python
import xarray as xr
import dask
import spot_the_blOb as blob


# Load Pre-processed Data
file_name = 'path/to/binary/extreme/data'
chunk_size = {'time': 25, 'lat': -1, 'lon': -1}
ds_hot = xr.open_dataset(file_name, chunks=chunk_size)

# Extract Extreme Binary Features and Modify Mask
extreme_bin = ds_hot.dat_stn
mask = ds_hot.mask.where((ds_hot.lat < 85) & (ds_hot.lat > -90), other=False)

# Spot the Blobs
tracker = blob.Spotter(extreme_bin, mask, R_fill=8, area_filter_quartile=0.5, allow_merging=True, overlap_threshold=0.5)
blobs = tracker.run()
```


## Installation

To install the required dependencies, you can use `pip`:

```bash
pip install -r requirements.txt
```

Please contact [Aaron Wienkers](mailto:aaron.wienkers@usys.ethz.ch) with any questions, comments, issues, or bugs.