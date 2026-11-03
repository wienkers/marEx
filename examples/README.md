# marEx Examples

Most notebooks run the same detect → track → visualise chain on their own input, with a few grid-specific arguments, and print the wall time and peak memory of every stage. The table below collects them with the cluster each ran on. The `4d/` notebooks download
the input, detect at depth, track the events in 4-D as volumes and visualise them.

## Notebooks

| Directory | Notebook | Input | Cluster | Measured stages |
| --- | --- | --- | --- | --- |
| `gridded/` | `01_preprocess_extremes` | OSTIA daily SST, 0.25° global, 1982-2022 | 512 SLURM workers × 7.8 GB | 3.5 min shifting baseline + Hobday, 0.8 min harmonic + global |
| | `02_id_track_events` | output of 01 | 1 node, 16 workers × 14 GB | 41.2 min tracking with merging, 1.8 min basic |
| | `03_visualise_events` | outputs of 01-02 | 1 node, 32 workers × 7 GB | 2.2-5.1 min per movie |
| `regional/` | `01_preprocess_extremes` | OSTIA over Europe, 1982-2022 | 512 SLURM workers × 7.8 GB | 2.9 min shifting baseline + Hobday, 0.9 min harmonic + global |
| | `02_id_track_events` | output of 01 | 4 workers × 12 GB | 15.3 min tracking with merging, 3.5 min basic |
| | `03_visualise_events` | outputs of 01-02 | 1 node, 32 workers × 7 GB | 0.8-3.2 min per movie |
| `unstructured/` | `01_preprocess_extremes` | ICON-O R02B09 (14.9 M cells), 1985-2014 | 48 SLURM workers × 31 GB | 36.6 min shifting baseline + Hobday |
| | `02_id_track_events` | output of 01, year 2014 | 1 node, 8 workers × 28 GB | 38.7 min nearest-neighbour partitioning, 31.4 min centroid |
| | `03_visualise_events` | outputs of 01-02 | 1 node, 32 workers × 7 GB | 2.5-2.8 min per movie |
| `4d/` | `00_download_glorys12` | GLORYS12V1 daily thetao, 1/12°, North-East Pacific (30-60°N, 170-120°W), 36 levels to 1150 m, 1993-2022 | 8 CPUs × 12 GB, own environment | about 16 min per year |
| | `01_preprocess_extremes` | output of 00, kept 2013-16 | 256 SLURM workers × 7.8 GB | 26.9 min fixed baseline + Hobday |
| | `02_track_extremes_4d` | output of 01 | 1 node, 8 workers × 6 GB (the timing is from 8 × 8 GB) | 74.7 min tracking with merging |
| | `03_visualise_events` | outputs of 01-02 | 1 node, 16 workers × 14 GB | 3.1 min isosurface movie, 0.2 min footprint movie |
| `applications/` | `wind_drought_europe` | ERA5 daily wind capacity factor over Europe, 1980-2024 | 8 workers × 6 GB | 1.6 min detect (lower tail), 4.5 min regional tracking |
| `larger-than-memory/` | `larger_than_memory` | IFS-FESOM2-SR 6-hourly SST, 0.25° global, 10 and 20 years | 32 GiB job, 3 workers × 9 GB | 63-192 min detect, 57-139 min track per batch run (the notebook reads their summaries) |

## Paths

The tutorials write output under `scratch_dir/marEx/examples/`. To run them on another system, change `scratch_dir` and the input paths in their setup cells. The input sources are:

- **gridded**: OSTIA daily SST (CMEMS OSTIA L4), regridded to 0.25° and stored as `ostia/ostia.zarr`.
- **regional**: the same OSTIA product at full resolution, read through the public EERIE intake catalogue.
- **unstructured**: ICON-ESM-ER ocean output (`hist-1950`), read through the EERIE intake catalogue.
- **4d**: GLORYS12V1 `thetao` from Copernicus Marine, downloaded by `4d/00_download_glorys12.ipynb` into
  `input/glorys12_nepacific/` as one zarr store per year, plus `glorys12_static.zarr` and `glorys12_valid.zarr`.
- **applications**: ERA5 wind, converted by `applications/prepare_era5_wind.sh`.

## Scripts

- **`batch_jobs/`**: a template for running detection and tracking as SLURM jobs: two driver scripts configured by one
  YAML file per dataset, with worked configs for the gridded, regional and unstructured examples. See
  its [README](batch_jobs/README.md).
- **`larger-than-memory/`**: detection and tracking of 20 years of 6-hourly global SST inside one 32 GiB job,
  comparing `compute_mode="streaming"` with `"persist"` by the memory and wall time of each run. See its [README](larger-than-memory/README.md).
- **`applications/prepare_era5_wind.sh`**: builds the daily wind input for the wind-drought notebook from ERA5 GRIB. See
  [applications/README.md](applications/README.md).
