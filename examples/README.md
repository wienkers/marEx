# marEx Examples

Every notebook runs the same detect → track → visualise chain, and only the input and a few grid-specific arguments
change. The notebooks are committed with their outputs, and each one prints the wall time and peak memory of every stage
on the cluster it ran on, copied into the table below. The 3-D notebook covers the detect step only. The inputs live on
DKRZ's Levante (OSTIA, EERIE ICON and FESOM output and ERA5 in the DKRZ pool), so outside Levante each notebook needs its
input path changed.

## Notebooks

| Directory | Notebook | Input | Cluster | Measured stages |
| --- | --- | --- | --- | --- |
| `gridded/` | `01_preprocess_extremes` | OSTIA daily SST, 0.25° global, 1982-2022 | 256 SLURM workers × 7.8 GB | 3.5 min shifting baseline + Hobday, 0.8 min harmonic + global |
| | `02_id_track_events` | output of 01 | 1 node, 16 workers × 14 GB | 41.2 min tracking with merging, 1.8 min basic |
| | `03_visualise_events` | outputs of 01-02 | 1 node, 32 workers × 7 GB | 2.2-5.1 min per movie |
| `regional/` | `01_preprocess_extremes` | OSTIA over Europe, 1982-2022 | 256 SLURM workers × 7.8 GB | 2.9 min shifting baseline + Hobday, 0.9 min harmonic + global |
| | `02_id_track_events` | output of 01 | 4 workers × 12 GB | 15.3 min tracking with merging, 3.5 min basic |
| | `03_visualise_events` | outputs of 01-02 | 1 node, 32 workers × 7 GB | 0.8-3.2 min per movie |
| `unstructured/` | `01_preprocess_extremes` | ICON-O R02B09 (14.9 M cells), 1985-2014 | 48 SLURM workers × 31 GB | 36.6 min shifting baseline + Hobday |
| | `02_id_track_events` | output of 01, year 2014 | 1 node, 8 workers × 28 GB | 38.7 min nearest-neighbour partitioning, 31.4 min centroid |
| | `03_visualise_events` | outputs of 01-02 | 1 node, 32 workers × 7 GB | 2.5-2.8 min per movie |
| `3d/` | `01_preprocess_extremes` | FESOM daily ocean temperature on 7 levels (2.5-1415 m), North Atlantic, 1950-2050 | 256 SLURM workers × 7.8 GB | 7.2 min shifting baseline + Hobday |
| `applications/` | `wind_drought_europe` | ERA5 daily wind capacity factor over Europe, 1980-2024 | 8 workers × 6 GB | 1.6 min detect (lower tail), 4.5 min regional tracking |

## Scripts

- **`batch_jobs/`**: the gridded detect and track notebooks as plain Python scripts with SLURM wrappers, for running the
  same workflow without Jupyter. See its [README](batch_jobs/README.md).
- **`larger-than-memory/`**: the measurement harness behind the `compute_mode="streaming"` claims in the performance
  guide, with the numbers and how to reproduce them. See its [README](larger-than-memory/README.md).
- **`applications/prepare_era5_wind.sh`**: builds the daily wind input for the wind-drought notebook from ERA5 GRIB. See
  [applications/README.md](applications/README.md).
