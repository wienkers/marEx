# Changelog

All notable changes to marEx are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [5.0.0]

### Added

- `marEx.anomaly.compute()`: a standalone anomaly stage with no threshold parameter anywhere. A smoothed daily climatology, detrended or standardised anomalies, on gridded or unstructured data, usable without ever detecting an event.
- `marEx.extremes.identify()`: standalone percentile thresholding on anomalies from marEx or from anywhere else (accepts a DataArray, or a Dataset carrying `dat_anomaly`).
- `tail="lower"` (on `identify` and `preprocess_data`) for cold spells, drought and any low-side extreme; `threshold_percentile=5, tail="lower"` flags the coldest 5 %.
- Fields with an extra dimension (for example depth or pressure level): `(time, level, lat, lon)` and `(time, level, cell)` run through the same four anomaly methods and both threshold methods, with a documented dimension contract and a slice-equivalence guarantee (a 3-D run equals the per-level 2-D runs).
- Non-daily time axes: monthly and sub-daily (for example 6-hourly, hourly) data are supported. The within-year axis is inferred (`dayofyear`, `month`, `hourofyear`) or passed as a `marEx.SeasonalCycle` via `cycle=`; `marEx.infer_cycle` is public.
- `compute_mode="persist" | "lazy" | "streaming"` and `scratch_dir=` on `preprocess_data`, `anomaly.compute` and `extremes.identify`; `compute_mode="persist" | "streaming"` on `tracker`. `streaming` keeps intermediate results on disk instead of pinning them in worker memory, so a run whose pinned data does not fit in cluster memory can still complete. `marEx.clear_staging(ds)` removes the staging directory after you have written your output.
- `tracker(..., prefilter_min_cells=N)`: drops objects smaller than N cells at each timestep before the morphology step, on both grid types (default `None` = off, output unchanged).
- `marEx.helper.ResourceMonitor`: wall time and memory per pipeline stage, with spill reported as growth during the stage.
- `tracker` and `regional_tracker` accept `mask=None` (every cell valid) and no longer need a land-sea mask on atmospheric fields.
- Data-derived histogram range for the approximate percentile path: `precision` alone sets the bin width; the binned range follows the data for any variable and units.
- A warning when a canonical-layout tile cannot fit its task budget, naming a rechunk that would.
- zarr-python 3 is supported alongside zarr 2 (`zarr>=2.18`), and current xarray releases.
- Python 3.13 in CI; dask is un-pinned (`dask[complete]>=2025.9.0`, was `==2025.3.0`).
- Logical subpackages: `marEx.anomaly`, `marEx.extremes`, `marEx.core`, `marEx.helper`, `marEx.track` (split from single modules).

### Changed

- Extreme-method names are domain neutral: `hobday_extreme` is now `seasonal_percentile`, `global_extreme` is now `global_percentile`.
- Parameter names: `window_year_baseline` -> `window_years`, `smooth_days_baseline` -> `smooth_days`, `window_days_hobday` -> `window_days`, `window_spatial_hobday` -> `window_spatial`, `std_normalise` -> `standardise`. The output attributes carry the new names.
- `fixed_baseline` and `detrend_fixed_baseline` now smooth their day-of-year climatology with a circular moving average (default `smooth_days=21`, as `shifting_baseline` always did; wraps the year, as in Hobday et al. 2016). `smooth_days=1` restores the unsmoothed climatology. Anomalies from these two methods change by default.
- Without `precision`/`max_anomaly`, the approximate percentile range and bin width are derived from the data (about 3000 bins over the tail's range) instead of a fixed `precision=0.01`, `max_anomaly=5.0`. A threshold that reaches the edge of the range regrows it automatically and recomputes.
- `tracker`'s `R_fill` and `mask` are keyword-optional in the signature (`mask=None` is valid; `R_fill` is still required and rejected by name when unset).
- Tracker merge and parent limits are record widths of 64 (were 20 merges / 10 parents), with the guard moved to fire only when a candidate actually consumes a slot.
- Outputs written to Zarr no longer inherit the input store's compressor; the writing store's default applies. Values are unchanged.
- Variable attributes on `dat_anomaly`, `extreme_events` and the tracker's time variables no longer pick up the input's attrs on current xarray; output matches xarray < 2025.11.
- `preprocessing_steps` and `window_spatial` attributes describe what actually ran: the cadence and tail appear in the step text, and `method_percentile="exact"` records no spatial window. The daily upper-tail wording is unchanged.
- The streaming staging-directory handshake lives on `ds.encoding["marex_staging_dir"]`, not `ds.attrs`, so it is not copied into files you write.
- Time chunk `"auto"` in `dask_chunks` is dask's byte budget (`array.chunk-size`), not an element budget; an integer is a step count.

### Deprecated

- `max_anomaly` and `n_bins` on `preprocess_data` / `identify` (`FutureWarning`; still honoured). `max_anomaly` pins the range; `n_bins` replaces the 3000-bin target. Use `precision`.

### Removed

- The `marEx.detect` module and the top-level `marEx.compute_normalised_anomaly`, `marEx.identify_extremes`, `marEx.rolling_climatology`, `marEx.smoothed_rolling_climatology` (no shims; the functions live under `marEx.anomaly` / `marEx.extremes`).
- The `use_temp_checkpoints` argument and the temp-checkpoint machinery in detect; `marEx.helper.checkpoint_to_zarr` and `marEx.helper.fix_dask_tuple_array`.

### Fixed

- Approximate percentiles: the 1-D (global) quantile now interpolates within the containing bin (the old interpolation branch never ran); values beyond the range are counted in the outermost bin instead of dropped; a threshold is masked only where a cell is NaN at every timestep, so cells valid for part of the year (for example under sea ice) keep a threshold.
- Exact percentiles: cells with no variance (for example permanent sea ice) no longer flag every timestep; a non-positive upper threshold is nudged to the smallest value past zero (mirrored for the lower tail).
- Histogram counts per (cell, day, bin) are no longer allowed to wrap at 65535 samples.
- Leap-year and calendar handling: `fixed_baseline` with a reference period containing no leap year no longer leaves day-of-year 366 NaN; `standardise` no longer fails on spans with no leap year; cftime / non-standard calendars work in the decimal-year step.
- A `NaT` in the time axis no longer makes a daily axis read as sub-daily.
- `detrend_harmonic` on sub-daily input now raises a clear error instead of silently leaving the diurnal cycle in the anomaly.
- A `dimensions=` mapping that names `y` but not `x` now raises instead of silently treating the longitude axis as an extra dimension.
- Tracker: the first object in raster order at the first timestep was dropped on every gridded and regional track; fixed.
- Tracker: area-filter ties are now handled identically on both grid types (objects exactly at the cutoff are kept), and the reported `accepted_area_fraction` agrees.
- Tracker: antimeridian margin scaled to the grid (grids with <= 200 longitude points mis-flagged mid-domain objects); latitude no longer wraps in the morphology padding; single-timestep centroids; radian-to-degree conversion no longer mutates the caller's array.
- Tracker: `fill_holes` padding at the periodic-longitude seam is now exact (4 x `R_fill` reach). Already in 4.1.2.
- Tracker merge loop: ID-range overrun that could silently fuse two events under one ID, off-by-one capacity guards that raised IndexError instead of a clear error, a dropped time axis for a size-1 time chunk, and property transfer that silently produced NaN.
- Unstructured tracker: events depended on the input time chunking and on worker layout (events fused or split differently). The merge loop now reruns a chunk from pristine labels until it settles, so events and merges are independent of time chunking, except for cells exactly equidistant between candidate parents.
- Unstructured tracker: two code paths that had never been executed (`partition_centroid_unstructured`, checkpoint reload after a restart) now work; the `MAX_PARENTS` guard no longer raises order-dependently.
- Tracker temp stores are unique per run; two runs sharing a scratch directory no longer corrupt each other.
- Tracker coordinate-unit auto-detection accepts both global conventions (endpoint-inclusive and exclusive) and no longer mis-reads a short 10-degree grid as radians.
- Cluster, logging and dependency probes: `has_dependency` always returned True, lazy logging configuration never fired, runtime was labelled hours when it was minutes.
- plotX: bare `.plotX()` on unstructured data, single-panel `multi_plot`, `plot_IDs=True` permanently altering the plotter's data, animation projection and coordinate handling, frame size from the projected domain, `animate` on a `where()`-cut dask coordinate.
- `to_netcdf` of outputs carrying tuple attributes. Already in 4.1.2.

### Performance

- Detect output no longer depends on how the input is chunked (time or space), for every anomaly and threshold method (verified on the test fixtures): reductions run on an internal canonical layout (time whole, spatial tiles) and the caller's layout is restored.
- Seasonal approximate percentile: the dense (cycle x bin x tile) histogram is gone; a per-cell kernel over the time-transposed bin slab completes a 0.25-degree global 40-year daily run in about an hour on one node (4 workers x 22 GB, 64 threads; single run).
- Tracker preprocessing: exact Euclidean-distance morphology and per-slice small-object labelling, frontier-queue BFS partitioners, an O(1) object-property store and a single labelling pass. The unstructured partitioning kernel, the real bottleneck there, was fixed.
- `streaming` completes the gridded track (3804 days, 0.25 degree) in a 24 GiB allocation where `persist` is OOM-killed; on the unstructured track `streaming` completed within 3 h 50 min where `persist` did not finish within 5 h at 32 GB. `streaming` reduces pinned bytes. Peak memory falls only where pinned data dominated, as on the long gridded track. Repeat counts are given where runs were repeated, and the other figures are single measurements on DKRZ Levante.
- Gridded 0.25-degree global detect (3438 x 720 x 1440): the data variables of the `streaming` output are byte-identical to `persist` (one configuration).
- Staging of the merge ledger, ID field and final relabel in the tracker; fewer whole-field pins.

## Earlier Versions

See [GitHub releases](https://github.com/wienkers/marEx/releases) for 4.1.x and earlier.
