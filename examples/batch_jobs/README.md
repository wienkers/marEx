# Batch Jobs

A template for running marEx detection and tracking as SLURM jobs, without Jupyter. Two driver scripts read one YAML
configuration per dataset; to run marEx on your own data, copy a config and edit it. The scripts themselves need no
changes.

| File | What it does |
| --- | --- |
| `run_detect.py` | opens the input, starts a Dask cluster, runs each configured detection with `marEx.preprocess_data` and writes it to zarr |
| `run_track.py` | reads a detection output and runs each configured tracker (`marEx.tracker` or `marEx.regional_tracker`), writing the events to zarr and the merge records to netCDF |
| `_common.py` | config loading, input and grid reading, cluster start-up (shared by both scripts) |
| `detect.sbatch`, `track.sbatch` | SLURM job templates for the two scripts |
| `configs/gridded.yaml` | global daily OSTIA SST, 0.25°: the settings of `gridded/01` and `gridded/02` |
| `configs/regional.yaml` | OSTIA at 0.05°, cut to Europe, with `regional_tracker`: the settings of `regional/01` and `regional/02` |
| `configs/unstructured.yaml` | ICON-ESM-ER SST on the native 14.9 M-cell mesh, with its grid file: the settings of `unstructured/01` and `unstructured/02` |

## Running

```bash
export MAREX_SCRATCH=/scratch/${USER:0:1}/$USER   # used by the example configs for every path
export MAREX_PYTHON=/path/to/env/bin/python       # an environment with marEx installed (default: python on PATH)
cd examples/batch_jobs

sbatch --account=<project> detect.sbatch configs/gridded.yaml
sbatch --account=<project> track.sbatch configs/gridded.yaml     # once detection has finished
```

Both scripts take `--smoke`, which merges the config's `smoke:` section over the rest of the file (a short period, a
small local cluster and a separate output directory), and `--only NAME ...` to run a subset of the configured runs.
A smoke run checks the plumbing in minutes. Run it before the full job, and submit it with an allocation that fits
its local cluster:

```bash
sbatch --account=<project> --cpus-per-task=16 --mem=64G detect.sbatch configs/gridded.yaml --smoke
sbatch --account=<project> --partition=shared --cpus-per-task=4 --mem=16G track.sbatch configs/gridded.yaml --smoke
```

The scripts also run directly (`python run_detect.py configs/gridded.yaml --smoke`) inside any allocation.

## Writing a Config

```yaml
output_dir: ...        # detect writes here; track reads its input from and writes its events to here
scratch_dir: ...       # Dask spill (dask/) and compute_mode="streaming" staging (staging/)
dask_config: {...}     # optional, passed to dask.config.set

input:                 # the field to detect extremes in
  path: ...            # a zarr store, or a netCDF file / glob (or `intake: {catalog, entry, kwargs}`)
  variable: sst
  drop_vars: [...]     # optional, then isel / sel / time / dtype, in that order
  sel: {lat: [30, 66]} #   a [start, stop] pair selects a range
  time: [1982-01-01, 2022-05-31]
  chunks: {...}
dimensions: {time: time, x: lon, y: lat}   # passed to preprocess_data; no "y" means an unstructured mesh
coordinates: {...}                          # unstructured only
grid: {file, rename, neighbours, cell_areas}  # unstructured only: neighbour indices and cell areas

detect:
  cluster: {type: slurm | local, ...}   # start_distributed_cluster / start_local_cluster arguments
  runs:
    - name: ...
      output: ....zarr
      kwargs: {...}                     # passed to marEx.preprocess_data unchanged

track:
  input: ....zarr      # a detect output in output_dir
  time: [...]          # optional sub-period
  chunks: {...}        # space must stay whole; chunk time
  mask_lat: [lo, hi]   # optional: drop cells outside this latitude band from the mask
  cluster: {type: local, n_workers, threads_per_worker, memory_limit}
  runs:
    - name: ...
      tracker: global | regional
      output: ....zarr
      merges: ....nc   # optional: also write the merge records
      kwargs: {...}    # passed to the tracker unchanged; compute_mode defaults to "streaming"

smoke: {...}           # merged over the rest by --smoke (dicts merge, lists and values replace)
```

Strings may use environment variables and `~`; a variable that is not set stops the script with its name. Any
parameter of `preprocess_data` or the tracker goes under `kwargs`, so a new method needs a new config, not a new
script. The parameters are documented in the [API reference](https://marex.readthedocs.io/en/latest/api/index.html).

Three rules that the configs follow, and that a new config should too:

- **Pass `memory_limit` for a local cluster.** On an HPC node Dask sizes workers from the whole node's memory, not from
  the job's allocation. `run_*.py` refuses a local cluster without it.
- **Chunking.** For detection on a lat/lon grid keep space whole and chunk time; on an unstructured mesh chunk space
  (see `configs/unstructured.yaml`). The tracker always needs space whole and is chunked in time.
- **Streaming staging outlives the call.** Both scripts clear it after writing each output, but a job killed mid-run
  leaves it in `scratch_dir/staging`: sweep that directory after a failed job.

## Cluster Shapes

`cluster.type: slurm` runs `run_detect.py` as a driver and launches the workers as their own short SLURM jobs on the
compute partition, so `detect.sbatch` asks only for a few cores. The worker walls (`runtime`, minutes) must cover the
whole detection, because dask's P2P rechunk does not survive a worker leaving mid-shuffle. `cluster.type: local` starts
the workers inside the job. Size the allocation to `n_workers x memory_limit` plus a few GB for the client.

The partitions and node sizes in the templates and configs are those of DKRZ Levante (256 GB nodes, `shared` and
`compute` partitions). On another cluster, change the `#SBATCH` lines, `node_memory` and `workers_per_node`. For the
timings of the full runs, see the stage summaries at the end of each example notebook.
