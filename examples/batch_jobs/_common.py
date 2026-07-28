"""
Shared helpers for run_detect.py and run_track.py: read a run configuration, open the input and start a Dask cluster.

A configuration is one YAML file (see configs/). Strings may use environment variables and `~`, which are expanded when
the file is read, so one file serves every user. The optional `smoke:` section is merged over the rest of the file when
a script runs with --smoke.
"""

import copy
import os
from pathlib import Path

import xarray as xr
import yaml

import marEx.helper as hpc


def _expand(value):
    """Expand environment variables and `~` in every string of a nested structure."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _strings(value):
    """Yield every string in a nested structure."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _merge(base, override):
    """Recursively merge `override` into a copy of `base`; dicts merge, everything else is replaced."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path, smoke=False):
    """Read a YAML run configuration, apply its `smoke:` overrides if asked, and expand paths."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    overrides = cfg.pop("smoke", None) or {}
    if smoke:
        cfg = _merge(cfg, overrides)
    cfg = _expand(cfg)
    unset = sorted({part for s in _strings(cfg) for part in s.split("/") if part.startswith("$")})
    if unset:
        raise SystemExit(f"{path}: environment variable(s) not set: {', '.join(unset)}")
    for key in ("output_dir", "scratch_dir"):
        if key not in cfg:
            raise SystemExit(f"{path}: `{key}` is required")
        cfg[key] = Path(cfg[key])
    return cfg


def open_input(spec):
    """
    Open the input field described by an `input:` section.

    `path` is a zarr store or a netCDF file / glob; alternatively `intake: {catalog, entry, kwargs}` reads an intake
    catalogue entry (`entry` is the list of keys from the catalogue root). `variable` picks the field, then the optional
    `drop_vars`, `isel`, `sel` (a [start, stop] pair is a range), `time` (a [start, end] pair) and `dtype` are applied, and finally
    the `chunks` rechunk.
    """
    if "intake" in spec:
        import intake

        node = intake.open_catalog(spec["intake"]["catalog"])
        for key in spec["intake"]["entry"]:
            node = node[key]
        kwargs = spec["intake"].get("kwargs", {})
        # intake 0.x sources open with .to_dask(); the newer reader-based entries with .reader(...).read()
        ds = node.reader(**kwargs).read() if hasattr(node, "reader") else node(**kwargs).to_dask()
    elif str(spec["path"]).endswith(".zarr"):
        ds = xr.open_zarr(spec["path"], chunks={})  # the store's own chunks; rechunked below
    else:
        ds = xr.open_mfdataset(spec["path"], chunks={})
    da = ds[spec["variable"]]
    if "drop_vars" in spec:
        da = da.drop_vars(spec["drop_vars"], errors="ignore")
    if "isel" in spec:
        da = da.isel(spec["isel"])
    if "sel" in spec:
        # a [start, stop] pair selects a range (a slice); any other value selects by label
        da = da.sel({k: slice(*v) if isinstance(v, list) else v for k, v in spec["sel"].items()})
    if "time" in spec:
        da = da.sel(time=slice(*spec["time"]))
    if "dtype" in spec:
        da = da.astype(spec["dtype"])
    if "chunks" in spec:
        da = da.chunk(spec["chunks"])
    return da


def open_grid(spec):
    """
    Read the neighbour indices and cell areas of an unstructured mesh from a `grid:` section
    (`file`, `rename`, `neighbours`, `cell_areas` variable names). Returns (neighbours, cell_areas).
    """
    grid = xr.open_dataset(spec["file"], chunks={})
    rename = spec.get("rename", {})
    grid = grid.rename({k: v for k, v in rename.items() if k in grid.dims or k in grid.variables})
    return grid[spec["neighbours"]], grid[spec["cell_areas"]]


def start_cluster(spec, scratch_dir):
    """
    Start the Dask cluster described by a `cluster:` section and return (client, monitor).

    `type: local` starts the workers inside this job (start_local_cluster). `memory_limit` is required there because
    on an HPC node Dask sizes workers from the whole node's memory, not from the job's allocation.
    `type: slurm` launches the workers as their own SLURM jobs (start_distributed_cluster), which needs an account:
    `account:` here, or $SLURM_ACCOUNT.
    """
    spec = dict(spec)
    kind = spec.pop("type")
    dask_dir = Path(scratch_dir) / "dask"
    if kind == "local":
        if "memory_limit" not in spec:
            raise SystemExit("cluster.memory_limit is required for a local cluster")
        client = hpc.start_local_cluster(scratch_dir=dask_dir, **spec)
    elif kind == "slurm":
        timeout = spec.pop("wait_timeout", 1800)
        client = hpc.start_distributed_cluster(scratch_dir=dask_dir, **spec)
        client.wait_for_workers(spec["n_workers"], timeout=timeout)
    else:
        raise SystemExit(f"cluster.type must be 'local' or 'slurm', not {kind!r}")
    # Records wall time, peak memory, spill and worker restarts for each `monitor.stage(...)` block
    return client, hpc.ResourceMonitor(client)
