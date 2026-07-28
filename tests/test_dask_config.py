"""The worker memory thresholds `configure_dask` applies.

A worker whose unmanaged memory alone sits above `pause` can no longer spill back down, so it stops taking
work for good and the cluster hangs without an error. marEx once paused at 0.6 and terminated at 0.8, which
put that line under the ordinary transient of a large tile. The local cluster now runs dask's own
thresholds, the ones SLURM-launched workers (which never see this process's config) already use.
"""

import copy
from pathlib import Path

import dask
import distributed
import yaml

from marEx.helper import start_local_cluster
from marEx.helper.dask_config import DEFAULT_DASK_CONFIG

FRACTIONS = ("target", "spill", "pause", "terminate")


def _dask_defaults():
    shipped = yaml.safe_load((Path(distributed.__file__).parent / "distributed.yaml").read_text())
    return {name: shipped["distributed"]["worker"]["memory"][name] for name in FRACTIONS}


def test_memory_thresholds_are_dasks_own():
    ours = {name: DEFAULT_DASK_CONFIG[f"distributed.worker.memory.{name}"] for name in FRACTIONS}
    assert ours == _dask_defaults()


def test_local_cluster_workers_run_with_them(tmp_path):
    # configure_dask sets the global config key by key, which `with dask.config.set({})` would not undo
    saved = copy.deepcopy(dask.config.config)
    try:
        client = start_local_cluster(n_workers=1, threads_per_worker=1, memory_limit="1GB", scratch_dir=tmp_path)
        try:
            seen = client.run(lambda: {name: dask.config.get(f"distributed.worker.memory.{name}") for name in FRACTIONS})
        finally:
            client.close()
    finally:
        dask.config.config.clear()
        dask.config.config.update(saved)
    assert list(seen.values()) == [_dask_defaults()]
