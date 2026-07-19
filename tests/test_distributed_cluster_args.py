"""What `start_distributed_cluster` asks SLURMCluster for: worker shape and the worker-side dask config.

The worker processes are launched by SLURM from a job script, so nothing set with `dask.config.set`
in the calling process reaches them; whatever they need has to be in the job script.
"""

import dask
import pytest

import marEx.helper.cluster as cluster_mod


class _FakeCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scaled = None

    def scale(self, n):
        self.scaled = n


class _FakeClient:
    def __init__(self, cluster):
        self.cluster = cluster


_ACCOUNT_VARS = ("SLURM_ACCOUNT", "SBATCH_ACCOUNT", "SLURM_JOB_ACCOUNT", "SLURM_JOB_ID")


def _set_account_env(monkeypatch, **values):
    for name in _ACCOUNT_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def captured(monkeypatch, tmp_path):
    made = []

    def fake_cluster(**kwargs):
        made.append(_FakeCluster(**kwargs))
        return made[-1]

    monkeypatch.setattr(cluster_mod, "SLURMCluster", fake_cluster)
    monkeypatch.setattr(cluster_mod, "Client", _FakeClient)
    monkeypatch.setattr(cluster_mod, "get_cluster_info", lambda client: None)
    monkeypatch.setattr(cluster_mod, "DKRZ_LOG_PATH", tmp_path / "logs")
    _set_account_env(monkeypatch, SLURM_ACCOUNT="test_account")
    with dask.config.set({}):  # configure_dask sets global config; restore it afterwards
        yield made, tmp_path


def test_worker_shape_is_processes_times_threads(captured):
    made, tmp_path = captured
    client = cluster_mod.start_distributed_cluster(
        n_workers=96, workers_per_node=32, threads_per_worker=4, runtime=60, scratch_dir=tmp_path
    )
    kw = made[0].kwargs
    assert kw["processes"] == 32
    assert kw["cores"] == 128
    assert client.cluster.scaled == 96


def test_single_thread_workers_by_default(captured):
    made, tmp_path = captured
    cluster_mod.start_distributed_cluster(n_workers=64, workers_per_node=32, runtime=60, scratch_dir=tmp_path)
    kw = made[0].kwargs
    assert kw["processes"] == 32
    assert kw["cores"] == 32


def test_explicit_account_wins_over_the_environment(captured):
    made, tmp_path = captured
    cluster_mod.start_distributed_cluster(n_workers=4, workers_per_node=4, runtime=60, scratch_dir=tmp_path, account="mine")
    assert made[0].kwargs["account"] == "mine"


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"SLURM_ACCOUNT": "a", "SBATCH_ACCOUNT": "b", "SLURM_JOB_ACCOUNT": "c"}, "a"),
        ({"SBATCH_ACCOUNT": "b", "SLURM_JOB_ACCOUNT": "c"}, "b"),
        ({"SLURM_JOB_ACCOUNT": "c"}, "c"),
    ],
)
def test_account_falls_back_to_the_slurm_environment(captured, monkeypatch, env, expected):
    made, tmp_path = captured
    _set_account_env(monkeypatch, **env)
    cluster_mod.start_distributed_cluster(n_workers=4, workers_per_node=4, runtime=60, scratch_dir=tmp_path)
    assert made[0].kwargs["account"] == expected


def test_account_falls_back_to_the_running_job(captured, monkeypatch):
    made, tmp_path = captured
    _set_account_env(monkeypatch, SLURM_JOB_ID="123")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return cluster_mod.subprocess.CompletedProcess(cmd, 0, stdout="from_job\n", stderr="")

    monkeypatch.setattr(cluster_mod.subprocess, "run", fake_run)
    cluster_mod.start_distributed_cluster(n_workers=4, workers_per_node=4, runtime=60, scratch_dir=tmp_path)
    assert made[0].kwargs["account"] == "from_job"
    assert calls[-1][:4] == ["squeue", "-h", "-j", "123"]


@pytest.mark.parametrize(
    "returncode, stdout, expected",
    [(1, "", None), (1, "stale\n", None), (0, "", None), (0, "acct\nacct\n", "acct")],
)
def test_squeue_lookup_edge_cases(monkeypatch, returncode, stdout, expected):
    _set_account_env(monkeypatch, SLURM_JOB_ID="123")
    monkeypatch.setattr(
        cluster_mod.subprocess,
        "run",
        lambda cmd, **kwargs: cluster_mod.subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=""),
    )
    assert cluster_mod._slurm_account() == expected


def test_missing_account_is_rejected(captured, monkeypatch):
    made, tmp_path = captured
    _set_account_env(monkeypatch)
    with pytest.raises(cluster_mod.ConfigurationError, match="account"):
        cluster_mod.start_distributed_cluster(n_workers=4, workers_per_node=4, runtime=60, scratch_dir=tmp_path)
    assert made == []


def test_comm_timeouts_reach_the_worker_job_script(captured):
    made, tmp_path = captured
    cluster_mod.start_distributed_cluster(
        n_workers=32,
        workers_per_node=32,
        runtime=60,
        scratch_dir=tmp_path,
        job_script_prologue=["module load foo"],
    )
    prologue = made[0].kwargs["job_script_prologue"]
    # configure_dask sets 300 s; without the export the workers keep dask's own 30 s default.
    assert "export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=300s" in prologue
    assert "export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=300s" in prologue
    assert prologue[-1] == "module load foo"


# The cases below render the real dask-jobqueue job script (no job is submitted), so they check what the
# workers would actually run rather than the keyword arguments marEx passes on.
dask_jobqueue = pytest.importorskip("dask_jobqueue")


@pytest.fixture
def rendered(monkeypatch, tmp_path):
    scripts = []

    def render(**kwargs):
        cluster = dask_jobqueue.SLURMCluster(**kwargs)
        scripts.append(cluster.job_script())
        cluster.close()
        return _FakeCluster(**kwargs)

    monkeypatch.setattr(cluster_mod, "SLURMCluster", render)
    monkeypatch.setattr(cluster_mod, "Client", _FakeClient)
    monkeypatch.setattr(cluster_mod, "get_cluster_info", lambda client: None)
    monkeypatch.setattr(cluster_mod, "DKRZ_LOG_PATH", tmp_path / "logs")
    _set_account_env(monkeypatch, SLURM_ACCOUNT="test_account")
    with dask.config.set({}):

        def start(**kwargs):
            cluster_mod.start_distributed_cluster(
                n_workers=4, workers_per_node=4, runtime=60, scratch_dir=tmp_path, dashboard_address=0, **kwargs
            )
            return [line for line in scripts[-1].splitlines() if not line.startswith("#")]

        yield start


def test_rendered_script_shape_and_exports(rendered):
    lines = rendered(threads_per_worker=8)
    worker = next(line for line in lines if "dask_worker" in line)
    assert "--nthreads 8" in worker and "--nworkers 4" in worker
    assert lines.index("export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=300s") < lines.index(worker)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_deprecated_env_extra_still_reaches_the_script(rendered):
    assert "module load legacy" in rendered(env_extra=["module load legacy"])


def test_jobqueue_config_prologue_still_reaches_the_script(rendered):
    with dask.config.set({"jobqueue.slurm.job-script-prologue": ["module load from-config"]}):
        assert "module load from-config" in rendered()


def test_prologue_none_and_str(rendered):
    assert "export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=300s" in rendered(job_script_prologue=None)
    assert "module load foo" in rendered(job_script_prologue="module load foo")


def test_threads_per_worker_below_one_is_rejected(captured):
    _, tmp_path = captured
    with pytest.raises(cluster_mod.ConfigurationError):
        cluster_mod.start_distributed_cluster(n_workers=4, workers_per_node=4, threads_per_worker=0, scratch_dir=tmp_path)
