"""Cluster helpers and the test-suite's own cluster fixtures.

``start_local_cluster`` logs the per-worker memory limit the workers actually get and runs dask's
own memory thresholds; ``start_distributed_cluster`` asks SLURMCluster for the right worker shape and
account, and puts the worker-side dask config into the job script (SLURM-launched workers never see
this process's ``dask.config``); ``ResourceMonitor`` records per-stage wall time, memory and restarts.
The last group checks that the shared fixtures see what they claim to see: a "zero memory warnings"
assertion means nothing unless a warning raised on a real worker would have been recorded.

``configure_dask`` sets the global dask config key by key, which ``with dask.config.set({}):`` does
not undo, so every test that reaches it restores a deep copy of the whole config.
"""

import contextlib
import copy
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import dask
import dask.array
import distributed
import numpy as np
import pytest
import yaml
from distributed import Client, LocalCluster

import marEx
import marEx.helper as hpc
import marEx.helper.cluster as cluster_mod
import marEx.logging_config as lc
from marEx.helper import dask_config, start_local_cluster
from marEx.helper.dask_config import DEFAULT_DASK_CONFIG
from marEx.helper.resources import _parse_slurm_mem

from .conftest import (
    MEMORY_LOGGERS,
    N_CLUSTER_WORKERS,
    assert_no_failures,
    capture_memory_warnings,
    check,
    on_client,
    worker_memory_limit,
)

FRACTIONS = ("target", "spill", "pause", "terminate")
_ACCOUNT_VARS = ("SLURM_ACCOUNT", "SBATCH_ACCOUNT", "SLURM_JOB_ACCOUNT", "SLURM_JOB_ID")


@contextlib.contextmanager
def _dask_config_restored():
    """Put back a deep copy of the whole global dask config when the block ends."""
    saved = copy.deepcopy(dask.config.config)
    try:
        yield
    finally:
        dask.config.config.clear()
        dask.config.config.update(saved)


# Every logger ``_configure_external_loggers`` (run by ``configure_logging``) sets a level on.
EXTERNAL_LOGGERS = (
    "distributed.scheduler",
    "distributed.worker",
    "distributed.shuffle._scheduler_plugin",
    "distributed.comm",
    "distributed.core",
    "tornado.access",
    "asyncio",
    "matplotlib.font_manager",
    "PIL.PngImagePlugin",
) + MEMORY_LOGGERS


def _logging_state():
    root = logging.getLogger("marEx")
    external = {name: logging.getLogger(name).level for name in EXTERNAL_LOGGERS}
    return (lc._current_verbosity_level, lc._logger_configured, root.level, root.propagate, list(root.handlers), external)


@pytest.fixture(autouse=True)
def _logging_state_restored():
    """``configure_logging`` (called by the cluster helpers and the fixture checks) changes module-global state.

    Only this process's state is put back: ``client.run(marEx.configure_logging)`` reconfigures the workers.
    """
    saved = _logging_state()
    yield
    root = logging.getLogger("marEx")
    lc._current_verbosity_level, lc._logger_configured = saved[0], saved[1]
    root.setLevel(saved[2])
    root.propagate = saved[3]
    root.handlers = list(saved[4])
    for name, level in saved[5].items():
        logging.getLogger(name).setLevel(level)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_local_cluster_logs_memory_limit_and_resources(tmp_path, monkeypatch):
    """With the cluster mocked out: the memory_limit passed on is the one logged, its absence is said plainly,
    and verbose=True adds the system-resources record."""
    monkeypatch.setattr(cluster_mod, "LocalCluster", mock.MagicMock())
    monkeypatch.setattr(cluster_mod, "Client", mock.MagicMock())
    monkeypatch.setattr(cluster_mod, "get_cluster_info", lambda client: None)
    logger = logging.getLogger(cluster_mod.logger.name)

    def run(**kwargs):
        handler = _ListHandler()
        old_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            with _dask_config_restored():
                cluster_mod.start_local_cluster(n_workers=2, threads_per_worker=1, scratch_dir=tmp_path, **kwargs)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
        return handler.records

    def with_limit():
        lines = [r.getMessage() for r in run(memory_limit="3GB") if "per worker" in r.getMessage().lower()]
        assert lines == ["Memory limit per worker: 3.00 GB"], lines
        assert cluster_mod.LocalCluster.call_args.kwargs["memory_limit"] == "3GB"

    def without_limit():
        (line,) = [r.getMessage() for r in run() if "per worker" in r.getMessage().lower()]
        assert "Dask default" in line and "pass memory_limit" in line, line

    def verbose_system_resources():
        lc.configure_logging(quiet=True)
        records = run(verbose=True)
        assert lc.is_verbose_mode(), "start_local_cluster(verbose=True) did not set verbose mode"
        resources = [r for r in records if r.getMessage().startswith("System resources - CPUs:")]
        assert len(resources) == 1 and resources[0].levelno == logging.DEBUG, [r.getMessage() for r in records]

    failures = []
    for row in (with_limit, without_limit, verbose_system_resources):
        check(failures, row.__name__, row)
    assert_no_failures(failures)


def _dask_defaults():
    shipped = yaml.safe_load((Path(distributed.__file__).parent / "distributed.yaml").read_text())
    return {name: shipped["distributed"]["worker"]["memory"][name] for name in FRACTIONS}


def test_local_cluster_runs_dask_memory_thresholds(tmp_path):
    """The worker memory fractions are dask's shipped ones, and a local cluster's workers run with them.

    A worker whose unmanaged memory alone sits above ``pause`` can no longer spill back down, so it stops
    taking work for good and the cluster hangs without an error. SLURM-launched workers, which never see
    this process's config, already use dask's own thresholds.
    """
    ours = {name: DEFAULT_DASK_CONFIG[f"distributed.worker.memory.{name}"] for name in FRACTIONS}
    assert ours == _dask_defaults()
    with _dask_config_restored():
        client = start_local_cluster(n_workers=1, threads_per_worker=1, memory_limit="1GB", scratch_dir=tmp_path)
        cluster = client.cluster
        try:
            seen = client.run(lambda: {name: dask.config.get(f"distributed.worker.memory.{name}") for name in FRACTIONS})
        finally:
            # Close the cluster too: left open, it outlives tmp_path and logs errors at session end.
            client.close()
            cluster.close()
    assert list(seen.values()) == [_dask_defaults()]


def test_default_scratch_falls_back_to_tempdir_off_dkrz(tmp_path, monkeypatch):
    """Off DKRZ the default scratch root is the system temp dir, never a path under a missing /scratch."""
    unreachable = Path("/nonexistent-marex-root")
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    monkeypatch.setattr(dask_config, "DKRZ_SCRATCH_PATH", unreachable / "x" / "clients")
    monkeypatch.setattr(tempfile, "tempdir", str(fallback))

    resolved = dask_config._resolve_scratch_path(None)

    assert resolved == fallback
    assert not unreachable.exists()


def test_default_scratch_is_the_dkrz_path_where_its_root_exists(tmp_path, monkeypatch):
    """Where the DKRZ path's top-level root exists, the default scratch root is that path, not the temp dir.

    The path is moved under ``tmp_path``, whose top-level root exists on any machine, so this runs off DKRZ too.
    """
    dkrz = tmp_path / "u" / "clients"
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    monkeypatch.setattr(dask_config, "DKRZ_SCRATCH_PATH", dkrz)
    monkeypatch.setattr(tempfile, "tempdir", str(fallback))
    assert Path(*dkrz.parts[:2]).exists()

    assert dask_config._resolve_scratch_path(None) == dkrz
    assert dask_config._resolve_scratch_path(tmp_path / "given") == tmp_path / "given"


def test_parse_slurm_memory():
    """SLURM memory strings are binary units; an unparseable one is None."""
    assert _parse_slurm_mem("15040") == 15040 * 2**20
    assert _parse_slurm_mem("15040M") == 15040 * 2**20
    assert _parse_slurm_mem("96G") == 96 * 2**30
    assert _parse_slurm_mem("garbage") is None


class _FakeCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scaled = None

    def scale(self, n):
        self.scaled = n


class _FakeClient:
    def __init__(self, cluster):
        self.cluster = cluster


def _set_account_env(monkeypatch, **values):
    for name in _ACCOUNT_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """``start_distributed_cluster`` with SLURMCluster and Client replaced: yields the clusters it asked for."""
    made = []

    def fake_cluster(**kwargs):
        made.append(_FakeCluster(**kwargs))
        return made[-1]

    monkeypatch.setattr(cluster_mod, "SLURMCluster", fake_cluster)
    monkeypatch.setattr(cluster_mod, "Client", _FakeClient)
    monkeypatch.setattr(cluster_mod, "get_cluster_info", lambda client: None)
    monkeypatch.setattr(cluster_mod, "DKRZ_LOG_PATH", tmp_path / "logs")
    _set_account_env(monkeypatch, SLURM_ACCOUNT="test_account")
    with _dask_config_restored():
        yield made, tmp_path


def test_slurm_worker_shape(captured):
    """processes x threads per node, single-threaded workers by default, threads < 1 rejected, and the global
    dask config that ``configure_dask`` changes is put back by the fixture's restore."""
    made, tmp_path = captured
    failures = []

    def processes_times_threads():
        made.clear()
        client = cluster_mod.start_distributed_cluster(
            n_workers=96, workers_per_node=32, threads_per_worker=4, runtime=60, scratch_dir=tmp_path
        )
        kw = made[0].kwargs
        assert (kw["processes"], kw["cores"], client.cluster.scaled) == (32, 128, 96), kw

    def single_thread_by_default():
        made.clear()
        cluster_mod.start_distributed_cluster(n_workers=64, workers_per_node=32, runtime=60, scratch_dir=tmp_path)
        kw = made[0].kwargs
        assert (kw["processes"], kw["cores"]) == (32, 32), kw

    def threads_below_one_rejected():
        made.clear()
        with pytest.raises(cluster_mod.ConfigurationError):
            cluster_mod.start_distributed_cluster(n_workers=4, workers_per_node=4, threads_per_worker=0, scratch_dir=tmp_path)
        assert made == []

    def global_dask_config_restored():
        before = copy.deepcopy(dask.config.config)
        with _dask_config_restored():
            cluster_mod.start_distributed_cluster(n_workers=4, workers_per_node=4, runtime=60, scratch_dir=tmp_path)
            assert dask.config.config != before, "precondition: start_distributed_cluster changed the global dask config"
        assert dask.config.config == before, "the global dask config was not restored"

    for row in (processes_times_threads, single_thread_by_default, threads_below_one_rejected, global_dask_config_restored):
        check(failures, row.__name__, row)
    assert_no_failures(failures)


def test_slurm_account_resolution(captured, monkeypatch):
    """Explicit account > SLURM_ACCOUNT > SBATCH_ACCOUNT > SLURM_JOB_ACCOUNT > the running job's (squeue) > error."""
    made, tmp_path = captured
    failures = []

    def start(**kwargs):
        made.clear()
        cluster_mod.start_distributed_cluster(n_workers=4, workers_per_node=4, runtime=60, scratch_dir=tmp_path, **kwargs)
        return made[0].kwargs["account"]

    def explicit_wins():
        _set_account_env(monkeypatch, SLURM_ACCOUNT="test_account")
        assert start(account="mine") == "mine"

    env_rows = [
        ({"SLURM_ACCOUNT": "a", "SBATCH_ACCOUNT": "b", "SLURM_JOB_ACCOUNT": "c"}, "a"),
        ({"SBATCH_ACCOUNT": "b", "SLURM_JOB_ACCOUNT": "c"}, "b"),
        ({"SLURM_JOB_ACCOUNT": "c"}, "c"),
    ]

    def from_environment(env, expected):
        _set_account_env(monkeypatch, **env)
        assert start() == expected

    def from_running_job():
        _set_account_env(monkeypatch, SLURM_JOB_ID="123")
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return cluster_mod.subprocess.CompletedProcess(cmd, 0, stdout="from_job\n", stderr="")

        with monkeypatch.context() as m:
            m.setattr(cluster_mod.subprocess, "run", fake_run)
            assert start() == "from_job"
        assert calls[-1][:4] == ["squeue", "-h", "-j", "123"], calls

    squeue_rows = [(1, "", None), (1, "stale\n", None), (0, "", None), (0, "acct\nacct\n", "acct")]

    def squeue_outcome(returncode, stdout, expected):
        _set_account_env(monkeypatch, SLURM_JOB_ID="123")
        with monkeypatch.context() as m:
            m.setattr(
                cluster_mod.subprocess,
                "run",
                lambda cmd, **kwargs: cluster_mod.subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=""),
            )
            assert cluster_mod._slurm_account() == expected

    def missing_is_rejected():
        _set_account_env(monkeypatch)
        made.clear()
        with pytest.raises(cluster_mod.ConfigurationError, match="account"):
            cluster_mod.start_distributed_cluster(n_workers=4, workers_per_node=4, runtime=60, scratch_dir=tmp_path)
        assert made == []

    check(failures, "explicit_wins", explicit_wins)
    for env, expected in env_rows:
        check(failures, f"environment{sorted(env)}", lambda env=env, expected=expected: from_environment(env, expected))
    check(failures, "from_running_job", from_running_job)
    for row in squeue_rows:
        check(failures, f"squeue{row}", lambda row=row: squeue_outcome(*row))
    check(failures, "missing_is_rejected", missing_is_rejected)
    assert_no_failures(failures)


def _loopback_interface():
    """Name of this machine's loopback interface (``lo`` on Linux, ``lo0`` on macOS)."""
    import psutil

    for name, addresses in psutil.net_if_addrs().items():
        if any(a.address == "127.0.0.1" for a in addresses):
            return name
    pytest.skip("no loopback interface")


@pytest.fixture
def rendered(monkeypatch, tmp_path):
    """``start(**kwargs)`` -> the non-comment lines of the real dask-jobqueue job script (no job is submitted)."""
    dask_jobqueue = pytest.importorskip("dask_jobqueue")
    scripts = []
    loopback = _loopback_interface()

    def render(**kwargs):
        # The scheduler is started on the requested interface; Levante's InfiniBand ``ib0`` exists
        # only there. These cases check the rendered script, so render it on the loopback.
        assert kwargs["interface"] == "ib0"
        cluster = dask_jobqueue.SLURMCluster(**{**kwargs, "interface": loopback})
        scripts.append(cluster.job_script())
        cluster.close()
        return _FakeCluster(**kwargs)

    monkeypatch.setattr(cluster_mod, "SLURMCluster", render)
    monkeypatch.setattr(cluster_mod, "Client", _FakeClient)
    monkeypatch.setattr(cluster_mod, "get_cluster_info", lambda client: None)
    monkeypatch.setattr(cluster_mod, "DKRZ_LOG_PATH", tmp_path / "logs")
    _set_account_env(monkeypatch, SLURM_ACCOUNT="test_account")

    def start(**kwargs):
        cluster_mod.start_distributed_cluster(
            n_workers=4, workers_per_node=4, runtime=60, scratch_dir=tmp_path, dashboard_address=0, **kwargs
        )
        return [line for line in scripts[-1].splitlines() if not line.startswith("#")]

    with _dask_config_restored():
        yield start


CONNECT = "export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=300s"
TCP = "export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=300s"


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_worker_job_script_carries_dask_config(rendered):
    """The rendered job script: worker shape, the comm-timeout exports ahead of the worker and of the caller's
    prologue, and every prologue source (argument, None, a string, the jobqueue config, the deprecated env_extra)."""
    failures = []

    def shape_and_exports():
        lines = rendered(threads_per_worker=8)
        worker = next(line for line in lines if "dask_worker" in line)
        assert "--nthreads 8" in worker and "--nworkers 4" in worker, worker
        assert lines.index(CONNECT) < lines.index(worker)

    def comm_timeouts_ahead_of_the_prologue():
        # configure_dask sets 300 s; without the export the workers keep dask's own 30 s default.
        lines = rendered(job_script_prologue=["module load foo"])
        assert CONNECT in lines and TCP in lines, lines
        assert max(lines.index(CONNECT), lines.index(TCP)) < lines.index("module load foo"), lines

    def deprecated_env_extra():
        assert "module load legacy" in rendered(env_extra=["module load legacy"])

    def jobqueue_config_prologue():
        with dask.config.set({"jobqueue.slurm.job-script-prologue": ["module load from-config"]}):
            assert "module load from-config" in rendered()

    def prologue_none_and_str():
        assert TCP in rendered(job_script_prologue=None)
        assert "module load foo" in rendered(job_script_prologue="module load foo")

    for row in (
        shape_and_exports,
        comm_timeouts_ahead_of_the_prologue,
        deprecated_env_extra,
        jobqueue_config_prologue,
        prologue_none_and_str,
    ):
        check(failures, row.__name__, row)
    assert_no_failures(failures)


def test_memory_warning_capture_is_live(client):
    """A memory warning raised on a worker, and one on the nanny logger, after marEx has configured logging, is
    recorded; the capture's own probes are not."""
    assert on_client(client)
    marEx.configure_logging()
    client.run(marEx.configure_logging)
    with capture_memory_warnings(client) as captured:
        assert captured.records == [], "probe records leaked into the workload interval"
        worker = list(client.scheduler_info()["workers"])[0]
        client.run(logging.getLogger(MEMORY_LOGGERS[0]).warning, "workload warning from a worker", workers=[worker])
        logging.getLogger(MEMORY_LOGGERS[1]).warning("workload warning from a nanny")
    assert sorted((r.name, r.getMessage()) for r in captured.records) == [
        (MEMORY_LOGGERS[1], "workload warning from a nanny"),
        (MEMORY_LOGGERS[0], "workload warning from a worker"),
    ]


def test_marex_records_survive_configure_logging(marex_records):
    """``configure_logging`` clears the package logger's handlers; the capture must not go with them."""
    records = marex_records(logging.INFO)
    marEx.configure_logging()
    logging.getLogger("marEx.anomaly").info("after configure_logging")
    assert "after configure_logging" in [r.getMessage() for r in records]


def test_a_test_that_requests_the_cluster_computes_on_it(client):
    """A test that asks for the session client computes on it."""
    scheduler = dask.base.get_scheduler()
    assert getattr(scheduler, "__self__", None) is client, f"scheduler {scheduler!r} is not the session client's"
    assert dask.array.ones(4, chunks=2).sum().compute() == 4


def test_a_later_test_that_does_not_request_it_runs_threaded():
    """After a test has started the session client, a test that does not ask for it computes on threads.

    Run in a child pytest so the order is fixed (a client test, then one without) whatever xdist
    does with this session; the child's own tests make the assertions.
    """
    probe = Path(__file__).with_name("_scheduler_order_probe.py")
    env = {**os.environ, "MAREX_TEST_WORKER_MEMORY": "1GB"}
    ids = [f"{probe}::test_starts_the_session_client", f"{probe}::test_then_computes_on_threads"]
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly", "-p", "no:xdist", *ids],
        cwd=probe.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert run.returncode == 0 and "2 passed" in run.stdout, run.stdout[-3000:] + run.stderr[-2000:]


def test_resource_monitor_stage_table(client, capsys):
    """One row per stage and a total, a stage recorded when its block raises, a printed line per stage, the
    largest single worker within the summed peak, and no total row for a single stage."""
    assert on_client(client)
    limit, _ = worker_memory_limit()
    monitor = hpc.ResourceMonitor(client, interval=0.1)
    assert (monitor.n_workers, monitor.threads) == (N_CLUSTER_WORKERS, N_CLUSTER_WORKERS)
    assert monitor.worker_memory_limit == pytest.approx(limit)
    with monitor.stage("sum"):
        dask.array.ones((2000, 2000), chunks=500).sum().compute()
    with monitor.stage("sleep"):
        time.sleep(0.3)
    with pytest.raises(ValueError, match="boom"):
        with monitor.stage("fails"):
            raise ValueError("boom")
    out = capsys.readouterr().out
    df = monitor.summary()
    assert list(df.index) == ["sum", "sleep", "fails", "total"]
    assert monitor._records[1]["wall (min)"] * 60 >= 0.29  # unrounded: 0.3 s is 0.005 min
    stages = ["sum", "sleep", "fails"]
    assert df.loc["total", "wall (min)"] == pytest.approx(df.loc[stages, "wall (min)"].sum(), abs=0.02)  # rows rounded to 0.01
    assert (df["peak worker memory (GB)"] > 0).all()
    assert (df["peak client memory (GB)"] > 0).all()
    assert (df["worker restarts"] == 0).all()
    assert 0 < df.loc["sum", "peak per worker (GB)"] <= df.loc["sum", "peak worker memory (GB)"]
    printed = out.splitlines()
    assert [line.split(":")[0] for line in printed] == stages, printed
    assert all(line.endswith("restarts 0") for line in printed), printed

    single = hpc.ResourceMonitor(client, interval=0.1)
    with single.stage("only"):
        np.zeros(10).sum()
    assert list(single.summary().index) == ["only"]


def test_resource_monitor_job_memory_warning(client, monkeypatch, marex_records):
    """Worker limits on this node above the job's memory warn; within it, or on workers on other hosts, they do not."""
    import marEx.helper.resources as res

    assert on_client(client)
    workers = client.scheduler_info(n_workers=-1)["workers"]
    total = float(sum(w["memory_limit"] for w in workers.values()))
    remote = {addr: {**w, "host": "192.0.2.1"} for addr, w in workers.items()}  # TEST-NET-1: never a local address
    failures = []

    def warned(job_memory, workers_seen=None):
        records = marex_records(logging.WARNING)
        with monkeypatch.context() as m:
            m.setattr(res, "_job_memory_bytes", lambda: job_memory)
            if workers_seen is not None:
                m.setattr(res.ResourceMonitor, "_workers", lambda self: workers_seen)
            hpc.ResourceMonitor(client)
        return any("the job may use" in r.getMessage() for r in records)

    def within_the_job():
        assert not warned(2 * total)

    def over_the_job():
        assert warned(total / 2), "local worker limits above the job's memory did not warn"

    def remote_workers_do_not_count():
        assert not warned(total / 2, remote), "workers on another host were counted against this job"

    for row in (within_the_job, over_the_job, remote_workers_do_not_count):
        check(failures, row.__name__, row)
    assert_no_failures(failures)


def test_resource_monitor_counts_restarts(tmp_path):
    """A worker restart inside a stage is counted.

    The restart runs on a private two-worker cluster made and closed here, never the session cluster, so a
    restart that hangs cannot take the later tests' cluster down with it.
    """
    n_workers = 2
    with (
        LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=True,
            memory_limit="1GB",
            local_directory=str(tmp_path),
            dashboard_address=None,
            silence_logs=logging.WARNING,
        ) as cluster,
        Client(cluster, set_as_default=False) as private,
    ):
        private.wait_for_workers(n_workers, timeout=60)
        monitor = hpc.ResourceMonitor(private, interval=0.1)
        with monitor.stage("steady"):
            time.sleep(0.3)
        with monitor.stage("restart"):
            private.restart()
            private.wait_for_workers(n_workers, timeout=60)
            time.sleep(0.3)
        summary = monitor.summary()
    assert summary.loc["steady", "worker restarts"] == 0, summary
    assert summary.loc["restart", "worker restarts"] == n_workers, summary
