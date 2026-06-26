"""marEx.helper.ResourceMonitor: per-stage wall time, memory, spill and restarts."""

import logging
import time

import dask.array as da
import numpy as np
import pytest
from dask.distributed import Client, LocalCluster

import marEx.helper as hpc


@pytest.fixture(scope="module")
def client():
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        memory_limit="1GB",
        dashboard_address=None,
        processes=True,
    )
    c = Client(cluster)
    yield c
    c.close()
    cluster.close()


def test_records_one_row_per_stage_and_a_total(client):
    monitor = hpc.ResourceMonitor(client, interval=0.1)
    assert monitor.n_workers == 2 and monitor.threads == 2
    assert monitor.worker_memory_limit == pytest.approx(1e9)
    with monitor.stage("sum"):
        da.ones((2000, 2000), chunks=500).sum().compute()
    with monitor.stage("sleep"):
        time.sleep(0.3)
    df = monitor.summary()
    assert list(df.index) == ["sum", "sleep", "total"]
    assert monitor._records[1]["wall (min)"] * 60 >= 0.29  # unrounded: 0.3 s is 0.005 min
    assert df.loc["total", "wall (min)"] == pytest.approx(
        df.loc[["sum", "sleep"], "wall (min)"].sum(), abs=0.015
    )  # each row rounded to 0.01
    assert (df["peak worker memory (GB)"] > 0).all()
    assert (df["peak client memory (GB)"] > 0).all()
    assert (df["worker restarts"] == 0).all()


def test_single_stage_has_no_total_row(client):
    monitor = hpc.ResourceMonitor(client, interval=0.1)
    with monitor.stage("only"):
        pass
    assert list(monitor.summary().index) == ["only"]


def test_stage_is_recorded_when_the_block_raises(client):
    monitor = hpc.ResourceMonitor(client, interval=0.1)
    with pytest.raises(ValueError):
        with monitor.stage("fails"):
            raise ValueError("boom")
    assert list(monitor.summary().index) == ["fails"]


def test_counts_worker_restarts(client):
    monitor = hpc.ResourceMonitor(client, interval=0.1)
    with monitor.stage("restart"):
        client.restart()
        client.wait_for_workers(2)
        time.sleep(0.3)
    assert monitor.summary().loc["restart", "worker restarts"] == 2


def test_prints_a_line_per_stage(client, capsys):
    monitor = hpc.ResourceMonitor(client, interval=0.1)
    with monitor.stage("print me"):
        np.zeros(10).sum()
    out = capsys.readouterr().out
    assert out.startswith("print me:") and "restarts 0" in out


def test_parses_slurm_memory_as_binary_units():
    from marEx.helper.resources import _parse_slurm_mem

    assert _parse_slurm_mem("15040") == 15040 * 2**20
    assert _parse_slurm_mem("15040M") == 15040 * 2**20
    assert _parse_slurm_mem("96G") == 96 * 2**30
    assert _parse_slurm_mem("garbage") is None


def test_warns_when_limits_exceed_the_job_memory(client, monkeypatch, caplog):
    import marEx.helper.resources as res

    monkeypatch.setattr(res, "_job_memory_bytes", lambda: 1e9)  # 1 GB for the job, 2 GB of worker limits
    caplog.set_level(logging.WARNING, logger="marEx")
    logging.getLogger("marEx").propagate = True
    hpc.ResourceMonitor(client)
    assert any("the job may use" in r.getMessage() for r in caplog.records)
