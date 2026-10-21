"""
start_local_cluster logs the per-worker memory limit the workers actually get.

Dask is cgroup-blind: without an explicit memory_limit a worker's limit is a share of the
whole machine, not of the job's allocation. Logging that machine share as if it were the
limit in force makes a memory diagnosis start from the wrong number, so the log must report
the memory_limit passed on, and say plainly when there is none.
"""

import logging
from unittest import mock

import pytest

from marEx.helper import cluster as cluster_mod


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def memory_lines(monkeypatch):
    """Run start_local_cluster with the cluster mocked out and return its memory log lines."""
    monkeypatch.setattr(cluster_mod, "LocalCluster", mock.MagicMock())
    monkeypatch.setattr(cluster_mod, "Client", mock.MagicMock())
    monkeypatch.setattr(cluster_mod, "get_cluster_info", lambda client: None)

    def run(**kwargs):
        handler = _ListHandler()
        logger = logging.getLogger(cluster_mod.logger.name)
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            cluster_mod.start_local_cluster(n_workers=2, threads_per_worker=1, **kwargs)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
        return [m for m in handler.messages if "per worker" in m.lower()]

    return run


def test_logs_the_memory_limit_passed_to_the_workers(memory_lines):
    lines = memory_lines(memory_limit="3GB")
    assert lines == ["Memory limit per worker: 3.00 GB"]
    assert cluster_mod.LocalCluster.call_args.kwargs["memory_limit"] == "3GB"


def test_without_a_limit_says_it_is_the_dask_default(memory_lines):
    (line,) = memory_lines()
    assert "Dask default" in line and "pass memory_limit" in line
