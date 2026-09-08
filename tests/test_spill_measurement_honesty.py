"""The spill/managed measurement in `examples/larger-than-memory/` must not report a guess.

Three consecutive adversarial reviews found real defects in this one surface, each of the same
shape: a number reached the README that no sampler was entitled to report.

1. `_probe` read `worker.data.disk.weight_by_key`, which does not exist on distributed 2025.9.1,
   and a bare `except` turned the AttributeError into `0`. Every leg of the campaign reported
   "spill 0.00 GB" and none of them had measured anything (D-038, D-041).
2. A sampler that never sampled -- `client.run` raising on every tick, or answering from no
   workers -- produced a summary bit-identical to one that sampled throughout and saw nothing.
   `samples_ok` was added to separate them.
3. `samples_ok` was then satisfiable by a client answering from 1 of 4 workers, because the
   expected width was pinned from the first sample instead of being required to match the
   cluster. A quarter of the cluster read as the whole of it.

The common thread is that the honesty check itself was never tested, so each fix shipped on the
strength of the code reading right. These tests are the standing version of the adversarial
table, and they need no cluster: every defect above lives in the arithmetic that CONSUMES
`client.run`, not in dask. `measurements/lm/verify_sampler.py` covers the real-worker half.
"""

import importlib.util
import time
from pathlib import Path

import pytest

LM_DIR = Path(__file__).resolve().parent.parent / "examples" / "larger-than-memory"


def _load(name):
    """Import one of the example scripts by path; `examples/` is not an importable package."""
    spec = importlib.util.spec_from_file_location(f"_lm_{name}", LM_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


squeeze_common = _load("squeeze_common")
report = _load("report")


class ScriptedClient:
    """A client whose `run` replays a fixed sequence of per-worker replies, then repeats."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def run(self, _func):
        """Return the next scripted reply, or raise it if it is an exception."""
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return reply


def workers(n, disk=0, managed=0):
    """`n` workers all reporting the same disk/managed pair, in `_probe`'s return shape."""
    return {f"tcp://w{i}": {"disk": disk, "managed": managed} for i in range(n)}


def summarise(replies, n_workers=4, ticks=6):
    """Drive a sampler over a scripted client and build the summary fields `report.py` reads."""
    sampler = squeeze_common.SpillSampler(ScriptedClient(replies), n_workers=n_workers, interval=0.001)
    sampler.start()
    deadline = time.monotonic() + 10
    while sampler.client.calls < ticks and time.monotonic() < deadline:
        time.sleep(0.005)
    sampler.stop()
    sampler.join(timeout=5)
    assert sampler.client.calls >= ticks, "the sampler thread never ran; the test would prove nothing"

    spill_dead = sampler.unmeasured or sampler.samples_ok == 0
    managed_dead = sampler.managed_unmeasured or sampler.managed_samples_ok == 0
    return {
        "spill_max_disk_bytes": None if spill_dead else sampler.max_disk,
        "spill_unmeasured": bool(spill_dead),
        "spill_samples_ok": sampler.samples_ok,
        "spill_workers_sampled": sampler.workers_expected,
        "managed_max_worker_bytes": None if managed_dead else sampler.max_managed_worker,
        "managed_unmeasured": bool(managed_dead),
        "managed_samples_ok": sampler.managed_samples_ok,
        "n_workers": n_workers,
        "n_workers_requested": n_workers,
    }


def prints_spill(row):
    """Would `report.py` print a spill figure for this summary row?"""
    return report._measured(row, "spill", "spill_max_disk_bytes")


def prints_managed(row):
    """Would `report.py` print a managed figure for this summary row?"""
    return report._measured(row, "managed", "managed_max_worker_bytes")


class TestPartialCoverageIsNotAMeasurement:
    """A sample that reached some of the cluster under-counts exactly like a failed read."""

    @pytest.mark.parametrize(
        "label,replies",
        [
            ("always 1 of 4, all zero", [workers(1)]),
            ("always 1 of 4, with bytes", [workers(1, disk=10**9)]),
            ("narrow first, then full", [workers(1), workers(4)]),
            ("full, then a fifth worker", [workers(4), workers(5)]),
            ("full, then one drops out", [workers(4), workers(3)]),
        ],
    )
    def test_narrow_sample_never_licenses_a_figure(self, label, replies):
        """This is gate 3's finding 4: the defect that made a quarter of a cluster read as all of it."""
        row = summarise(replies)
        assert not prints_spill(row), f"{label}: printed a spill figure from partial coverage"
        assert not prints_managed(row), f"{label}: printed a managed figure from partial coverage"

    def test_expected_width_cannot_be_taught_by_the_client(self):
        """The width comes from the REQUESTED count, so a persistently narrow client cannot set it.

        Pinning it from the first sample is what made the previous fix self-referential: the
        client that narrowed every sample also narrowed the expectation it was checked against.
        """
        row = summarise([workers(1)], n_workers=4)
        assert row["spill_workers_sampled"] == 4
        assert row["spill_samples_ok"] == 0

    def test_width_is_not_taken_from_the_cluster_derived_count(self):
        """`n_workers` is `len(client.run(...))`, so requiring only it would restate the undercount."""
        row = summarise([workers(1)], n_workers=4)
        row.update(spill_workers_sampled=1, n_workers=1, spill_unmeasured=False, spill_samples_ok=22)
        row["spill_max_disk_bytes"] = 0
        assert not prints_spill(row), "a summary narrowed consistently to 1 of 4 still printed a figure"


class TestAFullSampleOfZeroIsStillAMeasurement:
    """The fix must not buy honesty by making everything unmeasurable."""

    def test_full_coverage_of_true_zeros_prints_zero(self):
        row = summarise([workers(4)])
        assert prints_spill(row) and row["spill_max_disk_bytes"] == 0
        assert prints_managed(row)

    def test_full_coverage_of_real_bytes_prints_the_sum(self):
        row = summarise([workers(4, disk=10**9, managed=2 * 10**9)])
        assert prints_spill(row) and row["spill_max_disk_bytes"] == 4 * 10**9
        assert prints_managed(row) and row["managed_max_worker_bytes"] == 2 * 10**9


class TestSentinelsNeverSumIntoATotal:
    """Three workers at -1 and one at +3 must not cancel to a plausible zero."""

    def test_one_unreadable_worker_kills_that_quantity(self):
        row = summarise([{**workers(4), "tcp://w0": {"disk": -1, "managed": 0}}])
        assert not prints_spill(row)

    def test_all_unreadable_is_unmeasured_not_zero(self):
        row = summarise([{k: {"disk": -1, "managed": -1} for k in workers(4)}])
        assert not prints_spill(row) and not prints_managed(row)

    def test_a_bool_is_not_a_byte_count(self):
        """`isinstance(True, int)` is True, so `True` would otherwise sum as 1 byte."""
        row = summarise([{k: {"disk": True, "managed": True} for k in workers(4)}])
        assert not prints_spill(row) and not prints_managed(row)

    @pytest.mark.parametrize("replies", [[RuntimeError("scheduler gone")], [{}]])
    def test_a_sampler_that_never_sampled_is_unmeasured(self, replies):
        """Gate 1's finding: `sum(())` is 0, and 0 is also a legitimate answer."""
        row = summarise(replies)
        assert row["spill_unmeasured"] and row["managed_unmeasured"]
        assert not prints_spill(row) and not prints_managed(row)


class TestDiskAndManagedFailIndependently:
    """One quantity being unreadable must not suppress the other, nor silently license it."""

    def test_unreadable_managed_leaves_disk_measured(self):
        row = summarise([workers(4, disk=5, managed=-1)])
        assert prints_spill(row)
        assert not prints_managed(row)

    def test_unreadable_disk_leaves_managed_measured(self):
        row = summarise([workers(4, disk=-1, managed=5)])
        assert not prints_spill(row)
        assert prints_managed(row)

    def test_a_coverage_failure_kills_both(self):
        """A width mismatch is a failure of the round trip, so neither quantity was sampled."""
        row = summarise([workers(4), workers(3)])
        assert row["spill_unmeasured"] and row["managed_unmeasured"]


class TestReportLicence:
    """`report.py` is the last gate before a number reaches the README."""

    BASE = {
        "spill_max_disk_bytes": 0,
        "spill_unmeasured": False,
        "spill_samples_ok": 324,
        "spill_workers_sampled": 4,
        "n_workers": 4,
        "n_workers_requested": 4,
    }

    def test_the_fully_licensed_row_prints(self):
        assert prints_spill(dict(self.BASE))

    @pytest.mark.parametrize(
        "label,override",
        [
            ("pre-probe-fix leg, no unmeasured key", {"spill_unmeasured": None}),
            ("unmeasured is True", {"spill_unmeasured": True}),
            ("null byte count", {"spill_max_disk_bytes": None}),
            ("no successful samples", {"spill_samples_ok": 0}),
            ("samples_ok is a bool", {"spill_samples_ok": True}),
            ("pre-coverage leg, no width key", {"spill_workers_sampled": None}),
            ("width below the cluster", {"spill_workers_sampled": 1, "n_workers": 1}),
            ("width disagrees with the request", {"n_workers_requested": 8}),
        ],
    )
    def test_an_incomplete_row_reads_unmeasured(self, label, override):
        row = dict(self.BASE)
        row.update(override)
        for key, value in override.items():
            if value is None and key != "spill_max_disk_bytes":
                del row[key]
        assert not prints_spill(row), f"{label}: printed a figure it was not entitled to"

    def test_a_missing_key_is_absent_not_falsy(self):
        """A leg predating the probe fix has no `spill_unmeasured` at all; its recorded 0 is an
        artefact of a probe that returned 0 on failure, so ABSENT must read UNMEASURED."""
        assert not prints_spill({"spill_max_disk_bytes": 0, "spill_samples_ok": 324})
