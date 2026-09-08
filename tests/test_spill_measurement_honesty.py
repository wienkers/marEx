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
4a. `_probe` returned early on an unreadable `data` store WITHOUT the process key, so a worker
   handing back no store also reported its RSS unmeasurable -- although RSS comes from the
   SystemMonitor and had nothing to do with the store. `_probe` itself had no test at all.
4b. `prochist_covers_whole_run` is `count <= maxlen`, and a worker RESTART resets `count`, so a
   post-restart TAIL read as the whole run. The recorded series span is what catches that.
5. The per-worker MANAGED series was then reported against `memory.spill`'s threshold, but
   `memory.spill` and `memory.pause` threshold per-worker PROCESS memory (worker_memory.py:213)
   and only `memory.target` thresholds managed bytes. A leg at 44 % of target could still have
   crossed spill, so "the target path was never approached" was being read as "the leg never
   approached spilling" (D-042, gate 4 finding 12).

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


def workers(n, disk=0, managed=0, process=0):
    """`n` workers all reporting the same triple, in `_probe`'s return shape."""
    return {f"tcp://w{i}": {"disk": disk, "managed": managed, "process": process} for i in range(n)}


def history(n, rss=(1, 2, 3), limit=4_000_000_000, count=None, maxlen=7200):
    """`n` workers' `_worker_process_history` replies, all with the same RSS series."""
    times = [1000.0 + 0.5 * i for i in range(len(rss))]
    return {
        f"tcp://w{i}": {
            "memory": list(rss),
            "time": list(times),
            "count": len(rss) if count is None else count,
            "maxlen": maxlen,
            "limit": limit,
            "spill_fraction": 0.7,
            "target_fraction": 0.6,
            "pause_fraction": 0.8,
        }
        for i in range(n)
    }


class HistoryClient:
    """A client whose `run` answers one fixed `_worker_process_history` reply, or raises."""

    def __init__(self, reply):
        self.reply = reply

    def run(self, _func):
        """Return the fixed reply, or raise it if it is an exception."""
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def harvest(reply, n_workers=4):
    """Reduce one scripted history reply through the real `harvest_process_history`."""
    fields, _series = squeeze_common.harvest_process_history(HistoryClient(reply), n_workers)
    return fields


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
    process_dead = sampler.process_unmeasured or sampler.process_samples_ok == 0
    return {
        "process_max_worker_bytes": None if process_dead else sampler.max_process_worker,
        "process_unmeasured": bool(process_dead),
        "process_samples_ok": sampler.process_samples_ok,
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


def prints_process(row):
    """Is the POLLED per-worker RSS maximum a figure the sampler was entitled to report?

    The polled series is the 5 s cross-check, not the table column; the table is fed by the
    harvest below. It is held to the same licence so that a JSON reader cannot quote a number
    the sampler never measured either.
    """
    return report._measured(row, "process", "process_max_worker_bytes")


def prints_prochist(row):
    """Would `report.py` print the harvested per-worker RSS maximum for this row?

    The whole licence lives in ONE function in report.py, deliberately: the harvest's own
    observed width, the unwrapped deque, and the series span covering the run are three
    separate ways of getting a tail maximum, and a consumer that reimplemented two of them
    would print the third as if it were the run's maximum.
    """
    return report.prochist_measured(row)


class FakeMonitor:
    """A worker's `monitor`; `rss` may be an exception to raise instead of a reading."""

    def __init__(self, rss):
        self.rss = rss

    def get_process_memory(self):
        """Return the scripted RSS, or raise it."""
        if isinstance(self.rss, Exception):
            raise self.rss
        return self.rss


class FakeSpillBuffer(dict):
    """A `SpillBuffer` duck-type: `spilled_total.disk` bytes on disk, `fast.total_weight` managed."""

    class _Total:
        def __init__(self, disk):
            self.disk = disk

    class _Fast:
        def __init__(self, weight):
            self.total_weight = weight

    def __init__(self, disk, managed):
        super().__init__()
        self.spilled_total = self._Total(disk)
        self.fast = self._Fast(managed)
        self.slow = {}


class FakeWorker:
    """The minimum `_probe` reads: a `data` store and a `monitor`."""

    def __init__(self, data, rss):
        if data is not _MISSING:
            self.data = data
        self.monitor = FakeMonitor(rss)


_MISSING = object()


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
        assert not prints_process(row), f"{label}: printed an RSS figure from partial coverage"

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
        assert prints_process(row)

    def test_full_coverage_of_real_bytes_prints_the_sum(self):
        row = summarise([workers(4, disk=10**9, managed=2 * 10**9, process=3 * 10**9)])
        assert prints_spill(row) and row["spill_max_disk_bytes"] == 4 * 10**9
        assert prints_managed(row) and row["managed_max_worker_bytes"] == 2 * 10**9
        # Per WORKER, not summed: the threshold `memory.spill` applies is per worker, so a
        # cluster sum here would be the same category error D-042 was amended for.
        assert prints_process(row) and row["process_max_worker_bytes"] == 3 * 10**9


class TestSentinelsNeverSumIntoATotal:
    """Three workers at -1 and one at +3 must not cancel to a plausible zero."""

    def test_one_unreadable_worker_kills_that_quantity(self):
        row = summarise([{**workers(4), "tcp://w0": {"disk": -1, "managed": 0, "process": 0}}])
        assert not prints_spill(row)

    def test_all_unreadable_is_unmeasured_not_zero(self):
        row = summarise([{k: {"disk": -1, "managed": -1, "process": -1} for k in workers(4)}])
        assert not prints_spill(row) and not prints_managed(row) and not prints_process(row)

    def test_a_bool_is_not_a_byte_count(self):
        """`isinstance(True, int)` is True, so `True` would otherwise sum as 1 byte."""
        row = summarise([{k: {"disk": True, "managed": True, "process": True} for k in workers(4)}])
        assert not prints_spill(row) and not prints_managed(row) and not prints_process(row)

    @pytest.mark.parametrize("replies", [[RuntimeError("scheduler gone")], [{}]])
    def test_a_sampler_that_never_sampled_is_unmeasured(self, replies):
        """Gate 1's finding: `sum(())` is 0, and 0 is also a legitimate answer."""
        row = summarise(replies)
        assert row["spill_unmeasured"] and row["managed_unmeasured"] and row["process_unmeasured"]
        assert not prints_spill(row) and not prints_managed(row) and not prints_process(row)


class TestTheThreeQuantitiesFailIndependently:
    """One quantity being unreadable must not suppress another, nor silently license one.

    They answer different questions -- `memory.target` thresholds managed bytes, `memory.spill`
    and `memory.pause` threshold process RSS, and the disk figure is the outcome those two
    thresholds produce -- so each must be able to fail on its own.
    """

    @pytest.mark.parametrize(
        "unreadable",
        [("disk",), ("managed",), ("process",), ("disk", "managed"), ("disk", "process"), ("managed", "process")],
    )
    def test_each_quantity_fails_alone(self, unreadable):
        """Exactly the quantities that could not be read read UNMEASURED; the rest still print."""
        values = {"disk": 5, "managed": 5, "process": 5}
        values.update({k: -1 for k in unreadable})
        row = summarise([workers(4, **values)])
        printers = {"disk": prints_spill, "managed": prints_managed, "process": prints_process}
        for quantity, prints in printers.items():
            expected = quantity not in unreadable
            assert prints(row) is expected, f"{quantity} with {unreadable} unreadable"

    def test_a_coverage_failure_kills_all_three(self):
        """A width mismatch is a failure of the round trip, so no quantity was sampled."""
        row = summarise([workers(4), workers(3)])
        assert row["spill_unmeasured"] and row["managed_unmeasured"] and row["process_unmeasured"]


class TestProbeReadsRSSIndependentlyOfTheDataStore:
    """`_probe` had no test, and the gap was a real coupling (falsifier 2026-09-08, finding 6)."""

    UNREADABLE = squeeze_common.SpillSampler.UNREADABLE

    def test_a_readable_store_reports_all_three(self):
        probe = squeeze_common.SpillSampler._probe(FakeWorker(FakeSpillBuffer(disk=7, managed=11), rss=13))
        assert probe == {"disk": 7, "managed": 11, "process": 13}

    def test_no_data_store_still_reports_rss(self):
        """The defect: `data is None` returned early and dropped a perfectly readable RSS."""
        probe = squeeze_common.SpillSampler._probe(FakeWorker(None, rss=13))
        assert probe["process"] == 13
        assert probe["disk"] == self.UNREADABLE and probe["managed"] == self.UNREADABLE

    def test_a_worker_with_no_data_attribute_still_reports_rss(self):
        probe = squeeze_common.SpillSampler._probe(FakeWorker(_MISSING, rss=13))
        assert probe["process"] == 13

    def test_an_unreadable_monitor_does_not_touch_the_other_two(self):
        """The converse coupling, tested so the fix cannot be reversed into the other direction."""
        probe = squeeze_common.SpillSampler._probe(FakeWorker(FakeSpillBuffer(disk=7, managed=11), rss=RuntimeError("no psutil")))
        assert probe == {"disk": 7, "managed": 11, "process": self.UNREADABLE}

    def test_a_plain_dict_is_a_true_zero_on_disk_and_unreadable_managed(self):
        """`--no-spill` hands back a plain dict: 0 on disk is the true answer, managed is not."""
        probe = squeeze_common.SpillSampler._probe(FakeWorker({}, rss=13))
        assert probe == {"disk": 0, "managed": self.UNREADABLE, "process": 13}


class TestProcessHistoryHarvest:
    """`harvest_process_history` is the primary RSS instrument; the poller is its cross-check.

    It reads dask's own `SystemMonitor.quantities["memory"]`, which is `get_process_memory()`
    sampled on a 500 ms PeriodicCallback -- the same call `memory_monitor` thresholds on. The
    failure modes are the sampler's, one round trip later, so they are tested the same way.
    """

    def test_a_full_reply_yields_the_per_worker_maximum(self):
        fields = harvest(history(4, rss=(1_000, 2_000, 1_500)))
        assert fields["prochist_unmeasured"] is False
        assert fields["prochist_max_worker_bytes"] == 2_000
        assert fields["prochist_workers_sampled"] == 4
        assert fields["prochist_samples_ok"] == 3
        assert fields["prochist_covers_whole_run"] is True
        assert fields["prochist_spill_fraction"] == [0.7]
        # times are 1000.0, 1000.5, 1001.0 -> the series spans 1.0 s
        assert fields["prochist_span_s"] == pytest.approx(1.0)
        assert fields["prochist_span_s_source"] == "run"

    def test_the_fraction_is_per_worker_against_its_own_limit(self):
        """With unequal limits, max(RSS)/max(limit) is not the quantity `memory_monitor` computes."""
        reply = history(2, rss=(1_000,), limit=10_000)
        small = "tcp://w1"
        reply[small] = dict(reply[small], memory=[900], limit=1_000)
        fields = harvest(reply, n_workers=2)
        # 900/1000 = 0.9, not 1000/10000 = 0.1 and not 1000/1000 = 1.0.
        assert fields["prochist_max_worker_fraction"] == pytest.approx(0.9)

    @pytest.mark.parametrize(
        "label,reply,n_workers",
        [
            ("the round trip raised", RuntimeError("scheduler gone"), 4),
            ("nobody answered", {}, 4),
            ("a strict subset answered", history(1), 4),
            ("more workers than requested", history(5), 4),
        ],
    )
    def test_a_failed_round_trip_is_unmeasured(self, label, reply, n_workers):
        fields = harvest(reply, n_workers=n_workers)
        assert fields["prochist_unmeasured"] is True, label
        assert fields["prochist_max_worker_bytes"] is None, label

    def test_one_worker_erroring_kills_the_whole_harvest(self):
        """Three maxima with the fourth missing is a maximum over three quarters of a cluster."""
        reply = history(4)
        reply["tcp://w0"] = {"error": "monitor unreadable: AttributeError"}
        fields = harvest(reply)
        assert fields["prochist_unmeasured"] is True
        assert fields["prochist_max_worker_bytes"] is None

    def test_an_empty_series_is_not_a_zero(self):
        """A worker that recorded nothing has `max([])`, not a maximum of 0."""
        reply = history(4)
        reply["tcp://w0"] = dict(reply["tcp://w0"], memory=[], time=[])
        fields = harvest(reply)
        assert fields["prochist_unmeasured"] is True

    def test_a_wrapped_deque_is_measured_but_not_whole_run(self):
        """`log-length` samples is 3600 s at 500 ms; a longer run keeps only the tail."""
        fields = harvest(history(4, rss=(1, 2), count=9000, maxlen=7200))
        assert fields["prochist_unmeasured"] is False
        assert fields["prochist_covers_whole_run"] is False
        row = dict(fields, n_workers=4, n_workers_requested=4)
        assert not prints_prochist(row), "a tail-only maximum printed as the run's maximum"


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


class TestTheHarvestCarriesItsOwnCoverage:
    """The harvest is a second round trip, so the sampler's coverage does not extend to it."""

    BASE = {
        "prochist_max_worker_bytes": 2_500_000_000,
        "prochist_unmeasured": False,
        "prochist_samples_ok": 3300,
        "prochist_workers_sampled": 4,
        "prochist_covers_whole_run": True,
        "prochist_span_s": 1670.0,
        "prochist_span_s_source": "run",
        "elapsed_s": 1675.5,
        "spill_workers_sampled": 4,
        "n_workers": 4,
        "n_workers_requested": 4,
    }

    def test_the_fully_licensed_row_prints(self):
        assert prints_prochist(dict(self.BASE))

    def test_the_samplers_width_does_not_license_the_harvest(self):
        """The load-bearing case: 4 workers sampled every 5 s, then 1 of 4 answered the harvest.

        Reusing `spill_workers_sampled` as the witness would print a per-worker maximum taken
        over a quarter of the cluster, which is gate 3's finding 4 one round trip later.
        """
        row = dict(self.BASE, prochist_workers_sampled=1)
        assert row["spill_workers_sampled"] == 4
        assert not prints_prochist(row)

    @pytest.mark.parametrize(
        "label,override",
        [
            ("a leg predating the harvest", {"prochist_unmeasured": None}),
            ("unmeasured is True", {"prochist_unmeasured": True}),
            ("null byte count", {"prochist_max_worker_bytes": None}),
            ("no samples in the deque", {"prochist_samples_ok": 0}),
            ("the deque had wrapped", {"prochist_covers_whole_run": False}),
            ("no whole-run key at all", {"prochist_covers_whole_run": None}),
            ("width below the cluster", {"prochist_workers_sampled": 1, "n_workers": 1}),
            ("a post-restart tail, half the run", {"prochist_span_s": 800.0}),
            ("no span recorded at all", {"prochist_span_s": None}),
            ("no wall to compare the span against", {"elapsed_s": None}),
            # A lower bound alone accepts this: a number nobody measured passing a check
            # nobody can fail. The upper bound is what rejects it.
            ("an absurd span, far longer than the run", {"prochist_span_s": 999999.0}),
            ("no source for the span", {"prochist_span_s_source": None}),
            ("a source nothing recognises", {"prochist_span_s_source": "guessed"}),
        ],
    )
    def test_an_incomplete_row_reads_unmeasured(self, label, override):
        row = dict(self.BASE)
        row.update(override)
        for key, value in override.items():
            if value is None and key != "prochist_max_worker_bytes":
                del row[key]
        assert not prints_prochist(row), f"{label}: printed a figure it was not entitled to"

    def test_a_span_reduced_from_the_persisted_series_is_licensed_and_marked(self):
        """A value derived after the fact from an artefact the run persisted is legitimate.

        What is not legitimate is its being indistinguishable from one the run wrote, so the
        source is a key the licence READS -- not a note in a sibling key nothing consumes.
        """
        row = dict(self.BASE, prochist_span_s_source="npz-backfill")
        assert prints_prochist(row)

    def test_a_restarted_worker_is_not_caught_by_covers_whole_run_alone(self):
        """The finding, stated as a test: `count <= maxlen` is True on a post-restart tail."""
        row = dict(self.BASE, prochist_span_s=800.0)
        assert row["prochist_covers_whole_run"] is True
        assert report._measured(row, "prochist", "prochist_max_worker_bytes", "prochist_workers_sampled")
        assert not prints_prochist(row), "a half-length series printed as the run's maximum"
