"""Streaming detect and track complete on inputs larger than the summed worker memory.

Each larger-than-memory test owns a one-worker, one-thread ``LocalCluster`` with an explicit
``memory_limit`` (dask cannot see a cgroup or SLURM cap), reads the limit back, and asserts that
the governed bytes exceed the summed effective limits by >= 1.1 before it writes anything, so a
shape that accidentally fits fails in seconds. The governed bytes are the float32 input for detect
and the int32 ID field (the streaming output) for track. Spill thresholds are dask's defaults.
Each then asserts, over the workload: completion, no worker restarted, no
``distributed.{worker,nanny}.memory`` warning other than dask's pause/resume/unmanaged
back-pressure (a live capture), the driver's resident memory grew by < 0.25 x the governed bytes,
a value check that can fail, and the streaming staging directory gone after
``marEx.clear_staging``.

Not claimed: that ``persist`` would fail at these shapes. They prove that ``streaming`` completes
on an input larger than the memory budget, nothing more. For the tracker, only its int32 ID field
(the streaming output) exceeds the budget; its boolean input is a third of it, and the worker was
not under memory pressure on Levante (peak 0.95 GB of 1.5 GiB, nothing spilled).

Shape overrides (the flake policy allows a larger per-worker limit with more data, never a ratio
below 1.1): ``MAREX_LTM_DETECT_LIMIT``, ``MAREX_LTM_DETECT_YEARS``, ``MAREX_LTM_TRACK_LIMIT``,
``MAREX_LTM_TRACK_NT``.
"""

import contextlib
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dask.utils import format_bytes, parse_bytes
from distributed import Client, LocalCluster

import marEx
import marEx.helper as hpc

from ._synthetic_stores import TIME_CHUNK, DriverRSSMonitor, write_synthetic_blobs, write_synthetic_sst
from .conftest import _report_text, assert_no_failures, capture_memory_warnings, check, on_client

MIN_RATIO = 1.1
DRIVER_FRACTION = 0.25
NY, NX = 256, 720

# 4 GiB x 18 y paused the worker (80 %) at 3.21 GiB; at 25 y the worker peaked at 4.20 GB, 71 % of 5.5 GiB:
# the detect peak grows with record length at this shape (3.39 GB at 18 y).
DETECT_LIMIT = os.environ.get("MAREX_LTM_DETECT_LIMIT", "5.5GiB")
DETECT_YEARS = int(os.environ.get("MAREX_LTM_DETECT_YEARS", "25"))
DETECT_KWARGS = dict(
    method_anomaly="detrend_harmonic",
    method_extreme="global_percentile",
    threshold_percentile=95,
    dimensions={"time": "time", "x": "lon", "y": "lat"},
    dask_chunks={"time": TIME_CHUNK},
)

TRACK_LIMIT = os.environ.get("MAREX_LTM_TRACK_LIMIT", "1.5GiB")
TRACK_NT = int(os.environ.get("MAREX_LTM_TRACK_NT", "2850"))
# R_fill 4 leaves the planted radius-10 discs unchanged and cannot join them to anything (every
# planted event is > 8 cells from any other blob); planted discs (317 cells) are far above the
# area filter; T_fill 2 cannot bridge a planted event, which is gone for good once it ends.
TRACK_KWARGS = dict(
    R_fill=4,
    T_fill=2,
    area_filter_absolute=40,
    allow_merging=True,
    overlap_threshold=0.5,
    nn_partitioning=True,
    quiet=True,
)


class _NannyRestarts(logging.Handler):
    """Records on ``distributed.nanny`` that say a worker was restarted."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        if "restart" in record.getMessage().lower():
            self.records.append(record)


@pytest.fixture
def ltm_client(tmp_path):
    """``start(limit)`` -> a Client on this test's own one-worker, one-thread cluster; closed at teardown.

    Its spill directory is under ``tmp_path`` and removed after the cluster closes.
    """
    opened = []

    def start(limit):
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=True,
            memory_limit=limit,
            local_directory=str(tmp_path / "dask"),
            dashboard_address=None,
            silence_logs=logging.WARNING,
        )
        client = Client(cluster)
        opened.append((client, cluster))
        client.wait_for_workers(1, timeout=60)
        return client

    yield start
    with contextlib.ExitStack() as stack:  # every close runs even if an earlier one raises; newest first
        stack.callback(shutil.rmtree, tmp_path / "dask", ignore_errors=True)
        for client, cluster in opened:
            stack.callback(cluster.close)
            stack.callback(client.close)


def _budget(client, requested, input_bytes, label, extra=""):
    """Assert the effective limit equals the request and the input exceeds the budget by MIN_RATIO; returns the limits."""
    limits = sorted(w["memory_limit"] for w in client.scheduler_info()["workers"].values())
    ratio = input_bytes / sum(limits)
    print(
        f"\n[{label}] worker memory_limit {[format_bytes(x) for x in limits]} (requested {format_bytes(requested)}), "
        f"input {format_bytes(input_bytes)}, ratio {ratio:.3f}{extra}",
        flush=True,
    )
    assert limits == [requested], f"workers run with {limits}, requested {requested}"
    assert ratio >= MIN_RATIO, f"input {input_bytes} B / summed worker limits {sum(limits)} B = {ratio:.3f} < {MIN_RATIO}"
    return limits


# dask's worker-side back-pressure: the worker pausing at the pause fraction, resuming, or
# reporting unmanaged memory above the target. This is how a streaming run stays inside its
# budget, so it is reported, not failed. Every other memory warning fails the test, including
# the nanny's "exceeded 95% memory budget. Restarting...".
BACK_PRESSURE = ("Worker is at ", "Unmanaged memory use is high")


def _split_memory_warnings(records):
    pressure = [r for r in records if r.name == "distributed.worker.memory" and r.getMessage().startswith(BACK_PRESSURE)]
    return pressure, [r for r in records if r not in pressure]


class _Workload:
    """Everything the larger-than-memory claim needs watched while the workload runs."""

    def __init__(self, client, label):
        self.client, self.label = client, label

    def __enter__(self):
        assert on_client(self.client), "the workload would not compute on this test's cluster"
        self.addresses = set(self.client.scheduler_info()["workers"])
        self.resources = hpc.ResourceMonitor(self.client, interval=1.0)
        self.nanny = _NannyRestarts()
        with contextlib.ExitStack() as stack:
            logging.getLogger("distributed.nanny").addHandler(self.nanny)
            stack.callback(logging.getLogger("distributed.nanny").removeHandler, self.nanny)
            # The capture probes every worker and the nanny logger before the workload starts.
            self.captured = stack.enter_context(capture_memory_warnings(self.client))
            self.driver = stack.enter_context(DriverRSSMonitor())
            stack.enter_context(self.resources.stage(self.label))
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def check(self, input_bytes):
        row = self.resources.summary().loc[self.label]
        print(
            f"[{self.label}] wall {self.driver.wall:.0f} s, peak worker memory {row['peak worker memory (GB)']:.2f} GB, "
            f"driver RSS growth {format_bytes(self.driver.growth)} (guard {format_bytes(int(DRIVER_FRACTION * input_bytes))}), "
            f"worker restarts {row['worker restarts']}",
            flush=True,
        )
        pressure, danger = _split_memory_warnings(self.captured.records)
        print(f"[{self.label}] back-pressure warnings {len(pressure)} (pause/resume/unmanaged; not failures)", flush=True)
        assert danger == [], "memory warnings during the workload:\n" + _report_text(danger)
        assert set(self.client.scheduler_info()["workers"]) == self.addresses, "a worker was lost or restarted"
        assert self.nanny.records == [], _report_text(self.nanny.records)
        assert row["worker restarts"] == 0
        assert self.driver.growth < DRIVER_FRACTION * input_bytes, (
            f"driver RSS grew {self.driver.growth} B, >= {DRIVER_FRACTION} x input {input_bytes} B"
        )


def _du(path, when):
    out = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.strip()
    print(f"du -sh {when}: {out}", flush=True)


@pytest.mark.larger_than_memory
def test_streaming_detect_larger_than_memory(ltm_client, tmp_path):
    """detrend_harmonic + global_percentile (95th) on a 25-year 0.5 deg daily field larger than the worker budget.

    The extreme fraction of a global 95th-percentile run is 0.050 +- 0.001.
    """
    requested = parse_bytes(DETECT_LIMIT)
    client = ltm_client(requested)
    n_days = int((np.datetime64(f"{2000 + DETECT_YEARS}-01-01") - np.datetime64("2000-01-01")).astype(int))
    input_bytes = n_days * NY * NX * 4
    _budget(client, requested, input_bytes, "detect")

    store, out = tmp_path / "sst.zarr", tmp_path / "extremes.zarr"
    try:
        assert write_synthetic_sst(store, DETECT_YEARS, NY, NX) == input_bytes
        sst = xr.open_zarr(store, chunks={"time": TIME_CHUNK, "lat": -1, "lon": -1}).sst
        assert sst.size * 4 == input_bytes
        with _Workload(client, "detect") as run:
            ds = marEx.preprocess_data(sst, compute_mode="streaming", scratch_dir=str(tmp_path / "stage"), **DETECT_KWARGS)
            staging = Path(ds.encoding["marex_staging_dir"])
            assert staging.exists()
            ds[["extreme_events", "mask"]].to_zarr(out, mode="w")
            _du(tmp_path, "at the disk peak (input, staging, output)")
            marEx.clear_staging(ds)
            fraction = float(xr.open_zarr(out).extreme_events.mean().compute())
        print(f"[detect] extreme fraction {fraction:.5f}", flush=True)
        run.check(input_bytes)
        assert abs(fraction - 0.05) <= 0.001, fraction
        assert not staging.exists(), f"staging directory {staging} survived clear_staging"
    finally:
        _du(tmp_path, "after clear_staging")
        for path in (store, out, tmp_path / "stage"):
            shutil.rmtree(path, ignore_errors=True)


def planted_failures(events, merges, truth):
    """Failures of the tracked output against the planted truth; reads small slices only.

    - each planted lineage carries ONE event ID in its window, on both sides of the time-chunk
      boundary it crosses, with exactly the planted cells (317) at every step and none outside
      its lifetime; the event's ``presence`` is exactly the planted lifetime;
    - the two lineages are different events;
    - the planted merge: two different parent events the step before ``t_merge``, both rows of
      the merge ledger set at ``t_merge`` and not the step before, a two-parent merge recorded at
      that time, and only those two events in the capsule while it lives.
    """
    failures, ids = [], {}
    nt = events.sizes["time"]
    m = truth["merge"]
    r = m["radius"]

    for lin in truth["lineages"]:
        name = lin["name"]

        def lineage(lin=lin, name=name):
            t_lo, t_hi = max(lin["t_first"] - 2, 0), min(lin["t_end"] + 2, nt)
            rows = slice(lin["row"] - r - 2, lin["row"] + r + 3)
            col_lo, col_hi = min(lin["cols"]) - r - 2, max(lin["cols"]) + r + 3
            block = events.ID_field.isel(time=slice(t_lo, t_hi), lat=rows, lon=slice(col_lo, col_hi)).values
            found = np.unique(block[block > 0])
            assert found.size == 1, f"event IDs {found.tolist()} in its window, expected exactly one"
            eid = int(found[0])
            ids[name] = eid
            counts = (block == eid).sum(axis=(1, 2))
            expected = [lin["cells"] if lin["t_first"] <= t < lin["t_end"] else 0 for t in range(t_lo, t_hi)]
            assert counts.tolist() == expected, f"ID {eid}: cells per step {counts.tolist()}, planted {expected}"
            b = lin["crosses"]
            assert counts[b - 1 - t_lo] > 0 and counts[b - t_lo] > 0, f"ID {eid} not on both sides of the time-chunk boundary {b}"
            presence = np.flatnonzero(events.presence.sel(ID=eid).values)
            assert presence.tolist() == list(range(lin["t_first"], lin["t_end"])), (
                f"ID {eid}: present at steps {presence.min()}..{presence.max()} ({presence.size}), "
                f"planted {lin['t_first']}..{lin['t_end'] - 1}"
            )

        check(failures, name, lineage)

    def distinct_lineages():
        assert len(set(ids.values())) == len(truth["lineages"]), f"lineage IDs {ids}"

    check(failures, "distinct lineages", distinct_lineages)

    def merge():
        t = m["t_merge"]
        row, (c1, c2) = m["row"], m["cols"]
        before = events.ID_field.isel(time=t - 1, lat=row).values
        parents = sorted({int(before[c1]), int(before[c2])})
        assert len(parents) == 2 and 0 not in parents, f"parents at t_merge - 1: {parents}"
        ledger = events.merge_ledger.isel(time=slice(t - 1, t + 1)).sel(ID=parents).values  # (2, 2, sibling)
        assert not (ledger[0] != -1).any(), f"merge ledger already set for {parents} the step before the merge"
        assert (ledger[1] != -1).any(axis=-1).all(), f"merge ledger at t_merge for {parents}: {ledger[1].tolist()}"
        at = merges.merge_time.values == events.time.values[t]
        assert (merges.n_parents.values[at] == 2).any(), f"no two-parent merge recorded at step {t}"
        capsule = events.ID_field.isel(
            time=slice(t, m["t_end"]), lat=slice(row - r, row + r + 1), lon=slice(c1 - r, c2 + r + 1)
        ).values
        assert sorted(np.unique(capsule[capsule > 0]).tolist()) == parents, "the capsule holds events other than the two parents"

    check(failures, "merge", merge)
    return failures


@pytest.mark.larger_than_memory
def test_streaming_track_larger_than_memory(ltm_client, tmp_path):
    """The tracker on synthetic blobs whose int32 ID field is larger than the worker budget.

    The boolean input is a quarter of that and is not claimed to exceed it; both are printed.
    The oracle is the planted truth (``planted_failures``), not ``ID_field.max() == N_events``.
    """
    requested = parse_bytes(TRACK_LIMIT)
    client = ltm_client(requested)
    id_bytes, bool_bytes = TRACK_NT * NY * NX * 4, TRACK_NT * NY * NX
    note = f" (int32 ID field; boolean input {format_bytes(bool_bytes)}, ratio {bool_bytes / requested:.3f})"
    _budget(client, requested, id_bytes, "track", note)

    store, events_path = tmp_path / "blobs.zarr", tmp_path / "events.zarr"
    try:
        info = write_synthetic_blobs(store, TRACK_NT, NY, NX)
        assert (info["id_bytes"], info["bool_bytes"]) == (id_bytes, bool_bytes)
        data = xr.open_zarr(store, chunks={"time": TIME_CHUNK, "lat": -1, "lon": -1})
        with _Workload(client, "track") as run:
            tracker = marEx.tracker(
                data.extreme_events, data.mask, compute_mode="streaming", temp_dir=str(tmp_path / "stage"), **TRACK_KWARGS
            )
            events, merges = tracker.run(return_merges=True)
            staging = Path(events.encoding["marex_staging_dir"])
            assert staging.exists()
            merges = merges.load()
            events.to_zarr(events_path, mode="w")
            _du(tmp_path, "at the disk peak (input, staging, output)")
            marEx.clear_staging(events)
        events = xr.open_zarr(events_path)
        n_events, n_merges = int(events.attrs["N_events_final"]), int(merges.sizes.get("merge_ID", 0))
        print(f"[track] N_events {n_events}, merges {n_merges}", flush=True)
        run.check(id_bytes)
        assert n_events > 0 and n_merges > 0
        assert not staging.exists(), f"staging directory {staging} survived clear_staging"
        failures = planted_failures(events, merges, info["truth"])
        assert_no_failures(failures)
    finally:
        _du(tmp_path, "after clear_staging")
        for path in (store, events_path, tmp_path / "stage"):
            shutil.rmtree(path, ignore_errors=True)


def test_driver_guard_trips_on_a_client_side_gather(tmp_path):
    """Positive control for the driver guard: a whole-field ``.values`` on the client must trip it.

    A buffer larger than the gather is allocated and freed first, so the process's high-water
    mark already exceeds what the gather adds: a guard on ``ru_maxrss`` would not move, one on
    current RSS does. Every buffer is > 32 MB and filled, so glibc maps it and unmaps it on free.
    """
    store = tmp_path / "small.zarr"
    input_bytes = write_synthetic_sst(store, 4, ny=64, nx=180)  # 1461 days x 11520 cells x 4 B = 67 MB
    sst = xr.open_zarr(store).sst
    high_water = np.ones(4 * input_bytes // 8)  # float64: 4 x the gather
    del high_water
    with DriverRSSMonitor() as driver:
        gathered = sst.values
        time.sleep(1.0)  # held while the monitor samples
    print(f"\ngather {format_bytes(gathered.nbytes)}, driver RSS growth {format_bytes(driver.growth)}")
    assert gathered.nbytes == input_bytes
    assert driver.growth >= DRIVER_FRACTION * input_bytes, (
        f"a {format_bytes(gathered.nbytes)} client-side gather moved the guard by only {format_bytes(driver.growth)}"
    )
