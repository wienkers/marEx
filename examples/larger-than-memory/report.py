#!/usr/bin/env python
"""Collate the squeeze summaries into the results table.

Reads every ``*_summary.json`` produced by the leg scripts and prints a markdown table.
Legs with ``status == "started"`` were killed by the wall clock before finishing and are
reported as such rather than being silently dropped.
"""

import argparse
import json
from pathlib import Path

GB = 1e9


def fmt(value, digits=1, suffix=" GB"):
    """Format a byte count in GB, or '-' if value is missing or non-numeric."""
    if value is None:
        return "-"
    try:
        return f"{float(value) / GB:.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "-"


def _is_count(value) -> bool:
    """Report whether `value` is a non-negative integer, excluding bool.

    `isinstance(True, int)` is True, so a bool would otherwise pass as a count.
    """
    return type(value) is int and value >= 0


def _measured(row, prefix: str, bytes_key: str) -> bool:
    """Decide whether `row[bytes_key]` is a figure a sampler was entitled to have printed.

    Every clause is load-bearing, and each one is a defect that reached the README:

    * `<prefix>_unmeasured is False` -- an ABSENT key means a leg predating the 2026-09-08
      probe fix, whose recorded 0 came from a probe that returned 0 on failure (D-038/D-041).
    * the byte count is not None -- a null with `unmeasured` False otherwise renders "-",
      which reads as a formatting gap rather than as an absence of measurement.
    * `samples_ok` is a positive non-bool int -- a sampler that never sampled reports the same
      `max_disk = 0` as one that sampled throughout and saw nothing.
    * `workers_sampled == n_workers == n_workers_requested` -- a sampler that reached 1 of 4
      workers under-counts exactly like a failed read. Be honest about what this clause does:
      on any summary written by the current `execute()` all three are the same integer, because
      `build_cluster` ABORTs when they disagree, so PROSPECTIVELY it is a restatement and the
      real guard is `len(per_worker) != workers_expected` inside `SpillSampler.run`. It earns
      its place RETROSPECTIVELY: every summary written before 2026-09-08 is missing at least
      one of the three keys and must read UNMEASURED rather than be granted a coverage
      guarantee that its build could not make.
    """
    if row.get(f"{prefix}_unmeasured") is not False or row.get(bytes_key) is None:
        return False
    samples = row.get(f"{prefix}_samples_ok")
    if not _is_count(samples) or samples == 0:
        return False
    width, observed, requested = (
        row.get("spill_workers_sampled"),
        row.get("n_workers"),
        row.get("n_workers_requested"),
    )
    return _is_count(width) and width == observed and width == requested


def main() -> None:
    """Collate the outdir's summary JSONs into the feasibility and equivalence tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.outdir).glob("*_summary.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"<!-- unreadable: {path} -->")

    header = (
        "| leg | mode | n_time | input | input chunk | cluster RAM | outcome | peak | pinned "
        "| spill | managed/worker | nanny | wall |\n"
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"
    )
    print(header)
    for r in rows:
        # status "started" means the breadcrumb was written but the leg never finished:
        # either it is still running, or the wall clock killed it. Only squeue can tell
        # those apart, so name both rather than asserting the wrong one.
        if r.get("outcome"):
            outcome = r["outcome"]
        elif r.get("status") == "started":
            outcome = f"RUNNING-or-WALL-KILLED (job {r.get('slurm_job_id', '?')})"
        else:
            outcome = "-"
        pinned = (r.get("persist") or {}).get("total_bytes")
        # A leg predating the 2026-09-08 probe fix has no `spill_unmeasured` key at all, and its
        # recorded 0 is an artefact of a probe that returned 0 on failure -- so ABSENT must read
        # UNMEASURED, not "0.00 GB". Only an explicit False, written by a sampler that counted
        # its successful samples, licenses printing a number.
        # `spill_samples_ok` is the licence: a build without it could not tell a sampler that
        # saw nothing from one that never ran, so its 0 is not a measurement either.
        spill = fmt(r.get("spill_max_disk_bytes"), 2) if _measured(r, "spill", "spill_max_disk_bytes") else "UNMEASURED"
        # The per-worker managed series is the quantity `distributed.worker.memory.target`
        # thresholds on. `peak` beside it is MemorySampler's cluster-SUMMED PROCESS series;
        # the two are not comparable and the column exists so that they stop being compared.
        managed = fmt(r.get("managed_max_worker_bytes"), 2) if _measured(r, "managed", "managed_max_worker_bytes") else "UNMEASURED"
        wall = r.get("elapsed_s")
        print(
            f"| {r.get('label')} | {r.get('compute_mode')} | {r.get('n_time', '-')} | "
            f"{fmt(r.get('input_bytes'))} | {fmt(r.get('input_chunk_bytes'), 3)} | "
            f"{fmt(r.get('cluster_memory_limit_bytes'))} | {outcome} | "
            f"{fmt(r.get('peak_cluster_bytes'))} | {fmt(pinned, 3)} | {spill} | {managed} | "
            f"{r.get('nanny_memory_events', '-')} | {f'{wall:.0f} s' if wall else '-'} |"
        )

    print("\n### Cross-mode fingerprints (equivalence legs)\n")
    print("| leg | mode | n_extreme_cells | anomaly_checksum | id_field_sum | n_events | n_merges |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        if r.get("outcome") != "completed":
            continue
        print(
            f"| {r.get('label')} | {r.get('compute_mode')} | {r.get('n_extreme_cells', '-')} | "
            f"{r.get('anomaly_checksum', '-')} | {r.get('id_field_sum', '-')} | "
            f"{r.get('n_events', '-')} | {r.get('n_merges', '-')} |"
        )


if __name__ == "__main__":
    main()
