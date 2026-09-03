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
        "| leg | mode | n_time | input | input chunk | cluster RAM | outcome | peak | pinned | spill | nanny | wall |\n"
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"
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
        wall = r.get("elapsed_s")
        print(
            f"| {r.get('label')} | {r.get('compute_mode')} | {r.get('n_time', '-')} | "
            f"{fmt(r.get('input_bytes'))} | {fmt(r.get('input_chunk_bytes'), 3)} | "
            f"{fmt(r.get('cluster_memory_limit_bytes'))} | {outcome} | "
            f"{fmt(r.get('peak_cluster_bytes'))} | {fmt(pinned, 3)} | {fmt(r.get('spill_max_disk_bytes'), 2)} | "
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
