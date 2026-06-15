#!/usr/bin/env python
"""Collate the squeeze summaries into the results table.

Reads every ``*_summary.json`` produced by the leg scripts and prints a markdown table.
Legs with ``status == "started"`` were killed by the wall clock before finishing and are
reported as such rather than being silently dropped.
"""

import argparse
import json
import re
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


def _measured(row, prefix: str, bytes_key: str, width_key: str = "spill_workers_sampled") -> bool:
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

    `width_key` names the coverage witness, because not every quantity is witnessed by the
    same round trip.  The disk and managed series share `SpillSampler`'s samples, so they share
    `spill_workers_sampled`.  The process HISTORY is a separate `client.run` at the end of the
    run, and a coverage guarantee earned by the sampler says nothing about how wide that later
    call was, so it carries its own `prochist_workers_sampled` -- an OBSERVED width, unlike the
    sampler's, which is the width every counted sample was required to have.
    """
    if row.get(f"{prefix}_unmeasured") is not False or row.get(bytes_key) is None:
        return False
    samples = row.get(f"{prefix}_samples_ok")
    if not _is_count(samples) or samples == 0:
        return False
    width, observed, requested = (
        row.get(width_key),
        row.get("n_workers"),
        row.get("n_workers_requested"),
    )
    return _is_count(width) and width == observed and width == requested


def prochist_measured(row) -> bool:
    """Decide whether the harvested per-worker RSS maximum may be printed.

    Three clauses, and keeping them in ONE function is the point: `_measured` alone licenses a
    row whose monitor deque had WRAPPED, and a second consumer that forgot the extra conjunct
    would print a tail maximum as the run's maximum.

    * its OWN observed width, not the sampler's -- the harvest is a separate `client.run`.
    * `covers_whole_run` -- `count <= maxlen`, i.e. the deque never wrapped.
    * the series SPAN covers the run, at BOTH ends. A worker restart resets `count`, so
      `covers_whole_run` reads True over a post-restart tail; only the span catches that. The
      upper bound is not decoration: a lower bound alone accepts an absurd span, which is a
      figure nobody measured passing a check nobody can fail.
    * the span was produced by the run, or says so if it was not.
    """
    if not _measured(row, "prochist", "prochist_max_worker_bytes", "prochist_workers_sampled"):
        return False
    if row.get("prochist_covers_whole_run") is not True:
        return False
    if row.get("prochist_span_s_source") not in ("run", "npz-backfill"):
        return False
    span, elapsed = row.get("prochist_span_s"), row.get("elapsed_s")
    if not isinstance(span, (int, float)) or not isinstance(elapsed, (int, float)) or elapsed <= 0:
        return False
    # Two-sided, with an ABSOLUTE floor on the slack: the harvest happens after the work and the
    # monitor's first sample lands up to one 500 ms tick after startup, so an exact match never
    # holds, but on a short run a purely proportional slack would be tighter than that fixed
    # offset. 3 % of a 1621 s run is 49 s, so a restart inside the first ~3 % still slips
    # through; that residue is stated rather than hidden.
    slack = max(2.0, 0.03 * elapsed)
    return elapsed - slack <= span <= elapsed + slack


def input_reads(row, prefix: str = "open_dataset-sst"):
    """Completions of ONE named input-reading prefix, or why the figure may not be read.

    `taskcount_by_prefix` counts completed tasks per scheduler task prefix, cumulatively over
    the whole leg. Three separate things can make that number uninterpretable, and each is
    named rather than folded into a bare count:

    * a leg that did not COMPLETE. Falsifier gate 5, finding 9: a leg that raised
      `ConfigurationError` after 2.2 s, having computed nothing of the science, still reported
      `taskcount_unmeasured` False with 50 completions over 8 prefixes -- those are the setup
      graph. "The harvest answered" is not "the graph ran".
    * a leg with `nanny_memory_events > 0`. A restarted worker's lost tasks are re-run, and
      this counter counts that re-execution with the same increment as a mode's recompute.
    * summing across prefixes. `streaming` re-reads its own staged zarrs, which also carry
      `open_dataset*` prefixes (`compute_mode.py` stages `dat_anomaly`, `thresholds`,
      `dat_stn`), so an `open_dataset*` SUM is not comparable across modes. One named prefix,
      passed in, or nothing.

    A leg predating the instrument has no `taskcount_unmeasured` key at all; ABSENT reads
    UNMEASURED, never 0, for the same reason the spill column does (D-038).

    NOTE the figure this returns is NOT yet licensed as "how many times the anomaly graph ran".
    Falsifier gate 5, finding 10 falsified that reading of `open_dataset-sst`: dask fuses the
    prefix away in a plain `.sum()` graph and it survives only where a rechunk blocks fusion,
    so its count is not one-per-graph-execution. The column reports what was counted; it does
    not claim what the count means.
    """
    if row.get("taskcount_unmeasured") is not False:
        return "UNMEASURED"
    if row.get("outcome") != "completed":
        return f"VOID ({row.get('outcome')})"
    if row.get("nanny_memory_events"):
        return f"VOID ({row.get('nanny_memory_events')} nanny restarts)"
    by_prefix = row.get("taskcount_by_prefix") or {}
    if prefix not in by_prefix:
        return "no such prefix"
    return str(by_prefix[prefix])


def boundary_taskcount(row, outdir) -> str:
    """Return the task completions counted INSIDE one `preprocess_data` call, or why they are void.

    This column exists because the leg-end count answers a different question. A leg is one
    `preprocess_data` call plus however many consumers the harness runs, and
    `detect_gridded.py:fingerprint` runs three, so a whole-leg lazy/persist ratio measures
    the HARNESS (D-046, D-047). The boundary reading is the library's own fan-out.

    Its voiding rules are its OWN, deliberately not the leg's:

    * `outcome != "completed"` does NOT void it. The boundary is read before `fingerprint`,
      so a leg later killed by its deadline still has a valid boundary reading -- that is the
      whole reason the reading is written to its own file the moment it is taken. Voiding it
      on the leg's outcome would throw away the one number such a leg does deliver.
    * `boundary_nanny_memory_events > 0` DOES void it, and this is the reading that matters
      rather than the leg's total: a worker restart before the boundary makes dask re-run the
      lost tasks, and that re-execution increments the same counter as a mode's recompute
      (D-047). Restarts AFTER the boundary cannot contaminate it.
    * `boundary_settle_quiescent` false voids it. A count read while work is still in flight
      is not a boundary count, and the bias falls on `persist` alone, which is the mode
      `lazy` is compared against.
    * `boundary_settle_futures_error` not None voids it, on EITHER mode. A walk that raised
      reports zero futures, which is indistinguishable in the number from "nothing was
      persisted" -- D-052, where `hasattr` on a datetime64 coord raised and silently disabled
      D-049's exact wait for three whole replicates.
    * `boundary_settle_futures_waited == 0` voids a **persist** row and only a persist row.
      The prereg's mechanism discriminator is that persist has futures to wait on and lazy has
      none; a persist leg reading zero means the exact wait was a no-op, the undercount is not
      excluded, and the ratio is INCONCLUSIVE. Lazy's zero is the expected reading, never a
      fault. This rule lived only in the prereg's prose until 27325013 spent a whole run
      satisfying every enforced rule while failing this one.

    As everywhere else in this file, an ABSENT key reads UNMEASURED and never 0.
    """
    block = row
    if outdir is not None:
        label, phase = row.get("label"), row.get("boundary_phase", "preprocess_data")
        candidate = Path(outdir) / f"{label}_boundary_{phase}.json"
        if candidate.exists():
            try:
                # The standalone file outlives a summary the wall clock never let us write.
                block = json.loads(candidate.read_text())
            except json.JSONDecodeError:
                pass
    if block.get("boundary_taskcount_unmeasured") is not False:
        return "UNMEASURED"
    if block.get("boundary_nanny_memory_events"):
        return f"VOID ({block['boundary_nanny_memory_events']} nanny restarts by the boundary)"
    if block.get("boundary_settle_quiescent") is not True:
        return f"VOID (not quiescent: {block.get('boundary_settle_pending')})"
    ferr = block.get("boundary_settle_futures_error")
    if ferr:
        return f"VOID (futures walk failed: {ferr})"
    if row.get("compute_mode") == "persist":
        # ABSENT reads UNMEASURED, never 0, as everywhere else in this file: a leg predating
        # the instrument has no such key, and `block.get(...) == 0` would wave it through.
        waited = block.get("boundary_settle_futures_waited")
        if not _is_count(waited):
            return "UNMEASURED"
        if waited == 0:
            return "VOID (persist waited on 0 futures: exact wait was a no-op, undercount not excluded)"
    total = block.get("boundary_taskcount_total_completed")
    if not _is_count(total):
        return "UNMEASURED"
    return f"{total} / {block.get('boundary_taskcount_n_prefixes', '?')}p"


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
        "| spill | managed/worker | rss/worker | open_dataset-sst | inside preprocess_data "
        "| nanny | wall |\n"
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
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
        # The per-worker RSS maximum, harvested from dask's own 500 ms SystemMonitor history:
        # the quantity `memory.spill` (0.7) and `memory.pause` (0.8) threshold on, which
        # `managed` beside it is NOT.  A tail-only maximum is a lower bound rather than the
        # run's maximum, and `prochist_measured` is where all three ways of getting a tail are
        # rejected together.
        # The dagger travels WITH the row. A figure whose span was reduced from the npz after
        # the run is legitimate but is not what the run wrote, and a marker parked in a sibling
        # key no consumer reads would be lost the first time this table is copied.
        if prochist_measured(r):
            proc = fmt(r.get("prochist_max_worker_bytes"), 2)
            if r.get("prochist_span_s_source") != "run":
                proc += f" (span {r.get('prochist_span_s_source')})"
        else:
            proc = "UNMEASURED"
        wall = r.get("elapsed_s")
        print(
            f"| {r.get('label')} | {r.get('compute_mode')} | {r.get('n_time', '-')} | "
            f"{fmt(r.get('input_bytes'))} | {fmt(r.get('input_chunk_bytes'), 3)} | "
            f"{fmt(r.get('cluster_memory_limit_bytes'))} | {outcome} | "
            f"{fmt(r.get('peak_cluster_bytes'))} | {fmt(pinned, 3)} | {spill} | {managed} | {proc} | "
            f"{input_reads(r)} | {boundary_taskcount(r, args.outdir)} | "
            f"{r.get('nanny_memory_events', '-')} | {f'{wall:.0f} s' if wall else '-'} |"
        )

    print("\n### Cross-mode fingerprints (equivalence legs)\n")
    print("| leg | mode | n_extreme_cells | anomaly_checksum | thresholds_checksum | id_field_sum | n_events | n_merges |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    # A leg that did not complete is LISTED, never dropped. Falsifier gate 9 finding 15: the
    # old `continue` made a deadlined leg vanish from this table with no marker, so HARD
    # VOIDING RULE 4 (the two modes must agree on the fingerprints) silently went unchecked
    # while the boundary count from the same leg was still reported as good.
    for r in rows:
        if r.get("outcome") != "completed":
            print(
                f"| {r.get('label')} | {r.get('compute_mode')} | "
                f"NOT COMPUTED (outcome={r.get('outcome', '?')}) | - | - | - | - | - |"
            )
            continue
        print(
            f"| {r.get('label')} | {r.get('compute_mode')} | {r.get('n_extreme_cells', '-')} | "
            f"{r.get('anomaly_checksum', '-')} | {r.get('thresholds_checksum', '-')} | "
            f"{r.get('id_field_sum', '-')} | "
            f"{r.get('n_events', '-')} | {r.get('n_merges', '-')} |"
        )

    _rule4_verdict(rows)


# HARD VOIDING RULE 4 of prereg_q5_boundary.txt: the persist and lazy legs of one comparison
# must agree EXACTLY on n_extreme_cells, anomaly_checksum and thresholds_checksum. The rule
# lived only in the prereg's prose and nothing computed it (falsifier gate 9 finding 15).
RULE4_KEYS = ("n_extreme_cells", "anomaly_checksum", "thresholds_checksum")


def _rule4_verdict(rows):
    """Print the cross-mode agreement verdict, and say plainly when it could not be taken."""
    print("\n### HARD VOIDING RULE 4 -- cross-mode fingerprint agreement\n")
    groups = {}
    for r in rows:
        label = str(r.get("label", ""))
        # q5_dg_persist / q5_dg_lazy -> comparison key q5_dg
        stem = re.sub(r"_(persist|lazy|streaming)$", "", label)
        groups.setdefault(stem, []).append(r)

    any_pair = False
    for stem, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        any_pair = True
        done = [m for m in members if m.get("outcome") == "completed"]
        missing = [m for m in members if m.get("outcome") != "completed"]
        if missing:
            names = ", ".join(f"{m.get('label')} (outcome={m.get('outcome', '?')})" for m in missing)
            print(
                f"- **{stem}: NOT CHECKED** -- {names} produced no fingerprints. "
                f"Rule 4 is UNSATISFIED, not passed. A boundary count from these legs still "
                f"stands on its own voiding rules, but it carries no cross-mode equivalence check."
            )
            continue
        disagreements = []
        for key in RULE4_KEYS:
            vals = {m.get("label"): m.get(key) for m in done}
            if any(v is None for v in vals.values()):
                disagreements.append(f"{key}: MISSING {vals}")
            elif len(set(map(repr, vals.values()))) != 1:
                disagreements.append(f"{key}: {vals}")
        if disagreements:
            print(f"- **{stem}: FAIL** -- the modes disagree. " + "; ".join(disagreements))
        else:
            shown = ", ".join(f"{k}={done[0].get(k)}" for k in RULE4_KEYS)
            print(f"- **{stem}: PASS** -- {len(done)} legs agree exactly on {shown}")
    if not any_pair:
        print("- no comparison has two or more legs in this directory; rule 4 not applicable")


if __name__ == "__main__":
    main()
