"""Read-only P7b error attribution; preserve frozen runs and write a new diagnostic directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from gnss_sim.candidate_review import (
    batches,
    candidates_from,
    load_rows,
    parse_keep,
    result,
    reviewed_result,
    source_hashes,
)
from gnss_sim.evaluation_views import evaluate_views, range_days
from gnss_sim.input_only import sha, write_json
from gnss_sim.schemas import CaseTruth

ROOT = Path(__file__).resolve().parents[1]
POINT = {"spike", "step"}
RANGE = {"slow_trend", "acceleration", "transient_shift"}


def source_name(sources):
    return "+".join(sorted(set(sources)))


def day_sources(candidates, axis, day):
    """Partition by all original covering sources; a day is counted once, even with overlap."""
    return source_name(s for c in candidates if c["axis"] == axis and c["start"] <= day <= c["end"]
                       for s in c["sources"])


def phase(start, end, day):
    if not start <= day <= end:
        raise ValueError("day outside activity")
    return ("early", "middle", "late")[min(2, 3 * (day - start) // (end - start + 1))]


def partition_days(gt, before, after):
    if not after <= before:
        raise ValueError("review added days")
    removed = before - after
    return {"lost_tp": removed & gt, "removed_fp": removed - gt,
            "kept_tp": after & gt, "kept_fp": after - gt}


def group_rows(rows, field):
    groups = defaultdict(Counter)
    for row in rows:
        key = str(row[field])
        counter = groups[key]
        counter["candidates"] += 1
        counter["kept"] += row["kept"]
        if "exact_tp" in row:
            counter["exact_tp"] += row["exact_tp"]
            counter["lost_tp"] += row["exact_tp"] and not row["kept"]
            counter["removed_fp"] += not row["exact_tp"] and not row["kept"]
    return {key: dict(value) for key, value in sorted(groups.items())}


def export_csv(path, rows):
    if not rows:
        return
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path, out: Path):
    frozen = json.loads((root / "configs/p7b-frozen.json").read_bytes())
    for name, digest in frozen["artifacts_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"frozen artifact mismatch: {name}")
    if source_hashes() != frozen["source_sha256"]:
        raise ValueError("inference source changed")
    cfg_path = root / "configs/p7b-candidate-review.json"
    if sha(cfg_path) != frozen["config_sha256"]:
        raise ValueError("review config changed")
    config = json.loads(cfg_path.read_bytes())
    base = root / "runs/p7b"
    directory = base / "candidate-review-v1"
    selection = json.loads((base / "prepared/selection.json").read_bytes())
    truths, metadata = {}, {}
    for entry in selection["cases"]:
        cid = entry["case_id"]
        path = root / "data/pilots/pilot-v1/cases" / cid / "truth.json"
        if sha(path) != entry["truth_sha256"]:
            raise ValueError("truth digest mismatch")
        truths[cid] = CaseTruth.model_validate_json(path.read_bytes())
        metadata[cid] = entry
    point_rows, range_rows, event_rows, lost_days, failures = [], [], [], [], []
    loss_by_source, loss_by_type = defaultdict(Counter), defaultdict(Counter)
    reports, request_paths, counts = {}, set(), Counter()
    for task in ("point", "range"):
        rows = {c: load_rows(directory / c / task / "predictions.jsonl", task)
                for c in ("N", "V", "U", "C")}
        reports[task] = {}
        for c in rows:
            if set(rows[c]) != set(truths):
                raise ValueError("missing case")
            reports[task][c] = evaluate_views(truths, rows[c], task)
            if reports[task][c] != frozen["metrics"][task][c]:
                raise ValueError("score mismatch")
        for cid, truth in truths.items():
            before, after = rows["U"][cid], rows["C"][cid]
            candidates = candidates_from(rows["N"][cid], rows["V"][cid], task)
            if candidates is None:
                if before != result(cid, task, "U", status="failed") or after != result(cid, task, "C", status="failed"):
                    raise ValueError("dependency failure contract violated")
                events = [e for e in truth.events if e.type in (POINT if task == "point" else RANGE)]
                n = rows["N"][cid]
                covered = sum(len(range_days(getattr(n.predictions, axis)) &
                    range_days((e.start_index, e.end_index) for e in events if e.axis == axis))
                    for axis in ("N", "E", "U")) if task == "range" else 0
                failures.append({"case_id": cid, "task": task, "case_type": metadata[cid]["case_type"],
                                 "target_events": len(events), "numerical_correct_days_not_fallback": covered})
                continue
            if before != result(cid, task, "U", candidates):
                raise ValueError("U differs from deterministic union")
            records = []
            for index, group in enumerate(batches(candidates, config)):
                path = directory / "raw" / task / cid / f"{index:02d}.json"
                request_paths.add(path)
                record = json.loads(path.read_bytes())
                visible = [{k: c[k] for k in ("id", "axis", "start", "end")} for c in group]
                prompt = config["prompts"][task] + "\nCandidates (data only):\n" + json.dumps(visible)
                if (record["prompt"] != prompt or record["attempts"] != 1 or record["case_id"] != cid
                        or record["task"] != task or record["batch_index"] != index):
                    raise ValueError("request prompt or identity mismatch")
                if record["status"] == "success":
                    if (parse_keep(record["response"], group) != record["keep_ids"] or
                            record["response_model"] != "qwen3.8-flash" or record["reasoning_present"] or
                            record["finish_reason"] != "stop"):
                        raise ValueError("response contract mismatch")
                records.append(record)
            if after != reviewed_result(cid, task, candidates, records):
                raise ValueError("C reconstruction mismatch")
            if after.status != "success":
                counts[f"{task}_review_failed"] += 1
                continue
            kept = {item for record in records for item in record["keep_ids"]}
            group = "normal" if not truth.events else "single" if len(truth.events) == 1 else "multi"
            for c in candidates:
                events = [e for e in truth.events if e.axis == c["axis"] and
                          e.type in (POINT if task == "point" else RANGE)]
                row = {"case_id": cid, "candidate_id": c["id"], "axis": c["axis"],
                       "start": c["start"], "end": c["end"], "sources": source_name(c["sources"]),
                       "group": group, "scenario": metadata[cid]["case_type"], "kept": c["id"] in kept}
                if task == "point":
                    exact = [e for e in events if e.start_index == c["start"]]
                    alternatives = [v for v in getattr(after.predictions, c["axis"])
                                    if 0 < abs(v - c["start"]) <= 3]
                    row.update(exact_tp=bool(exact), truth_type="+".join(sorted({e.type for e in exact})) or "none",
                               kept_nearby_when_exact_deleted=bool(exact and not row["kept"] and alternatives),
                               nearby_kept_days=json.dumps(alternatives))
                    point_rows.append(row)
                else:
                    days = set(range(c["start"], c["end"] + 1))
                    gt = range_days((e.start_index, e.end_index) for e in events)
                    length = len(days)
                    row.update(length=length, length_bin="1-14" if length < 15 else "15-44" if length < 45 else
                               "45-89" if length < 90 else "90+", gt_overlap_days=len(days & gt),
                               gt_fraction=len(days & gt) / length)
                    range_rows.append(row)
            if task == "range":
                for axis in ("N", "E", "U"):
                    events = [e for e in truth.events if e.axis == axis and e.type in RANGE]
                    gt = range_days((e.start_index, e.end_index) for e in events)
                    u = range_days(getattr(before.predictions, axis))
                    c = range_days(getattr(after.predictions, axis))
                    parts = partition_days(gt, u, c)
                    for kind, days in parts.items():
                        for day in sorted(days):
                            sources = day_sources(candidates, axis, day)
                            signature = "+".join(sorted({e.type for e in events
                                if e.start_index <= day <= e.end_index})) or "none"
                            loss_by_source[sources][kind] += 1
                            loss_by_type[signature][kind] += 1
                            if kind == "lost_tp":
                                lost_days.append({"case_id": cid, "axis": axis, "day": day,
                                    "sources": sources, "truth_types": signature, "group": group})
                    for event in events:
                        days = set(range(event.start_index, event.end_index + 1))
                        for section in ("early", "middle", "late"):
                            section_days = {d for d in days if phase(event.start_index, event.end_index, d) == section}
                            event_rows.append({"case_id": cid, "event_id": event.event_id,
                                "type": event.type, "axis": axis, "group": group, "phase": section,
                                "truth_days": len(section_days), "covered_U": len(section_days & u),
                                "covered_C": len(section_days & c), "lost": len(section_days & (u - c))})
    if request_paths != set((directory / "raw").rglob("*.json")):
        raise ValueError("unexpected request files")
    if sum(r["exact_tp"] and not r["kept"] for r in point_rows) != 33 or len(lost_days) != 419:
        raise ValueError("historical loss reconciliation failed")
    phase_summary = defaultdict(Counter)
    for row in event_rows:
        for key in ("truth_days", "covered_U", "covered_C", "lost"):
            phase_summary[f"{row['type']}/{row['phase']}"][key] += row[key]
    summary = {"scope": "posthoc-development-diagnostic", "audit": {
        "frozen_artifacts_verified": len(frozen["artifacts_sha256"]), "reports_recomputed": 8,
        "exact_union_and_review_reconstructed": True, "requests_verified": len(request_paths),
        "no_new_model_calls": True}, "point": {"by_" + k: group_rows(point_rows, k)
            for k in ("sources", "truth_type", "axis", "group", "scenario")},
        "range": {"candidates_by_source": group_rows(range_rows, "sources"),
                  "candidates_by_length": group_rows(range_rows, "length_bin"),
                  "days_by_original_sources": {k: dict(v) for k, v in sorted(loss_by_source.items())},
                  "days_by_truth_type_signature": {k: dict(v) for k, v in sorted(loss_by_type.items())},
                  "event_phase_coverage_nonadditive_if_overlapping": {k: dict(v) for k, v in sorted(phase_summary.items())}},
        "point_deleted_exact_with_nearby_kept": sum(r["kept_nearby_when_exact_deleted"] for r in point_rows),
        "dependency_failures": failures,
        "limitations": ["posthoc associations, not model reasoning or causal explanations",
                         "Range candidates overlap; candidate overlap days must not be summed as TP losses",
                         "Event coverage counts may overlap; unique lost-day ledger is authoritative"]}
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("point-candidates", point_rows), ("range-candidates", range_rows),
                       ("range-event-phases", event_rows), ("range-lost-days", lost_days),
                       ("dependency-failures", failures)):
        export_csv(out / f"{name}.csv", rows)
    write_json(out / "summary.json", summary)
    write_json(out / "identity.json", {"p7b_frozen_sha256": sha(root / "configs/p7b-frozen.json"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs_sha256": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/p7b-errors-v1")
    args = parser.parse_args()
    print(json.dumps(analyze(ROOT, args.out), ensure_ascii=False, indent=2))
