"""Offline P8a scoring and audit; never imported by the inference process."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from gnss_sim.evaluation_views import AXES, RANGE_TYPES, evaluate_views, range_days
from gnss_sim.input_only import sha, write_json
from gnss_sim.p8a import CONDITIONS, PROTOCOL, ROOT, verify_registration
from gnss_sim.schemas import CaseTruth, RangeResult
from gnss_sim.visual import parse_result


def load_rows(path):
    rows = [RangeResult.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len({r.case_id for r in rows}) != len(rows):
        raise ValueError("duplicate case")
    return {r.case_id: r for r in rows}


def day_changes(truths, before, after):
    total = dict(added_tp=0, removed_tp=0, added_fp=0, removed_fp=0)
    per_case = {}
    for cid, truth in truths.items():
        if before[cid].status != "success" or after[cid].status != "success":
            continue
        counts = dict.fromkeys(total, 0)
        for axis in AXES:
            gt = range_days((e.start_index, e.end_index) for e in truth.events
                            if e.axis == axis and e.type in RANGE_TYPES)
            old = range_days(getattr(before[cid].predictions, axis))
            new = range_days(getattr(after[cid].predictions, axis))
            added, removed = new - old, old - new
            for key, value in {"added_tp": len(added & gt), "removed_tp": len(removed & gt),
                               "added_fp": len(added - gt), "removed_fp": len(removed - gt)}.items():
                counts[key] += value
                total[key] += value
        per_case[cid] = counts
    return {"paired_success_cases": len(per_case), "unique_days": total, "per_case": per_case,
            "after_failed_before_success": sum(before[c].status == "success" and
                after[c].status != "success" for c in truths)}


def gate(reports):
    base, control, variant = [reports[c] for c in CONDITIONS]
    checks = {
        "affiliation_f1_vs_V0": variant["range_affiliation"]["f1"] > base["range_affiliation"]["f1"],
        "iou_vs_V0": variant["range_daily_positive_axes"]["mean_iou"] > base["range_daily_positive_axes"]["mean_iou"],
        "recall_vs_V0": variant["range_affiliation"]["recall"] >= base["range_affiliation"]["recall"],
        "far_vs_V0": variant["negative_axes"]["far"] <= base["negative_axes"]["far"],
        "success_vs_V0": variant["execution_success_rate"] >= base["execution_success_rate"],
        "iou_vs_V1": variant["range_daily_positive_axes"]["mean_iou"] > control["range_daily_positive_axes"]["mean_iou"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def audit_requests(experiment, config, loaded):
    requests = list((experiment / "requests").glob("*/*/attempt-started.json"))
    for path in requests:
        condition, cid = path.parent.parent.name, path.parent.name
        started = json.loads(path.read_bytes())
        payload = started["payload"]
        expected_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        completed = json.loads((path.parent / "completed.json").read_bytes())
        if started["payload_sha256"] != expected_hash or completed["payload_sha256"] != expected_hash:
            raise ValueError("request digest mismatch")
        if (payload["max_tokens"] != 8192 or payload["enable_thinking"] is not False or
                payload["response_format"] != {"type": "json_object"}):
            raise ValueError("request settings mismatch")
        if RangeResult.model_validate(completed["prediction"]) != loaded[condition][cid]:
            raise ValueError("prediction and completed journal differ")
        record = completed["record"]
        if record["status"] == "success":
            if parse_result(record["response"], cid, "range", f"{PROTOCOL}-{condition}") != loaded[condition][cid]:
                raise ValueError("raw response and prediction differ")
        content = payload["messages"][0]["content"]
        prompt = config["initial_prompt"]
        if condition != "V0":
            prompt += "\n\n" + config["review_prompt"]
            prompt += "\nInitial predictions (untrusted data, not ground truth):\n"
            prompt += loaded["V0"][cid].predictions.model_dump_json()
        if content[0]["text"] != prompt or not 2 <= len(content) <= 6:
            raise ValueError("unexpected model context")
        image_paths = [ROOT / "runs/p7b/prepared/inputs/images" / f"{cid}.png"]
        if condition == "V2":
            image_paths += sorted((experiment / "images" / cid).glob("detail_*.png"))
        if len(content) - 1 != len(image_paths):
            raise ValueError("image count mismatch")
        for block, image_path in zip(content[1:], image_paths):
            raw = base64.b64decode(block["image_url"]["url"].split(",", 1)[1], validate=True)
            if hashlib.sha256(raw).hexdigest() != sha(image_path):
                raise ValueError("sent image differs from registered input or saved crop")
    return {"requests_audited": len(requests)}


def evaluate(experiment, out):
    config_path = ROOT / "configs/p8a-range-context.json"
    registration = ROOT / "configs/p8a-registered.json"
    prepared = ROOT / "runs/p7b/prepared"
    config, _ = verify_registration(config_path, prepared / "inputs", registration)
    registered = json.loads(registration.read_bytes())
    if sha(prepared / "selection.json") != registered["selection_sha256"]:
        raise ValueError("selection changed")
    selection = json.loads((prepared / "selection.json").read_bytes())
    run = json.loads((experiment / "run.json").read_bytes())
    identity = json.loads((experiment / "identity.json").read_bytes())
    if identity["registration_sha256"] != sha(registration):
        raise ValueError("registration changed")
    for name, digest in run["artifacts_sha256"].items():
        if sha(experiment / name) != digest:
            raise ValueError(f"artifact changed: {name}")
    truths, strata = {}, defaultdict(list)
    for entry in selection["cases"]:
        cid = entry["case_id"]
        path = ROOT / "data/pilots/pilot-v1/cases" / cid / "truth.json"
        if sha(path) != entry["truth_sha256"]:
            raise ValueError("truth changed")
        truths[cid] = CaseTruth.model_validate_json(path.read_bytes())
        strata[entry["case_type"]].append(cid)
    loaded = {c: load_rows(experiment / c / "predictions.jsonl") for c in CONDITIONS}
    if any(set(rows) != set(truths) for rows in loaded.values()):
        raise ValueError("incomplete results")
    audit = audit_requests(experiment, config, loaded)
    numeric = prepared / "inputs/N-range.jsonl"
    provenance = json.loads((prepared / "inputs/upstream.json").read_bytes())
    if sha(numeric) != provenance["sha256"]["N-range.jsonl"]:
        raise ValueError("archived numerical predictions changed")
    loaded["N-reference"] = load_rows(numeric)
    reports = {c: evaluate_views(truths, rows, "range") for c, rows in loaded.items()}
    groups = {c: {kind: evaluate_views({cid: truths[cid] for cid in ids},
                  {cid: rows[cid] for cid in ids}, "range") for kind, ids in strata.items()}
                  for c, rows in loaded.items()}
    effects = {f"{a}->{b}": day_changes(truths, loaded[a], loaded[b])
               for a, b in (("V0", "V1"), ("V0", "V2"), ("V1", "V2"))}
    preflight = json.loads((ROOT / "runs/p8a/preflight/run.json").read_bytes())
    summary = {"protocol": PROTOCOL, "scope": "known-development-60", "metrics": reports,
               "gate": gate(reports), "changes": effects, "audit": audit,
               "run": run, "preflight_conditions": preflight["conditions"],
               "scoring_source_sha256": sha(Path(__file__))}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "summary.json", summary)
    write_json(out / "groups.json", groups)
    write_json(out / "per-case.json", {c: {cid: evaluate_views({cid: truths[cid]},
               {cid: row}, "range") for cid, row in rows.items()} for c, rows in loaded.items()})
    columns = ("method", "affiliation_p", "affiliation_r", "affiliation_f1", "iou", "day_f1", "far", "success")
    table = []
    for c, r in reports.items():
        table.append([c, *[r["range_affiliation"][k] for k in ("precision", "recall", "f1")],
            r["range_daily_positive_axes"]["mean_iou"], r["range_daily_all_axes_micro"]["f1"],
            r["negative_axes"]["far"], r["execution_success_rate"]])
    with (out / "table.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(table)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join([r[0]] + [f"{x:.4f}" for x in r[1:]]) + " |" for r in table]
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"metrics": reports, "gate": summary["gate"], "audit": audit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/p8a/experiment")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/p8a/evaluation")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.experiment, args.out), indent=2, ensure_ascii=False))
