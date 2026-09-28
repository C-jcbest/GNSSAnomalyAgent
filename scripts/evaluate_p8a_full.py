"""Separate offline scoring/audit for the explicitly requested 300-case extension."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from gnss_sim import p8a
from gnss_sim.evaluation_views import evaluate_views
from gnss_sim.input_only import sha, write_json
from gnss_sim.p8a_full import DEFAULT, ROOT, validate
from gnss_sim.schemas import CaseTruth, RangeResult
from gnss_sim.visual import parse_result


def load_rows(path):
    rows = [RangeResult.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len({r.case_id for r in rows}):
        raise ValueError("duplicate case")
    return {r.case_id: r for r in rows}


def main():
    registered, _ = validate(DEFAULT)
    directory = DEFAULT / "experiment"
    run = json.loads((directory / "run.json").read_bytes())
    for name, digest in run["artifacts_sha256"].items():
        if sha(directory / name) != digest:
            raise ValueError("extension artifact changed")
    manifest_path = ROOT / "data/pilots/pilot-v1/manifest.json"
    if sha(manifest_path) != registered["pilot_manifest_sha256"]:
        raise ValueError("evaluation pilot changed")
    manifest = json.loads(manifest_path.read_bytes())
    truths, kinds = {}, {}
    for entry in manifest["cases"]:
        cid = entry["case_id"]
        path = manifest_path.parent / "cases" / cid / "truth.json"
        if sha(path) != entry["truth_sha256"]:
            raise ValueError("truth changed")
        truths[cid] = CaseTruth.model_validate_json(path.read_bytes())
        kinds[cid] = entry["case_type"]
    loaded = {c: load_rows(directory / c / "predictions.jsonl") for c in p8a.CONDITIONS}
    cached = set(registered["cached_cases"])
    for c, rows in loaded.items():
        if rows.keys() != truths.keys():
            raise ValueError("incomplete method results")
        old = load_rows(ROOT / "runs/p8a/experiment" / c / "predictions.jsonl")
        if any(rows[cid] != old[cid] for cid in cached):
            raise ValueError("cached results changed")
    config = p8a.load_config(ROOT / "configs/p8a-range-context.json")
    audited, failures = 0, []
    for c in p8a.CONDITIONS:
        for cid in truths:
            started = directory / "requests" / c / cid / "attempt-started.json"
            if not started.exists():
                if c == "V0" or loaded["V0"][cid].status != "failed" or loaded[c][cid].status != "failed":
                    raise ValueError("missing request without upstream failure")
                continue
            saved = json.loads(started.read_bytes())
            completed = json.loads(started.with_name("completed.json").read_bytes())
            paths = [DEFAULT / "inputs/images" / f"{cid}.png"]
            if c == "V2":
                paths += sorted((directory / "images" / cid).glob("detail_*.png"))
            expected = p8a.payload_for(config, c, [p.read_bytes() for p in paths], loaded["V0"][cid])
            digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
            if saved["payload"] != expected or saved["payload_sha256"] != digest or completed["payload_sha256"] != digest:
                raise ValueError("request/prompt/image contract mismatch")
            if RangeResult.model_validate(completed["prediction"]) != loaded[c][cid]:
                raise ValueError("journal/output mismatch")
            record = completed["record"]
            if record["status"] == "success":
                if parse_result(record["response"], cid, "range", f"{p8a.PROTOCOL}-{c}") != loaded[c][cid]:
                    raise ValueError("response parse mismatch")
            else:
                failures.append({"condition": c, "case_id": cid, "cached": cid in cached,
                    "error_type": record.get("error_type"), "finish_reason": record.get("finish_reason")})
            audited += 1
    # Numerical reference is deterministic, frozen, already available for all 300 cases.
    upstream = json.loads((ROOT / "runs/p7b/prepared/inputs/upstream.json").read_bytes())
    reference = upstream["provenance"]["N-range.jsonl"]
    numerical = ROOT / reference["source"]
    if sha(numerical) != reference["sha256"]:
        raise ValueError("archived numerical source changed")
    loaded["N-reference"] = load_rows(numerical)
    if loaded["N-reference"].keys() != truths.keys():
        raise ValueError("numerical case set mismatch")
    reports = {c: evaluate_views(truths, rows, "range") for c, rows in loaded.items()}
    groups = {c: {kind: evaluate_views({cid: t for cid, t in truths.items() if kinds[cid] == kind},
        {cid: r for cid, r in rows.items() if kinds[cid] == kind}, "range")
        for kind in sorted(set(kinds.values()))} for c, rows in loaded.items()}
    by_phase = {phase: {c: evaluate_views({cid: truths[cid] for cid in ids},
        {cid: rows[cid] for cid in ids}, "range") for c, rows in loaded.items()}
        for phase, ids in (("cached_60", cached), ("new_240", set(truths) - cached))}
    out = DEFAULT / "evaluation"
    out.mkdir(exist_ok=False)
    summary = {"scope": "known-development-300; cached60+new240; not independent test",
               "metrics": reports, "by_phase": by_phase, "failures": failures,
               "audit": {"artifacts": len(run["artifacts_sha256"]), "requests": audited,
                         "cached_cases_predictions_unchanged": 60},
               "registration_sha256": sha(DEFAULT / "registered.json"),
               "run_sha256": sha(directory / "run.json"),
               "scoring_source_sha256": sha(Path(__file__))}
    write_json(out / "summary.json", summary)
    write_json(out / "groups.json", groups)
    write_json(out / "per-case.json", {c: {cid: evaluate_views({cid: truths[cid]},
        {cid: row}, "range") for cid, row in rows.items()} for c, rows in loaded.items()})
    cols = ["method", "affiliation_p", "affiliation_r", "affiliation_f1", "iou", "day_f1", "far", "successful_cases"]
    table = [[c, *[r["range_affiliation"][k] for k in ("precision", "recall", "f1")],
        r["range_daily_positive_axes"]["mean_iou"], r["range_daily_all_axes_micro"]["f1"],
        r["negative_axes"]["far"], r["successful_cases"]] for c, r in reports.items()]
    with (out / "table.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(cols)
        writer.writerows(table)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines += ["| " + " | ".join([r[0]] + [f"{x:.4f}" for x in r[1:-1]] + [str(r[-1])]) + " |" for r in table]
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(summary["audit"]))


if __name__ == "__main__":
    main()
