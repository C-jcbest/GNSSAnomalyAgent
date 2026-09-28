"""P7b packaging and separate evaluation. Inference lives in candidate_review.py."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from gnss_sim.candidate_review import PROTOCOL, TASKS, load_rows, preflight, run
from gnss_sim.input_only import read_inputs, run_numerical, sha, write_json, write_rows

ROOT = Path(__file__).resolve().parents[2]


def select_cases(manifest):
    from gnss_sim.schemas import CASE_TYPES, SCENARIO_TYPES

    groups = defaultdict(list)
    for entry in sorted(manifest["cases"], key=lambda item: item["case_id"]):
        kind = entry["case_type"]
        key = ((kind, entry["axis"], entry["sign"]) if kind in CASE_TYPES[1:] else (kind,))
        groups[key].append(entry)
    expected = {("normal",): 6}
    expected.update({(kind, axis, sign): 1 for kind in CASE_TYPES[1:]
                     for axis in ("N", "E", "U") for sign in ("positive", "negative")})
    expected.update({(kind,): 4 for kind in SCENARIO_TYPES})
    if set(groups) != set(expected) or any(len(groups[k]) < count for k, count in expected.items()):
        raise ValueError("pilot strata incomplete")
    return sorted([entry for key, count in expected.items() for entry in groups[key][:count]],
                  key=lambda entry: entry["case_id"])


def prepare(pilot: Path, out: Path):
    from gnss_sim.visual_semantics import frozen_images

    p5_path = ROOT / "configs/p5-frozen.json"
    p5 = json.loads(p5_path.read_bytes())
    p7a = json.loads((ROOT / "configs/p7a-frozen.json").read_bytes())
    config = json.loads((ROOT / "configs/p7a-visual-semantics.json").read_bytes())
    if sha(pilot / "manifest.json") != p5["pilot_sha256"]["manifest"]:
        raise ValueError("pilot manifest changed")
    manifest = json.loads((pilot / "manifest.json").read_bytes())
    selected = select_cases(manifest)
    images = frozen_images(pilot, config, ROOT / "runs/p6/qwen3.8-flash")
    out.mkdir(parents=True, exist_ok=False)
    package = out / "inputs"
    entries = []
    for item in selected:
        cid = item["case_id"]
        target = package / "cases" / cid / "input.json"
        source = pilot / "cases" / cid / "input.json"
        if sha(source) != item["input_sha256"]:
            raise ValueError("input changed")
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
        (package / "images").mkdir(exist_ok=True)
        shutil.copyfile(images[cid], package / "images" / f"{cid}.png")
        entries.append({"case_id": cid, "input_sha256": sha(target), "image_sha256": sha(images[cid])})
    write_json(package / "manifest.json", {"protocol": "input-only-v1", "cases": entries})
    write_json(out / "selection.json", {"scope": "known-development-subset", "cases": selected,
        "pilot_manifest_sha256": sha(pilot / "manifest.json"),
        "selection": "lexicographic first IDs per metadata stratum; 6 normal/30 single/24 multi"})
    expected = {e["case_id"] for e in selected}
    provenance, hashes, cost = {}, {}, {}
    saved = json.loads((ROOT / "runs/comparison-views-v1/summary.json").read_bytes())
    for task, numerical in (("point", "sr"), ("range", "theilsen")):
        for condition in ("N", "V"):
            directory = (ROOT / "runs/p5" / numerical if condition == "N" else
                         ROOT / "runs/p7a/visual-semantics-v2" / task)
            path = directory / "predictions.jsonl"
            digest = (saved["methods"][f"{numerical}/{task}"]["provenance"]["predictions_sha256"]
                      if condition == "N" else p7a["predictions_sha256"][task])
            if sha(path) != digest:
                raise ValueError("archived predictions changed")
            rows = load_rows(path, task)
            if len(rows) != 300 or not expected <= rows.keys():
                raise ValueError("archived predictions incomplete")
            name = f"{condition}-{task}.jsonl"
            write_rows(package / name, [rows[cid] for cid in sorted(expected)])
            hashes[name] = sha(package / name)
            provenance[name] = {"source": str(path.relative_to(ROOT)), "sha256": digest}
            if condition == "V":
                raw = directory / "raw.jsonl"
                if sha(raw) != p7a["raw_sha256"][task]:
                    raise ValueError("archived raw response changed")
                records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
                records = [r for r in records if r["case_id"] in expected]
                cost[task] = {"historical_requests": len(records),
                    "usage_known": sum(isinstance(r.get("usage"), dict) for r in records),
                    "recorded_total_tokens": sum((r.get("usage") or {}).get("total_tokens", 0)
                                                  for r in records),
                    "summed_latency_ms": sum(r["latency_ms"] for r in records),
                    "completed_at": [r.get("completed_at") for r in records], "cost": None}
    write_json(package / "upstream.json", {"sha256": hashes, "provenance": provenance,
        "cost": {"visual": cost, "numeric_original_subset_seconds": None,
                 "original_render_seconds": None, "note": "Cached reuse; original full cost unknown."}})
    # A fresh input-only runner, using the frozen P5 thresholds, checks all 60 cases.
    replay = out / "numerical-replay"
    run_numerical(package, p5_path, replay)
    for task in TASKS:
        if load_rows(replay / task / "predictions.jsonl", task) != load_rows(package / f"N-{task}.jsonl", task):
            raise ValueError("input-only replay differs from archived N")
    report = {"cases": len(selected), "numerical_equivalence_cases_per_task": len(selected),
              "input_manifest_sha256": sha(package / "manifest.json"),
              "input_only": True, "truth_files_copied": 0}
    write_json(out / "prepared.json", report)
    return report


def evaluate(pilot: Path, prepared: Path, experiment: Path, out: Path):
    from gnss_sim.evaluation_views import evaluate_views
    from gnss_sim.schemas import CaseTruth

    selection = json.loads((prepared / "selection.json").read_bytes())
    if sha(pilot / "manifest.json") != selection["pilot_manifest_sha256"]:
        raise ValueError("evaluation pilot changed")
    run_record = json.loads((experiment / "run.json").read_bytes())
    identity = json.loads((experiment / "identity.json").read_bytes())
    if (sha(prepared / "inputs/manifest.json") != identity["input_manifest_sha256"] or
            sha(prepared / "inputs/upstream.json") != identity["upstream_sha256"]):
        raise ValueError("experiment input identity changed")
    input_ids = {case.case_id for _, case in read_inputs(prepared / "inputs")}
    if input_ids != {e["case_id"] for e in selection["cases"]}:
        raise ValueError("selection and inference case set differ")
    truths, metadata = {}, {}
    for entry in selection["cases"]:
        cid = entry["case_id"]
        path = pilot / "cases" / cid / "truth.json"
        if sha(path) != entry["truth_sha256"]:
            raise ValueError("truth digest mismatch")
        truths[cid] = CaseTruth.model_validate_json(path.read_bytes())
        metadata[cid] = entry["case_type"]
    out.mkdir(parents=True, exist_ok=False)
    reports, effects, per_case, grouped = {}, {}, {}, {}
    for task in TASKS:
        reports[task], loaded = {}, {}
        for condition in ("N", "V", "U", "C"):
            path = experiment / condition / task / "predictions.jsonl"
            if sha(path) != run_record["predictions_sha256"][f"{condition}/{task}"]:
                raise ValueError("experiment predictions changed")
            rows = load_rows(path, task)
            if rows.keys() != truths.keys() or any(r.method != f"{PROTOCOL}-{condition}" for r in rows.values()):
                raise ValueError("incomplete or inconsistent experiment rows")
            loaded[condition] = rows
            reports[task][condition] = evaluate_views(truths, rows, task)
            key = f"{condition}/{task}"
            per_case[key] = {cid: evaluate_views({cid: t}, {cid: rows[cid]}, task)
                             for cid, t in truths.items()}
            grouped[key] = {kind: evaluate_views({cid: t for cid, t in truths.items()
                if metadata[cid] == kind}, {cid: r for cid, r in rows.items() if metadata[cid] == kind}, task)
                for kind in sorted(set(metadata.values()))}
        success = {cid: t for cid, t in truths.items()
                   if all(loaded[c][cid].status == "success" for c in ("U", "C"))}
        paired = {c: evaluate_views(success, {cid: loaded[c][cid] for cid in success}, task)
                  for c in ("U", "C")}
        keys = ("point_exact", "point_tolerance_3d") if task == "point" else ("range_daily_all_axes_micro",)
        effects[task] = {"paired_success_cases": len(success),
            "review_failed_after_upstream_success": sum(loaded["U"][cid].status == "success" and
                    loaded["C"][cid].status == "failed" for cid in truths),
            "successful_review_deletions": {key: {
                "removed_fp": paired["U"][key]["fp"] - paired["C"][key]["fp"],
                "lost_tp": paired["U"][key]["tp"] - paired["C"][key]["tp"]} for key in keys}}
    report = {"protocol": PROTOCOL, "scope": "development-60", "metrics": reports,
              "review_effect": effects, "run": run_record}
    write_json(out / "summary.json", report)
    write_json(out / "per-case.json", per_case)
    write_json(out / "groups.json", grouped)
    lines = ["# P7b 开发子集结果", "", "60 例已知开发数据；不得视为独立确认。", "",
             "| 任务 | 条件 | 主 F1 | ±3 日 F1 / IoU | FAR | 成功率 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for task in TASKS:
        for condition, row in reports[task].items():
            primary = row["point_exact" if task == "point" else "range_affiliation"]["f1"]
            secondary = (row["point_tolerance_3d"]["f1"] if task == "point" else
                         row["range_daily_positive_axes"]["mean_iou"])
            values = [primary, secondary, row["negative_axes"]["far"], row["execution_success_rate"]]
            lines.append(f"| {task} | {condition} | " + " | ".join(
                "N/A" if v is None else f"{v:.4f}" for v in values) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"metrics": reports, "review_effect": effects}


def main():
    parser = argparse.ArgumentParser(description="P7b preparation, input-only inference and scoring")
    parser.add_argument("phase", choices=("prepare", "preflight", "run", "evaluate", "numerical"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/p7b-candidate-review.json")
    parser.add_argument("--pilot", type=Path, default=ROOT / "data/pilots/pilot-v1")
    parser.add_argument("--prepared", type=Path, default=ROOT / "runs/p7b/prepared")
    parser.add_argument("--preflight", type=Path, default=ROOT / "runs/p7b/preflight")
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/p7b/candidate-review-v1")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        report = prepare(args.pilot, args.out or args.prepared)
    elif args.phase == "preflight":
        report = preflight(args.config, args.out or args.preflight, args.env_file)
    elif args.phase == "run":
        report = run(args.config, args.prepared / "inputs", args.preflight,
                     args.out or args.experiment, args.env_file)
    elif args.phase == "numerical":
        if args.out is None:
            parser.error("numerical requires a new --out directory")
        report = run_numerical(args.prepared / "inputs", ROOT / "configs/p5-frozen.json", args.out)
    else:
        report = evaluate(args.pilot, args.prepared, args.experiment,
                          args.out or ROOT / "runs/p7b/evaluation")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
