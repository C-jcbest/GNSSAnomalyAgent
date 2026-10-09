"""Run three fixed stage heads on the complete frozen author activity packet."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_landslide_raw_adjudication import verify as verify_activity_run  # noqa: E402

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_baselines import observation_features, rule_stages  # noqa: E402
from gnss_sim.landslide_diagnostics import LandslideDiagnostics  # noqa: E402
from gnss_sim.landslide_evaluation import ActivityPrediction  # noqa: E402
from gnss_sim.landslide_frozen_stage_heads import (  # noqa: E402
    METHODS,
    apply_stage_support,
    compile_stage_answer,
    numeric_stage_rows,
    stage_pair_counts,
    stage_prompt,
    unclassified_rows,
    validate_activity,
)
from gnss_sim.landslide_visual import VisualRequests  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_frozen(source, target, expected):
    if sha(source) != expected:
        raise ValueError(f"Input changed: {source.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha(target) != expected:
        raise ValueError(f"Copied input changed: {target.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/landslide-frozen-stage-heads-v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/landslide-frozen-stage-heads-2026-10-08/run-v1")
    args = parser.parse_args()
    config = read_json(args.config)
    fixed = {"model": "qwen3.8-flash", "maximum_requests": 9, "repetitions": 1,
             "temperature": 0.1, "top_p": 0.3, "max_tokens": 2048, "retries": 0,
             "rule_relative_change_60d": 0.25, "scoring": False, "sealed_test_access": "none"}
    if any(config[key] != expected for key, expected in fixed.items()):
        raise ValueError("Stage configuration differs from the fixed protocol")
    author = ROOT / config["activity_run"]
    packet = ROOT / config["observation_packet"]
    numeric = ROOT / config["numeric_run"]
    verify_activity_run(author)
    author_manifest = read_json(author / "public/manifest.json")
    activity_freeze = read_json(author / "activity-freeze.json")
    packet_manifest = read_json(packet / "manifest.json")
    ledger = read_json(author / "run-ledger-start.json")
    prior_stage = read_json(ROOT / "artifacts/landslide-stage-diagnosis-2026-10-07/reference-v1/manifest.json")
    if sha(numeric / "stage-xgboost.json") != prior_stage["stage_model_sha256"]:
        raise ValueError("The saved stage model differs from the earlier frozen experiment")
    if sha(numeric / "frozen-selection.json") != prior_stage["selection_sha256"]:
        raise ValueError("The numeric parameter selection changed")
    selection = read_json(numeric / "frozen-selection.json")
    if selection["rule_stage"]["relative_change_60d"] != config["rule_relative_change_60d"]:
        raise ValueError("The rule threshold differs from the original validation selection")
    contract = read_json(numeric / "feature-contract.json")
    model = XGBClassifier()
    model.load_model(numeric / "stage-xgboost.json")
    if model.get_booster().num_features() != len(contract["names"]):
        raise ValueError("Saved model feature count differs from the feature contract")
    records = packet_manifest["cases"]
    expected_ids = [f"case_{number:04d}" for number in range(1, 13)]
    if [record["case_id"] for record in records] != expected_ids:
        raise ValueError("The complete twelve-case development packet is required")
    entries = {entry["case_id"]: entry for entry in author_manifest["cases"]}
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    public = out / "public"
    public.mkdir()
    source_names = ["src/gnss_sim/landslide_frozen_stage_heads.py", "src/gnss_sim/landslide_baselines.py",
                    "src/gnss_sim/landslide_diagnostics.py", "src/gnss_sim/landslide_evaluation.py",
                    "src/gnss_sim/landslide_reference_interface.py", "src/gnss_sim/landslide_visual.py",
                    "src/gnss_sim/visual.py", "scripts/run_landslide_frozen_stage_heads.py",
                    "scripts/analyze_landslide_frozen_stage_heads.py", "scripts/landslide_stage_heads_page.html",
                    "tests/test_landslide_frozen_stage_heads.py", "configs/landslide-frozen-stage-heads-v1.json",
                    "docs/landslide-frozen-stage-heads-protocol.md"]
    source_hashes = {name: sha(ROOT / name) for name in source_names}
    for name, expected in source_hashes.items():
        copy_frozen(ROOT / name, out / "source" / name, expected)
    copy_frozen(args.config, public / "protocol.json", sha(args.config))
    copy_frozen(packet / "manifest.json", public / "source-manifest.json", sha(packet / "manifest.json"))
    copy_frozen(author / "activity-freeze.json", public / "activity-freeze.json", sha(author / "activity-freeze.json"))
    copy_frozen(numeric / "stage-xgboost.json", out / "models/stage-xgboost.json", sha(numeric / "stage-xgboost.json"))
    copy_frozen(numeric / "feature-contract.json", public / "feature-contract.json", sha(numeric / "feature-contract.json"))
    cases, motions, reviews, activity_rows, packets = {}, {}, {}, {}, []
    # Assemble and freeze every model input before executing any stage head.
    for record in records:
        case_id = record["case_id"]
        names = [record["input"], record["csv"], *[view["image"] for view in record["views"]]]
        for window in (31, 61, 91):
            names.extend(record["diagnostics"][str(window)][key] for key in ("data", "image"))
        for name in names:
            copy_frozen(packet / name, public / name, ledger["source_packet_sha256"][name])
        entry = entries[case_id]
        for key in ("review", "daily"):
            copy_frozen(author / "public" / entry[key], public / "activity-source" / entry[key],
                        sha(author / "public" / entry[key]))
        cases[case_id] = LandslideInput.model_validate_json((public / record["input"]).read_bytes())
        reviews[case_id] = read_json(public / "activity-source" / entry["review"])
        activity_rows[case_id] = read_json(public / "activity-source" / entry["daily"])
        expected_hash = activity_freeze["reviews"][case_id]["activity_sha256"]
        validate_activity(cases[case_id], reviews[case_id], activity_rows[case_id], expected_hash)
        motions[case_id] = {window: LandslideDiagnostics.model_validate_json(
            (public / record["diagnostics"][str(window)]["data"]).read_bytes()) for window in (31, 61, 91)}
        if any(motion.case_id != case_id or motion.input_sha256 != record["input_sha256"]
               or motion.window_days != window for window, motion in motions[case_id].items()):
            raise ValueError("Auxiliary cache does not identify these observations/window")
        if any(row["activity_label"] == 1 for row in activity_rows[case_id]):
            images = [view["image"] for view in record["views"]]
            images += [record["diagnostics"][str(window)]["image"] for window in (31, 61, 91)]
            packets.append({"case_id": case_id, "request_id": case_id + "-frozen-stage",
                            "activity_sha256": expected_hash, "images": images,
                            "prompt": stage_prompt(case_id, reviews[case_id], expected_hash, images,
                                                   len(cases[case_id].dates))})
    if len(packets) != config["maximum_requests"]:
        raise ValueError("Unexpected count of records with confirmed activity")
    write_json(public / "packets.json", packets)
    prepared_hashes = {path.relative_to(public).as_posix(): sha(path) for path in public.rglob("*") if path.is_file()}
    write_json(out / "run-ledger-start.json", {"status": "running", "command": sys.argv,
                                               "started_utc": datetime.now(timezone.utc).isoformat(),
                                               "request_plan_sha256": sha(public / "packets.json"),
                                               "prepared_inputs_sha256": prepared_hashes,
                                               "activity_run": config["activity_run"],
                                               "activity_freeze_sha256": sha(author / "activity-freeze.json"),
                                               "stage_model_sha256": sha(numeric / "stage-xgboost.json"),
                                               "new_stage_reference": False, "maximum_requests": 9})
    transport = VisualRequests(public / "requests", ROOT / ".env", config["model"], 9)
    packets_by_case = {request["case_id"]: request for request in packets}
    results = []
    all_rows = []
    started = time.monotonic()
    for record in records:
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise TimeoutError("Fixed stage run reached its hard timeout")
        case_id = record["case_id"]
        case, diagnostics = cases[case_id], motions[case_id]
        rows, review = activity_rows[case_id], reviews[case_id]
        activity = np.array([row["activity_label"] for row in rows])
        features = observation_features(case, diagnostics)
        if features.names != contract["names"]:
            raise ValueError("Feature names/order differ from the saved training contract")
        rule = rule_stages(features, activity, config["rule_relative_change_60d"])
        stage = model.predict(features.matrix).astype(int) + 1
        stage[(activity != 1) | ~features.supported] = -1
        stage[activity == 0] = 0
        xgboost = ActivityPrediction(activity.copy(), stage)
        native = {"rule": numeric_stage_rows(rows, rule), "xgboost": numeric_stage_rows(rows, xgboost)}
        final, suppressed = {}, {}
        status = {method: "success" for method in METHODS}
        for method in ("rule", "xgboost"):
            final[method], suppressed[method] = apply_stage_support(native[method], rows, case, diagnostics[61])
        error = None
        if case_id in packets_by_case:
            request = packets_by_case[case_id]
            try:
                payload = transport.ask(request["request_id"], request["prompt"], [public / name for name in request["images"]])
                stage_review, native["visual"], final["visual"], suppressed["visual"] = compile_stage_answer(
                    payload, case, review, rows, request["activity_sha256"], diagnostics[61])
                write_json(public / f"stage-reviews/{case_id}.json", stage_review)
            except (RuntimeError, ValueError, TypeError, KeyError) as failure:
                status["visual"] = "failed_unknown_activity_stages"
                error = {"type": type(failure).__name__, "message": str(failure)[:400]}
                native["visual"] = unclassified_rows(rows)
                final["visual"] = unclassified_rows(rows)
                suppressed["visual"] = {}
        else:
            status["visual"] = "skipped_no_confirmed_activity"
            native["visual"] = unclassified_rows(rows)
            final["visual"] = unclassified_rows(rows)
            suppressed["visual"] = {}
        head_counts = {}
        for method in METHODS:
            write_json(public / f"native/{case_id}-{method}.json", native[method])
            write_json(public / f"daily/{case_id}-{method}.json", final[method])
            head_counts[method] = {
                "native_activity_features": dict(Counter(row["feature"] for row in native[method] if row["activity_label"] == 1)),
                "final_activity_features": dict(Counter(row["feature"] for row in final[method] if row["activity_label"] == 1)),
                "suppressed_support": suppressed[method], "status": status[method]}
            all_rows.extend({"method": method, **row} for row in final[method])
        velocity_support = np.array([value is not None for value in diagnostics[61].velocity_mm_day])
        tangent_support = np.array([value is not None for value in diagnostics[61].tangential_acceleration_mm_day2])
        pairs = {f"{first}_vs_{second}": stage_pair_counts(final[first], final[second])
                 for first, second in combinations(METHODS, 2)}
        result = {"case_id": case_id, "activity_sha256": activity_freeze["reviews"][case_id]["activity_sha256"],
                  "active_days": int(np.count_nonzero(activity == 1)),
                  "velocity_supported_activity_days": int(np.count_nonzero((activity == 1) & velocity_support)),
                  "tangent_supported_activity_days": int(np.count_nonzero((activity == 1) & tangent_support)),
                  "heads": head_counts, "pairs": pairs, "visual_error": error}
        results.append(result)
        write_json(out / f"case-status/{case_id}.json", result)
        print(f"{case_id}: visual={status['visual']}; active days={result['active_days']}", flush=True)
    receipts = [read_json(path) for path in (public / "requests").glob("*.response.json")]
    summary = {"protocol_id": config["protocol_id"], "cases": 12, "calls": len(receipts),
               "total_tokens": sum(receipt.get("usage", {}).get("total_tokens", 0) for receipt in receipts),
               "service_seconds": sum(receipt["seconds"] for receipt in receipts),
               "elapsed_seconds": time.monotonic() - started, "results": results,
               "performance_scores": None, "independent_stage_reference": False,
               "sealed_test_read": False, "retries": 0}
    write_json(public / "results.json", summary)
    with (public / "daily-stage-heads.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    verify_activity_run(author)
    if any(sha(public / name) != expected for name, expected in prepared_hashes.items()):
        raise ValueError("A prepared input changed during stage execution")
    write_json(out / "inference-freeze.json", {
        "evidence_sha256": {path.relative_to(out).as_posix(): sha(path) for path in public.rglob("*") if path.is_file()},
        "activity_unchanged": True, "new_stage_reference": False})
    write_json(out / "manifest.json", {"status": "complete", "source_sha256": source_hashes,
                                       "config": config, "inference_freeze_sha256": sha(out / "inference-freeze.json"),
                                       "model_sha256": sha(out / "models/stage-xgboost.json"),
                                       "calls": len(receipts), "performance_scores": None})
    print(f"COMPLETE: {len(receipts)} calls; frozen activity unchanged; stage predictions only", flush=True)


if __name__ == "__main__":
    main()
