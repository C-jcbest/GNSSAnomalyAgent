"""Finish a failed export from existing weights and outputs without retraining."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from encode_landslide_paired_persistence import verify_inputs
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import ROOT, export
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import save_json
from run_landslide_window_optimization import read_json
from score_landslide_claim_review import score_target
from train_landslide_paired_persistence import ARMS

from gnss_sim.artifacts import sha


def finalize(run):
    verify_inputs(run)
    verify_bindings(run, "encoding-freeze.json")
    for name, expected in read_json(run / "learning-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Failed execution source changed: " + name)
    if read_json(run / "state.json")["status"] != "export_failed":
        raise ValueError("Finalize only the recorded export failure")
    export(run / "finalization-plan.json", {"change": "Use validation-summary.json; prepared summary.json stays intact",
        "source_sha256": sha(Path(__file__)), "retraining": False, "changed_thresholds": False,
        "changed_scoring_or_gate": False, "missing_costs_retained_unknown": True})
    (run / "source/finalization.py").write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    names = list(ARMS) + ["xgb_raw18", "xgb_expert24", "xgb_chart384",
                          "original_robust_rule", "original_pelt_rule", "original_xgboost"]
    stored = read_json(run / "predictions.json")
    predictions = {name: stored[name] for name in names}
    labels = read_json(run / "validation-labels.json")
    observed, weak = {}, {}
    for cid in labels:
        with np.load(run / "original/validation" / (cid + ".npz")) as data:
            observed[cid], weak[cid] = data["observed"], data["weak"]
    results, summary, records = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json"):
        raise ValueError("Recovered common scores differ from original execution")
    corrections = []
    for name in names:
        if name == "visual_numeric_pair":
            continue
        for cid in labels:
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            known = actual >= 0
            before = np.asarray(predictions[name][cid]["activity"])
            after = np.asarray(predictions["visual_numeric_pair"][cid]["activity"])
            gained = int((known & (before != actual) & (after == actual)).sum())
            lost = int((known & (before == actual) & (after != actual)).sum())
            corrections.append({"control": name, "case_id": cid, "gained": gained, "lost": lost, "net": gained - lost})
    lookup = {row["method"]: row for row in summary}
    main = lookup["visual_numeric_pair"]
    plain, augment, shuffled, numeric = [lookup[name] for name in (
        "visual_numeric_plain", "visual_numeric_augmentation", "visual_numeric_shuffled_pairs", "numeric_pair")]
    # Exact eight conditions from the frozen training execution; export recovery
    # does not add a condition or reinterpret a tradeoff after seeing scores.
    checks = {
        "f1_nondecrease_augmentation": main["f1"] >= augment["f1"],
        "f1_nondecrease_shuffled": main["f1"] >= shuffled["f1"],
        "f1_gain_plain_at_least_005": main["f1"] - plain["f1"] >= .005,
        "strict_pairing_gain": main["f1"] > augment["f1"] or main["event_f1_iou03"] > augment["event_f1_iou03"],
        "weak_nondecrease": main["weak_recall"] >= max(plain["weak_recall"], augment["weak_recall"]),
        "visual_value_vs_numeric": main["f1"] >= numeric["f1"] or (
            main["event_f1_iou03"] > numeric["event_f1_iou03"] and main["f1"] >= numeric["f1"] - .01),
    }
    for control in ("visual_numeric_plain", "visual_numeric_augmentation"):
        checks["net_correction_" + control] = sum(row["net"] for row in corrections if row["control"] == control) > 0
    decision = "PROCEED_PAIRED_PERSISTENCE_DEVELOPMENT" if all(checks.values()) else "STOP_PAIRED_PERSISTENCE_CALIBRATION"
    export(run / "validation-summary.json", summary)
    export(run / "corrections.json", corrections)
    write_csv(run / "summary.csv", summary)
    write_csv(run / "per-case.csv", records)
    rounded_seconds = [("visual_numeric_pair", .617, .659), ("visual_numeric_augmentation", .369, .407),
        ("visual_numeric_plain", .370, .391), ("visual_numeric_shuffled_pairs", .568, .660),
        ("visual_pair_no_numeric", .561, .627), ("numeric_pair", .454, .461), ("numeric_augmentation", .309, .310)]
    costs = [{"arm": name, "width": width, "train_seconds_rounded": seconds,
              "source": "Actual terminal stdout, three-decimal print; exact timer row lost on export failure"}
             for name, left, right in rounded_seconds for width, seconds in ((16, left), (32, right))]
    costs.extend({"arm": name, "depth": depth, "train_seconds_rounded": None,
                  "source": "Not persisted before export failure"}
                 for name in ("xgb_raw18", "xgb_expert24", "xgb_chart384") for depth in (2, 4))
    export(run / "training-cost.json", {"rows": costs, "wall_seconds": None,
        "validation_head_seconds": None, "new_api_calls": 0, "shared_encoder_cost": "encoding-cost.json",
        "failure": "Exact timer rows unavailable; no retraining to fabricate original costs",
        "finalization_seconds": time.perf_counter() - started})
    export(run / "calibration-decision.json", {"decision": decision, "checks": checks,
        "independent_confirmation": False, "passed": sum(checks.values()), "total": len(checks)})
    verify_inputs(run)
    save_json(run / "state.json", {"status": "calibration_complete", "decision": decision,
                                   "export_failure_recovered": True, "retraining": False})
    export(run / "calibration-freeze.json", {path.relative_to(run).as_posix(): sha(path)
        for path in run.rglob("*") if path.is_file() and path.name not in {"calibration-freeze.json", "state.json"}})
    print(decision, sum(checks.values()), "/", len(checks), flush=True)
    print({name: {key: lookup[name][key] for key in ("f1", "weak_recall", "event_f1_iou03")}
           for name in names[:7]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    finalize(parser.parse_args().run.resolve())
