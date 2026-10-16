"""Score coordinate consistency and matched ordinary repeats on a common task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_observation_probe import evaluate_population
from analyze_landslide_review_utility_v2 import verify_receipts
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_coordinate_review import ARMS, verify
from run_landslide_evidence_display_confirmation import numerical_predictions
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_missed_motion_repair import write_rows
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_coordinate_review import agree_or_abstain
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_window_experiment import parse_window_activity


def analyze(run, out):
    verify(run)
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Only completed frozen pilot predictions can be scored")
    for name, digest in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != digest:
            raise ValueError("Coordinate pilot predictions changed")
    config = read_json(run / "protocol.json")
    records = {cid: load_case_features(run, cid) for cid in config["case_ids"]}
    prior = read_json(run / "prior-controls.json")
    recomputed = numerical_predictions(Path(config["data"]), records)
    for method, values in recomputed.items():
        if prior[method] != values:
            raise ValueError("Common fitted numerical controls do not reproduce")
    receipts = verify_receipts(run)
    local = read_json(run / "local-predictions.json")
    for packet in read_json(run / "packets.json"):
        key = packet["request_id"]
        feature = records[packet["case_id"]][1]
        if local[key]["status"] == "success":
            replay = parse_window_activity(json.loads(receipts[key]["output"]), feature.observed,
                                           packet["start"], packet["stop"])
        else:
            replay = np.full(len(feature.observed), -1)
        np.testing.assert_array_equal(replay, local[key]["activity"])
    head = XGBClassifier()
    head.load_model(run / "stage-xgboost.json")
    labels = load_labels(run / "reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    selections = read_json(run / "selections.json")
    predictions = dict(prior)
    used, permissions, disagreements = {}, {}, []
    for rep in (0, 1):
        for cid, (case, feature) in records.items():
            initial = np.array(prior[f"direct_visual_r{rep}"][cid]["activity"])
            choice = selections[f"{cid}-r{rep}"]
            scope = np.zeros(len(initial), dtype=bool)
            scope[choice["permission"]] = True
            readings = {arm: np.full(len(initial), -1) for arm in ARMS}
            keys = {arm: [] for arm in ARMS}
            for target in choice["selected"]:
                for arm in ARMS:
                    key = f"{cid}-window-{target['index']:02d}-{arm}-r{rep}"
                    activity = np.array(local[key]["activity"])
                    readings[arm][target["start"]:target["stop"]] = activity[target["start"]:target["stop"]]
                    keys[arm].append(key)
            combined = {}
            for arm in ("raw", "reflected"):
                combined[arm] = initial.copy()
                combined[arm][scope] = readings[arm][scope]
            for stem, second in (("coordinate_consensus", "reflected"), ("repeat_consensus", "raw_repeat")):
                combined[stem] = agree_or_abstain(initial, readings["raw"], readings[second], scope, feature.observed)
            codes = head.predict(feature.matrix).astype(int) + 1
            for stem, activity in combined.items():
                method = f"{stem}_r{rep}"
                prediction = learned_stages(codes, activity, feature.supported)
                predictions.setdefault(method, {})[cid] = prediction.to_dict()
                if stem == "coordinate_consensus":
                    request_keys = keys["raw"] + keys["reflected"]
                elif stem == "repeat_consensus":
                    request_keys = keys["raw"] + keys["raw_repeat"]
                else:
                    request_keys = keys[stem]
                used.setdefault(method, {})[cid] = request_keys
                permissions.setdefault(method, {})[cid] = choice["permission"]
                if np.any(activity[~scope] != initial[~scope]):
                    raise ValueError("Coordinate review expanded original revision authority")
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            for second in ("reflected", "raw_repeat"):
                disagreement = scope & (readings["raw"] != readings[second]) & (actual >= 0)
                raw_correct = readings["raw"] == actual
                disagreements.append({"repeat": rep, "case_id": cid, "second": second,
                                      "disagreement_days": int(disagreement.sum()),
                                      "raw_correct_abstained": int((disagreement & raw_correct).sum()),
                                      "raw_wrong_definite_abstained": int((disagreement & ~raw_correct & (readings["raw"] >= 0)).sum()),
                                      "raw_unknown_days": int((disagreement & (readings["raw"] == -1)).sum())})
    out.mkdir(parents=True, exist_ok=False)
    results, summaries, per_case = evaluate_population(predictions, labels)
    internal = verify_metrics(predictions, results, labels, records)
    changes, daily, cost_rows = {}, [], []
    for row in summaries:
        method = row["method"]
        row["internal_stage_macro_f1"] = internal[method]["macro_f1"]
        wrong_definite, correct_definite = 0, 0
        for cid, value in predictions[method].items():
            actual = np.array([int(item["activity_label"]) for item in labels[cid]])
            activity = np.array(value["activity"])
            known = actual >= 0
            wrong_definite += int((known & (activity >= 0) & (activity != actual)).sum())
            correct_definite += int((known & (activity == actual)).sum())
            for day, item in enumerate(labels[cid]):
                daily.append({"method": method, "case_id": cid, "day_index": day,
                              "actual_activity": item["activity_label"], "actual_stage": item["feature"],
                              "predicted_activity": value["activity"][day], "predicted_stage": value["stage"][day]})
            if method.startswith("coordinate_consensus"):
                for comparator in (f"raw_r{method[-1]}", f"repeat_consensus_r{method[-1]}"):
                    before = np.array(predictions[comparator][cid]["activity"])
                    changes.setdefault(method, {}).setdefault(comparator, {})[cid] = {
                        **correction_utility(actual, before, activity, known),
                        "gained_tp": int((known & (actual == 1) & (before != 1) & (activity == 1)).sum()),
                        "lost_tp": int((known & (actual == 1) & (before == 1) & (activity != 1)).sum()),
                        "added_fp": int((known & (actual == 0) & (before != 1) & (activity == 1)).sum()),
                        "removed_fp": int((known & (actual == 0) & (before == 1) & (activity != 1)).sum()),
                    }
        row["wrong_definite_days"] = wrong_definite
        row["correct_definite_days"] = correct_definite
        if method in used:
            keys = [key for values in used[method].values() for key in values]
            global_keys = [f"{cid}-direct-r{method[-1]}" for cid in config["case_ids"]]
            globals_saved = [read_json(Path(config["prior"]) / "requests" / f"{key}.response.json")
                             for key in global_keys]
            tokens = sum(receipts[key].get("usage", {}).get("total_tokens", 0) for key in keys)
            tokens += sum(receipt.get("usage", {}).get("total_tokens", 0) for receipt in globals_saved)
            row["deployment_requests"] = len(keys) + len(global_keys)
            row["reported_tokens"] = tokens
            cost_rows.append({"method": method, "requests": row["deployment_requests"], "tokens": tokens})
        else:
            row["deployment_requests"] = None
            row["reported_tokens"] = None
    indexed = {row["method"]: row for row in summaries}
    decisions = []
    for rep in (0, 1):
        main, raw, repeat = [indexed[f"{stem}_r{rep}"] for stem in
                             ("coordinate_consensus", "raw", "repeat_consensus")]
        utility = changes[main["method"]][raw["method"]]
        gate = {"f1_nonloss_vs_raw": main["f1"] >= raw["f1"],
                "weak_recall_nonloss_vs_raw": main["weak_recall"] >= raw["weak_recall"],
                "wrong_definite_reduced_vs_raw": main["wrong_definite_days"] < raw["wrong_definite_days"],
                "net_definite_correction_positive_vs_raw": sum(item["net_correct"] for item in utility.values()) > 0,
                "f1_nonloss_vs_repeat": main["f1"] >= repeat["f1"]}
        decisions.append({"repeat": rep, "gate": gate, "passed": all(gate.values())})
    gain = any(indexed[f"coordinate_consensus_r{rep}"]["f1"] > indexed[f"repeat_consensus_r{rep}"]["f1"]
               for rep in (0, 1))
    decision = {"repeats": decisions, "strict_gain_vs_repeat_in_one_repeat": gain,
                "continue": all(item["passed"] for item in decisions) and gain,
                "field_accuracy_confirmed": False, "independent_validation": False}
    actual_cost = {"started": len(list((run / "requests").glob("*.started.json"))),
                   "receipts": len(receipts), "returned_json": sum(item["status"] == "returned_json" for item in receipts.values()),
                   "reported_tokens": sum(item.get("usage", {}).get("total_tokens", 0) for item in receipts.values()),
                   "failures": len(read_json(run / "failures.json")), "new_records": 0}
    for name, value in (("predictions.json", predictions), ("results.json", results),
                        ("internal-stages.json", internal), ("changes.json", changes),
                        ("decision.json", decision), ("actual-costs.json", actual_cost),
                        ("used-requests.json", used), ("permissions.json", permissions)):
        save_json(out / name, value)
    for name, rows in (("summary.csv", summaries), ("per-case.csv", per_case),
                       ("daily.csv", daily), ("disagreements.csv", disagreements), ("deployment-costs.csv", cost_rows)):
        write_rows(out / name, rows)
    fig, panels = plt.subplots(1, 2, figsize=(10, 4))
    stems = ("raw", "reflected", "repeat_consensus", "coordinate_consensus")
    for rep, offset in ((0, -.18), (1, .18)):
        panels[0].bar(np.arange(4) + offset, [indexed[f"{stem}_r{rep}"]["f1"] for stem in stems],
                      width=.36, label=f"Repeat {rep}")
        panels[1].bar(np.arange(4) + offset, [indexed[f"{stem}_r{rep}"]["coverage"] for stem in stems], width=.36)
    for panel, title in zip(panels, ("Activity F1; unknown counted as missed activity", "Definite decision coverage"), strict=True):
        panel.set_xticks(np.arange(4), stems, rotation=20, ha="right", fontsize=8)
        panel.set_ylim(0, 1)
        panel.set_title(title, fontsize=10)
    panels[0].legend()
    fig.suptitle("Exposed controlled development records; coordinate consistency pilot")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    save_json(out / "checks.json", {"receipt_replays": len(local), "numerical_case_replays": 9 * len(records),
                                   "independent_metric_methods": len(predictions), "daily_rows": len(daily),
                                   "missing_stage_scope_checked": True})
    save_json(out / "manifest.json", {"files": {str(path.relative_to(out)): sha(path)
                                               for path in out.rglob("*") if path.is_file()}})
    print({"decision": decision, "cost": actual_cost, "methods": len(predictions)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    analyze(arguments.run, arguments.out)
