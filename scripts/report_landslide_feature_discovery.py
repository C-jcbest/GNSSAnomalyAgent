"""Replay reviewed-pool weights and independently score all selected methods."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import ROOT, csv_rows, export
from probe_landslide_feature_discovery import (
    PROPOSALS,
    arm_matrix,
    case_inputs,
    compile_reviewed,
    correction,
    probe_functions,
)
from probe_landslide_visual_invariance import TRAINING, calibration_labels
from report_landslide_core_lockbox import full_table
from report_landslide_visual_invariance_calibration import diagnostic_figure
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import calibration_objective
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_core_lockbox import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores
from gnss_sim.landslide_local_experiment import learned_stages


def verify_generation():
    """Reconstruct six complete payloads without another API call."""
    verify_bindings(PROPOSALS, "pre-proposal-freeze.json")
    verify_bindings(PROPOSALS, "proposal-freeze.json")
    packets = read_json(PROPOSALS / "packets.json")
    proposals = read_json(PROPOSALS / "proposals.json")
    rows = []
    for request_id, item in proposals.items():
        packet = packets[item["arm"]]
        receipt = read_json(PROPOSALS / "requests" / (request_id + ".response.json"))
        start = read_json(PROPOSALS / "requests" / (request_id + ".started.json"))
        images = [PROPOSALS / name for name in packet["images"]]
        content = [{"type": "text", "text": packet["prompt"]}]
        for path in images:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + encoded}})
        payload = {"model": "qwen3.8-flash", "messages": [{"role": "user", "content": content}],
                   "temperature": .1, "top_p": .3, "max_tokens": 2048,
                   "enable_thinking": False, "response_format": {"type": "json_object"}}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        for value in (receipt, start):
            if (value["request_sha256"] != expected or value["prompt"] != packet["prompt"]
                    or value["image_sha256"] != {path.name: sha(path) for path in images}):
                raise ValueError("Frozen generation payload does not reconstruct")
        if (receipt["status"] != "returned_json" or receipt["finish_reason"] != "stop"
                or json.loads(receipt["output"]) != item["proposal"]):
            raise ValueError("Proposal differs from original complete response")
        rows.append({"request_id": request_id, "arm": item["arm"], "repetition": item["repetition"],
                     "tokens": receipt["usage"]["total_tokens"], "seconds": receipt["seconds"],
                     "payload_verified": True})
    return rows


def report(run, out):
    verify_bindings(run, "freeze.json")
    verify_bindings(run, "result-freeze.json")
    for name, expected in read_json(run / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen fitting source/reference changed")
    if read_json(run / "state.json")["status"] != "complete":
        raise ValueError("Need terminal calibration")
    generation = verify_generation()
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    pools = read_json(run / "reviewed-pools.json")
    # NumPy initializes ndarray.any lazily. Initialize its own backend outside
    # the restricted generated namespace; do not grant generated code imports.
    np.isnan(np.zeros((13, 3))).any()
    np.linalg.norm(np.zeros(3))
    functions = compile_reviewed(pools)
    def probe_key(row):
        return row["pool"], row["feature"], row["shape"]

    replayed_probes = sorted(probe_functions(pools, functions), key=probe_key)
    original_probes = sorted(read_json(run / "pre-score-probes.json"), key=probe_key)
    if replayed_probes != original_probes:
        raise ValueError("Pre-score feature probe changed")
    protocol = read_json(run / "protocol.json")
    selections = read_json(run / "selection.json")
    trials = read_json(run / "validation-trials.json")
    corrections = csv_rows(run / "per-record-corrections.csv")
    all_corrections, costs, family_rows = [], [], []
    checked_days = 0
    summary_text = ["# 固定视觉特征池：同划分强基线、消融与权重重放", "",
                    read_json(run / "decision.json")["decision"], "",
                    "这些是历史开发验证分数，不是独立观测准确率。生成只用六个训练记录窗，"
                    "两遍三输入臂共六请求；14/18函数按原契约保留，无再生成或源代码修复。", ""]
    selected_numbers = []
    for target, source in TRAINING.items():
        folder = out / target
        folder.mkdir()
        ids = protocol["calibration_splits"][target]["validation"]
        pairs = {cid: load_case_features(source, cid) for cid in ids}
        labels = calibration_labels(source, pairs)
        inputs = {cid: case_inputs(*pairs[cid], functions) for cid in ids}
        stage = XGBClassifier()
        stage.load_model(source / "stage-xgboost.json")
        codes = {cid: stage.predict(pairs[cid][1].matrix).astype(int) + 1 for cid in ids}
        frozen = read_json(run / target / "validation-predictions.json")
        bank = read_json(run / target / "validation-probabilities.json")
        predictions, probabilities = {}, {}
        for name, arm in protocol["arms"].items():
            selected = selections[target][name]
            candidates = [row for row in trials if row["target"] == target and row["arm"] == name]
            if len(candidates) != 38 or max(candidates, key=lambda row: row["objective"]) != selected:
                raise ValueError("Selection does not match frozen opportunities")
            model = XGBClassifier()
            model.load_model(run / target / f"{name}-depth{selected['depth']}.json")
            predictions[name], probabilities[name] = {}, {}
            for cid in ids:
                started = time.perf_counter()
                probability = model.predict_proba(arm_matrix(inputs[cid][0], arm))[:, 1]
                costs.append({"target": target, "arm": name, "case_id": cid,
                              "inference_seconds": time.perf_counter() - started})
                if probability.tolist() != bank[name][cid]:
                    raise ValueError("Replayed probability changed")
                activity = activity_from_mask(probability >= selected["threshold"], inputs[cid][1])
                prediction = learned_stages(codes[cid], activity, pairs[cid][1].supported)
                if prediction.to_dict() != frozen[name][cid]:
                    raise ValueError("Replayed activity/stage changed")
                if np.any(activity[~inputs[cid][1]] != -1):
                    raise ValueError("Activity escaped original support")
                actual = np.array([int(row["activity_label"]) for row in labels[cid]])
                gained, lost = correction(actual, np.array(frozen["base18"][cid]["activity"]), activity)
                saved = next(row for row in corrections if row["target"] == target
                             and row["arm"] == name and row["case_id"] == cid)
                if (gained, lost, gained - lost) != tuple(int(saved[key]) for key in ("gained", "lost", "net")):
                    raise ValueError("Correction count differs")
                all_corrections.append({"target": target, "arm": name, "case_id": cid,
                                        "gained": gained, "lost": lost, "net": gained - lost})
                predictions[name][cid] = prediction.to_dict()
                probabilities[name][cid] = probability.tolist()
                checked_days += len(activity)
        observed = {cid: pair[1].observed for cid, pair in pairs.items()}
        weak = {cid: np.array([int(row.get("weak_signal", 0)) == 1 and int(row["activity_label"]) == 1
                               for row in labels[cid]]) for cid in ids}
        results, summary, records = score_target(labels, observed, weak, predictions, {}, {})
        for name, result in results.items():
            if result["aggregate"] != selections[target][name]["scores"]:
                raise ValueError("Replayed aggregate differs")
            if calibration_objective(result["aggregate"]) != selections[target][name]["objective"]:
                raise ValueError("Replayed objective differs")
        for row in summary:
            row["target"] = target
            changes = [r for r in all_corrections if r["target"] == target and r["arm"] == row["method"]]
            row["gained_vs_base"] = sum(r["gained"] for r in changes)
            row["lost_vs_base"] = sum(r["lost"] for r in changes)
            row["net_vs_base"] = sum(r["net"] for r in changes)
            row["benefiting_records"] = sum(r["net"] > 0 for r in changes)
        export(folder / "results.json", results)
        write_csv(folder / "summary.csv", summary)
        write_csv(folder / "per-case.csv", records)
        selected_numbers.extend(summary)
        candidates = [row for row in all_corrections if row["target"] == target
                      and row["arm"] == "visual_numeric-r1_union"]
        chosen = []
        for reverse in (False, True):
            for row in sorted(candidates, key=lambda row: (row["net"], row["case_id"]), reverse=reverse):
                if row["case_id"] not in chosen:
                    chosen.append(row["case_id"])
                    break
        export(folder / "diagnostic-selection.json", {"case_ids": chosen,
                "selection": "post-score min/max net vsbase, no reference revision or gate/subset change"})
        diagnostic_names = ["base18", "base_expert", "visual_numeric-r0_union",
                            "visual_numeric-r1_union", "numeric_only-r1_union"]
        for cid in chosen:
            diagnostic_figure(pairs[cid][0], labels[cid],
                              {name: predictions[name][cid] for name in diagnostic_names},
                              {name: probabilities[name][cid] for name in diagnostic_names},
                              folder / (cid + "-diagnostic.png"), target)
            values = np.array([row if row is not None else [np.nan] * 3
                               for row in pairs[cid][0].displacement_mm])
            blocks = []
            for start in range(0, len(values), 14):
                window = values[start:start + 14]
                complete = window[np.isfinite(window).all(axis=1)]
                blocks.append({"start": start, "stop": min(start + 14, len(values)),
                               "complete_days": len(complete),
                               "median_neu_mm": np.median(complete, axis=0).tolist() if len(complete) >= 8 else None})
            export(folder / (cid + "-raw-blocks.json"), blocks)
        if target == "primary_process":
            families = read_json(source / "protocol.json")["families"]
            for name, result in results.items():
                for family in families:
                    members = [cid for cid in ids if families[(int(cid.split("_")[-1]) - 1) % 6] == family]
                    score = aggregate_scores([result["per_case"][cid] for cid in members])
                    family_rows.append({"method": name, "family": family,
                                        "f1": score["daily"]["f1"], "fp": score["daily"]["fp"],
                                        "fn": score["daily"]["fn"]})
        summary_text.extend(["## " + target, "", full_table(summary), ""])
        print("Replayed and independently scored", target, flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    names = ["base18", "base_blocks", "base_expert", "visual_numeric-r0_union",
             "visual_numeric-r1_union", "visual_only-r0_union", "visual_only-r1_union",
             "numeric_only-r0_union", "numeric_only-r1_union"]
    for target, panel in zip(TRAINING, axes, strict=True):
        values = [selections[target][name]["scores"]["daily"]["f1"] for name in names]
        panel.barh(np.arange(len(names)), values, color=["#687d86", "#aa9a70", "#277c71"] + ["#5e8cac"] * 6)
        panel.set_yticks(np.arange(len(names)), names)
        panel.set_xlim(.8, 1)
        panel.invert_yaxis()
        panel.set_xlabel("Daily activity F1 | historical validation")
        panel.set_title(target)
        for index, value in enumerate(values):
            panel.text(value + .001, index, f"{value:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=140)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    write_csv(out / "summary.csv", selected_numbers)
    write_csv(out / "per-record-corrections.csv", all_corrections)
    write_csv(out / "generation-cost.csv", generation)
    write_csv(out / "inference-cost.csv", costs)
    write_csv(out / "family-scores.csv", family_rows)
    export(out / "cost.json", {"new_requests": len(generation), "new_receipts": len(generation),
                               "new_reported_tokens": sum(row["tokens"] for row in generation),
                               "generation_api_seconds": sum(row["seconds"] for row in generation),
                               "new_evaluation_calls": 0})
    export(out / "verification.json", {"generation_payloads_reconstructed": len(generation),
                                       "original_proposals_unchanged": True,
                                       "reviewed_probe_exact_replay": True,
                                       "selected_weights_probability_prediction_exact_match": True,
                                       "selected_metrics_sklearn_and_events_boundary_match": True,
                                       "all34_selection_opportunities_match": True,
                                       "calendar_days_checked": checked_days,
                                       "new_api_calls": 0, "retraining": False})
    (out / "results.md").write_text("\n".join(summary_text) + "\n", encoding="utf-8")
    export(out / "manifest.json", {"files": {path.relative_to(out).as_posix(): sha(path)
                                             for path in out.rglob("*") if path.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report(args.run.resolve(), args.out.resolve())
