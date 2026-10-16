"""Replay finite contrastive reviews and score original queried dates only."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_contrastive_exemplars import (
    ANCHOR,
    SOURCE,
    apply_labels,
    parse_labels,
    raw_values,
    serial_vector,
)
from probe_landslide_endpoint_fit import ROOT, export
from probe_landslide_visual_invariance import calibration_labels
from report_landslide_core_lockbox import full_table
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_core_lockbox import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_contrastive_exemplars import (
    contrastive_neighbors,
    fixed_class_examples,
    motion_shape,
)
from gnss_sim.landslide_feature_discovery import discovery_block_matrix


def local_prediction(prediction, start, stop):
    return {**prediction, "activity": prediction["activity"][start:stop],
            "stage": prediction["stage"][start:stop]}


def verify_payload(run, item, packet):
    request_id = item["request_id"]
    receipt = read_json(run / "requests" / (request_id + ".response.json"))
    start = read_json(run / "requests" / (request_id + ".started.json"))
    content = [{"type": "text", "text": packet["prompt"]}]
    images = [run / name for name in packet["images"]]
    for path in images:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
    payload = {"model": "qwen3.8-flash", "messages": [{"role": "user", "content": content}],
               "temperature": .1, "top_p": .3, "max_tokens": 2048,
               "enable_thinking": False, "response_format": {"type": "json_object"}}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    for value in (receipt, start):
        if (value["request_sha256"] != expected or value["prompt"] != packet["prompt"]
                or value["image_sha256"] != {path.name: sha(path) for path in images}):
            raise ValueError("Frozen request does not reconstruct")
    return receipt


def report(run, out):
    verify_bindings(run, "pre-inference-freeze.json")
    verify_bindings(run, "inference-freeze.json")
    for name, expected in read_json(run / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source changed")
    if read_json(run / "state.json")["status"] != "inference_complete":
        raise ValueError("Only terminal inference may be scored")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    bank = read_json(run / "training-bank.json")
    split = read_json(SOURCE / "protocol.json")["split"]
    pairs = {cid: load_case_features(SOURCE, cid) for cid in split["train"] + split["validation"]}
    labels = calibration_labels(SOURCE, pairs)
    blocks = {cid: discovery_block_matrix(raw_values(case), features.noise)
              for cid, (case, features) in pairs.items()}
    for cid, pair in pairs.items():
        if pair[0].model_dump(mode="json") != read_json(run / "inputs" / (cid + ".json")):
            raise ValueError("Original observations differ")
    for row in bank:
        cid, day = row["case_id"], row["day"]
        if cid not in split["train"]:
            raise ValueError("Nontraining record entered exemplar bank")
        descriptor = serial_vector(motion_shape(blocks[cid][0][day]))
        if descriptor != row["descriptor"]:
            raise ValueError("Training descriptor changed")
        actual = np.array([int(value["activity_label"]) for value in labels[cid][day - 7:day + 7]])
        observed = pairs[cid][1].observed[day - 7:day + 7]
        if np.any(actual[observed] != row["center_label"]):
            raise ValueError("Training center label not homogeneous")
    if fixed_class_examples(bank) != read_json(run / "fixed-examples.json"):
        raise ValueError("Fixed class medoids changed")
    selected = read_json(run / "selected-queries.json")
    for row in selected:
        cid, day = row["case_id"], row["day"]
        neighbors = contrastive_neighbors(motion_shape(blocks[cid][0][day]), bank)
        if neighbors != row["neighbors"] or cid not in split["validation"]:
            raise ValueError("Retrieval ranking or query partition changed")
    packets = read_json(run / "packets.json")
    outputs = read_json(run / "outputs.json")
    frozen = read_json(ANCHOR / "primary_process/validation-predictions.json")
    head = XGBClassifier()
    head.load_model(SOURCE / "stage-xgboost.json")
    predictions = {name: {} for name in ("base18", "base_expert", "base_blocks", "nearest_class")}
    local_labels, observed, weak, receipts, used = {}, {}, {}, {}, {}
    checked_days = 0
    for index, row in enumerate(selected):
        key = f"q{index + 1}"
        packet = packets[key + ":retrieved_contrast"]
        cid, day = row["case_id"], row["day"]
        start, stop = packet["query"]
        if packet["stage_codes"] != (head.predict(pairs[cid][1].matrix).astype(int) + 1).tolist():
            raise ValueError("Shared stage head changed")
        local_labels[key] = labels[cid][start:stop]
        observed[key] = pairs[cid][1].observed[start:stop]
        weak[key] = np.array([int(value.get("weak_signal", 0)) == 1
                              and int(value["activity_label"]) == 1 for value in local_labels[key]])
        for name in ("base18", "base_expert", "base_blocks"):
            predictions[name][key] = local_prediction(frozen[name][cid], start, stop)
        neighbors = row["neighbors"]
        code = int(neighbors["active"]["distance"] < neighbors["inactive"]["distance"])
        numeric = apply_labels(packet, frozen["base_expert"][cid], np.full(14, code))
        predictions["nearest_class"][key] = local_prediction(numeric, start, stop)
    for item in read_json(run / "schedule.json"):
        key, arm, repetition = item["query_id"], item["arm"], item["repetition"]
        request_id = item["request_id"]
        packet = packets[key + ":" + arm]
        receipt = verify_payload(run, item, packet)
        receipts[request_id] = receipt
        baseline = read_json(run / (key + "-baseline.json"))
        value = outputs[request_id]
        if value["status"] == "success":
            if receipt["status"] != "returned_json":
                raise ValueError("Successful decision lacks JSON")
            codes = parse_labels(json.loads(receipt["output"]))
            replayed = apply_labels(packet, baseline, codes)
            if codes.tolist() != value["labels"]:
                raise ValueError("Stored original-day codes differ")
        else:
            replayed = baseline
            try:
                parse_labels(json.loads(receipt["output"]))
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
            else:
                raise ValueError("Failed review no longer fails parsing")
        if replayed != value["prediction"]:
            raise ValueError("Frozen response replay differs")
        activity, stage = np.array(replayed["activity"]), np.array(replayed["stage"])
        initial = np.array(baseline["activity"])
        permission = np.zeros(len(activity), dtype=bool)
        start, stop = packet["query"]
        permission[start:stop] = True
        permission &= np.array(packet["support"]) & (initial >= 0)
        if (np.any(activity[~permission] != initial[~permission])
                or np.any(activity[~np.array(packet["observed"])] != -1)
                or np.any(stage[~np.array(packet["observed"])] != -1)
                or np.any(stage[activity != 1] > 0)
                or np.any(stage[(activity == 1) & ~np.array(packet["stage_support"])] != -1)):
            raise ValueError("Original support/activity/stage contract violated")
        name = arm + f"_r{repetition}"
        predictions.setdefault(name, {})[key] = local_prediction(replayed, start, stop)
        used.setdefault(name, {})[key] = [request_id]
        checked_days += len(activity)
    results, summary, per_case = score_target(local_labels, observed, weak, predictions, used, receipts)
    changes = []
    for method, cases in predictions.items():
        for key, prediction in cases.items():
            actual = np.array([int(row["activity_label"]) for row in local_labels[key]])
            known = actual >= 0
            before = np.array(predictions["base_expert"][key]["activity"])
            after = np.array(prediction["activity"])
            gained = int((known & (before != actual) & (after == actual)).sum())
            lost = int((known & (before == actual) & (after != actual)).sum())
            role = selected[int(key[1:]) - 1]["role"]
            changes.append({"method": method, "query_id": key, "role": role,
                            "case_id": selected[int(key[1:]) - 1]["case_id"],
                            "gained_vs_expert": gained, "lost_vs_expert": lost, "net": gained - lost})
    for row in summary:
        members = [change for change in changes if change["method"] == row["method"]]
        row["gained_vs_expert"] = sum(change["gained_vs_expert"] for change in members)
        row["lost_vs_expert"] = sum(change["lost_vs_expert"] for change in members)
        row["net_vs_expert"] = row["gained_vs_expert"] - row["lost_vs_expert"]
        row["benefiting_records"] = sum(change["net"] > 0 for change in members)
        row["control_correct_loss"] = sum(change["lost_vs_expert"] for change in members
                                         if change["role"] != "detector_disagreement")
    by_method = {row["method"]: row for row in summary}
    conditions = {}
    for repetition in range(2):
        main = by_method[f"retrieved_contrast_r{repetition}"]
        raw = by_method[f"target_only_r{repetition}"]
        fixed = by_method[f"fixed_contrast_r{repetition}"]
        base = by_method["base_expert"]
        conditions[f"r{repetition}_above_expert_f1"] = main["f1"] > base["f1"]
        conditions[f"r{repetition}_net_vs_expert"] = main["net_vs_expert"] > 0
        conditions[f"r{repetition}_above_target_f1"] = main["f1"] > raw["f1"]
        net_vs_raw = 0
        for key in local_labels:
            actual = np.array([int(row["activity_label"]) for row in local_labels[key]])
            known = actual >= 0
            a = np.array(predictions[f"retrieved_contrast_r{repetition}"][key]["activity"])
            b = np.array(predictions[f"target_only_r{repetition}"][key]["activity"])
            net_vs_raw += int((known & (a == actual)).sum() - (known & (b == actual)).sum())
        conditions[f"r{repetition}_net_vs_target"] = net_vs_raw > 0
        conditions[f"r{repetition}_benefit_two_records"] = main["benefiting_records"] >= 2
        conditions[f"r{repetition}_weak_no_loss"] = main["weak_recall"] >= base["weak_recall"]
        conditions[f"r{repetition}_controls_no_loss"] = main["control_correct_loss"] == 0
        conditions[f"r{repetition}_fixed_no_loss"] = main["f1"] >= fixed["f1"]
        conditions[f"r{repetition}_nearest_no_loss"] = main["f1"] >= by_method["nearest_class"]["f1"]
    conditions["fixed_strict_once"] = any(by_method[f"retrieved_contrast_r{i}"]["f1"]
                                         > by_method[f"fixed_contrast_r{i}"]["f1"] for i in range(2))
    conditions["numeric_feedback_strict_once"] = any(by_method[f"retrieved_contrast_r{i}"]["f1"]
                                         > by_method[f"retrieved_no_numeric_r{i}"]["f1"] for i in range(2))
    crop_checks = []
    for key in local_labels:
        main_path = run / (key + "-retrieved_contrast.png")
        with Image.open(main_path) as picture:
            main_image = np.array(picture)
        height, width = main_image.shape[:2]
        for arm in ("target_only", "fixed_contrast", "retrieved_no_numeric"):
            with Image.open(run / (key + "-" + arm + ".png")) as picture:
                other = np.array(picture)
            left, right = int(width * .385), int(width * .66)
            crop_checks.append({"query_id": key, "arm": arm,
                "target_crop_exact": bool(np.array_equal(main_image[70:height - 30, left:right],
                                                         other[70:height - 30, left:right])),
                "whole_image_exact": sha(main_path) == sha(run / (key + "-" + arm + ".png"))})
    export(out / "image-layout-checks.json", crop_checks)
    matched_target_pixels = all(row["target_crop_exact"] for row in crop_checks)
    # The original preregistered pixel matching claim was contradicted by actual
    # image checks. Preserve that protocol defect alongside the efficacy gate.
    decision = {"conditions": conditions, "passed": sum(conditions.values()), "total": len(conditions),
                "decision": "PROCEED_TO_FIXED_LAYOUT_CONFIRMATION" if all(conditions.values())
                            else "STOP_CONTRASTIVE_EXEMPLAR_PILOT",
                "matched_target_pixels": matched_target_pixels,
                "limitation": "tight_layout shifts target pixels in target-only/fixed controls; "
                              "main/no-numeric images identical; original data/scales unchanged. "
                              "Do not attribute all other arm differences solely to retrieval.",
                "independent_confirmation": False}
    export(out / "decision.json", decision)
    export(out / "results.json", results)
    export(out / "predictions.json", predictions)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    write_csv(out / "per-query-corrections.csv", changes)
    costs = [{"request_id": key, "tokens": receipt.get("usage", {}).get("total_tokens"),
              "seconds": receipt["seconds"], "status": outputs[key]["status"]} for key, receipt in receipts.items()]
    write_csv(out / "request-cost.csv", costs)
    export(out / "cost.json", {"new_requests": len(costs), "new_receipts": len(costs),
                               "reported_tokens": sum(row["tokens"] for row in costs),
                               "api_seconds": sum(row["seconds"] for row in costs),
                               "inference_loop_seconds": read_json(run / "state.json")["loop_seconds"],
                               "contract_failures": sum(row["status"] == "failed" for row in costs)})
    export(out / "verification.json", {"training_examples_original_labels_descriptors": len(bank),
                                       "fixed_and_retrieved_neighbors_exact_match": True,
                                       "original_input_calendars": len(pairs),
                                       "requests_payload_and_prediction_replay": len(costs),
                                       "support_unknown_stage_permission_days_checked": checked_days,
                                       "sklearn_daily_and_event_boundary_match": True,
                                       "target_pixel_matching": matched_target_pixels,
                                       "new_model_calls_during_report": 0})
    text = ["# 平衡中心正反训练图复核小试", "", decision["decision"], "",
            "8个由检测器预测提名的14日开发目标；不是完整活动方法或独立效果验证。",
            "主/去数值图完全相同；目标/固定对照有排版像素变化，不称严格像素配对。", "", full_table(summary), "",
            json.dumps(decision, ensure_ascii=False, indent=2)]
    (out / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(by_method)
    ax.bar(np.arange(len(names)), [by_method[name]["f1"] for name in names], color="#448b91")
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Queried activity F1 | exposed development reference")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=140)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    export(out / "manifest.json", {"files": {path.relative_to(out).as_posix(): sha(path)
                                             for path in out.rglob("*") if path.is_file()}})
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report(args.run.resolve(), args.out.resolve())
