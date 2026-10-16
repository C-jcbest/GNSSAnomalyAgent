"""Evaluate frozen paired heads on all 24 exposed core records and controls."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from encode_landslide_paired_persistence import verify_inputs
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_contrastive_exemplars import SOURCE, raw_values
from probe_landslide_endpoint_fit import BASE, ROOT, export
from probe_landslide_feature_discovery import expert_features
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_paired_persistence import (
    ARMS,
    calibrated_prediction,
    modality_inputs,
    probabilities,
)
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_paired_persistence import pretrained_activity_head, render_encoder_chart
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, invariance_inputs, numeric_input

DATA = BASE / "core-lockbox-data-v1"
OLD = BASE / "core-lockbox-analysis-v1/primary_process"


def run_development(calibration, out):
    verify_inputs(calibration)
    verify_bindings(calibration, "calibration-freeze.json")
    if read_json(calibration / "calibration-decision.json")["decision"] != "PROCEED_PAIRED_PERSISTENCE_DEVELOPMENT":
        raise ValueError("Complete development only after the registered calibration gate")
    out.mkdir(exist_ok=False)
    ids = read_json(DATA / "protocol.json")["case_ids"]
    selection = read_json(calibration / "selection.json")
    export(out / "protocol.json", {
        "scope": "All 24 exposed core-lockbox process records, development only",
        "case_ids": ids, "calibration": calibration.relative_to(ROOT).as_posix(), "selection": selection,
        "refitting": False, "new_api_calls": 0, "independent_confirmation": False,
        "old_controls": "All original37 primary-process methods, including directVLM/fixedAgent/rules/PELT/strongXGB",
        "stage": "Identical existing process stage head, same raw84 support and original stage gate",
        "gate": "MainF1>=augmentation and shuffled; main-plain>=.005; net corrections>plain and augmentation; "
                "weakRecall>=plain/augmentation; >=2 positive-net records vs each; main>=numeric_pair F1 "
                "or higher eventF1 with dailyF1 loss<=.01; F1 or eventF1 strictly higher than augmentation. "
                "All37 old methods and three same-augmentation XGB are reported, no universal numerical win required",
        "stop": "Stop fixed candidate on failing development, no seed/threshold/render/margin rescue",
        "next_if_pass": "Freeze one new complete controlled batch and its independent-of-selection evaluation; visual/numeric reference before specific truth/predictions",
    })
    sources = [Path(__file__), ROOT / "scripts/train_landslide_paired_persistence.py",
               ROOT / "src/gnss_sim/landslide_paired_persistence.py", ROOT / "scripts/score_landslide_claim_review.py"]
    bindings = {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}
    for index, path in enumerate(sources):
        destination = out / "source" / f"{index:02d}-{path.name}"
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(path.read_bytes())
    for cid in ids:
        for path in (DATA / "inputs" / cid).iterdir():
            if path.is_file():
                bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (OLD / "predictions.json", OLD / "reference-labels.json", calibration / "selection.json",
                 calibration / "numeric-scaling.json", SOURCE / "stage-xgboost.json",
                 BASE / "paired-persistence-runtime-v1/mobileclip2-s0-vision.pt"):
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    export(out / "root-bindings.json", bindings)
    export(out / "pre-inference-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "inference", "completed_records": 0})
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    config = open_clip.get_pretrained_cfg("MobileCLIP2-S0", "dfndr2b")
    encoder, _, transform = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained=None,
        image_mean=config["mean"], image_std=config["std"], image_interpolation=config["interpolation"],
        image_resize_mode=config["resize_mode"])
    encoder.visual.load_state_dict(torch.load(BASE / "paired-persistence-runtime-v1/mobileclip2-s0-vision.pt", weights_only=True))
    encoder.eval()
    encoder.requires_grad_(False)
    if str(transform) != read_json(calibration / "encoding-cost.json")["actual_transform"]:
        raise ValueError("Pretrained preprocessing changed")
    models = {}
    for name, chosen in selection.items():
        if name in ARMS:
            arm = ARMS[name]
            model = pretrained_activity_head(512 if arm["image"] else 0, 36 if arm["numeric"] else 0, chosen["width"])
            model.load_state_dict(torch.load(calibration / "models" / f"{name}-width{chosen['width']}.pt", weights_only=True))
        else:
            model = XGBClassifier()
            model.load_model(calibration / "models" / f"{name}-depth{chosen['depth']}.json")
        models[name] = model
    scaling = read_json(calibration / "numeric-scaling.json")
    stage_head = XGBClassifier()
    stage_head.load_model(SOURCE / "stage-xgboost.json")
    predictions = {name: {} for name in list(ARMS) + ["xgb_raw18", "xgb_expert24", "xgb_chart384"]}
    costs, observed = [], {}
    for cid in ids:
        case, features = load_case_features(DATA, cid)
        before = time.perf_counter()
        charts, numbers, support = invariance_inputs(case, features)
        feature_seconds = time.perf_counter() - before
        embeddings = np.zeros((len(support), 512), dtype=np.float32)
        indices = np.flatnonzero(support)
        before = time.perf_counter()
        with torch.inference_mode():
            for first in range(0, len(indices), 32):
                selected = indices[first:first + 32]
                inputs = torch.stack([transform(render_encoder_chart(charts[index])) for index in selected])
                embeddings[selected] = encoder.encode_image(inputs, normalize=True).numpy()
        encoding_seconds = time.perf_counter() - before
        descriptors = chart_numeric_descriptors(charts)
        expert = expert_features(discovery_block_matrix(raw_values(case), features.noise)[0])
        normalized = numeric_input(numbers, scaling)
        codes = stage_head.predict(features.matrix).astype(int) + 1
        probabilities_by_method = {}
        head_costs = {}
        for name, model in models.items():
            before = time.perf_counter()
            if name in ARMS:
                probability = probabilities(model, modality_inputs(embeddings, normalized, ARMS[name]))
            else:
                matrix = numbers
                if name == "xgb_expert24":
                    matrix = np.column_stack((numbers, expert))
                elif name == "xgb_chart384":
                    matrix = np.column_stack((numbers, descriptors))
                probability = model.predict_proba(matrix)[:, 1]
            predictions[name][cid] = calibrated_prediction(probability, support, codes, features.supported,
                                                         selection[name]["threshold"]).to_dict()
            head_costs[name] = time.perf_counter() - before
            probabilities_by_method[name] = probability.tolist()
        np.savez_compressed(out / (cid + "-inputs.npz"), embeddings=embeddings, numbers=numbers,
            supported=support, observed=features.observed, stage_support=features.supported, codes=codes,
            chart_descriptors=descriptors, expert_features=expert)
        export(out / (cid + "-probabilities.json"), probabilities_by_method)
        observed[cid] = features.observed
        costs.append({"case_id": cid, "original_dates": len(case.dates), "encoded_images": len(indices),
                      "feature_seconds": feature_seconds, "render_encoder_seconds": encoding_seconds, "heads": head_costs})
        save_json(out / "state.json", {"status": "inference", "completed_records": len(costs),
                                      "encoded_images": sum(row["encoded_images"] for row in costs)})
        print(cid, len(indices), round(encoding_seconds, 3), flush=True)
    export(out / "new-predictions.json", predictions)
    export(out / "inference-cost.json", {"rows": costs, "wall_seconds": time.perf_counter() - started, "new_api_calls": 0})
    export(out / "prediction-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file() and path.name != "state.json"})
    # Existing exposed reference is opened only after every new prediction is frozen.
    labels = read_json(OLD / "reference-labels.json")
    if set(labels) != set(ids):
        raise ValueError("Complete original evaluation population changed")
    predictions.update(read_json(OLD / "predictions.json"))
    weak = {cid: np.array([int(row["activity_label"]) == 1 and int(row.get("weak_signal", 0)) == 1
                           for row in labels[cid]]) for cid in ids}
    results, summary, records = score_target(labels, observed, weak, predictions, {}, {})
    lookup = {row["method"]: row for row in summary}
    main = lookup["visual_numeric_pair"]
    plain, augment, shuffled, numeric = [lookup[name] for name in (
        "visual_numeric_plain", "visual_numeric_augmentation", "visual_numeric_shuffled_pairs", "numeric_pair")]
    corrections = []
    for name, cases in predictions.items():
        if name == "visual_numeric_pair":
            continue
        for cid, value in cases.items():
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            known = actual >= 0
            before, after = np.asarray(value["activity"]), np.asarray(predictions["visual_numeric_pair"][cid]["activity"])
            gained = int((known & (before != actual) & (after == actual)).sum())
            lost = int((known & (before == actual) & (after != actual)).sum())
            corrections.append({"control": name, "case_id": cid, "gained": gained, "lost": lost, "net": gained - lost})
    checks = {
        "f1_nondecrease_augmentation": main["f1"] >= augment["f1"],
        "f1_nondecrease_shuffled": main["f1"] >= shuffled["f1"],
        "f1_gain_plain_005": main["f1"] - plain["f1"] >= .005,
        "strict_pairing_gain": main["f1"] > augment["f1"] or main["event_f1_iou03"] > augment["event_f1_iou03"],
        "weak_nondecrease": main["weak_recall"] >= max(plain["weak_recall"], augment["weak_recall"]),
        "visual_value_vs_numeric": main["f1"] >= numeric["f1"] or (
            main["event_f1_iou03"] > numeric["event_f1_iou03"] and main["f1"] >= numeric["f1"] - .01),
    }
    for control in ("visual_numeric_plain", "visual_numeric_augmentation"):
        relevant = [row for row in corrections if row["control"] == control]
        checks["net_" + control] = sum(row["net"] for row in relevant) > 0
        checks["two_records_" + control] = sum(row["net"] > 0 for row in relevant) >= 2
    decision = "PROCEED_PAIRED_PERSISTENCE_CONFIRMATION" if all(checks.values()) else "STOP_PAIRED_PERSISTENCE_DEVELOPMENT"
    export(out / "predictions.json", predictions)
    export(out / "results.json", results)
    export(out / "summary.json", summary)
    export(out / "corrections.json", corrections)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", records)
    export(out / "decision.json", {"decision": decision, "checks": checks, "passed": sum(checks.values()),
                                   "total": len(checks), "independent_confirmation": False})
    for name, expected in bindings.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Development source/evidence changed: " + name)
    save_json(out / "state.json", {"status": "complete", "decision": decision})
    export(out / "result-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()})
    print(decision, sum(checks.values()), "/", len(checks), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    run_development(arguments.calibration.resolve(), arguments.out.resolve())
