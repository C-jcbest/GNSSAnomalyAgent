"""Freeze raw-contrast stage repair before generating twelve new analytic records."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from fit_landslide_raw_persistence_baselines import raw_gate
from prepare_landslide_motion_probe import FAMILIES, PROTOCOL, controlled_case
from run_landslide_comparison import observation_features, predict_baselines, save_json, sha
from run_landslide_window_optimization import read_json
from xgboost import XGBClassifier

from gnss_sim.landslide import LandslideRequest
from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_local_experiment import learned_stages


def prepare(training_source, raw_baselines, out):
    development = training_source.parent / "stage-contrast-review-development-v2"
    if not read_json(development / "decision.json")["allows_new_confirmation"]:
        raise ValueError("Development stage repair did not satisfy the registered condition")
    out.mkdir(parents=True, exist_ok=False)
    case_ids = [f"case_{index:04d}" for index in range(97, 109)]
    protocol = {
        "date": "2026-10-09", "schema_version": "landslide-stage-contrast-confirmation-v1",
        "primary_method": "adaptive_numeric_feedback_conflict_scope_veto", "case_ids": case_ids,
        "generation": {"seed": PROTOCOL["seed"], "indices_zero_based": list(range(96, 108)),
                       "days": PROTOCOL["days"], "families": list(FAMILIES)},
        "selection_evidence": "stage-raw-consistency-development-v2, transfer-development-v1, "
                              "stage-contrast-review-development-v2; all are development evidence",
        "decision": "Freeze the development primary: raw42 stage conflicts, maximum one activity "
                    "review per record, numeric feedback, exact conflict-day permissions and "
                    "contradiction veto. Do not select pure image or full-target arms after confirmation.",
        "stage_policy": {"separation": 42, "relative_change": 0.25, "noise_multiplier": 2.0,
                         "minimum_conflict_days": 14, "maximum_reviews_per_record": 1},
        "activity_policy": "Unchanged direct raw VLM activity; all phase repairs keep it exactly fixed",
        "development_decision_sha256": sha(development / "decision.json"),
        "method_source_sha256": sha(Path(__file__).parent / "run_landslide_stage_contrast_confirmation.py"),
        "contrast_source_sha256": sha(Path(__file__).parent.parent / "src/gnss_sim/landslide_stage_raw_consistency_v2.py"),
        "controls": "Identical frozen model, policy, windows, review budget and contracts; "
                    "all previously calibrated numeric models reused at the same probe distribution",
        "training_reference": "Same original train24/validation12 records; neither the "
                              "any previous development nor new12 labels used for numerical fitting",
        "base_training_manifest_sha256": sha(training_source / "manifest.json"),
        "additional_training_selection_sha256": sha(raw_baselines / "frozen-selection.json"),
        "claims": "Controlled synthetic mechanism confirmation, not independent expert "
                  "observable references or real-site generalization",
        "test_use": "No retuning or repeat selection on these twelve new records"}
    save_json(out / "protocol.json", protocol)
    shutil.copyfile(Path(__file__), out / "source.py")
    for name in ("stage-xgboost.json", "activity-xgboost.json", "frozen-selection.json"):
        shutil.copyfile(training_source / name, out / name)
    request = LandslideRequest(seed=PROTOCOL["seed"], count=108, days=PROTOCOL["days"])
    features, evidence, reference_rows = {}, {}, []
    for index in range(96, 108):
        case, labels, truth = controlled_case(request, index)
        cid = case.case_id
        folder = out / "inputs" / cid
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        reference = out / "mechanism-reference" / cid
        reference.mkdir(parents=True)
        save_json(reference / "truth.json", truth)
        motions = {}
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            motions[window] = motion
            (folder / f"motion-{window}.json").write_text(motion.model_dump_json(indent=2),
                                                         encoding="utf-8")
        features[cid] = observation_features(case, motions)
        evidence[cid] = raw_motion_evidence(case, features[cid])
        reference_rows.extend(labels)
        print(f"Confirmation input {cid}", flush=True)
    with (out / "reference-labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)
    activity_model, stage_model, raw_model = XGBClassifier(), XGBClassifier(), XGBClassifier()
    activity_model.load_model(out / "activity-xgboost.json")
    stage_model.load_model(out / "stage-xgboost.json")
    raw_model.load_model(raw_baselines / "activity-xgboost-raw-persistence.json")
    numeric = predict_baselines(features, case_ids, read_json(out / "frozen-selection.json"),
                                activity_model, stage_model)
    save_json(out / "numeric-predictions.json", {
        name: {cid: value.to_dict() for cid, value in cases.items()}
        for name, cases in numeric.items()})
    choices = read_json(raw_baselines / "frozen-selection.json")
    additional = {name: {} for name in choices}
    for cid in case_ids:
        feature, long_evidence = features[cid], evidence[cid]
        matrix = np.column_stack((feature.matrix, long_evidence["rate"],
                                  long_evidence["signal_to_noise"], long_evidence["direction"]))
        rule = choices["raw_multiscale_rule"]
        learned = choices["xgboost_raw_persistence"]
        gates = {"raw_multiscale_rule": raw_gate(feature, long_evidence, rule["rate"], rule["snr"]),
                 "xgboost_raw_persistence": activity_from_mask(
                     raw_model.predict_proba(matrix)[:, 1] >= learned["threshold"], feature.supported)}
        stage_codes = stage_model.predict(feature.matrix).astype(int) + 1
        for method, gate in gates.items():
            additional[method][cid] = learned_stages(stage_codes, gate, feature.supported).to_dict()
    (out / "additional-baselines").mkdir()
    save_json(out / "additional-baselines" / "predictions.json", additional)
    save_json(out / "manifest.json", {"status": "complete", "files": {
        str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--raw-baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.training_source, args.raw_baselines, args.out)


if __name__ == "__main__":
    main()
