"""Freeze a simpler Agent before generating a separate confirmation probe batch."""
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
    out.mkdir(parents=True, exist_ok=False)
    case_ids = [f"case_{index:04d}" for index in range(49, 61)]
    protocol = {
        "date": "2026-10-09", "schema_version": "landslide-adaptive-confirmation-v1",
        "primary_method": "adaptive_no_feedback", "case_ids": case_ids,
        "generation": {"seed": PROTOCOL["seed"], "indices_zero_based": list(range(48, 60)),
                       "days": PROTOCOL["days"], "families": list(FAMILIES)},
        "selection_evidence": "pilot-v3 and mechanism-v1; the latter has now participated "
                              "in selecting the simpler default, so it is development evidence",
        "decision": "Retain numerical conflict selection and bounded image reviews. "
                    "Remove numeric text from the default because of weak-motion loss "
                    "in mechanism-v1 repeat 2. Keep numeric text only as a named ablation.",
        "controls": "Identical frozen model, policy, windows, review budget and contracts; "
                    "all previously calibrated numeric models reused at the same probe distribution",
        "training_reference": "Same original train24/validation12 records; neither the "
                              "previous12 nor new12 confirmation labels used for numerical fitting",
        "base_training_manifest_sha256": sha(training_source / "manifest.json"),
        "additional_training_selection_sha256": sha(raw_baselines / "frozen-selection.json"),
        "claims": "Controlled synthetic mechanism confirmation, not independent expert "
                  "observable references or real-site generalization",
        "test_use": "No retuning or repeat selection on these twelve new records"}
    save_json(out / "protocol.json", protocol)
    shutil.copyfile(Path(__file__), out / "source.py")
    for name in ("stage-xgboost.json", "activity-xgboost.json", "frozen-selection.json"):
        shutil.copyfile(training_source / name, out / name)
    request = LandslideRequest(seed=PROTOCOL["seed"], count=60, days=PROTOCOL["days"])
    features, evidence, reference_rows = {}, {}, []
    for index in range(48, 60):
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
