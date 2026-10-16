"""Freeze conflict-scoped revision and extend the controlled family to new indices."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from fit_landslide_raw_persistence_baselines import raw_gate
from prepare_landslide_motion_probe import FAMILIES, PROTOCOL, controlled_case
from probe_landslide_revision_scope import PROTOCOL as SCOPE_PROTOCOL
from run_landslide_comparison import ROOT, observation_features, predict_baselines, save_json, sha
from run_landslide_window_optimization import read_json
from xgboost import XGBClassifier

from gnss_sim.landslide import LandslideRequest
from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_local_experiment import learned_stages


def prepare(out):
    base = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
    training, raw = base / "motion-probe-v2", base / "raw-baselines-v1"
    out.mkdir(parents=True, exist_ok=False)
    indices = list(range(72, 84))
    case_ids = [f"case_{index + 1:04d}" for index in indices]
    source_paths = (
        Path(__file__), ROOT / "scripts/probe_landslide_revision_scope.py",
        ROOT / "scripts/prepare_landslide_motion_probe.py",
        ROOT / "scripts/fit_landslide_raw_persistence_baselines.py",
    )
    protocol = {
        "schema": "landslide-revision-scope-confirmation-v1", "case_ids": case_ids,
        "primary_candidate": "adaptive_no_feedback + conflict_scope",
        "selection_evidence": "revision-scope-probe-v1, three viewed controlled batches. "
                              "Previously confirmed intervals do not confirm this candidate.",
        "candidate_source_sha256": sha(ROOT / "scripts/probe_landslide_revision_scope.py"),
        "candidate_policy": SCOPE_PROTOCOL,
        "generation": {"seed": PROTOCOL["seed"], "indices_zero_based": indices,
                       "days": 730, "families": list(FAMILIES)},
        "training": "Same original train24/validation12 models and validation opportunities; "
                    "no new index used for fitting/calibration",
        "reference": "Known controlled latent activity and ASD with missing observations unscored; "
                     "not independent expert observable field reference",
        "decision_rule": "Do not promote if either repetition loses TP or weak recall, increases "
                         "FP, or reduces daily F1 relative to whole-target unchanged default. "
                         "Both repetitions must remove at least one FP. Otherwise reject or "
                         "retain as inconclusive; do not retune on these records.",
        "cost": "Same initial and local queries; no additional VLM query for scope mask. "
                "Measure deterministic mask cost separately. Do not claim field correction.",
        "controls": "Whole target, matched editable-day centre scope, four original workflows, "
                    "six numerical methods and same-data frozen TAMA GNSS adaptation",
        "source_bindings": {str(path): sha(path) for path in source_paths},
    }
    save_json(out / "protocol.json", protocol)
    (out / "source").mkdir()
    for path in source_paths:
        shutil.copyfile(path, out / "source" / path.name)
    for file in ("stage-xgboost.json", "activity-xgboost.json", "frozen-selection.json"):
        shutil.copyfile(training / file, out / file)
    save_json(out / "training-bindings.json", {str(path): sha(path) for path in (
        training / "manifest.json", training / "frozen-selection.json",
        training / "stage-xgboost.json", training / "activity-xgboost.json",
        raw / "frozen-selection.json", raw / "activity-xgboost-raw-persistence.json",
    )})
    request = LandslideRequest(seed=PROTOCOL["seed"], count=84, days=730)
    features, evidence, rows = {}, {}, []
    for index in indices:
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
            (folder / f"motion-{window}.json").write_text(
                motion.model_dump_json(indent=2), encoding="utf-8")
        features[cid] = observation_features(case, motions)
        evidence[cid] = raw_motion_evidence(case, features[cid])
        rows.extend(labels)
        print(f"New scope confirmation input {cid}", flush=True)
    with (out / "reference-labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    activity_model, stage_model, raw_model = XGBClassifier(), XGBClassifier(), XGBClassifier()
    activity_model.load_model(out / "activity-xgboost.json")
    stage_model.load_model(out / "stage-xgboost.json")
    raw_model.load_model(raw / "activity-xgboost-raw-persistence.json")
    predictions = predict_baselines(features, case_ids, read_json(out / "frozen-selection.json"),
                                    activity_model, stage_model)
    save_json(out / "numeric-predictions.json", {
        method: {cid: value.to_dict() for cid, value in records.items()}
        for method, records in predictions.items()
    })
    choices = read_json(raw / "frozen-selection.json")
    additional = {method: {} for method in choices}
    for cid in case_ids:
        feature, long = features[cid], evidence[cid]
        matrix = np.column_stack((feature.matrix, long["rate"], long["signal_to_noise"], long["direction"]))
        rule, learned = choices["raw_multiscale_rule"], choices["xgboost_raw_persistence"]
        gates = {
            "raw_multiscale_rule": raw_gate(feature, long, rule["rate"], rule["snr"]),
            "xgboost_raw_persistence": activity_from_mask(
                raw_model.predict_proba(matrix)[:, 1] >= learned["threshold"], feature.supported),
        }
        stage_codes = stage_model.predict(feature.matrix).astype(int) + 1
        for method, activity in gates.items():
            additional[method][cid] = learned_stages(stage_codes, activity, feature.supported).to_dict()
    (out / "additional-baselines").mkdir()
    save_json(out / "additional-baselines/predictions.json", additional)
    save_json(out / "manifest.json", {"files": {
        str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()
    }})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out)


if __name__ == "__main__":
    main()
