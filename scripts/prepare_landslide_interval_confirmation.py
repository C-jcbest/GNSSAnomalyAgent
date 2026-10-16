"""Freeze the interval-check candidate, then generate twelve new controlled records."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from fit_landslide_raw_persistence_baselines import raw_gate
from prepare_landslide_motion_probe import FAMILIES, PROTOCOL, controlled_case
from probe_landslide_interval_consistency import PROTOCOL as INTERVAL_PROTOCOL
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
    training = base / "motion-probe-v2"
    raw = base / "raw-baselines-v1"
    out.mkdir(parents=True, exist_ok=False)
    case_ids = [f"case_{index:04d}" for index in range(61, 73)]
    protocol = {
        "schema_version": "landslide-interval-confirmation-v1", "case_ids": case_ids,
        "primary_candidate": "adaptive_no_feedback + require_support interval check",
        "previous_default": "adaptive_no_feedback without interval check; remains frozen",
        "selection_source": "interval-consistency-probe-v2; old confirmation now viewed "
                            "for this candidate and cannot confirm it",
        "candidate_source_sha256": sha(ROOT / "scripts/probe_landslide_interval_consistency.py"),
        "interval_policy": INTERVAL_PROTOCOL,
        "generation": {"seed": PROTOCOL["seed"], "indices_zero_based": list(range(60, 72)),
                       "days": 730, "families": list(FAMILIES)},
        "reference": "Known controlled latent process, not expert observable field labels",
        "training": "Unchanged train24/validation12; new records never used for tuning",
        "comparison": "Original four visual arms and six numerical baselines plus their "
                      "same deterministic interval checks; TAMA GNSS adaptation with/without review",
        "decision_rule": "Do not promote if either repeat loses daily F1 or weak recall "
                         "relative to unchanged default. Stable reduction of FP is necessary. "
                         "Zero effect is inconclusive; field seasonality is not presumed solved.",
        "new_confirmatory_claims": "Small synthetic mechanism confirmation only; no "
                                   "significance or field superiority claimed",
    }
    save_json(out / "protocol.json", protocol)
    shutil.copyfile(Path(__file__), out / "source.py")
    shutil.copyfile(ROOT / "scripts/probe_landslide_interval_consistency.py", out / "candidate-source.py")
    for name in ("stage-xgboost.json", "activity-xgboost.json", "frozen-selection.json"):
        shutil.copyfile(training / name, out / name)
    save_json(out / "training-bindings.json", {
        str(path): sha(path) for path in (
            training / "manifest.json", training / "frozen-selection.json",
            training / "stage-xgboost.json", training / "activity-xgboost.json",
            raw / "frozen-selection.json", raw / "activity-xgboost-raw-persistence.json",
        )
    })
    request = LandslideRequest(seed=PROTOCOL["seed"], count=72, days=730)
    features, evidence, rows = {}, {}, []
    for index in range(60, 72):
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
        print(f"New frozen confirmation input {cid}", flush=True)
    with (out / "reference-labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    activity_model, stage_model, raw_model = XGBClassifier(), XGBClassifier(), XGBClassifier()
    activity_model.load_model(out / "activity-xgboost.json")
    stage_model.load_model(out / "stage-xgboost.json")
    raw_model.load_model(raw / "activity-xgboost-raw-persistence.json")
    numeric = predict_baselines(features, case_ids, read_json(out / "frozen-selection.json"),
                                activity_model, stage_model)
    save_json(out / "numeric-predictions.json", {
        method: {cid: value.to_dict() for cid, value in records.items()}
        for method, records in numeric.items()
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
        stages = stage_model.predict(feature.matrix).astype(int) + 1
        for method, gate in gates.items():
            additional[method][cid] = learned_stages(stages, gate, feature.supported).to_dict()
    (out / "additional-baselines").mkdir()
    save_json(out / "additional-baselines/predictions.json", additional)
    save_json(out / "manifest.json", {
        "status": "complete", "files": {str(path.relative_to(out)): sha(path)
                                         for path in out.rglob("*") if path.is_file()}
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out)


if __name__ == "__main__":
    main()
