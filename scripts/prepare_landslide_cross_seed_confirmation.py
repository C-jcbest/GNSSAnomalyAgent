"""Freeze the unchanged range method, then generate a different-seed confirmation."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from prepare_landslide_motion_probe import FAMILIES, controlled_case
from probe_landslide_revision_scope import PROTOCOL as SCOPE_PROTOCOL
from run_landslide_comparison import ROOT, observation_features, save_json, sha
from run_landslide_evidence_display_confirmation import numerical_predictions
from run_landslide_window_optimization import read_json

from gnss_sim.landslide import LandslideRequest
from gnss_sim.landslide_diagnostics import derive_motion

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"


def prepare(out):
    indices = list(range(108, 120))
    case_ids = [f"case_{index + 1:04d}" for index in indices]
    run_name = "cross-seed-confirmation-v1"
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "date": "2026-10-10", "schema": "cross-seed-confirmation-v1", "case_ids": case_ids,
        "primary": "adaptive_no_feedback + conflict_scope", "candidate_policy": SCOPE_PROTOCOL,
        "candidate_source_sha256": sha(ROOT / "scripts/probe_landslide_revision_scope.py"),
        "generation": {"seed": 2026101073, "indices_zero_based": indices, "days": 730,
                       "families": list(FAMILIES), "records_per_family": 2},
        "selection_evidence": "Unchanged original range method, selected before this seed existed; "
                              "all later background observation/approval extensions remain stopped",
        "scope": "Independent random realizations in the same controlled family, not a new site "
                 "or independent expert observable field reference",
        "training": "Original controlled train24/validation12; nine fitted controls retain their "
                    "complete existing calibration opportunities, no test fitting or threshold search",
        "scope_gate": "Both repeats lose no TP or weak recall, increase no FP and remove at least "
                      "one FP versus whole-target review; otherwise scope benefit is unconfirmed",
        "method_gate": "Both repeats activity F1 greater than direct and fixed whole-target review; "
                       "weak recall at least direct; net definite correction versus direct positive "
                       "in at least two records. Report all numerical/event/boundary/stage tradeoffs",
        "failure_action": "No retuning, best-arm replacement or another seed selected to rescue "
                          "this confirmation. A failure limits the corresponding contribution claim",
        "labels": "Exact controlled latent velocity/activity and ASD, raw observation gaps masked. "
                  "These references validate mechanism recovery only, not field-visible activity",
        "cost": "Two repeats, unchanged original runner, no retry, ceiling168 unique requests; "
                "fixed/adaptive share identical local answers. Scope replay adds no inference",
    }
    save_json(out / "protocol.json", protocol)
    model_folder = BASE / "evidence-display-confirmation-v2"
    model_files = (
        "stage-xgboost.json", "activity-xgboost.json", "original-selection.json", "raw-selection.json",
        "activity-xgboost-raw-persistence.json", "long-selection.json", "activity-xgboost-84.json",
        "matched-selection.json", "activity-xgboost-42-matched-support.json",
    )
    bindings = {}
    for name in model_files:
        source = model_folder / name
        shutil.copyfile(source, out / name)
        bindings[str(source)] = sha(source)
    save_json(out / "training-bindings.json", bindings)
    sources = list((ROOT / "src/gnss_sim").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    for source in sources:
        destination = out / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    config = {
        "schema_version": "landslide-adaptive-review-v1", "date": "2026-10-10",
        "scope": protocol["scope"], "source_bundle": out.relative_to(ROOT).as_posix(),
        "case_ids": case_ids, "model": "qwen3.8-flash", "repetitions": 2,
        "window_days": 180, "stride_days": 90, "policy": SCOPE_PROTOCOL["policy"],
        "maximum_requests": 168, "hard_timeout_seconds": 3600, "retries": 0,
        "primary_candidate": protocol["primary"], "stage_policy": "Common original trained numeric head",
        "confirmation_protocol": str(out / "protocol.json"), "run_name": run_name,
    }
    save_json(out / "runner-config.json", config)
    save_json(out / "pre-generation-freeze.json", {
        path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()
    })
    # Generation starts only after method, controls, scorer and conclusion gates freeze.
    request = LandslideRequest(seed=protocol["generation"]["seed"], count=120, days=730)
    records, reference_rows = {}, []
    for index in indices:
        case, labels, truth = controlled_case(request, index)
        folder = out / "inputs" / case.case_id
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        reference = out / "mechanism-reference" / case.case_id
        reference.mkdir(parents=True)
        save_json(reference / "truth.json", truth)
        motions = {}
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            motions[window] = motion
            (folder / f"motion-{window}.json").write_text(motion.model_dump_json(indent=2), encoding="utf-8")
        records[case.case_id] = (case, observation_features(case, motions))
        reference_rows.extend(labels)
        print(f"Prepared cross-seed input {case.case_id}", flush=True)
    with (out / "reference-labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)
    save_json(out / "numeric-predictions.json", numerical_predictions(out, records))
    for name, expected in read_json(out / "pre-generation-freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Pre-generation method or scorer changed")
    save_json(out / "manifest.json", {"files": {
        path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()
    }})
    print({"records": len(records), "seed": request.seed, "new_api": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out.resolve())
