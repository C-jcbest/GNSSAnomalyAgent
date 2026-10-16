"""Calibrate an axis-preserving observation witness before any new VLM requests."""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from analyze_landslide_review_utility_v2 import verify_receipts
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import calibration_objective, load_labels
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_axis_persistence import axis_persistence, persistence_review_plan
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
DATA = BASE / "core-lockbox-data-v1"
PARENT = BASE / "core-lockbox-confirmation-v1"
PRIOR = BASE / "core-lockbox-analysis-v1"


def calibration(source, head_path, out, target):
    """Offer the new rule and information-matched ML the original tuning budget."""
    out.mkdir()
    split = read_json(source / "protocol.json")["split"]
    ids = split["train"] + split["validation"]
    pairs = {cid: load_case_features(source, cid) for cid in ids}
    cases = {cid: pair[0] for cid, pair in pairs.items()}
    # The original process CSV also contains its old mechanism-test records.
    # Export only the fixed calibration split before using the shared loader.
    with (source / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        calibration_rows = [row for row in csv.DictReader(stream) if row["case_id"] in cases]
    write_csv(out / "calibration-labels.csv", calibration_rows)
    labels = load_labels(out / "calibration-labels.csv", cases)
    evidence = {
        cid: axis_persistence(case, feature.noise) for cid, (case, feature) in pairs.items()
    }
    stage = XGBClassifier()
    stage.load_model(head_path)
    codes = {
        cid: stage.predict(feature.matrix).astype(int) + 1 for cid, (_, feature) in pairs.items()
    }
    matrices = {
        cid: np.column_stack(
            (feature.matrix, evidence[cid]["axis_score"], evidence[cid]["rank_halves"])
        )
        for cid, (_, feature) in pairs.items()
    }
    support = {
        cid: feature.observed & np.isfinite(evidence[cid]["score"])
        for cid, (_, feature) in pairs.items()
    }

    def score(gates):
        scores = [
            evaluate_case(labels[cid], learned_stages(codes[cid], gate, pairs[cid][1].supported))
            for cid, gate in gates.items()
        ]
        return aggregate_scores(scores)

    trials = []
    for threshold in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        gates = {
            cid: activity_from_mask(evidence[cid]["score"] >= threshold, support[cid])
            for cid in split["validation"]
        }
        result = score(gates)
        trials.append(
            {
                "method": "axis_persistence_rule",
                "threshold": threshold,
                "objective": calibration_objective(result),
                "validation": result,
            }
        )
    train_matrix = np.concatenate([matrices[cid] for cid in split["train"]])
    train_y = np.array(
        [int(row["activity_label"]) for cid in split["train"] for row in labels[cid]]
    )
    eligible = np.concatenate([support[cid] for cid in split["train"]]) & (train_y >= 0)
    models = {}
    for depth in (2, 4):
        model = XGBClassifier(
            n_estimators=120, max_depth=depth, learning_rate=0.05, n_jobs=2, random_state=20261007
        )
        model.fit(train_matrix[eligible], train_y[eligible])
        models[depth] = model
        probability = {cid: model.predict_proba(matrices[cid])[:, 1] for cid in split["validation"]}
        for threshold in (0.35, 0.5, 0.65):
            gates = {
                cid: activity_from_mask(probability[cid] >= threshold, support[cid])
                for cid in split["validation"]
            }
            result = score(gates)
            trials.append(
                {
                    "method": "xgboost_axis_persistence",
                    "threshold": threshold,
                    "depth": depth,
                    "objective": calibration_objective(result),
                    "validation": result,
                }
            )
    choices = {
        method: max(
            (row for row in trials if row["method"] == method), key=lambda row: row["objective"]
        )
        for method in ("axis_persistence_rule", "xgboost_axis_persistence")
    }
    write_json(out / "validation-trials.json", trials)
    write_json(out / "selection.json", choices)
    models[choices["xgboost_axis_persistence"]["depth"]].save_model(
        out / "activity-xgboost-axis.json"
    )
    shutil.copyfile(head_path, out / "stage-xgboost.json")
    paths = [source / "protocol.json", source / "reference-labels.csv", head_path]
    paths.extend(
        path for cid in ids for path in (source / "inputs" / cid).iterdir() if path.is_file()
    )
    write_json(out / "source-bindings.json", {str(path): sha(path) for path in paths})
    write_json(
        out / "protocol.json",
        {
            "target": target,
            "split": {key: split[key] for key in ("train", "validation")},
            "selection": "Validation dailyF1 + eventF1; nine rule and six ML trials; no development record in fitting",
            "support": "Observed and seven supported raw blocks; common existing stage support",
            "scale": "14-day disjoint blocks at centers t-42,-28,-14,0,+14,+28,+42; >=8 complete days each",
            "limitations": "Rank consistency and full-record daily MAD are descriptive, not independent white-noise significance; isolated step median may create one proxy day",
        },
    )
    print(
        target,
        "calibrated rule",
        choices["axis_persistence_rule"]["threshold"],
        "ML",
        choices["xgboost_axis_persistence"]["depth"],
        choices["xgboost_axis_persistence"]["threshold"],
        flush=True,
    )
    return choices


def prepare(out, calibration_source=None):
    verify_manifest(PRIOR)
    receipts = verify_receipts(PARENT)
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_axis_persistence.py",
        ROOT / "tests/test_landslide_axis_persistence.py",
    ]
    (out / "source").mkdir()
    for path in sources:
        shutil.copyfile(path, out / "source" / path.name)
    protocol = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "New axis-preserving persistence witness; complete24 exposed development records, not rescue of old failed feedback or independent confirmation",
        "single_change": "Replace vector short/long binary witness with same-axis bilateral rank-and-amplitude persistence; VLM images and prompt unchanged, no numerical-text feedback",
        "observation": "Original180-day windows,14 conflict days,nonoverlap,at most2;49-day authority from unchanged98-day raw envelope",
        "calibration": "Process train24/validation12; author target6/6 for its numerical controls. Method schedule uses process validation rule only, for both targets",
        "probe_gate_before_calls": "Weak reference days covered by selected windows must increase in both repeats; full quiet+instrument-step selection counts must not exceed original per repeat; total calls <72 each repeat. Stop this candidate before new calls if any fails",
        "agent_gate_if_probe_passes": "Both repeats complete process F1 and net corrections > original core and matched-K early pure-image controls; average weak recall not lower than original; full fixed comparison reported, exact old gate cannot be rescued. Author target must have nonnegative mean F1/net corrections versus original. Independent new-record validation only after these development conditions",
        "failure": "No score-dependent threshold, seed, prompt, best-repeat or subset substitution; report numerical control strength and abstention costs",
        "new_remote_requests": 0,
        "old_decisions_unchanged": ["NOT_CONFIRMED", "STOP_FEEDBACK_PROMOTION"],
    }
    write_json(out / "protocol.json", protocol)
    if calibration_source is None:
        process = calibration(
            BASE / "motion-probe-v2",
            DATA / "stage-xgboost.json",
            out / "primary_process",
            "primary_process",
        )
        author_source = BASE / "ai-observable-calibration-v3"
        calibration(
            author_source,
            author_source / "stage-xgboost.json",
            out / "secondary_developer",
            "secondary_developer",
        )
    else:
        verify_bindings(calibration_source, "calibration-freeze.json")
        for target in ("primary_process", "secondary_developer"):
            for filename, digest in read_json(
                calibration_source / target / "source-bindings.json"
            ).items():
                if sha(Path(filename)) != digest:
                    raise ValueError("Calibration input changed: " + filename)
            shutil.copytree(calibration_source / target, out / target)
        process = read_json(out / "primary_process/selection.json")
        write_json(
            out / "calibration-reuse.json",
            {
                "source": str(calibration_source),
                "freeze_sha256": sha(calibration_source / "calibration-freeze.json"),
                "reason": "Completed fitting preserved after NaN JSON export failure; no retuning or refitting",
            },
        )
    # Fix both target calibrations before opening the development reference.
    write_json(
        out / "calibration-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    prior = read_json(PRIOR / "primary_process/predictions.json")
    old = read_json(PARENT / "selections.json")
    labels = read_json(PRIOR / "primary_process/reference-labels.json")
    case_ids = read_json(DATA / "protocol.json")["case_ids"]
    threshold = process["axis_persistence_rule"]["threshold"]
    evidence, selections, query_plan, opportunity = {}, {}, [], []
    for cid in case_ids:
        case, feature = load_case_features(DATA, cid)
        witness = axis_persistence(case, feature.noise)
        evidence[cid] = {
            key: np.where(np.isfinite(value), value, None).tolist()
            for key, value in witness.items()
        }
        for rep in (0, 1):
            previous = old[f"{cid}-r{rep}"]
            windows = [(row["start"], row["stop"]) for row in previous["priorities"]]
            initial = np.array(prior[f"direct_visual_r{rep}"][cid]["activity"])
            priorities, selected, scope = persistence_review_plan(
                initial, feature.observed, witness, windows, threshold
            )
            selections[f"{cid}-r{rep}"] = {
                "priorities": priorities,
                "selected": selected,
                "scope": scope.tolist(),
                "fixed_indices": previous["fixed_indices"],
            }
            for target in selected:
                request_id = f"{cid}-window-{target['index']:02d}-image-r{rep}"
                query_plan.append(
                    {
                        "case_id": cid,
                        "repeat": rep,
                        **target,
                        "request_id": request_id,
                        "cached": request_id in receipts,
                    }
                )
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]]) & (actual == 1)
            old_covered = np.zeros(len(actual), dtype=bool)
            new_covered = np.zeros(len(actual), dtype=bool)
            for target in previous["selected"]:
                old_covered[target["start"] : target["stop"]] = True
            for target in selected:
                new_covered[target["start"] : target["stop"]] = True
            opportunity.append(
                {
                    "case_id": cid,
                    "repeat": rep,
                    "old_reviews": len(previous["selected"]),
                    "new_reviews": len(selected),
                    "old_weak_covered": int((weak & old_covered).sum()),
                    "new_weak_covered": int((weak & new_covered).sum()),
                    "quiet_or_step": not np.any(actual == 1),
                    "new_observation_days": int(
                        (new_covered & ~old_covered & feature.observed).sum()
                    ),
                }
            )
    write_json(out / "raw-evidence.json", evidence)
    write_json(out / "selections.json", selections)
    write_json(out / "query-plan.json", query_plan)
    write_csv(out / "observation-opportunities.csv", opportunity)
    conditions = {}
    for rep in (0, 1):
        population = [row for row in opportunity if row["repeat"] == rep]
        conditions[f"weak_coverage_r{rep}"] = sum(
            row["new_weak_covered"] for row in population
        ) > sum(row["old_weak_covered"] for row in population)
        conditions[f"negative_selection_r{rep}"] = sum(
            row["new_reviews"] for row in population if row["quiet_or_step"]
        ) <= sum(row["old_reviews"] for row in population if row["quiet_or_step"])
        conditions[f"below_fixed_calls_r{rep}"] = (
            24 + sum(row["new_reviews"] for row in population) < 72
        )
    write_json(
        out / "decision.json",
        {
            "conditions": conditions,
            "decision": "ALLOW_BOUNDED_AGENT_DEVELOPMENT"
            if all(conditions.values())
            else "STOP_AXIS_WITNESS_BEFORE_CALLS",
            "new_queries_if_allowed": sum(not row["cached"] for row in query_plan),
            "warning": "Selected-window weak coverage is observation opportunity, not recognition recall or VLM correction",
        },
    )
    paths = sources + [
        PRIOR / "manifest.json",
        PARENT / "selections.json",
        PARENT / "protocol.json",
    ]
    paths.extend(path for path in (PARENT / "requests").iterdir() if path.is_file())
    write_json(out / "source-bindings.json", {str(path): sha(path) for path in paths})
    write_json(
        out / "manifest.json",
        {
            "files": {
                str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()
            }
        },
    )
    print(read_json(out / "decision.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration-from", type=Path)
    args = parser.parse_args()
    prepare(args.out.resolve(), args.calibration_from)
