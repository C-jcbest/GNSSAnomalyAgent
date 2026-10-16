"""Replay the registered core trial on full process and fixed developer targets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import paired_record_interval
from analyze_landslide_codex_observation_reference import scoring_inputs
from analyze_landslide_review_utility_v2 import verify_receipts
from fit_landslide_raw_persistence_baselines import raw_gate
from prepare_landslide_ai_reference_holdout import write_csv
from run_landslide_adaptive_review import verify_freeze
from run_landslide_ai_reference_holdout import replay, verify_bindings
from run_landslide_comparison import load_labels, predict_baselines
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_core_lockbox import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
AI_TRAINING = BASE / "ai-observable-calibration-v3"


def developer_predictions(data, features, visual):
    """Use the previously frozen observable-target training/validation models."""
    verify_bindings(AI_TRAINING, "training-freeze.json")
    models = {}
    for name, filename in {
        "activity": "primary-baselines/activity-xgboost.json",
        "stage": "stage-xgboost.json",
        "persistence": "raw-baselines/activity-xgboost-raw-persistence.json",
    }.items():
        models[name] = XGBClassifier()
        models[name].load_model(AI_TRAINING / filename)
    numerical = predict_baselines(
        {cid: pair[1] for cid, pair in features.items()},
        list(features),
        read_json(AI_TRAINING / "primary-baselines/frozen-selection.json"),
        models["activity"],
        models["stage"],
    )
    numerical.pop("ungated_rule")
    choices = read_json(AI_TRAINING / "raw-baselines/frozen-selection.json")
    numerical.update({name: {} for name in choices})
    predictions = {}
    for cid, (case, feature) in features.items():
        long = raw_motion_evidence(case, feature)
        matrix = np.column_stack(
            (feature.matrix, long["rate"], long["signal_to_noise"], long["direction"])
        )
        rule = choices["raw_multiscale_rule"]
        gates = {
            "raw_multiscale_rule": raw_gate(feature, long, rule["rate"], rule["snr"]),
            "xgboost_raw_persistence": activity_from_mask(
                models["persistence"].predict_proba(matrix)[:, 1]
                >= choices["xgboost_raw_persistence"]["threshold"],
                feature.supported,
            ),
        }
        codes = models["stage"].predict(feature.matrix).astype(int) + 1
        for name, gate in gates.items():
            numerical[name][cid] = learned_stages(codes, gate, feature.supported)
        for method, records in visual.items():
            value = records[cid]
            prediction = learned_stages(codes, np.array(value["activity"]), feature.supported)
            prediction.status = value["status"]
            if prediction.status == "failed":
                prediction.stage[:] = -1
            if prediction.activity.tolist() != value["activity"]:
                raise ValueError("Target-specific stage head changed visual activity")
            predictions.setdefault(method, {})[cid] = prediction.to_dict()
    predictions.update(
        {
            name: {cid: value.to_dict() for cid, value in cases.items()}
            for name, cases in numerical.items()
        }
    )
    return predictions


def compare(labels, observed, predictions, results, summaries):
    table = {row["method"]: row for row in summaries}
    rows, intervals = [], []
    for rep in (0, 1):
        main = f"adaptive_no_feedback_r{rep}__conflict_scope"
        controls = [
            name
            for name in predictions
            if name != main and (f"_r{rep}" in name or "_r" not in name)
        ]
        for control in controls:
            records = []
            for cid, labels_case in labels.items():
                counts = correction_utility(
                    np.array([int(row["activity_label"]) for row in labels_case]),
                    np.array(predictions[control][cid]["activity"]),
                    np.array(predictions[main][cid]["activity"]),
                    observed[cid],
                )
                records.append(counts)
            rows.append(
                {
                    "control": control,
                    "treatment": main,
                    **{
                        key: sum(row[key] for row in records)
                        for key in (
                            "eligible_days",
                            "gained_correct",
                            "lost_correct",
                            "net_correct",
                        )
                    },
                    "benefited_records": sum(row["net_correct"] > 0 for row in records),
                    "f1_difference": table[main]["f1"] - table[control]["f1"],
                }
            )
            intervals.append(
                {
                    "control": control,
                    "treatment": main,
                    **paired_record_interval(results, control, main),
                }
            )
    return rows, intervals


def analyze(data, run, out):
    verify_bindings(data, "data-freeze.json")
    verify_freeze(run)
    verify_bindings(run, "prediction-freeze.json")
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Complete and freeze all responses before scoring")
    reference = data / "codex-review/compiled/reference-freeze.json"
    binding = read_json(run / "developer-reference-binding.json")
    if sha(reference) != binding["sha256"]:
        raise ValueError("Developer reference changed since inference preparation")
    for filename, expected in read_json(reference)["files"].items():
        if sha(reference.parent / filename) != expected:
            raise ValueError("Developer reference file changed")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    receipts = verify_receipts(run)
    predictions, used, scopes = replay(run, receipts)
    write_json(out / "predictions.json", predictions)
    write_json(out / "used-requests.json", used)
    write_json(out / "scopes.json", scopes)
    protocol = read_json(data / "protocol.json")
    features = {cid: load_case_features(data, cid) for cid in protocol["case_ids"]}
    loaded = load_labels(
        data / "mechanism-sealed/reference-labels.csv",
        {cid: pair[0] for cid, pair in features.items()},
    )
    primary = {
        cid: [
            dict(
                row, activity_label=int(row["activity_label"]), weak_signal=int(row["weak_signal"])
            )
            for row in rows
        ]
        for cid, rows in loaded.items()
    }
    observed = {cid: pair[1].observed for cid, pair in features.items()}
    weak = {
        cid: np.array([row["activity_label"] == 1 and row["weak_signal"] == 1 for row in rows])
        for cid, rows in primary.items()
    }
    ai, ai_observed, _ = scoring_inputs(data / "codex-review")
    spans = read_json(data / "codex-review/activity-decisions.json")[
        "weak_interiors_authored_before_predictions"
    ]
    ai_weak = {cid: np.zeros(len(rows), dtype=bool) for cid, rows in ai.items()}
    for cid, rows in ai.items():
        for start, stop in spans.get(cid, []):
            ai_weak[cid][start:stop] = True
        ai_weak[cid] &= np.array([row["activity_label"] for row in rows]) == 1
    visual = {name: values for name, values in predictions.items() if name in used}
    ai_predictions = developer_predictions(data, {cid: features[cid] for cid in ai}, visual)
    gate = {}
    for target, labels, masks, weak_masks, values in (
        ("primary_process", primary, observed, weak, predictions),
        ("secondary_developer", ai, ai_observed, ai_weak, ai_predictions),
    ):
        folder = out / target
        folder.mkdir()
        target_requests = {
            method: {cid: keys for cid, keys in cases.items() if cid in labels}
            for method, cases in used.items()
        }
        results, summaries, records = score_target(
            labels, masks, weak_masks, values, target_requests, receipts
        )
        corrections, intervals = compare(labels, masks, values, results, summaries)
        for name, value in (
            ("results.json", results),
            ("predictions.json", values),
            ("reference-labels.json", labels),
            ("bootstrap.json", intervals),
        ):
            write_json(folder / name, value)
        for name, rows in (
            ("summary.csv", summaries),
            ("per-record.csv", records),
            ("corrections.csv", corrections),
        ):
            write_csv(folder / name, rows)
        if target == "primary_process":
            table = {row["method"]: row for row in summaries}
            conditions = {}
            for rep in (0, 1):
                main = f"adaptive_no_feedback_r{rep}__conflict_scope"
                for control in (
                    f"direct_visual_r{rep}",
                    f"fixed_image_review_r{rep}__whole_target",
                    f"same_count_fixed_early_r{rep}__conflict_scope",
                ):
                    row = next(
                        row
                        for row in corrections
                        if row["control"] == control and row["treatment"] == main
                    )
                    conditions[f"r{rep}:{control}:F1_and_net"] = (
                        row["f1_difference"] > 0 and row["net_correct"] > 0
                    )
                    if control.startswith("direct"):
                        conditions[f"r{rep}:two_benefited_records"] = row["benefited_records"] >= 2
                conditions[f"r{rep}:fewer_requests"] = (
                    table[main]["requests"]
                    < table[f"fixed_image_review_r{rep}__whole_target"]["requests"]
                )
            conditions["mean_weak_not_lower"] = bool(
                np.mean(
                    [
                        table[f"adaptive_no_feedback_r{rep}__conflict_scope"]["weak_recall"]
                        for rep in (0, 1)
                    ]
                )
                >= np.mean(
                    [
                        table[f"same_count_fixed_early_r{rep}__conflict_scope"]["weak_recall"]
                        for rep in (0, 1)
                    ]
                )
            )
            gate = {
                "conditions": conditions,
                "passed": all(conditions.values()),
                "decision": "CONFIRMED" if all(conditions.values()) else "NOT_CONFIRMED",
                "scope": "Retained core, one registered full24 process-target trial only",
            }
    write_json(out / "decision.json", gate)
    write_json(
        out / "audit.json",
        {
            "full_primary_records": 24,
            "developer_records": 12,
            "primary_methods": len(predictions),
            "secondary_methods": len(ai_predictions),
            "new_remote_requests": 0,
            "target_matched_numerical_models": True,
            "formal_field_accuracy_verified": False,
            "independent_reviewer": False,
        },
    )
    write_json(
        out / "manifest.json",
        {
            "files": {
                path.relative_to(out).as_posix(): sha(path)
                for path in sorted(out.rglob("*"))
                if path.is_file()
            }
        },
    )
    print(gate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.data.resolve(), args.run.resolve(), args.out.resolve())
