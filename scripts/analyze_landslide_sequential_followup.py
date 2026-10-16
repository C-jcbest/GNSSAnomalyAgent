"""Score a generic fresh sequential run against predeclared dual references."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from analyze_landslide_codex_observation_reference import scoring_inputs
from analyze_landslide_dual_reference_confirmation import score_target
from analyze_landslide_review_utility_v2 import verify_receipts
from fit_landslide_raw_persistence_baselines import raw_gate
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from run_landslide_ai_reference_holdout import replay, verify_bindings
from run_landslide_comparison import load_labels, predict_baselines
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility


def predict_process(data, training, out):
    """Freeze all target-matched predictions without reading new process labels."""
    verify_bindings(data, "data-freeze.json")
    verify_bindings(training, "training-freeze.json")
    out.mkdir(exist_ok=False)
    models = {}
    for name, filename in (("activity", "primary-baselines/activity-xgboost.json"),
                           ("stage", "stage-xgboost.json"),
                           ("persistence", "raw-baselines/activity-xgboost-raw-persistence.json")):
        models[name] = XGBClassifier()
        models[name].load_model(training / filename)
    ids = read_json(data / "protocol.json")["case_ids"]
    inputs = {cid: load_case_features(data, cid) for cid in ids}
    features = {cid: value[1] for cid, value in inputs.items()}
    predictions = predict_baselines(features, ids,
        read_json(training / "primary-baselines/frozen-selection.json"), models["activity"], models["stage"])
    predictions.pop("ungated_rule")
    choices = read_json(training / "raw-baselines/frozen-selection.json")
    predictions.update({name: {} for name in choices})
    codes = {}
    for cid, (case, feature) in inputs.items():
        long = raw_motion_evidence(case, feature)
        matrix = np.column_stack((feature.matrix, long["rate"], long["signal_to_noise"], long["direction"]))
        rule = choices["raw_multiscale_rule"]
        gates = {
            "raw_multiscale_rule": raw_gate(feature, long, rule["rate"], rule["snr"]),
            "xgboost_raw_persistence": activity_from_mask(
                models["persistence"].predict_proba(matrix)[:, 1] >= choices["xgboost_raw_persistence"]["threshold"],
                feature.supported),
        }
        codes[cid] = models["stage"].predict(feature.matrix).astype(int) + 1
        for method, gate in gates.items():
            predictions[method][cid] = learned_stages(codes[cid], gate, feature.supported)
    write_json(out / "numeric-predictions.json", {method: {cid: value.to_dict() for cid, value in records.items()}
        for method, records in predictions.items()})
    write_json(out / "stage-codes.json", {cid: value.tolist() for cid, value in codes.items()})
    write_json(out / "provenance.json", {"specific_process_labels_read": False,
        "training_freeze_sha256": sha(training / "training-freeze.json"),
        "data_freeze_sha256": sha(data / "data-freeze.json")})
    write_json(out / "prediction-freeze.json", {path.name: sha(path)
        for path in out.iterdir() if path.is_file()})


def confirmation_decision(labels, observed, predictions, summaries):
    comparisons = []
    for stem in ("adaptive_no_feedback", "followup_fixed_short", "first_only"):
        differences, weak_differences, net = [], [], 0
        for rep in (0, 1):
            main = f"followup_censored_r{rep}"
            control = f"{stem}_r{rep}"
            if stem == "adaptive_no_feedback":
                control += "__conflict_scope"
            differences.append(summaries[main]["f1"] - summaries[control]["f1"])
            weak_differences.append(summaries[main]["weak_recall"] - summaries[control]["weak_recall"])
            for cid, rows in labels.items():
                actual = np.array([row["activity_label"] for row in rows])
                net += correction_utility(actual, np.array(predictions[control][cid]["activity"]),
                    np.array(predictions[main][cid]["activity"]), observed[cid])["net_correct"]
        passed = np.mean(differences) > 0 and net > 0
        if stem == "adaptive_no_feedback":
            passed = passed and np.mean(weak_differences) >= 0
        comparisons.append({"control": stem, "f1_difference_by_repeat": differences,
            "mean_f1_difference": float(np.mean(differences)),
            "mean_weak_recall_difference": float(np.mean(weak_differences)),
            "net_correct_both_repeats": net, "passed": bool(passed)})
    return {"passed": all(row["passed"] for row in comparisons), "comparisons": comparisons,
            "default_changed": False}


def analyze(comparison, process_predictions, out):
    run = comparison / "run"
    data = Path(read_json(comparison / "source-bindings.json")["data"])
    if sha(Path(__file__)) != read_json(data / "protocol.json")["scorer_sha256"]:
        raise ValueError("Scorer differs from the pre-generation registration")
    verify_bindings(run, "prediction-freeze.json")
    verify_bindings(process_predictions, "prediction-freeze.json")
    verify_manifest(data / "codex-review", "delivery-manifest.json")
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Finish and freeze all visual predictions before reading process labels")
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "before-process-label-read.json", {
        "opened_at": datetime.now(timezone.utc).isoformat(), "all_predictions_frozen": True,
        "visual_freeze_sha256": sha(run / "prediction-freeze.json"),
        "process_prediction_freeze_sha256": sha(process_predictions / "prediction-freeze.json")})
    ai, observed, _ = scoring_inputs(data / "codex-review")
    authored = read_json(data / "codex-review/activity-decisions.json")["weak_interiors_authored_before_predictions"]
    weak_ai = {}
    for cid, rows in ai.items():
        mask = np.zeros(len(rows), bool)
        for start, stop in authored.get(cid, []):
            mask[start:stop] = True
        weak_ai[cid] = mask & (np.array([row["activity_label"] for row in rows]) == 1)
    cases = {cid: load_case_features(data, cid)[0] for cid in ai}
    loaded = load_labels(data / "mechanism-sealed/reference-labels.csv", cases)
    process = {cid: [dict(row, activity_label=int(row["activity_label"]), weak_signal=int(row["weak_signal"]))
        for row in rows] for cid, rows in loaded.items()}
    weak_process = {cid: np.array([row["weak_signal"] == 1 and row["activity_label"] == 1 for row in rows])
        for cid, rows in process.items()}
    receipts = verify_receipts(run)
    predictions, used, scopes = replay(run, receipts)
    predictions.update(read_json(run / "followup-predictions.json"))
    used.update(read_json(run / "followup-used-requests.json"))
    process_numeric = read_json(process_predictions / "numeric-predictions.json")
    codes = read_json(process_predictions / "stage-codes.json")
    features = {cid: load_case_features(data, cid)[1] for cid in ai}
    process_methods = dict(process_numeric)
    for method, records in predictions.items():
        if method in process_numeric:
            continue
        process_methods[method] = {cid: learned_stages(np.array(codes[cid]), np.array(value["activity"]),
            features[cid].supported).to_dict() for cid, value in records.items()}
    decisions = {}
    for target, labels, weak, methods in (("primary_ai", ai, weak_ai, predictions),
                                          ("secondary_process", process, weak_process, process_methods)):
        folder = out / target
        folder.mkdir()
        results, summaries, records = score_target(labels, observed, weak, methods, used, receipts)
        write_json(folder / "predictions.json", methods)
        write_json(folder / "reference-labels.json", labels)
        write_json(folder / "results.json", results)
        write_csv(folder / "summary.csv", summaries)
        write_csv(folder / "per-record.csv", records)
        decision = confirmation_decision(labels, observed, methods, {row["method"]: row for row in summaries})
        write_json(folder / "decision.json", decision)
        decisions[target] = decision
    write_json(out / "used-requests.json", used)
    write_json(out / "original-scopes.json", scopes)
    write_json(out / "decision.json", {"primary": decisions["primary_ai"], "secondary": decisions["secondary_process"],
        "secondary_does_not_override_primary": True, "default_changed": False})
    write_json(out / "manifest.json", {"files": {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()}})
    print({target: value["passed"] for target, value in decisions.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("predict-process", "analyze"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--process-predictions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "predict-process":
        predict_process(args.data.resolve(), args.training.resolve(), args.out.resolve())
    else:
        analyze(args.comparison.resolve(), args.process_predictions.resolve(), args.out.resolve())
