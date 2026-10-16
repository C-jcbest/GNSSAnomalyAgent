"""Freeze and replay a proposal acceptance probe using existing paired VLM answers."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from matplotlib.colors import ListedColormap
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_review_utility import training_labels, verify
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_missed_motion_repair import write_rows
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence, review_priorities
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_proposal_utility import (
    FEATURE_NAMES,
    apply_proposal,
    expected_proposal_utility,
    fit_proposal_model,
    proposal_features,
)
from gnss_sim.landslide_review_utility import correction_utility


def raw_prior_matrix(case, features):
    evidence = {scale: raw_motion_evidence(case, features, scale) for scale in (42, 84)}
    return np.column_stack((features.matrix, *[
        evidence[scale][name] for scale in (42, 84) for name in ("rate", "signal_to_noise", "direction")
    ])), evidence[42]


def prepare(pilot, out):
    verify(pilot)
    for file in ("calibration-freeze.json", "prediction-freeze.json", "selection-freeze.json"):
        for name, expected in read_json(pilot / file).items():
            if sha(pilot / name) != expected:
                raise ValueError("The completed calibration experiment changed")
    out.mkdir(parents=True, exist_ok=False)
    pilot_config = read_json(pilot / "protocol.json")
    data = Path(pilot_config["source"])
    split = read_json(data / "manifest.json")["split"]
    train_ids = pilot_config["split"]["train"]
    train_groups = {read_json(data / "mechanism-reference" / cid / "truth.json")["background_group"]
                    for cid in train_ids}
    prior_train = [cid for cid in split["train"] if read_json(
        data / "mechanism-reference" / cid / "truth.json")["background_group"] not in train_groups]
    save_json(out / "protocol.json", {
        "scope": "Already exposed development, six calibration and six validation records, one shared background each",
        "pilot": str(pilot.resolve()), "data": str(data.resolve()), "split": pilot_config["split"],
        "primary": "proposal_guard", "new_requests": 0, "new_records": 0,
        "hypothesis": "Choosing a promising view does not establish correctness of its actual revision. "
                      "Calibrate acceptance of the complete permitted VLM proposal, preserving initial on rejection",
        "features": FEATURE_NAMES, "ridge_alpha": 1, "threshold": 0,
        "training_unit": "All96 cached train local views; target net newly correct definite binary decisions "
                         "divided by changed evaluable days. Unknown is incorrect. Repeated/overlapping views not independent",
        "cross_fitted_prior": {"excluded_groups": sorted(train_groups), "train_ids": prior_train,
                              "parameters": "Same frozen raw84 XGBoost depth/120 trees/.05 rate; no tuning"},
        "inference": "Original raw conflicts select at most2 nonoverlapping180-day views; image-only answers unchanged. "
                     "Commit each complete original49-day permitted proposal iff predicted actual net utility>0",
        "ablations": "No guard; learned visual-only first3 features; uncalibrated numerical correctness gain; "
                     "same learned guard with fixed two views. No new observations or threshold searches",
        "continue_gate": "Both repeats mainF1>direct, weakR>=direct, eventF1>=direct, mainF1>=unguarded raw "
                         "and visual-only guard, at least2 records net definite correction>0. No arm substitution",
        "stage": "Unchanged commontrain24 stage head, gated by final activity, not native VLM ASD",
        "reference": "Known injection recovery only; no expert or residual-no-motion reference",
    })
    records = {cid: load_case_features(data, cid) for cid in prior_train + train_ids}
    labels = training_labels(data, prior_train)
    matrices, eligible = [], []
    for cid in prior_train:
        case, features = records[cid]
        matrix, _ = raw_prior_matrix(case, features)
        support = features.observed & (
            np.isfinite(features.progressive_rate)
            | np.isfinite(raw_motion_evidence(case, features, 42)["rate"])
            | np.isfinite(raw_motion_evidence(case, features, 84)["rate"])
        )
        matrices.append(matrix)
        eligible.extend((support & (labels[cid] >= 0)).tolist())
    depth = read_json(data / "long-selection.json")["xgboost_84"]["depth"]
    model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05,
                          n_jobs=2, random_state=20261007)
    matrix = np.concatenate(matrices)
    actual = np.concatenate([labels[cid] for cid in prior_train])
    model.fit(matrix[np.asarray(eligible)], actual[np.asarray(eligible)])
    model.save_model(out / "cross-fitted-prior.json")
    probabilities = {cid: model.predict_proba(raw_prior_matrix(*records[cid])[0])[:, 1].tolist() for cid in train_ids}
    probabilities.update({cid: read_json(pilot / "numeric-probabilities.json")[cid]
                          for cid in pilot_config["split"]["validation"]})
    save_json(out / "numeric-probabilities.json", probabilities)
    bindings = {}
    for folder in (ROOT / "scripts", ROOT / "src/gnss_sim"):
        for path in folder.glob("*.py"):
            target = out / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            bindings[str(path.resolve())] = sha(path)
    for name in ("initial-predictions.json", "local-predictions.json", "numeric-probabilities.json",
                 "analysis-v2/predictions.json", "analysis-v2/actual-costs.json"):
        bindings[str((pilot / name).resolve())] = sha(pilot / name)
    for name in ("reference-labels.csv", "stage-xgboost.json", "long-selection.json"):
        bindings[str((data / name).resolve())] = sha(data / name)
    save_json(out / "execution-bindings.json", bindings)
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})
    print({"prepared": str(out), "excluded_prior_training_backgrounds": sorted(train_groups),
           "prior_train_records": len(prior_train)}, flush=True)


def replay(out):
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Proposal probe freeze changed")
    for name, expected in read_json(out / "execution-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Proposal probe source or cache changed")
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Replay only a new prepared probe")
    config = read_json(out / "protocol.json")
    pilot, data = Path(config["pilot"]), Path(config["data"])
    train, validation = config["split"]["train"], config["split"]["validation"]
    records = {cid: load_case_features(pilot, cid) for cid in train + validation}
    initial_saved = read_json(pilot / "initial-predictions.json")
    local_saved = read_json(pilot / "local-predictions.json")
    probabilities = read_json(out / "numeric-probabilities.json")
    policy = ReviewPolicy()

    def candidate(cid, rep, index):
        case, features = records[cid]
        initial = np.asarray(initial_saved[f"{cid}-direct-r{rep}"]["activity"])
        reviewed = np.asarray(local_saved[f"{cid}-window-{index:02d}-image-r{rep}"]["activity"])
        start, stop = regular_windows(len(initial), 180, 90)[index]
        evidence = raw_motion_evidence(case, features)
        neighbourhood = conflict_neighbourhood(initial, features, evidence, policy)[1]
        scope = target_scopes(neighbourhood, features.observed, [{"start": start, "stop": stop}])[0]
        row, changed = proposal_features(initial, reviewed, scope, features, evidence,
                                        probabilities[cid], policy)
        return initial, reviewed, scope, row, changed

    actual_train = training_labels(data, train)
    rows, targets, training = [], [], []
    for rep in (0, 1):
        for cid in train:
            for index in range(len(regular_windows(len(records[cid][0].dates), 180, 90))):
                initial, reviewed, scope, row, changed = candidate(cid, rep, index)
                utility = correction_utility(actual_train[cid], initial, reviewed, changed)
                rows.append(row)
                targets.append(utility["utility_fraction"])
                training.append({"case_id": cid, "repeat": rep, "index": index,
                                 "features": row.tolist(), **utility})
    models = {"proposal_guard": fit_proposal_model(np.asarray(rows), targets, list(range(5))),
              "visual_proposal_guard": fit_proposal_model(np.asarray(rows), targets, [0, 1, 2])}
    save_json(out / "models.json", models)
    save_json(out / "training-utilities.json", training)
    save_json(out / "model-freeze.json", {name: sha(out / name) for name in ("models.json", "training-utilities.json")})
    head = XGBClassifier()
    head.load_model(data / "stage-xgboost.json")
    old = read_json(pilot / "analysis-v2/predictions.json")
    predictions = {name: values for name, values in old.items() if not name.startswith(
        ("visual_only_utility", "uncalibrated_prior", "same_count_fixed"))}
    proposal_log, permissions = [], {}
    for rep in (0, 1):
        for cid in validation:
            case, features = records[cid]
            initial = np.asarray(initial_saved[f"{cid}-direct-r{rep}"]["activity"])
            windows = regular_windows(len(initial), 180, 90)
            evidence = raw_motion_evidence(case, features)
            _, selected = review_priorities(initial, features, evidence, windows, policy)
            fixed = read_json(pilot / "selections.json")[f"{cid}-r{rep}"]["fixed_review"]["selected"]
            codes = head.predict(features.matrix).astype(int) + 1
            for stem in ("proposal_guard", "visual_proposal_guard", "numeric_vote_guard", "fixed_proposal_guard"):
                activity = initial.copy()
                authority = np.zeros(len(initial), dtype=bool)
                choices = fixed if stem == "fixed_proposal_guard" else selected
                for choice in choices:
                    index = choice["index"]
                    _, reviewed, scope, row, changed = candidate(cid, rep, index)
                    if stem == "numeric_vote_guard":
                        expected = float(row[4])
                    else:
                        selector = models["proposal_guard" if stem == "fixed_proposal_guard" else stem]
                        expected = expected_proposal_utility(selector, row)
                    accepted = expected > 0 and changed.any()
                    if accepted:
                        activity = apply_proposal(activity, reviewed, scope, expected)
                        authority |= scope
                    proposal_log.append({"method": f"{stem}_r{rep}", "case_id": cid, "index": index,
                                         "features": row.tolist(), "changed_days": int(changed.sum()),
                                         "expected_utility": expected, "accepted": bool(accepted)})
                if np.any(activity[~authority] != initial[~authority]) or np.any(activity[~features.observed] != -1):
                    raise ValueError("Proposal guard changed dates outside accepted visual authority")
                method = f"{stem}_r{rep}"
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, features.supported).to_dict()
                permissions[f"{method}-{cid}"] = np.flatnonzero(authority).tolist()
    save_json(out / "predictions.json", predictions)
    save_json(out / "proposals.json", proposal_log)
    save_json(out / "permissions.json", permissions)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in
                                              ("predictions.json", "proposals.json", "permissions.json")})
    # Validation references are loaded only after model and predicted approvals freeze.
    with (data / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        reference_rows = [row for row in csv.DictReader(handle) if row["case_id"] in validation]
    write_rows(out / "reference-subset.csv", reference_rows)
    labels = load_labels(out / "reference-subset.csv", {cid: records[cid][0] for cid in validation})
    results, summaries, changes = {}, [], {}
    with (pilot / "analysis-v2/summary.csv").open(encoding="utf-8", newline="") as handle:
        baseline = {row["method"]: row for row in csv.DictReader(handle)}
    for method, values in predictions.items():
        per_case, boundaries, corrected = {}, [], {}
        weak_total, weak_detected = 0, 0
        for cid, value in values.items():
            prediction = ActivityPrediction(np.asarray(value["activity"]), np.asarray(value["stage"]),
                                            status=value["status"], gated=value["gated"], stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            weak_total += int(weak.sum())
            weak_detected += int((weak & (prediction.activity == 1)).sum())
            boundaries.extend(boundary_errors(labels[cid], prediction.activity, .3)["matched_pairs"])
            if method.startswith(("proposal_guard", "visual_proposal_guard", "numeric_vote_guard", "fixed_proposal_guard")):
                actual = np.array([int(row["activity_label"]) for row in labels[cid]])
                initial = np.asarray(initial_saved[f"{cid}-direct-r{method[-1]}"]["activity"])
                corrected[cid] = correction_utility(actual, initial, prediction.activity, actual >= 0)
        aggregate = aggregate_scores(list(per_case.values()))
        results[method] = {"aggregate": aggregate, "per_case": per_case}
        changes[method] = corrected
        cost_method = f"fixed_review_r{method[-1]}" if method.startswith("fixed_proposal") else f"raw_conflict_r{method[-1]}"
        cost_row = baseline.get(method, baseline.get(cost_method, {}))
        summaries.append({"method": method, **aggregate["daily"], "coverage": aggregate["coverage"],
                          "weak_recall": weak_detected / weak_total, "weak_days": weak_total,
                          "event_f1_iou03": aggregate["events"]["f1"],
                          "onset_mae_days": float(np.mean([abs(row["onset_error_days"]) for row in boundaries])) if boundaries else None,
                          "offset_mae_days": float(np.mean([abs(row["offset_error_days"]) for row in boundaries])) if boundaries else None,
                          "stage_macro_f1": aggregate["stage_macro_f1"],
                          "requests": int(cost_row.get("requests", 0)),
                          "reported_tokens": int(cost_row.get("reported_tokens", 0))})
    stages = verify_metrics(predictions, results, labels, {cid: records[cid] for cid in validation})
    for row in summaries:
        row["internal_stage_macro_f1"] = stages[row["method"]]["macro_f1"]
    index = {row["method"]: row for row in summaries}
    decisions = []
    for rep in (0, 1):
        main, direct, raw, visual = [f"{stem}_r{rep}" for stem in
                                     ("proposal_guard", "direct_visual", "raw_conflict", "visual_proposal_guard")]
        a, d = index[main], index[direct]
        criteria = {"daily_gain": a["f1"] > d["f1"], "weak_nonloss": a["weak_recall"] >= d["weak_recall"],
                    "event_nonloss": a["event_f1_iou03"] >= d["event_f1_iou03"],
                    "raw_nonloss": a["f1"] >= index[raw]["f1"],
                    "visual_nonloss": a["f1"] >= index[visual]["f1"],
                    "multiple_actual_gains": sum(row["net_correct"] > 0 for row in changes[main].values()) >= 2}
        decisions.append({"repeat": rep, "criteria": criteria, "passed": all(criteria.values())})
    save_json(out / "decision.json", {"allows_full_prototype": all(row["passed"] for row in decisions), "repeats": decisions})
    for name, value in (("results.json", results), ("corrected-decisions.json", changes), ("internal-stages.json", stages)):
        save_json(out / name, value)
    write_rows(out / "summary.csv", summaries)
    for cid in validation:
        case = records[cid][0]
        raw = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        methods = [f"{stem}_r{rep}" for rep in (0, 1) for stem in
                   ("direct_visual", "proposal_guard", "visual_proposal_guard", "raw_conflict", "fixed_proposal_guard")]
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 3]})
        for axis in range(3):
            axes[axis].plot(raw[:, axis], color="#24887b", linewidth=.8)
            axes[axis].set_ylabel(f"{'NEU'[axis]} (mm)")
        matrix = [[int(row["activity_label"]) for row in labels[cid]]]
        matrix.extend(predictions[name][cid]["activity"] for name in methods)
        axes[3].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
                       cmap=ListedColormap(["#c5c9cc", "#fff4dc", "#377dab"]),
                       extent=[-.5, len(raw)-.5, len(matrix)-.5, -.5])
        axes[3].set_yticks(range(len(matrix)), ["Known injection"] + methods, fontsize=7)
        axes[3].set_xlabel("Original day | blue activity; cream quiet; gray unknown/missing")
        fig.suptitle(f"{cid} | Proposal-utility development replay | Not independent field truth")
        fig.tight_layout()
        fig.savefig(out / f"{cid}.png", dpi=120)
        plt.close(fig)
    save_json(out / "actual-costs.json", {"new_started": 0, "new_receipts": 0, "new_reported_tokens": 0,
                                         "new_records": 0, "cached_pilot_training_and_collection_requests": 216,
                                         "flow_costs": "All selected reviews charged, even rejected proposals; cross-fit CPU additional"})
    save_json(out / "state.json", {"status": "complete", "configs": len(results), "training_proposals": len(training),
                                  "prior_train_records": len(config["cross_fitted_prior"]["train_ids"]),
                                  "validation_proposals": len(proposal_log), "full_record_figures": len(validation)})
    print({"decision": read_json(out / "decision.json"), "main": [index[f"proposal_guard_r{rep}"] for rep in (0, 1)]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "replay"))
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.pilot is None:
            parser.error("prepare requires --pilot")
        prepare(args.pilot, args.out)
    else:
        replay(args.out)
