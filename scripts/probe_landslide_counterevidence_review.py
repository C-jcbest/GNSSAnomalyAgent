"""Freeze an observation-only edit-veto replay before opening scoring references."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from analyze_landslide_codex_observation_reference import scoring_inputs, stage_reference_arrays
from probe_landslide_conditional_numeric_review import BASE, BATCHES, check_bindings, write_csv
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import ROOT, load_labels
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_counterevidence_review import (
    raw_counterevidence,
    restrict_contradicted_edits,
)
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility

HOLDOUT = BASE / "ai-observable-holdout-comparison-v1"
CACHE = BASE / "conditional-numeric-followup-v2"
ARMS = ("counterevidence", "plateau_only", "motion_only", "numeric_vote", "matched_uniform", "matched_center")
PROTOCOL = {
    "schema": "counterevidence-edit-veto-development-v1",
    "scope": "54 old records and12 previously exposed new records; two AI references separate; development only",
    "candidate": "Reject only definite0->1 edits on dual-scale plateaus and definite1->0 edits on dual-scale sustained motion; retain original decisions, never create labels from numbers",
    "evidence": "21/42-day separation, three nonoverlapping14-day raw medians,8 complete observations/block, no gap>7 in envelope; plateau max normalized change<=1 at BOTH scales; motion min normalized change>=2, minimum rate>=.03mm/day, positive successive dot product at BOTH scales; otherwise inconclusive",
    "unchanged": "Original selection, disjoint targets,49-day authority, common population-specific stage head, observed and supported masks, cached actual image answers; unknown edits never vetoed",
    "vote": "For definite changed dates only, retain original if it agrees with existing short OR long raw motion state; at least one scale supported; no supported motion means numeric0. This is the two-out-of-three vote of initial,local,numeric",
    "matched_controls": "Same veto count within EACH selected target, choose definite changed dates uniformly or closest to target center; no reference or correctness selection",
    "controls": "direct,fixed,pure,numeric-first; plateau-only,motion-only,vote,two count-matched vetoes; original same-record strong numerical controls; all outcomes retained",
    "cost": "Zero new remote calls; deploy original pure receipts plus measured raw counterevidence time; no latency or energy saving claim",
    "continue_conditions": {
        "noninferiority": "Both repeats of mechanism,confirmation,background and both AI reference populations: F1 and weakR nondecreasing vs pure",
        "increment": "Each AI reference population: both repeats F1>=numeric_vote and at least one strict; net definite correction vs pure positive in at least one repeat",
        "stop": "Any failure stops this exact version; no threshold/seed/repeat sweep and no default change",
    },
    "independent_confirmation": False,
    "known_limit": "Even-block midpoint at a single step can produce isolated motion proxy support; this candidate only retains old labels and never establishes activity from it",
}


def bind(bindings, path):
    bindings[str(path.relative_to(ROOT))] = sha(path)


def matched_veto(eligible, rejected, targets, centered):
    """Retain the same number of old decisions per target without evidence placement."""
    selected = np.zeros(len(eligible), dtype=bool)
    for target in targets:
        start, stop = target["start"], target["stop"]
        count = int(rejected[start:stop].sum())
        dates = np.flatnonzero(eligible[start:stop]) + start
        if count > len(dates):
            raise ValueError("Matched veto exceeds eligible edits")
        if count == 0:
            continue
        if centered:
            order = np.lexsort((dates, np.abs(dates - (start + stop - 1) / 2)))
            chosen = dates[order[:count]]
        else:
            positions = np.floor((np.arange(count) + .5) * len(dates) / count).astype(int)
            chosen = dates[positions]
        selected[chosen] = True
        if selected[start:stop].sum() != count:
            raise ValueError("Matched veto budget differs")
    return selected


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", PROTOCOL)
    shutil.copyfile(Path(__file__), out / "source.py")
    module = ROOT / "src/gnss_sim/landslide_counterevidence_review.py"
    shutil.copyfile(module, out / module.name)
    check_bindings(CACHE, "prediction-freeze.json")
    cached = read_json(CACHE / "predictions.json")
    costs = read_json(CACHE / "deployment-costs.json")
    new_cached = read_json(HOLDOUT / "analysis/predictions.json")
    new_usage = read_json(HOLDOUT / "analysis/used-requests.json")
    predictions, vetoes, evidence_rows, timings, bindings = {}, {}, {}, [], {}
    for path in (CACHE / "predictions.json", CACHE / "deployment-costs.json",
                 HOLDOUT / "analysis/predictions.json", HOLDOUT / "analysis/used-requests.json", module):
        bind(bindings, path)
    for batch in (*BATCHES, "new-AI"):
        source = HOLDOUT / "run" if batch == "new-AI" else BASE / batch
        check_bindings(source, "freeze.json")
        check_bindings(source, "prediction-freeze.json")
        config = read_json(source / "protocol.json")
        choices = read_json(source / "selections.json")
        policy = ReviewPolicy(**config["policy"])
        head = XGBClassifier()
        head.load_model(source / "stage-xgboost.json")
        for name in ("freeze.json", "prediction-freeze.json", "protocol.json", "selections.json", "stage-xgboost.json"):
            bind(bindings, source / name)
        if batch == "new-AI":
            controls = read_json(source / "numeric-predictions.json")
            for repeat in (0, 1):
                for short, original in (("direct", f"direct_visual_r{repeat}"),
                                        ("fixed", f"fixed_image_review_r{repeat}__conflict_scope"),
                                        ("pure", f"adaptive_no_feedback_r{repeat}__conflict_scope"),
                                        ("numeric_first", f"adaptive_numeric_feedback_r{repeat}__conflict_scope")):
                    controls[f"{short}_r{repeat}"] = new_cached[original]
        else:
            controls = {name: records for name, records in cached[batch].items()
                        if not name.startswith(("conditional_", "matched_early_", "always_followup_"))}
        predictions[batch] = controls
        vetoes[batch], evidence_rows[batch] = {}, {}
        for cid in config["case_ids"]:
            case, feature = load_case_features(source, cid)
            for path in sorted((source / "inputs" / cid).glob("*.json")):
                bind(bindings, path)
            started = time.perf_counter()
            evidence = raw_counterevidence(case, feature.noise)
            elapsed = time.perf_counter() - started
            timings.append({"batch": batch, "case_id": cid, "counterevidence_seconds": elapsed})
            evidence_rows[batch][cid] = {key: np.flatnonzero(evidence[key]).tolist()
                                       for key in ("plateau", "motion", "supported")}
            long = raw_motion_evidence(case, feature)
            supported = feature.observed & (np.isfinite(feature.progressive_rate) | np.isfinite(long["rate"]))
            raw_motion = conflict_neighbourhood(np.where(feature.observed, 0, -1), feature, long, policy, margin=0)[0]
            numeric_state = raw_motion.astype(int)
            codes = head.predict(feature.matrix).astype(int) + 1
            for repeat in (0, 1):
                initial = np.array(controls[f"direct_r{repeat}"][cid]["activity"])
                pure = np.array(controls[f"pure_r{repeat}"][cid]["activity"])
                choice = choices[f"{cid}-r{repeat}"]
                targets = choice["selected"]
                neighbourhood = conflict_neighbourhood(initial, feature, long, policy)[1]
                permission = target_scopes(neighbourhood, feature.observed, targets)[0]
                eligible = permission & (initial >= 0) & (pure >= 0) & (initial != pure)
                activities, masks = {}, {}
                for arm, plateau, motion in (("counterevidence", True, True), ("plateau_only", True, False),
                                             ("motion_only", False, True)):
                    activities[arm], masks[arm] = restrict_contradicted_edits(
                        initial, pure, permission, feature.observed, evidence, plateau, motion)
                masks["numeric_vote"] = eligible & supported & (initial == numeric_state)
                for arm, centered in (("matched_uniform", False), ("matched_center", True)):
                    masks[arm] = matched_veto(eligible, masks["counterevidence"], targets, centered)
                for arm in ("numeric_vote", "matched_uniform", "matched_center"):
                    activities[arm] = pure.copy()
                    activities[arm][masks[arm]] = initial[masks[arm]]
                if batch == "new-AI":
                    request_ids = new_usage[f"adaptive_no_feedback_r{repeat}__conflict_scope"][cid]
                    receipts = [read_json(source / "requests" / f"{key}.response.json") for key in request_ids]
                    usage = {"requests": len(request_ids), "tokens": sum(row["usage"]["total_tokens"] for row in receipts)}
                else:
                    usage = next(row for row in costs[batch][f"pure_r{repeat}"] if row["case_id"] == cid)
                for arm in ARMS:
                    activity = activities[arm]
                    if np.any(activity[~permission] != initial[~permission]):
                        raise ValueError("Candidate changed a date outside original authority")
                    if np.any(activity[masks[arm]] != initial[masks[arm]]) or np.any(masks[arm] & ~eligible):
                        raise ValueError("Veto must retain only old definite changed labels")
                    method = f"{arm}_r{repeat}"
                    predictions[batch].setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                    vetoes[batch].setdefault(method, {})[cid] = {
                        "rejected_days": np.flatnonzero(masks[arm]).tolist(),
                        "eligible_days": np.flatnonzero(eligible).tolist(), "requests": usage["requests"],
                        "reported_tokens": usage["tokens"], "targets": targets,
                    }
    write_json(out / "source-bindings.json", bindings)
    write_json(out / "predictions.json", predictions)
    write_json(out / "vetoes.json", vetoes)
    write_json(out / "evidence.json", evidence_rows)
    write_csv(out / "timings.csv", timings)
    write_json(out / "prediction-freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print("Frozen66-record candidate and controls; scoring references not opened", flush=True)


def score(out):
    check_bindings(out, "prediction-freeze.json")
    for name, expected in read_json(out / "source-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Bound evidence changed: " + name)
    predictions = read_json(out / "predictions.json")
    populations, reference_bindings = {}, {}
    for batch in BATCHES:
        source = BASE / batch
        cases = {cid: load_case_features(source, cid)[0] for cid in predictions[batch]["direct_r0"]}
        reference = source / ("analysis-v1/reference-subset.csv" if batch == "background-review-v1" else "reference-subset.csv")
        labels = load_labels(reference, cases)
        weak = {cid: np.array([row["feature"] == "slow_displacement" if batch == "pilot-v3"
                              else int(row["weak_signal"]) == 1 for row in rows]) for cid, rows in labels.items()}
        populations[batch] = (batch, labels, weak)
        bind(reference_bindings, reference)
    for name, reference, batch in (("old-AI", BASE / "codex-observation-review-v1", "confirmation-v1"),
                                   ("new-AI", BASE / "ai-observable-holdout-data-v1/codex-review", "new-AI")):
        labels, _, weak = scoring_inputs(reference)
        if name == "new-AI":
            decisions = read_json(reference / "activity-decisions.json")
            weak = {cid: np.zeros(len(rows), dtype=bool) for cid, rows in labels.items()}
            for cid, spans in decisions["weak_interiors_authored_before_predictions"].items():
                for start, stop in spans:
                    weak[cid][start:stop] = True
            for cid, rows in labels.items():
                weak[cid] &= np.array([int(row["activity_label"]) == 1 for row in rows])
        populations[name] = (batch, labels, weak)
        for path in (reference / "compiled/daily-reference.json", reference / "activity-decisions.json"):
            bind(reference_bindings, path)
    results, summary, corrections, per_records = {}, [], [], []
    for population, (batch, labels, weak) in populations.items():
        results[population] = {}
        for method, records in predictions[batch].items():
            per_case, boundary, actual_stage, predicted_stage = {}, {}, [], []
            weak_tp = 0
            for cid, rows in labels.items():
                serialized = records[cid]
                prediction = ActivityPrediction(**{key: np.array(value) if key in {"activity", "stage"} else value
                                                   for key, value in serialized.items()})
                per_case[cid] = evaluate_case(rows, prediction)
                boundary[cid] = boundary_errors(rows, prediction.activity, .3)
                stage = stage_reference_arrays(rows, prediction)
                actual_stage.extend(stage["actual_stage"][stage["stage_interior"]].tolist())
                predicted_stage.extend(prediction.stage[stage["stage_interior"]].tolist())
                weak_tp += int(np.count_nonzero(weak[cid] & (prediction.activity == 1)))
                per_records.append({"population": population, "method": method, "case_id": cid,
                                    **per_case[cid]["daily"]})
            aggregate = aggregate_scores(list(per_case.values()))
            pairs = [pair for row in boundary.values() for pair in row["matched_pairs"]]
            internal_f1 = precision_recall_fscore_support(actual_stage, predicted_stage, labels=[1, 2, 3], zero_division=0)[2]
            event05 = aggregate_scores([evaluate_case(labels[cid], ActivityPrediction(**{
                key: np.array(value) if key in {"activity", "stage"} else value
                for key, value in records[cid].items()}), .5) for cid in labels])["events"]["f1"]
            row = {"population": population, "method": method, **aggregate["daily"],
                   "weak_recall": weak_tp / sum(int(mask.sum()) for mask in weak.values()),
                   "event_f1_iou03": aggregate["events"]["f1"], "event_f1_iou05": event05,
                   "onset_mae": float(np.mean([abs(pair["onset_error_days"]) for pair in pairs])) if pairs else None,
                   "offset_mae": float(np.mean([abs(pair["offset_error_days"]) for pair in pairs])) if pairs else None,
                   "internal_stage_f1": float(np.mean(internal_f1)), "end_to_end_stage_f1": aggregate["stage_macro_f1"],
                   "coverage": aggregate["coverage"]}
            summary.append(row)
            results[population][method] = {"summary": row, "per_case": per_case, "boundaries": boundary}
        for repeat in (0, 1):
            for control in ("pure", "direct", "fixed", "numeric_first", "numeric_vote", "matched_uniform", "matched_center"):
                for cid, rows in labels.items():
                    actual = np.array([int(row["activity_label"]) for row in rows])
                    before = np.array(predictions[batch][f"{control}_r{repeat}"][cid]["activity"])
                    after = np.array(predictions[batch][f"counterevidence_r{repeat}"][cid]["activity"])
                    corrections.append({"population": population, "case_id": cid, "repeat": repeat, "control": control,
                                        **correction_utility(actual, before, after, actual >= 0),
                                        "gained_tp": int(np.count_nonzero((actual == 1) & (before != 1) & (after == 1))),
                                        "lost_tp": int(np.count_nonzero((actual == 1) & (before == 1) & (after != 1))),
                                        "removed_fp": int(np.count_nonzero((actual == 0) & (before == 1) & (after != 1))),
                                        "gained_fp": int(np.count_nonzero((actual == 0) & (before != 1) & (after == 1)))})
    gates = []
    for population in ("mechanism-v1", "confirmation-v1", "background-review-v1", "old-AI", "new-AI"):
        for repeat in (0, 1):
            candidate = results[population][f"counterevidence_r{repeat}"]["summary"]
            pure = results[population][f"pure_r{repeat}"]["summary"]
            gates.append({"population": population, "repeat": repeat, "gate": "noninferiority",
                          "passed": candidate["f1"] >= pure["f1"] and candidate["weak_recall"] >= pure["weak_recall"]})
    for population in ("old-AI", "new-AI"):
        delta = [results[population][f"counterevidence_r{rep}"]["summary"]["f1"]
                 - results[population][f"numeric_vote_r{rep}"]["summary"]["f1"] for rep in (0, 1)]
        nets = [sum(row["net_correct"] for row in corrections if row["population"] == population
                    and row["repeat"] == rep and row["control"] == "pure") for rep in (0, 1)]
        gates.append({"population": population, "gate": "increment", "f1_delta_vs_vote": delta,
                      "net_vs_pure": nets, "passed": min(delta) >= 0 and max(delta) > 0 and max(nets) > 0})
    write_json(out / "results.json", results)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-record.csv", per_records)
    write_csv(out / "corrections.csv", corrections)
    write_json(out / "reference-bindings.json", reference_bindings)
    write_json(out / "decision.json", {"gates": gates, "continue": all(row["passed"] for row in gates),
                                       "new_remote_calls": 0, "independent_confirmation": False})
    write_json(out / "manifest.json", {"files": {p.name: sha(p) for p in out.iterdir() if p.is_file()}})
    print(read_json(out / "decision.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "score"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve())
    else:
        score(args.out.resolve())
