"""Freeze an interval-motion protection trial on all exposed cached records."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from analyze_landslide_codex_observation_reference import scoring_inputs, stage_reference_arrays
from probe_landslide_counterevidence_review import BASE, BATCHES, HOLDOUT, ROOT, bind, matched_veto
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import load_labels
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_event_motion_protection import (
    event_motion_support,
    protect_supported_denials,
)
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility

CACHE = BASE / "counterevidence-development-v2"
ARMS = ("event_protection", "unconditional_protection", "net_only", "matched_uniform", "matched_center")
PROTOCOL = {
    "schema": "ordered-event-denial-protection-v1",
    "scope": "All66 exposed records; controlled, residual injection and two authorAI references separate; no independent confirmation",
    "primary": "event_protection",
    "change": "Only a definite local1->0 revision of an initial visual activity can be vetoed using positive accumulated ordered raw motion inside that INITIAL event; no newly assigned activity from numeric evidence",
    "event": "Existing event_masks: at most7 unknown dates bridged, no negative dates bridged; no missing interpolation",
    "evidence": "Nonoverlapping14-day raw medians anchored to event start;>=8 complete dates/block and>=6 blocks; first-to-last net/noise>=2 and physical rate>=.03; project noise-scaled medians on noise-scaled endpoint direction; Kendalltau>=.6,>=3 positive increments, largest positive increment share<=.5",
    "thresholds": "2/.03 and14/8 from existing rule;6blocks (=84days) from existing long observation extent; .6 rank and .5 concentration fixed descriptive engineering criteria, not calibrated probabilities; no sweep",
    "unchanged": "Original VLM replies, raw21/42 selection, maximum2 disjoint targets,49-day permissions, observed/support masks and common population-specific stage head",
    "controls": "direct,fixed,pure,numeric-first,old day-motion protection,numeric majority; unconditional preservation, net-only (remove rank/concentration), matcheduniform andcenter at exact same protected-denial count per target; original strong numeric models",
    "cost": "Zero new calls; same pure receipts plus raw event checks; compute reported separately; no fewer-call claim",
    "continue_conditions": {
        "both_AI": "Both repetitions of BOTH authorAI populations: F1 and weakR>=pure, net definite correction>0, at least2 positive-net records; F1>=unconditional and at leastone strict repetition per population",
        "other_populations": "Both repetitions mechanism,confirmation,background: F1 and weakR>=pure; background net definite correction>=0",
        "ordinal_increment": "BothAI populations: at leastone repetition F1>net_only, with neither repetition worse",
        "stop": "Any failure stops this exact version; no threshold/seed/repeat rescue; keep original default",
    },
    "new_remote_calls": 0,
}


def write_csv(path, rows):
    import csv
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", PROTOCOL)
    shutil.copyfile(Path(__file__), out / "source.py")
    module = ROOT / "src/gnss_sim/landslide_event_motion_protection.py"
    shutil.copyfile(module, out / module.name)
    for name, digest in read_json(CACHE / "manifest.json")["files"].items():
        if sha(CACHE / name) != digest:
            raise ValueError("Prior frozen comparison changed")
    bindings = read_json(CACHE / "source-bindings.json")
    for name, digest in bindings.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Original cached inputs changed")
    bind(bindings, CACHE / "predictions.json")
    bind(bindings, CACHE / "vetoes.json")
    bind(bindings, module)
    cached_predictions = read_json(CACHE / "predictions.json")
    predictions = {}
    previous = read_json(CACHE / "vetoes.json")
    evidence_rows, protected_rows, timings = {}, {}, []
    for batch, methods in cached_predictions.items():
        methods = {name: records for name, records in methods.items()
                   if not name.endswith(("_r0", "_r1")) or name.startswith(("direct_", "fixed_", "pure_", "numeric_first_", "numeric_vote_"))}
        for repeat in (0, 1):
            methods[f"day_motion_protection_r{repeat}"] = cached_predictions[batch][f"motion_only_r{repeat}"]
        predictions[batch] = methods
        source = HOLDOUT / "run" if batch == "new-AI" else BASE / batch
        head = XGBClassifier()
        head.load_model(source / "stage-xgboost.json")
        config = read_json(source / "protocol.json")
        choices = read_json(source / "selections.json")
        policy = ReviewPolicy(**config["policy"])
        evidence_rows[batch], protected_rows[batch] = {}, {}
        for cid in methods["direct_r0"]:
            case, feature = load_case_features(source, cid)
            long = raw_motion_evidence(case, feature)
            codes = head.predict(feature.matrix).astype(int) + 1
            for repeat in (0, 1):
                initial = np.array(methods[f"direct_r{repeat}"][cid]["activity"])
                pure = np.array(methods[f"pure_r{repeat}"][cid]["activity"])
                started = time.perf_counter()
                support, details = event_motion_support(case, feature.noise, initial)
                timings.append({"batch": batch, "case_id": cid, "repeat": repeat, "event_check_seconds": time.perf_counter() - started})
                evidence_rows[batch][f"{cid}-r{repeat}"] = details
                targets = choices[f"{cid}-r{repeat}"]["selected"]
                neighbourhood = conflict_neighbourhood(initial, feature, long, policy)[1]
                permission = target_scopes(neighbourhood, feature.observed, targets)[0]
                eligible = permission & (initial == 1) & (pure == 0)
                activity, protected = protect_supported_denials(initial, pure, permission, feature.observed, support)
                masks = {"event_protection": protected, "unconditional_protection": eligible}
                net_support = np.zeros(len(initial), dtype=bool)
                for event in details:
                    if event.get("net_over_noise", 0) >= 2 and event.get("rate_mm_per_day", 0) >= .03:
                        net_support[event["start"]:event["stop"]] = True
                masks["net_only"] = eligible & net_support
                for arm, centered in (("matched_uniform", False), ("matched_center", True)):
                    masks[arm] = matched_veto(eligible, protected, targets, centered)
                for arm in ARMS:
                    activity = pure.copy()
                    activity[masks[arm]] = initial[masks[arm]]
                    if np.any(activity[~permission] != initial[~permission]) or np.any(masks[arm] & ~eligible):
                        raise ValueError("Protection exceeded the original definite-denial authority")
                    method = f"{arm}_r{repeat}"
                    methods.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                    old_cost = previous[batch][f"counterevidence_r{repeat}"][cid]
                    protected_rows[batch].setdefault(method, {})[cid] = {
                        "protected_days": np.flatnonzero(masks[arm]).tolist(), "eligible_days": np.flatnonzero(eligible).tolist(),
                        "targets": targets, "requests": old_cost["requests"], "reported_tokens": old_cost["reported_tokens"],
                    }
    write_json(out / "source-bindings.json", bindings)
    write_json(out / "predictions.json", predictions)
    write_json(out / "event-evidence.json", evidence_rows)
    write_json(out / "protections.json", protected_rows)
    write_csv(out / "timings.csv", timings)
    write_json(out / "prediction-freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print("Frozen66-record event-protection trial; scoring references not opened", flush=True)


def populations(predictions):
    groups = {}
    for batch in BATCHES:
        source = BASE / batch
        cases = {cid: load_case_features(source, cid)[0] for cid in predictions[batch]["direct_r0"]}
        reference = source / ("analysis-v1/reference-subset.csv" if batch == "background-review-v1" else "reference-subset.csv")
        labels = load_labels(reference, cases)
        weak = {cid: np.array([row["feature"] == "slow_displacement" if batch == "pilot-v3"
                              else int(row["weak_signal"]) == 1 for row in rows]) for cid, rows in labels.items()}
        groups[batch] = (batch, labels, weak)
    for name, directory, batch in (("old-AI", BASE / "codex-observation-review-v1", "confirmation-v1"),
                                    ("new-AI", BASE / "ai-observable-holdout-data-v1/codex-review", "new-AI")):
        labels, _, weak = scoring_inputs(directory)
        if name == "new-AI":
            weak = {cid: np.zeros(len(rows), dtype=bool) for cid, rows in labels.items()}
            for cid, spans in read_json(directory / "activity-decisions.json")["weak_interiors_authored_before_predictions"].items():
                for start, stop in spans:
                    weak[cid][start:stop] = True
            for cid, rows in labels.items():
                weak[cid] &= np.array([int(row["activity_label"]) == 1 for row in rows])
        groups[name] = (batch, labels, weak)
    return groups


def score(out):
    for name, digest in read_json(out / "prediction-freeze.json").items():
        if sha(out / name) != digest:
            raise ValueError("Frozen trial changed")
    for name, digest in read_json(out / "source-bindings.json").items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen inputs changed")
    predictions = read_json(out / "predictions.json")
    summary, results, corrections = [], {}, []
    for population, (batch, labels, weak) in populations(predictions).items():
        results[population] = {}
        for method, records in predictions[batch].items():
            per_case, boundaries, actual_stage, predicted_stage = {}, {}, [], []
            weak_tp = 0
            parsed = {}
            for cid, rows in labels.items():
                parsed[cid] = ActivityPrediction(**{key: np.array(value) if key in {"activity", "stage"} else value
                                                    for key, value in records[cid].items()})
                prediction = parsed[cid]
                per_case[cid] = evaluate_case(rows, prediction)
                boundaries[cid] = boundary_errors(rows, prediction.activity, .3)
                stage = stage_reference_arrays([dict(row, activity_label=int(row["activity_label"])) for row in rows], prediction)
                actual_stage.extend(stage["actual_stage"][stage["stage_interior"]])
                predicted_stage.extend(prediction.stage[stage["stage_interior"]])
                weak_tp += int(np.count_nonzero(weak[cid] & (prediction.activity == 1)))
            total = aggregate_scores(list(per_case.values()))
            matches = [pair for row in boundaries.values() for pair in row["matched_pairs"]]
            row = {"population": population, "method": method, **total["daily"],
                   "weak_recall": weak_tp / sum(int(mask.sum()) for mask in weak.values()),
                   "event_f1_iou03": total["events"]["f1"],
                   "event_f1_iou05": aggregate_scores([evaluate_case(labels[cid], parsed[cid], .5) for cid in labels])["events"]["f1"],
                   "onset_mae": float(np.mean([abs(pair["onset_error_days"]) for pair in matches])) if matches else None,
                   "offset_mae": float(np.mean([abs(pair["offset_error_days"]) for pair in matches])) if matches else None,
                   "internal_stage_f1": float(precision_recall_fscore_support(actual_stage, predicted_stage, labels=[1, 2, 3], zero_division=0)[2].mean()),
                   "end_to_end_stage_f1": total["stage_macro_f1"], "coverage": total["coverage"]}
            summary.append(row)
            results[population][method] = {"summary": row, "per_case": per_case, "boundaries": boundaries}
        for repeat in (0, 1):
            for control in ("pure", "direct", "fixed", "numeric_first", "numeric_vote", "unconditional_protection", "net_only", "matched_uniform", "matched_center", "day_motion_protection"):
                for cid, rows in labels.items():
                    actual = np.array([int(row["activity_label"]) for row in rows])
                    before = np.array(predictions[batch][f"{control}_r{repeat}"][cid]["activity"])
                    after = np.array(predictions[batch][f"event_protection_r{repeat}"][cid]["activity"])
                    corrections.append({"population": population, "repeat": repeat, "control": control, "case_id": cid,
                                        **correction_utility(actual, before, after, actual >= 0),
                                        "gained_tp": int(np.count_nonzero((actual == 1) & (before != 1) & (after == 1))),
                                        "lost_tp": int(np.count_nonzero((actual == 1) & (before == 1) & (after != 1))),
                                        "gained_fp": int(np.count_nonzero((actual == 0) & (before != 1) & (after == 1))),
                                        "removed_fp": int(np.count_nonzero((actual == 0) & (before == 1) & (after != 1)))})
    gates = []
    for population in ("old-AI", "new-AI", "mechanism-v1", "confirmation-v1", "background-review-v1"):
        delta_unconditional, delta_net = [], []
        for repeat in (0, 1):
            candidate = results[population][f"event_protection_r{repeat}"]["summary"]
            pure = results[population][f"pure_r{repeat}"]["summary"]
            changes = [row for row in corrections if row["population"] == population and row["repeat"] == repeat and row["control"] == "pure"]
            net = sum(row["net_correct"] for row in changes)
            beneficiaries = sum(row["net_correct"] > 0 for row in changes)
            passed = candidate["f1"] >= pure["f1"] and candidate["weak_recall"] >= pure["weak_recall"]
            if population in {"old-AI", "new-AI"}:
                passed = passed and net > 0 and beneficiaries >= 2
            if population == "background-review-v1":
                passed = passed and net >= 0
            gates.append({"population": population, "repeat": repeat, "net_vs_pure": net,
                          "beneficiaries": beneficiaries, "passed": bool(passed)})
            delta_unconditional.append(candidate["f1"] - results[population][f"unconditional_protection_r{repeat}"]["summary"]["f1"])
            delta_net.append(candidate["f1"] - results[population][f"net_only_r{repeat}"]["summary"]["f1"])
        if population in {"old-AI", "new-AI"}:
            for name, delta in (("conditional_vs_unconditional", delta_unconditional), ("ordering_vs_net_only", delta_net)):
                gates.append({"population": population, "gate": name, "f1_deltas": delta,
                              "passed": min(delta) >= 0 and max(delta) > 0})
    write_json(out / "results.json", results)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "corrections.csv", corrections)
    write_json(out / "decision.json", {"gates": gates, "continue": all(row["passed"] for row in gates),
                                       "independent_confirmation": False, "new_remote_calls": 0})
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
