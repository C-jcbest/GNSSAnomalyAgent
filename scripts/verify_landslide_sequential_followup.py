"""Reconstruct each sequential decision from raw inputs and actual receipts."""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_landslide_censored_followup_v2 import overlapping_scope
from analyze_landslide_review_utility_v2 import verify_receipts
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity
from run_landslide_ai_reference_holdout import replay, verify_bindings
from run_landslide_sequential_followup import ARMS
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import ReviewPolicy, combine_reviews, raw_motion_evidence
from gnss_sim.landslide_censored_followup import continuation_choice, followup_targets
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_window_experiment import parse_window_activity


def verify(comparison, out):
    run = comparison / "run"
    verify_bindings(run, "prediction-freeze.json")
    receipts = verify_receipts(run)
    original, _, _ = replay(run, receipts)
    schedules = read_json(run / "followup-schedules.json")
    selections = read_json(run / "selections.json")
    predictions = read_json(run / "followup-predictions.json")
    used = read_json(run / "followup-used-requests.json")
    config = read_json(run / "protocol.json")
    policy = ReviewPolicy(**config["policy"])
    head = XGBClassifier()
    head.load_model(run / "stage-xgboost.json")
    checked, triggered, temporal_checks = 0, [], 0
    for cid in config["case_ids"]:
        case, feature = load_case_features(run, cid)
        long = raw_motion_evidence(case, feature)
        codes = head.predict(feature.matrix).astype(int) + 1
        for rep in range(config["repetitions"]):
            key = f"{cid}-r{rep}"
            initial = np.array(original[f"direct_visual_r{rep}"][cid]["activity"])
            conflicts, authority = conflict_neighbourhood(initial, feature, long, policy)
            row = schedules[key]
            selected = selections[key]["selected"]
            if row["budget"] != len(selected):
                raise ValueError("Review budget differs from original selection")
            first = None
            if selected:
                target = min(selected, key=lambda value: (-value["conflict_days"], value["start"]))
                expected_key = f"{cid}-window-{target['index']:02d}-image-r{rep}"
                if row["first"]["request_id"] != expected_key:
                    raise ValueError("First observation is not the highest-conflict original target")
                first = local_reply(receipts, row["first"], feature.observed)
                if first != row["first"]:
                    raise ValueError("Saved first observation differs from its actual receipt")
                remaining = next((value for value in selected if value["index"] != target["index"]), None)
                choice = continuation_choice(target, np.array(first["activity"]), conflicts,
                                             feature.observed, policy.minimum_conflict_days)
                targets = followup_targets(target, remaining, choice, len(initial))
                if row["continuation"] != choice or row["targets"] != targets:
                    raise ValueError("Follow-up choice does not replay from the first answer")
                if choice["numeric_admitted"] and remaining is not None:
                    triggered.append(key)
            for arm in ARMS:
                method = f"first_only_r{rep}" if arm == "first_only" else f"followup_{arm}_r{rep}"
                reviews = [] if first is None else [first]
                request_ids = used[method][cid]
                if request_ids[0] != f"{cid}-direct-r{rep}":
                    raise ValueError("Wrong initial receipt")
                second = None
                if first is not None and arm != "first_only":
                    target = row["targets"]["censored" if arm == "censored_latest" else arm]
                    if target is not None:
                        second = local_reply(receipts, {"start": target["start"], "stop": target["stop"],
                                                       "request_id": request_ids[-1]}, feature.observed)
                        reviews.append(second)
                if request_ids != [request_ids[0]] + [value["request_id"] for value in reviews]:
                    raise ValueError("Method receipt count does not match its bounded observations")
                activity = initial.copy()
                if reviews:
                    reviewed = combine_reviews(initial, reviews, feature.observed).activity
                    if arm == "censored_latest" and second is not None:
                        start, stop = second["start"], second["stop"]
                        reviewed[start:stop] = np.array(second["activity"])[start:stop]
                    scope = overlapping_scope(authority, feature.observed, reviews)
                    activity = scoped_activity(initial, reviewed, scope, feature.observed)
                actual = learned_stages(codes, activity, feature.supported).to_dict()
                if actual != predictions[method][cid]:
                    raise ValueError("Sequential prediction does not replay")
                checked += 1
                if second is not None:
                    first_file = run / "requests" / (first["request_id"] + ".response.json")
                    second_file = run / "requests" / (second["request_id"] + ".started.json")
                    if first_file.stat().st_mtime_ns > second_file.stat().st_mtime_ns:
                        raise ValueError("Second observation was started before first answer saved")
                    temporal_checks += 1
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "audit.json", {"status": "passed", "original_methods_replayed": len(original),
        "candidate_record_repeat_pairs_replayed": checked, "first_before_second_mtime_checks": temporal_checks,
        "main_triggered_record_repeats": triggered, "new_remote_calls": 0,
        "all_receipts_verified": len(receipts), "registered_budget_and_targets_replayed": True,
        "original49day_authority_and_target_union_replayed": True,
        "prediction_freeze_sha256": sha(run / "prediction-freeze.json")})
    print(read_json(out / "audit.json"), flush=True)


def local_reply(receipts, target, observed):
    receipt = receipts[target["request_id"]]
    try:
        if receipt["status"] != "returned_json":
            raise ValueError("Failed visual response")
        activity = parse_window_activity(json.loads(receipt["output"]), observed,
                                        target["start"], target["stop"])
        status = "success"
    except (ValueError, KeyError):
        activity = np.full(len(observed), -1)
        status = "failed"
    return {"start": target["start"], "stop": target["stop"], "activity": activity.tolist(),
            "status": status, "request_id": target["request_id"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.comparison.resolve(), args.out.resolve())
