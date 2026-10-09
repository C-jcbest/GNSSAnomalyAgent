"""Explain support versus native abstention, without changing any frozen outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np
from landslide_stage_evidence import load_case
from landslide_stage_support_ablation import RUN, check_digest, read_json
from verify_landslide_stage_support_ablation import verify_metrics

from gnss_sim.artifacts import sha, write_json

STAGES = ("acceleration","steady_motion","deceleration")


def check_support_effect(rows,reported):
    counts = Counter()
    for row in rows:
        if row["native_feature"] == row["feature"]:
            continue
        assert row["native_feature"] in STAGES and row["feature"] == "unknown"
        counts["suppressed_activity_days"] += 1
        if row["reference_evaluable"] == "True":
            counts["suppressed_evaluable_days"] += 1
            relation = "native_agreement_to_abstention" if row["native_feature"] == row["reference_feature"] else "native_disagreement_to_abstention"
            counts[relation] += 1
        else:
            counts["suppressed_reference_unknown_days"] += 1
    assert reported == {key:counts[key] for key in reported}


def diagnose():
    public = RUN / "public"
    freeze = read_json(RUN / "analysis-freeze.json")
    for name in ("analysis.json","daily-stage-support.csv","daily-reference.json"):
        check_digest(public / name,freeze["public_files_sha256"][name])
    analysis = read_json(public / "analysis.json")
    grouped = defaultdict(list)
    with (public / "daily-stage-support.csv").open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["case_id"],row["method"],row["condition"]].append(row)
    for case in analysis["cases"]:
        for method in ("rule","xgboost","visual"):
            baseline = grouped[case["case_id"],method,"fixed_61"]
            native_rows = [{**row,"feature":row["native_feature"]} for row in baseline]
            verify_metrics(native_rows,case["native_scores"][method])
            for condition in analysis["summary"]["conditions"]:
                check_support_effect(grouped[case["case_id"],method,condition],case["summary"][method][condition]["support_effect"])
    for method in ("rule","xgboost","visual"):
        native_rows = [{**row,"feature":row["native_feature"]} for case in analysis["cases"]
                       for row in grouped[case["case_id"],method,"fixed_61"]]
        verify_metrics(native_rows,analysis["summary"]["native_scores"][method])
        for condition in analysis["summary"]["conditions"]:
            rows = [row for case in analysis["cases"] for row in grouped[case["case_id"],method,condition]]
            check_support_effect(rows,analysis["summary"]["methods"][method][condition]["support_effect"])
    records = read_json(public / "source/source-manifest.json")["cases"]
    entries = {entry["case_id"]:entry for entry in read_json(public / "source/activity-manifest.json")["cases"]}
    joint = []
    for record in records:
        case,review,_,_,motions = load_case(public,record,entries)
        case_id = case.case_id
        for day in range(len(case.dates)):
            baselines = {method:grouped[case_id,method,"fixed_61"][day] for method in ("rule","xgboost","visual")}
            if baselines["rule"]["reference_evaluable"] != "True" or any(row["feature"] != "unknown" for row in baselines.values()):
                continue
            span = next(span for span in review["spans"] if span["start"] <= day < span["stop"])
            velocity = motions[31].velocity_mm_day[day]
            tangential = motions[31].tangential_acceleration_mm_day2[day]
            joint.append({"case_id":case_id,"day_index":day,
                "native_features":{method:row["native_feature"] for method,row in baselines.items()},
                "short_velocity_finite":velocity is not None and bool(np.all(np.isfinite(velocity))),
                "short_tangential_finite":tangential is not None and bool(np.isfinite(tangential)),
                "short_contained":span["start"] <= day-15 and day+16 <= span["stop"],
                "fallback_features":{method:grouped[case_id,method,"fallback_31_contained"][day]["feature"] for method in baselines}})
    result = {"analysis_freeze_sha256":sha(RUN / "analysis-freeze.json"),
        "additional_native_sklearn_metric_groups":39,"support_effect_groups_checked":195,
        "joint_abstained_reference_days":len(joint),
        "joint_native_definite_days":{method:sum(row["native_features"][method] in STAGES for row in joint) for method in ("rule","xgboost","visual")},
        "joint_31_velocity_finite_days":sum(row["short_velocity_finite"] for row in joint),
        "joint_31_tangential_finite_days":sum(row["short_tangential_finite"] for row in joint),
        "joint_31_contained_tangential_days":sum(row["short_tangential_finite"] and row["short_contained"] for row in joint),
        "joint_fallback_definite_days":{method:sum(row["fallback_features"][method] in STAGES for row in joint) for method in ("rule","xgboost","visual")},
        "joint_daily":joint,"formal_performance_scores":None,"model_calls":0,"reference_use":"development_only"}
    write_json(RUN / "support-diagnosis.json",result)
    print(json.dumps({key:value for key,value in result.items() if key != "joint_daily"},indent=2))


if __name__ == "__main__":
    diagnose()
