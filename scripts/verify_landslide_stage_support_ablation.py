"""Independent mask recomputation, sklearn metrics, CSV and saved-source checks."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np
from landslide_stage_evidence import load_case
from landslide_stage_support_ablation import CONFIG, ROOT, RUN, check_digest, read_json
from verify_landslide_stage_agreement import check_class_metrics

from gnss_sim.artifacts import sha, write_json

STAGES = ("acceleration","steady_motion","deceleration")


def verify_metrics(rows,reported):
    check_class_metrics(rows,reported)
    active = [row for row in rows if row["activity_label"] == "1"]
    definite = [row for row in active if row["reference_evaluable"] == "True"]
    unknown = [row for row in active if row["reference_evaluable"] == "False"]
    assert reported["observed_activity_days"] == len(active)
    assert reported["active_prediction_features"] == dict(Counter(row["feature"] for row in active))
    assert reported["reference_unknown_prediction_features"] == dict(Counter(row["feature"] for row in unknown))
    for stage in STAGES:
        counts = Counter(row["feature"] for row in definite if row["reference_feature"] == stage)
        assert reported["confusion"][stage] == {feature:counts[feature] for feature in (*STAGES,"unknown")}
    all_classes = all(any(row["reference_feature"] == stage for row in definite) for stage in STAGES)
    assert reported["all_three_reference_classes_supported"] == all_classes
    assert (reported["development_agreement_macro_f1"] is not None) == all_classes


def verify_inventory(rows,inventory):
    active = [row for row in rows if row["activity_label"] == "1"]
    native_definite = [row for row in active if row["native_feature"] in STAGES]
    final_definite = [row for row in active if row["feature"] in STAGES]
    expected = {
        "observed_activity_days":len(active),"native_definite_days":len(native_definite),
        "native_unknown_days":len(active)-len(native_definite),"final_definite_days":len(final_definite),
        "suppressed_native_days":len(native_definite)-len(final_definite),
        "selected_cross_boundary_days":sum(row["selected_contained"] == "False" for row in active),
    }
    for key,value in expected.items():
        assert inventory[key] == value
    if "selected_window_days" in inventory:
        assert inventory["selected_window_days"] == dict(Counter(row["selected_window"] for row in final_definite))
        assert inventory["support_reasons"] == dict(Counter(row["support_reason"] for row in active))


def verify_changes(rows,baseline,reported):
    counts = Counter()
    changed = set()
    for row,before in zip(rows,baseline,strict=True):
        assert (row["case_id"],row["day_index"]) == (before["case_id"],before["day_index"])
        if row["feature"] == before["feature"]:
            continue
        assert row["activity_label"] == "1"
        restored = before["feature"] == "unknown" and row["feature"] in STAGES
        removed = before["feature"] in STAGES and row["feature"] == "unknown"
        assert restored or removed
        action = "restored" if restored else "removed"
        feature = row["feature"] if restored else before["feature"]
        relation = "reference_unknown"
        if row["reference_evaluable"] == "True":
            relation = "same" if row["reference_feature"] == feature else "different"
        counts[f"{action}_{relation}_days"] += 1
        changed.add((row["case_id"],int(row["day_index"]),before["feature"],row["feature"],relation))
    assert reported["counts"] == {f"{a}_{b}_days":counts[f"{a}_{b}_days"]
        for a in ("restored","removed") for b in ("same","different","reference_unknown")}
    restored_days = set()
    for span in reported["spans"]:
        for day in range(span["start"],span["stop"]):
            key = (span["case_id"],day,span["baseline_feature"],span["alternate_feature"],span["reference_relation"])
            assert key not in restored_days
            restored_days.add(key)
    assert restored_days == changed


def verify():
    config = read_json(CONFIG)
    public = RUN / "public"
    prediction_freeze = read_json(RUN / "prediction-freeze.json")
    analysis_freeze = read_json(RUN / "analysis-freeze.json")
    reference_ledger = read_json(RUN / "reference-load-ledger.json")
    assert prediction_freeze["reference_loaded_by_execution"] is False
    assert analysis_freeze["prediction_freeze_sha256"] == reference_ledger["prediction_freeze_sha256"] == sha(RUN / "prediction-freeze.json")
    assert prediction_freeze["frozen_utc"] < reference_ledger["loaded_utc"]
    assert analysis_freeze["reference_load_ledger_sha256"] == sha(RUN / "reference-load-ledger.json")
    for name,digest in prediction_freeze["source_sha256"].items():
        check_digest(ROOT / name,digest)
        check_digest(RUN / "source-code" / name,digest)
    for group in (prediction_freeze["input_sha256"],reference_ledger["input_sha256"]):
        for name,digest in group.items():
            check_digest(ROOT / name,digest)
    for group in (prediction_freeze["files_sha256"],analysis_freeze["public_files_sha256"]):
        for name,digest in group.items():
            check_digest(public / name,digest)
    analysis = read_json(public / "analysis.json")
    cases = {case["case_id"]:case for case in analysis["cases"]}
    reference = {(row["case_id"],row["day_index"]):row for row in read_json(public / "daily-reference.json")}
    groups = defaultdict(list)
    seen = set()
    with (public / "daily-stage-support.csv").open(encoding="utf-8-sig",newline="") as scored_stream, (public / "daily-support-predictions.csv").open(encoding="utf-8-sig",newline="") as frozen_stream:
        for row,frozen in zip(csv.DictReader(scored_stream),csv.DictReader(frozen_stream),strict=True):
            assert {key:row[key] for key in frozen} == frozen
            key = (row["case_id"],row["method"],row["condition"],int(row["day_index"]))
            assert key not in seen
            seen.add(key)
            actual = reference[row["case_id"],int(row["day_index"])]
            assert row["reference_feature"] == actual["feature"]
            assert row["reference_evaluable"] == str(actual["stage_evaluable"])
            groups[key[:3]].append(row)
    assert len(seen) == 197100 and len(groups) == 180
    records = read_json(public / "source/source-manifest.json")["cases"]
    entries = {entry["case_id"]:entry for entry in read_json(public / "source/activity-manifest.json")["cases"]}
    source_run = ROOT / config["native_prediction_run"]
    recomputed = 0
    for record in records:
        case,review,_,_,motions = load_case(public,record,entries)
        velocity = {window:np.array([value is not None and np.all(np.isfinite(value)) for value in motion.velocity_mm_day]) for window,motion in motions.items()}
        tangential = {window:np.array([value is not None and np.isfinite(value) for value in motion.tangential_acceleration_mm_day2]) for window,motion in motions.items()}
        contained = {window:np.zeros(len(case.dates),dtype=bool) for window in motions}
        for span in review["spans"]:
            if span["state"] == "activity":
                for window in motions:
                    half = window // 2
                    if span["stop"]-half > span["start"]+half:
                        contained[window][span["start"]+half:span["stop"]-half] = True
        for method in config["methods"]:
            native = read_json(public / f"native/{case.case_id}-{method}.json")
            assert native == read_json(source_run / f"public/native/{case.case_id}-{method}.json")
            baseline = groups[case.case_id,method,"fixed_61"]
            assert read_json(public / f"daily/{case.case_id}-{method}-fixed_61.json") == read_json(source_run / f"public/daily/{case.case_id}-{method}.json")
            for condition in analysis["summary"]["conditions"]:
                rows = groups[case.case_id,method,condition]
                output = read_json(public / f"daily/{case.case_id}-{method}-{condition}.json")
                trace = read_json(public / f"support-trace/{case.case_id}-{method}-{condition}.json")
                assert len(rows) == len(output) == len(trace) == 1095
                for day,(row,stored,evidence) in enumerate(zip(rows,output,trace,strict=True)):
                    for key in stored:
                        assert row[key] == str(stored[key])
                    for key in evidence:
                        assert row[key] == ("" if evidence[key] is None else str(evidence[key]))
                    before = native[day]["feature"]
                    assert row["native_feature"] == before
                    selected = None
                    if before in STAGES:
                        eligible = {window:bool(velocity[window][day] and (before == "steady_motion" or tangential[window][day])) for window in motions}
                        if condition == "fallback_31_contained":
                            if eligible[61]:
                                selected = 61
                            elif eligible[31] and contained[31][day]:
                                selected = 31
                        elif condition == "contained_61":
                            selected = 61 if eligible[61] and contained[61][day] else None
                        else:
                            window = int(condition.split("_")[1])
                            selected = window if eligible[window] else None
                        expected = before if selected is not None else "unknown"
                        assert row["feature"] == expected
                    else:
                        expected = "none" if int(row["activity_label"]) == 0 else before
                        assert row["feature"] == expected
                    assert row["selected_window"] == (str(selected) if selected is not None else "")
                    assert row["selected_contained"] == (str(bool(contained[selected][day])) if selected is not None else "")
                    assert int(row["day_index"]) == day and row["date"] == str(case.dates[day])
                    if case.displacement_mm[day] is None or int(row["activity_label"]) != 1:
                        assert row["feature"] not in STAGES
                    if condition == "fallback_31_contained" and baseline[day]["feature"] in STAGES:
                        assert row["feature"] == baseline[day]["feature"]
                    recomputed += 1
                group = cases[case.case_id]["summary"][method][condition]
                verify_metrics(rows,group["scores"])
                verify_inventory(rows,group["coverage"])
                verify_changes(rows,baseline,group["baseline_changes"])
                for interior in cases[case.case_id]["interiors"]:
                    start,stop = interior["start"],interior["stop"]
                    reported = interior["methods"][method][condition]
                    verify_metrics(rows[start:stop],reported["scores"])
                    verify_inventory(rows[start:stop],reported["coverage"])
                    verify_changes(rows[start:stop],baseline[start:stop],reported["baseline_changes"])
    for method in config["methods"]:
        baseline = [row for case in analysis["cases"] for row in groups[case["case_id"],method,"fixed_61"]]
        for condition in analysis["summary"]["conditions"]:
            rows = [row for case in analysis["cases"] for row in groups[case["case_id"],method,condition]]
            group = analysis["summary"]["methods"][method][condition]
            verify_metrics(rows,group["scores"])
            verify_inventory(rows,group["coverage"])
            verify_changes(rows,baseline,group["baseline_changes"])
    # Verify actual historical image/prompt associations, not the separate audit image version.
    for case in analysis["cases"]:
        request = case["visual_request"]
        if request is None:
            continue
        assert len(request["images"]) == 7
        source_id = request["started"]["request_id"]
        assert request["started"] == read_json(source_run / f"public/requests/{source_id}.started.json")
        assert request["response"] == read_json(source_run / f"public/requests/{source_id}.response.json")
        for name,digest in zip(request["images"],request["started"]["image_sha256"].values(),strict=True):
            check_digest(public / name,digest)
    result = {"status":"passed","independent_mask_rows":recomputed,"baseline_exact_cases_methods":36,
        "independent_metric_groups":510,"case_groups":180,"interior_groups":315,"aggregate_groups":15,
        "csv_rows":len(seen),"execution_source_hashes":len(prediction_freeze["source_sha256"]),
        "scored_files":len(analysis_freeze["public_files_sha256"]),"frozen_prediction_files":len(prediction_freeze["files_sha256"]),
        "prediction_freeze_before_reference_load":True,"historical_visual_images_checked":63,
        "model_calls":0,"browser_rendering":"not_verified","performance_scores":None}
    write_json(RUN / "verification-audit.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    verify()
