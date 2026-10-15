"""Comprehensive typed, range, generic, ablation, scenario and cost evaluation."""
from __future__ import annotations

import argparse
import copy
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp20_range_metrics import affiliation, aggregate_affiliation, evaluate, mean_or_none
from exp20_range_metrics import aggregate as range_aggregate
from experiment_exp22 import cases
from experiment_exp23 import recovery_graph
from experiment_exp25 import automatic_response, recover
from replay_author_parser import load_author_functions

from gnss_research.files import read_json, sha256, write_json
from gnss_research.fusion import DERIVED_METHODS
from gnss_research.metrics import aggregate as daily_aggregate
from gnss_research.metrics import evaluate_record
from gnss_research.schema import EVENT_TYPES, Prediction, event_masks, observed_values

PRIMARY = "sdr_gr_flash"
NUMERIC = ("random_forest", "rf_balanced", "extra_trees", "isolation_forest", "rf_split_temporal", "segment_numeric")
MAIN = (*NUMERIC, "visual_only", "visualtime_gnss", "visualtime_author_parser", "sdr_flash", "sdr_vl", PRIMARY, "sdr_gr_vl")
TITLES = {"transient": "暂态范围", "deformation": "持续形变范围", "range_union": "两类范围并集"}


def author_replay(observation, work, functions):
    """Supplemental original-parser replay; never rewrite strict run receipts."""
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    mask = np.zeros(len(observed), dtype=bool)
    receipts = []
    for axis in "NEU":
        for task in ("point", "range"):
            request = work / f"{axis}-{task}"
            receipt = {"axis": axis, "task": task, "status": "failed"}
            try:
                original = read_json(request / "receipt.json")
                if original.get("http_status") != 200:
                    raise ValueError("provider response unsuccessful")
                response = read_json(request / "response.json")
                parsed = functions["parse_output"](response["choices"][0]["message"]["content"],
                                                   "global" if task == "point" else "trend")
                if task == "point":
                    vector = functions["point_to_vector"](parsed, len_vector=len(observed))
                else:
                    vector = functions["interval_to_vector"](parsed, start=0, end=len(observed), pred=True)
                mask |= vector.astype(bool)
                receipt.update(status="success", parsed=parsed, response_sha256=sha256(request / "response.json"))
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                receipt["error_type"] = type(error).__name__
            receipts.append(receipt)
    mask &= observed
    return {"status": "failed" if any(r["status"] != "success" for r in receipts) else "success",
            "prediction": None, "generic_mask": mask.tolist(), "generic_score": None,
            "subrequests": receipts, "request_count": 0,
            "evaluation_dependency_workdirs": [str(work)]}


def load_test(visual: Path, pilot: bool):
    functions, author_source = load_author_functions()
    phases = ("pilot",) if pilot else ("pilot", "remaining")
    records = []
    for phase in phases:
        if not (visual / f"{phase}-derived.json").exists():
            raise ValueError("all planned stations must finish before evaluation")
        registration = read_json(visual / f"{phase}-base-registration.json")
        records.extend(registration["records"])
    dataset, baseline = Path(registration["dataset"]), Path(registration["baseline"])
    modes = sorted(path.name for path in (visual / "final").iterdir() if path.is_dir())
    entries = []
    for record in records:
        station = record["station_id"]
        predictions = {method: read_json(baseline / "predictions" / method / station / "result.json") for method in NUMERIC}
        predictions.update({method: read_json(visual / "final" / method / station / "result.json") for method in modes})
        predictions["visualtime_author_parser"] = author_replay(
            read_json(dataset / record["observation"]), visual / "predictions/visualtime_gnss" / station, functions)
        entries.append({"id": f"{registration['seed']}-{station}", "record": record, "dataset": dataset,
                        "seed": registration["seed"], "predictions": predictions})
    return entries, {"cohort": "test-pilot" if pilot else "test", "dataset": str(dataset),
                     "baseline": str(baseline), "visual": str(visual), "seed": registration["seed"],
                     "split": "test", "stations": len(entries), "method_selection_after_test": False,
                     "author_parser_supplement": {"source": author_source, "new_requests": 0,
                         "disclosure": "Post-registration replay of identical responses with unchanged author functions; strict results retained; partial union plus failed-station-zero sensitivity"}}


def load_historical():
    entries = []
    direct = {"sdr_flash": ("exp22-image-ablation-full-v1", "flash_image"),
              "sdr_vl": ("exp22-image-ablation-full-v1", "vl_image"),
              "sdr_flash_noimage": ("exp22-image-ablation-full-v1", "flash_noimage"),
              "sdr_vl_noimage": ("exp22-image-ablation-full-v1", "vl_noimage"),
              "sdr_gr_flash": ("exp24-historical-gap-v1", "gap_flash_image"),
              "sdr_gr_vl": ("exp24-historical-gap-v1", "gap_vl_image"),
              "sdr_gr_recovery_noimage": ("exp24-historical-gap-v1", "gap_flash_noimage"),
              "sdr_gr_vl_recovery_noimage": ("exp24-historical-gap-v1", "gap_vl_noimage"),
              "sdr_gr_numeric_recovery": ("exp24-historical-gap-v1", "gap_numeric"),
              "sdr_gr_candidate_all": ("exp24-historical-gap-v1", "gap_candidate_all"),
              "sdr_cr_flash": ("exp23-historical-full-v1", "flash_image"),
              "sdr_cr_vl": ("exp23-historical-full-v1", "vl_image"),
              "sdr_cr_recovery_noimage": ("exp23-historical-full-v1", "flash_noimage"),
              "sdr_cr_vl_recovery_noimage": ("exp23-historical-full-v1", "vl_noimage")}
    for case in cases()[0]:
        run, dataset = Path(case["run"]), Path(case["dataset"])
        record, station = case["record"], case["record"]["station_id"]
        registered = read_json(run / "registration.json")
        methods = [method for method in registered["methods"] if method not in ("pair_focus", "pair_raw")]
        predictions = {method: read_json(run / "predictions" / method / station / "result.json") for method in methods}
        for method, result in predictions.items():
            parent = DERIVED_METHODS.get(method, method)
            result["evaluation_dependency_workdirs"] = [str(run / "predictions" / parent / station)]
        for method, (folder, name) in direct.items():
            result = read_json(Path("results") / folder / "predictions" / name / case["id"] / "result.json")
            base_mode = name if folder.startswith("exp22") else "flash_image"
            dependencies = [str(Path("runs/exp22-image-ablation-v1/predictions") / case["id"] / base_mode)]
            if folder.startswith(("exp23", "exp24")):
                recovery_mode = {"gap_flash_image": "flash_image", "gap_vl_image": "vl_image",
                    "gap_flash_noimage": "flash_noimage", "gap_vl_noimage": "vl_noimage"}.get(name, name)
                if recovery_mode in ("flash_image", "flash_noimage", "vl_image", "vl_noimage"):
                    dependencies.append(str(Path("runs/exp23-historical-recovery-v2/predictions") / case["id"] / recovery_mode))
            result["evaluation_dependency_workdirs"] = dependencies
            predictions[method] = result
        predictions["sdr_raw"] = predictions["pair_raw_repair"]
        observation = read_json(dataset / record["observation"])
        numeric = predictions["segment_numeric"]
        graph = recovery_graph(observation, numeric, numeric)
        predictions["numeric_gr"] = recover(observation, numeric, graph, automatic_response(observation, numeric, graph), True)
        predictions["numeric_gr"]["evaluation_dependency_workdirs"] = []
        # EXP22 first-stage image/noimage signatures are identical. Still build the
        # full noimage graph from its own output instead of assuming equality.
        noimage = predictions["sdr_flash_noimage"]
        own_graph = recovery_graph(observation, numeric, noimage)
        image_graph = recovery_graph(observation, numeric, predictions["sdr_flash"])
        if own_graph != image_graph:
            raise ValueError("historical full-noimage reuse needs the same actual candidate graph")
        result = noimage
        deps = list(noimage["evaluation_dependency_workdirs"])
        if own_graph["pairs"]:
            work = Path("runs/exp23-historical-recovery-v2/predictions") / case["id"] / "flash_noimage"
            result = recover(observation, noimage, own_graph, read_json(work / "result.json"), True)
            deps.append(str(work))
        predictions["sdr_gr_pipeline_noimage"] = {**result, "evaluation_dependency_workdirs": deps}
        entries.append({**case, "dataset": dataset, "predictions": predictions})
    return entries, {"cohort": "historical", "stations": len(entries), "split": "previously observed test",
                     "interpretation": "Post-hoc historical analysis, not a new held-out test"}


def score_record(observation: dict, truth: dict, result: dict):
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    result = copy.deepcopy(result)
    failed = result.get("status") != "success"
    # Author partial successes retain their actual union. A failed typed visual
    # request has no validated prediction and is scored as an empty detector.
    if "generic_mask" not in result:
        result.update(prediction={"events": []}, generic_mask=[False] * len(observed), generic_score=None)
    if failed and result.get("prediction") is not None:
        result.update(prediction={"events": []}, generic_mask=[False] * len(observed), generic_score=None)
    original = evaluate_record(truth, result, observed, observation["background_end"])
    ranges = evaluate(observation, truth, result) if result.get("prediction") is not None else None
    start = observation["background_end"] + 1
    reference = Prediction.model_validate({"events": truth["events"]})
    target_masks = event_masks(reference, observed)
    generic_target = np.any(np.column_stack(list(target_masks.values())), axis=1)
    generic_target[:start] = False
    generic_pred = np.asarray(result["generic_mask"], dtype=bool) & observed
    generic_pred[:start] = False
    generic_aff = affiliation(generic_target, generic_pred, start)
    activity = None
    if result.get("prediction") is not None:
        predicted = event_masks(Prediction.model_validate(result["prediction"]), observed)["deformation"]
        activity_events = [event for event in truth["events"] if event["kind"] == "deformation"]
        activity = {"activity_stations": int(bool(activity_events)), "observed_activity_days": 0,
                    "missed_activity_days": 0, "false_activity_days_before_onset": 0,
                    "false_activity_days_after_arrest": 0}
        if activity_events:
            first = min(event["start"] for event in activity_events)
            last = max(event["end"] for event in activity_events)
            activity.update(observed_activity_days=int(target_masks["deformation"].sum()),
                missed_activity_days=int((target_masks["deformation"] & ~predicted).sum()),
                false_activity_days_before_onset=int(predicted[start:first].sum()),
                false_activity_days_after_arrest=int(predicted[last + 1:].sum()))
    zero_sensitivity = None
    zero_affiliation = generic_aff
    if failed and result.get("prediction") is None:
        zero = {**result, "generic_mask": [False] * len(observed), "generic_score": None}
        zero_sensitivity = evaluate_record(truth, zero, observed, observation["background_end"])
        zero_affiliation = affiliation(generic_target, np.zeros(len(observed), dtype=bool), start)
    return {"original": original, "range": ranges, "generic_affiliation": generic_aff,
            "status": result.get("status"), "activity": activity, "failed_station_zero_sensitivity": zero_sensitivity,
            "failed_station_zero_affiliation": zero_affiliation}


def aggregate_rows(rows):
    original = daily_aggregate([row["original"] for row in rows])
    original["failed_records"] = sum(row["status"] != "success" for row in rows)
    ranges = range_aggregate([{"metrics": row["range"]} for row in rows]) if rows[0]["range"] is not None else None
    generic_aff = aggregate_affiliation([{"affiliation": row["generic_affiliation"]} for row in rows], "affiliation")
    activity_rows = [row["activity"] for row in rows if row["activity"] is not None]
    activity = {key: sum(row[key] for row in activity_rows) for key in activity_rows[0]} if activity_rows else None
    sensitivity = None
    if any(row["failed_station_zero_sensitivity"] is not None for row in rows):
        sensitivity = daily_aggregate([row["failed_station_zero_sensitivity"] or row["original"] for row in rows])
    return {"original": original, "range": ranges, "generic_affiliation": generic_aff, "activity": activity,
            "generic_average_precision_station_mean": mean_or_none(row["original"]["generic_daily"].get("average_precision") for row in rows
                if row["original"]["generic_daily"].get("average_precision") is not None),
            "failed_station_zero_sensitivity": sensitivity,
            "failed_station_zero_affiliation": aggregate_affiliation(
                [{"affiliation": row.get("failed_station_zero_affiliation", row["generic_affiliation"])} for row in rows], "affiliation")}


def request_receipts(paths):
    receipts = {}
    for directory in paths:
        work = Path(directory)
        for request in work.rglob("request.json") if work.exists() else []:
            if (request.parent / "receipt.json").exists():
                result_path = request.parent / "receipt.json"
            else:
                result_path = request.parent / "result.json"
            if not result_path.exists():
                raise ValueError(f"completed request lacks receipt: {request}")
            result = read_json(result_path)
            if not result.get("request_count"):
                continue
            receipts[str(request.resolve())] = {"request_path": str(request), "receipt_path": str(result_path),
                "request_sha256": sha256(request), "receipt_sha256": sha256(result_path),
                "status": result["status"], "http_status": result.get("http_status"),
                "error_type": result.get("error_type"),
                "requested_model": result.get("requested_model"), "served_model": result.get("served_model"),
                "usage": result.get("usage"), "elapsed_seconds": result.get("elapsed_seconds"),
                "retry_count": result.get("retry_count", 0)}
    return receipts


def aggregate_cost(receipts):
    rows = list(receipts.values())
    return {"requests": len(rows), "failed_requests": sum(row["status"] != "success" for row in rows),
            "http_200": sum(row["http_status"] == 200 for row in rows),
            "output_validation_failed": sum(row["status"] != "success" and row["http_status"] == 200 for row in rows),
            "transport_or_http_failed": sum(row["status"] != "success" and row["http_status"] != 200 for row in rows),
            "retry_count": sum(row["retry_count"] for row in rows),
            "prompt_tokens": sum((row["usage"] or {}).get("prompt_tokens", 0) for row in rows),
            "completion_tokens": sum((row["usage"] or {}).get("completion_tokens", 0) for row in rows),
            "missing_token_receipts": sum(not row["usage"] for row in rows),
            "median_request_seconds": float(np.median([row["elapsed_seconds"] for row in rows])) if rows else None}


def paired_changes(entries, rows):
    def signature(result):
        prediction = result.get("prediction")
        if prediction is None:
            return None
        return sorted((event["kind"], event["start"], event["end"], event["status"])
                      for event in prediction["events"])

    output = {}
    for control in ("segment_numeric", "numeric_gr", "sdr_flash", "sdr_gr_pipeline_noimage",
                    "sdr_gr_recovery_noimage", "sdr_raw"):
        changes = []
        for index, entry in enumerate(entries):
            primary = entry["predictions"][PRIMARY]
            secondary = entry["predictions"][control]
            if signature(primary) == signature(secondary):
                continue
            item = {"id": entry["id"], "scenario": rows[PRIMARY][index]["scenario"],
                    "primary_events": signature(primary), "control_events": signature(secondary)}
            for kind in TITLES:
                a = rows[PRIMARY][index]["range"][kind]["affiliation"]["f1"]
                b = rows[control][index]["range"][kind]["affiliation"]["f1"]
                item[kind + "_aff_f1_delta"] = a - b if a is not None and b is not None else None
            changes.append(item)
        output[control] = {"changed_stations": len(changes), "changes": changes}
    return output


def bootstrap(rows, control):
    if [row["id"] for row in rows] != [row["id"] for row in control]:
        raise ValueError("paired bootstrap requires the same ordered stations")
    indices = np.random.default_rng(2026101525).integers(0, len(rows), (2000, len(rows)))
    output = {}
    for kind in TITLES:
        a = np.array([row["range"][kind]["affiliation"]["f1"] if row["range"][kind]["affiliation"]["f1"] is not None else np.nan for row in rows])
        b = np.array([row["range"][kind]["affiliation"]["f1"] if row["range"][kind]["affiliation"]["f1"] is not None else np.nan for row in control])
        delta = a - b
        valid = np.isfinite(delta[indices]).sum(axis=1)
        samples = np.nansum(delta[indices], axis=1)[valid > 0] / valid[valid > 0]
        output[kind + "_affiliation_f1"] = {"difference": float(np.nanmean(delta)),
            "percentile_ci_95": np.quantile(samples, [.025, .975]).tolist(), "undefined_resamples": int((valid == 0).sum())}
    def count_matrix(reports):
        return np.array([[[row["original"]["types"][kind]["daily"][key] for key in ("tp", "fp", "fn")]
                          for kind in EVENT_TYPES] for row in reports])
    def score(counts):
        numerator = 2 * counts[..., 0]
        denominator = numerator + counts[..., 1] + counts[..., 2]
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0).mean(axis=-1)
    a, b = count_matrix(rows), count_matrix(control)
    samples = score(a[indices].sum(axis=1)) - score(b[indices].sum(axis=1))
    output["four_type_daily_f1"] = {"difference": float(score(a.sum(axis=0)) - score(b.sum(axis=0))),
                                    "percentile_ci_95": np.quantile(samples, [.025, .975]).tolist()}
    return {"repeats": 2000, "seed": 2026101525, "endpoints": output,
            "scope": "Descriptive station bootstrap; common fitted models; no significance acceptance gate"}


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def csv_file(target, name, rows):
    with (target / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tables(target, summary):
    methods = summary["methods"]
    lines = ["# EXP25：全面同站测评与消融", "", f"数据：{summary['source']}。", "",
             "主方案固定为sdr_gr_flash；Affiliation为范围主评价，IoU仅补充。全部站点包括失败，无point adjustment。", "",
             "## 完整四类与通用任务", "",
             "| 方法 | 四类日F1 | spike | step | transient | deformation | 通用P | 通用R | 通用F1 | 通用Aff-F1 | 正常站误报/站年 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    comparison, range_rows, point_rows, scenario_rows = [], [], [], []
    for method, report in methods.items():
        original = report["original"]
        typed = original.get("types")
        daily = original["generic_daily_micro"]
        values = [original.get("types_daily_macro_f1"), *[typed[kind]["daily_micro"]["f1"] if typed else None for kind in EVENT_TYPES],
                  daily["precision"], daily["recall"], daily["f1"], report["generic_affiliation"]["station_macro_f1"],
                  original["false_alarm_runs_per_normal_station_year_controls"]]
        lines.append("| " + method + " | " + " | ".join(fmt(value) for value in values) + " |")
        comparison.append({"method": method, "four_type_daily_f1": values[0],
            **{kind + "_daily_f1": values[i + 1] for i, kind in enumerate(EVENT_TYPES)},
            "generic_precision": daily["precision"], "generic_recall": daily["recall"], "generic_f1": daily["f1"],
            "generic_affiliation_precision": report["generic_affiliation"]["precision"],
            "generic_affiliation_recall": report["generic_affiliation"]["recall"],
            "generic_affiliation_f1": report["generic_affiliation"]["station_macro_f1"],
            "generic_AP_station_mean": report["generic_average_precision_station_mean"],
            "normal_station_false_alarm_runs_per_year": values[-1], "failed_stations": original["failed_records"]})
        if typed:
            for kind in ("spike", "step"):
                point_rows.append({"method": method, "kind": kind,
                    **{"strict_" + key: typed[kind]["events_micro"]["iou_0.5"][key] for key in ("precision", "recall", "f1", "tp", "fp", "fn")},
                    **{"tolerance3_" + key: typed[kind]["events_micro"]["tolerance_3_days_secondary"][key] for key in ("precision", "recall", "f1")}})
    for kind, title in TITLES.items():
        lines += ["", f"## {title}：完整范围指标", "",
                  "| 方法 | Aff-P | Aff-R | Aff-F1 | 日P | 日R | 日F1 | 覆盖 | FP日 | 负站FP日 | 事件F1 .3/.5 | 起/止MAE | 匹配/真值 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
        for method, report in methods.items():
            if report["range"] is None:
                continue
            r = report["range"][kind]
            a, daily, event = r["affiliation"], r["daily_micro"], r["events_micro"]["iou_0.5"]
            values = [a["precision"], a["recall"], a["station_macro_f1"], daily["precision"], daily["recall"], daily["f1"], r["truth_event_mean_coverage"]]
            lines.append("| " + method + " | " + " | ".join(fmt(value) for value in values)
                + f" | {r['false_positive_days']} | {r['negative_station_fp_days']} | {fmt(r['events_micro']['iou_0.3']['f1'])}/{fmt(event['f1'])}"
                + f" | {fmt(event['matched_start_mae_days'],2)}/{fmt(event['matched_end_mae_days'],2)} | {event['tp']}/{r['truth_events']} |")
            range_rows.append({"method": method, "kind": kind, "positive_stations": a["positive_stations"],
                "aff_precision": a["precision"], "aff_recall": a["recall"], "aff_f1": a["station_macro_f1"],
                "aff_harmonic_macro_pr": a["harmonic_of_macro_pr"],
                "aff_gap_spans_sensitivity_f1": r["affiliation_gap_spans_sensitivity"]["station_macro_f1"],
                **{"daily_" + key: value for key, value in daily.items()},
                "coverage": r["truth_event_mean_coverage"], "best_iou": r["truth_event_mean_best_iou"],
                "event_f1_03": r["events_micro"]["iou_0.3"]["f1"], "event_f1_05": event["f1"],
                "start_mae_days": event["matched_start_mae_days"], "end_mae_days": event["matched_end_mae_days"],
                "negative_station_fp_days": r["negative_station_fp_days"],
                "negative_station_fp_runs_per_year": r["negative_station_fp_runs_per_station_year"],
                "matched_events": event["tp"], "truth_events": r["truth_events"]})
    lines += ["", "## 消融：相对完整SDR-GR-Flash", "",
              "| 方法 | 暂态Aff-F1 | 与主方案差值 | 形变Aff-F1 | 两类等权Aff-F1 | 四类日F1 | 暂态FP日 |", "|---|---:|---:|---:|---:|---:|---:|"]
    ablations = []
    primary = methods[PRIMARY]["range"]["transient"]["affiliation"]["station_macro_f1"]
    for method in (PRIMARY, "segment_numeric", "numeric_gr", "visual_only", "sdr_flash", "sdr_raw", "sdr_cr_flash", "sdr_cr_vl",
                   "sdr_gr_vl", "sdr_gr_recovery_noimage", "sdr_gr_vl_recovery_noimage", "sdr_gr_pipeline_noimage",
                   "sdr_gr_numeric_recovery", "sdr_gr_candidate_all", "sdr_flash_noimage", "sdr_vl_noimage"):
        if method not in methods:
            continue
        m = methods[method]
        r = m["range"]
        f1 = r["transient"]["affiliation"]["station_macro_f1"]
        row = {"method": method, "transient_aff_f1": f1, "delta_to_primary": f1 - primary,
            "deformation_aff_f1": r["deformation"]["affiliation"]["station_macro_f1"],
            "range_types_aff_macro_f1": r["range_types_affiliation_macro_f1"],
            "four_type_daily_f1": m["original"].get("types_daily_macro_f1"), "transient_fp_days": r["transient"]["false_positive_days"]}
        ablations.append(row)
        lines.append("| " + method + " | " + " | ".join(fmt(value) for key, value in row.items() if key != "method") + " |")
    lines += ["", "## 形变漏日与停止后假活动", "", "| 方法 | 形变站 | 有效活动日 | 漏日 | 起点前假活动日 | 停止后假活动日 |", "|---|---:|---:|---:|---:|---:|"]
    for method, report in methods.items():
        if report["activity"] is not None:
            activity = report["activity"]
            lines.append("| " + method + " | " + " | ".join(str(activity[key]) for key in ("activity_stations", "observed_activity_days", "missed_activity_days", "false_activity_days_before_onset", "false_activity_days_after_arrest")) + " |")
    lines += ["", "## 按模拟场景分层", "", "| 场景 | 方法 | 站数 | 四类日F1 | 暂态Aff-F1 | 形变Aff-F1 | 暂态FP日 |", "|---|---|---:|---:|---:|---:|---:|"]
    for scenario, grouped in summary["scenarios"].items():
        for method in ("segment_numeric", "sdr_flash", PRIMARY, "sdr_gr_pipeline_noimage", "numeric_gr", "visual_only"):
            if method not in grouped:
                continue
            r = grouped[method]
            row = {"scenario": scenario, "method": method, "stations": r["original"]["stations"],
                "four_type_daily_f1": r["original"].get("types_daily_macro_f1"),
                "transient_aff_f1": r["range"]["transient"]["affiliation"]["station_macro_f1"],
                "deformation_aff_f1": r["range"]["deformation"]["affiliation"]["station_macro_f1"],
                "transient_fp_days": r["range"]["transient"]["false_positive_days"]}
            scenario_rows.append(row)
            lines.append("| " + scenario + " | " + method + " | " + str(row["stations"]) + " | "
                + " | ".join(fmt(row[key]) for key in ("four_type_daily_f1", "transient_aff_f1", "deformation_aff_f1"))
                + f" | {row['transient_fp_days']} |")
    lines += ["", "## 请求与依赖成本", "", "同一响应供多项消融复用；下表各行不可相加作为总调用数。", "",
              "| 方法 | 依赖请求 | 失败请求 | 输入token | 输出token | token回执缺失 | 中位秒 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, cost in summary["costs"].items():
        lines.append(f"| {method} | {cost['requests']} | {cost['failed_requests']} | {cost['prompt_tokens']} | {cost['completion_tokens']} | {cost['missing_token_receipts']} | {fmt(cost['median_request_seconds'],2)} |")
    lines += ["", f"去重后实际请求：{summary['request_total']}。", "",
              "完整点事件严格/±3日结果见point-events.csv；描述性bootstrap见bootstrap.json；作者失败整站置零敏感性及全部未四舍五入指标见summary.json。", "",
              "作者VisualTimeAnomaly和Isolation Forest仅报告通用任务，空白不是零分。AP仅数值连续评分。Affiliation汇总仅正样本站，负站误报另列。", ""]
    (target / "results.md").write_text("\n".join(lines), encoding="utf-8")
    csv_file(target, "comparison.csv", comparison)
    csv_file(target, "range-metrics.csv", range_rows)
    csv_file(target, "point-events.csv", point_rows)
    csv_file(target, "ablation.csv", ablations)
    csv_file(target, "scenarios.csv", scenario_rows)


def plot(target, summary):
    methods = summary["methods"]
    names = [method for method in MAIN if method in methods and methods[method]["range"] is not None]
    x = np.arange(len(names))
    figure, panels = plt.subplots(2, 2, figsize=(14, 9), dpi=170)
    for panel, kind in zip(panels[0], ("transient", "deformation"), strict=True):
        values = [methods[name]["range"][kind]["affiliation"]["station_macro_f1"] for name in names]
        panel.bar(x, values, color=["#2b927c" if name == PRIMARY else "#5785ac" for name in names])
        panel.set_title(kind + " Affiliation F1")
        panel.set_ylim(0, 1.13)
        for i, value in enumerate(values):
            panel.text(i, value + .015, f"{value:.3f}", ha="center", fontsize=8)
    for panel, key, label in ((panels[1, 0], "types_daily_macro_f1", "Four-type daily F1"),
                              (panels[1, 1], "generic_daily_micro", "Generic daily F1")):
        values = [methods[name]["original"][key] if key != "generic_daily_micro" else methods[name]["original"][key]["f1"] for name in names]
        panel.bar(x, values, color=["#2b927c" if name == PRIMARY else "#5785ac" for name in names])
        panel.set_title(label)
        panel.set_ylim(0, 1.13)
        for i, value in enumerate(values):
            panel.text(i, value + .015, f"{value:.3f}", ha="center", fontsize=8)
    for panel in panels.flat:
        panel.set_xticks(x, names, rotation=35, ha="right", fontsize=8)
        panel.grid(axis="y", alpha=.2)
    figure.suptitle(f"EXP25 / {summary['source']['cohort']} / {summary['source']['stations']} stations")
    figure.tight_layout()
    figure.savefig(target / "multi-metric-comparison.png")
    figure.savefig(target / "multi-metric-comparison.svg")
    plt.close(figure)
    names = [method for method in MAIN if method in methods]
    x = np.arange(len(names))
    figure, panels = plt.subplots(1, 2, figsize=(15, 5.8), dpi=170)
    for panel, endpoint in zip(panels, ("daily", "affiliation"), strict=True):
        values = [methods[name]["original"]["generic_daily_micro"]["f1"] if endpoint == "daily"
                  else methods[name]["generic_affiliation"]["station_macro_f1"] for name in names]
        panel.bar(x, values, color=["#2b927c" if name == PRIMARY else "#5785ac" for name in names])
        panel.set_title("Generic " + endpoint + " F1 (all anomaly types)")
        panel.set_ylim(0, 1.13)
        panel.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
        panel.grid(axis="y", alpha=.2)
        for i, value in enumerate(values):
            panel.text(i, value + .015, f"{value:.3f}", ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(target / "generic-comparison.png")
    figure.savefig(target / "generic-comparison.svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("test", "historical"), default="test")
    parser.add_argument("--visual", type=Path, default=Path("runs/exp25-test-1516-visual-v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    entries, source = load_historical() if args.cohort == "historical" else load_test(args.visual, args.pilot)
    args.out.mkdir(parents=True, exist_ok=False)
    rows, paths = {}, {}
    expected_methods = set(entries[0]["predictions"])
    for entry in entries:
        if set(entry["predictions"]) != expected_methods:
            raise ValueError("all methods must have all the same stations")
        observation = read_json(entry["dataset"] / entry["record"]["observation"])
        truth = read_json(entry["dataset"] / entry["record"]["truth"])
        for method, result in entry["predictions"].items():
            rows.setdefault(method, []).append({"id": entry["id"], "station": entry["record"]["station_id"],
                "seed": entry["seed"], "scenario": truth["scenario"], **score_record(observation, truth, result)})
            paths.setdefault(method, set()).update(result.get("evaluation_dependency_workdirs", []))
    methods = {method: aggregate_rows(values) for method, values in rows.items()}
    scenarios = {scenario: {method: aggregate_rows([row for row in values if row["scenario"] == scenario]) for method, values in rows.items()}
                 for scenario in sorted({row["scenario"] for row in next(iter(rows.values()))})}
    receipt_sets = {method: request_receipts(directories) for method, directories in paths.items()}
    all_receipts = {key: value for receipts in receipt_sets.values() for key, value in receipts.items()}
    if args.cohort == "test":
        # Include diagnostic calls even when no published derived method uses them.
        selected = {entry["record"]["station_id"] for entry in entries}
        directories = [str(station) for mode in (args.visual / "predictions").iterdir()
                       if mode.is_dir() for station in mode.iterdir() if station.name in selected]
        all_receipts.update(request_receipts(directories))
    costs = {method: aggregate_cost(receipts) for method, receipts in receipt_sets.items()}
    comparisons = {control: bootstrap(rows[PRIMARY], rows[control]) for control in
                   ("segment_numeric", "numeric_gr", "sdr_flash", "sdr_gr_pipeline_noimage", "sdr_gr_recovery_noimage")}
    summary = {"experiment_id": "GNSS-VLM-EXP25", "primary_method": PRIMARY, "source": source,
        "methods": methods, "scenarios": scenarios, "costs": costs, "request_total": aggregate_cost(all_receipts),
        "protocol_sha256": sha256(Path("docs/experiment-exp25.md")), "report_script_sha256": sha256(Path(__file__)),
        "metric_sources": {"typed": sha256(Path("src/gnss_research/metrics.py")), "range": sha256(Path("scripts/exp20_range_metrics.py"))}}
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "per-station.json", rows)
    write_json(args.out / "bootstrap.json", comparisons)
    write_json(args.out / "paired-changes.json", paired_changes(entries, rows))
    write_json(args.out / "request-receipts.json", list(all_receipts.values()))
    frozen = args.out / "frozen-source"
    frozen.mkdir()
    for path in (Path(__file__), Path("scripts/experiment_exp25.py"), Path("scripts/exp20_range_metrics.py"), Path("docs/experiment-exp25.md")):
        shutil.copyfile(path, frozen / path.name)
    for receipt in all_receipts.values():
        work = Path(receipt["receipt_path"]).parent
        key = sha256(Path(receipt["request_path"]))[:16]
        dest = args.out / "responses" / key
        dest.mkdir(exist_ok=True, parents=True)
        for name in ("response.json", Path(receipt["receipt_path"]).name):
            if (work / name).exists() and not (dest / name).exists():
                shutil.copyfile(work / name, dest / name)
    tables(args.out, summary)
    plot(args.out, summary)
    print("cohort", source, "methods", len(methods), "actual requests", summary["request_total"], flush=True)


if __name__ == "__main__":
    main()
