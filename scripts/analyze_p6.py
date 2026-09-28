"""Post-hoc diagnosis of frozen outputs; no detector, API call or score replacement."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from gnss_sim.evaluation import (
    AXES,
    DAYS,
    POINT_TYPES,
    RANGE_TYPES,
    _point_case,
    _range_case,
    load_results_jsonl,
)
from gnss_sim.evaluation_views import matched_count, point_scores, range_days
from gnss_sim.pilot import verify_pilot
from gnss_sim.schemas import CaseTruth, PointResult, RangeResult

ROOT = Path(__file__).resolve().parents[1]


def summarize(truths: dict, results: dict, task: str) -> dict:
    tp = fp = fn = negative = alarms = success = 0
    aff, ious, coverage = [], [], []
    tolerance = {k: Counter() for k in (0, 1, 3, 7)}
    negative_groups = defaultdict(Counter)
    failed_positive_axes = failed_positive_days = 0
    for case_id, truth in truths.items():
        result = results[case_id]
        valid = result.status == "success"
        success += valid
        if task == "point":
            a, b, c, n, f = _point_case(truth, result)
            tp, fp, fn = tp + a, fp + b, fn + c
        else:
            scores, n, f = _range_case(truth, result)
            aff.extend(scores)
        negative += n
        alarms += f
        for axis in AXES:
            events = [event for event in truth.events if event.axis == axis]
            wanted = POINT_TYPES if task == "point" else RANGE_TYPES
            selected = [event for event in events if event.type in wanted]
            gt = ({event.start_index for event in selected} if task == "point" else
                  range_days((event.start_index, event.end_index) for event in selected))
            values = getattr(result.predictions, axis) if valid else []
            pred = set(values) if task == "point" else range_days(values)
            if not valid and gt:
                failed_positive_axes += 1
                failed_positive_days += len(gt)
            if not gt:
                group = "other_task_only" if events else "no_event_on_axis"
                negative_groups[group]["valid_axes"] += valid
                negative_groups[group]["alarm_axes"] += valid and bool(pred)
                negative_groups[group]["failed_axes"] += not valid
            if task == "point":
                for k, counts in tolerance.items():
                    hits = matched_count(gt, pred, k)
                    counts.update(tp=hits, fp=len(pred) - hits, fn=len(gt) - hits)
            elif gt:
                ious.append(len(gt & pred) / len(gt | pred))
                coverage.append(len(pred) / len(gt))
    output = {"cases": len(truths), "success": success,
              "execution_success_rate": success / len(truths),
              "negative_axes": negative, "false_alarm_axes": alarms,
              "far": alarms / negative if negative else None,
              "failed_positive_axes": failed_positive_axes,
              "failed_positive_days": failed_positive_days,
              "negative_axis_groups": {
                  key: dict(counts, far=counts["alarm_axes"] / counts["valid_axes"]
                            if counts["valid_axes"] else None)
                  for key, counts in sorted(negative_groups.items())}}
    if task == "point":
        output.update(point_scores(tp, fp, fn))
        output["tolerance_diagnostic"] = {
            str(k): point_scores(**counts) for k, counts in tolerance.items()}
        output["perfect_failed_cases_only_upper_bound"] = point_scores(
            tp + failed_positive_days, fp, fn - failed_positive_days)
    else:
        output.update({"positive_axes": len(aff),
                       "affiliation_precision": sum(x[0] for x in aff) / len(aff) if aff else 0,
                       "affiliation_recall": sum(x[1] for x in aff) / len(aff) if aff else 0,
                       "affiliation_f1": sum(x[2] for x in aff) / len(aff) if aff else 0,
                       "mean_daily_iou_positive_axes": sum(ious) / len(ious) if ious else None,
                       "mean_predicted_to_true_days_positive_axes":
                           sum(coverage) / len(coverage) if coverage else None})
    return output


def union_results(numerical: dict, visual: dict, task: str) -> dict:
    """Replay union; failed visual contributes nothing, numerical must succeed."""
    model = PointResult if task == "point" else RangeResult
    output = {}
    for case_id, num in numerical.items():
        if num.status != "success":
            raise ValueError("This replay requires successful frozen numerical results")
        vis = visual[case_id]
        predictions = {}
        for axis in AXES:
            left = getattr(num.predictions, axis)
            right = getattr(vis.predictions, axis) if vis.status == "success" else []
            predictions[axis] = sorted(set(left + right))
        output[case_id] = model(case_id=case_id, method="union_replay_diagnostic",
                               status="success", predictions=predictions)
    return output


def main() -> None:
    pilot = ROOT / "data/pilots/pilot-v1"
    manifest = verify_pilot(pilot)
    entries = {entry["case_id"]: entry for entry in manifest["cases"]}
    truths = {key: CaseTruth.model_validate_json(
        (pilot / "cases" / key / "truth.json").read_bytes()) for key in entries}
    paths = {"sr": ROOT / "runs/p5/sr/predictions.jsonl",
             "pelt": ROOT / "runs/p5/pelt/predictions.jsonl",
             "theilsen": ROOT / "runs/p5/theilsen/predictions.jsonl",
             "visual_point": ROOT / "runs/p6/qwen3.8-flash/point/predictions.jsonl",
             "visual_range": ROOT / "runs/p6/qwen3.8-flash/range/predictions.jsonl"}
    frozen = json.loads((ROOT / "configs/p6-frozen.json").read_bytes())
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    for task in ("point", "range"):
        if hashes[f"visual_{task}"] != frozen["predictions_sha256"][task]:
            raise ValueError("P6 predictions do not match frozen hashes")
    results = {}
    for name, path in paths.items():
        task = "point" if name in ("sr", "pelt", "visual_point") else "range"
        rows, errors = load_results_jsonl(path, task)
        if errors or len(rows) != len(truths) or {row.case_id for row in rows} != truths.keys():
            raise ValueError(f"Incomplete or duplicate frozen results: {name}")
        results[name] = {row.case_id: row for row in rows}
    results["union_point"] = union_results(results["sr"], results["visual_point"], "point")
    results["union_range"] = union_results(results["theilsen"], results["visual_range"], "range")
    results["all_positive_range"] = {
        key: RangeResult(case_id=key, method="all_positive_diagnostic", status="success",
                         predictions={axis: [(0, DAYS - 1)] for axis in AXES}) for key in truths}
    summary = {"protocol": "p6-analysis-v1", "scope": "posthoc_development_diagnostic_only",
               "source_sha256": hashes, "methods": {}, "single_type": {}}
    for name, rows in results.items():
        task = "point" if name in ("sr", "pelt", "visual_point", "union_point") else "range"
        summary["methods"][name] = summarize(truths, rows, task)
        summary["single_type"][name] = {
            kind: summarize({key: truth for key, truth in truths.items()
                             if entries[key]["case_type"] == kind}, rows, task)
            for kind in ("normal", "spike", "step", "slow_trend", "acceleration",
                         "transient_shift")}
        if name in ("sr", "pelt", "theilsen", "visual_point", "visual_range"):
            report = json.loads(paths[name].with_name("report.json").read_bytes())
            for metric, value in report.items():
                if metric != "task" and summary["methods"][name][metric] != value:
                    raise ValueError(f"Cannot reproduce frozen report: {name}/{metric}")
    failures = {}
    for task in ("point", "range"):
        raw_path = paths[f"visual_{task}"].with_name("raw.jsonl")
        raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        if len(raw) != 300 or {row["case_id"] for row in raw} != truths.keys():
            raise ValueError("Raw response coverage differs from frozen pilot")
        counts = Counter()
        for row in raw:
            if row["status"] != "success":
                try:
                    root_type = type(json.loads(row["response"])).__name__
                except (ValueError, TypeError):
                    root_type = "invalid_json"
                counts[root_type] += 1
        failures[task] = dict(counts)
    summary["failed_response_root_types"] = failures
    single_step = Counter()
    for key, entry in entries.items():
        if entry["case_type"] != "step":
            continue
        row = results["visual_range"][key]
        single_step["cases"] += 1
        if row.status == "success":
            single_step["valid_cases"] += 1
            axis = truths[key].events[0].axis
            intervals = getattr(row.predictions, axis)
            single_step["step_axis_range_alarm"] += bool(intervals)
            single_step["step_axis_range_reaches_last_day"] += any(
                end == DAYS - 1 for _, end in intervals)
    summary["visual_range_single_step_diagnostic"] = dict(single_step)
    output = ROOT / "runs/p6-analysis-v1/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"methods": summary["methods"], "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
