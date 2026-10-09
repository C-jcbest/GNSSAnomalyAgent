"""Run reference-activity stage diagnostics and preserve per-record error evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_landslide_comparison import (
    DEFAULT_EVIDENCE,
    ROOT,
    ActivityPrediction,
    LandslideDiagnostics,
    LandslideInput,
    aggregate_scores,
    evaluate_case,
    load_labels,
    observation_features,
    plot_stages,
    rule_stages,
    save_json,
    sha,
)
from run_landslide_visual_comparison import inclusive_spans, infer_stages
from xgboost import XGBClassifier

from gnss_sim.landslide_error_audit import (
    activity_disagreements,
    raw_interval_evidence,
    regular_windows,
    stage_error_partition,
)
from gnss_sim.landslide_visual import ACTIVITY_TASK, EVOLUTION_TASK, VisualRequests


def restore_prediction(row):
    return ActivityPrediction(np.array(row["activity"]), np.array(row["stage"]),
                              row["status"], row["gated"], row["stage_status"])


def render_review_atlas(case, feature, out):
    """Every calendar window appears, without labels or generator information."""
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    windows = regular_windows(len(values))
    paths = []
    for page in range(0, len(windows), 4):
        selected = windows[page:page + 4]
        fig, panels = plt.subplots(3, len(selected), figsize=(4.8 * len(selected), 7), squeeze=False)
        for column, (start, stop) in enumerate(selected):
            for axis, axis_name in enumerate(("N", "E", "U")):
                panel = panels[axis, column]
                series = values[start:stop, axis]
                panel.plot(np.arange(start, stop), series, color=("#147d72", "#cc6d4b", "#a38a32")[axis], linewidth=.8)
                finite = series[np.isfinite(series)]
                if len(finite):
                    low, high = float(finite.min()), float(finite.max())
                    center = (low + high) / 2
                    span = max(high - low, 6 * feature.noise[axis], 10)
                    panel.set_ylim(center - .75 * span, center + .75 * span)
                panel.set_xlim(start, stop - 1)
                panel.set_ylabel(axis_name + " (mm)")
                panel.grid(alpha=.2)
                if axis == 0:
                    panel.set_title(f"Days {start}..{stop - 1}")
                if axis == 2:
                    panel.set_xlabel("Original day index")
        fig.suptitle(case.case_id + " | Raw observations, 180-day windows, 90-day stride | NO labels")
        fig.tight_layout()
        path = out / f"{case.case_id}-local-{page // 4 + 1}.png"
        fig.savefig(path, dpi=135)
        plt.close(fig)
        paths.append(path)
    return paths


def write_html(out, summaries, case_ids, audit):
    display = []
    for name, result in summaries.items():
        score = result["aggregate"]
        display.append({"阶段器": name, "阶段macro-F1": score["stage_macro_f1"],
                        "加速F1": score["stages"]["acceleration"]["f1"],
                        "匀速F1": score["stages"]["steady_motion"]["f1"],
                        "减速F1": score["stages"]["deceleration"]["f1"],
                        "阶段失败记录": score["stage_failed_cases"]})
    headers = "".join(f"<th>{html.escape(key)}</th>" for key in display[0])
    body = ""
    for row in display:
        body += "<tr>" + "".join(f"<td>{value:.3f}</td>" if isinstance(value, float)
                                  else f"<td>{html.escape(str(value))}</td>" for value in row.values()) + "</tr>"
    cards = []
    for cid in case_ids:
        rows = []
        for name in ("robust_rule", "robust_xgboost_stage", "visual_model"):
            entry = audit[cid][name]
            activity = entry["activity"]
            counts = entry["stages"]["counts"]
            rows.append(f"<tr><td>{name}</td><td>{activity['false_activity']['days']}</td>"
                        f"<td>{activity['missed_activity']['days']}</td><td>{counts['stage_wrong_class']}</td>"
                        f"<td>{counts['stage_abstention']}</td><td>{counts['stage_output_failure']}</td></tr>")
        images = "".join(f'<img loading="lazy" src="{path.name}" alt="{cid} 局部原始位移">'
                         for path in sorted(out.glob(f"{cid}-local-*.png")))
        cards.append(f'<section><h2>{cid}</h2><table><tr><th>既有配置</th><th>活动误报日</th>'
                     '<th>活动漏检日</th><th>范围内错分阶段日</th><th>范围内阶段弃判日</th><th>阶段输出失败影响日</th></tr>'
                     + "".join(rows) + '</table>'
                     f'<img src="{cid}-reference-stages.png" alt="参考活动范围阶段对比">'
                     f'<details><summary>展开所有固定局部窗口（无标签）</summary>{images}</details></section>')
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>活动范围与阶段错误诊断</title>
<style>body{{font:15px system-ui;margin:32px;color:#233d36;background:#f8faf9}}p{{max-width:1100px;line-height:1.8}}table{{border-collapse:collapse;background:white}}th,td{{padding:10px;border:1px solid #cad6d0}}img{{width:100%;max-width:1800px}}section{{margin-top:40px}}summary{{cursor:pointer;padding:14px}}</style>
<h1>参考活动范围下的阶段诊断</h1><p>本页向三个阶段器提供同一份 AI 开发标注活动范围，目的是区分定位瓶颈与阶段判别瓶颈。
参考范围不是专家金标准；本诊断使用了评价活动标签，不能加入可部署方法排名，活动 F1 不作为成绩。
阶段标签未提供给分类器。局部图仅用于本轮审查，模型诊断仍复用上一轮全局原图与辅助图。全部是已查看仿真记录。</p>
<table><tr>{headers}</tr>{body}</table><p>灰色表示阶段未知、过渡或缺测；黄色只表示参考缓慢位移，不能自动当成匀速。
下方错误计数来自原有端到端输出；新增参考范围诊断另列，不修改原标注和预测。</p>{''.join(cards)}</html>'''
    (out / "index.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    numeric_manifest = json.loads((args.numeric / "manifest.json").read_text(encoding="utf-8"))
    visual_manifest = json.loads((args.visual / "visual-manifest.json").read_text(encoding="utf-8"))
    if visual_manifest["numeric_manifest_sha256"] != sha(args.numeric / "manifest.json"):
        raise ValueError("Prior runs do not share the same frozen numeric data/configuration")
    case_ids = numeric_manifest["split"]["development_evaluation"]
    cases = {path.parent.name: LandslideInput.model_validate_json(path.read_text(encoding="utf-8"))
             for path in (Path(numeric_manifest["data"]) / "cases").glob("*/input.json")}
    label_path = Path(numeric_manifest["labels"])
    if sha(label_path) != numeric_manifest["labels_sha256"]:
        raise ValueError("Reference annotation has changed")
    labels = load_labels(label_path, cases)
    previous = json.loads((args.visual / "predictions.json").read_text(encoding="utf-8"))
    choices = json.loads((args.numeric / "frozen-selection.json").read_text(encoding="utf-8"))
    model = XGBClassifier()
    model.load_model(args.numeric / "stage-xgboost.json")
    reference_activity = {cid: np.array([int(row["activity_label"]) for row in labels[cid]]) for cid in case_ids}
    save_json(args.out / "reference-activity-input.json", {cid: activity.tolist() for cid, activity in reference_activity.items()})
    run = {"status": "running", "purpose": "REFERENCE ACTIVITY DIAGNOSTIC ONLY; not deployable or independent evaluation",
           "stage_labels_supplied_to_models": False, "activity_labels_supplied_to_models": True,
           "case_ids": case_ids, "model": visual_manifest["model"], "budget": len(case_ids),
           "numeric_manifest_sha256": sha(args.numeric / "manifest.json"),
           "prior_predictions_sha256": sha(args.visual / "predictions.json"),
           "reference_labels_sha256": sha(label_path), "stage_model_sha256": sha(args.numeric / "stage-xgboost.json"),
           "reference_gate_sha256": sha(args.out / "reference-activity-input.json"),
           "selection_sha256": sha(args.numeric / "frozen-selection.json"),
           "prompt_policy": "Same stage prompt and same raw/auxiliary images as visual-v3; only activity gate changes"}
    paths = [Path(__file__), ROOT / "src/gnss_sim/landslide_error_audit.py",
             ROOT / "src/gnss_sim/landslide_evaluation.py", ROOT / "src/gnss_sim/landslide_baselines.py",
             ROOT / "src/gnss_sim/landslide_visual.py", ROOT / "src/gnss_sim/landslide_diagnostics.py",
             ROOT / "scripts/run_landslide_visual_comparison.py", ROOT / "scripts/run_landslide_comparison.py"]
    run["source_sha256"] = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = args.out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        run["source_sha256"][str(relative)] = sha(path)
    save_json(args.out / "manifest.json", run)
    requests = VisualRequests(args.out / "requests", ROOT / ".env", visual_manifest["model"], len(case_ids))
    methods = {name: {} for name in ("reference_rule", "reference_xgboost", "reference_visual_model")}
    errors, failures, evidence = {}, {}, {}
    for cid in case_ids:
        case = cases[cid]
        input_path = Path(numeric_manifest["data"]) / "cases" / cid / "input.json"
        if sha(input_path) != numeric_manifest["input_sha256"][cid]:
            raise ValueError("Observation input changed")
        motions = {window: LandslideDiagnostics.model_validate_json(
            (DEFAULT_EVIDENCE / cid / f"motion-{window}.json").read_text(encoding="utf-8")) for window in (31, 61, 91)}
        canonical = hashlib.sha256(case.model_dump_json().encode()).hexdigest()
        if any(motion.input_sha256 != canonical or motion.case_id != cid or motion.window_days != window
               for window, motion in motions.items()):
            raise ValueError("Observation auxiliary cache mismatch")
        feature = observation_features(case, motions)
        errors[cid] = {}
        for name, predictions in previous.items():
            prediction = restore_prediction(predictions[cid])
            errors[cid][name] = {"activity": activity_disagreements(labels[cid], prediction),
                                 "stages": stage_error_partition(labels[cid], prediction)}
        values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        evidence[cid] = [raw_interval_evidence(values, start, stop) for start, stop in regular_windows(len(values))]
        render_review_atlas(case, feature, args.out)
        activity = reference_activity[cid]
        methods["reference_rule"][cid] = rule_stages(feature, activity, choices["rule_stage"]["relative_change_60d"])
        stage = model.predict(feature.matrix).astype(int) + 1
        stage[(activity != 1) | ~feature.supported] = -1
        stage[activity == 0] = 0
        methods["reference_xgboost"][cid] = ActivityPrediction(activity.copy(), stage)
        raw = args.visual / f"{cid}-raw.png"
        auxiliary = args.visual / f"{cid}-auxiliary.png"
        prior_receipt = json.loads((args.visual / "requests" / f"{cid}-visual_model.started.json").read_text(encoding="utf-8"))
        if any(sha(path) != prior_receipt["image_sha256"][path.name] for path in (raw, auxiliary)):
            raise ValueError("Prior model image was modified")
        task = ACTIVITY_TASK.format(last=len(case.dates) - 1)
        prompt = task + EVOLUTION_TASK + "Only label stages inside the supplied confirmed activity intervals. Do not extend, revise or invent activity."
        prompt += "\nConfirmed inclusive activity intervals: " + json.dumps(inclusive_spans(activity))
        prompt += '\nReturn exactly {"stages":[[start,end,"A"],[start,end,"S"],[start,end,"D"]]} with actual intervals; empty lists are valid. No other keys.'
        marker = "\nConfirmed inclusive activity intervals: "
        before, after = prior_receipt["prompt"].split(marker)
        _, suffix = after.split("\n", 1)
        expected_prompt = before + marker + json.dumps(inclusive_spans(activity)) + "\n" + suffix
        if expected_prompt != prompt:
            raise ValueError("Stage diagnostic changed more than the supplied activity intervals")
        if np.any(activity == 1):
            methods["reference_visual_model"][cid] = infer_stages(requests, cid + "-reference-stage", activity,
                                                                  feature.observed, prompt, [raw, auxiliary], failures)
        else:
            empty_stage = np.where(activity == 0, 0, -1)
            methods["reference_visual_model"][cid] = ActivityPrediction(activity.copy(), empty_stage)
        save_json(args.out / "predictions.json", {name: {key: value.to_dict() for key, value in group.items()}
                                                  for name, group in methods.items()})
        save_json(args.out / "failures.json", failures)
        print(f"{cid}: reference-gate diagnostics complete; failed stage outputs={len(failures)}", flush=True)
    summaries = {}
    for name, predictions in methods.items():
        per_case = {cid: evaluate_case(labels[cid], prediction) for cid, prediction in predictions.items()}
        summaries[name] = {"aggregate": aggregate_scores(list(per_case.values())), "per_case": per_case}
    save_json(args.out / "reference-stage-results.json", summaries)
    save_json(args.out / "existing-error-audit.json", errors)
    save_json(args.out / "raw-window-evidence.json", evidence)
    for cid in case_ids:
        plot_stages(cases[cid], labels[cid], {name: predictions[cid] for name, predictions in methods.items()},
                    args.out / f"{cid}-reference-stages.png")
    with (args.out / "error-partition.csv").open("w", encoding="utf-8-sig", newline="") as output:
        rows = [{"case_id": cid, "method": name, **value["stages"]["counts"]}
                for cid, groups in errors.items() for name, value in groups.items()]
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_html(args.out, summaries, case_ids, errors)
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in (args.out / "requests").glob("*.response.json")]
    run.update(status="complete", seconds=time.monotonic() - started, calls=len(receipts), failures=len(failures),
               total_tokens=sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts))
    save_json(args.out / "manifest.json", run)
    print(json.dumps({name: result["aggregate"]["stage_macro_f1"] for name, result in summaries.items()}), flush=True)


if __name__ == "__main__":
    main()
