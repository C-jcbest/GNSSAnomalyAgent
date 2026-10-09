"""Freeze a grouped development split, calibrate baselines, then evaluate once."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_baselines import (  # noqa: E402
    activity_from_mask,
    observation_features,
    pelt_activity,
    pelt_segment_rates,
    robust_activity,
    rule_stages,
)
from gnss_sim.landslide_diagnostics import LandslideDiagnostics, derive_motion  # noqa: E402
from gnss_sim.landslide_evaluation import (  # noqa: E402
    STAGES,
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    runs,
)

DEFAULT_DATA = ROOT / "data/generated/landslide-v1-20261007-075217-7b9f8f2c"
DEFAULT_LABELS = ROOT / "artifacts/landslide-annotations-2026-10-07/revision-v2/daily-labels.csv"
DEFAULT_EVIDENCE = ROOT / "artifacts/landslide-annotations-2026-10-07/evidence"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_labels(path, cases):
    grouped = {case_id: [] for case_id in cases}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            grouped[row["case_id"]].append(row)
    for case_id, case in cases.items():
        rows = grouped[case_id]
        if len(rows) != len(case.dates):
            raise ValueError("Incomplete labels")
        for index, row in enumerate(rows):
            if int(row["day_index"]) != index or row["date"] != str(case.dates[index]):
                raise ValueError("Annotation calendar mismatch")
            if int(row["activity_label"]) not in {-1, 0, 1}:
                raise ValueError("Invalid activity annotation")
            if row["feature"] in STAGES and int(row["activity_label"]) != 1:
                raise ValueError("A reference stage must lie inside confirmed activity")
            if case.displacement_mm[index] is None and int(row["activity_label"]) != -1:
                raise ValueError("Missing observations must not be scored")
    return grouped


def development_split(labels, seed):
    rng = np.random.default_rng(seed)
    quiet, active = [], []
    for case_id, rows in sorted(labels.items()):
        group = active if any(int(row["activity_label"]) == 1 for row in rows) else quiet
        group.append(case_id)
    if len(quiet) != 4 or len(active) != 20:
        raise ValueError("This frozen pilot expects 20 active and four negative-control records")
    rng.shuffle(quiet)
    rng.shuffle(active)
    return {"train": sorted(active[:10] + quiet[:2]),
            "validation": sorted(active[10:15] + quiet[2:3]),
            "development_evaluation": sorted(active[15:] + quiet[3:])}


def summarize(labels, predictions, case_ids):
    return aggregate_scores([evaluate_case(labels[cid], predictions[cid]) for cid in case_ids])


def calibration_objective(score):
    # Declared before selection; stage quality does not select the activity gate.
    return (score["daily"]["f1"] or 0) + (score["events"]["f1"] or 0)


def fit_baselines(features, labels, split, out):
    validation = split["validation"]
    trials = []
    choices = {}
    for rate in (0.03, 0.06, 0.12):
        for snr in (1.0, 2.0, 3.0):
            predictions = {cid: rule_stages(features[cid], robust_activity(features[cid], rate, snr))
                           for cid in validation}
            score = summarize(labels, predictions, validation)
            trials.append({"method": "robust", "rate": rate, "snr": snr,
                           "objective": calibration_objective(score), "validation": score})
    choices["robust"] = max(trials, key=lambda row: row["objective"])
    print("Robust calibration complete", flush=True)

    pelt_cache = {}
    for penalty in (10.0, 40.0, 160.0):
        pelt_cache[penalty] = {cid: pelt_segment_rates(features[cid], penalty) for cid in validation}
        for rate in (0.03, 0.06, 0.12):
            predictions = {cid: rule_stages(features[cid], pelt_activity(features[cid], pelt_cache[penalty][cid], rate))
                           for cid in validation}
            score = summarize(labels, predictions, validation)
            trials.append({"method": "pelt", "rate": rate, "penalty": penalty,
                           "objective": calibration_objective(score), "validation": score})
    choices["pelt"] = max((row for row in trials if row["method"] == "pelt"), key=lambda row: row["objective"])
    print("PELT calibration complete", flush=True)

    train = split["train"]
    matrix = np.concatenate([features[cid].matrix for cid in train])
    support = np.concatenate([features[cid].supported for cid in train])
    activity_labels = np.array([int(row["activity_label"]) for cid in train for row in labels[cid]])
    stage_labels = np.array([STAGES.get(row["feature"], -1) for cid in train for row in labels[cid]])
    models = {}
    for depth in (2, 4):
        model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=0.05,
                              subsample=1.0, colsample_bytree=1.0, n_jobs=2, random_state=20261007)
        eligible = support & (activity_labels >= 0)
        model.fit(matrix[eligible], activity_labels[eligible])
        models[depth] = model
        probabilities = {cid: model.predict_proba(features[cid].matrix)[:, 1] for cid in validation}
        for threshold in (0.35, 0.5, 0.65):
            predictions = {cid: rule_stages(features[cid], activity_from_mask(
                probabilities[cid] >= threshold, features[cid].supported)) for cid in validation}
            score = summarize(labels, predictions, validation)
            trials.append({"method": "xgboost", "depth": depth, "threshold": threshold,
                           "objective": calibration_objective(score), "validation": score})
    choices["xgboost"] = max((row for row in trials if row["method"] == "xgboost"), key=lambda row: row["objective"])
    activity_model = models[choices["xgboost"]["depth"]]
    stage_model = XGBClassifier(n_estimators=120, max_depth=2, learning_rate=0.05,
                                n_jobs=2, random_state=20261007, objective="multi:softprob", num_class=3)
    eligible = support & (stage_labels > 0)
    if set(stage_labels[eligible]) != {1, 2, 3}:
        raise ValueError("Training split must contain all three scored evolution classes")
    stage_model.fit(matrix[eligible], stage_labels[eligible] - 1)
    activity_model.save_model(out / "activity-xgboost.json")
    stage_model.save_model(out / "stage-xgboost.json")

    rule_trials = []
    robust = choices["robust"]
    for change in (0.25, 0.5, 1.0):
        predictions = {cid: rule_stages(features[cid], robust_activity(features[cid], robust["rate"], robust["snr"]), change)
                       for cid in validation}
        score = summarize(labels, predictions, validation)
        rule_trials.append({"relative_change_60d": change, "validation": score})
    choices["rule_stage"] = max(rule_trials, key=lambda row: row["validation"]["stage_macro_f1"] or 0)
    save_json(out / "validation-trials.json", {"activity_trials": trials, "stage_trials": rule_trials})
    save_json(out / "frozen-selection.json", choices)
    return choices, activity_model, stage_model


def predict_baselines(features, case_ids, choices, activity_model, stage_model):
    methods = {name: {} for name in ("robust_rule", "pelt_rule", "xgboost", "robust_xgboost_stage", "ungated_rule")}
    robust, pelt, learned = (choices[key] for key in ("robust", "pelt", "xgboost"))
    change = choices["rule_stage"]["relative_change_60d"]
    for cid in case_ids:
        feature = features[cid]
        activity = robust_activity(feature, robust["rate"], robust["snr"])
        methods["robust_rule"][cid] = rule_stages(feature, activity, change)
        methods["ungated_rule"][cid] = rule_stages(feature, activity, change, gated=False)
        rates = pelt_segment_rates(feature, pelt["penalty"])
        methods["pelt_rule"][cid] = rule_stages(feature, pelt_activity(feature, rates, pelt["rate"]), change)
        probability = activity_model.predict_proba(feature.matrix)[:, 1]
        learned_activity = activity_from_mask(probability >= learned["threshold"], feature.supported)
        learned_stage = stage_model.predict(feature.matrix).astype(int) + 1
        for name, gate in (("xgboost", learned_activity), ("robust_xgboost_stage", activity)):
            stage = learned_stage.copy()
            stage[gate != 1] = -1
            stage[gate == 0] = 0
            methods[name][cid] = ActivityPrediction(gate.copy(), stage)
    return methods


def plot_case(case, rows, predictions, path):
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    fig, panels = plt.subplots(2 + len(predictions), 1, figsize=(14, 3.5 + .55 * len(predictions)),
                               sharex=True, gridspec_kw={"height_ratios": [5] + [1] * (1 + len(predictions))})
    for axis, name in enumerate(("N", "E", "U")):
        panels[0].plot(values[:, axis], linewidth=.7, alpha=.7, label=name)
    low, high = min(0, np.nanmin(values)), max(0, np.nanmax(values))
    middle, span = (low + high) / 2, max(60, high - low)
    panels[0].set_ylim(middle - .75 * span, middle + .75 * span)
    panels[0].set_ylabel("Displacement (mm)")
    panels[0].legend(ncol=3, fontsize=8)
    reference = np.array([int(row["activity_label"]) for row in rows])
    bands = [("AI development label", reference)] + [(name, value.activity) for name, value in predictions.items()]
    for panel, (name, activity) in zip(panels[1:], bands):
        for code, color in ((-1, "#dddddd"), (1, "#428b7c")):
            for start, stop in runs(activity == code):
                panel.axvspan(start - .5, stop - .5, color=color)
        panel.set_yticks([])
        panel.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    panels[-1].set_xlim(0, len(values) - 1)
    panels[-1].set_xlabel("Day index | green: confirmed activity; grey: unknown; white: not confirmed")
    fig.suptitle(f"{case.case_id} — simulated, retrospective DEVELOPMENT comparison")
    fig.subplots_adjust(left=.2, right=.98, bottom=.11, top=.93, hspace=.15)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_stages(case, rows, predictions, path):
    fig, panels = plt.subplots(1 + len(predictions), 1, figsize=(14, 1.5 + .55 * len(predictions)), sharex=True)
    reference = np.array([STAGES.get(row["feature"], -1) for row in rows])
    for index, row in enumerate(rows):
        if int(row["activity_label"]) == 0:
            reference[index] = 0
        if row["feature"] == "slow_displacement":
            reference[index] = 4
    bands = [("AI development label", reference)] + [(name, value.stage) for name, value in predictions.items()]
    colors = {-1: "#dddddd", 1: "#dc7455", 2: "#ab89c2", 3: "#72a9be", 4: "#ddbc55"}
    for panel, (name, stages) in zip(panels, bands):
        for code, color in colors.items():
            for start, stop in runs(stages == code):
                panel.axvspan(start - .5, stop - .5, color=color)
        panel.set_yticks([])
        panel.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    panels[-1].set_xlim(0, len(rows) - 1)
    panels[-1].set_xlabel("Day | orange: A; purple: S; blue: D; yellow: reference slow (not scored as S); grey: unknown/failure")
    fig.suptitle(case.case_id + " | End-to-end stages, including missed activity and failed outputs")
    fig.subplots_adjust(left=.2, right=.98, bottom=.12, top=.92, hspace=.18)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_report(out, cases, labels, methods, case_ids):
    results = {}
    for method, predictions in methods.items():
        per_case = {cid: evaluate_case(labels[cid], predictions[cid]) for cid in case_ids}
        results[method] = {"aggregate": aggregate_scores(list(per_case.values())), "per_case": per_case}
    save_json(out / "results.json", results)
    save_json(out / "predictions.json", {name: {cid: value.to_dict() for cid, value in rows.items()}
                                          for name, rows in methods.items()})
    table = []
    for method, result in results.items():
        score = result["aggregate"]
        table.append({"method": method, "daily_f1": score["daily"]["f1"], "event_f1_iou03": score["events"]["f1"],
                      "stage_macro_f1": score["stage_macro_f1"], "weak_recall": score["weak_recall"],
                      "false_activity_days": score["daily"]["fp"], "missed_activity_days": score["daily"]["fn"],
                      "false_acceleration_days": score["false_acceleration_days"], "coverage": score["coverage"],
                      "failed_cases": score["failed_cases"], "stage_failed_cases": score["stage_failed_cases"]})
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for cid in case_ids:
        plot_case(cases[cid], labels[cid], {name: rows[cid] for name, rows in methods.items()}, out / f"{cid}.png")
        plot_stages(cases[cid], labels[cid], {name: rows[cid] for name, rows in methods.items()}, out / f"{cid}-stages.png")
    header = "".join(f"<th>{html.escape(key)}</th>" for key in table[0])
    body = "".join("<tr>" + "".join(f"<td>{value:.3f}</td>" if isinstance(value, float)
                                      else f"<td>{html.escape(str(value))}</td>" for value in row.values()) + "</tr>"
                   for row in table)
    cards = "".join(f'<figure><img src="{cid}.png" alt="{cid} 开发对比图"><figcaption>{cid}</figcaption></figure>'
                    f'<details><summary>{cid}：展开阶段对比</summary><img src="{cid}-stages.png" alt="{cid} 阶段对比"></details>'
                    for cid in case_ids)
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>长期位移开发对比</title>
<style>body{{font:15px system-ui;margin:32px;color:#243c37;background:#f7f9f8}}table{{border-collapse:collapse;background:white}}td,th{{padding:10px;border:1px solid #ccd6d1}}img{{width:100%;max-width:1400px}}figure{{margin:24px 0}}.scroll{{overflow:auto}}p{{max-width:1000px;line-height:1.8}}</style>
<h1>长期位移：统一开发对比</h1><p>24 条既有仿真记录，按完整记录划分 12 条训练、6 条验证、6 条开发评价。
全部记录以前已查看过，标签是 AI 辅助开发标注，不是独立盲测或专家金标准。居中估计使用未来观测，结果仅支持离线回顾。
灰色表示未知，不能算作正确静稳；缓慢位移单独评价检出率，不等同匀速。阶段漏检计入端到端评分。
事件采用已知标注日期上的一对一 IoU≥0.3，容忍候选边界但不是预警提前量。ungated_rule 仅是明确取消活动约束的消融。</p>
<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>{cards}</html>'''
    (out / "index.html").write_text(document, encoding="utf-8")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    paths = sorted((args.data / "cases").glob("*/input.json"))
    cases = {path.parent.name: LandslideInput.model_validate_json(path.read_text(encoding="utf-8")) for path in paths}
    labels = load_labels(args.labels, cases)
    split = development_split(labels, 20261007)
    manifest = {"schema_version": "landslide-development-comparison-v1", "seed": 20261007,
                "status": "running", "split": split, "data": str(args.data.resolve()),
                "labels": str(args.labels.resolve()), "labels_sha256": sha(args.labels),
                "input_sha256": {path.parent.name: sha(path) for path in paths},
                "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                                  [Path(__file__), ROOT / "src/gnss_sim/landslide_baselines.py",
                                   ROOT / "src/gnss_sim/landslide_evaluation.py",
                                   ROOT / "src/gnss_sim/landslide_diagnostics.py"]},
                "selection_objective": "validation daily F1 + one-to-one event F1 at IoU 0.3; first candidate breaks ties",
                "protocol": "Offline, synthetic, already viewed; development labels, not blind test",
                "minimum_run_days": 14, "stage_codes": STAGES}
    save_json(args.out / "manifest.json", manifest)
    features = {}
    for cid, case in cases.items():
        motions = {}
        canonical = hashlib.sha256(case.model_dump_json().encode()).hexdigest()
        for window in (31, 61, 91):
            cache = args.evidence / cid / f"motion-{window}.json"
            if cache.exists():
                motion = LandslideDiagnostics.model_validate_json(cache.read_text(encoding="utf-8"))
                if motion.input_sha256 != canonical or motion.window_days != window or motion.case_id != cid:
                    raise ValueError("Observation-only diagnostic cache mismatch")
            else:
                motion = derive_motion(case, window)
            motions[window] = motion
        features[cid] = observation_features(case, motions)
    save_json(args.out / "feature-contract.json", {"names": next(iter(features.values())).names,
                                                 "source": "observations only; no truth.json read"})
    print("Verified observation caches and frozen split", split, flush=True)
    choices, activity_model, stage_model = fit_baselines(features, labels, split, args.out)
    evaluation = split["development_evaluation"]
    methods = predict_baselines(features, evaluation, choices, activity_model, stage_model)
    table = write_report(args.out, cases, labels, methods, evaluation)
    manifest.update(status="complete", seconds=time.monotonic() - started)
    save_json(args.out / "manifest.json", manifest)
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
