"""Prepare, execute once, and report the frozen local-view experiment and saved stage head."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import time
from pathlib import Path

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
    plot_case,
    plot_stages,
    save_json,
    sha,
)
from xgboost import XGBClassifier

from gnss_sim.landslide_error_audit import activity_disagreements, stage_error_partition
from gnss_sim.landslide_local_experiment import (
    ACTIVITY_METHODS,
    execute_packet,
    learned_stages,
    request_schedule,
)
from gnss_sim.landslide_trace import read_json
from gnss_sim.landslide_visual import ACTIVITY_TASK, VisualRequests

METHOD_LABELS = {
    "activity_global": "定位：全局原图",
    "activity_local": "定位：全局＋局部原图",
    "stage_global": "固定活动：全局原图＋辅助图",
    "stage_local": "固定活动：全局原图＋辅助图＋局部原图",
    "visual_xgboost_stage": "固定视觉活动＋XGBoost 阶段",
    "visual_rule": "原视觉活动＋规则阶段",
    "visual_model": "原视觉活动＋模型阶段（历史调用）",
    "robust_xgboost_stage": "数值活动＋XGBoost 阶段",
}


def verified_copy(source, destination, expected):
    if sha(source) != expected:
        raise ValueError(f"Frozen source changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def prepare(config_path, atlas_root, out):
    config = read_json(config_path)
    numeric = ROOT / config["numeric_run"]
    visual = ROOT / config["fixed_stage_gate_run"]
    manifest = read_json(numeric / "manifest.json")
    atlas_freeze = read_json(atlas_root / "local-view-freeze.json")
    if config["status"] != "frozen_not_executed" or config["maximum_requests"] != 24:
        raise ValueError("Expected the frozen 24-request local-view protocol")
    if config["case_ids"] != sorted(manifest["split"]["development_evaluation"]):
        raise ValueError("Protocol and calibrated evaluation records differ")
    if sha(config_path) != atlas_freeze["config_sha256"]:
        raise ValueError("Local-view protocol changed after its freeze")
    if sha(numeric / "manifest.json") != atlas_freeze["numeric_manifest_sha256"]:
        raise ValueError("Numerical calibration changed")
    if sha(visual / "predictions.json") != atlas_freeze["fixed_stage_predictions_sha256"]:
        raise ValueError("Fixed visual activity source changed")
    if sha(ROOT / config["labels"]) != config["labels_sha256"]:
        raise ValueError("Development labels changed")
    out.mkdir(parents=True, exist_ok=False)
    fixed_predictions = read_json(visual / "predictions.json")
    packets = request_schedule(config["case_ids"])
    for packet in packets:
        cid = packet["case_id"]
        input_path = Path(manifest["data"]) / "cases" / cid / "input.json"
        verified_copy(input_path, out / "inputs" / cid / "input.json", manifest["input_sha256"][cid])
        case = LandslideInput.model_validate_json(input_path.read_text(encoding="utf-8"))
        receipt = read_json(visual / "requests" / f"{cid}-visual_model.started.json")
        raw, auxiliary = f"{cid}-raw.png", f"{cid}-auxiliary.png"
        for name in (raw, auxiliary):
            verified_copy(visual / name, out / name, receipt["image_sha256"][name])
        locals_ = [f"{cid}-local-{page}.png" for page in (1, 2, 3)]
        for name in locals_:
            verified_copy(atlas_root / name, out / name, atlas_freeze["existing_local_review_image_sha256"][name])
        if packet["method"] in ACTIVITY_METHODS:
            packet["prompt"] = ACTIVITY_TASK.format(last=len(case.dates) - 1)
            packet["prompt"] += '\nReturn exactly {"activity":[[start,end]],"uncertain":[[start,end]]}. No stage labels or other keys.'
            packet["images"] = [raw]
        else:
            packet["prompt"] = receipt["prompt"]
            packet["images"] = [raw, auxiliary]
            packet["activity"] = fixed_predictions[config["fixed_stage_gate_method"]][cid]["activity"]
        if packet["method"].endswith("_local"):
            packet["images"].extend(locals_)
        packet["image_sha256"] = {name: sha(out / name) for name in packet["images"]}
    save_json(out / "packets.json", packets)
    save_json(out / "supplement-plan.json", {
        "method": "visual_xgboost_stage", "activity_source": config["fixed_stage_gate_method"],
        "stage_model_sha256": sha(numeric / "stage-xgboost.json"),
        "policy": "Saved model, no fitting or threshold search; same feature support mask as reference diagnostic",
        "new_model_service_calls": 0,
    })
    verified_copy(config_path, out / "protocol.json", sha(config_path))
    source_paths = [Path(__file__), ROOT / "scripts/run_landslide_comparison.py",
                    ROOT / "src/gnss_sim/landslide_local_experiment.py",
                    ROOT / "src/gnss_sim/landslide_visual.py", ROOT / "src/gnss_sim/visual.py",
                    ROOT / "src/gnss_sim/landslide_baselines.py", ROOT / "src/gnss_sim/landslide_evaluation.py",
                    ROOT / "src/gnss_sim/landslide_error_audit.py", ROOT / "src/gnss_sim/landslide_trace.py"]
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        source_hashes[str(relative)] = sha(path)
        verified_copy(path, out / "source" / relative, sha(path))
    run = {"status": "prepared", "case_ids": config["case_ids"], "model": config["model"],
           "protocol_sha256": sha(config_path), "packets_sha256": sha(out / "packets.json"),
           "supplement_plan_sha256": sha(out / "supplement-plan.json"),
           "source_sha256": source_hashes, "numeric_manifest_sha256": sha(numeric / "manifest.json"),
           "fixed_predictions_sha256": sha(visual / "predictions.json"),
           "input_sha256": {cid: manifest["input_sha256"][cid] for cid in config["case_ids"]},
           "calls": 0, "budget": len(packets), "hard_timeout_seconds": 1800,
           "purpose": "Previously viewed synthetic development data; local-evidence controlled experiments",
           "numeric_root": str(numeric.resolve()), "visual_root": str(visual.resolve()),
           "atlas_root": str(atlas_root.resolve()), "deviations": []}
    save_json(out / "manifest.json", run)
    print(f"Prepared {len(packets)} frozen requests; no calls made", flush=True)


def load_features(out, run):
    cases, features = {}, {}
    for cid in run["case_ids"]:
        path = out / "inputs" / cid / "input.json"
        if sha(path) != run["input_sha256"][cid]:
            raise ValueError("Frozen observation file changed")
        case = LandslideInput.model_validate_json(path.read_text(encoding="utf-8"))
        cases[cid] = case
        motions = {}
        canonical = hashlib.sha256(case.model_dump_json().encode()).hexdigest()
        for window in (31, 61, 91):
            path = DEFAULT_EVIDENCE / cid / f"motion-{window}.json"
            motion = LandslideDiagnostics.model_validate_json(path.read_text(encoding="utf-8"))
            if motion.input_sha256 != canonical or motion.case_id != cid or motion.window_days != window:
                raise ValueError("Observation auxiliary cache mismatch")
            motions[window] = motion
        features[cid] = observation_features(case, motions)
    return cases, features


def execute(out):
    run = read_json(out / "manifest.json")
    if run["status"] != "prepared":
        raise ValueError("Only a prepared, never-started experiment may execute; no automatic retries")
    for name, expected in (("protocol.json", run["protocol_sha256"]), ("packets.json", run["packets_sha256"]),
                           ("supplement-plan.json", run["supplement_plan_sha256"])):
        if sha(out / name) != expected:
            raise ValueError(f"Prepared evidence changed: {name}")
    for relative, expected in run["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Code changed after experiment freeze: {relative}")
    config = read_json(out / "protocol.json")
    cases, features = load_features(out, run)
    numeric, visual = Path(run["numeric_root"]), Path(run["visual_root"])
    if sha(numeric / "manifest.json") != run["numeric_manifest_sha256"]:
        raise ValueError("Numerical source changed after prepare")
    if sha(visual / "predictions.json") != run["fixed_predictions_sha256"]:
        raise ValueError("Fixed activity source changed after prepare")
    supplement = read_json(out / "supplement-plan.json")
    if sha(numeric / "stage-xgboost.json") != supplement["stage_model_sha256"]:
        raise ValueError("Saved stage model changed")
    model = XGBClassifier()
    model.load_model(numeric / "stage-xgboost.json")
    previous = read_json(visual / "predictions.json")
    methods = {name: {} for name in ("activity_global", "activity_local", "stage_global", "stage_local", "visual_xgboost_stage")}
    for cid in run["case_ids"]:
        feature = features[cid]
        codes = model.predict(feature.matrix).astype(int) + 1
        numeric_gate = np.array(previous["robust_xgboost_stage"][cid]["activity"])
        replay = learned_stages(codes, numeric_gate, feature.supported)
        if replay.to_dict() != previous["robust_xgboost_stage"][cid]:
            raise ValueError("Saved stage model did not reproduce the prior numerical-gate result")
        gate = np.array(previous[config["fixed_stage_gate_method"]][cid]["activity"])
        methods["visual_xgboost_stage"][cid] = learned_stages(codes, gate, feature.supported)
    requests = VisualRequests(out / "requests", ROOT / ".env", config["model"], run["budget"])
    failures = {}
    started = time.monotonic()
    run["status"] = "running"
    save_json(out / "manifest.json", run)
    packets = read_json(out / "packets.json")
    for index, packet in enumerate(packets):
        if time.monotonic() - started > run["hard_timeout_seconds"]:
            raise TimeoutError("Experiment hard timeout reached; saved calls will not be retried")
        for name, expected in packet["image_sha256"].items():
            if sha(out / name) != expected:
                raise ValueError("Frozen model input image changed")
        cid = packet["case_id"]
        images = [out / name for name in packet["images"]]
        prediction = execute_packet(requests, packet, features[cid].observed, images, failures)
        prediction.validate(len(cases[cid].dates))
        methods[packet["method"]][cid] = prediction
        save_json(out / "predictions.json", {name: {key: value.to_dict() for key, value in group.items()}
                                              for name, group in methods.items()})
        save_json(out / "failures.json", failures)
        run.update(calls=index + 1, failed_calls_or_contracts=len(failures), seconds=time.monotonic() - started)
        save_json(out / "manifest.json", run)
        print(f"{index + 1}/{len(packets)} {packet['request_id']}: {prediction.status}/{prediction.stage_status}; failures={len(failures)}", flush=True)
    receipts = [read_json(path) for path in (out / "requests").glob("*.response.json")]
    run.update(status="inference_complete", total_tokens=sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
               seconds=time.monotonic() - started)
    save_json(out / "manifest.json", run)
    render_report(out)


def render_report(out):
    run = read_json(out / "manifest.json")
    if run["status"] not in {"inference_complete", "complete"}:
        raise ValueError("Inference must finish before scoring; render never calls a model")
    config = read_json(out / "protocol.json")
    if sha(ROOT / config["labels"]) != config["labels_sha256"]:
        raise ValueError("Labels changed before scoring")
    numeric = Path(run["numeric_root"])
    numeric_manifest = read_json(numeric / "manifest.json")
    all_cases = {path.parent.name: LandslideInput.model_validate_json(path.read_text(encoding="utf-8"))
                 for path in (Path(numeric_manifest["data"]) / "cases").glob("*/input.json")}
    labels = load_labels(ROOT / config["labels"], all_cases)
    saved = read_json(out / "predictions.json")
    prior = read_json(Path(run["visual_root"]) / "predictions.json")
    for name in ("visual_rule", "visual_model", "robust_xgboost_stage"):
        saved[name] = prior[name]
    methods = {name: {cid: ActivityPrediction(np.array(row["activity"]), np.array(row["stage"]),
                                             row["status"], row["gated"], row["stage_status"])
                      for cid, row in records.items()} for name, records in saved.items()}
    results, audit = {}, {}
    for name, records in methods.items():
        per_case = {cid: evaluate_case(labels[cid], records[cid]) for cid in run["case_ids"]}
        results[name] = {"aggregate": aggregate_scores(list(per_case.values())), "per_case": per_case,
                         "scope": "localization_only" if name in ACTIVITY_METHODS else "activity_and_stage"}
        audit[name] = {cid: {"activity": activity_disagreements(labels[cid], records[cid]),
                             "stages": stage_error_partition(labels[cid], records[cid])} for cid in run["case_ids"]}
    save_json(out / "results.json", results)
    save_json(out / "error-audit.json", audit)
    for cid in run["case_ids"]:
        plot_case(all_cases[cid], labels[cid], {name: group[cid] for name, group in methods.items()}, out / f"{cid}.png")
        plot_stages(all_cases[cid], labels[cid], {name: group[cid] for name, group in methods.items() if name not in ACTIVITY_METHODS},
                    out / f"{cid}-stages.png")
    rows = []
    for name, result in results.items():
        score = result["aggregate"]
        stage_score = "—" if name in ACTIVITY_METHODS else f"{score['stage_macro_f1']:.3f}"
        stage_failures = "—" if name in ACTIVITY_METHODS else str(score["stage_failed_cases"])
        rows.append(f"<tr><td>{html.escape(METHOD_LABELS[name])}</td><td>{score['daily']['f1']:.3f}</td>"
                    f"<td>{score['events']['f1']:.3f}</td><td>{score['weak_recall']:.3f}</td>"
                    f"<td>{score['daily']['fp']}</td><td>{score['daily']['fn']}</td><td>{stage_score}</td>"
                    f"<td>{score['failed_cases']}/{stage_failures}</td></tr>")
    common_cases = [cid for cid in run["case_ids"] if all(
        methods[name][cid].status == "success" for name in ACTIVITY_METHODS)]
    common_scores = {name: aggregate_scores([results[name]["per_case"][cid] for cid in common_cases])
                     for name in ACTIVITY_METHODS}
    false_acceleration = {name: results[name]["aggregate"]["false_acceleration_days"]
                          for name in ("stage_global", "stage_local", "visual_xgboost_stage", "robust_xgboost_stage")}
    findings = (
        '<section aria-label="实验分析"><h2>结果如何解释</h2>'
        f'<p>完成 {run["calls"]} 次新调用，{run["failed_calls_or_contracts"]} 次契约失败，无重试。'
        '全局定位控制在 case_0023 输出活动与未知区间共享端点，整项按失败保留；主表改善受此影响。</p>'
        f'<p>共同成功 {len(common_cases)} 条记录中，定位 F1 为 '
        f'{common_scores["activity_global"]["daily"]["f1"]:.3f} → '
        f'{common_scores["activity_local"]["daily"]["f1"]:.3f}；两组均漏掉全部 81 日弱位移。'
        '固定局部图本轮未解决弱位移或平台合并。</p>'
        '<p>固定旧视觉活动门控时，增加局部图使阶段 macro-F1 从 '
        f'{results["stage_global"]["aggregate"]["stage_macro_f1"]:.3f} 到 '
        f'{results["stage_local"]["aggregate"]["stage_macro_f1"]:.3f}，XGBoost 阶段为 '
        f'{results["visual_xgboost_stage"]["aggregate"]["stage_macro_f1"]:.3f}。'
        '但阶段总分须结合非活动日期的虚假加速：</p>'
        f'<p>全局视觉 {false_acceleration["stage_global"]} 日；局部视觉 '
        f'{false_acceleration["stage_local"]} 日；视觉活动＋XGBoost '
        f'{false_acceleration["visual_xgboost_stage"]} 日；数值活动＋XGBoost '
        f'{false_acceleration["robust_xgboost_stage"]} 日。XGBoost 阶段头只训练 A/S/D，'
        '无法自行拒绝视觉活动门控误检的平台。</p>'
        '<p>下一步优先检验原始位移活动确认、多尺度累计候选与困难负例；'
        '这是单次仿真开发实验，尚未证明真实数据泛化或在线预警能力。</p></section>'
    )
    cards = "".join(f'<section><h2>{cid}</h2><img src="{cid}.png" alt="活动对比">'
                    f'<img src="{cid}-stages.png" alt="阶段对比"></section>' for cid in run["case_ids"])
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>局部证据受控实验</title><style>body{font:15px/1.8 "Microsoft YaHei",sans-serif;background:#f5f5ef;color:#213c36;margin:30px}table{border-collapse:collapse;background:white}td,th{padding:10px;border:1px solid #d9dfd6}img{display:block;width:100%;max-width:1700px;margin:16px 0}section{margin:35px 0}a{color:#246b59}</style>
<h1>局部原图与阶段分工：开发对照实验</h1><p>定位对照与阶段对照分别运行，每组单次调用。阶段对照使用旧 visual-v3 活动门控，不能把新定位输出拼接为新端到端成绩。XGBoost 复用旧模型与相同门控，不重训。</p>
<p>全部是已查看仿真记录及 AI 辅助开发标注。弱位移只有一个 81 日参考段。两组调用数相同，输入图数与 token 不同。</p>
<p><a href="trace/">查看本轮实际图片、完整提示词与原始回答 →</a> · <a href="results.json">完整结果 JSON</a> · <a href="analysis.md">分析报告</a></p>
<table><tr><th>配置</th><th>活动 F1</th><th>事件 F1 / IoU 0.3</th><th>弱位移召回</th><th>活动误报日</th><th>活动漏检日</th><th>阶段 macro-F1</th><th>定位/阶段失败例数</th></tr>'''
    (out / "index.html").write_text(document + "".join(rows) + "</table>" + findings + cards + "</html>", encoding="utf-8")
    run["status"] = "complete"
    save_json(out / "manifest.json", run)
    print(json.dumps({name: {"activity_f1": value["aggregate"]["daily"]["f1"],
                            "stage_macro_f1": value["aggregate"]["stage_macro_f1"]} for name, value in results.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "render"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/landslide-local-view-experiment-v1.json")
    parser.add_argument("--atlas", type=Path, default=ROOT / "artifacts/landslide-stage-diagnosis-2026-10-07/reference-v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.config, args.atlas, args.out)
    elif args.mode == "run":
        execute(args.out)
    else:
        render_report(args.out)


if __name__ == "__main__":
    main()
