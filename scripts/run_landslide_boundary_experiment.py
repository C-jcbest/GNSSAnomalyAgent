"""Run a frozen, repeated coordinate-check and block-grid robustness experiment."""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_landslide_candidate_confirmation import restore_prediction, verify_freeze
from run_landslide_comparison import (
    ROOT,
    LandslideInput,
    load_labels,
    plot_case,
    plot_stages,
    save_json,
    sha,
)
from run_landslide_window_optimization import copy_verified, load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_boundary_experiment import (
    ARMS,
    boundary_blocks,
    boundary_prompt,
    parse_boundary_response,
)
from gnss_sim.landslide_candidate_confirmation import confirmation_base, proposal_pool
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import aggregate_windows, apply_reviews

ARM_NAMES = {
    "b0_blocks": "B0 原分块",
    "b1_endpoints": "B1 明确端点检查",
    "b2_phase15": "B2 同端点检查＋分块偏移15日",
}
METHODS = {
    f"{arm}_rep{rep}": f"{ARM_NAMES[arm]} · 重复{rep + 1}"
    for rep in range(2) for arm in ARMS
}
HISTORICAL = {
    "history_r0": ("r0_candidate", "历史原候选确认"),
    "history_r1": ("r1_neutral", "历史中性说明"),
    "history_r2": ("r2_blocks", "历史分块观察"),
    "strict_single": ("strict_single", "历史严格多数单窗"),
    "numeric_reference": ("numeric_reference", "强数值活动＋同阶段头"),
}
METHODS.update({name: title for name, (_, title) in HISTORICAL.items()})


def prepare(config_path, out):
    config = read_json(config_path)
    parent = ROOT / config["source_run"]
    parent_manifest = read_json(parent / "manifest.json")
    verify_freeze(parent, parent_manifest)
    if parent_manifest["status"] != "complete" or parent_manifest["case_ids"] != config["case_ids"]:
        raise ValueError("Parent run must be complete with the same six records")
    settings = (config["arms"], config["repetitions"], config["maximum_requests"],
                config["block_days"], config["shift_days"], config["model"],
                config["temperature"], config["top_p"], config["max_output_tokens"],
                config["enable_thinking_requested"], config["retries"])
    if settings != (list(ARMS), 2, 102, 30, 15, parent_manifest["model"], .1, .3, 2048, False, 0):
        raise ValueError("Protocol differs from implemented transport and factors")
    targets = sorted([packet for packet in read_json(parent / "packets.json")
                      if packet["method"] == "r2_blocks"],
                     key=lambda packet: (packet["case_id"], packet["start"]))
    if len(targets) != 17:
        raise ValueError("Expected all 17 fixed targets")
    for packet in targets:
        if boundary_prompt(ARMS[0], packet["start"], packet["stop"]) != packet["prompt"]:
            raise ValueError("Concurrent control must be the exact historical R2 prompt")
    out.mkdir(parents=True, exist_ok=False)
    copy_verified(config_path, out / "protocol.json", sha(config_path))
    for name in ("manifest.json", "predictions.json", "results.json", "costs.json"):
        copy_verified(parent / name, out / ("parent-" + name), sha(parent / name))
    for name in ("reference-labels.csv", "stage-xgboost.json", "cached-screen-packets.json",
                 "candidate-pools.json"):
        copy_verified(parent / name, out / name, sha(parent / name))
    for path in (parent / "inputs").rglob("*.json"):
        copy_verified(path, out / path.relative_to(parent), sha(path))
    copied_images = set()
    packets = []
    for rep in range(2):
        for ordinal, target in enumerate(targets):
            for image, expected in target["image_sha256"].items():
                if image not in copied_images:
                    copy_verified(parent / image, out / image, expected)
                    copied_images.add(image)
            shift = (ordinal + rep) % len(ARMS)
            order = ARMS[shift:] + ARMS[:shift]
            for arm in order:
                packets.append({
                    "request_id": f"{target['case_id']}-target-{ordinal:02d}-{arm}-rep{rep}",
                    "case_id": target["case_id"], "method": f"{arm}_rep{rep}",
                    "arm": arm, "rep": rep, "start": target["start"], "stop": target["stop"],
                    "prompt": boundary_prompt(arm, target["start"], target["stop"]),
                    "images": target["images"], "image_sha256": target["image_sha256"],
                    "source_request_id": target["request_id"],
                })
    save_json(out / "packets.json", packets)
    source_names = set(parent_manifest["source_sha256"])
    source_names.update({str(Path(__file__).relative_to(ROOT)),
                         "src/gnss_sim/landslide_boundary_experiment.py",
                         "tests/test_landslide_boundary_experiment.py"})
    source_hashes = {}
    for name in sorted(source_names):
        source_hashes[name] = sha(ROOT / name)
        copy_verified(ROOT / name, out / "source" / name, source_hashes[name])
    frozen = {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}
    save_json(out / "manifest.json", {
        "status": "prepared", "date": config["date"], "case_ids": config["case_ids"],
        "model": config["model"], "planned_calls": len(packets), "calls": 0,
        "candidate_count": len(targets), "repetitions": 2, "evidence_sha256": frozen,
        "source_sha256": source_hashes, "labels_sha256": parent_manifest["labels_sha256"],
        "scope": config["limits"], "deviations": [],
    })
    print("Frozen all 17 targets × 3 arms × 2 repeats = 102 requests", flush=True)


def execute(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] != "prepared":
        raise ValueError("Only a never-started run may execute; no retries or resume selection")
    verify_freeze(out, manifest)
    config = read_json(out / "protocol.json")
    features = {case_id: load_case_features(out, case_id)[1] for case_id in config["case_ids"]}
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], 102)
    started = time.monotonic()
    manifest["status"] = "running"
    save_json(out / "manifest.json", manifest)
    outputs, failures = [], {}
    for index, packet in enumerate(read_json(out / "packets.json")):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise RuntimeError("Frozen run reached its hard timeout")
        observed = features[packet["case_id"]].observed
        try:
            payload = transport.ask(packet["request_id"], packet["prompt"],
                                    [out / name for name in packet["images"]])
            activity = parse_boundary_response(payload, observed, packet["start"],
                                               packet["stop"], packet["arm"])
            status = "success"
        except (RuntimeError, ValueError) as error:
            failures[packet["request_id"]] = str(error)
            activity = np.full(len(observed), -1, dtype=int)
            status = "failed"
        outputs.append({**packet, "status": status, "activity": activity.tolist()})
        save_json(out / "outputs.json", outputs)
        save_json(out / "failures.json", failures)
        manifest.update(calls=index + 1, failures=len(failures), seconds=time.monotonic() - started)
        save_json(out / "manifest.json", manifest)
        print(f"{index + 1}/102 {packet['request_id']}: {status}", flush=True)
    model = XGBClassifier()
    model.load_model(out / "stage-xgboost.json")
    previous = read_json(out / "parent-predictions.json")
    pools = read_json(out / "candidate-pools.json")
    cached = read_json(out / "cached-screen-packets.json")
    predictions = {method: {} for method in METHODS}
    for case_id, feature in features.items():
        windows = [row for row in cached if row["case_id"] == case_id]
        strict = aggregate_windows(windows, feature.observed)
        pool = proposal_pool(windows, feature.observed)
        if (pool.spans != [tuple(span) for span in pools[case_id]["spans_half_open"]]
                or strict.activity.tolist() != previous["strict_single"][case_id]["activity"]):
            raise ValueError("Frozen candidate or strict baseline changed")
        base = confirmation_base(strict.activity, pool, feature.observed)
        stage_codes = model.predict(feature.matrix).astype(int) + 1
        for rep in range(2):
            for arm in ARMS:
                method = f"{arm}_rep{rep}"
                reviews = [row for row in outputs if row["case_id"] == case_id and row["method"] == method]
                if [(row["start"], row["stop"]) for row in reviews] != pool.spans:
                    raise ValueError("A target was skipped, repeated or reordered")
                gate = apply_reviews(base, reviews, feature.observed)
                predictions[method][case_id] = learned_stages(
                    stage_codes, gate.activity, feature.supported,
                ).to_dict()
        for name, (parent_name, _) in HISTORICAL.items():
            predictions[name][case_id] = previous[parent_name][case_id]
    save_json(out / "predictions.json", predictions)
    receipts = [read_json(out / "requests" / (row["request_id"] + ".response.json")) for row in outputs]
    manifest.update(status="inference_complete", seconds=time.monotonic() - started,
                    total_tokens=sum(receipt.get("usage", {}).get("total_tokens", 0) for receipt in receipts))
    save_json(out / "manifest.json", manifest)
    report(out)


def report(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] not in {"inference_complete", "complete"}:
        raise ValueError("Every scheduled request is required before scoring")
    verify_freeze(out, manifest)
    predictions = read_json(out / "predictions.json")
    cases = {path.parent.name: LandslideInput.model_validate_json(path.read_text(encoding="utf-8"))
             for path in (out / "inputs").glob("*/input.json")}
    labels = load_labels(out / "reference-labels.csv", cases)
    outputs = read_json(out / "outputs.json")
    receipts = {row["request_id"]: read_json(out / "requests" / (row["request_id"] + ".response.json"))
                for row in outputs}
    if len(outputs) != 102 or len(receipts) != 102:
        raise ValueError("Incomplete or duplicated request receipts")
    verified_images = 0
    for row in outputs:
        receipt = receipts[row["request_id"]]
        if receipt["prompt"] != row["prompt"]:
            raise ValueError("Actual prompt differs from frozen packet")
        for image, expected in row["image_sha256"].items():
            if sha(out / image) != expected or receipt["image_sha256"][Path(image).name] != expected:
                raise ValueError("Actual image differs from frozen evidence")
            verified_images += 1
        observed = np.array([item is not None for item in cases[row["case_id"]].displacement_mm])
        if row["status"] == "success":
            replay = parse_boundary_response(json.loads(receipt["output"]), observed,
                                             row["start"], row["stop"], row["arm"])
            if replay.tolist() != row["activity"]:
                raise ValueError("Successful response changed or was repaired")
        elif any(value != -1 for value in row["activity"]):
            raise ValueError("Failed target did not remain entirely unknown")
    results = {}
    for method, records in predictions.items():
        scores = {case_id: evaluate_case(labels[case_id], restore_prediction(records[case_id]))
                  for case_id in manifest["case_ids"]}
        results[method] = {"per_case": scores, "aggregate": aggregate_scores(list(scores.values()))}
    cached_cost = read_json(out / "parent-costs.json")["r2_blocks"]
    screen_tokens = cached_cost["pipeline_tokens"] - cached_cost["new_tokens"]
    costs = {}
    onset = {}
    for rep in range(2):
        for arm in ARMS:
            method = f"{arm}_rep{rep}"
            rows = [row for row in outputs if row["method"] == method]
            selected_receipts = [receipts[row["request_id"]] for row in rows]
            tokens = sum(receipt.get("usage", {}).get("total_tokens", 0) for receipt in selected_receipts)
            costs[method] = {
                "new_calls": len(rows), "new_tokens": tokens, "pipeline_calls": 72 + len(rows),
                "pipeline_tokens": screen_tokens + tokens,
                "service_seconds_sum": sum(receipt["seconds"] for receipt in selected_receipts),
                "failed_targets": sum(row["status"] == "failed" for row in rows),
                "finish_reasons": [receipt.get("finish_reason") for receipt in selected_receipts],
            }
            onset[method] = {"successful_nonempty": 0, "starts_at_target": 0,
                             "shifted_starts": 0, "shifted_starts_at_grid_edge": 0,
                             "ends_before_target": 0, "internal_ends_at_grid_edge": 0}
            for row in rows:
                if row["status"] != "success":
                    continue
                intervals = json.loads(receipts[row["request_id"]]["output"])["activity"]
                if not intervals:
                    continue
                first, last = intervals[0][0], intervals[-1][1]
                blocks = boundary_blocks(row["start"], row["stop"], arm)
                onset[method]["successful_nonempty"] += 1
                onset[method]["starts_at_target"] += int(first == row["start"])
                onset[method]["shifted_starts"] += int(first > row["start"])
                onset[method]["shifted_starts_at_grid_edge"] += int(first in {left for left, _ in blocks[1:]})
                onset[method]["ends_before_target"] += int(last < row["stop"] - 1)
                onset[method]["internal_ends_at_grid_edge"] += int(last in {right - 1 for _, right in blocks[:-1]})
    development_checks = []
    for rep in range(2):
        control, treatment = f"b0_blocks_rep{rep}", f"b1_endpoints_rep{rep}"
        before, after = [results[name]["aggregate"] for name in (control, treatment)]
        development_checks.append({"rep": rep, "passes_frozen_check": bool(
            after["weak_detected_days"] == 81 and after["daily"]["fp"] <= before["daily"]["fp"]
            and after["daily"]["f1"] >= before["daily"]["f1"]
            and costs[treatment]["failed_targets"] <= costs[control]["failed_targets"]),
            "daily_f1_difference": after["daily"]["f1"] - before["daily"]["f1"]})
    comparisons = [(f"{arm}_rep0", f"{arm}_rep1") for arm in ARMS]
    comparisons += [(f"b1_endpoints_rep{rep}", f"b2_phase15_rep{rep}") for rep in range(2)]
    disagreements = []
    for left, right in comparisons:
        per_case = {}
        for case_id in manifest["case_ids"]:
            observed = np.array([item is not None for item in cases[case_id].displacement_mm])
            a, b = [np.array(predictions[name][case_id]["activity"]) for name in (left, right)]
            positives = observed & ((a == 1) | (b == 1))
            per_case[case_id] = {
                "observed_state_disagreement_days": int(np.count_nonzero(observed & (a != b))),
                "observed_positive_union_days": int(np.count_nonzero(positives)),
                "observed_positive_intersection_days": int(np.count_nonzero(observed & (a == 1) & (b == 1))),
            }
        disagreements.append({"left": left, "right": right, "per_case": per_case})
    analysis = {"status": "ANALYZED", "development_checks": development_checks,
                "both_repeats_pass": all(row["passes_frozen_check"] for row in development_checks),
                "onset_behavior": onset, "repeat_and_phase_disagreement": disagreements,
                "limits": manifest["scope"]}
    save_json(out / "results.json", results)
    save_json(out / "costs.json", costs)
    save_json(out / "analysis.json", analysis)
    save_json(out / "execution-audit.json", {
        "verified_calls": len(outputs), "verified_image_associations": verified_images,
        "failed_targets": len(read_json(out / "failures.json")), "no_retries": True,
        "predictions_sha256": sha(out / "predictions.json"),
    })
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    for case_id in manifest["case_ids"]:
        records = {METHODS[name]: restore_prediction(method[case_id]) for name, method in predictions.items()}
        plot_case(cases[case_id], labels[case_id], records, out / f"{case_id}.png")
        plot_stages(cases[case_id], labels[case_id], records, out / f"{case_id}-stages.png")
    render_pages(out, manifest, results, costs, analysis, outputs, receipts)
    manifest["status"] = "complete"
    save_json(out / "manifest.json", manifest)
    print(json.dumps({name: row["aggregate"]["daily"] for name, row in results.items()}, ensure_ascii=False), flush=True)


def render_pages(out, manifest, results, costs, analysis, outputs, receipts):
    style = 'body{font:16px system-ui,"Microsoft YaHei";margin:24px auto;padding:0 20px;max-width:1500px;color:#203e34;background:#f4f7f5}p{line-height:1.9}section,details{background:white;padding:20px;margin:20px 0;border:1px solid #d8e3df}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d8e3df;white-space:nowrap;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}summary{cursor:pointer;font-weight:600}a{color:#176757}'
    prefix = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>{style}</style>'
    page = prefix + '<title>端点与分块稳健性实验</title></head><body><h1>端点契约与分块位置：两次完整重复</h1><p><a href="trace.html">102次实际调用审查</a> · <a href="analysis.json">全部重复／相位诊断</a> · <a href="costs.json">实际成本</a></p>'
    page += f'<p>实际{manifest["calls"]}次、{manifest["total_tokens"]:,} token、失败{manifest["failures"]}次。相同17目标、原始图片、模型和保存阶段头；B0原分块，B1只增加明确端点检查，B2在B1上只将分块列表偏移15日。两次重复逐项列出，不挑最好回答；仍是六例开发材料。</p>'
    columns = ["配置", "TP", "FP日", "FN日", "活动F1", "阶段macro-F1", "弱日", "虚假A", "覆盖", "目标失败"]
    page += '<section><h2>全部成绩</h2><div class="scroll"><table><tr>' + ''.join(f'<th>{name}</th>' for name in columns) + '</tr>'
    md = ["# 端点与分块实验结果", "", "## Material Passport", "",
          "- Origin Skill: academic-research-suite / experiment-agent", "- Origin Mode: run + descriptive validation",
          "- Origin Date: 2026-10-08", "- Verification Status: ANALYZED；保存证据已重放，效果未独立验证",
          "- Version Label: boundary_v1", "", "| " + " | ".join(columns) + " |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, row in results.items():
        score = row["aggregate"]
        values = [METHODS[method], str(score["daily"]["tp"]), str(score["daily"]["fp"]),
                  str(score["daily"]["fn"]), f'{score["daily"]["f1"]:.6f}', f'{score["stage_macro_f1"]:.6f}',
                  str(score["weak_detected_days"]), str(score["false_acceleration_days"]),
                  f'{score["coverage"]:.2%}', str(costs[method]["failed_targets"]) if method in costs else "历史"]
        page += '<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>'
        md.append("| " + " | ".join(values) + " |")
    page += '</table></div><p>阶段分数包含错误活动中的虚假阶段与第一层漏检。参考未知／缺测不评分；预测未知在正日计FN，覆盖另报。失败整目标未知，没有裁剪。</p></section>'
    page += '<section><h2>冻结开发检查、起点与重复差异</h2><pre>' + html.escape(json.dumps(analysis, ensure_ascii=False, indent=2)) + '</pre></section>'
    for case_id in manifest["case_ids"]:
        page += f'<details><summary>{case_id} · 全记录活动与阶段图（事后评分，未送模）</summary><img loading="lazy" src="{case_id}.png" alt="{case_id}活动比较"><img loading="lazy" src="{case_id}-stages.png" alt="{case_id}阶段比较"></details>'
    page += '<p>短首尾块增加输出负担，相位对照不保证相同token；两次回答仅显示观测到的波动，不估计稳定成功概率。一个弱段、相关AI参考、离线居中特征与开发选择限制仍在。</p></body></html>'
    (out / "index.html").write_text(page, encoding="utf-8")
    md += ["", "```json", json.dumps(analysis, ensure_ascii=False, indent=2), "```", "", manifest["scope"], ""]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    trace = prefix + '<title>端点分块真实调用</title></head><body><h1>全部102次实际调用</h1><p><a href="index.html">全部结果</a> · 原始PNG没有标签、导数或数值摘要；observation为实际模型输出，不补造隐藏思维链。</p>'
    failures = read_json(out / "failures.json")
    for row in outputs:
        trace += f'<details><summary>{html.escape(row["request_id"])} · {row["start"]}–{row["stop"] - 1} · {row["status"]}</summary><p>{html.escape(failures.get(row["request_id"], "无契约失败"))}</p><h3>完整实际提示</h3><pre>{html.escape(row["prompt"])}</pre>'
        for image in row["images"]:
            trace += f'<a href="{image}"><img loading="lazy" src="{image}" alt="{html.escape(row["request_id"])}实际送模图"></a>'
        trace += '<h3>原始回答与请求记录</h3><pre>' + html.escape(json.dumps(receipts[row["request_id"]], ensure_ascii=False, indent=2)) + '</pre></details>'
    (out / "trace.html").write_text(trace + '</body></html>', encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "report"])
    parser.add_argument("--config", type=Path, default=Path("configs/landslide-boundary-v1.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.config, args.out)
    elif args.mode == "run":
        execute(args.out)
    else:
        report(args.out)
