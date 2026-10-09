"""Freeze, run and score the repeated activity-output-contract comparison."""
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

from gnss_sim.landslide_candidate_confirmation import confirmation_base, proposal_pool
from gnss_sim.landslide_contract_experiment import ARMS, contract_prompt, parse_contract_response
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case, runs
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import aggregate_windows, apply_reviews

ARM_NAMES = {"c0_redundant": "C0 原B1重复字段", "c1_intervals": "C1 唯一活动区间＋观察文字"}
METHODS = {
    f"{arm}_rep{rep}": f"{ARM_NAMES[arm]} · 重复{rep + 1}"
    for rep in range(2) for arm in ARMS
}
HISTORICAL = {
    "history_b1_rep0": ("b1_endpoints_rep0", "历史B1 · 重复1"),
    "history_b1_rep1": ("b1_endpoints_rep1", "历史B1 · 重复2"),
    "history_b2_rep0": ("b2_phase15_rep0", "历史B2偏移15日 · 重复1"),
    "history_b2_rep1": ("b2_phase15_rep1", "历史B2偏移15日 · 重复2"),
    "strict_single": ("strict_single", "历史严格多数单窗"),
    "numeric_reference": ("numeric_reference", "强数值活动＋同阶段头"),
}
METHODS.update({name: title for name, (_, title) in HISTORICAL.items()})


def prepare(config_path: Path, out: Path):
    config = read_json(config_path)
    parent = ROOT / config["source_run"]
    parent_manifest = read_json(parent / "manifest.json")
    verify_freeze(parent, parent_manifest)
    if parent_manifest["status"] != "complete" or parent_manifest["case_ids"] != config["case_ids"]:
        raise ValueError("Parent must be complete with the same six records")
    settings = (
        config["arms"], config["repetitions"], config["maximum_requests"],
        config["block_days"], config["block_phase_days"], config["model"],
        config["temperature"], config["top_p"], config["max_output_tokens"],
        config["enable_thinking_requested"], config["retries"],
    )
    if settings != (list(ARMS), 2, 68, 30, 0, parent_manifest["model"], .1, .3, 2048, False, 0):
        raise ValueError("Configuration differs from implemented frozen factors or transport")
    targets = sorted(
        [packet for packet in read_json(parent / "packets.json") if packet["method"] == "b1_endpoints_rep0"],
        key=lambda packet: (packet["case_id"], packet["start"]),
    )
    if len(targets) != 17:
        raise ValueError("Expected all 17 original phase0 targets")
    for target in targets:
        if contract_prompt(ARMS[0], target["start"], target["stop"]) != target["prompt"]:
            raise ValueError("Concurrent control differs from historical B1")
    out.mkdir(parents=True, exist_ok=False)
    copy_verified(config_path, out / "protocol.json", sha(config_path))
    for name in ("manifest.json", "predictions.json", "results.json", "costs.json"):
        copy_verified(parent / name, out / ("parent-" + name), sha(parent / name))
    for name in ("reference-labels.csv", "stage-xgboost.json", "cached-screen-packets.json", "candidate-pools.json"):
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
            for arm in ARMS[shift:] + ARMS[:shift]:
                packets.append({
                    "request_id": f"{target['case_id']}-target-{ordinal:02d}-{arm}-rep{rep}",
                    "case_id": target["case_id"], "method": f"{arm}_rep{rep}",
                    "arm": arm, "rep": rep, "start": target["start"], "stop": target["stop"],
                    "prompt": contract_prompt(arm, target["start"], target["stop"]),
                    "images": target["images"], "image_sha256": target["image_sha256"],
                    "source_request_id": target["request_id"],
                })
    if len(packets) != 68 or len({row["request_id"] for row in packets}) != 68:
        raise ValueError("Incomplete or duplicate schedule")
    save_json(out / "packets.json", packets)
    preflight = {"planned_unique_calls": 68, "same_image_target_groups": 17,
                 "control_exact_b1": True, "historical_control_replayed_success": 0,
                 "historical_control_replayed_failure": 0, "no_schema_fallback": True}
    features = {case_id: load_case_features(out, case_id)[1] for case_id in config["case_ids"]}
    for row in read_json(parent / "outputs.json"):
        if row["arm"] != "b1_endpoints":
            continue
        observed = features[row["case_id"]].observed
        receipt = read_json(parent / "requests" / (row["request_id"] + ".response.json"))
        try:
            activity = parse_contract_response(json.loads(receipt["output"]), observed,
                                               row["start"], row["stop"], ARMS[0])
        except ValueError:
            if row["status"] != "failed" or any(value != -1 for value in row["activity"]):
                raise ValueError("Historical failure changed during preflight") from None
            preflight["historical_control_replayed_failure"] += 1
        else:
            if row["status"] != "success" or activity.tolist() != row["activity"]:
                raise ValueError("Historical successful response changed during preflight")
            preflight["historical_control_replayed_success"] += 1
    save_json(out / "preflight-audit.json", preflight)
    source_names = set(parent_manifest["source_sha256"])
    source_names.update({str(Path(__file__).relative_to(ROOT)),
                         "src/gnss_sim/landslide_contract_experiment.py",
                         "tests/test_landslide_contract_experiment.py"})
    source_hashes = {}
    for name in sorted(source_names):
        source_hashes[name] = sha(ROOT / name)
        copy_verified(ROOT / name, out / "source" / name, source_hashes[name])
    frozen = {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}
    save_json(out / "manifest.json", {
        "status": "prepared", "date": config["date"], "case_ids": config["case_ids"],
        "model": config["model"], "planned_calls": 68, "calls": 0,
        "candidate_count": 17, "repetitions": 2, "evidence_sha256": frozen,
        "source_sha256": source_hashes, "labels_sha256": parent_manifest["labels_sha256"],
        "scope": config["limits"], "deviations": [],
    })
    print(json.dumps(preflight, ensure_ascii=False), flush=True)


def execute(out: Path):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] != "prepared":
        raise ValueError("Only a never-started run may execute; no retry or selective resume")
    verify_freeze(out, manifest)
    config = read_json(out / "protocol.json")
    features = {case_id: load_case_features(out, case_id)[1] for case_id in config["case_ids"]}
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], 68)
    started = time.monotonic()
    manifest["status"] = "running"
    save_json(out / "manifest.json", manifest)
    outputs, failures = [], {}
    for index, packet in enumerate(read_json(out / "packets.json")):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise RuntimeError("Frozen experiment reached its hard timeout")
        observed = features[packet["case_id"]].observed
        try:
            payload = transport.ask(packet["request_id"], packet["prompt"],
                                    [out / name for name in packet["images"]])
            activity = parse_contract_response(payload, observed, packet["start"],
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
        print(f"{index + 1}/68 {packet['request_id']}: {status}", flush=True)
    save_predictions(out, features, outputs)
    receipts = [read_json(out / "requests" / (row["request_id"] + ".response.json")) for row in outputs]
    manifest.update(status="inference_complete", seconds=time.monotonic() - started,
                    total_tokens=sum(receipt.get("usage", {}).get("total_tokens", 0) for receipt in receipts))
    save_json(out / "manifest.json", manifest)
    report(out)


def save_predictions(out, features, outputs):
    model = XGBClassifier()
    model.load_model(out / "stage-xgboost.json")
    parent = read_json(out / "parent-predictions.json")
    pools = read_json(out / "candidate-pools.json")
    cached = read_json(out / "cached-screen-packets.json")
    predictions = {method: {} for method in METHODS}
    for case_id, feature in features.items():
        windows = [row for row in cached if row["case_id"] == case_id]
        strict = aggregate_windows(windows, feature.observed)
        pool = proposal_pool(windows, feature.observed)
        if (pool.spans != [tuple(span) for span in pools[case_id]["spans_half_open"]]
                or strict.activity.tolist() != parent["strict_single"][case_id]["activity"]):
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
            predictions[name][case_id] = parent[parent_name][case_id]
    save_json(out / "predictions.json", predictions)


def audit_calls(out, cases, outputs):
    packets = read_json(out / "packets.json")
    if len(outputs) != 68 or len({row["request_id"] for row in outputs}) != 68:
        raise ValueError("All 68 unique scheduled outputs are required")
    if len(list((out / "requests").glob("*.started.json"))) != 68:
        raise ValueError("Actual started request count differs from schedule")
    if len(list((out / "requests").glob("*.response.json"))) != 68:
        raise ValueError("Actual response count differs from schedule")
    receipts = {}
    failures = read_json(out / "failures.json")
    verified_images = 0
    for packet, row in zip(packets, outputs):
        if any(row[key] != value for key, value in packet.items()):
            raise ValueError("Saved output packet differs from frozen schedule")
        receipt = read_json(out / "requests" / (row["request_id"] + ".response.json"))
        receipts[row["request_id"]] = receipt
        if receipt["prompt"] != packet["prompt"]:
            raise ValueError("Actual prompt differs from frozen prompt")
        for image, expected in packet["image_sha256"].items():
            if sha(out / image) != expected or receipt["image_sha256"][Path(image).name] != expected:
                raise ValueError("Actual image differs from frozen evidence")
            verified_images += 1
        observed = np.array([item is not None for item in cases[row["case_id"]].displacement_mm])
        if row["status"] == "success":
            replay = parse_contract_response(json.loads(receipt["output"]), observed,
                                             row["start"], row["stop"], row["arm"])
            if replay.tolist() != row["activity"] or row["request_id"] in failures:
                raise ValueError("Successful response changed or was repaired")
        elif row["status"] != "failed" or any(value != -1 for value in row["activity"]):
            raise ValueError("Failed target did not remain entirely unknown")
        elif row["request_id"] not in failures:
            raise ValueError("Failed target has no recorded error")
        elif receipt.get("status") == "returned_json":
            try:
                parse_contract_response(json.loads(receipt["output"]), observed,
                                        row["start"], row["stop"], row["arm"])
            except ValueError as error:
                if str(error) != failures[row["request_id"]]:
                    raise ValueError("Failure replay reason changed") from error
            else:
                raise ValueError("A rejected payload now parses successfully")
    save_json(out / "execution-audit.json", {
        "verified_calls": len(outputs), "verified_image_associations": verified_images,
        "failed_targets": len(failures), "no_retries": True,
        "predictions_sha256": sha(out / "predictions.json"),
    })
    return receipts


def describe_results(out, manifest, cases, labels, predictions, results, costs, outputs, receipts):
    errors = {}
    for method, records in predictions.items():
        errors[method] = {}
        for case_id in manifest["case_ids"]:
            actual = np.array([int(row["activity_label"]) for row in labels[case_id]])
            observed = np.array([item is not None for item in cases[case_id].displacement_mm])
            activity = np.array(records[case_id]["activity"])
            stage = np.array(records[case_id]["stage"])
            if (np.any(activity[~observed] != -1) or np.any(stage[~observed] != -1)
                    or np.any((stage > 0) & (activity != 1))):
                raise ValueError("Missing-date mask or activity-to-stage gate changed")
            errors[method][case_id] = {
                "fp_half_open": runs((actual == 0) & (activity == 1)),
                "fn_half_open": runs((actual == 1) & (activity != 1)),
                "false_A_days": int(np.count_nonzero((actual == 0) & (stage == 1))),
            }
        for field in ("fp", "fn"):
            count = sum(stop - start for record in errors[method].values()
                        for start, stop in record[field + "_half_open"])
            if count != results[method]["aggregate"]["daily"][field]:
                raise ValueError("Error intervals do not conserve official counts")
    checks, paired_changes = [], []
    for rep in range(2):
        control, treatment = f"{ARMS[0]}_rep{rep}", f"{ARMS[1]}_rep{rep}"
        before, after = [results[name]["aggregate"] for name in (control, treatment)]
        checks.append({
            "rep": rep,
            "passes_frozen_check": bool(
                after["weak_detected_days"] == 81
                and costs[treatment]["failed_targets"] <= costs[control]["failed_targets"]
                and after["daily"]["fp"] <= before["daily"]["fp"]
                and after["false_acceleration_days"] <= before["false_acceleration_days"]
                and after["daily"]["f1"] >= before["daily"]["f1"]
                and after["stage_macro_f1"] >= before["stage_macro_f1"]
            ),
            "activity_f1_delta": after["daily"]["f1"] - before["daily"]["f1"],
            "stage_macro_f1_delta": after["stage_macro_f1"] - before["stage_macro_f1"],
        })
        changes = {}
        for case_id in manifest["case_ids"]:
            actual = np.array([int(row["activity_label"]) for row in labels[case_id]])
            a, b = [np.array(predictions[name][case_id]["activity"]) for name in (control, treatment)]
            changes[case_id] = {
                "gained_TP": int(np.count_nonzero((actual == 1) & (a != 1) & (b == 1))),
                "lost_TP": int(np.count_nonzero((actual == 1) & (a == 1) & (b != 1))),
                "removed_FP": int(np.count_nonzero((actual == 0) & (a == 1) & (b != 1))),
                "added_FP": int(np.count_nonzero((actual == 0) & (a != 1) & (b == 1))),
            }
        if (sum(row["gained_TP"] - row["lost_TP"] for row in changes.values()) != after["daily"]["tp"] - before["daily"]["tp"]
                or sum(row["added_FP"] - row["removed_FP"] for row in changes.values()) != after["daily"]["fp"] - before["daily"]["fp"]):
            raise ValueError("Paired changes do not conserve official counts")
        paired_changes.append({"rep": rep, "per_case": changes})
    disagreements = {}
    for arm in ARMS:
        per_case = {}
        for case_id in manifest["case_ids"]:
            observed = np.array([item is not None for item in cases[case_id].displacement_mm])
            a, b = [np.array(predictions[f"{arm}_rep{rep}"][case_id]["activity"]) for rep in range(2)]
            per_case[case_id] = int(np.count_nonzero(observed & (a != b)))
        disagreements[arm] = {"per_case": per_case, "total": sum(per_case.values()),
                              "meaning": "Observed 0/1/unknown disagreement includes contract failures"}
    failure_diagnostics = []
    for row in outputs:
        if row["status"] == "failed":
            actual = np.array([int(item["activity_label"]) for item in labels[row["case_id"]]])
            failure_diagnostics.append({
                "request_id": row["request_id"], "method": row["method"],
                "target_half_open": [row["start"], row["stop"]],
                "error": read_json(out / "failures.json")[row["request_id"]],
                "reference_positive_days": int(np.count_nonzero(actual[row["start"]:row["stop"]] == 1)),
                "raw_output": receipts[row["request_id"]].get("output"),
            })
    analysis = {
        "status": "ANALYZED", "development_checks": checks,
        "both_repeats_pass": all(row["passes_frozen_check"] for row in checks),
        "paired_known_day_changes": paired_changes, "repeat_disagreement": disagreements,
        "per_case_errors": errors, "failure_diagnostics": failure_diagnostics,
        "count_conservation_verified": True, "missing_and_stage_masks_verified": True,
        "limits": manifest["scope"],
    }
    save_json(out / "analysis.json", analysis)
    return analysis


def report(out: Path):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] not in {"inference_complete", "complete"}:
        raise ValueError("Every scheduled request must finish before scoring")
    verify_freeze(out, manifest)
    predictions = read_json(out / "predictions.json")
    cases = {path.parent.name: LandslideInput.model_validate_json(path.read_text(encoding="utf-8"))
             for path in (out / "inputs").glob("*/input.json")}
    labels = load_labels(out / "reference-labels.csv", cases)
    outputs = read_json(out / "outputs.json")
    receipts = audit_calls(out, cases, outputs)
    results = {}
    for method, records in predictions.items():
        scores = {case_id: evaluate_case(labels[case_id], restore_prediction(records[case_id]))
                  for case_id in manifest["case_ids"]}
        results[method] = {"per_case": scores, "aggregate": aggregate_scores(list(scores.values()))}
    parent_cost = read_json(out / "parent-costs.json")["b1_endpoints_rep0"]
    screen_tokens = parent_cost["pipeline_tokens"] - parent_cost["new_tokens"]
    costs = {}
    for rep in range(2):
        for arm in ARMS:
            method = f"{arm}_rep{rep}"
            rows = [row for row in outputs if row["method"] == method]
            selected = [receipts[row["request_id"]] for row in rows]
            tokens = sum(receipt.get("usage", {}).get("total_tokens", 0) for receipt in selected)
            costs[method] = {
                "new_calls": len(rows), "new_tokens": tokens, "pipeline_calls": 72 + len(rows),
                "pipeline_tokens": screen_tokens + tokens,
                "service_seconds_sum": sum(receipt["seconds"] for receipt in selected),
                "failed_targets": sum(row["status"] == "failed" for row in rows),
                "finish_reasons": [receipt.get("finish_reason") for receipt in selected],
            }
    save_json(out / "results.json", results)
    save_json(out / "costs.json", costs)
    analysis = describe_results(out, manifest, cases, labels, predictions, results, costs, outputs, receipts)
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
    prefix = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px/1.65 system-ui;background:#f2f5f8;color:#172638;max-width:1450px;margin:28px auto;padding:0 20px}section,details{background:white;padding:20px;margin:16px 0;border:1px solid #dce3eb;border-radius:10px}h1{font-size:30px}h2{font-size:23px}a{color:#125fa7}table{border-collapse:collapse;min-width:100%;white-space:nowrap}th,td{padding:10px 13px;border-bottom:1px solid #dce3eb;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f6f8fb;padding:16px}img{max-width:100%;height:auto}summary{cursor:pointer;font-weight:600}</style>'
    page = prefix + '<title>活动输出契约对照</title></head><body><h1>减少重复字段：完整重复实验</h1><p><a href="analysis.html">效果与案例分析</a> · <a href="trace.html">全部真实调用</a> · <a href="protocol.json">冻结协议</a></p>'
    page += f'<p>本轮{manifest["calls"]}次、{manifest["total_tokens"]:,} token，目标失败{manifest["failures"]}次。C0逐字沿用原B1；C1只减少输出契约重复字段，图像、原分块、候选、阶段头和标签不变。未知／失败与所有目标保留，六例均为已查看开发记录。</p>'
    columns = ["配置", "TP", "FP日", "FN日", "活动F1", "阶段macro-F1", "弱日", "虚假A", "覆盖", "目标失败"]
    page += '<section><h2>全部成绩</h2><div class="scroll"><table><tr>' + ''.join(f'<th>{name}</th>' for name in columns) + '</tr>'
    md = ["# 活动输出契约实验结果", "", "## Material Passport", "",
          "- Origin Skill: academic-research-suite / experiment-agent", "- Origin Mode: run + descriptive validation",
          "- Origin Date: 2026-10-08", "- Verification Status: ANALYZED；保存证据已重放，独立验证未完成",
          "- Version Label: contract_v1", "", "| " + " | ".join(columns) + " |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, row in results.items():
        score = row["aggregate"]
        values = [METHODS[method], str(score["daily"]["tp"]), str(score["daily"]["fp"]),
                  str(score["daily"]["fn"]), f'{score["daily"]["f1"]:.6f}', f'{score["stage_macro_f1"]:.6f}',
                  str(score["weak_detected_days"]), str(score["false_acceleration_days"]),
                  f'{score["coverage"]:.2%}', str(costs[method]["failed_targets"]) if method in costs else "历史"]
        page += '<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>'
        md.append("| " + " | ".join(values) + " |")
    page += '</table></div><p>参考未知／缺测不评分；预测未知在正日计FN，负日不计TN，覆盖另报。阶段错误包含活动漏检及参考非活动日的虚假阶段。</p></section>'
    page += '<section><h2>冻结检查与错误分解</h2><pre>' + html.escape(json.dumps(analysis, ensure_ascii=False, indent=2)) + '</pre></section>'
    for case_id in manifest["case_ids"]:
        page += f'<details><summary>{case_id} · 全记录活动及阶段图（事后评分，未送模）</summary><img loading="lazy" src="{case_id}.png" alt="{case_id}活动比较"><img loading="lazy" src="{case_id}-stages.png" alt="{case_id}阶段比较"></details>'
    page += '<p>少拒判不能单独证明视觉识别更准。C1文字一致性另行审查，不用关键词自动改范围；不按成功回答筛选正式成绩，也不把两个重复当独立记录。</p></body></html>'
    (out / "index.html").write_text(page, encoding="utf-8")
    md += ["", "```json", json.dumps(analysis, ensure_ascii=False, indent=2), "```", "", manifest["scope"], ""]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    trace = prefix + '<title>活动契约真实调用</title></head><body><h1>全部68次实际调用</h1><p><a href="analysis.html">案例分析</a> · <a href="index.html">全部结果</a> · 只有两张原始PNG，没有标签、导数或数值摘要；观察文字来自真实回答，不补造隐藏思维链。</p>'
    failures = read_json(out / "failures.json")
    for row in outputs:
        trace += f'<details><summary>{html.escape(row["request_id"])} · {row["start"]}–{row["stop"] - 1} · {row["status"]}</summary><p>{html.escape(failures.get(row["request_id"], "无契约失败；文字一致性须另审查"))}</p><h3>完整实际提示</h3><pre>{html.escape(row["prompt"])}</pre>'
        for image in row["images"]:
            trace += f'<a href="{image}"><img loading="lazy" src="{image}" alt="{html.escape(row["request_id"])}实际送模图"></a>'
        trace += '<h3>原始回答与请求记录</h3><pre>' + html.escape(json.dumps(receipts[row["request_id"]], ensure_ascii=False, indent=2)) + '</pre></details>'
    (out / "trace.html").write_text(trace + '</body></html>', encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "report"])
    parser.add_argument("--config", type=Path, default=Path("configs/landslide-contract-v1.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.config, args.out)
    elif args.mode == "run":
        execute(args.out)
    else:
        report(args.out)
