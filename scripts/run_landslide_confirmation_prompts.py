"""Freeze, run and analyze concurrent raw-confirmation wording and block ablations."""
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
from gnss_sim.landslide_confirmation_prompt_experiment import (
    ARMS,
    confirmation_prompt,
    parse_confirmation,
)
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import aggregate_windows, apply_reviews

METHODS = {
    "r0_candidate": "R0 原候选确认（同期）",
    "r1_neutral": "R1 中性目标说明（同期）",
    "r2_blocks": "R2 中性说明＋逐块观察（同期）",
    "strict_single": "历史严格多数单窗",
    "historical_proposal_visual": "历史候选原提示确认",
    "historical_proposal_calendar": "历史候选缺测提示确认",
    "numeric_reference": "强数值活动＋同阶段头",
}


def prepare(config_path, out):
    config = read_json(config_path)
    parent = ROOT / config["source_run"]
    parent_manifest = read_json(parent / "manifest.json")
    if parent_manifest["status"] != "complete" or parent_manifest["case_ids"] != config["case_ids"]:
        raise ValueError("Parent experiment is incomplete or uses another evaluation corpus")
    verify_freeze(parent, parent_manifest)
    settings = (config["temperature"], config["top_p"], config["max_output_tokens"],
                config["enable_thinking_requested"], config["repetitions"], config["retries"])
    if settings != (.1, .3, 2048, False, 1, 0) or config["arms"] != list(ARMS):
        raise ValueError("Protocol differs from the implemented transport or arms")
    if config["calendar_metadata"] or config["block_days"] != 30 or config["model"] != parent_manifest["model"]:
        raise ValueError("This experiment must retain images and model without new calendar information")
    source_packets = sorted([packet for packet in read_json(parent / "packets.json")
                             if packet["method"] == "proposal_visual"], key=lambda packet: (packet["case_id"], packet["start"]))
    if len(source_packets) != 17 or config["maximum_requests"] != 3 * len(source_packets):
        raise ValueError("All 17 candidates and exactly 51 new calls are required")
    out.mkdir(parents=True, exist_ok=False)
    copy_verified(config_path, out / "protocol.json", sha(config_path))
    for name in ("manifest.json", "predictions.json", "results.json", "costs.json"):
        copy_verified(parent / name, out / ("parent-" + name), sha(parent / name))
    for name in ("reference-labels.csv", "stage-xgboost.json", "cached-screen-packets.json", "candidate-pools.json"):
        copy_verified(parent / name, out / name, sha(parent / name))
    for path in (parent / "inputs").rglob("*.json"):
        copy_verified(path, out / path.relative_to(parent), sha(path))
    copied_images, packets = set(), []
    for ordinal, source in enumerate(source_packets):
        expected_original = confirmation_prompt(ARMS[0], source["start"], source["stop"])
        if expected_original != source["prompt"]:
            raise ValueError("R0 changed the prior original confirmation prompt")
        for image, expected in source["image_sha256"].items():
            if image not in copied_images:
                copy_verified(parent / image, out / image, expected)
                copied_images.add(image)
        order = ARMS[ordinal % 3:] + ARMS[:ordinal % 3]
        for arm in order:
            packets.append({"case_id": source["case_id"], "method": arm,
                            "request_id": f"{source['case_id']}-target-{ordinal:02d}-{arm}",
                            "start": source["start"], "stop": source["stop"],
                            "prompt": confirmation_prompt(arm, source["start"], source["stop"]),
                            "images": source["images"], "image_sha256": source["image_sha256"],
                            "source_request_id": source["request_id"]})
    save_json(out / "packets.json", packets)
    source_names = list(parent_manifest["source_sha256"])
    source_names.extend([str(Path(__file__).relative_to(ROOT)),
                         "src/gnss_sim/landslide_confirmation_prompt_experiment.py"])
    source_hashes = {}
    for name in source_names:
        source_hashes[name] = sha(ROOT / name)
        copy_verified(ROOT / name, out / "source" / name, source_hashes[name])
    frozen = {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}
    save_json(out / "manifest.json", {"status": "prepared", "date": config["date"],
                                     "case_ids": config["case_ids"], "model": config["model"],
                                     "planned_calls": len(packets), "candidate_count": len(source_packets), "calls": 0,
                                     "evidence_sha256": frozen, "source_sha256": source_hashes,
                                     "labels_sha256": parent_manifest["labels_sha256"],
                                     "parent_manifest_sha256": sha(parent / "manifest.json"),
                                     "scope": config["limits"], "deviations": []})
    print("Frozen 17 identical targets and 51 new calls; no screen/image/stage change", flush=True)


def execute(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] != "prepared":
        raise ValueError("Only a never-started experiment may execute")
    verify_freeze(out, manifest)
    config = read_json(out / "protocol.json")
    features = {case_id: load_case_features(out, case_id)[1] for case_id in config["case_ids"]}
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_requests"])
    started = time.monotonic()
    manifest["status"] = "running"
    save_json(out / "manifest.json", manifest)
    outputs, failures = [], {}
    for index, packet in enumerate(read_json(out / "packets.json")):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise RuntimeError("Frozen prompt experiment timeout reached")
        observed = features[packet["case_id"]].observed
        try:
            payload = transport.ask(packet["request_id"], packet["prompt"], [out / name for name in packet["images"]])
            activity = parse_confirmation(payload, observed, packet["start"], packet["stop"], packet["method"])
            status = "success"
        except (RuntimeError, ValueError) as error:
            failures[packet["request_id"]] = str(error)
            activity = np.full(len(observed), -1, dtype=int)
            status = "failed"
        outputs.append({**packet, "activity": activity.tolist(), "status": status})
        save_json(out / "outputs.json", outputs)
        save_json(out / "failures.json", failures)
        manifest.update(calls=index + 1, failures=len(failures), seconds=time.monotonic() - started)
        save_json(out / "manifest.json", manifest)
        print(f"{index + 1}/{manifest['planned_calls']} {packet['request_id']}: {status}", flush=True)
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
        frozen_pool = pools[case_id]
        if (pool.spans != [tuple(span) for span in frozen_pool["spans_half_open"]]
                or pool.positive_vote_days.tolist() != frozen_pool["positive_vote_days"]
                or pool.negative_unknown_conflict_days.tolist() != frozen_pool["negative_unknown_conflict_days"]
                or strict.activity.tolist() != previous["strict_single"][case_id]["activity"]):
            raise ValueError("Cached source activity or targets changed")
        base = confirmation_base(strict.activity, pool, feature.observed)
        stage_codes = model.predict(feature.matrix).astype(int) + 1
        for arm in ARMS:
            reviews = [row for row in outputs if row["case_id"] == case_id and row["method"] == arm]
            if [(row["start"], row["stop"]) for row in reviews] != pool.spans:
                raise ValueError("Some candidates were skipped or reordered")
            gate = apply_reviews(base, reviews, feature.observed)
            predictions[arm][case_id] = learned_stages(stage_codes, gate.activity, feature.supported).to_dict()
        for name, parent_name in (("strict_single", "strict_single"),
                                  ("historical_proposal_visual", "proposal_visual"),
                                  ("historical_proposal_calendar", "proposal_calendar"),
                                  ("numeric_reference", "numeric_reference")):
            predictions[name][case_id] = previous[parent_name][case_id]
    save_json(out / "predictions.json", predictions)
    receipts = [read_json(out / "requests" / (row["request_id"] + ".response.json")) for row in outputs]
    manifest.update(status="inference_complete", seconds=time.monotonic() - started,
                    total_tokens=sum(receipt.get("usage", {}).get("total_tokens", 0) for receipt in receipts))
    save_json(out / "manifest.json", manifest)
    report(out)


def paired_intervals(results, case_ids, comparisons):
    draws = np.random.default_rng(20261008).integers(0, len(case_ids), size=(2000, len(case_ids)))
    comparisons_report = []
    for treatment, control in comparisons:
        differences = []
        for indices in draws:
            scores = [aggregate_scores([results[arm]["per_case"][case_ids[index]] for index in indices])
                      for arm in (treatment, control)]
            if any(score["daily"]["f1"] is None for score in scores):
                continue
            differences.append(scores[0]["daily"]["f1"] - scores[1]["daily"]["f1"])
        comparisons_report.append({"treatment": treatment, "control": control,
                                   "defined_draws": len(differences), "undefined_draws": len(draws) - len(differences),
                                   "descriptive_record_percentile95_daily_f1": np.quantile(differences, [.025, .975]).tolist()})
    return comparisons_report


def report(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] not in {"inference_complete", "complete"}:
        raise ValueError("Complete inference is required before scoring")
    verify_freeze(out, manifest)
    config, predictions = read_json(out / "protocol.json"), read_json(out / "predictions.json")
    cases = {path.parent.name: LandslideInput.model_validate_json(path.read_text(encoding="utf-8"))
             for path in (out / "inputs").glob("*/input.json")}
    labels = load_labels(out / "reference-labels.csv", cases)
    observed_by_case = {case_id: np.array([row is not None for row in cases[case_id].displacement_mm])
                        for case_id in manifest["case_ids"]}
    results = {}
    for method, records in predictions.items():
        per_case = {case_id: evaluate_case(labels[case_id], restore_prediction(records[case_id]))
                    for case_id in manifest["case_ids"]}
        results[method] = {"per_case": per_case, "aggregate": aggregate_scores(list(per_case.values()))}
    outputs = read_json(out / "outputs.json")
    receipts = {row["request_id"]: read_json(out / "requests" / (row["request_id"] + ".response.json")) for row in outputs}
    failures, costs, onset = read_json(out / "failures.json"), {}, {}
    verified_images = 0
    for row in outputs:
        receipt = receipts[row["request_id"]]
        if receipt["prompt"] != row["prompt"]:
            raise ValueError("Actual prompt differs from frozen packet")
        for image, expected in row["image_sha256"].items():
            if sha(out / image) != expected or receipt["image_sha256"][Path(image).name] != expected:
                raise ValueError("Actual image differs from frozen packet")
            verified_images += 1
        observed = observed_by_case[row["case_id"]]
        if row["status"] == "success":
            replay = parse_confirmation(json.loads(receipt["output"]), observed, row["start"], row["stop"], row["method"])
            if replay.tolist() != row["activity"]:
                raise ValueError("Successful response was repaired")
        elif any(value != -1 for value in row["activity"]):
            raise ValueError("Failed output was repaired")
    cached_cost = read_json(out / "parent-costs.json")["proposal_visual"]
    screen_tokens = cached_cost["pipeline_tokens"] - cached_cost["new_confirmation_tokens"]
    for arm in ARMS:
        rows = [row for row in outputs if row["method"] == arm]
        arm_receipts = [receipts[row["request_id"]] for row in rows]
        tokens = sum(receipt.get("usage", {}).get("total_tokens", 0) for receipt in arm_receipts)
        costs[arm] = {"new_calls": len(rows), "new_tokens": tokens, "cached_screen_calls": 72,
                      "pipeline_calls": 72 + len(rows), "pipeline_tokens": screen_tokens + tokens,
                      "service_seconds_sum": sum(receipt["seconds"] for receipt in arm_receipts),
                      "contract_or_transport_failures": sum(row["status"] == "failed" for row in rows),
                      "finish_reasons": [receipt.get("finish_reason") for receipt in arm_receipts]}
        nonempty = [row for row in rows if row["status"] == "success" and json.loads(receipts[row["request_id"]]["output"])["activity"]]
        onset[arm] = {"successful_nonempty": len(nonempty),
                      "starts_at_target": sum(json.loads(receipts[row["request_id"]]["output"])["activity"][0][0] == row["start"] for row in nonempty)}
    baseline = results[ARMS[0]]["aggregate"]
    decisions = {}
    for arm in ARMS[1:]:
        score = results[arm]["aggregate"]
        decisions[arm] = {"passes_frozen_development_check": bool(
            score["daily"]["fp"] < baseline["daily"]["fp"] and score["weak_detected_days"] == 81
            and score["daily"]["f1"] >= baseline["daily"]["f1"]),
            "interpretation": "Development selection only; not independent superiority"}
    save_json(out / "results.json", results)
    save_json(out / "costs.json", costs)
    save_json(out / "analysis.json", {"status": "ANALYZED", "onset_behavior": onset,
                                      "development_checks": decisions,
                                      "paired_record_intervals": paired_intervals(results, manifest["case_ids"], config["paired_comparisons"]),
                                      "limits": config["limits"]})
    save_json(out / "execution-audit.json", {"verified_calls": len(outputs), "verified_image_associations": verified_images,
                                             "no_retries": True, "failure_count": len(failures),
                                             "predictions_sha256": sha(out / "predictions.json")})
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    for case_id in manifest["case_ids"]:
        records = {METHODS[name]: restore_prediction(method[case_id]) for name, method in predictions.items()}
        plot_case(cases[case_id], labels[case_id], records, out / f"{case_id}.png")
        plot_stages(cases[case_id], labels[case_id], records, out / f"{case_id}-stages.png")
    render_pages(out, manifest, results, outputs, receipts, failures, decisions, onset)
    manifest["status"] = "complete"
    save_json(out / "manifest.json", manifest)
    print(json.dumps({name: {"daily": row["aggregate"]["daily"], "stage_macro_f1": row["aggregate"]["stage_macro_f1"],
                              "weak_days": row["aggregate"]["weak_detected_days"], "false_A": row["aggregate"]["false_acceleration_days"]}
                      for name, row in results.items()}, ensure_ascii=False), flush=True)


def render_pages(out, manifest, results, outputs, receipts, failures, decisions, onset):
    style = 'body{font:16px system-ui,"Microsoft YaHei";color:#203e34;background:#f4f7f5;margin:24px auto;max-width:1500px;padding:0 20px}p{line-height:1.9}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px;font-size:13px}img{width:100%;height:auto}section,details{background:white;padding:18px;margin:18px 0;border:1px solid #d8e3df}summary{cursor:pointer;font-weight:600}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #d8e3df;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}a{color:#176757}'
    prefix = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>{style}</style>'
    rows, markdown = [], ["# 确认提示因素实验", "", "## Material Passport", "",
                          "- Origin Skill: academic-research-suite / experiment-agent", "- Origin Mode: run + descriptive analysis",
                          "- Origin Date: 2026-10-08", "- Verification Status: ANALYZED；真实执行、冻结证据重放核查，开发效果未独立验证",
                          "- Version Label: confirmation_prompt_v1", "",
                          f"新增{manifest['calls']}次；相同17候选与相同原始图，三个同期组，失败{manifest['failures']}次，无重试。", "",
                          "| 配置 | 活动P | 活动R | 活动F1 | FP日 | FN日 | 阶段macro-F1 | 弱日 | 虚假A日 | 覆盖 |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, record in results.items():
        score, daily = record["aggregate"], record["aggregate"]["daily"]
        values = [METHODS[name], f"{daily['precision']:.3f}", f"{daily['recall']:.3f}", f"{daily['f1']:.3f}",
                  str(daily["fp"]), str(daily["fn"]), f"{score['stage_macro_f1']:.3f}", f"{score['weak_detected_days']}/81",
                  str(score["false_acceleration_days"]), f"{score['coverage']:.1%}"]
        rows.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")
        markdown.append("| " + " | ".join(values) + " |")
    markdown += ["", "## 开发检查与起点行为", "", "```json", json.dumps({"decisions": decisions, "onset": onset}, ensure_ascii=False, indent=2), "```", "",
                 "R1只替换三个候选／确认措辞；R2在R1上增加固定30日块观察、首次活动位置及扩展JSON契约。R2观察过程与输出校验作为组合因素，不能分离其内部机制，输出成本与失败必须同时报告。", "",
                 "每候选每组单次调用，历史R0另列，不用历史随机性替代同期控制；记录级重采样2000次仅描述六例组成，不估计调用方差。原始缺测两层未知，活动先于阶段。A/S/D支持866日，慢位移81日另评，过渡／未定340日不补阶段。", "",
                 "全部为已查看仿真开发记录与相关AI辅助标签，离线特征含未来信息，不是独立盲测、现场验证或在线预警。11/11风险检查：分组异质性、仿真外推、选择偏差、成功子集、基率、事后调整与均值回归、失败保留、多指标、选择自由度、因果归属和离线未来信息均检查。", "",
                 "证据与标签冻结不变，不挑最好重跑，不自动裁剪失败回答；网页展示实际观察摘要，未保存或补造内部思维链。", ""]
    (out / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    page = prefix + '<title>确认提示因素实验</title></head><body><h1>中性范围与逐块观察：实际实验</h1><p><a href="trace.html">全部51次实际调用：图片／完整提示／回答</a> · <a href="report.md">实验报告</a> · <a href="analysis.json">配对分析及开发检查</a> · <a href="costs.json">成本</a></p>'
    page += f'<p>新增真实调用{manifest["calls"]}次、{manifest["total_tokens"]:,} token、失败{manifest["failures"]}次，无重试。固定17候选、两张原始输入图、模型参数及阶段头；R2输出与校验负担增加，不能声称同调用数即同计算成本。</p>'
    headings = "".join(f"<th>{name}</th>" for name in ["配置", "P", "R", "活动F1", "FP日", "FN日", "阶段macro-F1", "弱日", "虚假A日", "覆盖"])
    page += f'<section><div class="scroll"><table><tr>{headings}</tr>{"".join(rows)}</table></div></section>'
    page += '<section><h2>预先冻结的开发检查及起点行为</h2><pre>' + html.escape(json.dumps({"decisions": decisions, "onset": onset}, ensure_ascii=False, indent=2)) + '</pre><p>减少FP、保留81弱日且活动F1不降，才通过本轮开发选择检查；这不等于优于历史严格单窗或强数值，更不等于独立优势。起点一致率只作诊断。</p></section>'
    for case_id in manifest["case_ids"]:
        page += f'<section><h2>{case_id}</h2><img loading="lazy" src="{case_id}.png" alt="{case_id}原始位移与活动"><details><summary>查看端到端阶段</summary><img loading="lazy" src="{case_id}-stages.png" alt="{case_id}阶段结果"></details></section>'
    page += '<p>六例已查看仿真、相关AI辅助标签、一个弱段和单次调用；阶段只评866个A/S/D日。结果仅作开发分析，未证明独立、现场或在线效果。</p></body></html>'
    (out / "index.html").write_text(page, encoding="utf-8")
    trace = prefix + '<title>确认提示因素调用审查</title></head><body><h1>全部同期实际确认调用</h1><p><a href="index.html">返回结果</a>。图片均无参考标签、无导数。blocks与evidence是模型可见观察摘要，不是内部思维链。</p>'
    for row in outputs:
        request_id = row["request_id"]
        trace += f'<details><summary>{html.escape(request_id)} · {row["start"]}–{row["stop"] - 1} · {row["status"]}</summary>'
        trace += '<p>' + html.escape(failures.get(request_id, "无契约失败")) + '</p><h3>完整提示</h3><pre>' + html.escape(row["prompt"]) + '</pre>'
        for image in row["images"]:
            trace += f'<a href="{html.escape(image)}"><img loading="lazy" src="{html.escape(image)}" alt="{html.escape(request_id)}实际送模图"></a>'
        trace += '<h3>原始回答与请求记录</h3><pre>' + html.escape(json.dumps(receipts[request_id], ensure_ascii=False, indent=2)) + '</pre></details>'
    (out / "trace.html").write_text(trace + '</body></html>', encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "report"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/landslide-confirmation-prompt-v1.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.config, args.out)
    elif args.mode == "run":
        execute(args.out)
    else:
        report(args.out)
