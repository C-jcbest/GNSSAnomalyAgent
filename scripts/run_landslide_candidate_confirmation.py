"""Freeze and execute proposal confirmation with a paired calendar-metadata ablation."""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_landslide_comparison import (
    ROOT,
    LandslideInput,
    load_labels,
    plot_case,
    plot_stages,
    save_json,
    sha,
)
from run_landslide_window_optimization import (
    copy_verified,
    execute_request,
    load_case_features,
    read_json,
    render_window,
)
from xgboost import XGBClassifier

from gnss_sim.landslide_candidate_confirmation import (
    calendar_information,
    confirmation_base,
    proposal_pool,
)
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import (
    REVIEW_TASK,
    aggregate_windows,
    apply_reviews,
    parse_window_activity,
)

ARMS = ("proposal_visual", "proposal_calendar")
METHODS = {
    "strict_single": "缓存单窗：严格多数",
    "strict_single_review": "历史严格多数＋复核",
    "proposal_visual": "阳性/冲突候选＋原提示确认",
    "proposal_calendar": "同候选确认＋真实缺测日期",
    "atlas_reference": "历史逐窗图册参考",
    "numeric_reference": "强数值活动参考",
}
CONTROLS = {
    "strict_single": "window_single", "strict_single_review": "window_single_review",
    "atlas_reference": "window_atlas", "numeric_reference": "numeric_xgboost",
}


def source_windows(out, case_id):
    return [row for row in read_json(out / "cached-screen-packets.json") if row["case_id"] == case_id]


def prepare(config_path, out):
    config = read_json(config_path)
    source = ROOT / config["source_run"]
    parent = read_json(source / "manifest.json")
    parent_protocol = read_json(source / "protocol.json")
    if parent["status"] != "complete" or parent["case_ids"] != config["case_ids"]:
        raise ValueError("Cached source is incomplete or uses a different evaluation set")
    settings = (config["temperature"], config["top_p"], config["max_output_tokens"],
                config["enable_thinking_requested"], config["repetitions"], config["retries"])
    if settings != (0.1, 0.3, 2048, False, 1, 0) or config["arms"] != list(ARMS):
        raise ValueError("Protocol differs from the fixed transport or paired arms")
    if config["model"] != parent["model"]:
        raise ValueError("A candidate experiment must retain the cached model")
    numeric_root = ROOT / parent_protocol["numeric_run"]
    if sha(numeric_root / "manifest.json") != parent["numeric_manifest_sha256"]:
        raise ValueError("Numerical training source changed")
    numeric_manifest = read_json(numeric_root / "manifest.json")
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    copy_verified(config_path, out / "protocol.json", sha(config_path))
    copy_verified(source / "reference-labels.csv", out / "reference-labels.csv", parent_protocol["labels_sha256"])
    for name in ("predictions.json", "costs.json", "manifest.json"):
        copy_verified(source / name, out / ("parent-" + name), sha(source / name))
    copy_verified(source / "stage-xgboost.json", out / "stage-xgboost.json", parent["evidence_sha256"]["stage-xgboost.json"])
    # The common annotation validator consumes the entire corpus; inference uses six cases.
    for case_id, expected in numeric_manifest["input_sha256"].items():
        path = Path(numeric_manifest["data"]) / "cases" / case_id / "input.json"
        copy_verified(path, out / "inputs" / case_id / "input.json", expected)
    for case_id in config["case_ids"]:
        for window in (31, 61, 91):
            name = Path("inputs") / case_id / f"motion-{window}.json"
            copy_verified(source / name, out / name, parent["evidence_sha256"][str(name)])
    case_features = {case_id: load_case_features(out, case_id) for case_id in config["case_ids"]}
    cached = [row for row in read_json(source / "screen-results.json") if row["method"] == "window_single"]
    if len(cached) != 72:
        raise ValueError("Expected all 72 cached single-window judgments")
    copied_images = set()
    for row in cached:
        request_id = row["request_id"]
        receipt = read_json(source / "requests" / (request_id + ".response.json"))
        if receipt["prompt"] != row["prompt"]:
            raise ValueError("Cached prompt differs from its receipt")
        case, features = case_features[row["case_id"]]
        if row["status"] == "success":
            replay = parse_window_activity(json.loads(receipt["output"]), features.observed, row["start"], row["stop"])
            if replay.tolist() != row["activity"]:
                raise ValueError("Cached successful judgment was repaired or changed")
        elif row["status"] != "failed" or any(value != -1 for value in row["activity"]):
            raise ValueError("Cached failed judgment did not remain unknown")
        for name, expected in row["image_sha256"].items():
            if receipt["image_sha256"][Path(name).name] != expected:
                raise ValueError("Cached image hash differs from actual request")
            if name not in copied_images:
                copy_verified(source / name, out / name, expected)
                copied_images.add(name)
        copy_verified(source / "requests" / (request_id + ".response.json"),
                      out / "cached-receipts" / (request_id + ".response.json"),
                      sha(source / "requests" / (request_id + ".response.json")))
    save_json(out / "cached-screen-packets.json", cached)
    packets, pools = [], {}
    for case_id in config["case_ids"]:
        case, features = case_features[case_id]
        rows = [row for row in cached if row["case_id"] == case_id]
        pool = proposal_pool(rows, features.observed)
        pools[case_id] = pool.to_dict()
        for index, (start, stop) in enumerate(pool.spans):
            plot_start = max(0, start - config["review_context_days"])
            plot_stop = min(len(case.dates), stop + config["review_context_days"])
            name = f"images/{case_id}-proposal-{index:02d}.png"
            render_window(case, features.noise, plot_start, plot_stop, out / name, config)
            images = [f"images/{case_id}-raw.png", name]
            prompt = REVIEW_TASK.format(start=start, end=stop - 1)
            ordered_arms = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
            for arm in ordered_arms:
                packet_prompt = prompt
                if arm == "proposal_calendar":
                    packet_prompt += calendar_information(features.observed, plot_start, plot_stop)
                packets.append({"case_id": case_id, "method": arm,
                                "request_id": f"{case_id}-proposal-{index:02d}-{arm}",
                                "start": start, "stop": stop, "prompt": packet_prompt, "images": images,
                                "image_sha256": {image: sha(out / image) for image in images}})
    if len(packets) > config["maximum_requests"]:
        raise ValueError("Full proposal schedule exceeds budget; selective review is forbidden")
    save_json(out / "candidate-pools.json", pools)
    save_json(out / "packets.json", packets)
    source_paths = [Path(__file__), ROOT / "scripts/run_landslide_window_optimization.py",
                    ROOT / "scripts/run_landslide_comparison.py"]
    source_paths += [ROOT / "src/gnss_sim" / name for name in (
        "landslide_candidate_confirmation.py", "landslide_window_experiment.py",
        "landslide_local_experiment.py", "landslide_evaluation.py", "landslide_visual.py",
        "visual.py", "landslide_baselines.py", "landslide_diagnostics.py", "landslide.py")]
    source_hashes = {}
    for path in source_paths:
        name = str(path.relative_to(ROOT))
        source_hashes[name] = sha(path)
        copy_verified(path, out / "source" / name, sha(path))
    frozen = {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}
    save_json(out / "manifest.json", {
        "status": "prepared", "date": config["date"], "case_ids": config["case_ids"], "model": config["model"],
        "planned_calls": len(packets), "candidate_count": len(packets) // 2, "calls": 0,
        "evidence_sha256": frozen, "source_sha256": source_hashes,
        "parent_manifest_sha256": sha(source / "manifest.json"),
        "parent_screen_results_sha256": sha(source / "screen-results.json"),
        "labels_sha256": parent_protocol["labels_sha256"],
        "scope": config["limits"], "deviations": [],
    })
    print(f"Frozen {len(packets) // 2} candidates and {len(packets)} calls; reused 72 source judgments", flush=True)


def verify_freeze(out, manifest):
    for name, expected in manifest["evidence_sha256"].items():
        if sha(out / name) != expected:
            raise ValueError(f"Frozen evidence changed: {name}")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected or sha(out / "source" / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def execute(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] != "prepared":
        raise ValueError("Only a never-started experiment may run")
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
            raise RuntimeError("Predeclared candidate experiment timeout reached")
        outputs.append(execute_request(transport, packet, features[packet["case_id"]].observed, out, failures))
        save_json(out / "outputs.json", outputs)
        save_json(out / "failures.json", failures)
        manifest.update(calls=index + 1, failures=len(failures), seconds=time.monotonic() - started)
        save_json(out / "manifest.json", manifest)
        print(f"{index + 1}/{manifest['planned_calls']} {packet['request_id']}: {outputs[-1]['status']}", flush=True)
    model = XGBClassifier()
    model.load_model(out / "stage-xgboost.json")
    previous = read_json(out / "parent-predictions.json")
    methods = {name: {} for name in METHODS}
    for case_id in config["case_ids"]:
        feature = features[case_id]
        rows = source_windows(out, case_id)
        strict = aggregate_windows(rows, feature.observed)
        if strict.activity.tolist() != previous["window_single"][case_id]["activity"]:
            raise ValueError("Source majority did not reproduce its frozen activity")
        pool = proposal_pool(rows, feature.observed)
        base = confirmation_base(strict.activity, pool, feature.observed)
        codes = model.predict(feature.matrix).astype(int) + 1
        for arm in ARMS:
            reviews = [row for row in outputs if row["case_id"] == case_id and row["method"] == arm]
            if [(row["start"], row["stop"]) for row in reviews] != pool.spans:
                raise ValueError("Not all frozen candidates were reviewed")
            if strict.status == "failed":
                methods[arm][case_id] = strict.to_dict()
                continue
            gate = apply_reviews(base, reviews, feature.observed)
            methods[arm][case_id] = learned_stages(codes, gate.activity, feature.supported).to_dict()
        for name, source_name in CONTROLS.items():
            methods[name][case_id] = previous[source_name][case_id]
    save_json(out / "predictions.json", methods)
    receipts = [read_json(path) for path in (out / "requests").glob("*.response.json")]
    manifest.update(status="inference_complete", seconds=time.monotonic() - started,
                    total_tokens=sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts))
    save_json(out / "manifest.json", manifest)
    report(out)


def restore_prediction(row):
    return ActivityPrediction(np.array(row["activity"]), np.array(row["stage"]),
                              status=row["status"], gated=row["gated"], stage_status=row["stage_status"])


def report(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] not in {"inference_complete", "complete"}:
        raise ValueError("All inference must finish before scoring")
    verify_freeze(out, manifest)
    all_cases = {path.parent.name: LandslideInput.model_validate_json(path.read_text(encoding="utf-8"))
                 for path in (out / "inputs").glob("*/input.json")}
    labels = load_labels(out / "reference-labels.csv", all_cases)
    predictions = {name: {case_id: restore_prediction(row) for case_id, row in records.items()}
                   for name, records in read_json(out / "predictions.json").items()}
    results = {}
    for name, records in predictions.items():
        per_case = {case_id: evaluate_case(labels[case_id], row) for case_id, row in records.items()}
        results[name] = {"aggregate": aggregate_scores(list(per_case.values())), "per_case": per_case}
    save_json(out / "results.json", results)
    outputs = read_json(out / "outputs.json")
    costs, audit = {}, {"new_verified_calls": 0, "new_image_associations": 0, "failures_by_arm": {}}
    for arm in ARMS:
        rows = [row for row in outputs if row["method"] == arm]
        receipts = []
        for row in rows:
            receipt = read_json(out / "requests" / (row["request_id"] + ".response.json"))
            if receipt["prompt"] != row["prompt"]:
                raise ValueError("Actual confirmation prompt changed")
            for name, expected in row["image_sha256"].items():
                if receipt["image_sha256"][Path(name).name] != expected or sha(out / name) != expected:
                    raise ValueError("Actual confirmation image changed")
                audit["new_image_associations"] += 1
            if row["status"] == "success":
                case = all_cases[row["case_id"]]
                observed = np.array([value is not None for value in case.displacement_mm])
                replay = parse_window_activity(json.loads(receipt["output"]), observed, row["start"], row["stop"])
                if replay.tolist() != row["activity"]:
                    raise ValueError("Confirmation response was altered")
            elif any(value != -1 for value in row["activity"]):
                raise ValueError("Failed confirmation was repaired")
            receipts.append(receipt)
            audit["new_verified_calls"] += 1
        cached_cost = read_json(out / "parent-costs.json")["window_single"]
        tokens = sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts)
        costs[arm] = {"new_confirmation_calls": len(rows), "new_confirmation_tokens": tokens,
                      "cached_screen_calls": cached_cost["additional_calls"],
                      "pipeline_calls": len(rows) + cached_cost["additional_calls"],
                      "pipeline_tokens": tokens + cached_cost["additional_tokens"],
                      "service_seconds_sum": sum(row["seconds"] for row in receipts)}
        audit["failures_by_arm"][arm] = sum(row["status"] == "failed" for row in rows)
    save_json(out / "costs.json", costs)
    save_json(out / "execution-audit.json", audit)
    candidate_support = {}
    pools = read_json(out / "candidate-pools.json")
    for case_id in manifest["case_ids"]:
        covered = np.zeros(len(labels[case_id]), dtype=bool)
        for start, stop in pools[case_id]["spans_half_open"]:
            covered[start:stop] = True
        actual = np.array([int(row["activity_label"]) for row in labels[case_id]])
        weak = np.array([row["feature"] == "slow_displacement" for row in labels[case_id]])
        candidate_support[case_id] = {"covered_positive_days": int(np.count_nonzero(covered & (actual == 1))),
                                      "uncovered_positive_days": int(np.count_nonzero(~covered & (actual == 1))),
                                      "covered_negative_days": int(np.count_nonzero(covered & (actual == 0))),
                                      "covered_weak_days": int(np.count_nonzero(covered & weak)),
                                      "interpretation": "Candidate coverage diagnostic, not an activity score"}
    save_json(out / "candidate-support.json", candidate_support)
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    figures = []
    for case_id in manifest["case_ids"]:
        rows = {METHODS[name]: records[case_id] for name, records in predictions.items()}
        plot_case(all_cases[case_id], labels[case_id], rows, out / f"{case_id}.png")
        plot_stages(all_cases[case_id], labels[case_id], rows, out / f"{case_id}-stages.png")
        figures.append(f'<h2>{case_id}</h2><img src="{case_id}.png" alt="原始位移与活动对比"><img src="{case_id}-stages.png" alt="活动内阶段对比">')
    table = []
    def number(value):
        return "未定义" if value is None else f"{value:.3f}"
    for name, record in results.items():
        score = record["aggregate"]
        values = [METHODS[name], number(score["daily"]["precision"]), number(score["daily"]["recall"]),
                  number(score["daily"]["f1"]), score["daily"]["fp"], score["daily"]["fn"],
                  number(score["stage_macro_f1"]), f'{score["weak_detected_days"]}/81',
                  score["false_acceleration_days"], number(score["coverage"])]
        table.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    style = '<style>body{font:15px system-ui,"Microsoft YaHei";background:#f4f7f5;color:#223e35;margin:28px}p{line-height:1.9;max-width:1200px}table{border-collapse:collapse;background:white;min-width:950px}th,td{padding:10px;border:1px solid #cad8ce}img{width:100%;max-width:1900px}.scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}details{margin:12px 0;padding:12px;border:1px solid #cad8ce}summary{cursor:pointer}a{color:#196653}</style>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>候选保留与原始位移确认</title>' + style
    page += '<h1>候选保留 → 原始位移确认 → 活动内阶段</h1><p>缓存72次单窗判断；任一阳性或成功窗负/未知冲突进入候选，候选不能直接赋活动或阶段。两个确认组使用相同候选、相同图片与原提示，仅第二组添加真实缺测日期。图像纵轴、模型、阶段头不变。全部候选统一处理，失败保留未知。</p>'
    page += f'<p>本轮实际新增{manifest["calls"]}次调用、{manifest["total_tokens"]} token，失败{manifest["failures"]}次。先推理后评分；全部是已查看开发记录、AI辅助标签，不能声称独立或现场泛化。</p>'
    page += '<div class="scroll"><table><tr><th>配置</th><th>P</th><th>R</th><th>活动F1</th><th>FP日</th><th>FN日</th><th>阶段macro-F1</th><th>弱段</th><th>虚假A日</th><th>覆盖</th></tr>' + ''.join(table) + '</table></div>'
    page += '<p><a href="trace.html">缓存定位与全部新增确认：真实图片、完整提示和原始回答</a> · <a href="results.json">逐例指标</a> · <a href="candidate-support.json">候选覆盖（不是确认成绩）</a> · <a href="costs.json">成本</a> · <a href="protocol.json">冻结协议</a></p>'
    page += '<pre>' + html.escape(json.dumps(costs, ensure_ascii=False, indent=2)) + '</pre>' + ''.join(figures) + '</html>'
    (out / "index.html").write_text(page, encoding="utf-8")
    cards = []
    failures = read_json(out / "failures.json")
    for cached, rows in ((True, read_json(out / "cached-screen-packets.json")), (False, outputs)):
        for row in rows:
            receipt_dir = "cached-receipts" if cached else "requests"
            receipt = read_json(out / receipt_dir / (row["request_id"] + ".response.json"))
            pictures = ''.join(f'<a href="{html.escape(name)}"><img loading="lazy" src="{html.escape(name)}" alt="实际送模原图"></a>' for name in row["images"])
            provenance = "缓存定位，未重跑" if cached else "本轮新增确认"
            cards.append(f'<details><summary>{provenance} · {row["request_id"]} · {row["start"]}–{row["stop"] - 1}</summary><p>{html.escape(failures.get(row["request_id"], "无本轮契约失败"))}</p>{pictures}<h3>完整提示</h3><pre>{html.escape(row["prompt"])}</pre><h3>原始回答与调用记录</h3><pre>{html.escape(json.dumps(receipt, ensure_ascii=False, indent=2))}</pre></details>')
    trace = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>候选确认调用审查</title>' + style
    trace += '<h1>缓存定位与新增候选确认的实际调用</h1><p>evidence仅为模型可见观察摘要，不是内部思维链。定位与确认来源分别标识，不把缓存当新增调用；图片没有评价标签或导数。</p>' + ''.join(cards) + '</html>'
    (out / "trace.html").write_text(trace, encoding="utf-8")
    manifest["status"] = "complete"
    save_json(out / "manifest.json", manifest)
    print(json.dumps({name: record["aggregate"] for name, record in results.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/landslide-candidate-confirmation-v1.json")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.config, args.out)
    elif args.action == "run":
        execute(args.out)
    else:
        report(args.out)
