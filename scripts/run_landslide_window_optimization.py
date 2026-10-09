"""Freeze, execute and report paired window images and visual candidate confirmation."""
from __future__ import annotations

import argparse
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
    LandslideDiagnostics,
    LandslideInput,
    load_labels,
    observation_features,
    plot_case,
    plot_stages,
    save_json,
    sha,
)
from xgboost import XGBClassifier

from gnss_sim.landslide_error_audit import regular_windows, stage_error_partition
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import (
    REVIEW_TASK,
    WINDOW_TASK,
    aggregate_windows,
    apply_reviews,
    parse_window_activity,
    review_candidates,
)

METHODS = {
    "window_atlas": "逐窗：图册输入",
    "window_single": "逐窗：单窗输入",
    "window_single_review": "单窗＋纯视觉活动复核",
    "historical_record_atlas": "历史整记录图册＋同阶段头",
    "historical_visual_gate": "历史视觉活动＋同阶段头",
    "numeric_xgboost": "强数值活动＋同阶段头",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_verified(source, destination, expected):
    if sha(source) != expected:
        raise ValueError(f"Source changed: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def load_case_features(out, case_id):
    case = LandslideInput.model_validate_json(
        (out / "inputs" / case_id / "input.json").read_text(encoding="utf-8")
    )
    motions = {}
    canonical = hashlib.sha256(case.model_dump_json().encode()).hexdigest()
    for window in (31, 61, 91):
        motion = LandslideDiagnostics.model_validate_json(
            (out / "inputs" / case_id / f"motion-{window}.json").read_text(encoding="utf-8")
        )
        if motion.input_sha256 != canonical or motion.window_days != window:
            raise ValueError("Auxiliary observations do not match input")
        motions[window] = motion
    return case, observation_features(case, motions)


def render_window(case, noise, start, stop, path, config):
    values = np.array([
        row if row is not None else [np.nan] * 3 for row in case.displacement_mm
    ])
    fig, panels = plt.subplots(
        3, 1, figsize=config["single_plot_inches"], sharex=True
    )
    for axis, name in enumerate(("N", "E", "U")):
        panel = panels[axis]
        series = values[start:stop, axis]
        panel.plot(
            np.arange(start, stop), series, linewidth=0.8,
            color=("#147d72", "#cc6d4b", "#a38a32")[axis],
        )
        finite = series[np.isfinite(series)]
        if len(finite):
            low, high = float(finite.min()), float(finite.max())
            center = (low + high) / 2
            span = max(high - low, 6 * noise[axis], 10)
            panel.set_ylim(center - 0.75 * span, center + 0.75 * span)
        panel.set_xlim(start, stop - 1)
        panel.set_ylabel(name + " displacement (mm)")
        panel.grid(alpha=0.2)
    ticks = np.unique(np.r_[np.arange(start, stop, 30), stop - 1])
    panels[-1].set_xticks(ticks)
    panels[-1].set_xlabel("Original day index; missing observations remain gaps")
    fig.suptitle(f"{case.case_id} | Days {start}..{stop - 1} | Raw only; NO labels")
    fig.tight_layout()
    fig.savefig(path, dpi=config["single_plot_dpi"])
    plt.close(fig)


def prepare(config_path, out):
    config = read_json(config_path)
    source = ROOT / config["source_run"]
    numeric = ROOT / config["numeric_run"]
    numeric_manifest = read_json(numeric / "manifest.json")
    source_manifest = read_json(source / "manifest.json")
    if (config["temperature"], config["top_p"], config["max_output_tokens"],
            config["enable_thinking_requested"], config["repetitions"], config["retries"]) != (0.1, 0.3, 2048, False, 1, 0):
        raise ValueError("Protocol settings differ from the fixed VisualRequests transport")
    if source_manifest["status"] != "complete":
        raise ValueError("Historical source run must be complete")
    if config["case_ids"] != source_manifest["case_ids"]:
        raise ValueError("Case set differs from the common development comparison")
    if source_manifest["numeric_manifest_sha256"] != sha(numeric / "manifest.json"):
        raise ValueError("Historical and numeric runs do not share a calibrated source")
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    copy_verified(config_path, out / "protocol.json", sha(config_path))
    copy_verified(ROOT / config["labels"], out / "reference-labels.csv", config["labels_sha256"])
    copy_verified(numeric / "stage-xgboost.json", out / "stage-xgboost.json", sha(numeric / "stage-xgboost.json"))
    copy_verified(numeric / "predictions.json", out / "numeric-predictions.json", sha(numeric / "predictions.json"))
    copy_verified(source / "predictions.json", out / "historical-predictions.json", sha(source / "predictions.json"))
    packets = []
    for case_id in sorted(config["case_ids"]):
        input_path = Path(numeric_manifest["data"]) / "cases" / case_id / "input.json"
        copy_verified(input_path, out / "inputs" / case_id / "input.json", numeric_manifest["input_sha256"][case_id])
        for window in (31, 61, 91):
            path = DEFAULT_EVIDENCE / case_id / f"motion-{window}.json"
            copy_verified(path, out / "inputs" / case_id / path.name, sha(path))
        case, features = load_case_features(out, case_id)
        receipt = read_json(source / "requests" / f"{case_id}-locate-local.started.json")
        for name, expected in receipt["image_sha256"].items():
            copy_verified(source / name, out / "images" / name, expected)
        windows = regular_windows(len(case.dates), config["window_days"], config["stride_days"])
        for index, (start, stop) in enumerate(windows):
            single = f"images/{case_id}-single-{index:02d}.png"
            render_window(case, features.noise, start, stop, out / single, config)
            atlas = f"images/{case_id}-local-{index // 4 + 1}.png"
            arms = ("atlas", "single") if index % 2 == 0 else ("single", "atlas")
            for arm in arms:
                images = [f"images/{case_id}-raw.png", atlas if arm == "atlas" else single]
                packets.append({
                    "case_id": case_id, "method": "window_" + arm,
                    "request_id": f"{case_id}-window-{index:02d}-{arm}",
                    "start": start, "stop": stop,
                    "prompt": WINDOW_TASK.format(start=start, end=stop - 1),
                    "images": images,
                    "image_sha256": {name: sha(out / name) for name in images},
                })
    if len(packets) != config["screening_calls"]:
        raise ValueError("Screening schedule differs from configured request count")
    save_json(out / "screen-packets.json", packets)
    frozen_paths = [path for path in out.rglob("*") if path.is_file()]
    evidence_hashes = {str(path.relative_to(out)): sha(path) for path in frozen_paths}
    source_paths = [
        Path(__file__), ROOT / "scripts/run_landslide_comparison.py",
        ROOT / "src/gnss_sim/landslide_window_experiment.py",
        ROOT / "src/gnss_sim/landslide_local_experiment.py",
        ROOT / "src/gnss_sim/landslide_visual.py", ROOT / "src/gnss_sim/visual.py",
        ROOT / "src/gnss_sim/landslide_evaluation.py",
        ROOT / "src/gnss_sim/landslide_error_audit.py",
        ROOT / "src/gnss_sim/landslide_baselines.py",
    ]
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        source_hashes[str(relative)] = sha(path)
        copy_verified(path, out / "source" / relative, sha(path))
    save_json(out / "manifest.json", {
        "status": "prepared", "date": config["date"], "case_ids": config["case_ids"],
        "model": config["model"], "maximum_requests": config["maximum_requests"],
        "evidence_sha256": evidence_hashes, "source_sha256": source_hashes,
        "historical_manifest_sha256": sha(source / "manifest.json"),
        "numeric_manifest_sha256": sha(numeric / "manifest.json"),
        "new_calls": 0, "deviations": [], "scope": config["scope"],
    })
    print(f"Prepared {len(packets)} paired requests; no model calls", flush=True)


def execute_request(transport, packet, observed, out, failures):
    try:
        for name, expected in packet["image_sha256"].items():
            if sha(out / name) != expected:
                raise ValueError("Frozen image changed")
        payload = transport.ask(
            packet["request_id"], packet["prompt"], [out / name for name in packet["images"]]
        )
        activity = parse_window_activity(payload, observed, packet["start"], packet["stop"])
        return {**packet, "activity": activity.tolist(), "status": "success"}
    except (ValueError, RuntimeError) as error:
        failures[packet["request_id"]] = str(error)
        return {**packet, "activity": [-1] * len(observed), "status": "failed"}


def execute(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] != "prepared":
        raise ValueError("Only a never-started prepared experiment can execute")
    for name, expected in manifest["evidence_sha256"].items():
        if sha(out / name) != expected:
            raise ValueError("Prepared evidence changed")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Source changed after freeze")
    config = read_json(out / "protocol.json")
    cases, features = {}, {}
    for case_id in manifest["case_ids"]:
        cases[case_id], features[case_id] = load_case_features(out, case_id)
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_requests"])
    started = time.monotonic()
    manifest["status"] = "running_screen"
    save_json(out / "manifest.json", manifest)
    screen_results, failures = [], {}
    for index, packet in enumerate(read_json(out / "screen-packets.json")):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise RuntimeError("Experiment exceeded the predeclared 40-minute wall limit")
        result = execute_request(transport, packet, features[packet["case_id"]].observed, out, failures)
        screen_results.append(result)
        save_json(out / "screen-results.json", screen_results)
        save_json(out / "failures.json", failures)
        if (index + 1) % 12 == 0:
            print(f"Screen {index + 1}/144; failures={len(failures)}", flush=True)
    gates = {name: {} for name in ("window_atlas", "window_single", "window_single_review")}
    for name in ("window_atlas", "window_single"):
        for case_id in manifest["case_ids"]:
            selected = [row for row in screen_results if row["case_id"] == case_id and row["method"] == name]
            gates[name][case_id] = aggregate_windows(selected, features[case_id].observed)
    review_packets = []
    for case_id in manifest["case_ids"]:
        case = cases[case_id]
        candidates = review_candidates(gates["window_single"][case_id].activity)
        for index, (start, stop) in enumerate(candidates):
            name = f"images/{case_id}-review-{index:02d}.png"
            padding = config["review_context_days"]
            render_window(case, features[case_id].noise, max(0, start - padding), min(len(case.dates), stop + padding), out / name, config)
            images = [f"images/{case_id}-raw.png", name]
            review_packets.append({
                "case_id": case_id, "method": "window_single_review",
                "request_id": f"{case_id}-review-{index:02d}", "start": start, "stop": stop,
                "prompt": REVIEW_TASK.format(start=start, end=stop - 1), "images": images,
                "image_sha256": {image: sha(out / image) for image in images},
            })
    if len(review_packets) > config["review_maximum_candidates"]:
        raise ValueError("Candidate count exceeds predeclared review limit; no selective review")
    # Candidate spans and rendered evidence freeze before the first confirmation request.
    save_json(out / "review-packets.json", review_packets)
    manifest["review_packets_sha256"] = sha(out / "review-packets.json")
    manifest["review_candidate_count"] = len(review_packets)
    manifest["status"] = "running_review"
    save_json(out / "manifest.json", manifest)
    reviews = []
    for index, packet in enumerate(review_packets):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise RuntimeError("Experiment exceeded the predeclared 40-minute wall limit")
        reviews.append(execute_request(transport, packet, features[packet["case_id"]].observed, out, failures))
        save_json(out / "review-results.json", reviews)
        save_json(out / "failures.json", failures)
        print(f"Review {index + 1}/{len(review_packets)}; failures={len(failures)}", flush=True)
    for case_id in manifest["case_ids"]:
        selected = [row for row in reviews if row["case_id"] == case_id]
        if gates["window_single"][case_id].status == "failed":
            gates["window_single_review"][case_id] = gates["window_single"][case_id]
            continue
        gates["window_single_review"][case_id] = apply_reviews(
            gates["window_single"][case_id].activity, selected, features[case_id].observed
        )
    model = XGBClassifier()
    model.load_model(out / "stage-xgboost.json")
    historical = read_json(out / "historical-predictions.json")
    numeric = read_json(out / "numeric-predictions.json")
    methods = {name: {} for name in METHODS}
    for case_id in manifest["case_ids"]:
        feature = features[case_id]
        stage_codes = model.predict(feature.matrix).astype(int) + 1
        for name in ("window_atlas", "window_single", "window_single_review"):
            gate = gates[name][case_id]
            if gate.status == "failed":
                methods[name][case_id] = gate
            else:
                methods[name][case_id] = learned_stages(stage_codes, gate.activity, feature.supported)
        old = historical["activity_local"][case_id]
        if old["status"] == "failed":
            methods["historical_record_atlas"][case_id] = ActivityPrediction(
                np.array(old["activity"]), np.array(old["stage"]), status="failed"
            )
        else:
            methods["historical_record_atlas"][case_id] = learned_stages(stage_codes, np.array(old["activity"]), feature.supported)
        saved = historical["visual_xgboost_stage"][case_id]
        methods["historical_visual_gate"][case_id] = ActivityPrediction(
            np.array(saved["activity"]), np.array(saved["stage"]),
            status=saved["status"], stage_status=saved["stage_status"],
        )
        original = numeric["robust_xgboost_stage"][case_id]
        replay = learned_stages(stage_codes, np.array(original["activity"]), feature.supported)
        for field in ("activity", "stage", "status", "gated"):
            if replay.to_dict()[field] != original[field]:
                raise ValueError("Saved stage model failed to reproduce the numerical reference")
        methods["numeric_xgboost"][case_id] = replay
    save_json(out / "predictions.json", {
        name: {case_id: row.to_dict() for case_id, row in predictions.items()}
        for name, predictions in methods.items()
    })
    responses = [read_json(path) for path in sorted((out / "requests").glob("*.response.json"))]
    manifest.update(
        status="complete", new_calls=len(responses), failures=len(failures),
        total_tokens=sum(row.get("usage", {}).get("total_tokens", 0) for row in responses),
        seconds=time.monotonic() - started,
    )
    save_json(out / "manifest.json", manifest)
    report(out)


def report(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Scoring requires all planned inference to finish")
    config = read_json(out / "protocol.json")
    if sha(out / "reference-labels.csv") != config["labels_sha256"]:
        raise ValueError("Reference labels changed")
    cases = {case_id: load_case_features(out, case_id)[0] for case_id in manifest["case_ids"]}
    numeric_root = ROOT / config["numeric_run"]
    if sha(numeric_root / "manifest.json") != manifest["numeric_manifest_sha256"]:
        raise ValueError("Numerical source changed before reporting")
    numeric_manifest = read_json(numeric_root / "manifest.json")
    reference_cases = {}
    # The shared validator checks the entire label file, including non-evaluation cases.
    for case_id, expected in numeric_manifest["input_sha256"].items():
        path = Path(numeric_manifest["data"]) / "cases" / case_id / "input.json"
        if sha(path) != expected:
            raise ValueError("Reference observation calendar changed")
        reference_cases[case_id] = LandslideInput.model_validate_json(path.read_text(encoding="utf-8"))
    labels = load_labels(out / "reference-labels.csv", reference_cases)
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    methods = {}
    for name, records in read_json(out / "predictions.json").items():
        methods[name] = {
            case_id: ActivityPrediction(
                np.array(row["activity"]), np.array(row["stage"]),
                status=row["status"], stage_status=row["stage_status"],
            ) for case_id, row in records.items()
        }
    results, errors = {}, {}
    for name, records in methods.items():
        per_case = {case_id: evaluate_case(labels[case_id], prediction) for case_id, prediction in records.items()}
        results[name] = {"aggregate": aggregate_scores(list(per_case.values())), "per_case": per_case}
        errors[name] = {case_id: stage_error_partition(labels[case_id], prediction) for case_id, prediction in records.items()}
    save_json(out / "results.json", results)
    save_json(out / "stage-errors.json", errors)
    packets = read_json(out / "screen-packets.json") + read_json(out / "review-packets.json")
    costs = {}
    for name in ("window_atlas", "window_single", "window_single_review"):
        selected = [packet for packet in packets if packet["method"] == name]
        receipts = [read_json(out / "requests" / (packet["request_id"] + ".response.json")) for packet in selected]
        costs[name] = {
            "additional_calls": len(selected),
            "additional_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
            "service_seconds_sum": sum(row["seconds"] for row in receipts),
        }
    costs["window_single_review"]["pipeline_calls"] = costs["window_single"]["additional_calls"] + costs["window_single_review"]["additional_calls"]
    costs["window_single_review"]["pipeline_tokens"] = costs["window_single"]["additional_tokens"] + costs["window_single_review"]["additional_tokens"]
    save_json(out / "costs.json", costs)
    failures = read_json(out / "failures.json")
    table_rows = []
    def format_score(value):
        return "未定义" if value is None else f"{value:.3f}"

    for name, record in results.items():
        score = record["aggregate"]
        values = [
            METHODS[name], format_score(score["daily"]["precision"]),
            format_score(score["daily"]["recall"]), format_score(score["daily"]["f1"]),
            score["daily"]["fp"], score["daily"]["fn"],
            format_score(score["stage_macro_f1"]),
            format_score(score["weak_recall"]), score["false_acceleration_days"],
            f'{score["coverage"]:.3f}',
        ]
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    figures = []
    for case_id in manifest["case_ids"]:
        predictions = {METHODS[name]: rows[case_id] for name, rows in methods.items()}
        plot_case(cases[case_id], labels[case_id], predictions, out / f"{case_id}.png")
        plot_stages(cases[case_id], labels[case_id], predictions, out / f"{case_id}-stages.png")
        figures.append(f'<section><h2>{case_id}</h2><img src="{case_id}.png" alt="活动范围对比"><img src="{case_id}-stages.png" alt="同阶段头端到端对比"></section>')
    trace_cards = []
    for packet in packets:
        receipt = read_json(out / "requests" / (packet["request_id"] + ".response.json"))
        pictures = "".join(f'<a href="{html.escape(name)}"><img loading="lazy" src="{html.escape(name)}" alt="实际送模原图"></a>' for name in packet["images"])
        failure = failures.get(packet["request_id"], "无契约失败")
        trace_cards.append(
            f'<details><summary>{packet["request_id"]} · Days {packet["start"]}..{packet["stop"] - 1}</summary>'
            f'<p>{html.escape(failure)}</p>{pictures}<h3>实际完整提示</h3><pre>{html.escape(packet["prompt"])}</pre>'
            f'<h3>实际原始回答与调用记录</h3><pre>{html.escape(json.dumps(receipt, ensure_ascii=False, indent=2))}</pre></details>'
        )
    style = '<style>body{font:15px system-ui;margin:28px;background:#f5f8f7;color:#233d36}table{border-collapse:collapse;background:white}td,th{padding:9px;border:1px solid #ccd8d1}img{width:100%;max-width:1900px}p{line-height:1.8;max-width:1200px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:15px}details{margin:14px 0;border:1px solid #ccd8d1;padding:12px}summary{cursor:pointer}section{margin-top:35px}</style>'
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>视觉窗口优化实验</title>' + style
    document += '<h1>视觉窗口与平台确认：实际开发实验</h1><p>同一模型、每窗相同提示、相同调用数比较图册与单窗；单窗候选再做纯视觉活动复核。所有配置使用同一保存的XGBoost阶段头，未重新训练。历史整记录和数值方法为参考，调用预算不同。本批已查看、AI辅助标签，只有一个弱活动段；不代表独立泛化或在线预警。</p>'
    document += f'<p>本轮实际调用 {manifest["new_calls"]} 次，{manifest["total_tokens"]} token；失败请求／契约 {manifest["failures"]} 次。部分窗口失败保留未知，并进入投票分母；记录级评分之外单列全部调用失败。</p>'
    document += '<table><tr><th>配置</th><th>活动P</th><th>活动R</th><th>活动F1</th><th>FP日</th><th>FN日</th><th>阶段macro-F1</th><th>弱段召回</th><th>虚假A日</th><th>覆盖</th></tr>' + ''.join(table_rows) + '</table>'
    document += '<p><a href="trace.html">查看全部真实输入图、提示、观察依据、原始回答和失败</a> · <a href="results.json">逐例完整指标</a> · <a href="costs.json">调用成本</a> · <a href="protocol.json">冻结协议</a></p>'
    document += '<pre>' + html.escape(json.dumps(costs, ensure_ascii=False, indent=2)) + '</pre>' + ''.join(figures) + '</html>'
    (out / "index.html").write_text(document, encoding="utf-8")
    trace = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实际视觉调用审查</title>' + style
    trace += '<h1>实际视觉调用审查</h1><p>以下均为本轮实际发送的图片和提示。evidence是模型输出的可见观察摘要，不是内部思维链；没有补造缺失判断理由。</p>' + ''.join(trace_cards) + '</html>'
    (out / "trace.html").write_text(trace, encoding="utf-8")
    print(json.dumps({name: record["aggregate"]["daily"] for name, record in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/landslide-window-optimization-v1.json")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.config, args.out)
    elif args.action == "run":
        execute(args.out)
    else:
        report(args.out)


if __name__ == "__main__":
    main()
