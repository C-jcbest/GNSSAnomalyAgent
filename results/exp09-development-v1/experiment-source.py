"""Fixed three-member visual votes over original GNSS return-pair proposals."""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiment_exp07 import score
from summarize_exp06 import pooled_bootstrap

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.vision import SYSTEM, call_visual, credentials

PROTOCOL = Path("configs/experiment-exp09.json")


def consensus_repair(observation: dict, responses: list[dict], candidates: dict) -> dict:
    if len(responses) != 3:
        raise ValueError("EXP09 requires the three planned members")
    member_results = [repair_return_pairs(observation, response, candidates) for response in responses]
    votes = Counter()
    for result in member_results:
        # Count actual repairs, so near dates map through the same original policy.
        votes.update({repair["proposal_id"] for repair in result["semantic_repairs"]})
    selected = []
    for proposal in candidates["proposals"]:
        if votes[proposal["id"]] < 2:
            continue
        event = {key: value for key, value in proposal.items() if key != "id"}
        event.update(status="confirmed", evidence=f"Planned independent visual votes: {votes[proposal['id']]}/3")
        selected.append(event)
    aggregated = {"status": "success", "prediction": {"events": selected}}
    result = repair_return_pairs(observation, aggregated, candidates)
    result.update(member_votes=dict(votes), member_statuses=[response["status"] for response in responses],
                  aggregation="2 of 3 planned members must select the same numerical return-pair proposal")
    return result


def read_run(run: Path):
    registration = read_json(run / "registration.json")
    if registration["split"] != "validation":
        raise ValueError("this frozen EXP09 stage is development only")
    if registration["source_hashes"] != source_hashes():
        raise ValueError("production source differs from baseline registration")
    dataset = Path(registration["dataset"])
    manifest = read_json(dataset / "manifest.json")
    if sha256(dataset / "manifest.json") != registration["manifest_sha256"]:
        raise ValueError("registered dataset changed")
    records = [record for record in manifest["records"] if record["station_id"] in registration["station_ids"]]
    return registration, dataset, records


def prepare(run: Path, visual: Path, env: Path):
    registered, dataset, records = read_run(run)
    protocol = read_json(PROTOCOL)
    if registered["seed"] != protocol["development_seed"]:
        raise ValueError("use the newly registered development seed")
    settings = credentials(env)
    if settings[2] != protocol["model"]:
        raise ValueError("model differs from the frozen protocol")
    visual.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, visual / "experiment-source.py")
    pending = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numeric = read_json(run / "predictions/rf_split_temporal" / station / "result.json")
        base = visual / "inputs" / station
        payload = prepare_pair_focus(observation, numeric, base, settings[2], SYSTEM)
        for member in range(1, 4):
            work = visual / f"member{member}" / station
            work.mkdir(parents=True, exist_ok=False)
            if payload is None:
                shutil.copyfile(base / "result.json", work / "result.json")
                continue
            shutil.copyfile(base / "request.json", work / "request.json")
            pending.append({"station": station, "member": member,
                            "request_sha256": sha256(work / "request.json")})
    pilot = sorted({job["station"] for job in pending})[:6]
    write_json(visual / "registration.json", {
        "dataset": str(dataset.resolve()), "baseline_run": str(run.resolve()),
        "station_ids": registered["station_ids"], "split": registered["split"],
        "source_hashes": source_hashes(), "script_sha256": sha256(Path(__file__)),
        "helper_sha256": sha256(Path(__file__).with_name("experiment_exp07.py")),
        "protocol": protocol, "protocol_sha256": sha256(PROTOCOL),
        "protocol_text_sha256": sha256(Path(protocol["protocol_path"])),
        "pending": pending, "pilot_ids": pilot, "request_upper_bound": len(pending)})
    print({"pilot_ids": pilot, "requests": len(pending)}, flush=True)


def infer(visual: Path, env: Path, phase: str):
    frozen = read_json(visual / "registration.json")
    if frozen["source_hashes"] != source_hashes() or frozen["script_sha256"] != sha256(Path(__file__)):
        raise ValueError("registered experiment source changed")
    if frozen["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("registered protocol changed")
    if frozen["helper_sha256"] != sha256(Path(__file__).with_name("experiment_exp07.py")):
        raise ValueError("registered scoring helper changed")
    if frozen["protocol_text_sha256"] != sha256(Path(frozen["protocol"]["protocol_path"])):
        raise ValueError("registered protocol text changed")
    sentinel = visual / f"{phase}-started.json"
    if sentinel.exists():
        raise ValueError("phase already started; no implicit retries")
    settings = credentials(env)
    if settings[2] != frozen["protocol"]["model"]:
        raise ValueError("registered model changed")
    dataset = Path(frozen["dataset"])
    records = {record["station_id"]: record for record in read_json(dataset / "manifest.json")["records"]}
    jobs = [job for job in frozen["pending"]
            if (job["station"] in frozen["pilot_ids"]) == (phase == "pilot")]
    write_json(sentinel, {"requests": len(jobs), "retry_count": 0})

    def request(job):
        work = visual / f"member{job['member']}" / job["station"]
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("planned member input changed")
        observation = read_json(dataset / records[job["station"]]["observation"])
        result = call_visual(observation, "pair_focus", work, settings)
        print(job["station"], f"member{job['member']}", result["status"], flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(request, jobs))
    write_json(visual / f"{phase}-completed.json", {"requests": len(jobs)})


def summarize(run: Path, visual: Path, output: Path):
    registered, dataset, records = read_run(run)
    frozen = read_json(visual / "registration.json")
    if frozen["baseline_run"] != str(run.resolve()):
        raise ValueError("visual members belong to a different baseline run")
    for phase in ("pilot", "remaining"):
        if not (visual / f"{phase}-completed.json").exists():
            raise ValueError("both planned execution phases must finish")
    output.mkdir(parents=True, exist_ok=False)
    methods = ["consensus_repair", "member1_repair", "member2_repair", "member3_repair",
               "segment_numeric", "rf_split_temporal", "rf_balanced"]
    predictions = {method: {} for method in methods}
    costs = {f"member{member}": {"requests": 0, "failures": 0} for member in range(1, 4)}
    disagreements = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        candidates = read_json(visual / "inputs" / station / "candidates.json")
        responses = []
        repair_sets = []
        for member in range(1, 4):
            response = read_json(visual / f"member{member}" / station / "result.json")
            result = repair_return_pairs(observation, response, candidates)
            predictions[f"member{member}_repair"][station] = result
            responses.append(response)
            repair_sets.append(sorted(repair["proposal_id"] for repair in result["semantic_repairs"]))
            costs[f"member{member}"]["requests"] += response["request_count"]
            costs[f"member{member}"]["failures"] += response["status"] != "success"
        if any(repairs != repair_sets[0] for repairs in repair_sets[1:]):
            disagreements.append({"station": station, "member_repairs": repair_sets})
        predictions["consensus_repair"][station] = consensus_repair(observation, responses, candidates)
        for method in ("segment_numeric", "rf_split_temporal", "rf_balanced"):
            predictions[method][station] = read_json(run / "predictions" / method / station / "result.json")
    summaries, rows = score(dataset, records, predictions)
    candidate = summaries["consensus_repair"]
    single = summaries["member1_repair"]
    member_mean = float(np.mean([summaries[f"member{member}_repair"]["types_daily_macro_f1"]
                                 for member in range(1, 4)]))
    difference = candidate["types_daily_macro_f1"] - single["types_daily_macro_f1"]
    mean_difference = candidate["types_daily_macro_f1"] - member_mean
    worst = min(candidate["types"][kind]["daily_micro"]["f1"] - single["types"][kind]["daily_micro"]["f1"]
                for kind in single["types"])
    accepted = difference >= .02 and mean_difference >= .02 and worst >= -.05
    summary = {"experiment_id": "GNSS-VLM-EXP09", "split": "validation", "seed": registered["seed"],
               "methods": summaries, "member_mean_primary_f1": member_mean, "costs": costs,
               "disagreements": disagreements, "registered_visual_run": str(visual.resolve()),
               "decision": {"eligible_for_new_test": accepted, "difference_from_first_member": difference,
                            "difference_from_member_mean": mean_difference, "worst_type_difference": worst},
               "paired_bootstrap_vs_first": pooled_bootstrap(rows["consensus_repair"], rows["member1_repair"], registered["seed"])}
    write_json(output / "summary.json", summary)
    write_json(output / "per-station.json", rows)
    shutil.copyfile(__file__, output / "experiment-source.py")
    shutil.copyfile(visual / "registration.json", output / "registration.json")
    for method in methods[:4]:
        for station, result in predictions[method].items():
            write_json(output / "predictions" / method / station / "result.json", result)
    lines = ["# EXP09：GNSS歧义返回对视觉重复判断与多数修复开发结果", "",
             "新开发种子2026101414，24验证站；主指标四类等权日F1，不做point adjustment。", "",
             "| 方法 | 类型日F1 | 类型事件F1≥0.5 | 通用日F1 |", "|---|---:|---:|---:|"]
    for method in methods:
        values = summaries[method]
        lines.append(f"| {method} | {values['types_daily_macro_f1']:.4f} | {values['types_event_macro_f1']['iou_0.5']:.4f} | {values['generic_daily_micro']['f1']:.4f} |")
    total = sum(cost["requests"] for cost in costs.values())
    failures = sum(cost["failures"] for cost in costs.values())
    lines.extend(["", f"单次三成员主指标均值{member_mean:.4f}；多数相对首成员{difference:+.4f}、相对均值{mean_difference:+.4f}。",
                  f"最差类型相对首成员{worst:+.4f}；不同修复成员的站数{len(disagreements)}。",
                  f"计划内实际请求{total}，失败{failures}，无重试；多数成本为首成员的3倍，仅歧义站请求。", "",
                  f"决定：**{'通过开发条件，需另登记新正式测试' if accepted else '未通过开发条件，取消集成并保留EXP06'}**。",
                  "多数须同时比首成员和三成员均值至少+0.02，任一类型下降不超过0.05。",
                  "本次为开发结果，不能称未见正式测试或近期作者方法完整复现。", ""])
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    labels = ["Member 1 (single SDR-VLM)", "Member 2", "Member 3", "Three-member mean", "Majority repair"]
    values = [summaries[f"member{member}_repair"]["types_daily_macro_f1"] for member in range(1, 4)]
    values += [member_mean, candidate["types_daily_macro_f1"]]
    bars = axis.barh(labels, values, color=["#8a9caa"] * 4 + ["#245c70"])
    for bar, value in zip(bars, values):
        axis.text(value + .01, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Four-type pooled daily macro F1")
    axis.set_title("EXP09: fixed three-member visual repair, 24 development stations")
    figure.tight_layout()
    figure.savefig(output / "comparison.png")
    figure.savefig(output / "comparison.svg")
    plt.close(figure)
    print(summary["decision"])
    print({method: value["types_daily_macro_f1"] for method, value in summaries.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "remaining", "summarize"))
    parser.add_argument("--run", type=Path, default=Path("runs/exp09-development-baseline-v1"))
    parser.add_argument("--visual", type=Path, default=Path("runs/exp09-development-visual-v1"))
    parser.add_argument("--out", type=Path, default=Path("results/exp09-development-v1"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.visual, args.env_file)
    elif args.stage == "summarize":
        summarize(args.run, args.visual, args.out)
    else:
        infer(args.visual, args.env_file, args.stage)


if __name__ == "__main__":
    main()
