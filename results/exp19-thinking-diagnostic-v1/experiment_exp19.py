"""Paired native-thinking intervention on byte-identical visual/text content."""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from exp14_fingerprint import apply, call
from experiment_exp07 import score
from experiment_exp08 import data_records
from experiment_exp17 import settings

from gnss_research.files import read_json, sha256, source_hashes, write_json

MODES = ("reference_distribution", "distribution_text", "no_evidence")
PROTOCOL = Path("docs/experiment-exp19.md")


def prepare(run: Path, parent: Path, visual: Path):
    registered, dataset, manifest, records = data_records(run)
    visual.mkdir(parents=True, exist_ok=False)
    pending = []
    copied = []
    for record in records:
        station = record["station_id"]
        for mode in MODES:
            original = parent / "predictions" / mode / station
            work = visual / "predictions" / mode / station
            work.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(original / "candidates.json", work / "candidates.json")
            if not (original / "request.json").exists():
                shutil.copyfile(original / "result.json", work / "result.json")
                continue
            request = read_json(original / "request.json")
            request.update(enable_thinking=True, thinking_budget=2048)
            write_json(work / "request.json", request)
            pending.append({"mode": mode, "station": station,
                            "request_sha256": sha256(work / "request.json")})
            copied.append({"mode": mode, "station": station,
                "original_request_path": str((original / "request.json").resolve()),
                "original_request_sha256": sha256(original / "request.json")})
    protocol = PROTOCOL
    write_json(visual / "registration.json", {"experiment_id": "GNSS-VLM-EXP19",
        "baseline_run": str(run.resolve()), "parent": str(parent.resolve()),
        "parent_registration_sha256": sha256(parent / "registration.json"),
        "baseline_registration_sha256": sha256(run / "registration.json"),
        "dataset": str(dataset.resolve()), "seed": manifest["seed"], "split": registered["split"],
        "station_ids": registered["station_ids"], "model": settings()[2], "pending": pending,
        "pilot_ids": sorted({row["station"] for row in pending})[:1], "original_requests": copied,
        "source_hashes": source_hashes(), "source_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(protocol), "thinking_budget": 2048})
    shutil.copyfile(__file__, visual / Path(__file__).name)
    print({"requests": len(pending), "pilot_ids": sorted({row["station"] for row in pending})[:1]}, flush=True)


def infer(visual: Path, phase: str):
    frozen = read_json(visual / "registration.json")
    if source_hashes() != frozen["source_hashes"] or sha256(Path(__file__)) != frozen["source_sha256"]:
        raise ValueError("source changed")
    if sha256(PROTOCOL) != frozen["protocol_sha256"]:
        raise ValueError("protocol changed")
    dataset = Path(frozen["dataset"])
    records = {row["station_id"]: row for row in read_json(dataset / "manifest.json")["records"]}
    jobs = [job for job in frozen["pending"]
            if (job["station"] in frozen["pilot_ids"]) == (phase == "pilot")]
    marker = visual / f"{phase}-started.json"
    if marker.exists():
        raise ValueError("phase already started; no retry")
    write_json(marker, {"requests": len(jobs), "retry_count": 0})

    def request(job):
        work = visual / "predictions" / job["mode"] / job["station"]
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("request changed")
        observation = read_json(dataset / records[job["station"]]["observation"])
        result = call(observation, work, settings())
        print(job["station"], job["mode"], result["status"],
              result.get("pair_decision", {}).get("pair_ids"), flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(request, jobs))
    write_json(visual / f"{phase}-completed.json", {"requests": len(jobs)})


def summarize(run: Path, visual: Path, output: Path):
    registered, dataset, manifest, records = data_records(run)
    frozen = read_json(visual / "registration.json")
    parent = Path(frozen["parent"])
    if sha256(parent / "registration.json") != frozen["parent_registration_sha256"]:
        raise ValueError("parent changed")
    for phase in ("pilot", "remaining"):
        if not (visual / f"{phase}-completed.json").exists():
            raise ValueError("incomplete experiment")
    output.mkdir(parents=True, exist_ok=False)
    predictions = {mode + suffix: {} for mode in MODES for suffix in ("", "_nonthinking")}
    costs = {mode: {"requests": 0, "failures": 0, "visible_reasoning_responses": 0,
                    "reasoning_tokens": 0} for mode in MODES}
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        for mode in MODES:
            for suffix, root in (("", visual), ("_nonthinking", parent)):
                work = root / "predictions" / mode / station
                result = read_json(work / "result.json")
                candidates = read_json(work / "candidates.json")
                predictions[mode + suffix][station] = apply(observation, candidates["numerical_prediction"],
                                                           candidates["graph"], result)
                if suffix:
                    continue
                costs[mode]["requests"] += result.get("request_count", 0)
                costs[mode]["failures"] += result["status"] != "success"
                if (work / "response.json").exists():
                    raw = read_json(work / "response.json")
                    message = raw["choices"][0]["message"]
                    costs[mode]["visible_reasoning_responses"] += bool(message.get("reasoning_content"))
                    costs[mode]["reasoning_tokens"] += raw.get("usage", {}).get(
                        "completion_tokens_details", {}).get("reasoning_tokens", 0)
                for name in ("response.json", "result.json"):
                    path = work / name
                    if path.exists():
                        target = output / "responses" / mode / station / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
    summaries, rows = score(dataset, records, predictions)
    main = summaries["reference_distribution"]["types_daily_macro_f1"]
    delta = main - summaries["reference_distribution_nonthinking"]["types_daily_macro_f1"]
    text_delta = main - summaries["distribution_text"]["types_daily_macro_f1"]
    summary = {"experiment_id": "GNSS-VLM-EXP19", "seed": manifest["seed"],
        "station_ids": registered["station_ids"], "methods": summaries, "costs": costs,
        "difference_vs_nonthinking": delta, "difference_vs_same_mode_text": text_delta,
        "eligible_for_1512_development": delta >= .01 and text_delta >= -.005}
    write_json(output / "summary.json", summary)
    write_json(output / "per-station.json", rows)
    shutil.copyfile(visual / "registration.json", output / "registration.json")
    shutil.copyfile(__file__, output / Path(__file__).name)
    lines = ["# EXP19：固定输入的原生思考模式对照（NTV-VLM）", "",
             f"种子{manifest['seed']}，{len(records)}已见开发站，thinking_budget=2048。", "",
             "| 方法 | 四类日F1 | 四类事件F1≥.5 |", "|---|---:|---:|"]
    for method, values in summaries.items():
        lines.append(f"| {method} | {values['types_daily_macro_f1']:.4f} | "
                     f"{values['types_event_macro_f1']['iou_0.5']:.4f} |")
    lines.extend(["", f"进入新1512模式对照：{summary['eligible_for_1512_development']}。",
                  f"相对非思考{delta:+.4f}，相对相同模式文字{text_delta:+.4f}。", ""])
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "remaining", "summarize"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.parent is None:
            parser.error("--parent required")
        prepare(args.run, args.parent, args.visual)
    elif args.stage == "summarize":
        if args.out is None:
            parser.error("--out required")
        summarize(args.run, args.visual, args.out)
    else:
        infer(args.visual, args.stage)


if __name__ == "__main__":
    main()
