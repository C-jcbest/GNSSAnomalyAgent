"""Recover discarded PELT boundaries for optional local image review after SDR."""
from __future__ import annotations

import argparse
import base64
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp14_fingerprint import PairDecision, apply, call
from exp20_range_metrics import aggregate, evaluate
from experiment_exp08 import data_records
from experiment_exp22 import MODES, cases
from matplotlib.ticker import MultipleLocator

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, write_json
from gnss_research.schema import observed_values
from gnss_research.segmentation import returning_pair
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.vision import SYSTEM, call_visual, credentials

PROTOCOL = Path("docs/experiment-exp23.md")
PROMPT = """Review proposed temporary shift-and-return intervals in GNSS displacement.
Reply with one JSON object {"pair_ids":[],"evidence":"short observed reason"}.
Choose supplied IDs only; empty is valid. The earlier boundary was found by piecewise
linear segmentation but was NOT accepted as a jump. It may be a smooth bend, noise, or
an actual overlooked offset. The later boundary is an unpaired numerical step; it may
be independent. Neither boundary is ground truth. Only accept if the original N/E/U
curves support an offset platform and subsequent return toward the continuing local
trajectory, allowing underlying gradual displacement. A smoothly changing slope alone
is not a temporary level shift. Do not accept merely because a candidate is supplied.
Left panels are ORIGINAL observations; right panels subtract normal-history linear and
annual reference only. This reference may extrapolate poorly: verify original panels.
Axes have their own labeled vertical scales; dotted lines mark candidate boundaries.
Missing observations remain absent. If the earlier boundary is the first observation
after a gap, decide only whether the OBSERVED elevated platform returns later; its start
is the first observed affected day, not a claimed exact jump date inside the gap.
Without actual curve evidence, leave unresolved pairs unselected. Do not infer an event
count, use a universal amplitude/speed threshold, or label physical activity. No two
selected pairs may share an endpoint. Existing activity/spikes are handled separately.
"""


def recovery_graph(observation: dict, numeric: dict, base: dict) -> dict:
    """Restore only unused segment starts, without duplicating refined known jumps."""
    known = [jump["index"] for jump in numeric["jumps"]]
    steps = sorted(event["start"] for event in base["prediction"]["events"]
                   if event["kind"] == "step" and event["status"] == "confirmed")
    existing = [event for event in base["prediction"]["events"] if event["kind"] == "transient"]
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    restored = [segment["start"] for segment in numeric["segments"][1:]
                if segment["start"] > observation["background_end"]
                and observed[segment["start"]]
                and not any(abs(segment["start"] - day) <= 3 for day in known)]
    pairs = []
    for first in restored:
        for second in steps:
            if not 0 < second - first <= 120:
                continue
            if any(max(first, event["start"]) <= min(second - 1, event["end"]) for event in existing):
                continue
            pairs.append({"id": f"R{len(pairs) + 1}", "start": first, "end": second - 1,
                          "first_id": f"D{first}", "second_id": f"J{second}"})
    indices = sorted({day for pair in pairs for day in (pair["start"], pair["end"] + 1)})
    return {"pairs": pairs, "jumps": [{"id": f"B{day}", "index": day} for day in indices]}


def render_local(observation: dict, graph: dict, target: Path):
    evidence = numerical_evidence(observation)
    indices = [jump["index"] for jump in graph["jumps"]]
    left = max(observation["background_end"] + 1, min(indices) - 30)
    right = min(len(evidence.values), max(indices) + 31)
    time = np.arange(left, right)
    figure, panels = plt.subplots(3, 2, figsize=(14, 8), dpi=180, sharex=True)
    for column, values in enumerate((evidence.values, evidence.residual)):
        for axis in range(3):
            panel = panels[axis, column]
            panel.plot(time, values[left:right, axis], ".-", markersize=3, linewidth=.9)
            for index in indices:
                panel.axvline(index, color="#a34040", linestyle=":", linewidth=.8)
                panel.text(index, .95, str(index), transform=panel.get_xaxis_transform(), fontsize=8)
            panel.set_ylabel(f"{'NEU'[axis]} (mm)")
            panel.xaxis.set_major_locator(MultipleLocator(10))
            panel.xaxis.set_minor_locator(MultipleLocator(1))
            panel.grid(alpha=.22)
            if axis == 2:
                panel.set_xlabel("Original daily index; gaps unfilled")
    panels[0, 0].set_title("Original displacement")
    panels[0, 1].set_title("Normal-history residual (auxiliary)")
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)


def configured(model: str):
    base, key, _ = credentials(Path(".env"))
    return base, key, model


def prepare_base(baseline: Path, output: Path):
    registered, dataset, manifest, records = data_records(baseline)
    output.mkdir(parents=True, exist_ok=False)
    selected = []
    for record in sorted(records, key=lambda item: item["station_id"]):
        case = {"run": str(baseline), "dataset": str(dataset), "seed": manifest["seed"],
                "record": record, "id": f"{manifest['seed']}-{record['station_id']}"}
        observation = read_json(dataset / record["observation"])
        numeric = read_json(baseline / "predictions/segment_numeric" / record["station_id"] / "result.json")
        prepare_pair_focus(observation, numeric, output / "base" / case["id"], "qwen3.8-flash", SYSTEM)
        selected.append(case)
    write_json(output / "base-registration.json", {"cases": selected, "split": registered["split"],
        "source": str(baseline), "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__))})
    print("Original Flash base prepared:", len(selected), "stations", flush=True)


def infer_base(output: Path):
    frozen = read_json(output / "base-registration.json")
    if (output / "base-started.json").exists():
        raise ValueError("no retries")
    write_json(output / "base-started.json", {"retry_count": 0})

    def process(case):
        work = output / "base" / case["id"]
        observation = read_json(Path(case["dataset"]) / case["record"]["observation"])
        if (work / "request.json").exists():
            response = call_visual(observation, "pair_focus", work, configured("qwen3.8-flash"))
        else:
            response = read_json(work / "result.json")
        result = repair_return_pairs(observation, response, read_json(work / "candidates.json"))
        write_json(work / "final.json", result)
        print(case["id"], "base", response["status"], response["request_count"], flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(process, frozen["cases"]))
    write_json(output / "base-completed.json", {"stations": len(frozen["cases"])})


def prepare(output: Path):
    if (output / "base-registration.json").exists():
        if not (output / "base-completed.json").exists():
            raise ValueError("complete base before recovery")
        selected = read_json(output / "base-registration.json")["cases"]
        historical = False
    else:
        output.mkdir(parents=True, exist_ok=False)
        selected = cases()[0]
        historical = True
    if (output / "registration.json").exists():
        raise ValueError("already prepared")
    jobs, triggered = [], []
    for case in selected:
        if historical:
            base_path = Path("results/exp22-image-ablation-full-v1/predictions/flash_image") / case["id"] / "result.json"
        else:
            base_path = output / "base" / case["id"] / "final.json"
        numeric_path = Path(case["run"]) / "predictions/segment_numeric" / case["record"]["station_id"] / "result.json"
        observation = read_json(Path(case["dataset"]) / case["record"]["observation"])
        base, numeric = read_json(base_path), read_json(numeric_path)
        graph = recovery_graph(observation, numeric, base)
        case.update(base_path=str(base_path), numeric_path=str(numeric_path), graph=graph,
                    base_sha256=sha256(base_path), numeric_sha256=sha256(numeric_path))
        if not graph["pairs"]:
            continue
        triggered.append(case["id"])
        directory = output / "inputs" / case["id"]
        directory.mkdir(parents=True)
        image = directory / "local.png"
        render_local(observation, graph, image)
        indices = np.flatnonzero(~np.isfinite(observed_values(observation)).all(axis=1)).tolist()
        for mode, model in MODES.items():
            work = output / "predictions" / case["id"] / mode
            work.mkdir(parents=True)
            write_json(work / "candidates.json", {"graph": graph, "numerical_prediction": base})
            content = [{"type": "text", "text": json.dumps({"graph": graph, "missing_indices": indices})}]
            if not mode.endswith("noimage"):
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                + base64.b64encode(image.read_bytes()).decode("ascii")}})
            payload = {"model": model, "temperature": 0, "enable_thinking": False,
                "response_format": {"type": "json_schema", "json_schema": {"name": "gnss_recovery",
                    "strict": True, "schema": PairDecision.model_json_schema()}},
                "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}]}
            write_json(work / "request.json", payload)
            jobs.append({"case_id": case["id"], "mode": mode, "request_sha256": sha256(work / "request.json")})
        print(case["id"], graph["pairs"], flush=True)
    write_json(output / "registration.json", {"experiment_id": "GNSS-VLM-EXP23", "cases": selected,
        "historical": historical, "triggered": triggered, "pilot_ids": triggered[:2], "jobs": jobs,
        "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__))})
    (output / "experiment-source.py").write_bytes(Path(__file__).read_bytes())
    print({"stations": len(selected), "recovery_triggered": len(triggered), "requests": len(jobs)}, flush=True)


def infer(output: Path, phase: str):
    frozen = read_json(output / "registration.json")
    if sha256(PROTOCOL) != frozen["protocol_sha256"] or sha256(Path(__file__)) != frozen["script_sha256"]:
        raise ValueError("registered source changed")
    if phase == "remaining" and not read_json(output / "pilot-summary.json")["expand"]:
        raise ValueError("pilot did not justify expansion")
    if (output / f"{phase}-started.json").exists():
        raise ValueError("no retries")
    jobs = [job for job in frozen["jobs"] if (job["case_id"] in frozen["pilot_ids"]) == (phase == "pilot")]
    write_json(output / f"{phase}-started.json", {"requests": len(jobs), "retry_count": 0})
    lookup = {case["id"]: case for case in frozen["cases"]}

    def process(job):
        case = lookup[job["case_id"]]
        work = output / "predictions" / case["id"] / job["mode"]
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("request changed")
        observation = read_json(Path(case["dataset"]) / case["record"]["observation"])
        result = call(observation, work, configured(MODES[job["mode"]]))
        print(case["id"], job["mode"], result["status"], result.get("pair_decision"), flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(process, jobs))
    write_json(output / f"{phase}-completed.json", {"requests": len(jobs)})


def numeric_decision(observation: dict, numeric: dict, graph: dict) -> list[str]:
    evidence = numerical_evidence(observation)
    keep = evidence.observed.copy()
    for event in numeric["prediction"]["events"]:
        if event["kind"] == "spike":
            keep[event["start"]:event["end"] + 1] = False
    time = np.flatnonzero(keep)
    values = evidence.residual[keep] / evidence.noise
    return [pair["id"] for pair in graph["pairs"]
            if returning_pair(time, values, pair["start"], pair["end"] + 1)["paired"]]


def summarize(visual: Path, target: Path, phase: str):
    frozen = read_json(visual / "registration.json")
    selected = frozen["cases"] if phase == "full" else [case for case in frozen["cases"] if case["id"] in frozen["pilot_ids"]]
    rows = {method: [] for method in ("segment_numeric", "base_flash", "recovery_numeric", "candidate_all", *MODES)}
    target.mkdir(parents=True, exist_ok=False)
    for case in selected:
        observation = read_json(Path(case["dataset"]) / case["record"]["observation"])
        truth = read_json(Path(case["dataset"]) / case["record"]["truth"])
        base, numeric = read_json(Path(case["base_path"])), read_json(Path(case["numeric_path"]))
        graph = case["graph"]
        for method in rows:
            result = copy.deepcopy(base)
            if method == "segment_numeric":
                result = numeric
            elif graph["pairs"] and method != "base_flash":
                if method in MODES:
                    response = read_json(visual / "predictions" / case["id"] / method / "result.json")
                else:
                    ids = numeric_decision(observation, numeric, graph) if method == "recovery_numeric" else [p["id"] for p in graph["pairs"]]
                    # Numerical proposals can conflict: use candidate order consistently.
                    used, compatible = set(), []
                    for pair in graph["pairs"]:
                        edges = {pair["start"], pair["end"] + 1}
                        if pair["id"] in ids and not edges & used:
                            compatible.append(pair["id"])
                            used.update(edges)
                    response = {"status": "success", "pair_decision": {"pair_ids": compatible,
                                "evidence": "Candidate-only automatic control"}}
                result = apply(observation, base, graph, response)
            write_json(target / "predictions" / method / case["id"] / "result.json", result)
            rows[method].append({"seed": case["seed"], "station": case["record"]["station_id"],
                                 "metrics": evaluate(observation, truth, result)})
    metrics = {method: aggregate(values) for method, values in rows.items()} if selected else {}
    base_f1 = metrics.get("base_flash", {}).get("transient", {}).get("affiliation", {}).get("station_macro_f1")
    expand = base_f1 is not None and any(metrics[m]["transient"]["affiliation"]["station_macro_f1"] > base_f1 for m in ("flash_image", "vl_image"))
    summary = {"experiment_id": "GNSS-VLM-EXP23", "stations": len(selected), "phase": phase,
               "historical": frozen["historical"], "methods": metrics, "expand": expand}
    write_json(target / "summary.json", summary)
    write_json(target / "per-station.json", rows)
    if phase == "pilot":
        write_json(visual / "pilot-summary.json", summary)
    lines = ["# EXP23：SDR-CR候选恢复复核", "", f"{len(selected)}站；{phase}；历史数据诊断={frozen['historical']}。", "",
             "| 方法 | Aff-P | Aff-R | 暂态Aff-F1 | 日F1 | 覆盖 | 总FP日 | 负站FP日 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, metric in metrics.items():
        r = metric["transient"]
        a = r["affiliation"]
        def fmt(value):
            return "NA" if value is None else f"{value:.4f}"
        lines.append(f"| {method} | {fmt(a['precision'])} | {fmt(a['recall'])} | {fmt(a['station_macro_f1'])}"
                     f" | {fmt(r['daily_micro']['f1'])} | {fmt(r['truth_event_mean_coverage'])}"
                     f" | {r['false_positive_days']} | {r['negative_station_fp_days']} |")
    lines += ["", f"按Aff-F1扩量条件：{expand}。IoU等补充指标见summary.json。", ""]
    (target / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare-base", "infer-base", "prepare", "infer", "summarize"))
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--phase", choices=("pilot", "remaining", "full"), default="pilot")
    args = parser.parse_args()
    if args.stage == "prepare-base":
        prepare_base(args.baseline, args.visual)
    elif args.stage == "infer-base":
        infer_base(args.visual)
    elif args.stage == "prepare":
        prepare(args.visual)
    elif args.stage == "infer":
        infer(args.visual, args.phase)
    else:
        summarize(args.visual, args.out, args.phase)


if __name__ == "__main__":
    main()
