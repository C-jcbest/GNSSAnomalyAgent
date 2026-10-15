"""Post-hoc gap-boundary constraint on unchanged EXP23 saved model decisions."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
from exp14_fingerprint import apply
from exp20_range_metrics import aggregate, evaluate
from experiment_exp22 import MODES
from experiment_exp23 import numeric_decision

from gnss_research.files import read_json, sha256, write_json
from gnss_research.schema import observed_values


def gap_guard(observation: dict, graph: dict, response: dict) -> dict:
    """Only recover first observed platform days with unobserved preceding days."""
    guarded = copy.deepcopy(response)
    if response["status"] != "success":
        return guarded
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    eligible = {pair["id"] for pair in graph["pairs"]
                if pair["start"] > 0 and observed[pair["start"]] and not observed[pair["start"] - 1]}
    proposed = response["pair_decision"]["pair_ids"]
    guarded["pair_decision"]["pair_ids"] = [pair for pair in proposed if pair in eligible]
    guarded["gap_guard_rejected_ids"] = [pair for pair in proposed if pair not in eligible]
    return guarded


def summarize(source: Path, target: Path):
    registration = read_json(source / "registration.json")
    target.mkdir(parents=True, exist_ok=False)
    methods = ("segment_numeric", "base_flash", "gap_numeric", "gap_candidate_all",
               *("gap_" + mode for mode in MODES))
    rows = {method: [] for method in methods}
    rejected = {method: [] for method in methods}
    for case in registration["cases"]:
        observation = read_json(Path(case["dataset"]) / case["record"]["observation"])
        truth = read_json(Path(case["dataset"]) / case["record"]["truth"])
        base = read_json(Path(case["base_path"]))
        numeric = read_json(Path(case["numeric_path"]))
        graph = case["graph"]
        for method in methods:
            result = base
            if method == "segment_numeric":
                result = numeric
            elif method != "base_flash" and graph["pairs"]:
                if method[4:] in MODES:
                    response = read_json(source / "predictions" / case["id"] / method[4:] / "result.json")
                else:
                    ids = numeric_decision(observation, numeric, graph) if method == "gap_numeric" else [p["id"] for p in graph["pairs"]]
                    used, compatible = set(), []
                    for pair in graph["pairs"]:
                        edges = {pair["start"], pair["end"] + 1}
                        if pair["id"] in ids and not edges & used:
                            compatible.append(pair["id"])
                            used.update(edges)
                    response = {"status": "success", "pair_decision": {
                        "pair_ids": compatible, "evidence": "Same-candidate automatic control"}}
                response = gap_guard(observation, graph, response)
                result = apply(observation, base, graph, response)
                rejected[method].append({"case_id": case["id"], "rejected_ids": response.get("gap_guard_rejected_ids", [])})
                result["gap_guard_rejected_ids"] = response.get("gap_guard_rejected_ids", [])
            write_json(target / "predictions" / method / case["id"] / "result.json", result)
            rows[method].append({"seed": case["seed"], "station": case["record"]["station_id"],
                                 "metrics": evaluate(observation, truth, result)})
    metrics = {method: aggregate(values) for method, values in rows.items()}
    write_json(target / "summary.json", {"experiment_id": "GNSS-VLM-EXP24", "source": str(source),
        "source_registration_sha256": sha256(source / "registration.json"),
        "script_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(Path("docs/experiment-exp24.md")),
        "stations": len(registration["cases"]), "historical": registration["historical"],
        "analysis": "post-hoc structural constraint using unchanged saved model requests/responses",
        "new_requests": 0, "methods": metrics, "rejected": rejected})
    write_json(target / "per-station.json", rows)
    lines = ["# EXP24：SDR-GR缺测边界恢复", "", "事后开发分析；0新增调用；复用未修改输入的EXP23响应。", "",
             "| 方法 | Aff-P | Aff-R | 暂态Aff-F1 | 日F1 | 覆盖 | 总FP日 | 负站FP日 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, metric in metrics.items():
        r = metric["transient"]
        a = r["affiliation"]
        lines.append(f"| {method} | {a['precision']:.4f} | {a['recall']:.4f} | {a['station_macro_f1']:.4f}"
                     f" | {r['daily_micro']['f1']:.4f} | {r['truth_event_mean_coverage']:.4f}"
                     f" | {r['false_positive_days']} | {r['negative_station_fp_days']} |")
    lines += ["", "IoU等补充指标见summary.json；旧站与新开发站分别报告，不称新独立测试。", ""]
    (target / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source, args.out)
