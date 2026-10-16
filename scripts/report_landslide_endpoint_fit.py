"""Export complete endpoint-fit controls and raw-first diagnostic figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_endpoint_fit import (
    BASE,
    DATA,
    PRIOR,
    TARGETS,
    csv_rows,
    method_name,
    verify,
)
from report_landslide_core_lockbox import full_table
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json


def diagnostic_figure(case, feature, labels, predictions, evidence, path, target):
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    days = np.arange(len(values))
    figure = plt.figure(figsize=(16, 11))
    grid = figure.add_gridspec(5, 2, width_ratios=(1.6, 1), height_ratios=(1, 1, 1, 0.65, 0.65))
    for axis, component in enumerate(("N", "E", "U")):
        panel = figure.add_subplot(grid[axis, 0])
        panel.plot(days, values[:, axis], linewidth=0.7)
        panel.set_xlim(0, len(values))
        panel.set_ylabel(component + " (mm)")
        panel.grid(alpha=0.2)
    for rep in (0, 1):
        panel = figure.add_subplot(grid[3 + rep, 0])
        base = f"direct_visual_r{rep}"
        names = [base, method_name(base, "selective"), method_name(base, "unconditional_fit")]
        matrix = [[int(row["activity_label"]) for row in labels]]
        matrix.extend(predictions[name][case.case_id]["activity"] for name in names)
        panel.imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
                     cmap=ListedColormap(["#bcbcbc", "#fafafa", "#277e8c"]),
                     extent=(-0.5, len(values) - 0.5, len(matrix) - 0.5, -0.5))
        panel.set_yticks(range(4), ["Reference", "Direct", "Selective", "Unconditional"])
        panel.set_ylabel(f"Repeat {rep}")
        panel.set_xlabel("Original day | teal=activity,white=stationary,gray=unknown")
    # Largest accepted movement in this already scored record is a disclosed
    # post-score diagnostic. It never changes the full-population decision.
    candidates = []
    for rep in (0, 1):
        rows = evidence[f"direct_visual_r{rep}"][case.case_id]["rules"]["selective"]
        candidates.extend((rep, row) for row in rows if row["accepted"])
    if candidates:
        rep, row = max(candidates, key=lambda item: abs(item[1]["selected_boundary"] - item[1]["original_boundary"]))
        start, stop = row["evidence_start"], row["evidence_stop"]
        times = np.array(row["block_times"])
        medians = np.array(row["block_medians_mm"])
        fitted = np.array(row["full_transition_fit"]["fitted_medians_mm"])
        for axis, component in enumerate(("N", "E", "U")):
            panel = figure.add_subplot(grid[axis, 1])
            panel.plot(days[start:stop], values[start:stop, axis], linewidth=0.7, label="Raw")
            panel.scatter(times, medians[:, axis], s=12, label="14d median")
            panel.plot(times, fitted[:, axis], linewidth=1.2, label="Transition fit")
            panel.axvline(row["original_boundary"], color="#666", linestyle=":", label="Original")
            panel.axvline(row["selected_boundary"], color="#a13d40", linestyle="--", label="Proposed")
            panel.set_ylabel(component + " (mm)")
            panel.grid(alpha=0.2)
            if axis == 0:
                panel.legend(fontsize=7, ncol=2)
        text = (
            f"Post-score diagnostic: r{rep}, event {row['event_index']}, {row['side']}\n"
            f"Original {row['original_boundary']} -> proposed {row['selected_boundary']}\n"
            f"Knot span {row['knot_span_days']} days\n"
            "Held-out MSE: transition / flat / drift / step\n"
        )
        for fold in row["folds"]:
            errors = fold["held_out_mse"]
            text += " / ".join(f"{errors[key]:.3f}" for key in ("transition", "stationary", "linear_drift", "single_step")) + "\n"
        text += "Alternating blocks are descriptive, not independent tests.\nRaw displacement fit does not establish geological cause."
    else:
        text = "No selective endpoint accepted; original prediction retained."
    panel = figure.add_subplot(grid[3:, 1])
    panel.axis("off")
    panel.text(0, 1, text, va="top", fontsize=9)
    figure.suptitle(case.case_id + " | " + target + " | exposed development diagnostic")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def report(run, analysis, out):
    verify(run)
    verify_manifest(analysis)
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    figures = out / "figures"
    figures.mkdir()
    decision = read_json(analysis / "decision.json")
    timings = csv_rows(run / "cpu-timings.csv")
    fits_seconds = sum(float(row["fit_seconds"]) for row in timings)
    application_seconds = sum(float(row["application_seconds"]) for row in timings)
    accepted, transitions = [], []
    text = ["# 局部原始位移端点拟合：开发结果", "",
            "完整24过程/预指定12作者记录；原预测冻结后进行局部拟合，修订输出再次冻结后读参考评分。"
            "无新增API、无重训练、无标签调参。作者参考已开发暴露，非独立专家或现场金标准。",
            "", "决定：" + decision["decision"], "",
            f"相同参考事件与预测身份的过程共同端点MAE变化：{decision['pooled_process_common_endpoint_mae_change']:+.6f}日。",
            f"本地计时路径：拟合/缓存查找{fits_seconds:.3f}秒，三个规则合计应用{application_seconds:.3f}秒；"
            "这是perf_counter墙钟，非纯CPU时间，不包含特征加载、阶段头预测或评分。", ""]
    for target in TARGETS:
        summaries = csv_rows(analysis / target / "summary.csv")
        predictions = read_json(analysis / target / "predictions.json")
        old_results = read_json(PRIOR / target / "results.json")
        results = read_json(analysis / target / "results.json")
        if any(results[name] != result for name, result in old_results.items()):
            raise ValueError("Historical result changed")
        old_summaries = csv_rows(PRIOR / target / "summary.csv")
        if summaries[:len(old_summaries)] != old_summaries:
            raise ValueError("Historical deployment cost changed")
        evidence = read_json(run / target / "endpoint-evidence.json")
        labels = read_json(BASE / "core-lockbox-analysis-v1" / target / "reference-labels.json")
        text.extend(["## " + target, "", full_table(summaries), ""])
        for base, cases in evidence.items():
            for cid, facts in cases.items():
                for rule, diagnostics in facts["rules"].items():
                    for row in diagnostics:
                        accepted.append({"target": target, "base": base, "case_id": cid, "rule": rule,
                                         "event_index": row["event_index"], "side": row["side"],
                                         "status": row["status"], "accepted": row["accepted"],
                                         "original_boundary": row["original_boundary"],
                                         "selected_boundary": row.get("selected_boundary"),
                                         "knot_span_days": row.get("knot_span_days"),
                                         "reason": row.get("reason", row.get("joint_rejection", ""))})
        for rep in (0, 1):
            base = f"direct_visual_r{rep}"
            for rule in ("selective", "no_step_rival", "unconditional_fit"):
                for cid, reference in labels.items():
                    actual = np.array([int(row["activity_label"]) for row in reference])
                    before = np.array(predictions[base][cid]["activity"])
                    after = np.array(predictions[method_name(base, rule)][cid]["activity"])
                    for ref in (-1, 0, 1):
                        for previous in (-1, 0, 1):
                            for revised in (-1, 0, 1):
                                transitions.append({"target": target, "repeat": rep, "rule": rule, "case_id": cid,
                                                    "reference": ref, "before": previous, "after": revised,
                                                    "days": int(((actual == ref) & (before == previous) & (after == revised)).sum())})
        # Two exposed author-target cases and two process cases: losses and a
        # gain. This post-score illustration never selects the evaluated subset.
        ids = ("case_0171", "case_0176") if target == "secondary_developer" else ("case_0169", "case_0188")
        for cid in ids:
            case, feature = load_case_features(DATA, cid)
            diagnostic_figure(case, feature, labels[cid], predictions, evidence,
                              figures / (cid + "-endpoint.png"), target)
    write_csv(out / "endpoints.csv", accepted)
    write_csv(out / "direct-state-transitions.csv", transitions)
    (out / "all-results.md").write_text("\n".join(text), encoding="utf-8")
    write_json(out / "verification.json", {"old_results_metrics_and_api_costs": "unchanged",
                                          "local_fit_wall_seconds": fits_seconds,
                                          "local_rule_application_wall_seconds": application_seconds,
                                          "new_remote_requests": 0,
                                          "endpoint_diagnostic_rows": len(accepted),
                                          "direct_transition_rows": len(transitions)})
    paths = list(run.rglob("*")) + list(analysis.rglob("*"))
    write_json(out / "source-bindings.json", {str(path): sha(path) for path in paths if path.is_file()})
    write_json(out / "manifest.json", {"files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Endpoint figures and complete results saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.analysis.resolve(), args.out.resolve())
